// SPDX-License-Identifier: MIT OR Apache-2.0
//! The scope and type checker for [`crate::kernel::KernelFn`] ([`check_kernel`]), split from
//! `kernel.rs` so neither file outgrows the size the analysers accept.

use unpopped_vocab::ElementKind;

use crate::exec::{Extent, MemorySpace, Scope as ExecScope};
use crate::ir::ScalarExpr;
use crate::kernel::{Block, BufferRole, KernelError, KernelFn, Param, ParamKind, Place, Stmt};
use crate::types::{self, Ty};

/// Per-walk state: which locals are in scope, which are protected from `Assign`.
struct Scope {
    in_scope: Vec<bool>,
    ever_declared: Vec<bool>,
    param_local: Vec<bool>,
    counter_local: Vec<bool>,
    tys: Vec<Ty>,
    loop_depth: u32,
}

/// Check that `k` is well formed (`docs/ir-kernel-layer.md` section 4).
///
/// # Errors
///
/// The first [`KernelError`], walking statements in program order and each statement's
/// expressions before its effects.
pub fn check_kernel(k: &KernelFn) -> Result<(), KernelError> {
    check_launch(k)?;
    let n = k.locals.len();
    let mut sc = Scope {
        in_scope: vec![false; n],
        ever_declared: vec![false; n],
        param_local: vec![false; n],
        counter_local: vec![false; n],
        tys: k.locals.iter().map(|d| d.ty()).collect(),
        loop_depth: 0,
    };
    for p in &k.params {
        if let ParamKind::Scalar { ty, local } = &p.kind {
            let i = *local as usize;
            let Some(decl) = k.locals.get(i) else {
                return Err(KernelError::UnknownLocal { local: *local });
            };
            if sc.in_scope[i] {
                return Err(KernelError::Redeclared { local: *local });
            }
            if decl.ty() != Ty::Scalar(*ty) {
                return Err(KernelError::TypeMismatch {
                    what: "scalar parameter and its local",
                });
            }
            sc.in_scope[i] = true;
            sc.ever_declared[i] = true;
            sc.param_local[i] = true;
        }
    }
    check_allocs(k)?;
    check_block(k, &k.body, &mut sc)?;
    crate::kernel_uniform::check_barriers(k)
}

/// A launch has no zero extent, and a subgroup range is non-empty and starts at 1 or more.
fn check_launch(k: &KernelFn) -> Result<(), KernelError> {
    let Some(l) = &k.launch else { return Ok(()) };
    let sg_ok = l.subgroup.is_none_or(|s| s.min >= 1 && s.min <= s.max);
    if l.block.contains(&0) || l.grid.contains(&0) || !sg_ok {
        return Err(KernelError::BadLaunch);
    }
    Ok(())
}

/// Every allocation is in a declarable space and has a usable extent.
fn check_allocs(k: &KernelFn) -> Result<(), KernelError> {
    for (i, a) in k.allocs.iter().enumerate() {
        let alloc = i as u32;
        if a.space == MemorySpace::Global {
            return Err(KernelError::AllocSpace { alloc });
        }
        let ok = match a.extent {
            Extent::Static(n) => n > 0,
            Extent::Param(local) => k.params.iter().any(|p| {
                matches!(p.kind, ParamKind::Scalar { ty, local: l }
                    if l == local && types::int_range(ty).is_some())
            }),
        };
        if !ok {
            return Err(KernelError::AllocExtent { alloc });
        }
    }
    Ok(())
}

fn check_block(k: &KernelFn, b: &Block, sc: &mut Scope) -> Result<(), KernelError> {
    let mut declared_here: Vec<usize> = Vec::new();
    let mut terminated = false;
    for st in &b.0 {
        if terminated {
            return Err(KernelError::Unreachable);
        }
        terminated = check_stmt(k, st, sc, &mut declared_here)?;
    }
    for i in declared_here {
        sc.in_scope[i] = false;
    }
    Ok(())
}

/// Check one statement; `Ok(true)` when it unconditionally leaves the block (`Break`,
/// `Continue`, `Return`).
fn check_stmt(
    k: &KernelFn,
    st: &Stmt,
    sc: &mut Scope,
    declared_here: &mut Vec<usize>,
) -> Result<bool, KernelError> {
    match st {
        Stmt::Let { local, init } => {
            let ty = check_expr(k, init, sc)?;
            declare(k, *local, ty, sc, declared_here)?;
        }
        Stmt::Load { local, place } => check_load(k, *local, place, sc, declared_here)?,
        Stmt::Assign { local, value } => check_assign(k, *local, value, sc)?,
        Stmt::Store { place, value } => check_store(k, place, value, sc)?,
        Stmt::If { cond, then, els } => {
            scalar_cond(k, cond, sc)?;
            check_block(k, then, sc)?;
            check_block(k, els, sc)?;
        }
        Stmt::For {
            local,
            start,
            end,
            step,
            body,
        } => check_for(k, *local, [start, end, step], body, sc)?,
        Stmt::While { cond, body } => {
            scalar_cond(k, cond, sc)?;
            in_loop(k, body, sc)?;
        }
        Stmt::Break | Stmt::Continue => {
            if sc.loop_depth == 0 {
                return Err(KernelError::BreakOutsideLoop);
            }
            return Ok(true);
        }
        Stmt::Return => return Ok(true),
        Stmt::Barrier { scope } => {
            if !matches!(scope, ExecScope::Subgroup | ExecScope::Block) {
                return Err(KernelError::BadBarrierScope);
            }
            if k.launch.is_none() {
                return Err(KernelError::ExecWithoutLaunch);
            }
        }
    }
    Ok(false)
}

fn check_load(
    k: &KernelFn,
    local: u32,
    place: &Place,
    sc: &mut Scope,
    declared_here: &mut Vec<usize>,
) -> Result<(), KernelError> {
    let (elem, role) = check_place(k, place, sc)?;
    if !role.reads() {
        return Err(match place {
            Place::Logical { buffer, .. } => KernelError::ReadFromWriteOnly { buffer: *buffer },
            Place::Alloc { alloc, .. } => KernelError::BadPlace { buffer: *alloc },
        });
    }
    declare(k, local, Ty::Scalar(elem), sc, declared_here)
}

fn check_store(
    k: &KernelFn,
    place: &Place,
    value: &ScalarExpr,
    sc: &Scope,
) -> Result<(), KernelError> {
    let (elem, role) = check_place(k, place, sc)?;
    if !role.writes() {
        return Err(match place {
            Place::Logical { buffer, .. } => KernelError::WriteToReadOnly { buffer: *buffer },
            Place::Alloc { alloc, .. } => KernelError::WriteToConstant { alloc: *alloc },
        });
    }
    if check_expr(k, value, sc)? != Ty::Scalar(elem) {
        return Err(KernelError::TypeMismatch { what: "Store" });
    }
    Ok(())
}

fn check_assign(
    k: &KernelFn,
    local: u32,
    value: &ScalarExpr,
    sc: &Scope,
) -> Result<(), KernelError> {
    let ty = check_expr(k, value, sc)?;
    use_local(k, local, sc)?;
    let i = local as usize;
    if sc.param_local[i] {
        return Err(KernelError::ParamAssigned { local });
    }
    if sc.counter_local[i] {
        return Err(KernelError::CounterAssigned { local });
    }
    if sc.tys[i] != ty {
        return Err(KernelError::TypeMismatch { what: "Assign" });
    }
    Ok(())
}

/// A `For`: three integer bounds of one type, a counter declared by the loop itself, and a body
/// that may `Break` and `Continue`.
fn check_for(
    k: &KernelFn,
    local: u32,
    bounds: [&ScalarExpr; 3],
    body: &Block,
    sc: &mut Scope,
) -> Result<(), KernelError> {
    let t0 = check_expr(k, bounds[0], sc)?;
    let t1 = check_expr(k, bounds[1], sc)?;
    let t2 = check_expr(k, bounds[2], sc)?;
    let int = matches!(t0, Ty::Scalar(e) if types::int_range(e).is_some());
    if t0 != t1 || t1 != t2 || !int {
        return Err(KernelError::TypeMismatch { what: "For bounds" });
    }
    let mut counter = Vec::new();
    declare(k, local, t0, sc, &mut counter)?;
    sc.counter_local[local as usize] = true;
    let r = in_loop(k, body, sc);
    sc.counter_local[local as usize] = false;
    sc.in_scope[local as usize] = false;
    r
}

/// Check `body` one loop deeper.
fn in_loop(k: &KernelFn, body: &Block, sc: &mut Scope) -> Result<(), KernelError> {
    sc.loop_depth += 1;
    let r = check_block(k, body, sc);
    sc.loop_depth -= 1;
    r
}

fn declare(
    k: &KernelFn,
    local: u32,
    ty: Ty,
    sc: &mut Scope,
    declared_here: &mut Vec<usize>,
) -> Result<(), KernelError> {
    let i = local as usize;
    if i >= k.locals.len() {
        return Err(KernelError::UnknownLocal { local });
    }
    if sc.in_scope[i] {
        return Err(KernelError::Redeclared { local });
    }
    if sc.tys[i] != ty {
        return Err(KernelError::TypeMismatch {
            what: "declaration and its local",
        });
    }
    sc.in_scope[i] = true;
    sc.ever_declared[i] = true;
    declared_here.push(i);
    Ok(())
}

fn use_local(k: &KernelFn, local: u32, sc: &Scope) -> Result<(), KernelError> {
    let i = local as usize;
    if i >= k.locals.len() {
        return Err(KernelError::UnknownLocal { local });
    }
    if sc.in_scope[i] {
        Ok(())
    } else if sc.ever_declared[i] {
        Err(KernelError::OutOfScope { local })
    } else {
        Err(KernelError::UseBeforeLet { local })
    }
}

fn scalar_cond(k: &KernelFn, cond: &ScalarExpr, sc: &Scope) -> Result<(), KernelError> {
    match check_expr(k, cond, sc)? {
        Ty::Scalar(_) => Ok(()),
        Ty::Packed(..) => Err(KernelError::TypeMismatch {
            what: "condition (a packed value has no truth value)",
        }),
    }
}

fn check_place(
    k: &KernelFn,
    p: &Place,
    sc: &Scope,
) -> Result<(ElementKind, BufferRole), KernelError> {
    let (buffer, coords) = match p {
        Place::Logical { buffer, coords } => (buffer, coords),
        Place::Alloc { alloc, index } => return check_alloc_place(k, *alloc, index, sc),
    };
    let bad = KernelError::BadPlace { buffer: *buffer };
    let Some(Param {
        kind: ParamKind::Buffer { elem, role, rank },
        ..
    }) = k.params.get(*buffer as usize)
    else {
        return Err(bad);
    };
    if rank.is_some_and(|r| usize::from(r) != coords.len()) {
        return Err(bad);
    }
    for c in coords {
        let int = matches!(check_expr(k, c, sc)?, Ty::Scalar(e) if types::int_range(e).is_some());
        if !int {
            return Err(bad);
        }
    }
    Ok((*elem, *role))
}

/// An `Alloc` place: the allocation exists, `Shared` has a launch, the index is an integer. A
/// `Constant` allocation reads like a read-only buffer; the others like a read-write one.
fn check_alloc_place(
    k: &KernelFn,
    alloc: u32,
    index: &ScalarExpr,
    sc: &Scope,
) -> Result<(ElementKind, BufferRole), KernelError> {
    let Some(a) = k.allocs.get(alloc as usize) else {
        return Err(KernelError::BadPlace { buffer: alloc });
    };
    if a.space == MemorySpace::Shared && k.launch.is_none() {
        return Err(KernelError::SharedWithoutLaunch { alloc });
    }
    let int = matches!(check_expr(k, index, sc)?, Ty::Scalar(e) if types::int_range(e).is_some());
    if !int {
        return Err(KernelError::BadPlace { buffer: alloc });
    }
    let role = if a.space == MemorySpace::Constant {
        BufferRole::Read
    } else {
        BufferRole::ReadWrite
    };
    Ok((a.elem, role))
}

/// Type an expression inside a kernel: reject intent-layer leaves, check every `Local` is in
/// scope, then defer to [`types::infer_ty_in`] with the declared local types.
fn check_expr(k: &KernelFn, e: &ScalarExpr, sc: &Scope) -> Result<Ty, KernelError> {
    walk(k, e, sc)?;
    types::infer_ty_in(e, k.compute, &sc.tys).map_err(KernelError::Type)
}

/// An `Exec` leaf needs a launch, an axis in `0..=2`, and (for the subgroup ids) a launch that
/// names a subgroup size.
fn check_exec(k: &KernelFn, id: crate::exec::ExecId) -> Result<(), KernelError> {
    let Some(l) = &k.launch else {
        return Err(KernelError::ExecWithoutLaunch);
    };
    if let Some(axis) = id.axis().filter(|a| *a > 2) {
        return Err(KernelError::ExecAxisOutOfRange { axis });
    }
    if id.needs_subgroup() && l.subgroup.is_none() {
        return Err(KernelError::SubgroupIdWithoutSubgroup);
    }
    Ok(())
}

fn walk(k: &KernelFn, e: &ScalarExpr, sc: &Scope) -> Result<(), KernelError> {
    match e {
        ScalarExpr::Local(i) => return use_local(k, *i, sc),
        ScalarExpr::Exec(id) => return check_exec(k, *id),
        ScalarExpr::Input(_)
        | ScalarExpr::Param(_)
        | ScalarExpr::Reduced(_)
        | ScalarExpr::Coord(_)
        | ScalarExpr::CoordInt(..) => {
            return Err(KernelError::IntentLeaf {
                node: types::node_name(e),
            });
        }
        _ => {}
    }
    for x in types::operands(e) {
        walk(k, x, sc)?;
    }
    Ok(())
}
