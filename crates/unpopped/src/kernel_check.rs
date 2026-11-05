// SPDX-License-Identifier: MIT OR Apache-2.0
//! The scope and type checker for [`crate::kernel::KernelFn`] ([`check_kernel`]), split from
//! `kernel.rs` so neither file outgrows the size the analysers accept.

mod async_copy;
mod collective;
mod extern_call;
mod mma;

use unpopped_vocab::ElementKind;

use crate::exec::{Extent, MemorySpace, Scope as ExecScope};
use crate::ir::ScalarExpr;
use crate::kernel::{
    AtomicOp, BarrierKind, Block, BufferRole, CallArg, KernelError, KernelFn, MemoryOrder, Param,
    ParamKind, Place, Stmt,
};
use crate::types::{self, Ty};

/// Per-walk state: which locals are in scope, which are protected from `Assign`.
struct Scope {
    in_scope: Vec<bool>,
    ever_declared: Vec<bool>,
    param_local: Vec<bool>,
    counter_local: Vec<bool>,
    tys: Vec<Ty>,
    loop_depth: u32,
    /// The signatures of the module's functions; empty outside a module (a `Call` is then refused).
    callees: Vec<CalleeSig>,
    /// The signatures of the module's extern declarations.
    externs: Vec<ExternSig>,
}

/// What an `ExternCall` is checked against: a declaration's parameter and result types and the effects the checker
/// and the uniformity rule read.
#[derive(Clone)]
pub(crate) struct ExternSig {
    pub(crate) params: Vec<ElementKind>,
    pub(crate) results: Vec<ElementKind>,
    /// No effect: the results are as uniform as the arguments. Otherwise they may differ between threads.
    pub(crate) pure: bool,
    /// The scope of a rendezvous, making a call a barrier site.
    pub(crate) sync: Option<ExecScope>,
}

/// What a `Call` is checked against: a callee's parameters and the type of its result local.
#[derive(Clone)]
pub(crate) struct CalleeSig {
    pub(crate) params: Vec<ParamKind>,
    pub(crate) result: Option<Ty>,
    /// The function reaches a `Barrier`, directly or through its own calls: a call of it is a barrier site.
    pub(crate) barrier: bool,
}

/// Check that `k` is well formed (`docs/ir-kernel-layer.md` section 4).
///
/// # Errors
///
/// The first [`KernelError`], walking statements in program order and each statement's
/// expressions before its effects.
pub fn check_kernel(k: &KernelFn) -> Result<(), KernelError> {
    check_kernel_with(k, Vec::new(), Vec::new())
}

/// [`check_kernel`] with the signatures of the module the function lives in, so a `Call` resolves.
pub(crate) fn check_kernel_with(
    k: &KernelFn,
    callees: Vec<CalleeSig>,
    externs: Vec<ExternSig>,
) -> Result<(), KernelError> {
    check_launch(k)?;
    let n = k.locals.len();
    let mut sc = Scope {
        in_scope: vec![false; n],
        ever_declared: vec![false; n],
        param_local: vec![false; n],
        counter_local: vec![false; n],
        tys: k.locals.iter().map(|d| d.ty()).collect(),
        loop_depth: 0,
        callees,
        externs,
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
    let barrier_callees: Vec<bool> = sc.callees.iter().map(|c| c.barrier).collect();
    crate::kernel_uniform::check_barriers(k, &barrier_callees, &sc.externs)
}

/// A launch has no zero extent, a subgroup range is non-empty and starts at 1 or more, and clusters (if any) tile the grid.
fn check_launch(k: &KernelFn) -> Result<(), KernelError> {
    let Some(l) = &k.launch else { return Ok(()) };
    let sg_ok = l.subgroup.is_none_or(|s| s.min >= 1 && s.min <= s.max);
    // clusters tile the grid: every cluster extent is positive and divides the grid's
    let cluster_ok = l
        .cluster
        .is_none_or(|c| (0..3).all(|a| c[a] >= 1 && l.grid[a] % c[a] == 0));
    if l.block.contains(&0) || l.grid.contains(&0) || !sg_ok || !cluster_ok {
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

/// An `AtomicRmw` or an `AtomicCas`.
fn check_atomic_stmt(
    k: &KernelFn,
    st: &Stmt,
    sc: &mut Scope,
    declared_here: &mut Vec<usize>,
) -> Result<(), KernelError> {
    match st {
        Stmt::AtomicRmw {
            op,
            place,
            value,
            old,
            scope,
            ..
        } => {
            let form = AtomicForm {
                op: *op,
                site: AtomicSite {
                    old: *old,
                    scope: *scope,
                },
            };
            check_atomic(k, place, value, form, sc, declared_here)?;
        }
        Stmt::AtomicCas {
            place,
            expected,
            desired,
            old,
            scope,
            ..
        } => {
            let site = AtomicSite {
                old: *old,
                scope: *scope,
            };
            check_cas(k, place, [expected, desired], site, sc, declared_here)?;
        }
        _ => unreachable!("check_atomic_stmt is called for atomic statements only"),
    }
    Ok(())
}

/// The statements that synchronize or call: atomics, fences, calls and barriers.
fn check_sync_stmt(
    k: &KernelFn,
    st: &Stmt,
    sc: &mut Scope,
    declared_here: &mut Vec<usize>,
) -> Result<(), KernelError> {
    match st {
        Stmt::AtomicRmw { .. } | Stmt::AtomicCas { .. } => {
            check_atomic_stmt(k, st, sc, declared_here)?;
        }
        Stmt::Fence { scope, order } => {
            if matches!(order, MemoryOrder::Relaxed) {
                return Err(KernelError::SyncUnsupported {
                    what: "a Relaxed Fence (it would order nothing; use Acquire, Release, AcqRel or SeqCst)",
                });
            }
            if !scope_ok(k, *scope) {
                return Err(KernelError::SyncUnsupported {
                    what: "a Fence scope other than Subgroup, Block, Device or (with launch cluster dims) Cluster",
                });
            }
        }
        Stmt::Collective { .. } => collective::check_collective(k, st, sc, declared_here)?,
        Stmt::Mma { .. } => mma::check_mma(k, st)?,
        Stmt::CopyAsync { .. } | Stmt::AsyncCommit | Stmt::AsyncWait { .. } => {
            async_copy::check_async(k, st, sc)?;
        }
        Stmt::ExternCall {
            decl,
            args,
            results,
        } => extern_call::check_extern_call(k, (*decl, args, results), sc, declared_here)?,
        Stmt::Call {
            callee,
            args,
            result,
        } => check_call(k, (*callee, args, *result), sc, declared_here)?,
        Stmt::Barrier { scope, kind } => {
            let BarrierKind::Full = kind;
            if !scope_ok(k, *scope) {
                return Err(KernelError::BadBarrierScope);
            }
            if k.launch.is_none() {
                return Err(KernelError::ExecWithoutLaunch);
            }
        }
        _ => unreachable!("check_sync_stmt is called for sync statements only"),
    }
    Ok(())
}

/// Whether a barrier, fence or atomic may name `scope`: `Subgroup`, `Block` and `Device` always, `Cluster` only in a
/// launch that has cluster dimensions; `Thread` has no semantics.
fn scope_ok(k: &KernelFn, scope: ExecScope) -> bool {
    match scope {
        ExecScope::Subgroup | ExecScope::Block | ExecScope::Device => true,
        ExecScope::Cluster => k.launch.is_some_and(|l| l.cluster.is_some()),
        _ => false,
    }
}

/// A `Call` against the signature of its callee: arity, each argument's kind and type, the buffer roles
/// the caller really has, and the result local's type (`docs/options-u3-calls.md`, Q2 and Q5).
fn check_call(
    k: &KernelFn,
    (callee, args, result): (u32, &[CallArg], Option<u32>),
    sc: &mut Scope,
    declared_here: &mut Vec<usize>,
) -> Result<(), KernelError> {
    if sc.callees.is_empty() {
        return Err(KernelError::CallUnsupported {
            what: "a Call outside a module: check it with check_module",
        });
    }
    let Some(sig) = sc.callees.get(callee as usize).cloned() else {
        return Err(KernelError::BadCallee { callee });
    };
    if args.len() != sig.params.len() {
        return Err(KernelError::CallArity {
            expected: sig.params.len(),
            found: args.len(),
        });
    }
    for (i, (a, p)) in args.iter().zip(&sig.params).enumerate() {
        check_call_arg(k, i, (a, p), sc)?;
    }
    match (result, sig.result) {
        (Some(local), Some(ty)) => declare(k, local, ty, sc, declared_here)?,
        (Some(_), None) => {
            return Err(KernelError::CallResult {
                what: "a result is asked of a callee that has none",
            });
        }
        (None, _) => {}
    }
    Ok(())
}

/// One argument against the parameter in its position.
fn check_call_arg(
    k: &KernelFn,
    i: usize,
    (arg, param): (&CallArg, &ParamKind),
    sc: &Scope,
) -> Result<(), KernelError> {
    let bad = |what| Err(KernelError::CallArgMismatch { arg: i, what });
    match (arg, param) {
        (CallArg::Scalar(e), ParamKind::Scalar { ty, .. }) => {
            if check_expr(k, e, sc)? == Ty::Scalar(*ty) {
                Ok(())
            } else {
                bad("a scalar argument of a different type than its parameter")
            }
        }
        (CallArg::Buffer(b), ParamKind::Buffer { elem, role, rank }) => {
            match k.params.get(*b as usize).map(|p| &p.kind) {
                Some(ParamKind::Buffer {
                    elem: e,
                    role: r,
                    rank: n,
                }) => buffer_fits((*elem, *role, *rank), (*e, *r, *n)).map_or(Ok(()), bad),
                _ => bad("a buffer argument that is not a buffer parameter of the caller"),
            }
        }
        (CallArg::Alloc(a), ParamKind::Buffer { elem, role, rank }) => {
            let Some(d) = k.allocs.get(*a as usize) else {
                return bad("an allocation the caller does not declare");
            };
            alloc_fits((*elem, *role, *rank), (d.elem, d.space)).map_or(Ok(()), bad)
        }
        _ => bad("an argument of a different kind than its parameter"),
    }
}

/// Why the caller's buffer `(elem, role, rank)` cannot be passed to the callee's `(elem, role, rank)`.
fn buffer_fits(
    callee: (ElementKind, BufferRole, Option<u8>),
    caller: (ElementKind, BufferRole, Option<u8>),
) -> Option<&'static str> {
    let ((ce, cr, cn), (e, r, n)) = (callee, caller);
    if ce != e {
        Some("a buffer of a different element type than its parameter")
    } else if (cr.reads() && !r.reads()) || (cr.writes() && !r.writes()) {
        Some("a buffer without the role the callee uses it with")
    } else if matches!((cn, n), (Some(x), Some(y)) if x != y) {
        Some("a buffer of a different rank than its parameter")
    } else {
        None
    }
}

/// Why the caller's allocation `(elem, space)` cannot be passed to the callee's buffer parameter.
fn alloc_fits(
    callee: (ElementKind, BufferRole, Option<u8>),
    alloc: (ElementKind, MemorySpace),
) -> Option<&'static str> {
    let ((ce, cr, cn), (e, space)) = (callee, alloc);
    if ce != e {
        Some("an allocation of a different element type than its parameter")
    } else if space == MemorySpace::Constant && cr.writes() {
        Some("a Constant allocation passed to a callee that writes it")
    } else if cn != Some(1) {
        Some("an allocation is one-dimensional, its parameter must declare rank 1")
    } else {
        None
    }
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
        Stmt::AtomicRmw { .. }
        | Stmt::AtomicCas { .. }
        | Stmt::Fence { .. }
        | Stmt::Collective { .. }
        | Stmt::Mma { .. }
        | Stmt::CopyAsync { .. }
        | Stmt::AsyncCommit
        | Stmt::AsyncWait { .. }
        | Stmt::Call { .. }
        | Stmt::ExternCall { .. }
        | Stmt::Barrier { .. } => check_sync_stmt(k, st, sc, declared_here)?,
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

/// The parts of an `AtomicRmw` other than its place and operand.
#[derive(Clone, Copy)]
struct AtomicForm {
    op: AtomicOp,
    site: AtomicSite,
}

/// An `AtomicRmw` (v1): a `ReadWrite` global buffer or `Shared` element, a value of the element's type,
/// a `Subgroup`, `Block` or `Device` scope, any memory order, and an `old` local (if any) of the element's type that this
/// statement declares.
fn check_atomic(
    k: &KernelFn,
    place: &Place,
    value: &ScalarExpr,
    form: AtomicForm,
    sc: &mut Scope,
    declared_here: &mut Vec<usize>,
) -> Result<(), KernelError> {
    let AtomicForm { op, site } = form;
    let elem = check_atomic_site(k, place, site, sc)?;
    let float_ok = matches!(op, AtomicOp::Add | AtomicOp::Exchange)
        && matches!(elem, ElementKind::F32 | ElementKind::F64);
    if crate::types::int_range(elem).is_none() && !float_ok {
        return Err(KernelError::AtomicUnsupported {
            what: "an atomic on a non-integer element (only an F32/F64 Add or Exchange is admitted)",
        });
    }
    if check_expr(k, value, sc)? != Ty::Scalar(elem) {
        return Err(KernelError::TypeMismatch { what: "AtomicRmw" });
    }
    if let Some(local) = site.old {
        declare(k, local, Ty::Scalar(elem), sc, declared_here)?;
    }
    Ok(())
}

/// An `AtomicCas`: an integer element at an atomic site, `expected` and `desired` of that type, and an
/// `old` local (if any) of that type.
fn check_cas(
    k: &KernelFn,
    place: &Place,
    [expected, desired]: [&ScalarExpr; 2],
    site: AtomicSite,
    sc: &mut Scope,
    declared_here: &mut Vec<usize>,
) -> Result<(), KernelError> {
    let elem = check_atomic_site(k, place, site, sc)?;
    if crate::types::int_range(elem).is_none() {
        return Err(KernelError::AtomicUnsupported {
            what: "an AtomicCas on a non-integer element (compare-and-exchange is integer only)",
        });
    }
    for e in [expected, desired] {
        if check_expr(k, e, sc)? != Ty::Scalar(elem) {
            return Err(KernelError::TypeMismatch { what: "AtomicCas" });
        }
    }
    if let Some(local) = site.old {
        declare(k, local, Ty::Scalar(elem), sc, declared_here)?;
    }
    Ok(())
}

/// What every atomic statement states about where it runs: the old-value local and the scope (the memory
/// order needs no check: every order is meaningful on a read-modify-write).
#[derive(Clone, Copy)]
struct AtomicSite {
    old: Option<u32>,
    scope: ExecScope,
}

/// The checks every atomic shares: a `Subgroup`, `Block` or `Device` scope, a `ReadWrite` global buffer or a
/// `Shared` allocation element. Returns the element type.
fn check_atomic_site(
    k: &KernelFn,
    place: &Place,
    site: AtomicSite,
    sc: &Scope,
) -> Result<ElementKind, KernelError> {
    // every memory order is meaningful on a read-modify-write (`SeqCst` is treated as `AcqRel`)
    if !scope_ok(k, site.scope) {
        return Err(KernelError::AtomicUnsupported {
            what: "an atomic scope other than Subgroup, Block, Device or (with launch cluster dims) Cluster",
        });
    }
    let outside_shared = match place {
        Place::Alloc { alloc, .. } => k
            .allocs
            .get(*alloc as usize)
            .is_some_and(|a| a.space != MemorySpace::Shared),
        _ => false,
    };
    if outside_shared {
        return Err(KernelError::AtomicUnsupported {
            what: "an atomic on an allocation outside Shared memory",
        });
    }
    let (elem, role) = check_place(k, place, sc)?;
    if !(role.reads() && role.writes()) {
        return Err(KernelError::AtomicUnsupported {
            what: "an atomic on a buffer that is not ReadWrite",
        });
    }
    Ok(elem)
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
