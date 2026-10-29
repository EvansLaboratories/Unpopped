// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The kernel layer, v0** (`docs/ir-kernel-layer.md`): the program of one logical thread, as
//! typed statements over locals and buffer elements.
//!
//! A separate artifact from [`crate::ir::OpDef`] (PM ruling 2026-10-28): nothing here is a field
//! of `OpDef` or `KernelPlan`, no `Backend` or `Frontend` trait mentions it, and the plan gate has
//! nothing to decline because nothing in v0 reaches a plan. The only change to the intent layer is
//! the [`ScalarExpr::Local`] leaf.
//!
//! [`check_kernel`] is the scope and type checker. Its error for each rule is pinned by a
//! known-bad mutation in `tests/kernel_layer.rs`, each paired with the unmutated program passing.
//!
//! Deliberately **not** here yet (each is named in the spec): `Stmt::Call` (needs a module type,
//! spec unknown U-2), pointers and memory spaces (U3), the effect set (computed when a consumer
//! asks), and any producer other than `lift_opdef`.

use serde::{Deserialize, Serialize};
use unpopped_vocab::ElementKind;

use crate::ir::ScalarExpr;
use crate::types::{self, Ty, TypeError};

/// What a buffer parameter may do. Checked by [`check_kernel`] against every `Load` and `Store`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BufferRole {
    /// Loaded from only.
    Read,
    /// Stored to only.
    Write,
    /// Loaded from and stored to.
    ReadWrite,
}

impl BufferRole {
    const fn reads(self) -> bool {
        matches!(self, Self::Read | Self::ReadWrite)
    }

    const fn writes(self) -> bool {
        matches!(self, Self::Write | Self::ReadWrite)
    }
}

/// What a parameter is.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ParamKind {
    /// A buffer of `elem` elements addressed by logical coordinates.
    Buffer {
        /// Element type.
        #[serde(with = "crate::text::ek")]
        elem: ElementKind,
        /// Which accesses are legal.
        role: BufferRole,
        /// The number of coordinates a `Place::Logical` must carry, when known.
        rank: Option<u8>,
    },
    /// A scalar argument, readable as the local `local` (in scope everywhere, never assigned).
    Scalar {
        /// Type of the argument.
        #[serde(with = "crate::text::ek")]
        ty: ElementKind,
        /// The local id it is bound to.
        local: u32,
    },
}

/// One kernel parameter.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Param {
    /// Name, for diagnostics.
    pub name: String,
    /// What it is.
    pub kind: ParamKind,
}

impl Param {
    /// A buffer parameter.
    #[must_use]
    pub fn buffer(
        name: impl Into<String>,
        elem: ElementKind,
        role: BufferRole,
        rank: Option<u8>,
    ) -> Self {
        Self {
            name: name.into(),
            kind: ParamKind::Buffer { elem, role, rank },
        }
    }

    /// A scalar parameter bound to local `local`.
    #[must_use]
    pub fn scalar(name: impl Into<String>, ty: ElementKind, local: u32) -> Self {
        Self {
            name: name.into(),
            kind: ParamKind::Scalar { ty, local },
        }
    }
}

/// The declared type of a local: `lanes == 1` is a scalar, `2` or `4` a packed value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct LocalDecl {
    /// Element type.
    #[serde(with = "crate::text::ek")]
    pub elem: ElementKind,
    /// Lane count: `1` (scalar), `2` or `4`.
    pub lanes: u8,
}

impl LocalDecl {
    /// A scalar local.
    #[must_use]
    pub const fn scalar(elem: ElementKind) -> Self {
        Self { elem, lanes: 1 }
    }

    /// A packed local of `lanes` lanes.
    #[must_use]
    pub const fn packed(elem: ElementKind, lanes: u8) -> Self {
        Self { elem, lanes }
    }

    /// The [`Ty`] this declaration denotes.
    #[must_use]
    pub const fn ty(self) -> Ty {
        if self.lanes == 1 {
            Ty::Scalar(self.elem)
        } else {
            Ty::Packed(self.elem, self.lanes)
        }
    }
}

/// A storage location named by **logical** coordinates: the element the operand's view
/// (strides, broadcast, reversal) selects, as the oracle's writeback does. Pointers and memory
/// spaces are U3's and add variants; this one keeps meaning "through the view".
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Place {
    /// The element of buffer parameter `buffer` at `coords`.
    Logical {
        /// Index into [`KernelFn::params`].
        buffer: u32,
        /// One integer expression per axis.
        coords: Vec<ScalarExpr>,
    },
}

/// A sequence of statements; a block opens a scope.
#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct Block(pub Vec<Stmt>);

/// A statement. Exhaustive on purpose: a new statement breaks every consumer at compile time,
/// as a new `ScalarExpr` node does.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Stmt {
    /// Declare `local` and initialize it; in scope for the rest of the block.
    Let {
        /// The local id.
        local: u32,
        /// Initial value.
        init: ScalarExpr,
    },
    /// Declare `local` as the element at `place`.
    Load {
        /// The local id.
        local: u32,
        /// Where to read.
        place: Place,
    },
    /// Mutate a declared local.
    Assign {
        /// The local id.
        local: u32,
        /// New value (same type).
        value: ScalarExpr,
    },
    /// Write one element.
    Store {
        /// Where to write.
        place: Place,
        /// The value (the buffer's element type).
        value: ScalarExpr,
    },
    /// Branch on `cond != 0`.
    If {
        /// Condition, any scalar type.
        cond: ScalarExpr,
        /// Taken when non-zero.
        then: Block,
        /// Taken otherwise.
        els: Block,
    },
    /// A counted loop: `local` takes `start, start+step, ...` while below `end` (above, for a
    /// negative step). The three bounds are evaluated once, before the first iteration.
    For {
        /// The counter's local id, declared by this statement.
        local: u32,
        /// First value.
        start: ScalarExpr,
        /// Exclusive limit.
        end: ScalarExpr,
        /// Increment; zero traps.
        step: ScalarExpr,
        /// Loop body.
        body: Block,
    },
    /// Loop while `cond != 0`.
    While {
        /// Condition, re-evaluated each pass.
        cond: ScalarExpr,
        /// Loop body.
        body: Block,
    },
    /// Leave the innermost loop.
    Break,
    /// Start the innermost loop's next pass.
    Continue,
    /// Leave the kernel.
    Return,
}

/// A kernel: the program of one logical thread.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct KernelFn {
    /// Name, for diagnostics.
    pub name: String,
    /// The dtype of the untyped leaves (`Const`), the role the plan dtype plays for an `OpDef`.
    #[serde(with = "crate::text::ek")]
    pub compute: ElementKind,
    /// Parameters, indexed by [`Place::Logical::buffer`].
    pub params: Vec<Param>,
    /// Declared type of every local id (`Local(i)`, `Let { local: i }`, ...).
    pub locals: Vec<LocalDecl>,
    /// The program.
    pub body: Block,
}

impl KernelFn {
    /// Build a kernel.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        compute: ElementKind,
        params: Vec<Param>,
        locals: Vec<LocalDecl>,
        body: Block,
    ) -> Self {
        Self {
            name: name.into(),
            compute,
            params,
            locals,
            body,
        }
    }
}

/// Why a [`KernelFn`] is not well formed. The first violation, in program order.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum KernelError {
    /// A local id is not in `locals`.
    UnknownLocal {
        /// The id.
        local: u32,
    },
    /// A local is read or assigned before any statement declares it.
    UseBeforeLet {
        /// The id.
        local: u32,
    },
    /// A local is used after the block that declared it closed.
    OutOfScope {
        /// The id.
        local: u32,
    },
    /// A `Let`, `Load` or `For` declares an id that is already in scope.
    Redeclared {
        /// The id.
        local: u32,
    },
    /// Two types that must be equal are not (`what` names the position).
    TypeMismatch {
        /// Where.
        what: &'static str,
    },
    /// An expression does not type-check.
    Type(TypeError),
    /// An intent-layer leaf (`Input`, `Param`, `Reduced`, `Coord`, `CoordInt`) inside a kernel.
    IntentLeaf {
        /// The node kind.
        node: &'static str,
    },
    /// `Break` or `Continue` outside a loop.
    BreakOutsideLoop,
    /// A statement after an unconditional `Break`, `Continue` or `Return` in the same block.
    Unreachable,
    /// A `Store` to a buffer that is not `Write` or `ReadWrite`.
    WriteToReadOnly {
        /// The buffer parameter index.
        buffer: u32,
    },
    /// A `Load` from a buffer that is not `Read` or `ReadWrite`.
    ReadFromWriteOnly {
        /// The buffer parameter index.
        buffer: u32,
    },
    /// A `Place` names a parameter that is not a buffer, or carries the wrong number of
    /// coordinates, or a coordinate that is not an integer.
    BadPlace {
        /// The buffer parameter index.
        buffer: u32,
    },
    /// An `Assign` to a `For` counter inside that loop's body.
    CounterAssigned {
        /// The id.
        local: u32,
    },
    /// An `Assign` to a scalar-parameter local.
    ParamAssigned {
        /// The id.
        local: u32,
    },
}

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
    check_block(k, &k.body, &mut sc)
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
        let Place::Logical { buffer, .. } = place;
        return Err(KernelError::ReadFromWriteOnly { buffer: *buffer });
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
        let Place::Logical { buffer, .. } = place;
        return Err(KernelError::WriteToReadOnly { buffer: *buffer });
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
    let Place::Logical { buffer, coords } = p;
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

/// Type an expression inside a kernel: reject intent-layer leaves, check every `Local` is in
/// scope, then defer to [`types::infer_ty_in`] with the declared local types.
fn check_expr(k: &KernelFn, e: &ScalarExpr, sc: &Scope) -> Result<Ty, KernelError> {
    walk(k, e, sc)?;
    types::infer_ty_in(e, k.compute, &sc.tys).map_err(KernelError::Type)
}

fn walk(k: &KernelFn, e: &ScalarExpr, sc: &Scope) -> Result<(), KernelError> {
    match e {
        ScalarExpr::Local(i) => return use_local(k, *i, sc),
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
