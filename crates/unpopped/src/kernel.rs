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
//! U3 row 4 adds execution ids ([`ScalarExpr::Exec`]), a launch shape, memory-space allocations
//! ([`Place::Alloc`]) and [`Stmt::Barrier`] (`docs/u3-row4-scopes-and-memory-spaces.md`); the checker
//! knows them, and `oracle::kernel_launch::run_launch` runs them (`run_kernel` runs unlaunched
//! kernels only).
//!
//! Deliberately **not** here yet (each is named in the spec): `Stmt::Call` (needs a module type,
//! spec unknown U-2), pointer values (`Place::Ptr`), the effect set (computed when a consumer
//! asks), and any producer other than `lift_opdef`.

use serde::{Deserialize, Serialize};
use unpopped_vocab::ElementKind;

pub use crate::kernel_check::check_kernel;

use crate::exec::{AllocDecl, LaunchShape, Scope};
use crate::ir::ScalarExpr;
use crate::types::{Ty, TypeError};

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
    pub(crate) const fn reads(self) -> bool {
        matches!(self, Self::Read | Self::ReadWrite)
    }

    pub(crate) const fn writes(self) -> bool {
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

/// A storage location. `Logical` is named by **logical** coordinates: the element the operand's
/// view (strides, broadcast, reversal) selects, as the oracle's writeback does; it keeps meaning
/// "through the view". `Alloc` is an element of a declared array in a memory space.
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
    /// The element at flat `index` of the allocation `alloc` (decision D-R4-3: arrays by id, no
    /// pointer values).
    Alloc {
        /// Index into [`KernelFn::allocs`].
        alloc: u32,
        /// An integer expression: the flat element index.
        index: ScalarExpr,
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
    /// Wait until every thread of `scope` (`Subgroup` or `Block`) reaches this statement. Must not
    /// sit under thread-divergent control (`KernelError::BarrierDivergent`).
    Barrier {
        /// The group that synchronizes.
        scope: Scope,
    },
    /// An atomic read-modify-write of one element of a global buffer: `*place = *place (op) value`
    /// as one indivisible step (**relaxed**: it orders nothing else, and a [`Stmt::Barrier`] is the
    /// only ordering a kernel has). The old value is not returned. Two atomics on one element never
    /// race, whatever threads or blocks issue them; an atomic and a plain access to one element do,
    /// unless a barrier orders them (`docs/ir-kernel-layer.md`, "Atomics"). v1: a global buffer
    /// (`ReadWrite`) place or a `Shared` allocation element, an integer element type (the result
    /// wraps at the element width) or an `F32`/`F64` `Add` (accumulation order unspecified, so a
    /// float result is defined only inside a band: section 5a).
    AtomicRmw {
        /// The combine.
        op: AtomicOp,
        /// The element updated.
        place: Place,
        /// The operand (the buffer's element type).
        value: ScalarExpr,
    },
}

/// The combine of a [`Stmt::AtomicRmw`]. Each is associative and commutative on integers, so the
/// result does not depend on the order the threads issue them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum AtomicOp {
    /// `old + value`, wrapping at the element width.
    Add,
    /// The larger of the two.
    Max,
    /// The smaller of the two.
    Min,
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
    /// The launch geometry; `None` is one logical thread (the whole of U2).
    #[serde(default)]
    pub launch: Option<LaunchShape>,
    /// Arrays in memory spaces, indexed by [`Place::Alloc::alloc`].
    #[serde(default)]
    pub allocs: Vec<AllocDecl>,
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
            launch: None,
            allocs: Vec::new(),
        }
    }

    /// The same kernel launched with `launch`.
    #[must_use]
    pub fn with_launch(mut self, launch: LaunchShape) -> Self {
        self.launch = Some(launch);
        self
    }

    /// The same kernel with the allocations `allocs`.
    #[must_use]
    pub fn with_allocs(mut self, allocs: Vec<AllocDecl>) -> Self {
        self.allocs = allocs;
        self
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
    /// An `AtomicRmw` outside the v1 form: a `ReadWrite` global buffer or `Shared` element of an
    /// integer type.
    AtomicUnsupported {
        /// What is wrong.
        what: &'static str,
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
    /// An `Exec` leaf or a `Barrier` in a kernel whose `launch` is `None`.
    ExecWithoutLaunch,
    /// An `Exec` axis above 2.
    ExecAxisOutOfRange {
        /// The axis.
        axis: u8,
    },
    /// A launch with a zero extent, or a subgroup range that is empty or starts at 0.
    BadLaunch,
    /// `LaneId`, `SubgroupId` or `SubgroupSize` in a launch that names no subgroup size.
    SubgroupIdWithoutSubgroup,
    /// An allocation declared in `MemorySpace::Global` (a global buffer is a buffer parameter).
    AllocSpace {
        /// The allocation index.
        alloc: u32,
    },
    /// An allocation whose extent is a zero `Static`, or a `Param` that is not an integer scalar
    /// parameter local.
    AllocExtent {
        /// The allocation index.
        alloc: u32,
    },
    /// A `Store` to a `Constant` allocation.
    WriteToConstant {
        /// The allocation index.
        alloc: u32,
    },
    /// A `Shared` allocation used in a kernel with no launch (shared memory is per block).
    SharedWithoutLaunch {
        /// The allocation index.
        alloc: u32,
    },
    /// A `Barrier` whose scope is not `Subgroup` or `Block`.
    BadBarrierScope,
    /// A `Barrier` under control that may differ between the threads of its scope. The rule is
    /// conservative (decision D-R4-5): it rejects some uniform programs.
    BarrierDivergent,
}
