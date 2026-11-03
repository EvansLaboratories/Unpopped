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
pub use crate::kernel_module::check_module;

use crate::exec::{AllocDecl, LaunchShape, Scope};
use crate::ir::{ReduceOp, ScalarExpr};
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
        /// Which barrier. Only [`BarrierKind::Full`] exists; the field is here so a named or a
        /// split arrive/wait barrier is an additive variant of the kind, not another break of `Stmt`
        /// (`docs/options-u4-sync-and-atomics.md`, option B).
        kind: BarrierKind,
    },
    /// An atomic read-modify-write of one element of a global buffer: `*place = *place (op) value`
    /// as one indivisible step (**relaxed**: it orders nothing else, and a [`Stmt::Barrier`] is the
    /// only ordering a kernel has). With `old` set, the value the element held just before this
    /// thread's step is stored in that local (a fetch-op); which value that is depends on the order
    /// the threads reach the element, which the IR does not specify (section 5a). Two atomics on one
    /// element never race, whatever threads or blocks issue them; an atomic and a plain access to one
    /// element do, unless a barrier orders them (`docs/ir-kernel-layer.md`, "Atomics"). v1: a global
    /// buffer (`ReadWrite`) place or a `Shared` allocation element, an integer element type (the
    /// result wraps at the element width) or an `F32`/`F64` `Add` or `Exchange` (accumulation order
    /// unspecified, so a float sum is defined only inside a band: section 5a). `order` must be
    /// any [`MemoryOrder`] (since 0.42.7 a release-class atomic and an acquire-class one on one element order the
    /// plain accesses around them in the oracle; `SeqCst` is treated as `AcqRel`) and `scope` `Subgroup`, `Block` or
    /// `Device` (`Thread` and `Cluster` are a typed refusal).
    AtomicRmw {
        /// The combine.
        op: AtomicOp,
        /// The element updated.
        place: Place,
        /// The operand (the buffer's element type).
        value: ScalarExpr,
        /// The local that receives the old value, if the kernel wants it.
        old: Option<u32>,
        /// The memory order.
        order: MemoryOrder,
        /// The scope the atomicity holds at: `Subgroup`, `Block` or `Device`.
        scope: Scope,
    },
    /// Compare-and-exchange: if the element equals `expected`, set it to `desired`; `old` receives the
    /// value found either way. **Shape reserved: `check_kernel` refuses it** until its semantics land
    /// (`docs/options-u4-sync-and-atomics.md`, S3).
    AtomicCas {
        /// The element.
        place: Place,
        /// The value it is compared with.
        expected: ScalarExpr,
        /// The value stored on a match.
        desired: ScalarExpr,
        /// The local that receives the value found.
        old: Option<u32>,
        /// The memory order.
        order: MemoryOrder,
        /// The scope the atomicity holds at.
        scope: Scope,
    },
    /// A memory fence of `scope` with `order`. Since 0.42.8 the oracle runs it as the C++ fence rules do: a release
    /// fence (`Release`, `AcqRel`, `SeqCst`) before a relaxed atomic publishes what the thread knew; an acquire fence
    /// (`Acquire`, `AcqRel`, `SeqCst`) after a relaxed atomic read takes in what that read saw. `check_kernel` refuses a
    /// `Relaxed` fence and the scopes `Thread` and `Cluster`.
    Fence {
        /// The group the fence orders.
        scope: Scope,
        /// The order.
        order: MemoryOrder,
    },
    /// A collective over the threads of `scope`: every thread of the scope contributes `value`, and each
    /// receives its result in the local `result` (PM ruling 2026-11-02 on
    /// `docs/options-u5-collectives-and-richer-reductions.md`, P5). A statement and not an expression because it
    /// rendezvouses like a [`Stmt::Barrier`]. Since 0.42.3 the oracle runs it (full participation of a `Subgroup`
    /// or a `Block`; `unpopped::oracle::kernel_launch`), and `check_kernel` admits the v1 forms
    /// (`unpopped::kernel_check`'s collective checker); the shape came in the 0.42.0 break. Since 0.42.4
    /// `unpopped-cpu-c` emits it as a block-level construct between two thread loops. A call that reaches a
    /// collective is a barrier site, as one that reaches a `Barrier` is.
    Collective {
        /// Which collective.
        kind: CollectiveKind,
        /// The value this thread contributes.
        value: ScalarExpr,
        /// The local that receives this thread's result.
        result: u32,
        /// The group that takes part.
        scope: Scope,
    },
    /// Call another kernel of the same [`Module`]: `result = callee(args...)`. The callee is a
    /// [`KernelFn`] (PM ruling 2026-11-02 on `docs/options-u3-remainder.md`, R1a); `callee` indexes
    /// [`Module::fns`]. **Shape reserved: `check_kernel` refuses it** until the module checker and the
    /// call semantics land (acyclic call graph; a callee holding a `Barrier` called from uniform
    /// control only). It is in the 0.41.0 break so that `Stmt` breaks downstream once.
    Call {
        /// Index of the callee in the module's functions.
        callee: u32,
        /// The arguments, in the callee's parameter order (`docs/options-u3-calls.md`, Q2).
        args: Vec<CallArg>,
        /// The local that receives the callee's result, if it has one.
        result: Option<u32>,
    },
}

/// One argument of a [`Stmt::Call`], matched to the callee's parameter of the same position
/// (`docs/options-u3-calls.md`, Q2; PM ruling 2026-11-02). **Shape only**: nothing checks or runs it yet.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum CallArg {
    /// A scalar passed by value to a [`ParamKind::Scalar`] parameter.
    Scalar(ScalarExpr),
    /// The caller's buffer parameter (by index), passed by alias to a [`ParamKind::Buffer`] parameter.
    Buffer(u32),
    /// The caller's allocation (by index into [`KernelFn::allocs`]), passed by alias.
    Alloc(u32),
}

/// A set of kernels, any of which may [`Stmt::Call`] another. **Shape only** (0.41.0): nothing checks or
/// runs a module yet; the type exists so `Stmt::Call` has a callee to name.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Module {
    /// The kernels; a call names one by index.
    pub fns: Vec<KernelFn>,
}

impl Module {
    /// A module of `fns`.
    #[must_use]
    pub fn new(fns: Vec<KernelFn>) -> Self {
        Self { fns }
    }
}

/// Which barrier a [`Stmt::Barrier`] is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum BarrierKind {
    /// Every thread of the scope arrives, then every thread leaves.
    Full,
}

/// Which [`Stmt::Collective`] (ROADMAP U5 P5). **Shapes only** until the semantics lands.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum CollectiveKind {
    /// Each lane receives the value of another lane of the scope, named by `operand` under `mode`
    /// (`Idx`: the lane index, `Up`/`Down`: the distance, `Xor`: the lane mask).
    Shuffle {
        /// How `operand` names the source lane.
        mode: ShuffleMode,
        /// The lane index, distance or mask.
        operand: ScalarExpr,
    },
    /// A predicate over the scope's contributed values.
    Vote(VoteKind),
    /// Every lane receives the fold of the scope's values under `op`.
    Reduce(ReduceOp),
    /// Every lane receives the prefix fold up to itself (before itself when `exclusive`).
    Scan {
        /// The fold.
        op: ReduceOp,
        /// Whether the lane's own value is left out.
        exclusive: bool,
    },
    /// Every lane receives the value of lane `lane`.
    Broadcast {
        /// The source lane.
        lane: ScalarExpr,
    },
}

/// How a shuffle names its source lane.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ShuffleMode {
    /// The lane with this index.
    Idx,
    /// The lane this far below.
    Up,
    /// The lane this far above.
    Down,
    /// The lane whose index is this lane's index xor the mask.
    Xor,
}

/// The predicate of a vote.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum VoteKind {
    /// Every lane's value is non-zero.
    All,
    /// Some lane's value is non-zero.
    Any,
    /// A mask with bit `i` set when lane `i`'s value is non-zero.
    Ballot,
}

/// The memory order of an atomic or a fence.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum MemoryOrder {
    /// No ordering of other accesses.
    Relaxed,
    /// Later accesses cannot move before this one.
    Acquire,
    /// Earlier accesses cannot move after this one.
    Release,
    /// Both.
    AcqRel,
    /// A single total order of all such operations.
    SeqCst,
}

impl Stmt {
    /// A full barrier of `scope` (the only kind there is).
    #[must_use]
    pub fn barrier(scope: Scope) -> Self {
        Stmt::Barrier {
            scope,
            kind: BarrierKind::Full,
        }
    }

    /// A relaxed, device-scope atomic that does not return the old value: the form 0.40 had.
    #[must_use]
    pub fn atomic_rmw(op: AtomicOp, place: Place, value: ScalarExpr) -> Self {
        Stmt::AtomicRmw {
            op,
            place,
            value,
            old: None,
            order: MemoryOrder::Relaxed,
            scope: Scope::Device,
        }
    }
}

/// The combine of a [`Stmt::AtomicRmw`]. `Add`, `Max`, `Min`, `And`, `Or` and `Xor` are associative and
/// commutative on integers, so the final value does not depend on the order the threads issue them
/// (the old value, if requested, still does); `Exchange` is not, so even its final value does.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum AtomicOp {
    /// `old + value`, wrapping at the element width.
    Add,
    /// The larger of the two.
    Max,
    /// The smaller of the two.
    Min,
    /// `old & value` (integers).
    And,
    /// `old | value` (integers).
    Or,
    /// `old ^ value` (integers).
    Xor,
    /// Store `value`; the old value is what was there (integers, `F32`, `F64`: the bits move unchanged).
    Exchange,
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
    /// The local whose value a [`Stmt::Call`] of this function receives: its value at `Return` or at the
    /// end of the body. `None` for a kernel and for a callee with no result. Shape only (0.41.4).
    #[serde(default)]
    pub result: Option<u32>,
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
            result: None,
        }
    }

    /// The same kernel launched with `launch`.
    #[must_use]
    pub fn with_launch(mut self, launch: LaunchShape) -> Self {
        self.launch = Some(launch);
        self
    }

    /// The same function with `local` as its result (what a [`Stmt::Call`] of it receives).
    #[must_use]
    pub fn with_result(mut self, local: u32) -> Self {
        self.result = Some(local);
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
    /// An `AtomicRmw` or `AtomicCas` outside the v1 form: a `ReadWrite` global buffer or `Shared`
    /// element of an integer type, `Relaxed` order, `Device` scope, no compare-and-exchange yet.
    AtomicUnsupported {
        /// What is wrong.
        what: &'static str,
    },
    /// A sync form whose semantics has not landed (a `Relaxed` or `Thread`/`Cluster` `Fence`, a barrier kind): the
    /// shape is reserved, the checker refuses it, so no producer builds a kernel an interpreter would then have to guess about.
    SyncUnsupported {
        /// What is wrong.
        what: &'static str,
    },
    /// A `Call` names a function that is not in the module (or `check_module` names an entry that is not).
    BadCallee {
        /// The index.
        callee: u32,
    },
    /// A `Call` passes a different number of arguments than the callee has parameters.
    CallArity {
        /// The callee's parameter count.
        expected: usize,
        /// The call's argument count.
        found: usize,
    },
    /// A `Call` argument does not fit its parameter: the wrong kind, a different element type, a role the
    /// caller's buffer does not have.
    CallArgMismatch {
        /// The argument position.
        arg: usize,
        /// What is wrong.
        what: &'static str,
    },
    /// The call graph has a cycle through this function.
    CallCycle {
        /// The index of a function on the cycle.
        callee: u32,
    },
    /// A `Call` asks for a result the callee does not have.
    CallResult {
        /// What is wrong.
        what: &'static str,
    },
    /// A `Stmt::Collective` before its semantics has landed: the shape is reserved.
    CollectiveUnsupported {
        /// What is wrong.
        what: &'static str,
    },
    /// A `Call` outside a module (`check_kernel` alone), or something a callee may not yet be: the shape is
    /// accepted only where the module checker can resolve it.
    CallUnsupported {
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
