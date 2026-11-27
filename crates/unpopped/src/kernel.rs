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
//! Deliberately **not** here yet (each is named in the spec): pointer values (`Place::Ptr`) and the effect
//! set (computed when a consumer asks). `Stmt::Call` and the module type landed later (`check_module`, `run_module`).

use serde::{Deserialize, Serialize};
use unpopped_vocab::ElementKind;

pub use crate::kernel_check::check_kernel;
pub use crate::kernel_module::check_module;

use crate::exec::{AllocDecl, LaunchShape, MemorySpace, Scope};
use crate::ir::{ReduceOp, ScalarExpr};
use crate::mma::MmaAtom;
use crate::types::{Ty, TypeError};

/// What a buffer parameter may do. Checked by [`check_kernel`] against every `Load` and `Store`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BufferRole {
    /// Loaded from only. For an **entry** function (one nothing in its module calls) the launcher guarantees the buffer is
    /// distinct storage from every other parameter of the launch (the oracle passes distinct buffers; an emitted kernel states it
    /// as `BufferGuarantee::restrict`), so no thread of the launch writes it: the barrier-uniformity check treats a load from it
    /// at a uniform index as uniform. A function reached through a `Call` gets no such guarantee (`CallArg::Buffer` aliases).
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
    /// A group of scalar arguments passed together (ROADMAP U23): each field is readable as its own local, exactly as a
    /// [`ParamKind::Scalar`] is. The group describes a layout (offsets, size, alignment) and carries no thread-set, memory-kind
    /// or copy-count annotation and chooses no storage: that no-annotation rule is the U23 patent guard (`docs/PATENT-SWEEP.md`,
    /// U23 section; its test module is still named `kernel_interface::layout_only_guard`, a historical name).
    /// An emitter may pass it by value or by address, or flatten it; `KernelInterface::params_block` records the layout.
    Struct {
        /// The fields, in declaration order.
        fields: Vec<StructField>,
    },
}

/// One scalar field of a [`ParamKind::Struct`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct StructField {
    /// Name, for diagnostics.
    pub name: String,
    /// Type of the field.
    #[serde(with = "crate::text::ek")]
    pub ty: ElementKind,
    /// The local id it is bound to (in scope everywhere, never assigned).
    pub local: u32,
}

impl StructField {
    /// A field `name` of type `ty` bound to local `local`.
    #[must_use]
    pub fn new(name: impl Into<String>, ty: ElementKind, local: u32) -> Self {
        Self {
            name: name.into(),
            ty,
            local,
        }
    }
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

    /// A struct parameter: scalar `fields`, each bound to its own local (ROADMAP U23).
    #[must_use]
    pub fn structure(name: impl Into<String>, fields: Vec<StructField>) -> Self {
        Self {
            name: name.into(),
            kind: ParamKind::Struct { fields },
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
    /// The `elem`-typed value at **byte** `byte_offset` of the storage of buffer parameter `buffer` (ROADMAP U9c, level (b);
    /// `docs/options-u9c-pointer-values.md`): the `(elem*)((char*)buffer + byte_offset)` of C and the byte-addressed `ld`/`st` of
    /// PTX. There is no pointer value, only a base named by a parameter and an integer offset. Byte 0 is the buffer's first
    /// element (its view origin); strides and shape are **ignored** (raw is bytes of the storage, not "through the view").
    /// `elem` need not be the buffer's element type (a width-punned access). Traps, never UB: a `byte_offset` that is not a
    /// multiple of `elem`'s size (misaligned), or an access outside the buffer's bytes. `elem` is a scalar of a whole number
    /// of bytes (no sub-byte or untyped kind). The race detector sees the access at the granularity of the buffer's own
    /// elements: every element the bytes touch is logged, so two threads writing different bytes of one wider element are
    /// reported as a race the hardware would not have (conservative). Taken by `Load`, `Store`, `MaskedLoad` and `MaskedStore`, and as the source of a
    /// `CopyAsync`.
    Raw {
        /// Index into [`KernelFn::params`]; a buffer.
        buffer: u32,
        /// An integer expression: the byte offset from the buffer's first element.
        byte_offset: ScalarExpr,
        /// The type of the value accessed.
        #[serde(with = "crate::text::ek")]
        elem: ElementKind,
    },
    /// The `elem`-typed value at **byte** `byte_offset` of the allocation `alloc` (a `Shared` or `Private` one; ROADMAP U9c): the
    /// byte-addressed twin of [`Place::Raw`], with the same rules (byte 0 is the allocation's first element, a misaligned or
    /// out-of-range offset is a typed trap, `elem` is a whole number of bytes and need not be the allocation's type). The race
    /// detector sees every allocation element the bytes touch. An element is initialized once a store has covered all of its
    /// bytes; a load of an element not yet fully covered is the usual uninitialized-read trap (so a byte-wise fill of a wider
    /// element never initializes it: conservative). `Constant` and `Tensor` allocations are refused. Taken by `Load`, `Store`,
    /// `MaskedLoad` and `MaskedStore`.
    AllocRaw {
        /// Index into [`KernelFn::allocs`].
        alloc: u32,
        /// An integer expression: the byte offset from the allocation's first element.
        byte_offset: ScalarExpr,
        /// The type of the value accessed.
        #[serde(with = "crate::text::ek")]
        elem: ElementKind,
    },
    /// The element at flat `index` of the `Shared` allocation `alloc` **of another block of this block's cluster**
    /// (distributed shared memory: PTX `mapa`, ROADMAP U12 slice 1). `block` is an integer expression: the rank of
    /// the block inside its cluster, `x` fastest (`%cluster_ctarank`), so `0..cluster x * y * z`; the rank of this very
    /// block names its own array. The kernel's launch must name cluster dimensions. A peer access is ordered like any
    /// other access to `Shared` memory: by a `Cluster` barrier or a cluster-scope release/acquire pair; unordered, it is
    /// a data race. `Load`, `Store`, `AtomicRmw` and `AtomicCas` take it; `CopyAsync` does not.
    ClusterPeer {
        /// Index into [`KernelFn::allocs`]; a `Shared` allocation.
        alloc: u32,
        /// An integer expression: the rank of the owning block inside the cluster.
        block: ScalarExpr,
        /// An integer expression: the flat element index in that block's array.
        index: ScalarExpr,
    },
    /// The element of buffer parameter `buffer` at `coords` **on another device** of a multi-device launch (ROADMAP
    /// U12 slice 2): every device holds its own copy of each buffer, and this names the copy of device `device`, an
    /// integer expression in `0..LaunchShape::devices` (the rank of the running device names its own copy). The
    /// launch must span more than one device. A cross-device access is ordered only by a `System`-scope barrier or a
    /// `System`-scope release/acquire pair; unordered, it is a data race. `Load`, `Store`, `AtomicRmw` and `AtomicCas`
    /// take it; `CopyAsync` does not.
    Peer {
        /// Index into [`KernelFn::params`]; a buffer.
        buffer: u32,
        /// An integer expression: the rank of the device that owns the copy.
        device: ScalarExpr,
        /// One integer expression per axis.
        coords: Vec<ScalarExpr>,
    },
    /// The element of buffer parameter `buffer` at `coords` **on every device at once** (ROADMAP U12 slice 3; PTX
    /// `multimem`, one multicast address for the same buffer on all devices). Only two statements take it: a `Store`
    /// (`multimem.st`) writes the value to every device's copy, and an `AtomicRmw` (`multimem.red`) combines the value
    /// into every device's copy. The launch must span more than one device. A `Store` is admitted for `I32`, `U32`,
    /// `F32`, `F16` and `Bf16`. An `AtomicRmw` must have `scope: System`, `order: Release` (the hardware always
    /// releases at system scope) and no `old` (a reduction has no single old value); its op and element are `Add`,
    /// `Min` or `Max` on `I32`/`U32`, or `Add` on `F32` (`F16`/`Bf16` `Add` is refused until the oracle has a 16-bit
    /// float atomic). A `Load`, `AtomicCas` or `CopyAsync` is refused (a reducing read is a [`Stmt::MulticastLoadReduce`]). Each device's copy is
    /// ordered like any cross-device access: by a `System` barrier or a `System`-scope release/acquire pair, and an
    /// unordered access is a data race. Float `Add` order across devices is unspecified in a band, as a float atomic's.
    Multicast {
        /// Index into [`KernelFn::params`]; a buffer.
        buffer: u32,
        /// One integer expression per axis.
        coords: Vec<ScalarExpr>,
    },
}

/// The unroll hint of a [`Stmt::For`] (ROADMAP U18): a request to the emitter, never a meaning.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Unroll {
    /// No hint.
    #[default]
    None,
    /// Unroll completely (`#pragma unroll`).
    Full,
    /// Unroll by this factor (`#pragma unroll N`); 0 is refused.
    Count(u32),
}

/// The cache operator of a [`Stmt::CopyAsync`] (ROADMAP U13): a hint, never a meaning.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum CacheHint {
    /// None given.
    #[default]
    Auto,
    /// Cache at all levels (`cp.async.ca`).
    All,
    /// Cache at L2 only (`cp.async.cg`); 16-byte copies only.
    Global,
}

/// Which dimension of a `Wgmma` operand tile is contiguous (`wgmma`'s `imm-trans-a` / `imm-trans-b`; PTX ISA 9.5
/// 9.7.17.5.1.2: "If the argument imm-trans-a / imm-trans-b ... is 0, then K-major is used ... If the value of
/// argument imm-trans-a is 1 then M-major is used for matrix A. If the value of the argument imm-trans-b is 1, then
/// N-major is used for matrix B"). Transposed operands exist for 16-bit element types only.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Major {
    /// `K` is the contiguous dimension (`imm-trans` 0).
    #[default]
    K,
    /// `M` (for `A`) or `N` (for `B`) is the contiguous dimension (`imm-trans` 1).
    MN,
}

/// The swizzle mode of a shared-memory matrix (the PTX matrix-descriptor swizzle field; the descriptor's own wire codes
/// are the emitter's). **Only `None` is built**: the XOR the others apply is not yet confirmed against the ISA figures
/// (`docs/design-wgmma-descriptors.md`, slice 2), and the checker refuses the rest.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Swizzle {
    /// No swizzle (the "interleave" layouts).
    #[default]
    None,
    /// 32-byte swizzle.
    B32,
    /// 64-byte swizzle.
    B64,
    /// 128-byte swizzle.
    B128,
}

/// How the elements of one `Wgmma` operand tile (`MN` rows of 16 `K` values) sit in its `Shared` allocation.
///
/// `RowMajor` is what the statement meant before this field: row `mn` is 16 contiguous elements. It is **not a layout the
/// PTX matrix descriptor can read** (a core matrix is 8 rows x 16 bytes packed), so it is for the oracle only and an
/// emitter for hardware refuses it. `Canonical` is the PTX canonical layout (PTX ISA 9.5 9.7.17.5.1.2.1.3, CuTe notation,
/// `T = 8` for a 16-bit element): K-major `((8,m),(T,2)):((1T,SBO),(1,LBO))`, MN-major
/// `((T,1,m),(8,k)):((1,T,SBO),(1T,LBO))`, with `LBO` and `SBO` in **bytes**, as the descriptor carries them.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OperandLayout {
    /// Row `mn` is 16 contiguous elements (oracle only).
    #[default]
    RowMajor,
    /// The PTX canonical layout.
    Canonical {
        /// The contiguous dimension.
        major: Major,
        /// The swizzle mode (only `None` is built).
        swizzle: Swizzle,
        /// The leading dimension byte offset. `None` where the ISA says it is not used (a swizzled K-major operand).
        lbo: Option<u32>,
        /// The stride dimension byte offset.
        sbo: u32,
    },
}

impl Swizzle {
    /// `B` of the PTX `Swizzle<B,4,3>` (0 none, 1 for 32B, 2 for 64B, 3 for 128B): the number of address bits XORed.
    const fn bits(self) -> u32 {
        match self {
            Self::None => 0,
            Self::B32 => 1,
            Self::B64 => 2,
            Self::B128 => 3,
        }
    }

    /// The bytes after which the swizzle pattern repeats (1024 / 512 / 256 for 128B / 64B / 32B; 16 for none): a tile
    /// start must be a multiple of this for the descriptor's base offset to be 0 (PTX ISA 9.5 9.7.17.5.1.2.2).
    #[must_use]
    pub const fn pattern_bytes(self) -> u32 {
        match self {
            Self::None => 16,
            other => 128 << other.bits(),
        }
    }
}

impl OperandLayout {
    /// The element offset of `(mn, k)` from the tile's first element, for a 16-bit element, or `None` when a `Canonical`
    /// layout lacks the offset it needs (`lbo` for no swizzle or MN-major). `mn` is the row of `A` / `B`, `k < 16`.
    ///
    /// The canonical layout (PTX ISA 9.5 9.7.17.5.1.2.1.3, `T = 8`, `s = 2^B`): K-major `((8,m),(T,2)):((sT,SBO),(1,T))`
    /// swizzled (`(1T,SBO),(1,LBO)` with no swizzle), MN-major `((T,s,m),(8,k)):((1,T,LBO),(sT,SBO))` swizzled
    /// (`((T,1,m),(8,k)):((1,T,SBO),(1T,LBO))` with no swizzle); then `Swizzle<B,4,3>` on the **byte** address, which XORs
    /// address bits `[4, 4+B)` with bits `[7, 7+B)` (read from PTX Figures 159-168, `docs/design-wgmma-descriptors.md`
    /// section 5c). The tile start is taken to be pattern-aligned (the checker requires it), so the XOR on the offset
    /// is the XOR on the address.
    #[must_use]
    pub fn element_offset(&self, mn: usize, k: usize) -> Option<usize> {
        self.element_offset_sized(mn, k, 2)
    }

    /// [`Self::element_offset`] for an element of `elem_bytes` bytes (1 for the 8-bit floats, 2 for the 16-bit kinds): the
    /// 16-byte core-matrix row holds `T = 16 / elem_bytes` elements (PTX ISA 9.5: `T = 128 / sizeof_bits`, so 16 for 8-bit),
    /// and `k` runs to `2T` (32 for 8-bit). `RowMajor` is `mn * 2T + k`. The XOR works on the byte address, so a swizzled
    /// 8-bit offset is XORed in bytes (an element is a byte).
    #[must_use]
    pub fn element_offset_sized(&self, mn: usize, k: usize, elem_bytes: usize) -> Option<usize> {
        let t = 16 / elem_bytes;
        match *self {
            Self::RowMajor => Some(mn * 2 * t + k),
            Self::Canonical {
                major,
                swizzle,
                lbo,
                sbo,
            } => {
                let sbo = sbo as usize / elem_bytes;
                let s = 1usize << swizzle.bits();
                let pre = match (major, swizzle) {
                    (Major::K, Swizzle::None) => {
                        let lbo = lbo? as usize / elem_bytes;
                        (mn % 8) * t + (mn / 8) * sbo + k % t + (k / t) * lbo
                    }
                    (Major::MN, Swizzle::None) => {
                        let lbo = lbo? as usize / elem_bytes;
                        mn % t + (mn / t) * sbo + (k % 8) * t + (k / 8) * lbo
                    }
                    (Major::K, _) => (mn % 8) * (s * t) + (mn / 8) * sbo + k,
                    (Major::MN, _) => {
                        let lbo = lbo? as usize / elem_bytes;
                        mn % t
                            + ((mn / t) % s) * t
                            + (mn / (t * s)) * lbo
                            + (k % 8) * (s * t)
                            + (k / 8) * sbo
                    }
                };
                let byte = pre * elem_bytes;
                let swizzled = byte ^ (((byte >> 7) & (s - 1)) << 4);
                Some(swizzled / elem_bytes)
            }
        }
    }
}

/// The default `m` of a [`Stmt::Tcgen05Mma`]: 128.
const fn default_tcgen05_m() -> u32 {
    128
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
    /// Declare `local` as the element at `place` where `mask != 0`, and as `fill` where it is zero (ROADMAP R5.7 C4;
    /// PTX predicated `ld`, Triton `tl.load(mask, other)`, Mojo `masked_load(.., passthrough)`). Where the mask is false
    /// the memory is **not touched**: no bounds check, no race record, so an out-of-range index under a false mask is
    /// legal. `fill` has the element's type, and any value, not only zero.
    MaskedLoad {
        /// The local id.
        local: u32,
        /// Where to read when the mask holds.
        place: Place,
        /// The predicate: any scalar type, tested `!= 0`.
        mask: ScalarExpr,
        /// The value the local takes where the mask is false (the buffer's element type).
        fill: ScalarExpr,
    },
    /// Write one element where `mask != 0`; where it is zero nothing happens and the memory is not touched (no bounds
    /// check, no race record).
    MaskedStore {
        /// Where to write when the mask holds.
        place: Place,
        /// The value (the buffer's element type).
        value: ScalarExpr,
        /// The predicate: any scalar type, tested `!= 0`.
        mask: ScalarExpr,
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
        /// A non-semantic unroll hint (`#pragma unroll`, `[unroll]`; ROADMAP U18). Every value gives the same result; an
        /// emitter may ignore it.
        #[serde(default)]
        unroll: Unroll,
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
        /// Which barrier (`Subgroup`, `Block` and, since 0.42.13, `Device`, which synchronizes the whole grid with every block
        /// resident; since 0.42.18 `Cluster`, in a launch that names cluster dimensions). Only [`BarrierKind::Full`] exists; the field is here so a named or a
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
    /// `Device`, or `Cluster` in a launch that names cluster dimensions (`Thread` is a typed refusal).
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
        /// The scope the atomicity holds at: `Subgroup`, `Block`, `Device`, or `Cluster` in a launch that names cluster dimensions.
        scope: Scope,
    },
    /// Compare-and-exchange: if the element equals `expected`, set it to `desired`; `old` receives the
    /// value found either way. Checked (an integer element, `expected` and `desired` of its type) and run by the
    /// oracle and `unpopped-cpu-c` (`docs/options-u4-sync-and-atomics.md`, S3).
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
    /// Declare `local` as the **reduction over every device's copy** of the element a [`Place::Multicast`] names (PTX
    /// `multimem.ld_reduce`, ROADMAP U12 slice 4; the one `Stmt` U12 adds, so a breaking change). `op` combines the
    /// devices' values: `Add`, `Min` or `Max` on `I32`/`U32` (exact; `Add` wraps), `Add` on `F32`, and `Add`, `Min`
    /// or `Max` on `F16`/`Bf16`, whose sum is accumulated in `f32` and rounded to the element type once. A float `Add`
    /// has unspecified order across devices (a band, as a float atomic's; the oracle adds in device order); a NaN
    /// operand of a float `Min`/`Max` is unspecified. `order` is `Relaxed` or `Acquire` (the hardware's `weak` form is
    /// `Relaxed`); the access is at `System` scope. Each device's copy is read as an atomic access, so the read races
    /// only with a plain write, and `Acquire` takes in what a `System`-scope release (a `multimem.red`, an atomic or a
    /// barrier) published. The place must be a [`Place::Multicast`], in a launch of several devices, on a buffer the
    /// kernel can read; `local` has the element's type.
    MulticastLoadReduce {
        /// The local id.
        local: u32,
        /// The element on every device; a [`Place::Multicast`].
        place: Place,
        /// The combine.
        op: AtomicOp,
        /// The memory order: `Relaxed` or `Acquire`.
        order: MemoryOrder,
    },
    /// `griddepcontrol.wait` (programmatic dependent launch, ROADMAP U17): block this thread until every prerequisite
    /// grid of the launch has completed and its memory writes are visible. With no prerequisite grid (the only case the
    /// oracle and the serial C can see: one launch) it does nothing. Which grids are prerequisites is a launch-time
    /// property of the host API (a stream attribute or a graph edge), not of the kernel, so the IR does not carry it.
    GridDepWait,
    /// `griddepcontrol.launch_dependents`: tell the hardware that the grids that depend on this one may begin to launch
    /// once every block of this grid has executed it or exited. It is a scheduling hint: it orders nothing, and a
    /// dependent still [`Stmt::GridDepWait`]s before it reads this grid's results. The oracle and the serial C ignore it.
    GridDepLaunchDependents,
    /// A memory fence of `scope` with `order`. Since 0.42.8 the oracle runs it as the C++ fence rules do: a release
    /// fence (`Release`, `AcqRel`, `SeqCst`) before a relaxed atomic publishes what the thread knew; an acquire fence
    /// (`Acquire`, `AcqRel`, `SeqCst`) after a relaxed atomic read takes in what that read saw. `check_kernel` refuses a
    /// `Relaxed` fence and the scope `Thread`, and `Cluster` outside a launch that names cluster dimensions.
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
    ///
    /// **Patent risk note (`docs/PATENT-SWEEP.md` hit 23, ruled 2026-11-13):** US8539204B2 (NVIDIA) claims 1 and 12 recite a barrier aggregation
    /// instruction that combines a thread's value with an aggregation result, communicates the result to each thread and waits until all
    /// threads have received it. The oracle and `unpopped-cpu-c` perform this statement over simulated threads (shipped since 0.43.2);
    /// an emitter that lowers it to a hardware barrier instruction carries that exposure. Not a clearance.
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
    /// A combining collective over a **tuple** state (ROADMAP R5.7 D4b, `docs/options-d4b-tuple-state-combine.md`,
    /// option B): Triton's `tl.reduce` and `tl.associative_scan` over a tuple, such as `(value, index)` for an argmax
    /// or `(mean, m2, count)` for Welford. Every thread of `scope` contributes `values` (one scalar per state
    /// component, each its own type) and receives the folded state in the locals `results`, one per component. `kind`
    /// is [`CollectiveKind::ReduceWith`] or [`CollectiveKind::ScanWith`], and its `combine` names a module function of
    /// `2k` scalar parameters (`a0..a(k-1)` then `b0..b(k-1)`, the component types twice) that returns `k` values (its
    /// [`KernelFn::result`] then [`KernelFn::results`]), with no barrier. As for the scalar form the author asserts
    /// associativity, the oracle folds left to right in lane order, a float component is compared within a tolerance
    /// and there is no exclusive scan. A rendezvous like [`Stmt::Collective`], and a barrier site.
    CollectiveTuple {
        /// [`CollectiveKind::ReduceWith`] or [`CollectiveKind::ScanWith`].
        kind: CollectiveKind,
        /// The state components this thread contributes.
        values: Vec<ScalarExpr>,
        /// The locals that receive this thread's state, one per component, declared by this statement.
        results: Vec<u32>,
        /// The group that takes part.
        scope: Scope,
    },
    /// A warp-level matrix multiply-accumulate (ROADMAP U7 S2): every lane of `atom`'s scope contributes its
    /// fragments and receives its share of `D = A.B + C`. The fragments are the `Private` arrays `a`, `b`, `c` and
    /// `d` (indices into [`KernelFn::allocs`]; `d` may be `c`), each `atom.values(..)` elements of the atom's type.
    /// A rendezvous like [`Stmt::Collective`]: it must be reached in uniform control by the whole group. The
    /// oracle gathers the lanes' fragments into the logical matrices through the atom's layouts, computes with
    /// exact products summed in `f64` (or exact integers, wrapping at the accumulator type) and rounded once to the
    /// accumulator type, and scatters `D` back. A device may sum in another order, so compare floats within a
    /// tolerance. It orders no memory.
    Mma {
        /// The atom: shape, types and fragment layouts.
        atom: MmaAtom,
        /// The `Private` allocation holding the A fragment.
        a: u32,
        /// The `Private` allocation holding the B fragment.
        b: u32,
        /// The `Private` allocation holding the C fragment.
        c: u32,
        /// The `Private` allocation that receives the D fragment.
        d: u32,
        /// The `Private` allocations of the atom's extra operands (2:4 metadata, block scales), in the order of
        /// [`MmaAtom::extras`]; empty for a dense, unscaled atom.
        #[serde(default)]
        extra: Vec<u32>,
    },
    /// A warp-level matrix load (`ldmatrix`; ROADMAP U8b, `docs/options-u8-copy-atoms.md` S2): `count` (1, 2 or 4) 8 x 8
    /// tiles of 16-bit elements are read from a `Shared` allocation and distributed over the 32 lanes of the warp. Lane
    /// `8 * t + r` (for `t < count`, `r < 8`) supplies, in `src`, the first of the **8 contiguous elements of row `r` of
    /// tile `t`**; the other lanes' `src` is not evaluated. Lane `l` then receives, in `dst` (a `Private` allocation of
    /// `2 * count` elements of the source's type), for each tile `t`: elements `2t` and `2t + 1` are, without `trans`,
    /// row `l / 4`, columns `2 * (l % 4)` and `2 * (l % 4) + 1` of tile `t`; with `trans`, rows `2 * (l % 4)` and
    /// `2 * (l % 4) + 1`, column `l / 4` (the tile transposed). A rendezvous like [`Stmt::Mma`]: reached in uniform
    /// control by the whole warp. The 8 elements each supplying lane names are read (and race-checked) as ordinary
    /// `Shared` loads. The lane distribution is the one PTX `ldmatrix.sync.aligned.m8n8` fixes; see the source note in
    /// `docs/options-u8-copy-atoms.md`.
    ///
    /// **Patent:** see `docs/PATENT-SWEEP.md` (U8b section; no claim found, claims of adjacent hits unread): a lane-to-element distribution of a warp load
    /// is the hardware's published behaviour; this statement describes it and does not claim an implementation.
    MatrixLoad {
        /// How many 8 x 8 tiles: 1, 2 or 4.
        count: u32,
        /// Whether each tile is delivered transposed (`ldmatrix .trans`).
        trans: bool,
        /// The first element of this lane's row, when it supplies one: a `Place::Alloc` of a `Shared` allocation.
        src: Place,
        /// The `Private` allocation (index into [`KernelFn::allocs`]) that receives this lane's `2 * count` elements.
        dst: u32,
    },
    /// Initialise the `mbarrier` object at `bar` (U8c; PTX `mbarrier.init`): its arrival count is `count` (at least 1), its
    /// pending arrivals are `count`, its transaction count is 0 and its phase is 0. `bar` is a place of a `Shared`
    /// allocation of 64-bit elements; the oracle keeps the object's state beside the element (reading the element as data
    /// is not modelled), and the initialisation is a logged write to that element, so another thread's use needs a barrier
    /// after it, as on hardware. Not a rendezvous.
    ///
    /// **Patent:** `mbarrier` is the published PTX synchronisation object; this describes its behaviour and claims no
    /// implementation. No patent search specific to it was run (`docs/PATENT-SWEEP.md`, U8c section).
    ///
    /// **Patent note (CireSnave 2026-11-16):** lowering this to a hardware operation on non-NVIDIA hardware may implicate
    /// NVIDIA's US11803380B2; see `docs/PATENT-SWEEP.md`. Emitters for other manufacturers decline it.
    MbarInit {
        /// The `Shared` 64-bit element holding the object.
        bar: Place,
        /// The arrival count, at least 1.
        count: ScalarExpr,
    },
    /// Arrive at the `mbarrier` at `bar` (`mbarrier.arrive`): one pending arrival is consumed, after first adding `tx`
    /// bytes to the transaction count when given (`arrive.expect_tx`). When no arrival is pending and the transaction
    /// count is 0 the phase completes: the phase number advances and the pending arrivals return to the arrival count.
    /// A release-class atomic access at `Block` scope: what the thread wrote before it is visible to a thread whose
    /// [`Stmt::MbarWait`] later sees the phase complete. Arriving at an uninitialised barrier is
    /// `KernelTrap::MbarrierMisuse`.
    ///
    /// **Patent note (CireSnave 2026-11-16):** lowering this to a hardware operation on non-NVIDIA hardware may implicate
    /// NVIDIA's US11803380B2; see `docs/PATENT-SWEEP.md`. Emitters for other manufacturers decline it.
    MbarArrive {
        /// The barrier.
        bar: Place,
        /// Bytes to add to the transaction count first (`expect_tx`), when the arrival also announces a transfer.
        tx: Option<ScalarExpr>,
    },
    /// Subtract `bytes` from the transaction count of the `mbarrier` at `bar` (`mbarrier.complete_tx`): what an
    /// asynchronous copy does as its data lands. Completes the phase when it brings the count to 0 with no arrival
    /// pending. A release-class atomic access like [`Stmt::MbarArrive`].
    ///
    /// **Patent note (CireSnave 2026-11-16):** lowering this to a hardware operation on non-NVIDIA hardware may implicate
    /// NVIDIA's US11803380B2; see `docs/PATENT-SWEEP.md`. Emitters for other manufacturers decline it.
    MbarCompleteTx {
        /// The barrier.
        bar: Place,
        /// The bytes that landed.
        bytes: ScalarExpr,
    },
    /// Wait until the phase of the `mbarrier` at `bar` whose parity is `parity` (0 or 1) has completed
    /// (`mbarrier.try_wait.parity` in a loop): true once the barrier's current phase number has the other parity, so
    /// `parity` 0 waits for the first phase and `parity` 1 passes at once on a fresh barrier (the previous phase of
    /// the first one is complete by convention). The thread blocks; it resumes when some other thread's arrival
    /// completes the phase, and a block in which every remaining thread waits is `KernelTrap::MbarrierDeadlock`. An
    /// acquire-class atomic access at `Block` scope. Not a rendezvous: any subset of the block's threads may wait.
    ///
    /// **Patent note (CireSnave 2026-11-16):** lowering this to a hardware operation on non-NVIDIA hardware may implicate
    /// NVIDIA's US11803380B2; see `docs/PATENT-SWEEP.md`. Emitters for other manufacturers decline it.
    MbarWait {
        /// The barrier.
        bar: Place,
        /// Which phase parity to wait for: 0 or 1.
        parity: ScalarExpr,
    },
    /// A warp-level matrix store (`stmatrix`; ROADMAP U8b slice 2): the inverse of [`Stmt::MatrixLoad`]. Lane `l` holds
    /// in `src` (a `Private` allocation of `2 * count` 16-bit elements) the fragment it would have received from a load
    /// of the same `count` and `trans`; lane `8 * t + r` (`t < count`, `r < 8`) supplies in `dst` (a place of a
    /// `Shared` allocation of the same element type) the first of the **8 contiguous elements of row `r` of tile `t`**
    /// and **writes** them, so the elements the warp assembled land where the supplying lanes point (the other lanes'
    /// `dst` is not evaluated). The writes are ordinary logged `Shared` writes by the supplying lane, ordered by the
    /// next barrier like any other store; this statement orders no memory itself. A rendezvous like
    /// [`Stmt::MatrixLoad`], reached in uniform control by the whole warp. `stmatrix` exists from `sm_90`.
    ///
    /// **Patent:** see `docs/PATENT-SWEEP.md` (U8b section, the same read as `MatrixLoad`; claims of adjacent hits
    /// unread): this describes the published instruction behaviour and does not claim an implementation.
    MatrixStore {
        /// How many 8 x 8 tiles: 1, 2 or 4.
        count: u32,
        /// Whether each tile is stored transposed (`stmatrix .trans`).
        trans: bool,
        /// The first element of this lane's row, when it supplies one: a `Place::Alloc` of a `Shared` allocation.
        dst: Place,
        /// The `Private` allocation (index into [`KernelFn::allocs`]) holding this lane's `2 * count` elements.
        src: u32,
    },
    /// Load a tile (ROADMAP T2, `docs/design-opaque-tile-mma.md`): the `rows x cols` elements of the [`MemorySpace::Tile`]
    /// allocation `d` are read from `src`: element `(r, c)` is the element at `src + (origin[0] + r) * elements_per_row +
    /// origin[1] + c`, in `src`'s allocation or buffer (`src` is a `Place::Alloc` of a `Shared` allocation or a one-index
    /// `Place::Logical`; its index names the element the offsets count from). Every element is an ordinary logged load. A
    /// tile is held by the whole subgroup and no element of it is addressable, so the statement is reached in uniform control
    /// (a divergent one is `BarrierDivergent`) and every lane of the subgroup executes it identically. The element type of
    /// `src` is the tile's.
    ///
    /// **Patent:** lifted by CireSnave 2026-11-26 (board 519 B) as his risk call: the PM did not support it with a reading; not
    /// a clearance; no attorney; the Intel/NVIDIA tile-instruction method and medium claims (US11086623B2 = app 16/487,787,
    /// US11288069B2 = 16/487,755, US10884734B2 = 16/459,191) remain NEEDS MORE as read. There is no `transpose` option and no
    /// transpose statement: US12405770B2 claims 8 and 15 are NEEDS MORE against one (findings, `docs/PATENT-SWEEP.md`).
    TileLoad {
        /// The tile (index into [`KernelFn::allocs`], a [`MemorySpace::Tile`] allocation).
        d: u32,
        /// The element the offsets count from.
        src: Place,
        /// The distance in elements between two rows of the source.
        elements_per_row: ScalarExpr,
        /// The row and the column of the source the tile starts at.
        origin: [ScalarExpr; 2],
    },
    /// `d = a * b + c` over tiles: `a` is `m x k`, `b` is `k x n`, `c` and `d` are `m x n` (`d` may be `c`). The element of
    /// `D` is the sum of `C` and the `k` products, the products exact, summed in ascending `k` and rounded **once** to the
    /// accumulator type (the contract of [`Stmt::Mma`]), so a device that sums in another order or at another width is compared
    /// within a tolerance, never by bytes. `a` and `b` have one element type and `c` and `d` another (`F32`, or `F64` over `F64`).
    /// Uniform control, like [`Stmt::TileLoad`]. The tile patent text on [`Stmt::TileLoad`] applies.
    TileMma {
        /// The result tile.
        d: u32,
        /// The left operand.
        a: u32,
        /// The right operand.
        b: u32,
        /// The accumulator.
        c: u32,
    },
    /// Set every element of the tile `d` to `value` (converted to the tile's element type with the rounding of an `Assign`).
    /// Uniform control. The tile patent text on [`Stmt::TileLoad`] applies.
    TileFill {
        /// The tile.
        d: u32,
        /// The value.
        value: ScalarExpr,
    },
    /// Store the tile `a` to `dst`, the inverse of [`Stmt::TileLoad`]: element `(r, c)` goes to `dst + (origin[0] + r) *
    /// elements_per_row + origin[1] + c`. The subgroup's first lane performs the writes (one logged writer, as on hardware
    /// where one operation stores the tile); uniform control. Every element of the tile must have been written (by a load,
    /// a fill or an MMA). The tile patent text on [`Stmt::TileLoad`] applies.
    TileStore {
        /// The tile.
        a: u32,
        /// The element the offsets count from.
        dst: Place,
        /// The distance in elements between two rows of the destination.
        elements_per_row: ScalarExpr,
        /// The row and the column of the destination the tile starts at.
        origin: [ScalarExpr; 2],
    },
    /// `fence.proxy.async.shared::cta` (ROADMAP U8c, slice 0): order this thread's earlier **generic-proxy** writes to
    /// `Shared` memory before later **async-proxy** reads of the same memory by [`Stmt::Wgmma`], [`Stmt::Tcgen05Mma`]
    /// and the like. PTX ISA 9.5: "Accessing the same memory location across multiple proxies needs a cross-proxy
    /// fence. For the async proxy, fence.proxy.async should be used to synchronize memory between generic proxy and the
    /// async proxy" (9.7.17.4, 9.7.10.28.2), and for `wgmma` "An async proxy fence must be used to establish an
    /// ordering between prior writes to shared memory matrices and subsequent reads of the same matrices in a
    /// wgmma.mma_async instruction" (9.7.17.7.1). The fence "take\[s\] effect within a single thread and therefore do\[es\]
    /// not have a .scope qualifier": it covers the writes **this thread** made, and its effect composes with a barrier
    /// as the memory model says (so a thread that reads another thread's stores still needs the barrier, and that
    /// writer needs its own `ProxyFence`). The completion of an asynchronous-proxy operation (`CopyBulk`, a `wgmma`)
    /// is followed by an implicit fence the other way (9.7.17.4), so the bytes a `CopyBulk` wrote need none. The
    /// oracle traps a missing fence: [`KernelTrap::MissingProxyFence`](crate::oracle::kernel_eval::KernelTrap).
    /// `check_kernel` admits only `Shared` (the `.global` and `.shared::cluster` forms are not modelled). Not modelled:
    /// the opposite hazard (an async-proxy read followed by a generic write of the same bytes).
    ProxyFence {
        /// The state space the fence orders: `Shared` only.
        space: MemorySpace,
    },
    /// A warpgroup-level asynchronous matrix multiply-accumulate (`wgmma.mma_async.m64nNk16`; ROADMAP U7 S4, **option
    /// A: no swizzle and no descriptor bits are modelled, and there was no ISA text to check them against**). The 128
    /// threads of a warpgroup (a block of `Scope::Warpgroup` groups, 32-lane subgroups, a launch block `[x, 1, 1]` with
    /// `x` a multiple of 128) issue it together, in uniform control: `D = A.B + C` for a 64 x 16 `A` at `a` (row-major,
    /// K-major) and an `n` x 16 `B` at `b` (row-major, K-major), both places of `Shared` allocations of `ab` (16-bit
    /// float) elements, with `D` the thread's `Private` fragment `d` of `n / 2` `acc` elements in the `wgmma` layout
    /// (see `oracle/kernel_wgmma.rs`). `scale_d` zero overwrites `D` instead of accumulating into it. The operand
    /// tiles are read when the statement executes (ordinary logged `Shared` reads, so they must be ordered after the
    /// stores that fill them); the result **lands at the [`Stmt::WgmmaWait`] that covers the group a
    /// [`Stmt::WgmmaCommit`] closed**, and an operation never committed is never waited for. Reading `d` before then
    /// returns its old value and is **not** trapped. Register-resident `A`, transposed operands, 8-bit and `tf32`
    /// inputs, and a half-precision accumulator are not modelled.
    ///
    /// **Patent:** see `docs/PATENT-SWEEP.md` (U7 S4 section): this describes the published instruction behaviour
    /// and does not claim an implementation.
    Wgmma {
        /// Columns of `B` and `D`: a multiple of 8, 8 to 256.
        n: u32,
        /// The input element type: `F16` or `Bf16`.
        #[serde(with = "crate::text::ek")]
        ab: ElementKind,
        /// The accumulator element type: `F32`.
        #[serde(with = "crate::text::ek")]
        acc: ElementKind,
        /// The first element of the `A` tile: a `Place::Alloc` of a `Shared` allocation.
        a: Place,
        /// The first element of the `B` tile: a `Place::Alloc` of a `Shared` allocation.
        b: Place,
        /// The `Private` allocation (index into [`KernelFn::allocs`]) holding this thread's `D` fragment.
        d: u32,
        /// Zero: `D = A.B`; nonzero: `D = A.B + D`. An integer expression.
        scale_d: ScalarExpr,
        /// How the `A` tile sits in its allocation (default `RowMajor`, the meaning before this field).
        #[serde(default)]
        a_layout: OperandLayout,
        /// How the `B` tile sits in its allocation (default `RowMajor`).
        #[serde(default)]
        b_layout: OperandLayout,
        /// `A` from **registers** (`wgmma.mma_async` with `A` in registers; PTX ISA 9.5 9.7.17.5.1.2, Figure 151): the
        /// `Private` allocation (index into [`KernelFn::allocs`]) holding this thread's fragment of `A`, 8 elements of
        /// `ab` `a0 .. a7` ("four .f16x2 registers"). Element `i` of the thread `l` of warp `w` is row
        /// `16 w + l / 4 + 8 * ((i / 2) % 2)`, column `2 * (l % 4) + i % 2 + 8 * (i / 4)` of the 64 x 16 `A`. When
        /// `Some`, `a` and `a_layout` are not read (give `a` as the fragment's own `Place::Alloc`, index 0) and a layout
        /// on `A` is refused by the checker. The fragment is read when the statement executes. Register-`A` has no
        /// transposed form (the ISA's register-`A` syntax has only `imm-trans-b`), and `wgmma.fence` before it is the
        /// emitter's, as for the accumulator.
        #[serde(default)]
        a_regs: Option<u32>,
    },
    /// Close this thread's open [`Stmt::Wgmma`] operations into one group (`wgmma.commit_group`). A commit with no
    /// open operation still makes a group, so that wait counts line up.
    WgmmaCommit,
    /// Wait until at most `pending` of this thread's committed [`Stmt::Wgmma`] groups are still in flight
    /// (`wgmma.wait_group`); the older ones land in their `d` fragments.
    ///
    /// **Patent note (CireSnave 2026-11-16):** lowering this to a hardware operation on non-NVIDIA hardware may implicate
    /// NVIDIA's US12204897B2 and its continuation US12710961B2 (claims 1, 9, 15 and 9, 15 name no hardware by their words);
    /// the oracle lands the results here and does not block, and no fence is modelled. See `docs/PATENT-SWEEP.md`.
    /// Emitters for other manufacturers decline it.
    WgmmaWait {
        /// How many of the newest groups may stay in flight.
        pending: u32,
    },
    /// One thread issues a `tcgen05.mma`: `D = A.B + scale_d * D`, a 128 x 16 `A` tile times an `n x 16` `B` tile (both
    /// K-major, row-major, in `Shared` allocations of the 16-bit float type `ab`), into the `128 x n` block of the
    /// **tensor-memory** allocation `d` (`MemorySpace::Tensor`) starting at column `d_col`: row `m` of `D` is lane `m`.
    /// Products are exact in `f64`, summed in ascending `k` from the old `D` (or from zero when `scale_d` is 0) and
    /// rounded once to `F32`, as for [`Stmt::Mma`]. The oracle reads the tiles and writes `D` at issue as logged accesses
    /// by the issuing thread, so a thread that reads `D` without an acquire that follows a [`Stmt::Tcgen05Commit`] is a
    /// `DataRace`. **Not modelled, and so not to be counted as covered:** instruction and shared-memory descriptors and
    /// swizzle modes (no ISA text was read), `tcgen05.alloc`/`dealloc` (the allocation is static), tensor-memory `A`,
    /// `cta_group::2` pairs, transposed operands, 8-bit/`tf32`/sparse/block-scaled kinds, half-precision accumulate,
    /// `tcgen05.st`/`cp`/`shift`, and ordering of an in-flight product against a later product.
    ///
    /// **Patent:** see `docs/PATENT-SWEEP.md` (U7 S5 section): this describes the published instruction behaviour and
    /// does not claim an implementation.
    Tcgen05Mma {
        /// Columns of `B` and `D`: a multiple of 8, 8 to 256.
        n: u32,
        /// The input element type: `F16` or `Bf16`.
        #[serde(with = "crate::text::ek")]
        ab: ElementKind,
        /// The accumulator element type (`dtype`, instruction-descriptor bits 4-5; `F16 = 0` and `F32 = 1` for `.kind::f16` and
        /// `.kind::f8f6f4`, PTX ISA 9.5 Tables 52-53; CUTLASS `CFormat` agrees): `F32`, or `F16`. With `F16`, "the
        /// sub-word elements of matrix D are expected not to be packed within a 32-bit Tensor Memory word ... a Tensor
        /// Memory word would contain a single 16-bit element in its lower 16 bits" (PTX 9.7.18.10.5): the `Tensor`
        /// allocation is `U32` (words), the oracle writes the binary16 bits into the low 16 bits and **zero** into the upper
        /// 16 (PTX does not say what they hold: a kernel must not read them). The sum is rounded to binary16 once, from the
        /// exact `f64` sum plus the old `D` (PTX does not say how an `F16` accumulator rounds inside the instruction: this is
        /// the oracle's contract, not a measurement of hardware).
        #[serde(with = "crate::text::ek")]
        acc: ElementKind,
        /// The first element of the `A` tile: a `Place::Alloc` of a `Shared` allocation.
        a: Place,
        /// The first element of the `B` tile: a `Place::Alloc` of a `Shared` allocation.
        b: Place,
        /// The `Tensor` allocation (index into [`KernelFn::allocs`]) holding `D`.
        d: u32,
        /// The first column of `D` in tensor memory. An integer expression.
        d_col: ScalarExpr,
        /// Zero: `D = A.B`; nonzero: `D = A.B + D`. An integer expression.
        scale_d: ScalarExpr,
        /// How the `m`-row `A` tile sits in its allocation (default `RowMajor`; see [`OperandLayout`]).
        #[serde(default)]
        a_layout: OperandLayout,
        /// How the `B` tile sits in its allocation (default `RowMajor`).
        #[serde(default)]
        b_layout: OperandLayout,
        /// Rows of `A` and `D`: 128 (the default) or 64 (`tcgen05.mma.cta_group::1` with `M = 64`, non-`.ws`, **Layout F**,
        /// PTX ISA 9.5 9.7.18.10.5.6, Figures 221 and 222: "4x1 (1/2 datapath utilized)"). With `m = 128` row `r` of `D` is
        /// tensor-memory lane `r`. With `m = 64` row `r` is lane `32 * (r / 16) + d_lane + r % 16`: rows `16 w .. 16 w + 15`
        /// fill 16 of the 32 lanes of warp `w`'s quadrant, the lower half (`d_lane` 0) or the upper half (`d_lane` 16).
        #[serde(default = "default_tcgen05_m")]
        m: u32,
        /// The tensor-memory lane alignment of `D` for `m = 64`: 0 or 16 (the table's "Layout 0 or 16"; Figure 222's
        /// addresses `0x0000.0XXX | 0x0010.0XXX`, ...). Must be 0 for `m = 128`.
        #[serde(default)]
        d_lane: u32,
        /// The element type of `B` when it differs from `A`'s (`ab`): `None` (the default) means `ab`. PTX ISA 9.5 Table 53
        /// gives `.kind::f8f6f4` separate `atype` and `btype` fields (bits 7-9 and 10-12), so `A` and `B` may be different
        /// 8-bit float types (`Fp8E4M3FN`, `Fp8E5M2`); CUTLASS `mma_sm100_desc.hpp` agrees. For `F16`/`Bf16` the two must
        /// be the same type (`.kind::f16` has one `atype`/`btype` pair and PTX names no mixed form here). With 8-bit
        /// inputs `K` is 32 per instruction (`.kind::f8f6f4` dense) and a `Canonical` layout's core-matrix row is 16
        /// elements. `acc` is `F32` (a `Tensor` allocation of `F32`, `I32` or `U32`) or `F16` (see `acc`).
        #[serde(default, with = "crate::text::ek_opt")]
        b_ty: Option<ElementKind>,
        /// Negate every element of `A` (instruction-descriptor bit 13, PTX ISA 9.5 Tables 52-54 and 9.7.18.10.1 Table 64:
        /// supported for `.kind::f16`, `.kind::tf32` and `.kind::f8f6f4`; CUTLASS `mma_sm100_desc.hpp` `a_negate_` agrees,
        /// "valid only for F32F16Format and MXF8F6F4Format"). The product's sign flips; both negated cancel.
        #[serde(default)]
        negate_a: bool,
        /// Negate every element of `B` (bit 14); see `negate_a`.
        #[serde(default)]
        negate_b: bool,
    },
    /// `tcgen05.alloc.cta_group::1` (ROADMAP U8c slice 4): allocate `ncols` columns of tensor memory, the whole of the
    /// declared `Tensor` allocation `d` (`128 * ncols` elements). PTX ISA 9.5 9.7.18.1.2 and 9.7.18.7.1: "The unit of
    /// allocation is 32 columns and the number of columns being allocated must be a power of 2", `nCols` "in the range
    /// [32, 512]", "All of the Tensor Memory that was allocated in a kernel, must be explicitly deallocated before the
    /// kernel exits", "it is illegal for a CTA to perform tcgen05.alloc after any of its constituent threads execute
    /// tcgen05.relinquish_alloc_permit", "The number of columns allocated must not increase between any two allocations
    /// in the execution order within the CTA". `.sync.aligned` means a whole warp issues it: the oracle acts on the
    /// warp's first lane and ignores the other lanes. A `Tcgen05Mma` or `Tcgen05Ld` on a `Tensor` allocation that is not
    /// allocated traps (`KernelTrap::TmemMisuse`). The address `tcgen05.alloc` writes to `[dst]` is not modelled: the
    /// allocation is named by its index and addressed by column. `.exclusive` and `cta_group::2` are not modelled.
    Tcgen05Alloc {
        /// The `Tensor` allocation (index into [`KernelFn::allocs`]).
        d: u32,
        /// Columns: a power of two from 32 to 512, equal to the allocation's declared columns.
        ncols: u32,
    },
    /// `tcgen05.dealloc.cta_group::1`: release the `ncols` columns of the allocated `Tensor` allocation `d` (the same
    /// count it was allocated with). Issued by one warp, as [`Stmt::Tcgen05Alloc`].
    Tcgen05Dealloc {
        /// The `Tensor` allocation.
        d: u32,
        /// Columns, as allocated.
        ncols: u32,
    },
    /// `tcgen05.relinquish_alloc_permit.cta_group::1`: this CTA will allocate no more tensor memory. A later
    /// [`Stmt::Tcgen05Alloc`] in the block traps.
    Tcgen05RelinquishPermit,
    /// `tcgen05.commit`: arrive (a release-class arrival, like [`Stmt::MbarArrive`] without a transaction count) on the
    /// `mbarrier` at `bar` once the thread's earlier [`Stmt::Tcgen05Mma`] have completed. In the oracle they completed at
    /// issue.
    ///
    /// **Patent note (CireSnave 2026-11-16):** lowering this to a hardware operation on non-NVIDIA hardware may implicate
    /// NVIDIA's US11803380B2; see `docs/PATENT-SWEEP.md`. Emitters for other manufacturers decline it.
    Tcgen05Commit {
        /// The barrier: a `Place::Alloc` of a `Shared` 64-bit element.
        bar: Place,
    },
    /// `tcgen05.ld.32x32b`: each thread reads `count` consecutive columns from `col` of its tensor-memory lane into
    /// elements `0..count` of its `Private` allocation `dst`. Warp `w` of the block reaches lanes `32 * (w % 4)` ..
    /// `+ 32`, and its lane `l` reads lane `32 * (w % 4) + l`. Not a rendezvous in the oracle (the reads are logged
    /// accesses, ordered by the barrier the consumer waited on); `tcgen05.wait::ld`, `.16x64b` and the other shapes are not
    /// modelled.
    Tcgen05Ld {
        /// The `Tensor` allocation read.
        src: u32,
        /// The first column. An integer expression.
        col: ScalarExpr,
        /// How many columns: a power of two, 1 to 128.
        count: u32,
        /// The `Private` allocation receiving them (at least `count` elements of the tensor element type).
        dst: u32,
    },
    /// An asynchronous copy of `elems` contiguous elements from the global buffer place `src` to the `Shared`
    /// allocation place `dst`, issued by one thread (`cp.async`; ROADMAP U8 S1, `docs/options-u8-copy-atoms.md`). It
    /// lands only when a later [`Stmt::AsyncWait`] covers its group: `elems` times the element size is 4, 8 or 16
    /// bytes. The oracle reads the source at issue and writes the destination at the wait (issue and landing are both
    /// logged writes, as the hardware may write any time after issue), so reading or
    /// overwriting those elements in between is the typed `KernelTrap::AsyncRace`; a copy never waited for is
    /// `KernelTrap::AsyncUnwaited`. `unpopped-cpu-c` performs the copy at issue (a correct kernel cannot tell).
    ///
    /// **Patent risk note (`docs/PATENT-SWEEP.md` hit 22, ruled 2026-11-13):** US11294713B2 and US12118382B2 (NVIDIA) claim an API that blocks
    /// a thread's instructions while other instructions of the same thread run, around an asynchronous shared-memory copy. This crate only
    /// describes the statement and its oracle runs a thread in order; an emitter that lowers it to a device wait carries that exposure.
    /// Not a clearance.
    CopyAsync {
        /// The first destination element: a `Shared` allocation place.
        dst: Place,
        /// The first source element: a place of a readable global buffer of the same element type.
        src: Place,
        /// How many consecutive elements.
        elems: u32,
        /// How many of the `elems` source elements are read (`cp.async ... src-size`; ROADMAP U13): the rest of the
        /// destination is **zero-filled**, and `0` reads nothing at all, so a predicated-off copy may point anywhere. `None` is
        /// all of them. An integer expression; above `elems` is a trap.
        #[serde(default)]
        valid: Option<ScalarExpr>,
        /// The cache operator (`.ca`, `.cg`); the oracle ignores it.
        #[serde(default)]
        cache: CacheHint,
        /// The L2 prefetch size in bytes (`.L2::64B`, `128B`, `256B`); the oracle ignores it.
        #[serde(default)]
        l2_prefetch: Option<u32>,
    },
    /// A bulk copy of `elems` contiguous elements from the global buffer place `src` to the `Shared` allocation place
    /// `dst`, issued by one thread, that **completes on an `mbarrier`** (`cp.async.bulk.shared::cluster.global.
    /// mbarrier::complete_tx::bytes`; ROADMAP U8c slice 2). The copy is `elems` times the element size bytes, a multiple of
    /// 16. When the data has landed the hardware does a [`Stmt::MbarCompleteTx`] of those bytes on `bar`; the issuing
    /// thread (or another) announces them with [`Stmt::MbarArrive`]'s `tx`, and a consumer that
    /// [`Stmt::MbarWait`]s on the phase then sees the tile. The oracle lands the data at issue (a logged `Shared` write by
    /// the issuing thread) and then does the `complete_tx`, a release-class access, so a thread that reads the tile
    /// without an acquire on the barrier is a `DataRace`. This is the 1-D bulk form: a tensor-map (descriptor) copy is a
    /// later slice. `unpopped-cpu-c` refuses it (it cannot block a thread on the barrier).
    ///
    /// **Patent:** see `docs/PATENT-SWEEP.md` (U8c section): this describes the published instruction behaviour and does
    /// not claim an implementation.
    ///
    /// **Patent note (CireSnave 2026-11-16):** lowering this to a hardware operation on non-NVIDIA hardware may implicate
    /// NVIDIA's US11803380B2; see `docs/PATENT-SWEEP.md`. Emitters for other manufacturers decline it.
    CopyBulk {
        /// The first destination element: a `Shared` allocation place.
        dst: Place,
        /// The first source element: a place of a readable global buffer of the same element type.
        src: Place,
        /// How many consecutive elements; `elems` times the element size is a multiple of 16 bytes.
        elems: u32,
        /// The `mbarrier` (a `Shared` 64-bit element) the copy completes on.
        bar: Place,
    },
    /// Fill a bit-addressed FP4 tile: copy `groups` groups of 16 `F4E2M1` elements from the packed global buffer place `src`
    /// into the **byte** (`U8`) `Shared` allocation place `dst`, one 16-byte slot per group: the 8 data bytes (KISS packing,
    /// two per byte, low nibble = even index) then 8 bytes of padding (PTX ISA 9.5 Figure 207, "16 elements of 4-bit
    /// floating point type, with 64-bit padding at the end"), issued by one thread and completing on an `mbarrier` like
    /// [`Stmt::CopyBulk`] (`groups * 16` bytes). It moves whole bytes, so it is not a sub-byte `Store` (U8c design note
    /// 5j; U9b's read-only sub-byte buffers are unchanged). The pad bytes are written zero by the oracle and are
    /// unspecified by PTX. FP6 sources wait on a KISS pin for their buffer packing. **Tags:** the padding rule is PTX's for
    /// `.kind::mxf8f6f4`; for the unscaled `.kind::f8f6f4` it is "inferred, CUTLASS-shared tags, PTX silent" (PM ruling
    /// 2026-11-26), and the nibble order inside the slot is assumed to carry over from the buffer packing.
    ///
    /// **Patent note:** the patent note on [`Stmt::CopyBulk`] applies; emitters for other manufacturers decline it.
    CopyUnpack {
        /// The first destination byte: a `Shared` `U8` allocation place.
        dst: Place,
        /// The first source element: a place of a readable global `F4E2M1` buffer.
        src: Place,
        /// How many groups of 16 elements (each becoming one 16-byte slot).
        groups: u32,
        /// The `mbarrier` (a `Shared` 64-bit element) the copy completes on.
        bar: Place,
    },
    /// Close this thread's open [`Stmt::CopyAsync`] copies into one group (`cp.async.commit_group`). A commit with
    /// no open copy still makes a group, as on hardware, so that wait counts line up.
    ///
    /// **Patent risk note (`docs/PATENT-SWEEP.md` hit 22, ruled 2026-11-13):** US11294713B2 and US12118382B2 (NVIDIA) claim an API that blocks
    /// a thread's instructions while other instructions of the same thread run, around an asynchronous shared-memory copy. This crate only
    /// describes the statement and its oracle runs a thread in order; an emitter that lowers it to a device wait carries that exposure.
    /// Not a clearance.
    AsyncCommit,
    /// Wait until at most `pending` of this thread's committed groups are still in flight
    /// (`cp.async.wait_group`); the older ones land. Copies not yet committed are not waited for.
    ///
    /// **Patent risk note (`docs/PATENT-SWEEP.md` hit 22, ruled 2026-11-13):** US11294713B2 and US12118382B2 (NVIDIA) claim an API that blocks
    /// a thread's instructions while other instructions of the same thread run, around an asynchronous shared-memory copy. This crate only
    /// describes the statement and its oracle runs a thread in order; an emitter that lowers it to a device wait carries that exposure.
    /// Not a clearance.
    AsyncWait {
        /// How many of the newest groups may stay in flight.
        pending: u32,
    },
    /// Call another kernel of the same [`Module`]: `result = callee(args...)`. The callee is a
    /// [`KernelFn`] (PM ruling 2026-11-02 on `docs/options-u3-remainder.md`, R1a); `callee` indexes
    /// [`Module::fns`]. Alone, `check_kernel` refuses it (the callee lives in the module); `check_module` resolves it
    /// (acyclic call graph; a callee holding a `Barrier` called from uniform control only) and `run_module` runs it.
    /// It entered in the 0.41.0 break so that `Stmt` broke downstream once.
    Call {
        /// Index of the callee in the module's functions.
        callee: u32,
        /// The arguments, in the callee's parameter order (`docs/options-u3-calls.md`, Q2).
        args: Vec<CallArg>,
        /// The local that receives the callee's result, if it has one.
        result: Option<u32>,
    },
    /// Call a kernel of the same [`Module`] that returns several values (ROADMAP R5.7 D4b,
    /// `docs/options-d4b-tuple-state-combine.md`, option B): `results = callee(args...)`. The callee's results are
    /// its [`KernelFn::result`] (if any) followed by its [`KernelFn::results`], and `results` names one local per
    /// value, in that order, each declared by this statement. A [`Stmt::Call`] still takes the first value only.
    /// Checked by `check_module`, run by `run_module` and `run_launch_module`.
    CallTuple {
        /// Index of the callee in the module's functions.
        callee: u32,
        /// The arguments, in the callee's parameter order.
        args: Vec<CallArg>,
        /// The locals that receive the callee's values, one per value, in order.
        results: Vec<u32>,
    },
    /// Call the opaque [`ExternDecl`] `decl` of the module (ROADMAP U11; `docs/options-u11-extern-and-inline-asm.md`):
    /// `results = decl(args...)`. A statement, not an expression, because an effectful call has an order. The
    /// declaration states the effects (`pure`, `sync`, `volatile`), which the checker holds it to and which every
    /// pass honours: a call is opaque, never fused across, and a `sync` call is a barrier site like a `Barrier`.
    /// `check_kernel` alone refuses it (the declaration lives in the module); `check_module` resolves it. The
    /// oracle and `unpopped-cpu-c` run the declaration's `reference` kernel in its place and refuse when it has none.
    ExternCall {
        /// Index of the declaration in [`Module::externs`].
        decl: u32,
        /// The scalar arguments, in the declaration's parameter order.
        args: Vec<ScalarExpr>,
        /// The locals that receive the results, one per declared result, declared by this statement.
        results: Vec<u32>,
    },
}

/// One argument of a [`Stmt::Call`], matched to the callee's parameter of the same position
/// (`docs/options-u3-calls.md`, Q2; PM ruling 2026-11-02). Checked by `check_module` and run by `run_module`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum CallArg {
    /// A scalar passed by value to a [`ParamKind::Scalar`] parameter.
    Scalar(ScalarExpr),
    /// The caller's buffer parameter (by index), passed by alias to a [`ParamKind::Buffer`] parameter.
    Buffer(u32),
    /// The caller's allocation (by index into [`KernelFn::allocs`]), passed by alias.
    Alloc(u32),
    /// A struct passed by value to a [`ParamKind::Struct`] parameter (ROADMAP U23): one expression per field, in the
    /// callee's field order, each of that field's type. Layout only: the callee sees its fields as scalars.
    Struct(Vec<ScalarExpr>),
}

/// A set of kernels, any of which may [`Stmt::Call`] another. Checked by `check_module` and run
/// by `run_module` (unlaunched entries) and by `run_launch` (launched).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Module {
    /// The kernels; a call names one by index.
    pub fns: Vec<KernelFn>,
    /// The opaque declarations a [`Stmt::ExternCall`] names, by index.
    #[serde(default)]
    pub externs: Vec<ExternDecl>,
}

impl Module {
    /// A module of `fns` with no extern declarations.
    #[must_use]
    pub fn new(fns: Vec<KernelFn>) -> Self {
        Self {
            fns,
            externs: Vec::new(),
        }
    }

    /// This module with the extern declarations `externs`.
    #[must_use]
    pub fn with_externs(mut self, externs: Vec<ExternDecl>) -> Self {
        self.externs = externs;
        self
    }
}

/// What an opaque call is allowed to do, **declared** by whoever writes it (Unpopped cannot look inside asm). The
/// checker holds a declaration to its own words (`check_module`): `pure` excludes everything else, `sync` makes a
/// call a barrier site, and a call that is not `pure` has results that may differ between threads.
///
/// v1 has no pointer operands, so `reads` and `writes` must be empty; they exist so that memory-touching asm
/// (which needs U8's descriptors) is not a second break of [`ExternDecl`].
#[derive(Clone, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Effects {
    /// No effect at all: the call may be reordered, duplicated or dropped if its results are unused.
    pub pure: bool,
    /// Memory spaces it may read through a pointer operand (empty in v1).
    pub reads: Vec<MemorySpace>,
    /// Memory spaces it may write through a pointer operand (empty in v1).
    pub writes: Vec<MemorySpace>,
    /// A rendezvous at this scope: the call is a barrier site, reached in uniform control by the whole group.
    pub sync: Option<Scope>,
    /// Must not be moved or removed (CUDA `asm volatile`).
    pub volatile: bool,
}

impl Effects {
    /// A pure call.
    #[must_use]
    pub fn pure() -> Self {
        Self {
            pure: true,
            ..Self::default()
        }
    }

    /// A call with an effect Unpopped does not model: not pure, never removed or moved.
    #[must_use]
    pub fn opaque() -> Self {
        Self {
            volatile: true,
            ..Self::default()
        }
    }

    /// This call also rendezvouses at `scope`.
    #[must_use]
    pub fn with_sync(mut self, scope: Scope) -> Self {
        self.sync = Some(scope);
        self
    }
}

/// The text of one target's body of an [`ExternDecl`]. Unpopped never parses it and cannot check it is correct:
/// only its shape (`check_module`) and, on a device leg, its result against the declaration's `reference`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum BodyText {
    /// Inline assembly (PTX, SPIR-V asm, AMDGCN). Operands are `{0}`, `{1}`, ... : the parameters first, then the
    /// results.
    Asm {
        /// The template.
        template: String,
        /// The target's constraint string (`"=r,r"`), opaque.
        constraints: String,
    },
    /// A symbol of a library the target links (libdevice, ocml, a linked object).
    Symbol {
        /// The library.
        library: String,
        /// The symbol.
        symbol: String,
    },
}

/// A body for one target. `target` is the backend's own name (`"cuda-ptx"`, `"spirv"`); a target nobody has is never used.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct TargetBody {
    /// The target's key.
    pub target: String,
    /// The body.
    pub body: BodyText,
}

impl TargetBody {
    /// A body for `target`.
    #[must_use]
    pub fn new(target: impl Into<String>, body: BodyText) -> Self {
        Self {
            target: target.into(),
            body,
        }
    }
}

/// An opaque function the IR cannot express (ROADMAP U11): typed scalar parameters and results, declared
/// [`Effects`], a body per target, and optionally a `reference` kernel that says what it computes.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ExternDecl {
    /// For diagnostics.
    pub name: String,
    /// The scalar parameter types.
    #[serde(with = "crate::text::ek_vec")]
    pub params: Vec<ElementKind>,
    /// The result types (none, one, or a tuple).
    #[serde(with = "crate::text::ek_vec")]
    pub results: Vec<ElementKind>,
    /// What it may do.
    pub effects: Effects,
    /// One body per target, at most one each.
    pub bodies: Vec<TargetBody>,
    /// Index into [`Module::fns`] of a plain kernel with this signature, run by the oracle and cpu-c in place of
    /// the bodies. Only the device leg shows that a body means what its reference means.
    pub reference: Option<u32>,
}

impl ExternDecl {
    /// A declaration with no body and no reference.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        params: Vec<ElementKind>,
        results: Vec<ElementKind>,
        effects: Effects,
    ) -> Self {
        Self {
            name: name.into(),
            params,
            results,
            effects,
            bodies: Vec::new(),
            reference: None,
        }
    }

    /// This declaration with a body for one more target.
    #[must_use]
    pub fn with_body(mut self, body: TargetBody) -> Self {
        self.bodies.push(body);
        self
    }

    /// This declaration with the reference kernel `fn_index`.
    #[must_use]
    pub fn with_reference(mut self, fn_index: u32) -> Self {
        self.reference = Some(fn_index);
        self
    }
}

/// Which barrier a [`Stmt::Barrier`] is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum BarrierKind {
    /// Every thread of the scope arrives, then every thread leaves.
    Full,
    /// `bar.sync id, count`: a barrier of a `Block` that `count` threads (any subset of the block, whole warps) arrive at
    /// and wait for; `id` is 0 to 15 and names the barrier, so subsets use different ids. Threads that disagree on
    /// `count` are `KernelTrap::BarrierMismatch`. Unlike [`BarrierKind::Full`] it may sit under thread-dependent control:
    /// the participants are whoever reach it. Every arrival is release-ordered before every later wait on the barrier.
    Named {
        /// Which barrier, 0 to 15.
        id: u32,
        /// A count of arriving threads; threads that disagree on it trap (`KernelTrap::BarrierMismatch`).
        count: u32,
    },
    /// `bar.arrive id, count`: arrive at the named barrier without waiting (a producer signalling a consumer that waits
    /// with [`BarrierKind::Named`]). It counts toward the same `count`.
    NamedArrive {
        /// Which barrier, 0 to 15.
        id: u32,
        /// A count of arriving threads; threads that disagree on it trap (`KernelTrap::BarrierMismatch`).
        count: u32,
    },
}

/// Which [`Stmt::Collective`] (ROADMAP U5 P5). **Shapes only** until the semantics lands.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum CollectiveKind {
    /// Each lane receives the value of another lane of the scope, named by `operand` under `mode`
    /// (`Idx`: the lane index, `Up`/`Down`: the distance, `Xor`: the lane mask).
    ///
    /// **Edge rules (the IR defines them; an emitter must SPELL them, never inherit a device's, and is device-tested at the edge
    /// lanes or declines by name; `docs/design-kernel-collectives-edges.md`).** For `n` lanes numbered `0..n`: `Idx` reads lane
    /// `operand mod n` (a Euclidean remainder, so an `operand` of `n` reads lane 0 and `-1` reads lane `n - 1`; **design**, the
    /// portable defined rule: PTX `shfl.sync.idx` takes `b[4:0]`, which is the same only when `n` is 32). `Up` reads lane
    /// `i - operand`, `Down` lane `i + operand` and `Xor` lane `i ^ operand`; **when that lane is outside `0..n` the lane reads its
    /// OWN value**, including `operand == 0` (every lane reads itself) and `operand >= n` for `Up`/`Down` (every lane reads
    /// itself). This is **inherited from CUDA, with its source**: PTX ISA 9.5, section 9.7.10.6 `shfl.sync` ("If the computed
    /// source lane index j is in range, the thread will copy the input operand a from lane j into its own destination register d;
    /// otherwise, the thread will simply copy its own input a to destination d", and `if (!pval) j = lane`), taken for a clamp
    /// that equals the scope's size. The targets leave these cases undefined or indeterminate (WGSL out-of-range sources
    /// indeterminate, HLSL has no up/down/xor intrinsic: vulkane's `docs/design/506-subgroup-collectives.md`).
    Shuffle {
        /// How `operand` names the source lane.
        mode: ShuffleMode,
        /// The lane index, distance or mask.
        operand: ScalarExpr,
    },
    /// A predicate over the scope's contributed values. `Ballot` is a `U32` whose bit `i` is lane `i`'s predicate, so it is defined for
    /// up to **32 lanes** and a scope of more is a typed oracle trap (**design**: the result is a `U32`; PTX ISA 9.5
    /// `vote.sync.ballot.b32` is also a `.b32` mask of the warp's 32 lanes).
    Vote(VoteKind),
    /// Each lane receives a `U32` mask whose bit `i` is set when lane `i`'s value is **bit-for-bit equal** to this lane's (a
    /// float compares by its bit pattern, so `NaN` matches an identical `NaN` and `0.0` does not match `-0.0`). A
    /// **Subgroup** operation of up to 32 lanes, as `Ballot` (a Block is refused, a larger scope is a typed oracle trap).
    /// Source: PTX ISA, `match.any.sync.type` ("the mask of threads whose operand `a` equals this thread's"); CUDA
    /// `__match_any_sync`.
    MatchAny,
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
    /// Every lane receives the fold of the scope's values under a **user combine function**: the function `combine`
    /// of the [`Module`] (`fns[combine]`), a plain function `(a: T, b: T) -> T` over the collective's value type, with
    /// no barrier (ROADMAP R5.7 D4; PM ruling 2026-11-09: the custom combine stays at the kernel layer, as a callee).
    /// The oracle folds left to right in lane order. The author asserts the function is **associative**, which lets
    /// a device reassociate it (a shuffle tree); a float combine is therefore compared within a tolerance, not bytes.
    /// It needs a module: `check_kernel` alone refuses it, `check_module` resolves the function.
    ReduceWith {
        /// Index of the combine function in [`Module::fns`].
        combine: u32,
    },
    /// Every lane receives the prefix fold up to itself (inclusive) under the combine function `combine`; see
    /// [`Self::ReduceWith`]. There is no exclusive form: a user combine has no identity to hand lane 0.
    ScanWith {
        /// Index of the combine function in [`Module::fns`].
        combine: u32,
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
    /// Every lane's predicate (is the value non-zero) is the same: all true or all false. Source: PTX ISA, `vote.sync.uni.pred`
    /// ("true if source predicate has the same value in all active threads"); the result is `1` or `0` (a `U32`) like `All`/`Any`.
    Uniform,
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
    /// Wrapping increment: `old >= value ? 0 : old + 1` (`U32` only). Not commutative, so even the final value depends on the
    /// order of the threads. Source: CUDA C++ Programming Guide, `atomicInc`; PTX ISA `atom.inc.u32`.
    Inc,
    /// Wrapping decrement: `(old == 0 || old > value) ? value : old - 1` (`U32` only). Not commutative. Source: CUDA C++
    /// Programming Guide, `atomicDec`; PTX ISA `atom.dec.u32`.
    Dec,
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
    /// The locals that carry the values after [`Self::result`] when a [`Stmt::CallTuple`] calls this function: a
    /// function that returns `(a, b)` has `result = a` and `results = [b]`. Empty for every function that returns at
    /// most one value (R5.7 D4b).
    #[serde(default)]
    pub results: Vec<u32>,
    /// Scalar parameters whose value is fixed when the kernel is built (a specialization constant), not passed at launch.
    /// Empty for every kernel that has none; an empty list is not serialized, so a kernel without one keeps its text and its
    /// revision hash.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub spec_consts: Vec<SpecConst>,
}

/// The value a [`SpecConst`] takes when nothing overrides it.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum SpecValue {
    /// For an integer scalar parameter.
    Int(i128),
    /// For a floating-point scalar parameter.
    Float(f64),
}

/// A scalar parameter that is a specialization constant: the program is built for one value of it, so it is not among the
/// launch arguments. The oracle binds the default; an emitter either bakes the default into the program or declines by name.
/// Vocabulary only: this adds no statement and no expression, the parameter is read as its local like any other scalar.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct SpecConst {
    /// Index into [`KernelFn::params`] of a [`ParamKind::Scalar`] parameter.
    pub param: u32,
    /// Its value.
    pub default: SpecValue,
}

impl SpecConst {
    /// A specialization constant for the scalar parameter `param`.
    #[must_use]
    pub fn new(param: u32, default: SpecValue) -> Self {
        Self { param, default }
    }
}

impl KernelFn {
    /// Where the integer `local` comes from as a launch argument: `(parameter index, field index, type)`. A scalar
    /// parameter has no field index; a struct parameter's field has `Some`. `None` when no parameter binds `local`.
    #[must_use]
    pub(crate) fn argument_of_local(
        &self,
        local: u32,
    ) -> Option<(usize, Option<usize>, ElementKind)> {
        self.params
            .iter()
            .enumerate()
            .find_map(|(i, p)| match &p.kind {
                ParamKind::Scalar { ty, local: l } if *l == local => Some((i, None, *ty)),
                ParamKind::Struct { fields } => fields
                    .iter()
                    .position(|f| f.local == local)
                    .map(|j| (i, Some(j), fields[j].ty)),
                _ => None,
            })
    }

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
            results: Vec::new(),
            spec_consts: Vec::new(),
        }
    }

    /// The same kernel with scalar parameter `param` a specialization constant of value `default` (see [`SpecConst`]).
    #[must_use]
    pub fn with_spec_const(mut self, param: u32, default: SpecValue) -> Self {
        self.spec_consts.push(SpecConst::new(param, default));
        self
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

    /// The same function with `locals` as the values after its result (what a [`Stmt::CallTuple`] of it also
    /// receives): `with_result(a).with_results([b, c])` returns `(a, b, c)`.
    #[must_use]
    pub fn with_results(mut self, locals: impl Into<Vec<u32>>) -> Self {
        self.results = locals.into();
        self
    }

    /// Every value this function returns, in order: [`Self::result`] then [`Self::results`].
    pub fn all_results(&self) -> impl Iterator<Item = u32> + '_ {
        self.result.into_iter().chain(self.results.iter().copied())
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
    /// A `Stmt::Mma` the checker refuses: what is wrong with the atom, the launch or a fragment.
    MmaUnsupported {
        /// What is wrong.
        what: &'static str,
    },
    /// An `mbarrier` statement the checker refuses: a barrier place that is not a `Shared` 64-bit element, or an operand
    /// that is not an integer.
    MbarrierUnsupported {
        /// What is wrong.
        what: &'static str,
    },
    /// A `Stmt::For` unroll hint the checker refuses (a factor of 0).
    UnrollUnsupported {
        /// What is wrong.
        what: &'static str,
    },
    /// A `Stmt::CopyBulk` the checker refuses: a destination that is not `Shared`, a source that is not a readable global
    /// buffer, mismatched element types, a size that is not a multiple of 16 bytes, or a barrier that is not a `Shared`
    /// 64-bit element.
    BulkCopyUnsupported {
        /// What is wrong.
        what: &'static str,
    },
    /// A `Stmt::Wgmma` the checker refuses: a launch that is not whole warpgroups of 32-lane subgroups, an `n` or element
    /// type outside the modelled set, a tile that is not a `Shared` place, or a fragment of the wrong space or length.
    WgmmaUnsupported {
        /// What is wrong.
        what: &'static str,
    },
    /// A named barrier the checker refuses: a scope other than `Block`, an id above 15, or a thread count that is not a
    /// positive whole number of 32-lane warps within the block.
    NamedBarrierUnsupported {
        /// What is wrong.
        what: &'static str,
    },
    /// A `Stmt::Tcgen05Mma`, `Tcgen05Commit` or `Tcgen05Ld` the checker refuses, or an ordinary access to a `Tensor`
    /// allocation: a launch whose subgroup is not 32 lanes, an `n` or element type outside the modelled set, a tile that
    /// is not a `Shared` place, a tensor allocation that is not a multiple of 128 32-bit elements, or a fragment of the
    /// wrong space or length.
    Tcgen05Unsupported {
        /// What is wrong.
        what: &'static str,
    },
    /// A `Stmt::MatrixLoad` the checker refuses: what is wrong with the launch, the source or the destination.
    MatrixLoadUnsupported {
        /// What is wrong.
        what: &'static str,
    },
    /// A `Stmt::MatrixStore` the checker refuses: what is wrong with the launch, the fragment or the destination.
    MatrixStoreUnsupported {
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
    /// A `Stmt::ExternCall` names a declaration that is not in the module.
    BadExtern {
        /// The index.
        decl: u32,
    },
    /// An [`ExternDecl`] or a `Stmt::ExternCall` the checker refuses: a declaration at odds with its own effects, a
    /// body of the wrong shape, a reference of the wrong signature, a call with the wrong arity, argument or result
    /// type, or an `ExternCall` outside a module.
    ExternUnsupported {
        /// What is wrong.
        what: &'static str,
    },
    /// A [`Stmt::CopyAsync`] the checker refuses: a destination that is not a `Shared` allocation, a source that is
    /// not a global buffer, or a size other than 4, 8 or 16 bytes.
    AsyncUnsupported {
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
    /// A kernel declares (as its compute type, a parameter, a local or an allocation) a dtype that is recognised but not
    /// computed yet: `F128`, `I128`, `U128`, `I256`, `U256`, `F8E3M4`, `F8UE4M3` (union batch 1, KISS 0.24.0; plan E1a). It is
    /// declined by name, never run: the oracle holds no value of these types until the casts, the wide integer and the software
    /// binary128 land (plan E1b, E2, E3).
    DtypeNotComputed {
        /// The dtype.
        dtype: ElementKind,
    },
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
    /// An allocation's `layout` or `align_bytes` breaks a rule of `docs/design-swizzled-allocation.md` (the reason is named): a layout
    /// only on `Shared` with a `Static` extent equal to its size, an injective layout inside the extent, a swizzle whose period
    /// divides the byte extent and whose base is at least the element width, and a swizzled allocation aligned to the period.
    AllocLayout {
        /// The allocation index.
        alloc: u32,
        /// The rule, in words.
        why: &'static str,
    },
    /// A tile allocation or tile statement breaks a rule (the reason is named): a `Tile` space and a tile shape agree, a tile
    /// statement names a tile of the right shape and element types, and no ordinary access reaches a tile.
    Tile {
        /// The allocation index (or the first tile the statement names).
        alloc: u32,
        /// The rule, in words.
        why: &'static str,
    },
    /// A specialization constant that names no scalar parameter, names one twice, or whose default is not of the parameter's
    /// kind (an integer default for a floating parameter, a float for an integer, or an integer outside the type's range).
    BadSpecConst {
        /// The index into [`KernelFn::spec_consts`].
        index: u32,
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
    /// A [`Place::ClusterPeer`] in a kernel whose launch names no cluster dimensions, or on an allocation that is not
    /// `Shared`.
    ClusterPeerUnsupported {
        /// What is wrong.
        what: &'static str,
    },
    /// A [`Place::Peer`] in a launch that spans one device.
    PeerUnsupported {
        /// What is wrong.
        what: &'static str,
    },
    /// A [`Place::Multicast`] that a `Load`, `AtomicCas` or `AtomicRmw` form, element or scope does not admit.
    MulticastUnsupported {
        /// What is wrong.
        what: &'static str,
    },
    /// A `Barrier` whose scope is not `Subgroup` or `Block`.
    BadBarrierScope,
    /// A `Barrier` under control that may differ between the threads of its scope. The rule is
    /// conservative (decision D-R4-5): it rejects some uniform programs.
    BarrierDivergent,
    /// A `Subgroup`-scope [`Stmt::Collective`] or [`Stmt::CollectiveTuple`] reached after a `Return` that some lanes of
    /// its subgroup may have taken (the `Return` sat under control that may differ between the lanes of one subgroup).
    /// The oracle runs a collective over every lane of the subgroup, while a device operates on the lanes still active,
    /// so a kernel that exits a subset of the lanes first has no single meaning at the collective (ROADMAP U63; board 506).
    /// A `Return` under warp-uniform control, which removes whole subgroups, is not refused.
    CollectiveAfterReturn,
}
