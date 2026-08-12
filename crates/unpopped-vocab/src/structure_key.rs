//! Structure-class keying for AOT kernel specialization.
//!
//! A [`StructureKey`] is the canonical identity of an *input/output layout
//! class* — the join token shared across three consumers (per the Baracuda↔Fuel
//! boundary contract in `baracuda:docs/design/kernel-specialization.md`):
//!
//! 1. **runtime dispatch** — pick the specialized kernel registered for a key;
//! 2. **FKC predicate generation** — a generated kernel contract's admissibility
//!    predicate *is* its structure key, so the planner's miss signal is honest;
//! 3. **telemetry tagging** — Fuel tags each dispatch/miss record with the
//!    key's [`StructureKey::to_token`] string.
//!
//! The key is computed by [`structure_key`] from a slice of [`OperandDesc`] —
//! the **minimal operand-description projection** the key reads. Each
//! `OperandDesc` is populated by the consuming runtime from its own device
//! tensor / buffer view; that view-to-`OperandDesc` adapter lives in the runtime
//! crate, not in this driver-free vocabulary. No consumer reimplements the key —
//! they all call this one function, so the build matrix and the runtime lookup
//! speak the same language by construction.
//!
//! # Scope (v1)
//!
//! This first cut targets the elementwise specialization pilot. It derives the
//! per-operand and whole-key predicate axes (contiguity, broadcast, flip,
//! vector width, inner-extent divisibility, index width, effective rank, work
//! class) from raw shape/stride/alignment. The following are deliberately left
//! for follow-ups and are called out at their use sites:
//!
//! - **Reduction keying** ([`StructureKey::reduce_axes`] is always empty here).
//! - **Quant-aware keying** ([`OperandDesc::quant`] is carried so Fuel can bind
//!   the interface, but v1 does not fold it into the key — quant operands are
//!   out of scope until the quant pilot).
//! - **Full canonicalization** (size-1 squeeze is applied; adjacent-contiguous
//!   merge feeds the effective rank; legality-aware axis *reordering*
//!   to maximize cell-merging is a follow-up).

use crate::{ArchSku, ElementKind, OpCategory};

/// Maximum tensor rank the structure key supports — matches the rank ceiling
/// of every strided baracuda kernel (`baracuda::coord` `MAX_RANK`).
pub const MAX_RANK: usize = 8;

/// Maximum number of operands (inputs + output) a single key describes.
pub const MAX_OPERANDS: usize = 8;

/// Structure-key schema version. Bumped when a predicate axis is added or
/// altered; old-version tokens stay distinguishable by this field.
///
/// v2 (D8, 2026-07-19): aligned the token codec to KISS-Classify — the
/// namespaced `target_capability` (`cuda:sm89`, §6.8) replaces the bare `sm89`
/// arch token, and the index-width field is spelled `ix32`/`ix64` (§6.7-0003,
/// deliberately distinct from the `i32`/`i64` dtype tokens).
///
/// v3 (sk3 RFC D1+D4+D5, 2026-07-22): the gem precision coordinates, landed as
/// ONE bump per the "grow the key once" pin. The gem contraction field gains a
/// REQUIRED trailing `/<wdt>/<acc>/<out>/<mp>` group (weight dtype, accumulator
/// dtype, output dtype, math-precision `st`/`rm`) after the optional
/// `/b<class>`; the spec-forbidden `f32s` dtype token retires into `<mp>` (D4 —
/// strict-SIMT f32 = `f32`-primary + `st`, TF32 = `f32`-primary + `rm`); and
/// the FP8 spellings go variant-explicit (`e4m3` → `e4m3fn`; `e5m2` is already
/// variant-explicit and unchanged; the AMD `e4m3fnuz`/`e5m2fnuz` spellings are
/// reserved, unused). Non-gem cells change only the version prefix.
pub const STRUCTURE_KEY_VERSION: u16 = 4;

// ===========================================================================
// Predicate axes
// ===========================================================================

/// Width of the integer arithmetic used for element offsets.
///
/// `int32` offset math is materially cheaper (fewer registers, tighter loops);
/// the boundary is 2³¹ elements and is architecture-independent.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum IdxWidth {
    /// All element offsets fit in `i32` (`< 2³¹`).
    Idx32,
    /// At least one offset needs `i64`.
    Idx64,
}

/// Per-operand memory-layout class — the single most codegen-relevant axis.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Hash)]
pub enum Contiguity {
    /// Row-major packed: linear addressing, no coordinate unravel.
    #[default]
    Contig,
    /// Innermost (fastest-varying) axis has stride 1, outer axes strided:
    /// the inner loop vectorizes even though the outer walk is strided.
    InnerContig,
    /// Arbitrary strides — full coordinate unravel per element.
    Strided,
    /// At least one axis has stride 0 — the load can be hoisted out of the loop.
    Broadcast,
}

/// Achievable vectorized access width, derived from base-pointer alignment,
/// innermost stride/extent, and dtype size. `ld.128`/`st.128` (V4 for f32, V8
/// for f16) versus scalar is 2–4× on bandwidth-bound ops.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Hash)]
pub enum VecWidth {
    /// One element per access.
    #[default]
    Scalar,
    /// Two elements per access.
    V2,
    /// Four elements per access.
    V4,
    /// Eight elements per access.
    V8,
}

/// Divisibility bucket of an operand's innermost extent — drives remainder-loop
/// elimination and full unrolling. The ladder is `Div16 ⊐ Div8 ⊐ Div4 ⊐ Div2 ⊐ Any`.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Hash)]
pub enum DivBucket {
    /// Inner extent divisible by 16.
    Div16,
    /// Divisible by 8 (but not 16).
    Div8,
    /// Divisible by 4 (but not 8).
    Div4,
    /// Divisible by 2 (but not 4).
    Div2,
    /// No useful power-of-two divisor.
    #[default]
    Any,
}

/// Total-work size class — replaces a "stepped max-dim" axis. Tiny work wants a
/// single-warp or single-block kernel (no grid-stride, no millions of idle
/// threads); everything larger is one grid-stride kernel.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum WorkClass {
    /// Fits in one warp (≤ 32 elements).
    OneWarp,
    /// Fits in one block (≤ 1024 elements).
    OneBlock,
    /// Larger — a grid-stride loop.
    GridStride,
}

/// A bitmask over canonical axes (bit `i` ⇒ axis `i`). Used for the broadcast
/// axis set and the reduction axis set; rank is capped at [`MAX_RANK`] so a
/// single byte suffices.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Hash)]
pub struct AxisMask(pub u8);

impl AxisMask {
    /// The empty mask (no axes set).
    pub const EMPTY: AxisMask = AxisMask(0);

    /// `true` if `axis` is set.
    #[inline]
    #[must_use]
    pub const fn is_set(self, axis: u8) -> bool {
        axis < 8 && (self.0 >> axis) & 1 == 1
    }

    /// Set `axis` (no-op if `axis >= 8`).
    #[inline]
    pub fn set(&mut self, axis: u8) {
        if axis < 8 {
            self.0 |= 1 << axis;
        }
    }

    /// `true` if no axes are set.
    #[inline]
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }
}

/// Structural size class of one contraction dimension — **classes, never
/// literal extents** (the §1 non-negotiable). Thresholds are v1 and tunable;
/// they exist to split the vendor-owned regime (all-`Large` → cuBLAS/CUTLASS)
/// from the generated long tail (`Tiny` M/N = the FlashDecoding++ flat-GEMM /
/// decode GEMV-adjacent cell).
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum SizeClass {
    /// ≤ 8 — the skinny/decode regime.
    Tiny,
    /// 9..=128.
    Small,
    /// 129..=2048.
    Mid,
    /// > 2048 — the vendor-tuned regime when all three dims are here.
    Large,
}

impl SizeClass {
    /// Classify an extent.
    #[must_use]
    pub fn of(extent: i64) -> SizeClass {
        match extent {
            i64::MIN..=8 => SizeClass::Tiny,
            9..=128 => SizeClass::Small,
            129..=2048 => SizeClass::Mid,
            _ => SizeClass::Large,
        }
    }
}

/// Math-precision coordinate of the sk3 gem cell (RFC D4 / KISS-Ops §6.17):
/// the input-rounding axis that replaced the spec-forbidden `f32s` dtype token.
///
/// Deliberately a 2-value key code, not the full `MathPrecision` enum — the key
/// only needs the mantissa-reduction axis; the §6.17 semantics layer resolves a
/// code to a concrete mode per `(primary_dtype, target_capability)` (`rm` on
/// `f32`-primary `cuda:sm80+` = TF32: 10 mantissa bits, RNE, exponent carry).
/// Codes never begin with `b` (that prefix is the batch coordinate's, §4.1.2 of
/// the RFC), so the geometry and precision groups never collide in spelling.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum MpCode {
    /// `st` — bit-stable / strict: no input rounding; each operand enters the
    /// multiply at its full storage-dtype mantissa (the retired `F32Strict`
    /// semantics on an `f32`-primary cell).
    St,
    /// `rm` — reduced-mantissa: operand mantissas are rounded to a pinned
    /// narrower width before compute (per §6.17's `(bits, rounding_mode)` pin).
    Rm,
}

/// Storage-order class of a contraction operand: `perm[d]` is the storage axis
/// read at role/logical position `d` (identity = canonical row-major). Copy,
/// heap-free; the identity default serializes byte-identically (additive codec).
///
/// `Eq`/`Hash` are HAND-WRITTEN, not derived, and are keyed on `(rank,
/// perm())` only — the don't-care tail of `perm` past `rank` is excluded.
/// `identity(rank)` fills the *entire* `[u8; MAX_RANK]` array with
/// `[0,1,…,MAX_RANK-1]`, while `from_perm(p)` fills only `perm[..p.len()]`
/// and leaves the tail zeroed; the two constructors can therefore produce
/// byte-different arrays for the SAME logical permutation (e.g. `identity(2)`
/// = `[0,1,2,3,4,5,6,7]` vs `from_perm(&[0,1])` = `[0,1,0,0,0,0,0,0]`), and a
/// derived `Eq`/`Hash` (which compares/hashes the full array) would wrongly
/// treat them as distinct. `ContractionKey` embeds this type and derives
/// `Eq`/`Hash` itself for dispatch-table keying, so padding-invariance here
/// is load-bearing for the whole key.
#[derive(Copy, Clone, Debug)]
pub struct LayoutOrder {
    perm: [u8; MAX_RANK],
    rank: u8,
}
impl Default for LayoutOrder {
    fn default() -> Self {
        Self::identity(0)
    }
}
impl PartialEq for LayoutOrder {
    fn eq(&self, other: &Self) -> bool {
        self.rank == other.rank && self.perm() == other.perm()
    }
}
impl Eq for LayoutOrder {}
impl std::hash::Hash for LayoutOrder {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.rank.hash(state);
        self.perm().hash(state);
    }
}
impl LayoutOrder {
    /// The identity (canonical row-major) order for a `rank`-dimensional operand.
    #[must_use]
    pub fn identity(rank: u8) -> Self {
        let mut perm = [0u8; MAX_RANK];
        for (i, p) in perm.iter_mut().enumerate() {
            *p = i as u8;
        }
        Self { perm, rank }
    }
    /// Build a `LayoutOrder` from an explicit permutation.
    ///
    /// # Panics (debug only)
    /// Debug-asserts `p.len() <= MAX_RANK`; a release build silently panics
    /// later on the slice-index copy instead (unchanged from before — this is
    /// a clearer diagnostic for a miscall, not a new invariant).
    #[must_use]
    pub fn from_perm(p: &[u8]) -> Self {
        debug_assert!(
            p.len() <= MAX_RANK,
            "LayoutOrder::from_perm: permutation length {} exceeds MAX_RANK {MAX_RANK}",
            p.len()
        );
        let mut perm = [0u8; MAX_RANK];
        perm[..p.len()].copy_from_slice(p);
        Self {
            perm,
            rank: p.len() as u8,
        }
    }
    /// `true` if this order is the canonical row-major identity.
    #[must_use]
    pub fn is_identity(&self) -> bool {
        (0..self.rank as usize).all(|i| self.perm[i] as usize == i)
    }
    /// The permutation (`perm[d]` = storage axis at logical position `d`).
    #[must_use]
    pub fn perm(&self) -> &[u8] {
        &self.perm[..self.rank as usize]
    }
}

/// Contraction-only structure facts (design §5.4 / the item-10 spike), carried
/// as `StructureKey::contraction` — `None` for every non-contraction cell so
/// non-GEMM tokens serialize **byte-identically** to the pre-contraction codec
/// (the token gains an optional trailing field only when these facts exist).
///
/// v1 scope = the canonical rank-2 row-major dense cell (`lhs [M,K] · rhs
/// [K,N] → out [M,N]`): the M/N/K size classes drive the vendor gate, the
/// K-alignment class the (future) MMA fragment / tail handling. Layout classes
/// join when the node grows past the pilot.
///
/// sk3 (D1+D4+D5) adds the precision/compute coordinate set — `wdt`/`acc`/
/// `out`/`mp` — the RFC's fix for the §6.6-0018 collision (a mixed-input FP8
/// GEMM and a homogeneous one, or SIMT-f32 and TF32-f32, previously derived
/// byte-identical tokens).
///
/// The lhs/rhs `LayoutOrder` fields (item 1 of the layout/shape ramp) are
/// additive like `batch`: an identity order serializes byte-identically to
/// the pre-order codec, so only a transposed/permuted operand adds a token
/// component. `derive_contraction` derives real (possibly non-identity)
/// values via [`classify_mat_layout`] — a packed transpose/permutation of
/// lhs/rhs is accepted (sub-spec A); a genuinely non-packed operand still
/// declines the whole cell to `None` (sub-spec D).
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct ContractionKey {
    /// Size class of M (lhs rows / out rows).
    pub m: SizeClass,
    /// Size class of N (rhs cols / out cols).
    pub n: SizeClass,
    /// Size class of K (the contracted dim).
    pub k: SizeClass,
    /// Divisibility of K (remainder/tail handling; future MMA-k legality).
    pub k_div: DivBucket,
    /// Size class of the leading batch dim for a rank-3 batched contraction
    /// (`[B,M,K]·[B,K,N] → [B,M,N]`); `None` for the plain rank-2 cell. Additive:
    /// a `None` batch serializes byte-identically to the pre-batch codec — only a
    /// batched cell carries the `/b<class>` token component (before the
    /// precision group, per the RFC: batch is an iteration-structure fact, so
    /// it sits with the geometry).
    pub batch: Option<SizeClass>,
    /// Storage-order class of the lhs operand (identity = canonical row-major).
    pub lhs_order: LayoutOrder,
    /// Storage-order class of the rhs operand (identity = canonical row-major).
    pub rhs_order: LayoutOrder,
    /// Operand-1 (weight) dtype — canonical (never `F32Strict`; the strict axis
    /// rides [`ContractionKey::mp`]).
    pub wdt: ElementKind,
    /// Accumulator / compute dtype (D5's key half). Derived by the canonical
    /// accumulator lattice ([`contraction_acc`]); the cell's contract MUST
    /// declare the same dtype as `accumulation_type` (KISS-Contract §6.8 pin).
    pub acc: ElementKind,
    /// Output operand dtype (Fuel #22) — canonical (never `F32Strict`).
    pub out: ElementKind,
    /// Math-precision coordinate (D4) — the `f32s` replacement.
    pub mp: MpCode,
}

/// The sk4 non-contraction precision coordinate: `<acc>/<mp>`
/// (KISS-CLASSIFY-6.7-0013).
///
/// The gem-symmetric twin of [`ContractionKey`]'s `acc`/`mp` pair, for cells
/// that are not contractions. It occupies the **same** optional-trailing token
/// slot the contraction group occupies for `gem`, and a cell carries **at most
/// one** precision field — the two never coexist. That is what makes the
/// nine-or-ten-field decode unambiguous: the tenth field is resolved by the
/// op-family code, never by counting fields. See [`carries_contraction_field`].
///
/// # Why this exists
///
/// Before sk4, a non-gem cell had nowhere to say "this reduction accumulates
/// wider than it computes". An `f16` trailing-axis reduction accumulating in
/// `f32` and one accumulating in `f16` derived byte-identical tokens, so a
/// dispatch table could not hold both and a cache could hand one cell's kernel
/// to the other. This field is the coordinate that separates them, and it
/// extends the strict-vs-reduced-mantissa axis (§6.7-0006) to reductions and
/// scans, where it previously collapsed.
///
/// It **declares** the accumulator/precision coordinate for identity. It does
/// not pin bit-level determinism — float accumulation may still be
/// order-invariant-nondeterministic (§6.17-0007).
///
/// # Construction
///
/// Prefer [`AccMp::new`], which returns `None` for a non-deviating pair. Rule
/// (d) makes the all-default spelling *invalid on the wire*, so a redundant
/// `AccMp` is a value that cannot be legally encoded; `new` refuses to build one
/// rather than deferring the problem to the encoder.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct AccMp {
    /// Accumulator dtype. A §6.1 member, never reserved, never `F32Strict`
    /// (the strict axis rides [`AccMp::mp`], exactly as it does on a gem cell).
    pub acc: ElementKind,
    /// Math-precision coordinate — the same code set as the gem group's `<mp>`.
    pub mp: MpCode,
}

impl AccMp {
    /// The field as it must be spelled for a cell computing in `compute`, or
    /// `None` when neither coordinate deviates and the field must be **omitted**.
    ///
    /// This is rules (a), (c) and (d) of KISS-CLASSIFY-6.7-0013 expressed as a
    /// constructor: emitted iff the accumulator differs from the compute dtype
    /// **or** the math-precision differs from the default `st`; omitted entirely
    /// otherwise (not `-`, not empty); and never emitted all-default, which is a
    /// forbidden redundant emission that a decoder MUST reject.
    ///
    /// `compute` is folded through [`canonical_dtype`] first, so an `F32Strict`
    /// compute dtype compares against the `F32` its token actually spells — an
    /// `f32` accumulator on an `F32Strict` cell is *not* a deviation, and
    /// claiming otherwise would emit a field that says nothing.
    #[must_use]
    pub fn new(compute: ElementKind, acc: ElementKind, mp: MpCode) -> Option<Self> {
        let acc = canonical_dtype(acc);
        if acc == canonical_dtype(compute) && mp == MpCode::St {
            return None;
        }
        Some(Self { acc, mp })
    }
}

/// Whether a cell of this op family carries the **contraction** group in the
/// optional-trailing token slot (`gem`), as opposed to the non-contraction
/// `(acc + mp)` field (everything else).
///
/// The single source of truth for the tenth-field dispatch, deliberately shared
/// by the encoder and the decoder. KISS-CLASSIFY-6.7-0013 resolves the
/// nine-or-ten-field ambiguity "by the op-family code" — **not** by field count,
/// and not by sniffing the payload's shape. A decoder that splits positionally
/// on a fixed count breaks silently the moment a cell widens from nine fields to
/// ten, and one that sniffs the payload would accept a contraction group on a
/// `red` cell instead of declining it.
const fn carries_contraction_field(op: OpCategory) -> bool {
    matches!(op, OpCategory::Gemm)
}

// ===========================================================================
// The key
// ===========================================================================

/// Per-operand predicate sub-key. One of these is carried for every input and
/// the output.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Hash)]
pub struct OperandKey {
    /// Memory-layout class.
    pub contig: Contiguity,
    /// Which axes broadcast (stride 0).
    pub bcast: AxisMask,
    /// Achievable vectorized access width.
    pub vec_width: VecWidth,
    /// Innermost-extent divisibility bucket.
    pub inner_div: DivBucket,
    /// `true` if any axis has a negative stride (a flipped / reversed view).
    pub flipped: bool,
}

/// The canonical identity of an input/output layout class.
///
/// Construct via [`structure_key`]. Two layouts that canonicalize to the same
/// `StructureKey` are served by the same specialized kernel. Heap-free and
/// `Copy` so it can be hashed into a dispatch table or an autotuner cache
/// directly; [`StructureKey::to_token`] gives the stable string form used on
/// the telemetry wire.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub struct StructureKey {
    /// Schema version ([`STRUCTURE_KEY_VERSION`]).
    pub version: u16,
    /// The op taxonomy this key was computed for (drives canonicalization
    /// legality).
    pub op: OpCategory,
    /// Primary dtype (operand 0). Mixed-dtype ops fold per-operand dtype in a
    /// follow-up; v1 assumes a uniform operand dtype.
    pub dtype: ElementKind,
    /// Compute capability the specialized kernel targets.
    pub arch: ArchSku,
    /// Offset-arithmetic width.
    pub idx: IdxWidth,
    /// Total-work size class.
    pub work: WorkClass,
    /// Raw iteration rank — the maximum operand rank (for elementwise ops the
    /// operands are rank-aligned, broadcasting via stride 0, so this is the
    /// shared logical rank the strided schedules unravel over). Size-1 squeeze
    /// and contiguous-axis collapse are deferred optimizations.
    pub rank: u8,
    /// Number of valid entries in [`StructureKey::operands`].
    pub n_operands: u8,
    /// Per-operand sub-keys; only `operands[0..n_operands]` are meaningful, the
    /// tail is [`OperandKey::default`] so equal keys hash equal.
    pub operands: [OperandKey; MAX_OPERANDS],
    /// Reduced-axis set for reduction-class ops; [`AxisMask::EMPTY`] otherwise
    /// (always empty in v1).
    pub reduce_axes: AxisMask,
    /// Contraction structure facts ([`ContractionKey`]); `None` for every
    /// non-contraction cell — in which case the token is byte-identical to the
    /// pre-contraction codec.
    pub contraction: Option<ContractionKey>,
    /// The sk4 non-contraction precision coordinate ([`AccMp`]);
    /// `None` on a `gem` cell (which carries [`StructureKey::contraction`]
    /// instead) and on any cell whose accumulator and math-precision both sit at
    /// their defaults. The two precision fields never coexist — see
    /// [`carries_contraction_field`].
    ///
    /// A `None` here serializes byte-identically to the pre-sk4 codec (modulo
    /// the §6.1 dtype renames), so the sk4 regen diff is exactly the cells whose
    /// accumulator or precision actually deviates (§6.7-0013 byte-stability).
    pub acc_mp: Option<AccMp>,
}

// ===========================================================================
// Operand description (the minimal projection the key reads)
// ===========================================================================

/// Quant family, mirroring the FDX `FDXQuant.family` codes (FDX is the
/// normative owner; this is the Baracuda-side projection).
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum QuantFamily {
    /// GGUF block layout, scale baked inline.
    Ggml,
    /// OCP microscaling (per-block F8E8M0 scale).
    Mx,
    /// Dynamic per-tensor/token/channel affine integer.
    AffineInt,
    /// Dynamic per-tensor/token/channel affine float.
    AffineFloat,
    /// NF4/QLoRA — low-bit data plus a separate per-block absmax scale.
    AffineBlock,
}

/// Where a quant scale lives, mirroring FDX `FDXScalePlacement`.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum ScalePlacement {
    /// Scale baked inline with the data block.
    Inline,
    /// Scale in a separate buffer.
    SeparateBuffer,
    /// Scale broadcast per axis.
    BroadcastPerAxis,
}

/// Quantization facts for a quant operand. Carried so Fuel can bind the
/// interface; **[`structure_key`] does not key on these** — verified by
/// `tests/operand_facts_reach_the_key.rs`, not merely believed.
///
/// # Do not build on this shape — it is superseded, not merely un-keyed
///
/// The wording here used to read as though the *shape* were right and only the
/// keying were deferred to a later pilot. It is the other way round. sk4 §3.2
/// settles the model: the **element dtype** and the **block structure** are
/// separate axes, and a block's shared scale is a **sibling operand** — its own
/// entry in the operand list — not a field hanging off the operand it scales.
///
/// That distinction is what closes the key collision, and it needs no schema
/// event, because operands already reach the key. Under the sibling model an
/// unquantized `i4` operand and a Q4 operand differ in the operand list itself
/// (one carries a scale sibling, the other does not), so they cannot derive
/// byte-identical tokens. Expressed as a field, they can and do.
///
/// The residual the sibling model does **not** close is block granularity:
/// blk32 and blk128 both contribute one scale sibling of the same rank, so they
/// still collide. That part is a genuine sk5 item and is tracked as such — it is
/// not something this type can fix.
///
/// **Why this is still here.** Removing the field is a breaking change, and per
/// sk4 §6 the hardening/removal fixes ride the coordinated schema cut rather
/// than preceding it — a separate major beforehand would cost downstream two
/// breaking releases instead of one. So the field stays until that cut, and this
/// doc is the guard in the meantime: a consumer reached for `quant` as a
/// precedent for extending [`OperandDesc`] once already, on the assumption it
/// was load-bearing. It is not.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct QuantFacts {
    /// Quant family.
    pub family: QuantFamily,
    /// Sub-byte bit width (e.g. 4 for Q4), or 0 if not sub-byte.
    pub sub_byte_bits: u8,
    /// Block extent in logical elements, or 0 if not block-quantized.
    pub block_elems: u16,
    /// Scale placement.
    pub scale: ScalePlacement,
}

/// Kind of a symbolic (live-vs-capacity) extent, mirroring FDX `FDXExtent.kind`.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum SymKind {
    /// A single dynamic scalar bound.
    Scalar,
    /// A `[min, capacity]` range.
    Range,
    /// An affine form `c0 + Σ cᵢ·symᵢ` (e.g. `k_len = cached + new`).
    Affine,
}

/// A symbolic extent on one axis. The axis's *capacity* is its
/// [`OperandDesc::shape`] entry (which keys strides and index width); this flags
/// that the live length is dynamic, which is itself a specialization axis for
/// attention-class ops (static `k_len == capacity` vs dynamic).
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct SymExtent {
    /// Which axis is symbolic.
    pub axis: u8,
    /// The kind of symbolic bound.
    pub kind: SymKind,
}

/// The per-operand description [`structure_key`] is given.
///
/// NOT all of it reaches the key. `quant` and `symbolic` are carried here but
/// are **not encoded into the token** and are read by nothing in this workspace —
/// two operands differing only in quantization block size, or only in which axis
/// is symbolic, produce byte-identical keys. Under KISS-CLASSIFY-6.8-0002 (byte-
/// exact matching, subset/implication logic forbidden) that collision does not
/// degrade gracefully, so do not populate these fields expecting the key to
/// distinguish them. Pinned by `tests/operand_facts_reach_the_key.rs`; closing the
/// gap changes the token grammar and is a schema event, not a patch.
///
/// Owning and `Copy` (inline `[i64; MAX_RANK]` arrays, no lifetimes) so every
/// consumer constructs it by value from whatever tensor or buffer view its own
/// runtime uses. Only `shape[0..rank]` / `strides[0..rank]` are meaningful.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub struct OperandDesc {
    /// Tensor rank (`≤ MAX_RANK`).
    pub rank: u8,
    /// Per-axis extents (capacity for symbolic axes).
    pub shape: [i64; MAX_RANK],
    /// Per-axis signed element strides. `0` = broadcast, `< 0` = flipped.
    pub strides: [i64; MAX_RANK],
    /// Base/logical operand dtype.
    pub dtype: ElementKind,
    /// Base-pointer alignment in bytes (drives vector width).
    pub align_bytes: u32,
    /// Quantization facts, if this is a quant operand.
    pub quant: Option<QuantFacts>,
    /// Symbolic-extent facts, if any axis is live-vs-capacity.
    pub symbolic: Option<SymExtent>,
}

impl OperandDesc {
    /// Build a plain (non-quant, non-symbolic) operand description from `rank`
    /// extents, strides, dtype, and base-pointer alignment.
    ///
    /// # Panics
    /// Panics if `rank > MAX_RANK` or if `shape`/`strides` are shorter than
    /// `rank`. Callers with statically-valid input use this; a path that reads
    /// an operand shape from a peer uses [`OperandDesc::try_new`] instead.
    #[must_use]
    pub fn new(
        rank: usize,
        shape: &[i64],
        strides: &[i64],
        dtype: ElementKind,
        align_bytes: u32,
    ) -> Self {
        Self::try_new(rank, shape, strides, dtype, align_bytes)
            .expect("OperandDesc::new: rank exceeds MAX_RANK or shape/strides shorter than rank")
    }

    /// Fallible constructor for untrusted or uncertain input: returns `None`
    /// instead of panicking when `rank > MAX_RANK` or when `shape`/`strides` are
    /// shorter than `rank`. Use this on any path that reads an operand shape from
    /// a peer; [`OperandDesc::new`] is the infallible convenience for callers
    /// whose `rank`/slices are statically valid.
    #[must_use]
    pub fn try_new(
        rank: usize,
        shape: &[i64],
        strides: &[i64],
        dtype: ElementKind,
        align_bytes: u32,
    ) -> Option<Self> {
        if rank > MAX_RANK || shape.len() < rank || strides.len() < rank {
            return None;
        }
        let mut s = [0i64; MAX_RANK];
        let mut st = [0i64; MAX_RANK];
        s[..rank].copy_from_slice(&shape[..rank]);
        st[..rank].copy_from_slice(&strides[..rank]);
        Some(Self {
            rank: rank as u8,
            shape: s,
            strides: st,
            dtype,
            align_bytes,
            quant: None,
            symbolic: None,
        })
    }

    // NOTE: `from_tensor_ref` (build an OperandDesc from a borrowed device
    // `TensorRef`) moved to `baracuda-kernels-types::operand_desc_ext` as the
    // `OperandDescExt` trait — it needs the device-view `TensorRef`, which stays
    // in the driver-coupled crate. All `OperandDesc` fields are `pub`, so the
    // extension constructs it directly. `OperandDesc::new` (above) is the
    // driver-free constructor callers use here.
}

// ===========================================================================
// Derivation
// ===========================================================================

/// Compute the [`StructureKey`] for an op over its operands on a target arch.
///
/// `operands` is inputs followed by the output; the first operand is treated as
/// the primary (it sets [`StructureKey::dtype`], the work class, and the
/// effective rank). An empty slice yields a rank-0 scalar key.
///
/// This is the single canonical key function — Fuel calls it rather than
/// reimplementing the derivation, so telemetry and the build matrix join on the
/// same token.
#[must_use]
pub fn structure_key(op: OpCategory, operands: &[OperandDesc], arch: ArchSku) -> StructureKey {
    let n = operands.len().min(MAX_OPERANDS);
    let mut keys = [OperandKey::default(); MAX_OPERANDS];
    let mut max_off: i64 = 0;
    for (slot, od) in keys.iter_mut().zip(operands.iter()).take(n) {
        *slot = derive_operand_key(od);
        max_off = max_off.max(max_touched_offset(od));
    }

    let idx = if max_off >= (1i64 << 31) {
        IdxWidth::Idx64
    } else {
        IdxWidth::Idx32
    };

    // The primary DTYPE is operand-0's (the §6.6-0005 primary-dtype path — the
    // STRUCT keeps the raw dtype, `F32Strict` included, as the in-process
    // MathPrecision carrier; only the TOKEN codec folds it to `f32` per sk3 D4).
    // The WORK class is FRAME-MAX across ALL operands (§6.5-0010/§6.6-0013),
    // NOT operand-0 alone — orthogonal axes, computed from different reads.
    let dtype = operands.first().map_or(ElementKind::F32, |p| p.dtype);
    let work = frame_work_class(operands);
    // Raw iteration rank = the widest operand rank (output rank for elementwise).
    let rank = operands.iter().map(|o| o.rank).max().unwrap_or(0);

    StructureKey {
        version: STRUCTURE_KEY_VERSION,
        op,
        dtype,
        arch,
        idx,
        work,
        rank,
        n_operands: n as u8,
        operands: keys,
        reduce_axes: derive_reduce_axes(op, operands),
        contraction: derive_contraction(op, operands),
        acc_mp: derive_acc_mp(op, dtype),
    }
}

/// The non-contraction `(acc + mp)` coordinate for a derived cell
/// (KISS-CLASSIFY-6.7-0013).
///
/// §6.7-0013 does not impose a normative accumulator lattice on non-contraction
/// cells the way §6.7-0006 does for `gem`. The field **declares** what the
/// producer's cell actually does, so the derivation is this producer's policy —
/// and a wrong declaration is worse than none, because it separates keys that
/// should collide and collides keys that should separate.
///
/// **This generator's policy is: accumulate at the compute dtype, bit-stable.**
/// Not an assumption — every schedule it can currently lower is elementwise.
/// Both shipping emitters decline `Reduction`/`RowReduce`/`Scan`/`Contraction`
/// outright, so there is no cell in which an accumulator could differ from its
/// compute dtype, and no path that reduces a mantissa before computing.
///
/// It is written as a policy fed through [`AccMp::new`] rather than as a bare
/// `None` on purpose: the day a reduction emitter lands with an `f32`
/// accumulator over `f16` data, the change belongs *here*, and the field starts
/// being emitted without anyone having to remember that this coordinate exists.
/// A bare `None` would silently keep declaring "accumulates at compute dtype"
/// after that stopped being true — and that declaration would be a lie the
/// dispatch table acts on.
fn derive_acc_mp(op: OpCategory, dtype: ElementKind) -> Option<AccMp> {
    if carries_contraction_field(op) {
        // A gem cell's precision facts ride the contraction group; the two
        // fields never coexist, and they share one token slot.
        return None;
    }
    AccMp::new(dtype, dtype, MpCode::St)
}

/// The key's canonical dtype: `F32Strict` folds to `F32` (sk3 D4 — the `f32s`
/// token retired from the closed set per §6.1-0005; the strict-vs-TF32 axis is
/// the gem cell's `<mp>` coordinate, derived from the pre-fold dtype). The
/// `F32Strict` Rust type itself stays — it still drives kernel selection and
/// is the derivation INPUT that says "strict SIMT math" on the operand channel.
const fn canonical_dtype(dt: ElementKind) -> ElementKind {
    match dt {
        ElementKind::F32Strict => ElementKind::F32,
        other => other,
    }
}

/// The canonical accumulator lattice for a dense-contraction cell, keyed off
/// the primary (operand-0) dtype: `int8/int4/bin → i32`; `f64 → f64`;
/// everything else (fp8/f16/bf16/f32) `→ f32`. Mirrors
/// `GemmSku::precision_guarantee` (`baracuda-cutlass/src/types.rs`) — the two
/// MUST stay in lock-step, and a provider kernel serving the cell MUST
/// accumulate in this dtype (the KISS-Contract §6.8 `accumulation_type` ↔
/// `<acc>` consistency pin).
const fn contraction_acc(primary: ElementKind) -> ElementKind {
    match primary {
        ElementKind::F64 => ElementKind::F64,
        ElementKind::I8 | ElementKind::U8 | ElementKind::I4 | ElementKind::U4 | ElementKind::B1 => {
            ElementKind::I32
        }
        _ => ElementKind::F32,
    }
}

/// The gem cell's math-precision coordinate, keyed off the PRE-fold primary
/// dtype. Mirrors the GEMM routing policy `GemmSku::precision_guarantee` pins
/// (`baracuda-cutlass/src/types.rs`): a plain-`F32` GEMM routes through TF32
/// tensor cores (`MathPrecision::Tf32` — reduced mantissa, `rm`), `F32Strict`
/// through SIMT CUDA cores at full binary32 (`MathPrecision::F32` — `st`).
/// Every other element kind multiplies at its declared storage precision (no
/// hidden input rounding) — `st`.
const fn contraction_mp(primary: ElementKind) -> MpCode {
    match primary {
        ElementKind::F32 => MpCode::Rm,
        _ => MpCode::St,
    }
}

/// Classify a contraction operand's storage order from its strides. Returns the
/// permutation `perm` such that role/logical axis `d` reads storage axis `perm[d]`,
/// iff the operand is a PACKED permutation of a contiguous tensor over its
/// NON-broadcast axes (every non-broadcast axis' |stride| equals the product of
/// the non-broadcast extents storage-inner to it). `None` if genuinely non-packed
/// among its non-broadcast axes (arbitrary strides → sub-spec D).
///
/// Stride-0 (broadcast) axes are ADMITTED: they occupy no storage extent, so they
/// are excluded from both the packed check and the running extent product — a
/// GQA broadcast-KV operand (e.g. batch broadcast over `[B,K,N]`) classifies
/// successfully as long as its remaining (real) axes are packed. The storage
/// order this returns is over those non-broadcast axes; a broadcast axis still
/// gets a deterministic (but address-irrelevant) `perm` slot, since the emitter's
/// `operand_stride_binding` filters broadcast axes out before consuming `perm`.
///
/// Role-agnostic: storage order derives from strides alone, so this takes no
/// `AxisRole`/`ContractionAxes` argument (roles enter at the plan/emitter layer).
/// Rank-generic — subsumes both the rank-2 and rank-3 `dense` checks below: a
/// canonical row-major operand's axes already sort into ascending order, so
/// `perm` comes out as the identity permutation (byte-identity with the
/// pre-Task-2 codec for every canonical cell).
fn classify_mat_layout(od: &OperandDesc) -> Option<LayoutOrder> {
    let rank = od.rank as usize;
    // Storage order = axes sorted by descending |stride| (outermost first). A
    // stride-0 (broadcast) axis sorts smallest (innermost). Ties (equal |stride|
    // — extent-1 axes, or multiple stride-0 broadcast axes) break on the axis
    // index so equivalent layouts canonicalize to the SAME order deterministically
    // (a bare `sort_by_key` is already stable, so this only makes the tie-break
    // explicit + robust to a future switch to an unstable sort).
    let mut axes: Vec<usize> = (0..rank).collect();
    axes.sort_by(|&a, &b| {
        core::cmp::Reverse(od.strides[a].abs())
            .cmp(&core::cmp::Reverse(od.strides[b].abs()))
            .then(a.cmp(&b))
    });
    // Verify packed over the non-broadcast axes: walking storage-inner→outer,
    // |stride| must equal the running extent product. Broadcast axes occupy no
    // storage extent, so they are skipped — neither packedness-checked nor
    // multiplied into `acc`.
    let mut acc: i64 = 1;
    for &d in axes.iter().rev() {
        if od.strides[d] == 0 {
            continue; // broadcast axis: no storage extent — not part of the packed layout
        }
        if od.strides[d].abs() != acc {
            return None; // non-packed → decline
        }
        acc = acc.saturating_mul(od.shape[d]);
    }
    // perm[logical d] = storage position of axis d. Here logical == operand axis
    // index; storage position is `axes.iter().position(|&a| a == d)`.
    let mut perm = [0u8; MAX_RANK];
    for (d, p) in perm.iter_mut().enumerate().take(rank) {
        *p = axes.iter().position(|&a| a == d).unwrap() as u8;
    }
    Some(LayoutOrder {
        perm,
        rank: od.rank,
    })
}

/// [item 10] Derive contraction structure facts for the canonical rank-2
/// row-major dense GEMM cell: `operands = [lhs [M,K], rhs [K,N], out [M,N]]`.
/// `out` stays required-canonical (row-major) in v1 — output views are
/// sub-spec B's scope. `lhs`/`rhs` classify through [`classify_mat_layout`]:
/// a packed transpose/permutation now derives real `lhs_order`/`rhs_order`
/// facts instead of being rejected outright; a genuinely non-packed operand
/// still declines (`None`) to sub-spec D. Any other shape/arity yields `None`
/// — an honest "no contraction facts", never a guess.
fn derive_contraction(op: OpCategory, operands: &[OperandDesc]) -> Option<ContractionKey> {
    // 3 operands = plain `[lhs, rhs, out]`; 4 = fused bias `[lhs, rhs, bias, out]`
    // (the per-column `[N]` bias the epilogue reads). The bias rides the existing
    // per-operand OperandKey list — it does NOT change the ContractionKey facts
    // (m/n/k/k_div), so the token is codec-compatible (no version bump).
    if op != OpCategory::Gemm || (operands.len() != 3 && operands.len() != 4) {
        return None;
    }
    let has_bias = operands.len() == 4;
    let (lhs, rhs, out) = (&operands[0], &operands[1], &operands[operands.len() - 1]);
    // sk3 precision/compute coordinates (D1+D4+D5). `wdt`/`out` are the
    // canonical operand spellings; `acc`/`mp` derive from the PRE-fold primary
    // dtype (the `F32Strict`-vs-`F32` distinction is exactly the `<mp>` input).
    let (wdt, acc, out_dt, mp) = (
        canonical_dtype(rhs.dtype),
        contraction_acc(lhs.dtype),
        canonical_dtype(out.dtype),
        contraction_mp(lhs.dtype),
    );
    // rank-2 = plain `[M,K]·[K,N]`; rank-3 = batched `[B,M,K]·[B,K,N]`. All three
    // core operands (lhs/rhs/out) must share the rank.
    let rank = lhs.rank;
    if rhs.rank != rank || out.rank != rank {
        return None;
    }
    match rank {
        2 => {
            let (m, k) = (lhs.shape[0], lhs.shape[1]);
            let (k2, n) = (rhs.shape[0], rhs.shape[1]);
            let dense = |o: &OperandDesc| o.strides[0] == o.shape[1] && o.strides[1] == 1;
            if k != k2 || out.shape[0] != m || out.shape[1] != n || !dense(out) {
                return None;
            }
            let lhs_order = classify_mat_layout(lhs)?;
            let rhs_order = classify_mat_layout(rhs)?;
            // A fused bias must be the DENSE per-column `[N]` vector (unit stride,
            // broadcast over M). The emitter reads a hardcoded `in2[col]`, so a
            // strided or broadcast (stride-0) bias would silently mis-read / read
            // out of bounds — decline (honest miss) rather than emit a wrong load.
            if has_bias {
                let bias = &operands[2];
                if bias.rank != 1 || bias.shape[0] != n || bias.strides[0] != 1 {
                    return None;
                }
            }
            Some(ContractionKey {
                m: SizeClass::of(m),
                n: SizeClass::of(n),
                k: SizeClass::of(k),
                k_div: div_bucket(k),
                batch: None,
                lhs_order,
                rhs_order,
                wdt,
                acc,
                out: out_dt,
                mp,
            })
        }
        3 => {
            // Batched `[B,M,K]·[B,K,N] → [B,M,N]`. v1 does not combine batch with a
            // fused bias (a follow-up).
            if has_bias {
                return None;
            }
            let (b, m, k) = (lhs.shape[0], lhs.shape[1], lhs.shape[2]);
            let (b2, k2, n) = (rhs.shape[0], rhs.shape[1], rhs.shape[2]);
            // Dense row-major rank-3: strides `[shape1*shape2, shape2, 1]`.
            let dense3 = |o: &OperandDesc| {
                o.strides[0] == o.shape[1] * o.shape[2]
                    && o.strides[1] == o.shape[2]
                    && o.strides[2] == 1
            };
            if b != b2
                || k != k2
                || out.shape[0] != b
                || out.shape[1] != m
                || out.shape[2] != n
                || !dense3(out)
            {
                return None;
            }
            let lhs_order = classify_mat_layout(lhs)?;
            let rhs_order = classify_mat_layout(rhs)?;
            // The v1 emitter binds the batch axis as the OUTERMOST storage stride
            // (`blockIdx.z`), so `operand_stride_binding` never treats batch as a
            // stride FACTOR (`ext(Batch)` is `unreachable!`). `classify_mat_layout`
            // is role-agnostic and admits any packed permutation, including one that
            // sorts a real batch axis storage-inner — decline that honestly here
            // (sub-spec D) rather than let it reach the emitter's `ext(Batch)` panic.
            // A BROADCAST batch (stride 0) is exempt: the emitter filters bcast axes
            // before the factor loop, so it never reaches `ext(Batch)` either.
            if (lhs.strides[0] != 0 && lhs_order.perm()[0] != 0)
                || (rhs.strides[0] != 0 && rhs_order.perm()[0] != 0)
            {
                return None;
            }
            Some(ContractionKey {
                m: SizeClass::of(m),
                n: SizeClass::of(n),
                k: SizeClass::of(k),
                k_div: div_bucket(k),
                batch: Some(SizeClass::of(b)),
                lhs_order,
                rhs_order,
                wdt,
                acc,
                out: out_dt,
                mp,
            })
        }
        _ => None,
    }
}

/// [item 03] Derive the reduced-axis set for a reduction cell from **keepdim-form**
/// operands: bit `d` is set where the input varies (`shape[d] > 1`) but the output
/// is size-1 (`shape[d] == 1`). This is unambiguous *only* in keepdim form (same
/// rank, reduced axes present as size-1) — a collapsed (rank-reduced) output is
/// un-inferable (input `[2,2,4]` reducing axis 0 vs 1 both give `[2,4]`, byte-
/// identical operands), so it yields `EMPTY` = *undetermined*. Gated on
/// [`OpCategory::Reduction`]: non-reduction ops (and the fused RowReduce family,
/// whose output == input shape leaves no size-1 trace) stay `EMPTY`; those carry
/// the reduced axis explicitly at the seam (item 05). `EMPTY` is thus reserved for
/// "non-reduction / undetermined", never overloaded as a "last-axis" sentinel.
/// See `baracuda:docs/design/axis-role-vocabulary.md` — this is the `{Reduced}` projection.
fn derive_reduce_axes(op: OpCategory, operands: &[OperandDesc]) -> AxisMask {
    if op != OpCategory::Reduction || operands.len() < 2 {
        return AxisMask::EMPTY;
    }
    let input = &operands[0];
    let output = &operands[operands.len() - 1];
    // Keepdim-form precondition: same rank, reduced axes present as size-1.
    if input.rank != output.rank {
        return AxisMask::EMPTY; // collapsed / rank-reduced output ⇒ undetermined
    }
    let mut axes = AxisMask::EMPTY;
    for d in 0..input.rank as usize {
        if input.shape[d] > 1 && output.shape[d] == 1 {
            axes.set(d as u8);
        }
    }
    axes
}

/// Convenience: compute the [`StructureKey`] and return its wire token in one
/// call — the form a caller's trampoline uses to tag a dispatch/miss record or
/// match an FKC `accept` predicate. Equivalent to
/// `structure_key(op, operands, arch).to_token()`.
///
/// This is the **single canonical entry point** for the cross-boundary use: Fuel
/// builds each [`OperandDesc`] from its `FdxOperandDesc` (rank, shape, strides,
/// dtype, alignment — plus quant / symbolic facts when present) and calls this,
/// rather than re-deriving the key, so the build matrix and the runtime lookup
/// join on the same token by construction. (Two Rust projects integrate via this
/// callable directly; an FFI C-ABI trampoline would additionally need the FDX
/// numeric dtype codes — review item E5 — and is deferred to that.)
///
/// ```
/// use unpopped_vocab::{
///     structure_key_token, ArchSku, ElementKind, OpCategory, OperandDesc,
/// };
/// // a [128, 256] row-major f32 (in, in, out) triple for a binary elementwise add.
/// let a = OperandDesc::new(2, &[128, 256], &[256, 1], ElementKind::F32, 256);
/// let token = structure_key_token(OpCategory::BinaryElementwise, &[a, a, a], ArchSku::Sm89);
/// assert!(token.starts_with("sk4|bin|f32|cuda:sm89|"));
/// ```
#[must_use]
pub fn structure_key_token(op: OpCategory, operands: &[OperandDesc], arch: ArchSku) -> String {
    structure_key(op, operands, arch).to_token()
}

/// Innermost non-unit axis of an operand, or `None` if the operand is all
/// size-≤1 axes (a scalar).
fn inner_axis(od: &OperandDesc) -> Option<usize> {
    (0..od.rank as usize).rev().find(|&d| od.shape[d] > 1)
}

fn derive_operand_key(od: &OperandDesc) -> OperandKey {
    let rank = od.rank as usize;

    // Broadcast mask: extent-> 1 axes with stride 0.
    let mut bcast = AxisMask::EMPTY;
    let mut flipped = false;
    for d in 0..rank {
        if od.shape[d] > 1 && od.strides[d] == 0 {
            bcast.set(d as u8);
        }
        if od.strides[d] < 0 {
            flipped = true;
        }
    }

    let inner = inner_axis(od);
    let contig = classify_contiguity(od, bcast, inner);
    let vec_width = classify_vec_width(od, inner, bcast);
    let inner_div = match inner {
        Some(d) => div_bucket(od.shape[d]),
        None => DivBucket::Any,
    };

    OperandKey {
        contig,
        bcast,
        vec_width,
        inner_div,
        flipped,
    }
}

fn classify_contiguity(od: &OperandDesc, bcast: AxisMask, inner: Option<usize>) -> Contiguity {
    if !bcast.is_empty() {
        return Contiguity::Broadcast;
    }
    let Some(inner) = inner else {
        return Contiguity::Contig; // scalar
    };
    let rank = od.rank as usize;

    // Expected row-major contiguous |stride| per axis (over non-unit axes).
    let mut acc: i64 = 1;
    let mut all_match = true;
    for d in (0..rank).rev() {
        if od.shape[d] <= 1 {
            continue;
        }
        if od.strides[d].abs() != acc {
            all_match = false;
        }
        acc = acc.saturating_mul(od.shape[d]);
    }
    if all_match {
        Contiguity::Contig
    } else if od.strides[inner].abs() == 1 {
        Contiguity::InnerContig
    } else {
        Contiguity::Strided
    }
}

fn classify_vec_width(od: &OperandDesc, inner: Option<usize>, bcast: AxisMask) -> VecWidth {
    let (Some(inner), Some(dsz)) = (inner, dtype_size_bytes(od.dtype)) else {
        return VecWidth::Scalar;
    };
    // Only forward unit-stride contiguous inner runs vectorize in v1.
    if od.strides[inner] != 1 || !bcast.is_empty() {
        return VecWidth::Scalar;
    }
    let ext = od.shape[inner].max(0) as u64;
    let align = u64::from(od.align_bytes);
    let dsz = u64::from(dsz);
    for &v in &[8u64, 4, 2] {
        let vbytes = v * dsz;
        if vbytes <= 16 && align % vbytes == 0 && ext % v == 0 {
            return match v {
                8 => VecWidth::V8,
                4 => VecWidth::V4,
                _ => VecWidth::V2,
            };
        }
    }
    VecWidth::Scalar
}

fn div_bucket(extent: i64) -> DivBucket {
    let e = extent.max(0);
    if e % 16 == 0 {
        DivBucket::Div16
    } else if e % 8 == 0 {
        DivBucket::Div8
    } else if e % 4 == 0 {
        DivBucket::Div4
    } else if e % 2 == 0 {
        DivBucket::Div2
    } else {
        DivBucket::Any
    }
}

/// Largest linear element offset the operand can touch (`Σ |strideₐ|·(extₐ−1)`),
/// used to pick the index width.
fn max_touched_offset(od: &OperandDesc) -> i64 {
    let mut off: i64 = 0;
    for d in 0..od.rank as usize {
        let span = od.strides[d]
            .saturating_abs()
            .saturating_mul((od.shape[d] - 1).max(0));
        off = off.saturating_add(span);
    }
    off
}

/// Total-work size class from the **iteration-frame numel** — the per-axis max
/// extent across ALL operands (KISS-CLASSIFY §6.5-0010 / §6.6-0013 FRAME-MAX,
/// the ruled work-class semantics), NOT operand-0's numel and NOT the output
/// frame. The frame extent on axis `d` is `max` over operands of that operand's
/// `shape[d]` (or `1` where `d >= operand.rank`), matching the rank-aligned
/// broadcast frame; its product, thresholded, is the work class.
///
/// This distinguishes a skinny cell whose operand-0 is small but whose frame is
/// large — e.g. a contraction `lhs[8,8]·rhs[8,4096]→out[8,4096]`: operand-0
/// numel is `64` (block), but the frame is `max(8,8,8)·max(8,4096,4096) =
/// 8·4096 = 32768` (grid). Reading operand-0 alone mislabels it block; frame-max
/// (and Fuel's deriver, and the KISS golden) say grid.
fn frame_work_class(operands: &[OperandDesc]) -> WorkClass {
    let max_rank = operands.iter().map(|o| o.rank as usize).max().unwrap_or(0);
    let mut numel: i64 = 1;
    for d in 0..max_rank {
        // Per-axis frame extent = max across operands (absent axis ⇒ extent 1,
        // the rank-aligned broadcast identity).
        let frame_d = operands
            .iter()
            .map(|o| {
                if d < o.rank as usize {
                    o.shape[d].max(0)
                } else {
                    1
                }
            })
            .max()
            .unwrap_or(1);
        numel = numel.saturating_mul(frame_d);
    }
    if numel <= 32 {
        WorkClass::OneWarp
    } else if numel <= 1024 {
        WorkClass::OneBlock
    } else {
        WorkClass::GridStride
    }
}

/// Byte size of a byte-addressable dtype, or `None` for sub-byte dtypes (which
/// are treated as non-vectorizable in v1).
fn dtype_size_bytes(dt: ElementKind) -> Option<u32> {
    use ElementKind::{
        B1, Bf16, Bool, Complex64, Complex128, F8E6M2, F8E8M0, F16, F32, F32Strict, F64, Fp8E4M3FN,
        Fp8E4M3FNUZ, Fp8E5M2, Fp8E5M2FNUZ, I4, I8, I16, I32, I64, U4, U8, U16, U32, U64,
    };
    Some(match dt {
        // The `fnuz` FP8 variants are RESERVED (no computation semantics at this
        // schema version) but their *storage* width is pinned by §6.1 all the
        // same — a reserved dtype is still a known dtype, and answering "how wide
        // is it" is not the same as agreeing to compute with it.
        I8 | U8 | Bool | Fp8E4M3FN | Fp8E5M2 | Fp8E4M3FNUZ | Fp8E5M2FNUZ | F8E8M0 | F8E6M2 => 1,
        F16 | Bf16 | I16 | U16 => 2,
        // U32: 4-byte index dtype (the `indices` operand's vec-width side-channel;
        // never a compute operand). Same width class as I32.
        F32 | F32Strict | I32 | U32 => 4,
        F64 | I64 | U64 | Complex64 => 8,
        Complex128 => 16,
        I4 | U4 | B1 => return None,
    })
}

// ===========================================================================
// Token codec
// ===========================================================================

/// Why a `structure_key` token was not honoured.
///
/// Exists because §6.1-0001 requires the reserved-dtype decline to be *distinct*
/// from the unknown-token decline — see [`StructureKey::parse_token`].
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum TokenDecline {
    /// A dtype position named a spelling that is in the closed vocabulary but
    /// has no computation semantics at this schema version (§6.1-0001).
    ReservedDtype {
        /// The reserved spelling as it appeared in the token.
        spelling: String,
    },
    /// The token is not one this schema version can honour — an unknown
    /// spelling, a retired one, or a malformed field.
    Unrecognized,
    /// A **well-formed** token from a schema version this build does not accept.
    ///
    /// Distinct from [`BadVersionPrefix`](Self::BadVersionPrefix) on purpose, and
    /// distinct from every other decline: this is a *recognized* token, correctly
    /// spelled, that simply belongs to another vocabulary. The caller's move is to
    /// **re-derive** it from the request that produced it — not to reach for a
    /// compatibility path. Carries the version so the diagnosis names it.
    UnsupportedSchemaVersion {
        /// The canonical version the token declared.
        version: u16,
    },
    /// The version field is **malformed** — no `sk` prefix, a non-numeric
    /// remainder, or a non-canonical spelling such as a leading zero (`sk04`) or a
    /// sign (`sk+4`). Not a token at all, as opposed to a token from elsewhere.
    BadVersionPrefix,
    /// The non-contraction `(acc + mp)` field (§6.7-0013) is **malformed** — not
    /// exactly two `/`-parts, or an unrecognized `<mp>` code.
    ///
    /// A contraction-shaped payload on a non-`gem` cell lands here, and that is
    /// the intended verdict rather than a near-miss: at sk4 the tenth slot on a
    /// non-`gem` cell **is** the `(acc + mp)` field, so `ctll/d16/f32/f32/f32/st`
    /// there is a six-part spelling of a two-part field, not a contraction group
    /// that wandered. Declining it as malformed is what stops a positional
    /// decoder from silently reinterpreting one cell's precision facts as
    /// another's geometry.
    BadAccMpField,
    /// The `(acc + mp)` field was spelled with **both slots at their defaults**
    /// (accumulator == compute dtype and `<mp>` == `st`).
    ///
    /// Rule (d) of §6.7-0013: a canonical producer omits the field entirely in
    /// that case (rule (c)), so its presence at default is invalid — the same
    /// token would otherwise have two legal spellings, and two spellings of one
    /// cell is a byte-match failure waiting to happen. Distinct from
    /// [`BadAccMpField`](Self::BadAccMpField): this field is *well-formed*, it
    /// just says nothing, and a producer emitting it has a real bug.
    RedundantAccMpField,
    /// Field 8 spelled a reduced-axis set with the **wrong one** of the four
    /// distinct encodings — an `x<hh>` bitmask naming a set that has a sentinel
    /// (`rall` / `rlast`), or `x00` naming the empty set that is `-`.
    ///
    /// Not pedantry about style. §6.6-0009's `x<hh>` form is **domain-restricted**
    /// to "any reduced-axis set that is neither all-axes nor the lone trailing
    /// axis", so such a spelling is not an instance of the fourth value — it is
    /// outside all four, and §6.7-0005 requires a reader to reject any other
    /// field-8 spelling with a typed decline. The clause's "MUST NOT overload a
    /// single sentinel across two of them" binds the reader that accepts it, not
    /// only the producer that emits it.
    ///
    /// Accepting-and-normalizing instead would make `from_token` → `to_token`
    /// byte-unstable for these tokens, and **a key with two spellings for one
    /// meaning is not an identity** — which is the entire basis for using a
    /// `structure_key` as a cache key. (Ruled by the KISS architect, KISS #160.)
    NonCanonicalReduceField,
}

impl StructureKey {
    /// Encode as the stable string token carried on the telemetry wire.
    /// Round-trips through [`StructureKey::from_token`] for every CANONICAL
    /// key. The one deliberately lossy facet (sk3 D4): an `F32Strict`-keyed
    /// struct spells the canonical `f32` — on a gem cell the strictness
    /// re-surfaces as `<mp>=st`, on a non-gem cell it is out-of-band
    /// (§6.6-0018) — so parsing yields the canonical `F32` twin, which
    /// re-emits the identical token.
    ///
    /// Form: `sk<ver>|<op>|<dtype>|<arch>|<idx>|<work>|r<rank>|<op0>;…|<reduce>`
    /// where each operand is `<contig>/<bcasthex>/<vec>/<div>/<flip>`; a gem
    /// cell appends the contraction field
    /// `c<m><n><k>/<kdiv>[/b<class>][/ol<digits>][/or<digits>]/<wdt>/<acc>/<out>/<mp>`.
    #[must_use]
    pub fn to_token(&self) -> String {
        let mut ops = String::new();
        for (i, o) in self
            .operands
            .iter()
            .take(self.n_operands as usize)
            .enumerate()
        {
            if i > 0 {
                ops.push(';');
            }
            ops.push_str(&format!(
                "{}/{:02x}/{}/{}/{}",
                contig_code(o.contig),
                o.bcast.0,
                vec_code(o.vec_width),
                div_code(o.inner_div),
                if o.flipped { 'r' } else { 'f' },
            ));
        }
        let reduce = reduce_field_code(self.reduce_axes, self.rank);
        let mut token = format!(
            "sk{}|{}|{}|{}|{}|{}|r{}|{}|{}",
            self.version,
            op_code(self.op),
            dtype_code(self.dtype),
            arch_code(self.arch),
            idx_code(self.idx),
            work_code(self.work),
            self.rank,
            ops,
            reduce,
        );
        // The OPTIONAL trailing slot. Which field lives here is decided by the
        // OP FAMILY, never by which struct field happens to be populated —
        // encoder and decoder must agree on that question or a ten-field token
        // round-trips into a different key than it came from. §6.7-0013.
        if !carries_contraction_field(self.op) {
            // Non-gem: the `(acc + mp)` field. Re-derived through `AccMp::new`
            // rather than trusted from the struct, so a caller who hand-built a
            // redundant all-default value cannot make us emit a token that our
            // own decoder is required to reject (rule (d)). Rules (b)/(c): both
            // slots spelled when emitted, nothing at all when not — no `-`, no
            // trailing `|`.
            if let Some(a) = self
                .acc_mp
                .and_then(|a| AccMp::new(self.dtype, a.acc, a.mp))
            {
                token.push_str(&format!("|{}/{}", dtype_code(a.acc), mp_code(a.mp)));
            }
            return token;
        }
        // Contraction facts ride as an OPTIONAL trailing field, emitted only
        // when present — every non-contraction token stays byte-identical to
        // the pre-contraction codec (and the wire stays opaque to Fuel).
        if let Some(c) = self.contraction {
            token.push_str(&format!(
                "|c{}{}{}/{}",
                size_code(c.m),
                size_code(c.n),
                size_code(c.k),
                div_code(c.k_div),
            ));
            // Batched cells append `/b<class>` — conditionally present, BEFORE
            // the precision group (batch is an iteration-structure fact, so it
            // sits with the geometry — sk3 RFC §4.1.2).
            if let Some(b) = c.batch {
                token.push_str(&format!("/b{}", size_code(b)));
            }
            // lhs/rhs `LayoutOrder` (item 1 of the layout/shape ramp): additive
            // like `batch` — emitted only when non-identity, so a canonical
            // (row-major) operand pair adds no token component. Each is
            // independent (only rhs transposed ⇒ only `/or...` appears).
            // Ordered lhs-then-rhs, both after `/b<class>` and before the
            // precision group (geometry facts sit together).
            if !c.lhs_order.is_identity() {
                token.push_str(&format!("/ol{}", order_digits(&c.lhs_order)));
            }
            if !c.rhs_order.is_identity() {
                token.push_str(&format!("/or{}", order_digits(&c.rhs_order)));
            }
            // sk3 (D1+D4+D5): the REQUIRED trailing precision group
            // `/<wdt>/<acc>/<out>/<mp>`. `<mp>` codes never begin with `b`, and
            // no dtype/mp code begins with `ol`/`or` either, so the geometry
            // (batch + order) and precision groups never collide in spelling.
            token.push_str(&format!(
                "/{}/{}/{}/{}",
                dtype_code(c.wdt),
                dtype_code(c.acc),
                dtype_code(c.out),
                mp_code(c.mp),
            ));
        }
        token
    }

    /// Parse a token, distinguishing a **reserved** decline from an
    /// **unrecognized** one.
    ///
    /// KISS-Classify §6.1-0001 requires that a `structure_key` using a reserved
    /// dtype be "answered with a typed decline (§6.7-0009) **distinct from the
    /// unknown-token decline**". [`StructureKey::from_token`] returns `Option`,
    /// which by construction cannot express that distinction — both declines
    /// collapse to `None`. This is the conformant surface.
    ///
    /// The distinction carries real information for a consumer. *Unrecognized*
    /// means the peer is speaking a vocabulary this reader does not have, so the
    /// reader may be out of date and upgrading could help. *Reserved* means the
    /// vocabulary is shared and agreed, and this member simply has no semantics
    /// at this schema version — upgrading will not help, and the correct response
    /// is to route around the dtype rather than to suspect a version skew.
    ///
    /// # Errors
    ///
    /// [`TokenDecline::ReservedDtype`] if any dtype position names a reserved
    /// spelling; [`TokenDecline::Unrecognized`] for anything else this schema
    /// version cannot honour.
    pub fn parse_token(token: &str) -> Result<StructureKey, TokenDecline> {
        // Version verdicts first, and in this order: a malformed field is "not a
        // token" (a wall), while a canonical field from another schema is a
        // recognized token that must be RE-DERIVED (a signpost). Collapsing them
        // would leave a consumer unable to tell "you sent me garbage" from "you
        // sent me sk3 and I speak sk4".
        match parse_canonical_version(token.split('|').next().unwrap_or("")) {
            Err(()) => return Err(TokenDecline::BadVersionPrefix),
            Ok(v) if v != STRUCTURE_KEY_VERSION => {
                return Err(TokenDecline::UnsupportedSchemaVersion { version: v });
            }
            Ok(_) => {}
        }
        // Scan atoms rather than positions: the reserved spellings are not
        // legitimate anywhere in a token, so finding one as a whole atom is
        // sufficient and cannot false-positive on a substring (`e4m3fn` is a
        // different atom from `e4m3fnuz`, and atom equality keeps them apart).
        for atom in token.split(['|', '/']) {
            if dtype_from_code(atom).is_some_and(ElementKind::is_reserved) {
                return Err(TokenDecline::ReservedDtype {
                    spelling: atom.to_string(),
                });
            }
        }
        // The §6.7-0013 field's own verdicts, which `from_token` cannot express
        // — it returns `Option`, so "malformed", "redundant" and "unknown op
        // code" all collapse to `None`. Reached through the SAME `parse_acc_mp`
        // the strict decode uses, so the reason reported here is by construction
        // the reason the decode actually failed.
        //
        // Dispatch is by op family, as everywhere else. An unparseable op or
        // dtype code falls through untouched: those are `Unrecognized`, and
        // diagnosing the tenth field of a token whose second field is already
        // gibberish would name a symptom instead of the cause.
        let parts: Vec<&str> = token.split('|').collect();

        // Field 8's own verdict, for the same reason: a non-canonical `x<hh>` is
        // a *specific* refusal (§6.6-0009 / §6.7-0005), and collapsing it into
        // `Unrecognized` would tell a producer "I don't know this token" when
        // the truth is "you spelled a set I understand with the wrong one of
        // four encodings" — actionable versus not.
        if let (Some(field), Some(rank)) = (
            parts.get(8),
            parts
                .get(6)
                .and_then(|s| s.strip_prefix('r')?.parse::<u8>().ok()),
        ) {
            parse_reduce_field(field, rank)?;
        }

        // The §6.7-0013 field's own verdicts, which `from_token` cannot express
        // — it returns `Option`, so "malformed", "redundant" and "unknown op
        // code" all collapse to `None`. Reached through the SAME `parse_acc_mp`
        // the strict decode uses, so the reason reported here is by construction
        // the reason the decode actually failed.
        //
        // Dispatch is by op family, as everywhere else. An unparseable op or
        // dtype code falls through untouched: those are `Unrecognized`, and
        // diagnosing the tenth field of a token whose second field is already
        // gibberish would name a symptom instead of the cause.
        if let (10, Some(op), Some(dt)) = (
            parts.len(),
            parts.get(1).and_then(|s| op_from_code(s)),
            parts.get(2).and_then(|s| dtype_from_code(s)),
        ) {
            if !carries_contraction_field(op) {
                parse_acc_mp(parts[9], dt)?;
            }
        }
        Self::from_token(token).ok_or(TokenDecline::Unrecognized)
    }

    /// Parse a token produced by [`StructureKey::to_token`]. Returns `None` on
    /// any malformed field or an unknown op short-code (a future op category
    /// with no token code assigned).
    #[must_use]
    pub fn from_token(token: &str) -> Option<StructureKey> {
        let parts: Vec<&str> = token.split('|').collect();
        // 9 fields = the base codec; a 10th is the optional contraction field.
        if parts.len() != 9 && parts.len() != 10 {
            return None;
        }
        // Canonical form first, then the ACCEPTED-version check. Before this,
        // the codec accepted any version at all — an `sk4` token decoded happily
        // under the sk3 vocabulary, where `c64` means a pair of f64 rather than a
        // pair of f32. That is the silent cross-vocabulary misdecode
        // §6.7-0014 names, and it was live.
        let version = parse_canonical_version(parts[0]).ok()?;
        if version != STRUCTURE_KEY_VERSION {
            return None;
        }
        let op = op_from_code(parts[1])?;
        let dtype = dtype_from_code(parts[2])?;
        // KISS-Classify §6.1-0001: a RESERVED dtype in ANY dtype position is a
        // typed decline. Recognizing the spelling (above) and refusing the key
        // (here) are two different obligations, and both are required — the
        // spelling must be distinguishable from an unknown token, and the key
        // must not be honoured. See [`StructureKey::parse_token`], which is how
        // a caller tells those two declines apart.
        if dtype.is_reserved() {
            return None;
        }
        let arch = arch_from_code(parts[3])?;
        let idx = match parts[4] {
            // `ix32`/`ix64` (§6.7-0003), not the `i32`/`i64` dtype spellings.
            "ix32" => IdxWidth::Idx32,
            "ix64" => IdxWidth::Idx64,
            _ => return None,
        };
        let work = match parts[5] {
            "warp" => WorkClass::OneWarp,
            "block" => WorkClass::OneBlock,
            "grid" => WorkClass::GridStride,
            _ => return None,
        };
        let rank: u8 = parts[6].strip_prefix('r')?.parse().ok()?;
        // A rank beyond the ceiling is malformed: downstream consumers index
        // MAX_RANK-sized arrays with it. Typed-decline rather than accept.
        if rank as usize > MAX_RANK {
            return None;
        }

        let mut operands = [OperandKey::default(); MAX_OPERANDS];
        let mut n_operands = 0u8;
        if !parts[7].is_empty() {
            // Reject an over-MAX_OPERANDS list rather than silently truncating it
            // to the first MAX_OPERANDS (which would collapse two distinct operand
            // lists onto one key). The reader parses untrusted wire tokens.
            let fields: Vec<&str> = parts[7].split(';').collect();
            if fields.len() > MAX_OPERANDS {
                return None;
            }
            for (slot, field) in operands.iter_mut().zip(fields) {
                *slot = parse_operand(field)?;
                n_operands += 1;
            }
        }

        let reduce_axes = parse_reduce_field(parts[8], rank).ok()?;

        // The tenth field, dispatched by OP FAMILY (§6.7-0013) — mirroring
        // `to_token`. A non-gem cell's tenth field is the `(acc + mp)` field and
        // is never read as a contraction group, so a contraction-shaped payload
        // there declines instead of decoding into geometry facts the cell does
        // not have.
        let acc_mp = match parts.get(9) {
            Some(f) if !carries_contraction_field(op) => Some(parse_acc_mp(f, dtype).ok()?),
            _ => None,
        };
        let contraction = match parts.get(9).filter(|_| carries_contraction_field(op)) {
            None => None,
            Some(f) => {
                // sk3 grammar: `c<m><n><k>/<kdiv>[/b<class>][/ol<digits>]
                // [/or<digits>]/<wdt>/<acc>/<out>/<mp>` — e.g. `ctll/d16/f32/
                // f32/f32/rm`, `ctll/d16/bt/f32/f32/f32/rm`, or
                // `ctll/d16/or10/f32/f32/f32/rm` (rhs transposed, no batch).
                // The precision group is REQUIRED and is always the trailing 4
                // components; the geometry tail between `<kdiv>` and the
                // precision group holds 0-3 OPTIONAL components in the fixed
                // order batch, lhs_order, rhs_order (mirroring emission order)
                // — a component in the wrong slot, or a leftover unrecognized
                // one, is a typed decline, never a silent partial accept.
                let rest = f.strip_prefix('c')?;
                let comps: Vec<&str> = rest.split('/').collect();
                if comps.len() < 6 {
                    return None;
                }
                let prec = &comps[comps.len() - 4..];
                let mut mid = comps[2..comps.len() - 4].iter();
                let mut cur = mid.next();

                let batch = if let Some(comp) = cur {
                    if let Some(b_rest) = comp.strip_prefix('b') {
                        let mut bcs = b_rest.chars();
                        let b = size_from_code(bcs.next()?)?;
                        if bcs.next().is_some() {
                            return None;
                        }
                        cur = mid.next();
                        Some(b)
                    } else {
                        None
                    }
                } else {
                    None
                };
                // A batched cell is rank-3, a plain cell rank-2 (the same rule
                // `derive_contraction` uses) — the rank an absent `/ol`/`/or`
                // must reconstruct as its identity order, so the struct
                // round-trips byte-for-byte through a canonical cell.
                let contraction_rank: u8 = if batch.is_some() { 3 } else { 2 };

                let lhs_order = if let Some(comp) = cur {
                    if let Some(digits) = comp.strip_prefix("ol") {
                        let o = order_from_digits(digits, contraction_rank)?;
                        cur = mid.next();
                        o
                    } else {
                        LayoutOrder::identity(contraction_rank)
                    }
                } else {
                    LayoutOrder::identity(contraction_rank)
                };
                let rhs_order = if let Some(comp) = cur {
                    if let Some(digits) = comp.strip_prefix("or") {
                        let o = order_from_digits(digits, contraction_rank)?;
                        cur = mid.next();
                        o
                    } else {
                        LayoutOrder::identity(contraction_rank)
                    }
                } else {
                    LayoutOrder::identity(contraction_rank)
                };
                // Anything left unconsumed is a malformed/out-of-order geometry
                // component.
                if cur.is_some() {
                    return None;
                }

                let mut cs = comps[0].chars();
                let m = size_from_code(cs.next()?)?;
                let n = size_from_code(cs.next()?)?;
                let k = size_from_code(cs.next()?)?;
                if cs.next().is_some() {
                    return None;
                }
                Some(ContractionKey {
                    m,
                    n,
                    k,
                    k_div: div_from_code(comps[1])?,
                    batch,
                    lhs_order,
                    rhs_order,
                    // §6.1-0001 applies to EVERY dtype position, so the
                    // contraction precision group is guarded exactly like the
                    // top-level dtype: a reserved spelling is recognized by
                    // `dtype_from_code` and refused here.
                    wdt: reject_reserved(dtype_from_code(prec[0])?)?,
                    acc: reject_reserved(dtype_from_code(prec[1])?)?,
                    out: reject_reserved(dtype_from_code(prec[2])?)?,
                    mp: mp_from_code(prec[3])?,
                })
            }
        };

        Some(StructureKey {
            version,
            op,
            dtype,
            arch,
            idx,
            work,
            rank,
            n_operands,
            operands,
            reduce_axes,
            contraction,
            acc_mp,
        })
    }
}

/// One-letter token code for a [`SizeClass`].
const fn size_code(s: SizeClass) -> char {
    match s {
        SizeClass::Tiny => 't',
        SizeClass::Small => 's',
        SizeClass::Mid => 'm',
        SizeClass::Large => 'l',
    }
}

fn size_from_code(c: char) -> Option<SizeClass> {
    Some(match c {
        't' => SizeClass::Tiny,
        's' => SizeClass::Small,
        'm' => SizeClass::Mid,
        'l' => SizeClass::Large,
        _ => return None,
    })
}

fn div_from_code(s: &str) -> Option<DivBucket> {
    Some(match s {
        "d16" => DivBucket::Div16,
        "d8" => DivBucket::Div8,
        "d4" => DivBucket::Div4,
        "d2" => DivBucket::Div2,
        "da" => DivBucket::Any,
        _ => return None,
    })
}

/// Encode a [`LayoutOrder`]'s permutation as the token's `<digits>` component
/// — one decimal digit per axis. `MAX_RANK` (8) keeps every value a single
/// digit (0-7), so no separator between axes is needed.
fn order_digits(o: &LayoutOrder) -> String {
    o.perm().iter().map(u8::to_string).collect()
}

/// Decode a token's `<digits>` component (the [`order_digits`] encoding) back
/// into a [`LayoutOrder`] for an operand of rank `expect_rank`. This parses an
/// UNTRUSTED wire token, so it fully validates that the digits form a true
/// permutation of `0..expect_rank` and returns `None` otherwise — a malformed
/// order (wrong digit count for the contraction's rank, an out-of-range digit,
/// or a duplicate) would build an inconsistent key that later panics when the
/// emitter indexes `perm()[d]` past the operand's axes. Rejects, rather than
/// panicking on the `[u8; MAX_RANK]` write, on:
/// - a non-digit character;
/// - a digit count `!= expect_rank` (a rank-2 order inside a rank-3 cell, etc.);
/// - a digit `>= expect_rank` (out of range for a `0..rank` permutation);
/// - a repeated digit (not a permutation).
fn order_from_digits(s: &str, expect_rank: u8) -> Option<LayoutOrder> {
    let n = s.chars().count();
    let rank = expect_rank as usize;
    if n == 0 || n > MAX_RANK || n != rank {
        return None;
    }
    let mut perm = [0u8; MAX_RANK];
    let mut seen: u16 = 0; // bitset of digits already used — rejects duplicates
    for (i, ch) in s.chars().enumerate() {
        let d = ch.to_digit(10)? as u8;
        if (d as usize) >= rank {
            return None; // out of range for a 0..rank permutation
        }
        let bit = 1u16 << d;
        if seen & bit != 0 {
            return None; // duplicate digit → not a permutation
        }
        seen |= bit;
        perm[i] = d;
    }
    Some(LayoutOrder::from_perm(&perm[..n]))
}

#[cfg(test)]
mod contraction_key_tests {
    use super::*;
    use crate::{ArchSku, ElementKind, OpCategory};

    /// Canonical rank-2 dense matmul key (dense row-major lhs/rhs/out) — the
    /// exact operand shapes/strides of [`gemm_rank2_dense_derives_and_round_trips`],
    /// factored out so layout-order tests can build on it.
    fn sample_matmul_key() -> StructureKey {
        let lhs = OperandDesc::new(2, &[8, 4096], &[4096, 1], ElementKind::F32, 256);
        let rhs = OperandDesc::new(2, &[4096, 4096], &[4096, 1], ElementKind::F32, 256);
        let out = OperandDesc::new(2, &[8, 4096], &[4096, 1], ElementKind::F32, 256);
        structure_key(OpCategory::Gemm, &[lhs, rhs, out], ArchSku::Sm89)
    }

    #[test]
    fn layout_order_eq_and_hash_are_padding_invariant() {
        // `identity(2)` fills the WHOLE [u8; MAX_RANK] array ([0,1,2,3,4,5,6,7]);
        // `from_perm(&[0,1])` fills only perm[..2] and leaves the tail zeroed
        // ([0,1,0,0,0,0,0,0]). Both spell the same logical rank-2 identity
        // permutation and MUST compare/hash equal — a derived Eq/Hash over the
        // full array would wrongly treat them as distinct, which would silently
        // break ContractionKey dispatch-table dedup (it derives Eq/Hash and
        // embeds LayoutOrder).
        use std::collections::HashSet;
        let a = LayoutOrder::identity(2);
        let b = LayoutOrder::from_perm(&[0, 1]);
        assert_eq!(
            a, b,
            "identity(2) and from_perm(&[0,1]) are the same logical permutation"
        );
        let mut set = HashSet::new();
        assert!(set.insert(a), "first insert always succeeds");
        assert!(
            !set.insert(b),
            "b must hash+eq into a's bucket — inserting it must be a no-op"
        );
        assert_eq!(
            set.len(),
            1,
            "padding-invariant Eq/Hash collapse both to one entry"
        );
    }

    #[test]
    fn contraction_layout_order_lhs_only_transposed_round_trips() {
        // lhs storage order [1,0], rhs stays identity → only `/ol10` appears.
        let canon = sample_matmul_key();
        let mut lhs_trans = canon;
        lhs_trans.contraction.as_mut().unwrap().lhs_order = LayoutOrder::from_perm(&[1, 0]);
        let tok = lhs_trans.to_token();
        assert!(tok.contains("/ol10"), "transposed lhs emits /ol10: {tok}");
        assert!(
            !tok.contains("/or"),
            "rhs stays identity, no /or component: {tok}"
        );
        assert_eq!(
            StructureKey::from_token(&tok).unwrap(),
            lhs_trans,
            "lhs-only transposed round-trips"
        );
    }

    #[test]
    fn contraction_layout_order_batch_and_both_orders_round_trip() {
        // Same batched shapes as `gemm_batched_derives_and_round_trips_with_batch_class`,
        // with BOTH operands additionally transposed — batch + lhs_order +
        // rhs_order all present together in the token's geometry tail.
        let lhs = OperandDesc::new(
            3,
            &[8, 8, 4096],
            &[8 * 4096, 4096, 1],
            ElementKind::F32,
            256,
        );
        let rhs = OperandDesc::new(
            3,
            &[8, 4096, 4096],
            &[4096 * 4096, 4096, 1],
            ElementKind::F32,
            256,
        );
        let out = OperandDesc::new(
            3,
            &[8, 8, 4096],
            &[8 * 4096, 4096, 1],
            ElementKind::F32,
            256,
        );
        let mut k = structure_key(OpCategory::Gemm, &[lhs, rhs, out], ArchSku::Sm89);
        let c = k.contraction.as_mut().expect("batched gemm derives facts");
        assert_eq!(c.batch, Some(SizeClass::Tiny));
        c.lhs_order = LayoutOrder::from_perm(&[0, 2, 1]);
        c.rhs_order = LayoutOrder::from_perm(&[1, 0, 2]);
        let tok = k.to_token();
        assert!(
            tok.ends_with("|ctll/d16/bt/ol021/or102/f32/f32/f32/rm"),
            "batch, lhs_order, rhs_order all present in the fixed order: {tok}"
        );
        assert_eq!(
            StructureKey::from_token(&tok),
            Some(k),
            "batch + both orders round-trip together"
        );
    }

    #[test]
    fn from_token_declines_reordered_and_duplicated_layout_order_components() {
        let base = "sk4|gem|f32|cuda:sm89|ix32|grid|r2|\
                    co/00/v4/d16/f;co/00/v4/d16/f;co/00/v4/d16/f|-";
        // `/or...` before `/ol...` violates the fixed emission order (batch,
        // then lhs_order, then rhs_order) the position-driven parser walks —
        // it must decline rather than silently accept an out-of-order
        // geometry tail.
        assert_eq!(
            StructureKey::from_token(&format!("{base}|ctll/d16/or10/ol10/f32/f32/f32/rm")),
            None,
            "reordered or-before-ol declines"
        );
        // A duplicated `/ol...` component: the second one is unconsumed
        // leftover after the (single) lhs_order slot is filled — declines.
        assert_eq!(
            StructureKey::from_token(&format!("{base}|ctll/d16/ol10/ol10/f32/f32/f32/rm")),
            None,
            "duplicated ol component declines"
        );
    }

    #[test]
    fn from_token_declines_malformed_order_digits() {
        // A valid order component's digits must be a TRUE permutation of the
        // contraction's `0..rank`. `from_token` parses an untrusted wire token, so
        // malformed digits must decline rather than build an inconsistent key that
        // panics when the emitter indexes `perm()[d]`.
        let r2 = "sk4|gem|f32|cuda:sm89|ix32|grid|r2|\
                  co/00/v4/d16/f;co/00/v4/d16/f;co/00/v4/d16/f|-";
        // out-of-range digit (7 >= rank 2)
        assert_eq!(
            StructureKey::from_token(&format!("{r2}|ctll/d16/or77/f32/f32/f32/rm")),
            None,
            "out-of-range order digit declines"
        );
        // duplicate digit — not a permutation
        assert_eq!(
            StructureKey::from_token(&format!("{r2}|ctll/d16/or00/f32/f32/f32/rm")),
            None,
            "duplicate order digit declines"
        );
        // digit count != contraction rank (3 digits in a rank-2 cell)
        assert_eq!(
            StructureKey::from_token(&format!("{r2}|ctll/d16/or102/f32/f32/f32/rm")),
            None,
            "order digit-count must match rank-2"
        );
        // A rank-3 (batched) cell with a rank-2 (2-digit) order — the `/bt/ol10`
        // case: a rank-2 LayoutOrder inside a rank-3 contraction would later panic
        // at `perm()[2]`. Must decline on the digit-count/rank mismatch.
        let r3 = "sk4|gem|f32|cuda:sm89|ix32|grid|r3|\
                  co/00/v4/d16/f;co/00/v4/d16/f;co/00/v4/d16/f|-";
        assert_eq!(
            StructureKey::from_token(&format!("{r3}|ctll/d16/bt/ol10/f32/f32/f32/rm")),
            None,
            "a 2-digit order in a rank-3 batched cell declines"
        );
        // Guard against over-rejection: a genuine 2-digit permutation still parses
        // in a rank-2 cell, and a genuine 3-digit one in a rank-3 batched cell.
        assert!(
            StructureKey::from_token(&format!("{r2}|ctll/d16/or10/f32/f32/f32/rm")).is_some(),
            "a valid 2-digit permutation order still parses (rank-2)"
        );
        assert!(
            StructureKey::from_token(&format!("{r3}|ctll/d16/bt/or201/f32/f32/f32/rm")).is_some(),
            "a valid 3-digit permutation order still parses (rank-3 batched)"
        );
    }

    #[test]
    fn contraction_layout_order_token_is_additive() {
        // A canonical rank-2 matmul key: both orders identity → token unchanged.
        let canon = sample_matmul_key();
        let tok_canon = canon.to_token();
        assert!(
            !tok_canon.contains("/ol") && !tok_canon.contains("/or"),
            "identity layout must emit no order component: {tok_canon}"
        );
        assert_eq!(
            StructureKey::from_token(&tok_canon).unwrap(),
            canon,
            "canonical round-trips"
        );

        // A transposed-rhs key: rhs storage order [1,0] → adds `/or10`, round-trips.
        let mut trans = canon;
        trans.contraction.as_mut().unwrap().rhs_order = LayoutOrder::from_perm(&[1, 0]);
        let tok_trans = trans.to_token();
        assert!(
            tok_trans.contains("/or10"),
            "transposed rhs emits /or10: {tok_trans}"
        );
        assert_eq!(
            StructureKey::from_token(&tok_trans).unwrap(),
            trans,
            "transposed round-trips"
        );
        assert_ne!(tok_trans, tok_canon, "transposed cell re-keys");
    }

    #[test]
    fn gemm_rank2_dense_derives_and_round_trips() {
        // Skinny decode cell: [8,4096]·[4096,4096] → [8,4096].
        let lhs = OperandDesc::new(2, &[8, 4096], &[4096, 1], ElementKind::F32, 256);
        let rhs = OperandDesc::new(2, &[4096, 4096], &[4096, 1], ElementKind::F32, 256);
        let out = OperandDesc::new(2, &[8, 4096], &[4096, 1], ElementKind::F32, 256);
        let k = structure_key(OpCategory::Gemm, &[lhs, rhs, out], ArchSku::Sm89);
        let c = k.contraction.expect("gemm cell derives contraction facts");
        assert_eq!(c.m, SizeClass::Tiny);
        assert_eq!(c.n, SizeClass::Large);
        assert_eq!(c.k, SizeClass::Large);
        assert_eq!(c.k_div, DivBucket::Div16);
        // sk3 precision coordinates: plain-F32 operands derive the TF32-routed
        // cell (f32 weight/out, f32 accumulator, reduced-mantissa math).
        assert_eq!(c.wdt, ElementKind::F32);
        assert_eq!(c.acc, ElementKind::F32);
        assert_eq!(c.out, ElementKind::F32);
        assert_eq!(c.mp, MpCode::Rm);
        let tok = k.to_token();
        assert!(
            tok.ends_with("|ctll/d16/f32/f32/f32/rm"),
            "optional trailing field carries the required precision group: {tok}"
        );
        assert_eq!(StructureKey::from_token(&tok), Some(k), "round-trips");
    }

    #[test]
    fn derive_contraction_accepts_transposed_rhs() {
        // rhs logical [K,N]=[16,4] but physically stored [N,K]: K unit-stride,
        // N strided by k=16 → transposed. lhs canonical [M,K].
        let lhs = OperandDesc::new(2, &[8, 16], &[16, 1], ElementKind::F32, 256); // [M,K] row-major
        let rhs = OperandDesc::new(2, &[16, 4], &[1, 16], ElementKind::F32, 256); // [K,N] but N strided by k=16, K unit → transposed store
        let out = OperandDesc::new(2, &[8, 4], &[4, 1], ElementKind::F32, 256); // [M,N] row-major
        let key = structure_key(OpCategory::Gemm, &[lhs, rhs, out], ArchSku::Sm89);
        let c = key
            .contraction
            .expect("transposed rhs must still be a contraction");
        assert!(c.lhs_order.is_identity(), "canonical lhs stays identity");
        assert_eq!(c.rhs_order.perm(), &[1, 0], "transposed rhs order = [1,0]");
    }

    #[test]
    fn derive_contraction_declines_nonpacked() {
        // rhs with a stride that is neither unit nor an extent-product of the other axis
        // (a genuine non-packed slice) → declines to sub-spec D (None).
        let lhs = OperandDesc::new(2, &[8, 16], &[16, 1], ElementKind::F32, 256);
        let rhs = OperandDesc::new(2, &[16, 4], &[9, 1], ElementKind::F32, 256); // K-stride 9 ≠ n(=4), N-stride 1 → non-packed
        let out = OperandDesc::new(2, &[8, 4], &[4, 1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::Gemm, &[lhs, rhs, out], ArchSku::Sm89);
        assert!(
            key.contraction.is_none(),
            "non-packed operand declines in v1 (sub-spec D)"
        );
    }

    #[test]
    fn derive_contraction_accepts_broadcast_batch_rhs() {
        // Batched matmul, rhs KV broadcast over the batch/head axis (stride 0):
        // GQA broadcast-KV. lhs [B,M,K] real batch; rhs [B,K,N] with B broadcast.
        let lhs = OperandDesc::new(3, &[2, 8, 16], &[8 * 16, 16, 1], ElementKind::F32, 256);
        let rhs = OperandDesc::new(3, &[2, 16, 4], &[0, 4, 1], ElementKind::F32, 256); // B stride 0
        let out = OperandDesc::new(3, &[2, 8, 4], &[8 * 4, 4, 1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::Gemm, &[lhs, rhs, out], ArchSku::Sm89);
        let c = key
            .contraction
            .expect("broadcast-batch rhs must still derive a contraction");
        // Hand-trace (rhs strides [0,4,1]): sort desc |stride| -> axes=[1,2,0]
        // (axis1 stride 4, axis2 stride 1, axis0 stride 0). Walk reversed
        // [0,2,1]: d=0 (axis0, stride 0) -> skipped (broadcast). d=2 (axis2,
        // stride 1): |1|==acc(1) ok, acc=1*shape[2]=4. d=1 (axis1, stride 4):
        // |4|==acc(4) ok, acc=4*shape[1]=64. No decline -> admitted.
        // perm[0]=pos(axis0 in [1,2,0])=2; perm[1]=pos(axis1)=0;
        // perm[2]=pos(axis2)=1 -> perm=[2,0,1].
        assert_eq!(
            c.rhs_order.perm(),
            &[2, 0, 1],
            "broadcast-batch rhs storage order"
        );
        // lhs is a real (non-broadcast) canonical batched operand -> identity order.
        assert!(c.lhs_order.is_identity(), "real-batch lhs stays identity");
    }

    #[test]
    fn derive_contraction_broadcast_batch_rhs_round_trips() {
        let lhs = OperandDesc::new(3, &[2, 8, 16], &[8 * 16, 16, 1], ElementKind::F32, 256);
        let rhs = OperandDesc::new(3, &[2, 16, 4], &[0, 4, 1], ElementKind::F32, 256);
        let out = OperandDesc::new(3, &[2, 8, 4], &[8 * 4, 4, 1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::Gemm, &[lhs, rhs, out], ArchSku::Sm89);
        assert!(
            key.contraction.is_some(),
            "broadcast-batch rhs derives a contraction"
        );
        // The rhs operand's bcast mask marks axis 0 (batch broadcast) — an
        // existing, already-tokenized per-operand class (item 03/OperandKey).
        assert!(
            key.operands[1].bcast.is_set(0),
            "rhs operand carries the /bcast mask for axis 0"
        );
        let tok = key.to_token();
        assert!(
            tok.contains("/or"),
            "non-identity rhs order emits an /or component: {tok}"
        );
        assert_eq!(
            StructureKey::from_token(&tok),
            Some(key),
            "broadcast-batch rhs round-trips: both /or (order) and /bcast (mask) survive"
        );
    }

    #[test]
    fn classify_mat_layout_broadcast_admission_is_byte_identical_for_canonical_cells() {
        // Rank-2: same fixture and pinned token suffix as
        // `gemm_rank2_dense_derives_and_round_trips` — no stride-0 axis anywhere,
        // so the new broadcast skip never fires and the token must be unchanged.
        let canon2 = sample_matmul_key();
        let tok2 = canon2.to_token();
        assert!(
            tok2.ends_with("|ctll/d16/f32/f32/f32/rm"),
            "rank-2 canonical token unchanged by broadcast admission: {tok2}"
        );
        // Rank-3: same fixture and pinned token suffix as
        // `gemm_batched_derives_and_round_trips_with_batch_class`.
        let lhs = OperandDesc::new(
            3,
            &[8, 8, 4096],
            &[8 * 4096, 4096, 1],
            ElementKind::F32,
            256,
        );
        let rhs = OperandDesc::new(
            3,
            &[8, 4096, 4096],
            &[4096 * 4096, 4096, 1],
            ElementKind::F32,
            256,
        );
        let out = OperandDesc::new(
            3,
            &[8, 8, 4096],
            &[8 * 4096, 4096, 1],
            ElementKind::F32,
            256,
        );
        let canon3 = structure_key(OpCategory::Gemm, &[lhs, rhs, out], ArchSku::Sm89);
        let tok3 = canon3.to_token();
        assert!(
            tok3.ends_with("|ctll/d16/bt/f32/f32/f32/rm"),
            "rank-3 canonical token unchanged by broadcast admission: {tok3}"
        );
    }

    #[test]
    fn derive_contraction_declines_nonpacked_among_nonbroadcast_axes() {
        // rhs [2,16,4] broadcast batch (stride 0) but K-stride 9 != n(=4) among
        // the non-broadcast axes -> genuinely non-packed there. Broadcast
        // admission must not accidentally admit this: still declines.
        let lhs = OperandDesc::new(3, &[2, 8, 16], &[8 * 16, 16, 1], ElementKind::F32, 256);
        let rhs = OperandDesc::new(3, &[2, 16, 4], &[0, 9, 1], ElementKind::F32, 256);
        let out = OperandDesc::new(3, &[2, 8, 4], &[8 * 4, 4, 1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::Gemm, &[lhs, rhs, out], ArchSku::Sm89);
        assert!(
            key.contraction.is_none(),
            "broadcast admission must not admit an operand non-packed among its non-broadcast axes"
        );
    }

    #[test]
    fn derive_contraction_declines_batch_inner_packed_layout() {
        // A PACKED batched operand that stores the batch axis storage-INNER (not
        // outermost) is out of v1 scope: the emitter binds batch as blockIdx.z
        // (outermost), so a batch-inner batch would reach ext(Batch)'s
        // `unreachable!`. classify_mat_layout is role-agnostic and ADMITS it
        // (packed), so derive_contraction must decline it honestly (sub-spec D)
        // rather than let it reach the emitter's panic.
        //
        // lhs [B,M,K]=[2,8,16] strides [16,32,1]: storage order M-outer (32),
        // batch-middle (16), K-inner (1). classify is packed (max offset
        // 7*32+1*16+15 = 255 < 256) and yields perm=[1,0,2] — batch (axis 0) at
        // storage position 1, NOT 0 — so the guard declines. rhs/out canonical.
        let lhs = OperandDesc::new(3, &[2, 8, 16], &[16, 32, 1], ElementKind::F32, 256);
        let rhs = OperandDesc::new(3, &[2, 16, 4], &[16 * 4, 4, 1], ElementKind::F32, 256);
        let out = OperandDesc::new(3, &[2, 8, 4], &[8 * 4, 4, 1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::Gemm, &[lhs, rhs, out], ArchSku::Sm89);
        assert!(
            key.contraction.is_none(),
            "a packed-but-batch-inner batched operand must decline (else it panics at the emitter's ext(Batch))"
        );
    }

    #[test]
    fn gemm_strict_and_tf32_f32_cells_hold_distinct_tokens_via_mp() {
        // The D4/D1 collision fix: SIMT-f32 (F32Strict operands, full binary32)
        // and TF32-f32 (plain F32 operands, reduced mantissa) are numerically
        // and determinism-distinct cells — under sk2 the `f32s` dtype spelling
        // was their only discriminator; under sk3 BOTH spell dtype `f32` and
        // the `<mp>` coordinate separates them.
        let mk = |dt| {
            let lhs = OperandDesc::new(2, &[8, 4096], &[4096, 1], dt, 256);
            let rhs = OperandDesc::new(2, &[4096, 4096], &[4096, 1], dt, 256);
            let out = OperandDesc::new(2, &[8, 4096], &[4096, 1], dt, 256);
            structure_key(OpCategory::Gemm, &[lhs, rhs, out], ArchSku::Sm89)
        };
        let strict = mk(ElementKind::F32Strict);
        let tf32 = mk(ElementKind::F32);
        // The STRUCT keeps `F32Strict` (the in-process MathPrecision carrier —
        // plan/kernel selection reads it); only the TOKEN folds to `f32`.
        assert_eq!(strict.dtype, ElementKind::F32Strict);
        let (st_tok, rm_tok) = (strict.to_token(), tf32.to_token());
        assert!(!st_tok.contains("f32s"), "no f32s spelling: {st_tok}");
        assert!(
            st_tok.starts_with("sk4|gem|f32|"),
            "strict cell spells the canonical f32 dtype: {st_tok}"
        );
        assert!(
            st_tok.ends_with("|ctll/d16/f32/f32/f32/st"),
            "strict cell = f32-primary + st: {st_tok}"
        );
        assert!(
            rm_tok.ends_with("|ctll/d16/f32/f32/f32/rm"),
            "TF32 cell = f32-primary + rm: {rm_tok}"
        );
        assert_ne!(st_tok, rm_tok, "the sk2 f32s collision is fixed by <mp>");
        // The plain-F32 key round-trips exactly; the strict key parses to its
        // CANONICAL twin (dtype `f32`, identical contraction incl. `mp=st`)
        // that re-emits the identical token — token-level round-trip holds,
        // and the strictness facet survives on the wire via `<mp>` alone.
        assert_eq!(StructureKey::from_token(&rm_tok), Some(tf32));
        let twin = StructureKey::from_token(&st_tok).expect("st token parses");
        assert_eq!(twin.dtype, ElementKind::F32);
        assert_eq!(twin.contraction, strict.contraction);
        assert_eq!(twin.to_token(), st_tok, "canonical twin re-emits the token");
    }

    #[test]
    fn gemm_mixed_fp8_cells_hold_distinct_tokens_via_wdt_and_out() {
        // The RFC §2 motivating collision: a mixed-input FP8 GEMM and a
        // homogeneous one previously derived byte-identical tokens (the key
        // spelled only operand-0's dtype). sk3's `<wdt>`/`<out>` disambiguate,
        // and the FP8 spellings are variant-explicit (`e4m3fn`, bare `e5m2`).
        let lhs = |dt| OperandDesc::new(2, &[8, 4096], &[4096, 1], dt, 256);
        let rhs = |dt| OperandDesc::new(2, &[4096, 4096], &[4096, 1], dt, 256);
        let out = |dt| OperandDesc::new(2, &[8, 4096], &[4096, 1], dt, 256);
        let mixed = structure_key(
            OpCategory::Gemm,
            &[
                lhs(ElementKind::Fp8E4M3FN),
                rhs(ElementKind::Fp8E5M2),
                out(ElementKind::F32),
            ],
            ArchSku::Sm89,
        );
        let homog = structure_key(
            OpCategory::Gemm,
            &[
                lhs(ElementKind::Fp8E4M3FN),
                rhs(ElementKind::Fp8E4M3FN),
                out(ElementKind::F16),
            ],
            ArchSku::Sm89,
        );
        let (mt, ht) = (mixed.to_token(), homog.to_token());
        assert!(
            mt.starts_with("sk4|gem|f8e4m3fn|"),
            "variant-explicit: {mt}"
        );
        assert!(
            mt.ends_with("|ctll/d16/f8e5m2/f32/f32/st"),
            "mixed cell spells its e5m2 weight + f32 out: {mt}"
        );
        assert!(
            ht.ends_with("|ctll/d16/f8e4m3fn/f32/f16/st"),
            "homogeneous cell spells its e4m3fn weight + f16 out: {ht}"
        );
        assert_ne!(mt, ht, "the sk2 FP8 collision is fixed by <wdt>/<out>");
        assert_eq!(StructureKey::from_token(&mt), Some(mixed));
        assert_eq!(StructureKey::from_token(&ht), Some(homog));
    }

    #[test]
    fn from_token_declines_sk2_shaped_and_malformed_gem_precision_groups() {
        let base = "sk4|gem|f32|cuda:sm89|ix32|grid|r2|\
                    co/00/v4/d16/f;co/00/v4/d16/f;co/00/v4/d16/f|-";
        // The precision group is REQUIRED for an sk3 gem cell: the sk2 shapes
        // (`c<mnk>/<kdiv>` and `c<mnk>/<kdiv>/b<class>`) are typed declines.
        assert_eq!(StructureKey::from_token(&format!("{base}|ctll/d16")), None);
        assert_eq!(
            StructureKey::from_token(&format!("{base}|ctll/d16/bt")),
            None
        );
        // 7 components whose third is NOT `b<class>` — malformed.
        assert_eq!(
            StructureKey::from_token(&format!("{base}|ctll/d16/f32/f32/f32/st/st")),
            None
        );
        // Retired / renamed / reserved dtype spellings inside the precision
        // group are typed declines: `f32s` (D4), bare `e4m3` (renamed
        // `e4m3fn`), and the reserved-unused AMD `e4m3fnuz`.
        for wdt in ["f32s", "e4m3", "e4m3fnuz"] {
            assert_eq!(
                StructureKey::from_token(&format!("{base}|ctll/d16/{wdt}/f32/f32/st")),
                None,
                "retired/unknown wdt spelling must decline: {wdt}"
            );
        }
        // An unknown mp code declines (future `rm10`/`rm7` are additive-later).
        assert_eq!(
            StructureKey::from_token(&format!("{base}|ctll/d16/f32/f32/f32/rm10")),
            None
        );
        // The `f32s` retirement also holds at the token's DTYPE field.
        let f32s_dtype = "sk4|gem|f32s|cuda:sm89|ix32|grid|r2|\
                          co/00/v4/d16/f;co/00/v4/d16/f;co/00/v4/d16/f|-|ctll/d16/f32/f32/f32/st";
        assert_eq!(StructureKey::from_token(f32s_dtype), None);
    }

    #[test]
    fn gemm_bias_dense_derives_but_strided_or_broadcast_bias_declines() {
        let lhs = OperandDesc::new(2, &[8, 4096], &[4096, 1], ElementKind::F32, 256);
        let rhs = OperandDesc::new(2, &[4096, 4096], &[4096, 1], ElementKind::F32, 256);
        let out = OperandDesc::new(2, &[8, 4096], &[4096, 1], ElementKind::F32, 256);
        // A DENSE per-column [N] bias (unit stride) derives — the bias does not
        // change the ContractionKey facts (token stays `ctll/d16`).
        let bias = OperandDesc::new(1, &[4096], &[1], ElementKind::F32, 256);
        let k = structure_key(OpCategory::Gemm, &[lhs, rhs, bias, out], ArchSku::Sm89);
        assert!(k.contraction.is_some(), "dense bias cell derives facts");
        assert!(
            k.to_token().ends_with("|ctll/d16/f32/f32/f32/rm"),
            "bias unchanged token"
        );
        // A STRIDED bias (every-other element) declines: the emitter reads a
        // hardcoded in2[col], so a non-unit stride would silently mis-read.
        let strided = OperandDesc::new(1, &[4096], &[2], ElementKind::F32, 256);
        let ks = structure_key(OpCategory::Gemm, &[lhs, rhs, strided, out], ArchSku::Sm89);
        assert!(ks.contraction.is_none(), "strided bias must decline");
        // A BROADCAST bias (stride 0) declines: in2[col] over 0..N would read past
        // a 1-element allocation (OOB device read).
        let bcast = OperandDesc::new(1, &[4096], &[0], ElementKind::F32, 256);
        let kb = structure_key(OpCategory::Gemm, &[lhs, rhs, bcast, out], ArchSku::Sm89);
        assert!(kb.contraction.is_none(), "broadcast bias must decline");
    }

    #[test]
    fn gemm_batched_derives_and_round_trips_with_batch_class() {
        // [8,8,4096]·[8,4096,4096] → [8,8,4096] (B/M Tiny).
        let lhs = OperandDesc::new(
            3,
            &[8, 8, 4096],
            &[8 * 4096, 4096, 1],
            ElementKind::F32,
            256,
        );
        let rhs = OperandDesc::new(
            3,
            &[8, 4096, 4096],
            &[4096 * 4096, 4096, 1],
            ElementKind::F32,
            256,
        );
        let out = OperandDesc::new(
            3,
            &[8, 8, 4096],
            &[8 * 4096, 4096, 1],
            ElementKind::F32,
            256,
        );
        let k = structure_key(OpCategory::Gemm, &[lhs, rhs, out], ArchSku::Sm89);
        let c = k.contraction.expect("batched gemm derives facts");
        assert_eq!(c.batch, Some(SizeClass::Tiny));
        let tok = k.to_token();
        assert!(
            tok.ends_with("|ctll/d16/bt/f32/f32/f32/rm"),
            "batched token carries /b<class> BEFORE the precision group: {tok}"
        );
        assert_eq!(
            StructureKey::from_token(&tok),
            Some(k),
            "batched round-trips"
        );
        // A genuinely non-packed rank-3 lhs still declines (Task 2: only a PACKED
        // permutation now derives lhs_order/rhs_order — arbitrary strides still
        // decline to sub-spec D). Note strides `[1, 8, 64]` would NOT qualify as
        // "non-dense" any more: axis0 stride 1, axis1 stride 8 (=shape[0]),
        // axis2 stride 64 (=shape[0]*shape[1]) is a valid packed permutation
        // that Task 2 now accepts — this fixture instead perturbs the middle
        // stride (9, not 8) so it fails the packed check.
        let nd = OperandDesc::new(3, &[8, 8, 4096], &[1, 9, 64], ElementKind::F32, 256);
        let knd = structure_key(OpCategory::Gemm, &[nd, rhs, out], ArchSku::Sm89);
        assert!(
            knd.contraction.is_none(),
            "genuinely non-packed rank-3 declines"
        );
    }

    #[test]
    fn work_class_is_frame_max_across_operands_not_operand_zero() {
        // KISS-CLASSIFY §6.5-0010/§6.6-0013 FRAME-MAX ruling (Eric 2026-07-23,
        // KISS #82 finding 2 / PR #85): the work class is the per-axis max
        // extent across ALL operands, NOT operand-0's numel and NOT the output
        // frame. The two #85 disambiguating goldens pin frame-max against both
        // alternative readings.

        // Cell A — lhs[8,4096]·rhs[4096,8]→out[8,8]. frame-max =
        // max(8,4096)·max(4096,8) = 4096·4096 → grid; the OUTPUT frame (8·8=64)
        // would read block. Catches the output-frame reading.
        let a_lhs = OperandDesc::new(2, &[8, 4096], &[4096, 1], ElementKind::F32, 256);
        let a_rhs = OperandDesc::new(2, &[4096, 8], &[8, 1], ElementKind::F32, 256);
        let a_out = OperandDesc::new(2, &[8, 8], &[8, 1], ElementKind::F32, 256);
        let ka = structure_key(OpCategory::Gemm, &[a_lhs, a_rhs, a_out], ArchSku::Sm89);
        assert_eq!(
            ka.work,
            WorkClass::GridStride,
            "frame-max (4096²) is grid, not the output-frame block reading"
        );

        // Cell B — lhs[8,8]·rhs[8,4096]→out[8,4096]. frame-max =
        // max(8,8,8)·max(8,4096,4096) = 8·4096 = 32768 → grid; OPERAND-0's numel
        // (8·8=64) would read block. Catches the operand-0-numel reading (the
        // bug this fix closes — under the old `work_class(operands.first())`
        // this derived block).
        let b_lhs = OperandDesc::new(2, &[8, 8], &[8, 1], ElementKind::F32, 256);
        let b_rhs = OperandDesc::new(2, &[8, 4096], &[4096, 1], ElementKind::F32, 256);
        let b_out = OperandDesc::new(2, &[8, 4096], &[4096, 1], ElementKind::F32, 256);
        let kb = structure_key(OpCategory::Gemm, &[b_lhs, b_rhs, b_out], ArchSku::Sm89);
        assert_eq!(
            kb.work,
            WorkClass::GridStride,
            "frame-max (8·4096) is grid, not the operand-0-numel block reading"
        );

        // Elementwise sanity: rank-aligned operands ⇒ frame-max ≡ operand-0
        // numel (the zero-churn case for every shipped elementwise golden).
        let e = OperandDesc::new(2, &[128, 256], &[256, 1], ElementKind::F32, 256);
        let ke = structure_key(OpCategory::BinaryElementwise, &[e, e, e], ArchSku::Sm89);
        assert_eq!(ke.work, WorkClass::GridStride);

        // A genuine size-1 operand-0 broadcast still reads the frame from the
        // larger operand: a[1,1] (numel 1, warp) with b[64,64] (4096, grid) ⇒
        // frame-max grid, never warp.
        let s = OperandDesc::new(2, &[1, 1], &[1, 1], ElementKind::F32, 256);
        let big = OperandDesc::new(2, &[64, 64], &[64, 1], ElementKind::F32, 256);
        let ks = structure_key(OpCategory::BinaryElementwise, &[s, big, big], ArchSku::Sm89);
        assert_eq!(
            ks.work,
            WorkClass::GridStride,
            "a small operand-0 must not shrink the frame below the larger operand"
        );
    }

    #[test]
    fn zero_length_inner_run_derives_scalar_vec_width_not_v4() {
        // §6.5-0009(c) E=0 divisibility trap (KISS #82/#87, cross-impl sweep
        // 2026-07-24): `inner_extent % L == 0` is VACUOUSLY TRUE at E==0, so a
        // ladder without a zero guard emits v4 for a zero-length run. Baracuda
        // is structurally immune — `inner_axis` selects ONLY a strictly-non-unit
        // axis (`shape[d] > 1`), so the divisibility ladder never sees E ∈ {0,1}.
        // This pins the OUTCOME (E=0 → Scalar/v1) so a future `inner_axis`
        // refactor that stopped excluding non-unit axes is caught. (KISS had the
        // bug + fixed it #87; Fuel had it + fixing; Baracuda never had it.)

        // A bare empty operand `[0]` / `[0,0]`: no axis is > 1, so inner_axis is
        // None → Scalar via the guard, and div bucket is Any — a consistent
        // (v1, da) pair, never the (v4, da) inconsistency the bug produces.
        for shape in [&[0i64][..], &[0, 0][..]] {
            let z = OperandDesc::new(
                shape.len(),
                shape,
                &vec![1i64; shape.len()],
                ElementKind::F32,
                256,
            );
            let k = structure_key(OpCategory::UnaryElementwise, &[z, z], ArchSku::Sm89);
            assert_eq!(
                k.operands[0].vec_width,
                VecWidth::Scalar,
                "empty operand {shape:?} must derive Scalar, never a vectorized width"
            );
            assert_eq!(k.operands[0].inner_div, DivBucket::Any);
        }

        // MIXED case `[4, 0]` (zero-length INNER axis): `inner_axis` skips axis 1
        // (0, not > 1) and picks axis 0 (4); a dense `[4,0]` has axis-0 stride 0,
        // so it lands Scalar before the divisibility loop. Pins the outcome
        // (a zero-inner operand → Scalar, never v4).
        let inner_zero = OperandDesc::new(2, &[4, 0], &[0, 1], ElementKind::F32, 256);
        let kz = structure_key(
            OpCategory::UnaryElementwise,
            &[inner_zero, inner_zero],
            ArchSku::Sm89,
        );
        assert_eq!(
            kz.operands[0].vec_width,
            VecWidth::Scalar,
            "[4,0] must derive Scalar"
        );

        // The load-bearing MECHANISM pin — `[0, 256]` (zero OUTER axis, non-zero
        // unit-stride inner): `inner_axis` SKIPS the zero axis 0 and picks axis 1
        // (256), so the divisibility ladder DOES run, on ext=256 (never the 0),
        // and correctly derives V4. This proves the vacuous-`E % L == 0`-at-0 path
        // is unreachable: a zero axis present in the operand does not corrupt the
        // ladder, because `inner_axis` fed it the real 256, not the 0.
        let outer_zero = OperandDesc::new(2, &[0, 256], &[256, 1], ElementKind::F32, 256);
        let ko = structure_key(
            OpCategory::UnaryElementwise,
            &[outer_zero, outer_zero],
            ArchSku::Sm89,
        );
        assert_eq!(
            ko.operands[0].vec_width,
            VecWidth::V4,
            "[0,256] must derive V4 from the real 256-inner — the zero axis is \
             skipped by inner_axis, never fed to the divisibility ladder"
        );
    }

    #[test]
    fn non_gemm_and_malformed_gemm_stay_none_and_byte_identical() {
        // A non-GEMM cell: no contraction facts, token has exactly 9 fields —
        // byte-identical to the pre-contraction codec.
        let a = OperandDesc::new(2, &[128, 256], &[256, 1], ElementKind::F32, 256);
        let k = structure_key(OpCategory::BinaryElementwise, &[a, a, a], ArchSku::Sm89);
        assert!(k.contraction.is_none());
        assert_eq!(k.to_token().split('|').count(), 9);
        // GEMM with a shape mismatch (K disagreement) → honest None.
        let bad = OperandDesc::new(2, &[100, 256], &[256, 1], ElementKind::F32, 256);
        let kb = structure_key(OpCategory::Gemm, &[a, bad, a], ArchSku::Sm89);
        assert!(kb.contraction.is_none());
        // GEMM with a transposed (column-major) operand → None (v1 is dense
        // row-major only).
        let t = OperandDesc::new(2, &[256, 128], &[1, 256], ElementKind::F32, 256);
        let kt = structure_key(OpCategory::Gemm, &[a, t, a], ArchSku::Sm89);
        assert!(kt.contraction.is_none());
    }
}

fn parse_operand(field: &str) -> Option<OperandKey> {
    let f: Vec<&str> = field.split('/').collect();
    if f.len() != 5 {
        return None;
    }
    let contig = match f[0] {
        "co" => Contiguity::Contig,
        "ic" => Contiguity::InnerContig,
        "st" => Contiguity::Strided,
        "br" => Contiguity::Broadcast,
        _ => return None,
    };
    let bcast = AxisMask(u8::from_str_radix(f[1], 16).ok()?);
    let vec_width = match f[2] {
        "v1" => VecWidth::Scalar,
        "v2" => VecWidth::V2,
        "v4" => VecWidth::V4,
        "v8" => VecWidth::V8,
        _ => return None,
    };
    let inner_div = match f[3] {
        "d16" => DivBucket::Div16,
        "d8" => DivBucket::Div8,
        "d4" => DivBucket::Div4,
        "d2" => DivBucket::Div2,
        "da" => DivBucket::Any,
        _ => return None,
    };
    let flipped = match f[4] {
        "f" => false,
        "r" => true,
        _ => return None,
    };
    Some(OperandKey {
        contig,
        bcast,
        vec_width,
        inner_div,
        flipped,
    })
}

const fn idx_code(v: IdxWidth) -> &'static str {
    match v {
        // `ix32`/`ix64` per KISS-CLASSIFY-6.7-0003 — deliberately DISTINCT from
        // the `i32`/`i64` dtype tokens so the index-width field never aliases a
        // dtype spelling.
        IdxWidth::Idx32 => "ix32",
        IdxWidth::Idx64 => "ix64",
    }
}

const fn work_code(v: WorkClass) -> &'static str {
    match v {
        WorkClass::OneWarp => "warp",
        WorkClass::OneBlock => "block",
        WorkClass::GridStride => "grid",
    }
}

const fn contig_code(v: Contiguity) -> &'static str {
    match v {
        Contiguity::Contig => "co",
        Contiguity::InnerContig => "ic",
        Contiguity::Strided => "st",
        Contiguity::Broadcast => "br",
    }
}

const fn vec_code(v: VecWidth) -> &'static str {
    match v {
        VecWidth::Scalar => "v1",
        VecWidth::V2 => "v2",
        VecWidth::V4 => "v4",
        VecWidth::V8 => "v8",
    }
}

const fn div_code(v: DivBucket) -> &'static str {
    match v {
        DivBucket::Div16 => "d16",
        DivBucket::Div8 => "d8",
        DivBucket::Div4 => "d4",
        DivBucket::Div2 => "d2",
        DivBucket::Any => "da",
    }
}

const fn arch_code(v: ArchSku) -> &'static str {
    // Namespaced `target_capability` per KISS-CLASSIFY-6.8 (`<namespace>:<cap>`,
    // matched byte-exact, §6.8-0002). `ArchSku` stays a CUDA-only enum internally;
    // only its token gains the `cuda:` namespace, so a future cpu:/vulkan: backend
    // slots in without perturbing these CUDA tokens.
    match v {
        ArchSku::Sm80 => "cuda:sm80",
        ArchSku::Sm89 => "cuda:sm89",
        ArchSku::Sm90a => "cuda:sm90a",
    }
}

fn arch_from_code(s: &str) -> Option<ArchSku> {
    Some(match s {
        "cuda:sm80" => ArchSku::Sm80,
        "cuda:sm89" => ArchSku::Sm89,
        "cuda:sm90a" => ArchSku::Sm90a,
        _ => return None,
    })
}

/// The KISS-Classify closed-set dtype token spelling (§6.1). Public because the
/// KISS-Contract §6.8 `accumulation_type` field MUST use the SAME spelling as
/// the key's `<acc>` coordinate (one dtype, two surfaces — the sk3 RFC §4.2
/// pin), so the contract emitter spells through this one function.
#[must_use]
pub const fn dtype_token(v: ElementKind) -> &'static str {
    dtype_code(v)
}

const fn dtype_code(v: ElementKind) -> &'static str {
    use ElementKind::{
        B1, Bf16, Bool, Complex64, Complex128, F8E6M2, F8E8M0, F16, F32, F32Strict, F64, Fp8E4M3FN,
        Fp8E4M3FNUZ, Fp8E5M2, Fp8E5M2FNUZ, I4, I8, I16, I32, I64, U4, U8, U16, U32, U64,
    };
    match v {
        F16 => "f16",
        Bf16 => "bf16",
        F32 => "f32",
        // sk3 D4: the `f32s` spelling is RETIRED from the closed set
        // (§6.1-0005 forbids a strict-precision dtype token). Derived keys
        // never carry `F32Strict` (`canonical_dtype` folds it), so this arm is
        // the total-match backstop for a hand-built key: it spells the
        // canonical `f32`, and the strict axis rides the gem `<mp>` coordinate.
        F32Strict => "f32",
        F64 => "f64",
        I8 => "i8",
        U8 => "u8",
        I32 => "i32",
        I64 => "i64",
        // U32: index dtype. New code, so no pre-existing token changes (no
        // pre-existing cell keys a u32 operand) — STRUCTURE_KEY_VERSION stays 1.
        U32 => "u32",
        Bool => "bool",
        // sk3 D4: variant-explicit FP8. `e4m3fn` (OCP, SATFINITE, no-inf, max
        // 448) renames the bare `e4m3`; `e5m2` is ALREADY the variant-explicit
        // IEEE-style spelling (inf/NaN, max 57344) and stays.
        Fp8E4M3FN => "f8e4m3fn",
        Fp8E5M2 => "f8e5m2",
        // The AMD `fnuz` variants are RESERVED by §6.1-0001: part of the closed
        // vocabulary, no computation semantics at this schema version. They
        // spell so a reader can RECOGNIZE them and tell them apart from an
        // unknown token — the clause requires exactly that distinction. Using
        // one in a dtype position is a typed decline, not a parse failure.
        Fp8E4M3FNUZ => "f8e4m3fnuz",
        // MX shared block SCALES — active at sk4, not reserved.
        F8E8M0 => "f8e8m0",
        F8E6M2 => "f8e6m2",
        Fp8E5M2FNUZ => "f8e5m2fnuz",
        I16 => "i16",
        U16 => "u16",
        U64 => "u64",
        I4 => "i4",
        U4 => "u4",
        B1 => "b1",
        Complex64 => "c64",
        Complex128 => "c128",
    }
}

/// Parse the `sk<N>` version field under the canonical-form rule.
///
/// `Ok(v)` for a canonical decimal (`0` or `[1-9][0-9]*`); `Err(())` for anything
/// malformed.
///
/// **The canonical check runs before the numeric parse, and that ordering is the
/// whole point** (KISS-CLASSIFY-6.7-0015). A bare `parse::<u16>()` accepts both
/// `sk04` and `sk+4` as `4` — Rust's integer parser tolerates leading zeros and a
/// leading sign. Both spellings were accepted by this codec before this guard,
/// which is a wall that was not there.
fn parse_canonical_version(field: &str) -> Result<u16, ()> {
    let digits = field.strip_prefix("sk").ok_or(())?;
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return Err(());
    }
    // Canonical: "0", or no leading zero.
    if digits.len() > 1 && digits.starts_with('0') {
        return Err(());
    }
    digits.parse::<u16>().map_err(|_| ())
}

/// `None` for a dtype that is RESERVED at this schema version, else the dtype.
///
/// The gate for "recognized but not usable" (KISS-Classify §6.1-0001). Separate
/// from [`dtype_from_code`] on purpose: recognizing a spelling and honouring a
/// key that uses it are different obligations, and the clause requires the first
/// while forbidding the second.
fn reject_reserved(dt: ElementKind) -> Option<ElementKind> {
    if dt.is_reserved() { None } else { Some(dt) }
}

/// The all-axes mask for a `rank`-dimensional iteration frame.
///
/// Truncating to `u8` mirrors [`AxisMask`]'s width exactly, so the encoder and
/// the decoder agree on the saturating case at `rank >= 8` instead of disagreeing
/// about a token neither can represent.
const fn all_axes_mask(rank: u8) -> u8 {
    ((1u16 << rank) - 1) as u8
}

/// Spell field 8, the reduce spec — KISS-CLASSIFY-6.6-0009 / §6.7-0005.
///
/// Four **distinctly-encoded** values, and the clause forbids overloading one
/// sentinel across two of them: `-` (not a reduction), `rall` (every
/// iteration-frame axis), `rlast` (exactly the lone innermost axis), and
/// `x<hh>` (any other reduced set).
///
/// # This is a producer MUST, and it was previously violated
///
/// This codec used to emit `x<hh>` for *every* non-empty mask, including the two
/// cases §6.7-0005 requires be spelled `rall`/`rlast` — "byte-different but
/// semantically identical", as the decoder's comment put it. Semantically
/// identical is not the standard the reduce field is held to: §6.6-0009 says the
/// `x<hh>` form MUST NOT be used for those two cases, precisely so that two
/// conforming implementations produce the same bytes for the same cell. A
/// reduction cell keyed by this crate and the same cell keyed by KISS would have
/// differed on every all-axes and every trailing-axis reduction.
///
/// # The rank-1 tie-break
///
/// A rank-1 reduction reduces a set that is *simultaneously* all axes and the
/// lone innermost axis. §6.6-0009 pins `rall` as the winner — checked first
/// here — "so two conforming implementations never disagree on the rank-1
/// encoding". Ordering these two branches the other way is a silent byte
/// divergence on the single most common reduction shape there is.
fn reduce_field_code(axes: AxisMask, rank: u8) -> String {
    if axes.is_empty() {
        return "-".to_string();
    }
    if rank > 0 {
        // `rall` BEFORE `rlast` — the §6.6-0009 precedence, load-bearing at rank 1.
        if axes.0 == all_axes_mask(rank) {
            return "rall".to_string();
        }
        if axes.0 == 1u8 << (rank - 1) {
            return "rlast".to_string();
        }
    }
    format!("x{:02x}", axes.0)
}

/// Decode field 8, the reduce spec — the reader half of [`reduce_field_code`].
///
/// The four §6.6-0009 encodings, and **only** the four. The sentinels resolve
/// against the token's `rank`: the widest-operand / input-axis space the mask
/// indexes (see [`structure_key`], `rank` = max operand rank).
///
/// # Rejecting a non-canonical spelling is required of the READER
///
/// §6.6-0009's fourth value is domain-restricted — it is an explicit bitmask
/// "for any reduced-axis set that is **neither all-axes nor the lone trailing
/// axis**". An `x03` at rank 2, whose bits are all axes, is therefore not an
/// instance of that value; it is outside all four, and §6.7-0005 says a reader
/// "MUST reject any other field-8 spelling with a typed decline". The clause's
/// prohibition on overloading one sentinel across two values binds the reader
/// that accepts such a token, not only the producer that emits it.
///
/// This codec accepted them until now, on the reasoning that declining was a
/// stronger claim than the clause made. It isn't, and the cost of the lenient
/// reading is concrete: normalizing on read makes `from_token` → `to_token`
/// byte-unstable, and **a key admitting two spellings for one meaning is not an
/// identity** — which is the entire basis for using a `structure_key` as a cache
/// key. §6.7-0008's round-trip byte-identity is scoped to keys whose fields
/// satisfy the §6.5–§6.6 domains, which these do not. (KISS architect ruling,
/// KISS #160.)
///
/// `x00` is rejected on the same footing: the empty set is `-`, and spelling it
/// as a bitmask is the same overload pointing at the first value instead of the
/// second or third.
///
/// # Errors
///
/// [`TokenDecline::NonCanonicalReduceField`] for a well-formed spelling of a set
/// that has a different canonical encoding; [`TokenDecline::Unrecognized`] for a
/// field that is not one of the four forms at all (including `rlast` in a rank-0
/// space, which has no trailing axis, and `rall` there, which would name the
/// empty set that `-` already owns).
fn parse_reduce_field(field: &str, rank: u8) -> Result<AxisMask, TokenDecline> {
    match field {
        "-" => Ok(AxisMask::EMPTY),
        "rall" if rank > 0 => Ok(AxisMask(all_axes_mask(rank))),
        "rlast" if rank > 0 => Ok(AxisMask(1u8 << (rank - 1))),
        // rank 0: neither sentinel has a referent. `rlast` has no trailing axis,
        // and `rall` would denote the empty set — which is `-`'s value, so
        // accepting it would be the overload §6.6-0009 forbids.
        "rall" | "rlast" => Err(TokenDecline::Unrecognized),
        s => {
            let hex = s.strip_prefix('x').ok_or(TokenDecline::Unrecognized)?;
            let mask = u8::from_str_radix(hex, 16).map_err(|_| TokenDecline::Unrecognized)?;
            // Well-formed, but is this set's canonical spelling one of the
            // sentinels (or `-`)? If so the token is outside all four values.
            let canonical_is_elsewhere = mask == 0
                || (rank > 0 && (mask == all_axes_mask(rank) || mask == 1u8 << (rank - 1)));
            if canonical_is_elsewhere {
                return Err(TokenDecline::NonCanonicalReduceField);
            }
            Ok(AxisMask(mask))
        }
    }
}

/// Decode the non-contraction `(acc + mp)` field (§6.7-0013) for a cell whose
/// compute dtype is `compute`.
///
/// The **single** implementation of this field's rules, deliberately: it is
/// called by [`StructureKey::from_token`] (which discards the reason) and by
/// [`StructureKey::parse_token`] (which reports it). Two copies would be two
/// opportunities for the strict path and the typed path to disagree about what
/// is legal, and a consumer would then get a decline reason describing a
/// different verdict than the one that was actually reached.
///
/// # Errors
///
/// - [`TokenDecline::BadAccMpField`] — not exactly two `/`-parts, or an
///   unrecognized `<mp>` code.
/// - [`TokenDecline::ReservedDtype`] — an accumulator that is a recognized but
///   reserved §6.1 member. Distinct from an unknown spelling, per §6.1-0001;
///   every dtype position is governed by that clause, and the accumulator slot
///   is a dtype position.
/// - [`TokenDecline::Unrecognized`] — an accumulator outside the closed set.
/// - [`TokenDecline::RedundantAccMpField`] — well-formed but all-default
///   (rule (d)).
fn parse_acc_mp(field: &str, compute: ElementKind) -> Result<AccMp, TokenDecline> {
    let parts: Vec<&str> = field.split('/').collect();
    // Exactly two. A contraction group (six parts) fails here, which is the
    // point: on a non-gem cell this slot is not the contraction group's.
    let [acc_code, mp_str] = parts[..] else {
        return Err(TokenDecline::BadAccMpField);
    };
    let acc = dtype_from_code(acc_code).ok_or(TokenDecline::Unrecognized)?;
    if acc.is_reserved() {
        return Err(TokenDecline::ReservedDtype {
            spelling: acc_code.to_string(),
        });
    }
    let mp = mp_from_code(mp_str).ok_or(TokenDecline::BadAccMpField)?;
    // Rule (d). `AccMp::new` is the rule; asking it is how the encoder's notion
    // of "worth emitting" and the decoder's notion of "legal to have emitted"
    // stay the same notion.
    AccMp::new(compute, acc, mp).ok_or(TokenDecline::RedundantAccMpField)
}

fn dtype_from_code(s: &str) -> Option<ElementKind> {
    use ElementKind::{
        B1, Bf16, Bool, Complex64, Complex128, F8E6M2, F8E8M0, F16, F32, F64, Fp8E4M3FN,
        Fp8E4M3FNUZ, Fp8E5M2, Fp8E5M2FNUZ, I4, I8, I16, I32, I64, U4, U8, U16, U32, U64,
    };
    Some(match s {
        "f16" => F16,
        "bf16" => Bf16,
        "f32" => F32,
        // `"f32s"` deliberately ABSENT (sk3 D4 retirement) — a peer token still
        // spelling it is a typed decline. Likewise the bare `"e4m3"` (renamed
        // `e4m3fn`).
        //
        // The reserved `"e4m3fnuz"`/`"e5m2fnuz"` ARE recognized below. They were
        // absent before, which made them parse as UNKNOWN — the one thing
        // §6.1-0001 forbids, since it requires a reader to distinguish a
        // reserved member of the shared vocabulary from a token it has never
        // heard of. Recognizing is not accepting: they still decline at use.
        "f64" => F64,
        "i8" => I8,
        "i16" => I16,
        "u8" => U8,
        "u16" => U16,
        "u64" => U64,
        "f8e4m3fnuz" => Fp8E4M3FNUZ,
        "f8e8m0" => F8E8M0,
        "f8e6m2" => F8E6M2,
        "f8e5m2fnuz" => Fp8E5M2FNUZ,
        "i32" => I32,
        "i64" => I64,
        "u32" => U32,
        "bool" => Bool,
        "f8e4m3fn" => Fp8E4M3FN,
        "f8e5m2" => Fp8E5M2,
        "i4" => I4,
        "u4" => U4,
        "b1" => B1,
        "c64" => Complex64,
        "c128" => Complex128,
        _ => return None,
    })
}

const fn mp_code(v: MpCode) -> &'static str {
    match v {
        // `st` (not `bs`): `<mp>` codes never begin with `b` — that prefix is
        // the batch coordinate's (sk3 RFC §4.1.2/§7).
        MpCode::St => "st",
        MpCode::Rm => "rm",
    }
}

fn mp_from_code(s: &str) -> Option<MpCode> {
    Some(match s {
        "st" => MpCode::St,
        "rm" => MpCode::Rm,
        // A future additive sub-code (`rm10`/`rm7`, §6.17) is an unknown
        // spelling here until adopted — typed decline, never a silent accept.
        _ => return None,
    })
}

fn op_code(v: OpCategory) -> &'static str {
    match v {
        OpCategory::Gemm => "gem",
        OpCategory::UnaryElementwise => "une",
        OpCategory::BinaryElementwise => "bin",
        OpCategory::TernaryElementwise => "ter",
        OpCategory::GatedActivation => "gat",
        OpCategory::Reduction => "red",
        OpCategory::Scan => "scn",
        OpCategory::Normalization => "nrm",
        OpCategory::Softmax => "sft",
        OpCategory::Convolution => "cnv",
        OpCategory::Pooling => "pol",
        OpCategory::Attention => "att",
        OpCategory::Indexing => "idx",
        OpCategory::Embedding => "emb",
        OpCategory::ShapeLayout => "shp",
        OpCategory::Sorting => "srt",
        OpCategory::Quantization => "qnt",
        OpCategory::Random => "rnd",
        OpCategory::Loss => "los",
        OpCategory::SegmentOps => "seg",
        OpCategory::Image => "img",
        OpCategory::Fft => "fft",
        OpCategory::Linalg => "lin",
        OpCategory::Moe => "moe",
        // Deliberately exhaustive (no `_` arm): `OpCategory` is the defining
        // crate's own enum here, so a newly added category surfaces as a build
        // break and forces a token code rather than silently encoding as "unk".
    }
}

fn op_from_code(s: &str) -> Option<OpCategory> {
    Some(match s {
        "gem" => OpCategory::Gemm,
        "une" => OpCategory::UnaryElementwise,
        "bin" => OpCategory::BinaryElementwise,
        "ter" => OpCategory::TernaryElementwise,
        "gat" => OpCategory::GatedActivation,
        "red" => OpCategory::Reduction,
        "scn" => OpCategory::Scan,
        "nrm" => OpCategory::Normalization,
        "sft" => OpCategory::Softmax,
        "cnv" => OpCategory::Convolution,
        "pol" => OpCategory::Pooling,
        "att" => OpCategory::Attention,
        "idx" => OpCategory::Indexing,
        "emb" => OpCategory::Embedding,
        "shp" => OpCategory::ShapeLayout,
        "srt" => OpCategory::Sorting,
        "qnt" => OpCategory::Quantization,
        "rnd" => OpCategory::Random,
        "los" => OpCategory::Loss,
        "seg" => OpCategory::SegmentOps,
        "img" => OpCategory::Image,
        "fft" => OpCategory::Fft,
        "lin" => OpCategory::Linalg,
        "moe" => OpCategory::Moe,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `parse_acc_mp` guards the accumulator slot against a RESERVED dtype on its
    /// own, independently of [`StructureKey::parse_token`]'s atom scan.
    ///
    /// # Why this test is a unit test and not an integration one
    ///
    /// It cannot be reached from either public entry point. `parse_token` scans
    /// every atom for a reserved spelling *before* it looks at the tenth field,
    /// so it answers first; `from_token` discards the reason entirely. A seeded
    /// mutation collapsing this branch into the unknown-token decline therefore
    /// survived the whole integration suite — the observable behaviour is
    /// identical, because the atom scan is what produces it.
    ///
    /// The branch stays because the two guards answer different questions: the
    /// atom scan is a coarse net over the whole token, this is the positional
    /// guard for one slot. Narrow the scan to dtype positions later — a
    /// reasonable thing to want — and this becomes the live enforcement of
    /// §6.1-0001 for the accumulator. Untested defence-in-depth is just untested
    /// code that looks reassuring, so it is tested here at the only level where
    /// it is observable.
    #[test]
    fn the_accumulator_slot_declines_a_reserved_dtype_on_its_own() {
        for spelling in ["f8e4m3fnuz", "f8e5m2fnuz"] {
            assert_eq!(
                parse_acc_mp(&format!("{spelling}/rm"), ElementKind::F32),
                Err(TokenDecline::ReservedDtype {
                    spelling: spelling.to_string()
                }),
                "a reserved accumulator must be recognized-and-declined, never unknown"
            );
        }
        // Distinctness is the clause's actual requirement (§6.1-0001), so pin the
        // other verdict too: an unknown spelling must NOT report as reserved.
        assert_eq!(
            parse_acc_mp("f99/rm", ElementKind::F32),
            Err(TokenDecline::Unrecognized)
        );
        // Positive control — every assertion above is a refusal.
        assert_eq!(
            parse_acc_mp("f64/rm", ElementKind::F32),
            Ok(AccMp {
                acc: ElementKind::F64,
                mp: MpCode::Rm
            })
        );
    }

    /// spec/namespaces/cuda.md §4 — BACKS the "a `cuda:` capability-set is a single
    /// scalar, no variable-length list" claim (which makes the §6.8-0007 digest
    /// structurally unreachable) with a test rather than prose: every emitted token is a
    /// single scalar, and a list-shaped token does not parse. A future range/list token
    /// fails HERE rather than silently falsifying the annex.
    #[test]
    fn cuda_tokens_are_single_scalar() {
        for v in [ArchSku::Sm80, ArchSku::Sm89, ArchSku::Sm90a] {
            let t = arch_code(v);
            let body = t
                .strip_prefix("cuda:sm")
                .unwrap_or_else(|| panic!("token {t} must start with `cuda:sm`"));
            let digits = body.strip_suffix('a').unwrap_or(body);
            assert!(
                !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit()),
                "token {t}: sm-number must be non-empty ASCII digits, got {digits:?}"
            );
            assert!(
                !t.bytes()
                    .any(|b| matches!(b, b'+' | b',' | b'|' | b' ' | b'-')),
                "token {t} carries a list/range separator; cuda: sets are single scalars (annex §4)"
            );
        }
        // A list-shaped token must not parse — there is no multi-arch cuda: token.
        assert!(arch_from_code("cuda:sm80+sm90a").is_none());
        assert!(arch_from_code("cuda:sm80,sm90a").is_none());
    }

    fn od(shape: &[i64], strides: &[i64], dtype: ElementKind, align: u32) -> OperandDesc {
        OperandDesc::new(shape.len(), shape, strides, dtype, align)
    }

    #[test]
    fn contiguous_f32_vectorizes_to_v4() {
        // [128, 256] row-major f32, 256-byte aligned: inner extent 256 (%16),
        // f32 caps at V4 (float4 = 16 bytes).
        let a = od(&[128, 256], &[256, 1], ElementKind::F32, 256);
        let k = structure_key(OpCategory::BinaryElementwise, &[a, a, a], ArchSku::Sm89);
        assert_eq!(k.n_operands, 3);
        assert_eq!(k.operands[0].contig, Contiguity::Contig);
        assert_eq!(k.operands[0].vec_width, VecWidth::V4);
        assert_eq!(k.operands[0].inner_div, DivBucket::Div16);
        assert_eq!(k.idx, IdxWidth::Idx32);
        assert_eq!(k.work, WorkClass::GridStride);
        assert_eq!(k.rank, 2); // raw iteration rank (collapse is deferred)
        assert!(!k.operands[0].flipped);
    }

    #[test]
    fn f16_contiguous_vectorizes_to_v8() {
        let a = od(&[64, 128], &[128, 1], ElementKind::F16, 256);
        let k = structure_key(OpCategory::UnaryElementwise, &[a, a], ArchSku::Sm89);
        assert_eq!(k.operands[0].vec_width, VecWidth::V8); // f16 V8 = 16 bytes
    }

    #[test]
    fn broadcast_axis_detected() {
        // Second operand broadcasts axis 0 (stride 0 over extent 128).
        let a = od(&[128, 256], &[256, 1], ElementKind::F32, 256);
        let b = od(&[128, 256], &[0, 1], ElementKind::F32, 256);
        let k = structure_key(OpCategory::BinaryElementwise, &[a, b, a], ArchSku::Sm89);
        assert_eq!(k.operands[1].contig, Contiguity::Broadcast);
        assert!(k.operands[1].bcast.is_set(0));
        assert!(!k.operands[1].bcast.is_set(1));
    }

    #[test]
    fn negative_stride_is_flipped() {
        // Reversed innermost axis.
        let a = od(&[128, 256], &[256, -1], ElementKind::F32, 256);
        let k = structure_key(OpCategory::UnaryElementwise, &[a, a], ArchSku::Sm89);
        assert!(k.operands[0].flipped);
        assert_eq!(k.operands[0].vec_width, VecWidth::Scalar); // reversed ⇒ no vec in v1
    }

    #[test]
    fn transposed_view_is_strided() {
        // [128, 256] transposed: strides [1, 128] — inner axis stride 128 ≠ 1.
        let a = od(&[128, 256], &[1, 128], ElementKind::F32, 256);
        let k = structure_key(OpCategory::UnaryElementwise, &[a, a], ArchSku::Sm89);
        assert_eq!(k.operands[0].contig, Contiguity::Strided);
        assert_eq!(k.operands[0].vec_width, VecWidth::Scalar);
    }

    #[test]
    fn large_tensor_needs_idx64() {
        // 2^31 elements ⇒ max offset ≥ 2^31.
        let big: i64 = 1 << 16;
        let a = od(&[big, big], &[big, 1], ElementKind::F16, 256);
        let k = structure_key(OpCategory::UnaryElementwise, &[a, a], ArchSku::Sm90a);
        assert_eq!(k.idx, IdxWidth::Idx64);
    }

    #[test]
    fn token_round_trips() {
        let a = od(&[128, 256], &[256, 1], ElementKind::F32, 256);
        let b = od(&[128, 256], &[0, 1], ElementKind::F32, 256);
        let c = od(&[128, 256], &[256, -1], ElementKind::F32, 256);
        let k = structure_key(OpCategory::BinaryElementwise, &[a, b, c], ArchSku::Sm89);
        let token = k.to_token();
        let parsed = StructureKey::from_token(&token).expect("round-trip parse");
        assert_eq!(k, parsed);
        // Token is human-greppable.
        assert!(token.starts_with("sk4|bin|f32|cuda:sm89|"));
    }

    #[test]
    fn from_token_rejects_more_than_max_operands() {
        // A token carrying MORE than MAX_OPERANDS operand fields MUST be a typed
        // decline (None), never a silent truncation to the first MAX_OPERANDS —
        // otherwise two distinct operand lists collapse to the same parsed key,
        // violating the KISS-Classify closed-membership / never-silently-accept
        // discipline (an untrusted peer supplies these tokens over the wire).
        let op = "co/00/v4/d16/f";
        let too_many = std::iter::repeat_n(op, MAX_OPERANDS + 1)
            .collect::<Vec<_>>()
            .join(";");
        let token = format!("sk4|bin|f32|cuda:sm89|ix32|grid|r2|{too_many}|-");
        assert_eq!(
            StructureKey::from_token(&token),
            None,
            "an over-MAX_OPERANDS token must be rejected, not truncated"
        );
    }

    #[test]
    fn from_token_rejects_rank_above_max() {
        // The rank field is bounded by MAX_RANK; a token claiming a larger rank is
        // malformed and MUST be a typed decline, not silently accepted (downstream
        // consumers index MAX_RANK-sized arrays with it).
        let token = format!(
            "sk4|bin|f32|cuda:sm89|ix32|grid|r{}|co/00/v4/d16/f;co/00/v4/d16/f;co/00/v4/d16/f|-",
            MAX_RANK + 1
        );
        assert_eq!(
            StructureKey::from_token(&token),
            None,
            "a rank above MAX_RANK must be rejected"
        );
    }

    #[test]
    fn try_new_rejects_rank_above_max() {
        let big = [1i64; MAX_RANK + 1];
        assert_eq!(
            OperandDesc::try_new(MAX_RANK + 1, &big, &big, ElementKind::F32, 256),
            None,
            "try_new must decline rank > MAX_RANK, not panic"
        );
    }

    #[test]
    fn try_new_rejects_slices_shorter_than_rank() {
        // rank claims 3 axes but only 2 extents/strides are supplied.
        assert_eq!(
            OperandDesc::try_new(3, &[8, 8], &[8, 1], ElementKind::F32, 256),
            None,
            "try_new must decline shape/strides shorter than rank, not panic"
        );
    }

    #[test]
    fn try_new_accepts_a_valid_operand() {
        let d = OperandDesc::try_new(2, &[8, 4], &[4, 1], ElementKind::F32, 256)
            .expect("a well-formed operand must construct");
        assert_eq!(d.rank, 2);
        assert_eq!(&d.shape[..2], &[8, 4]);
        assert_eq!(&d.strides[..2], &[4, 1]);
        assert_eq!(d.dtype, ElementKind::F32);
    }

    #[test]
    fn u32_variant_is_additive_preexisting_token_byte_identical() {
        // The ElementKind::U32 addition (Model-A gather/scatter contract wiring)
        // is codec-additive: the token codec is spelling-keyed, not
        // discriminant-keyed, so adding a dtype shifts no existing dtype's code.
        // (The schema version moves for UNRELATED reasons — sk3's gem precision
        // coordinates, then sk4's respell + `(acc+mp)`; the U32 addition itself
        // required no bump.) Pin the canonical f32 token verbatim so a future
        // reshuffle that perturbs it is caught.
        assert_eq!(
            STRUCTURE_KEY_VERSION, 4,
            "codec sits at the KISS-aligned schema version 4 (sk4 schema event)"
        );
        let a = od(&[128, 256], &[256, 1], ElementKind::F32, 256);
        let k = structure_key(OpCategory::BinaryElementwise, &[a, a, a], ArchSku::Sm89);
        assert_eq!(
            k.to_token(),
            "sk4|bin|f32|cuda:sm89|ix32|grid|r2|co/00/v4/d16/f;co/00/v4/d16/f;co/00/v4/d16/f|-",
            "the canonical f32 cell serializes to the KISS-aligned sk3 token"
        );
    }

    #[test]
    fn u32_dtype_round_trips_through_the_codec() {
        // The u32 dtype codec (dtype_code / dtype_from_code) round-trips both
        // directly AND through a full token (a top-level-u32 key — operand 0's
        // dtype is the token's dtype field; the per-operand index dtype rides the
        // OP, not the key, so it is intentionally NOT a per-operand token field).
        assert_eq!(dtype_code(ElementKind::U32), "u32");
        assert_eq!(dtype_from_code("u32"), Some(ElementKind::U32));
        let a = od(&[128, 64], &[64, 1], ElementKind::U32, 256);
        let k = structure_key(OpCategory::UnaryElementwise, &[a, a], ArchSku::Sm89);
        assert_eq!(k.dtype, ElementKind::U32);
        assert_eq!(
            k.operands[0].vec_width,
            VecWidth::V4,
            "u32 is a 4-byte dtype"
        );
        let token = k.to_token();
        assert!(
            token.starts_with("sk4|une|u32|"),
            "the token spells u32: {token}"
        );
        let parsed = StructureKey::from_token(&token).expect("u32 token round-trips");
        assert_eq!(k, parsed, "a u32 key round-trips byte-identically");
    }

    #[test]
    fn reduction_reduce_axes_derived_from_keepdim_form() {
        let x = od(&[4, 8], &[8, 1], ElementKind::F32, 256);
        // Keepdim reduce of the LAST axis: [4,8] -> [4,1] sets bit 1.
        let out_last = od(&[4, 1], &[1, 1], ElementKind::F32, 256);
        let k_last = structure_key(OpCategory::Reduction, &[x, out_last], ArchSku::Sm89);
        assert_eq!(k_last.reduce_axes, AxisMask(0b10));
        // Keepdim reduce of AXIS 0: [4,8] -> [1,8] sets bit 0 — a DIFFERENT cell.
        let out_ax0 = od(&[1, 8], &[8, 1], ElementKind::F32, 256);
        let k_ax0 = structure_key(OpCategory::Reduction, &[x, out_ax0], ArchSku::Sm89);
        assert_eq!(k_ax0.reduce_axes, AxisMask(0b01));
        assert_ne!(k_last.to_token(), k_ax0.to_token()); // honest miss: axis-0 != last
        // Collapsed (rank-reduced) output is un-inferable ⇒ empty (undetermined).
        let out_collapse = od(&[4], &[1], ElementKind::F32, 256);
        let k_col = structure_key(OpCategory::Reduction, &[x, out_collapse], ArchSku::Sm89);
        assert_eq!(k_col.reduce_axes, AxisMask::EMPTY);
        // Non-reduction op stays empty even with a size-1 axis present.
        let k_ew = structure_key(
            OpCategory::BinaryElementwise,
            &[out_last, out_last, out_last],
            ArchSku::Sm89,
        );
        assert_eq!(k_ew.reduce_axes, AxisMask::EMPTY);
        // A non-empty reduce_axes round-trips through the token.
        let parsed = StructureKey::from_token(&k_ax0.to_token()).expect("round-trip");
        assert_eq!(k_ax0, parsed);
    }

    /// Field 8 admits **exactly one** spelling per reduced-axis set.
    ///
    /// This test previously asserted the opposite for the bitmask cases — that
    /// `x07` at rank 3 was accepted alongside `rall`, "semantically identical to
    /// the explicit mask Baracuda itself emits". That leniency is now a decline,
    /// per the KISS architect's ruling (KISS #160): §6.6-0009's `x<hh>` value is
    /// domain-restricted to sets that are *neither* all-axes *nor* the lone
    /// trailing axis, so `x07`-at-rank-3 is not an instance of it, and §6.7-0005
    /// requires a reader to reject any other field-8 spelling.
    ///
    /// The old assertion was not merely permissive, it was self-defeating: two
    /// accepted spellings for one set makes `from_token` → `to_token`
    /// byte-unstable, and a key with two spellings for one meaning is not an
    /// identity — which is the only reason to have a `structure_key` at all.
    #[test]
    fn field_8_admits_exactly_one_spelling_per_reduced_set() {
        let base = "sk4|bin|f32|cuda:sm89|ix32|grid|r3|\
                    co/00/v4/d16/f;co/00/v4/d16/f;co/00/v4/d16/f";

        // The two sentinels resolve against `rank`.
        let k_all = StructureKey::from_token(&format!("{base}|rall")).expect("rall accepted");
        assert_eq!(k_all.reduce_axes, AxisMask(0b111));
        let k_last = StructureKey::from_token(&format!("{base}|rlast")).expect("rlast accepted");
        assert_eq!(k_last.reduce_axes, AxisMask(0b100));

        // And their bitmask equivalents are now REFUSED, with a verdict that
        // says which of the four encodings was owed rather than "unknown token".
        for (field, set) in [("x07", "all axes"), ("x04", "the lone trailing axis")] {
            assert_eq!(
                StructureKey::parse_token(&format!("{base}|{field}")),
                Err(TokenDecline::NonCanonicalReduceField),
                "`{field}` at rank 3 spells {set}, whose canonical encoding is a sentinel"
            );
        }
        // `x00` is the same overload aimed at `-`.
        assert_eq!(
            StructureKey::parse_token(&format!("{base}|x00")),
            Err(TokenDecline::NonCanonicalReduceField)
        );

        // Positive control: a set with NO sentinel keeps the bitmask form, so
        // the assertions above are about canonicalization and not about `x<hh>`
        // having been refused wholesale.
        let k_mid = StructureKey::from_token(&format!("{base}|x03")).expect("x03 accepted");
        assert_eq!(k_mid.reduce_axes, AxisMask(0b011));
        assert_eq!(k_mid.to_token(), format!("{base}|x03"), "round-trip stable");

        // Neither sentinel has a referent in a rank-0 space: `rlast` has no
        // trailing axis, and `rall` would name the empty set that `-` owns.
        let r0 = "sk4|une|f32|cuda:sm89|ix32|grid|r0|co/00/v1/d16/f;co/00/v1/d16/f";
        assert_eq!(StructureKey::from_token(&format!("{r0}|rlast")), None);
        assert_eq!(StructureKey::from_token(&format!("{r0}|rall")), None);
        assert!(
            StructureKey::from_token(&format!("{r0}|-")).is_some(),
            "control: a rank-0 cell is spelled `-`"
        );
    }

    #[test]
    fn scalar_operand_is_contiguous() {
        let s = od(&[], &[], ElementKind::F32, 256);
        let k = structure_key(OpCategory::UnaryElementwise, &[s, s], ArchSku::Sm80);
        assert_eq!(k.rank, 0);
        assert_eq!(k.operands[0].contig, Contiguity::Contig);
        assert_eq!(k.work, WorkClass::OneWarp);
    }

    #[test]
    fn from_tensor_ref_projects_dtype_and_shape() {
        // Build a TensorRef-shaped desc through the adapter path is exercised in
        // integration tests with a real DeviceSlice; here we validate the plain
        // constructor parity the adapter relies on.
        let a = OperandDesc::new(2, &[8, 16], &[16, 1], ElementKind::Bf16, 256);
        assert_eq!(a.rank, 2);
        assert_eq!(a.dtype, ElementKind::Bf16);
        assert_eq!(a.shape[1], 16);
    }

    /// Tiny deterministic PRNG for fuzz coverage — no external dependency.
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0
        }
    }

    #[test]
    fn from_token_never_panics_on_arbitrary_input() {
        // `from_token` parses tokens from an untrusted peer (telemetry / wire): it
        // must NEVER panic (no unbounded alloc, no index OOB, no parse crash) — only
        // `Some(key)` or `None`, whatever the bytes.
        fn junk(rng: &mut Lcg, n: usize) -> String {
            (0..n)
                .map(|_| char::from((rng.next() % 94 + 33) as u8))
                .collect()
        }
        let mut rng = Lcg(0x0000_F00D);

        // (1) Pure-random junk — exercises the outer split + field-count + magic
        // guards. (A random alphabet almost never lands the 9/10 pipes with an
        // in-range rank needed to reach the field parsers — hence stage (2).)
        let alpha = b"sk012|binuecmpgemredf32f64s8u8i32i64u32sm89sm80cuda:ix\
                      gridwarpblockr;co/icstbrvda-cxtsml0123456789";
        for _ in 0..4000 {
            let len = (rng.next() % 140) as usize;
            let s: String = (0..len)
                .map(|_| {
                    let pick = (rng.next() as usize) % (alpha.len() + 4);
                    char::from(if pick < alpha.len() {
                        alpha[pick]
                    } else {
                        (rng.next() % 128) as u8
                    })
                })
                .collect();
            let _ = StructureKey::from_token(&s);
        }

        // (2) STRUCTURAL mutation of VALID tokens — the only way to fuzz the DEEP
        // parsers (`op`/`dtype`/`arch`, per-operand `parse_operand`, the 10th
        // contraction sub-parser) past the 9/10-part + `rank ≤ MAX_RANK` guards.
        // Both bases are asserted to fully round-trip first, so from_token
        // provably reaches every field parser on them; the mutations then fuzz
        // around that reachable point.
        let a = OperandDesc::new(1, &[1 << 16], &[1], ElementKind::F32, 4);
        let ew = structure_key_token(OpCategory::BinaryElementwise, &[a, a, a], ArchSku::Sm89);
        let lhs = OperandDesc::new(2, &[8, 4096], &[4096, 1], ElementKind::F32, 256);
        let rhs = OperandDesc::new(2, &[4096, 4096], &[4096, 1], ElementKind::F32, 256);
        let out = OperandDesc::new(2, &[8, 4096], &[4096, 1], ElementKind::F32, 256);
        let gemm = structure_key_token(OpCategory::Gemm, &[lhs, rhs, out], ArchSku::Sm89);
        assert!(
            StructureKey::from_token(&ew).is_some(),
            "base elementwise token must round-trip: {ew}"
        );
        assert!(
            StructureKey::from_token(&gemm).is_some(),
            "base gemm token must round-trip: {gemm}"
        );
        assert_eq!(
            gemm.split('|').count(),
            10,
            "gemm token must carry the 10th contraction field: {gemm}"
        );
        for _ in 0..8000 {
            let base = if rng.next() & 1 == 0 { &ew } else { &gemm };
            let mut parts: Vec<String> = base.split('|').map(String::from).collect();
            for _ in 0..(1 + rng.next() % 3) {
                if parts.is_empty() {
                    parts.push(junk(&mut rng, 3));
                    continue;
                }
                let i = (rng.next() as usize) % parts.len();
                match rng.next() % 8 {
                    0 => {
                        parts.remove(i);
                    }
                    1 => {
                        let f = parts[i].clone();
                        parts.insert(i, f);
                    }
                    2 => {
                        let n = (rng.next() % 12) as usize;
                        parts[i] = junk(&mut rng, n);
                    }
                    3 => {
                        let j = junk(&mut rng, 4);
                        parts[i].push_str(&j);
                    }
                    4 => parts[i] = format!("r{}", rng.next() % 100_000), // wild rank
                    5 => {
                        let l = parts[i].len();
                        if l > 0 {
                            parts[i].truncate((rng.next() as usize) % l);
                        }
                    }
                    6 => parts.insert(i, String::new()),
                    _ => {
                        let l = parts[i].len();
                        if l > 0 {
                            let k = (rng.next() as usize) % l;
                            let mut b = parts[i].clone().into_bytes();
                            b[k] = (rng.next() % 94 + 33) as u8;
                            parts[i] = String::from_utf8_lossy(&b).into_owned();
                        }
                    }
                }
            }
            let _ = StructureKey::from_token(&parts.join("|"));
        }

        // (3) Adversarial DoS-shaped + boundary inputs, plus targeted vectors that
        // reach the operand + contraction parsers with an IN-RANGE rank (r2), which
        // the over-MAX_RANK `r250` vector below never does.
        let _ = StructureKey::from_token("");
        let _ = StructureKey::from_token(&"|".repeat(10_000));
        let _ = StructureKey::from_token(&"a".repeat(200_000));
        // Over-MAX_RANK rank — declines at the rank guard (never reaches operands).
        let _ = StructureKey::from_token(&format!(
            "sk4|bin|f32|cuda:sm89|ix32|grid|r250|{}|-",
            "co/00/v4/d16/f;".repeat(50)
        ));
        // In-range rank + over-MAX_OPERANDS operand list — MUST reach parse_operand,
        // then decline gracefully.
        let _ = StructureKey::from_token(&format!(
            "sk4|bin|f32|cuda:sm89|ix32|grid|r2|{}|-",
            "co/00/v1/d16/f;".repeat(50)
        ));
        // A valid gemm base with only its 10th (contraction) field corrupted —
        // reaches the contraction sub-parser with everything else well-formed.
        let mut gp: Vec<String> = gemm.split('|').map(String::from).collect();
        gp[9] = "garbage/contraction/field".to_string();
        let _ = StructureKey::from_token(&gp.join("|"));
    }
}
