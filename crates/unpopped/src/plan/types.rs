// SPDX-License-Identifier: MIT OR Apache-2.0
//! Section of the plan gate; split from `plan.rs` (ROADMAP U30). Paths unchanged.

use super::*;

/// How the kernel iterates the data — the backend-neutral schedule.
///
/// `#[non_exhaustive]`: strided / broadcast / reduction schedules are the growth
/// path. Out-of-crate backends (CUDA lives in the sibling `baracuda-cuda-emit`)
/// match with a trailing wildcard arm, so a new schedule is a minor-version
/// addition that declines at the backend's `_` arm, not a breaking change.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Schedule {
    /// Linear access, `width` elements at a time (e.g. `float4` for width 4).
    Vectorized {
        /// Vector width in elements.
        width: u32,
    },
    /// Linear access, one element at a time. Contiguous operands only.
    Scalar,
    /// Per-element coordinate unravel over the cell's iteration rank — for
    /// non-contiguous operands (strided / broadcast). The emitter specializes
    /// it per cell: the rank is unrolled, broadcast axes drop their offset
    /// terms, and a fully-broadcast operand is hoisted to a loop-invariant load.
    Strided,
    /// One thread per output element; sequential fold over the contiguous trailing
    /// axis. The v1 reduction schedule — block/warp-parallel reduction is the perf
    /// follow-up.
    Reduction {
        /// The associative combine to apply along the axis.
        op: ReduceOp,
        /// Reduce-axis geometry (design-doc predicate #9). All classes lower to
        /// the sequential fold in v2; the class reserves the dispatch token for a
        /// future block-parallel outer-axis kernel. The full axis mask rides on
        /// [`KernelPlan::key`]'s `reduce_axes` (this enum is `Copy`).
        class: ReduceAxisClass,
        /// Keep reduced axes as size-1 (broadcast-back) vs. collapse them.
        keepdim: bool,
    },
    /// Fused reduce → broadcast → elementwise, one block per output row (warp-
    /// shuffle + shared-memory tree reduce): `n_stages` reductions then a full-width
    /// epilogue. The stages + epilogue ride on [`KernelPlan::access`] (this enum is
    /// `Copy`, so a `Vec` can't live here). `block` selects the block-parallel tree
    /// (v1 always `true`) over a sequential fallback.
    RowReduce {
        /// Number of reduction stages (each produces a `Reduced(i)`).
        n_stages: u8,
        /// Block-parallel tree reduce (`true`, v1) vs the sequential fallback.
        block: bool,
    },
    /// Batched contraction (`out[m,n] = epi(Σ_k lhs[m,k]·rhs[k,n])`) — the
    /// terminal ORDER-3 schedule. v1: the skinny SIMT kernel (thread per output
    /// column, M-row register accumulators, coalesced K-streaming of the rhs) —
    /// the decode / flat-GEMM long-tail cell; tiled/MMA schedules join as
    /// bench-gated variants. Axes/accum/epilogue ride on [`KernelPlan::access`].
    Contraction,
    /// **Prefix scan** along a single axis (increment 6) — a full-width cumulative
    /// output, one row per output slot. `block = false` is the serial-fold BASE
    /// (thread 0 walks the axis sequentially — the deterministic bit-reference,
    /// [`crate::backend::VariantFidelity::BitIdentical`]); `block = true` is the
    /// cooperative block-scan VARIANT (Kogge-Stone warp scan + cross-warp carry,
    /// produced by `cuda::scan_blockscan_variant`, never by `build_plan`). The
    /// monoid/axis/flags ride here (this enum is `Copy`); `pre`/`post` ride on
    /// [`KernelPlan::access`].
    Scan {
        /// The associative monoid combine along the axis.
        op: ReduceOp,
        /// The scanned axis (v1: `rank - 1`, innermost/contiguous).
        axis: u8,
        /// Walk the axis descending.
        reverse: bool,
        /// Exclusive (shift-by-one, identity at the first visited position) scan.
        exclusive: bool,
        /// `false` = serial-fold base; `true` = cooperative block-scan variant.
        block: bool,
    },
    /// **Sliding-window reduction** along one axis (increment 7) — the POOLING
    /// family (max_pool / avg_pool / sum_pool / min_pool). One thread per output
    /// element (grid-stride) walks the local window of `size` taps, reduces with
    /// `op`, and stores the downsampled output — [`crate::backend::VariantFidelity::BitIdentical`]
    /// (each output is an independent fixed-order fold; no cross-output
    /// dependence, unlike [`Schedule::Scan`]). The window geometry rides here
    /// (this enum is `Copy`); `pre`/`post` ride on [`KernelPlan::access`].
    Window {
        /// The window combine (`Max`/`Min`/`Sum`/`Mean`).
        op: ReduceOp,
        /// The pooled axis (v1: `rank - 1`, innermost/contiguous).
        axis: u8,
        /// Window length in taps.
        size: u8,
        /// Output downsampling stride.
        stride: u8,
        /// Inter-tap dilation.
        dilation: u8,
        /// Low-side padding.
        pad_lo: u8,
        /// High-side padding.
        pad_hi: u8,
        /// Mean divisor policy: `size` (`true`) vs. valid-tap count (`false`).
        count_include_pad: bool,
    },
    /// **Row sort / argsort** along the innermost axis (increment 8). Base
    /// (`bitonic` produced by the emitter's flag, never keyed) is a per-output
    /// RANK sort — one thread per output element scans its row and computes the
    /// element's rank under the total order (O(k²), no smem, no barriers, any
    /// `k`), [`crate::backend::VariantFidelity::BitIdentical`] and stable by
    /// construction. The cooperative smem **bitonic** pair-sort is a
    /// `cuda::row_sort_bitonic_variant` VARIANT (also `BitIdentical` — a pair
    /// sort is a pure permutation), contract-bounded to `k <= 1024` via
    /// `launch_note`. The policy (`order`/`stable`/`out`/`limit`) rides here (this
    /// enum is `Copy`); `build_plan` always derives the base.
    RowSort {
        /// Ascending / descending (NaN orders greatest either way).
        order: crate::ir::SortOrder,
        /// Always `true` in v1 (`validate_row_sort` enforces the pair-sort).
        stable: bool,
        /// Which buffer(s) the sort writes: `Values` (raw-bit permutation),
        /// `Indices` (the `I32` sort permutation), or `Both` (the fused
        /// two-output kernel — values to `out_val`, `I32` indices to `out_idx`).
        /// See [`crate::ir::SortOut`].
        out: SortOut,
        /// Whether the writeback is capped to a runtime top-`k_out` (increment
        /// 10): `Full` = today's whole-row sort; `TopK` = the first `k_out` ranks
        /// under `order` (topk = `Desc`, bottomk = `Asc`), a `long long` launch
        /// arg. ORTHOGONAL to `out`. See [`crate::ir::SortLimit`].
        limit: SortLimit,
    },
    /// **2-D im2col / unfold** along the two spatial axes (increment 11) — the
    /// conv-lowering expanding gather. One thread per OUTPUT cell (grid-stride over
    /// `N*C*kh*kw*oH*oW`) unravels its linear index into `(n,c,ki,kj,oh,ow)`,
    /// computes the source coord (`in_h = oh*stride_h - pad_h + ki*dilation_h`,
    /// `in_w` symmetric), and RAW-BIT copies the in-bounds input element or stores
    /// the typed zero for an out-of-bounds tap —
    /// [`crate::backend::VariantFidelity::BitIdentical`] (each output is an
    /// independent read-or-zero + store; no fold, no cross-output dependence). The
    /// conv geometry rides here (this enum is `Copy`); `body == Input(0)` (no
    /// pre/post). Never routed to a variant.
    Im2Col {
        /// `(kh, kw)` — window taps per spatial axis.
        kernel: (u8, u8),
        /// `(stride_h, stride_w)` — output downsampling stride per axis.
        stride: (u8, u8),
        /// `(pad_h, pad_w)` — zero-padding per axis (symmetric).
        pad: (u8, u8),
        /// `(dilation_h, dilation_w)` — inter-tap dilation per axis.
        dilation: (u8, u8),
    },
}

/// Reduce-axis geometry (design-doc predicate #9). All classes lower to the same
/// sequential fold in v2; the class reserves the dispatch token so a later
/// block-parallel outer-axis kernel is an additive drop-in, not a re-key.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ReduceAxisClass {
    /// Empty mask (legacy last-axis sentinel) or a single contiguous trailing
    /// axis — today's sequential fast path.
    InnerContig,
    /// A single outermost (axis 0) reduced axis.
    Outer,
    /// A single interior reduced axis (not axis 0, not a contiguous trailing axis).
    Middle,
    /// Two or more reduced axes.
    Multi,
}

/// A language-agnostic description of the kernel to emit.
#[derive(Clone, Debug)]
pub struct KernelPlan<'a> {
    /// Op name (used to form the generated symbol).
    pub op_name: &'a str,
    /// Number of input operands.
    pub n_inputs: u8,
    /// Element dtype.
    pub dtype: ElementKind,
    /// Output element dtype — the *resolved* [`crate::ir::OpDef::out_dtype`]
    /// (`out_dtype.unwrap_or(key.dtype)`), so `out_dtype == dtype` for every
    /// uniform-dtype op and `U8` only for a validated predicate op. Backends
    /// read this for the output pointer type + store conversion.
    pub out_dtype: ElementKind,
    /// The chosen schedule.
    pub schedule: Schedule,
    /// The structure cell this plan targets. Backends read structural detail
    /// from it (rank, per-operand broadcast mask, flip) for strided lowering,
    /// and its token for traceability.
    pub key: &'a StructureKey,
    /// Output `= body`, evaluated per coordinate. For [`Schedule::RowReduce`] this
    /// is the epilogue (`OpDef::row_reduce` sets `body = epilogue`). For a
    /// multi-output op this is output 0; the further outputs are in
    /// [`Self::extra_out_bodies`].
    pub body: &'a ScalarExpr,
    /// Number of outputs the kernel writes (`1` for every single-output op —
    /// `OpDef::n_outputs`). `> 1` only for a validated multi-output
    /// `Access::Elementwise` op (increment 1); the emitter then writes the last
    /// `n_outputs` operands of the key from one shared body-DAG.
    pub n_outputs: u8,
    /// Additional output bodies (`OpDef::extra_out_bodies`) — **empty for every
    /// single-output op**, so the single-output emitters are byte-identical. The
    /// multi-output emitter interns `[body] ++ extra_out_bodies` into one
    /// [`crate::ir::ExprDag`] for cross-body CSE.
    pub extra_out_bodies: &'a [ScalarExpr],
    /// Per-EXTRA-output element dtypes ([`crate::ir::OpDef::extra_out_dtypes`], the
    /// hetero multi-output / dropout-class increment) — **empty for every
    /// single-output op and every UNIFORM multi-output op**, so the store emitters
    /// are byte-identical (`out_dtype_of(j)` resolves to the compute dtype for
    /// every output). A non-empty slice has length `extra_out_bodies.len()`; a
    /// `Some(U8)` entry is a per-output hetero keep-mask whose store converts the
    /// exact `0.0/1.0` `Cmp*` predicate to `unsigned char` at the store site.
    /// Backends read per-output dtype via [`KernelPlan::out_dtype_of`].
    pub extra_out_dtypes: &'a [Option<ElementKind>],
    /// The op's access pattern — the [`Schedule::RowReduce`] emitter reads its
    /// `stages` (and epilogue) off here, since `Schedule` is `Copy` and can't carry
    /// the stage `Vec`.
    pub access: &'a Access,
    /// Per-input layout [`crate::ir::View`]s (item 01) — index `i` ↔ `Input(i)`.
    /// **Empty for every view-free op** (every pre-item-01 constructor), so the
    /// strided emitter's per-operand offset is byte-identical; a non-empty slice
    /// has length `n_inputs`. Only a [`crate::ir::View::Permute`] entry changes
    /// emission (the stride-index remap in `cuda::offset_expr`); `Identity` /
    /// same-rank `Reshape` read at the iteration coordinate, and `Broadcast` is a
    /// key-driven validation-only declaration in v1. Validated at the top of
    /// [`build_plan`] (`assert_valid_views`) with an independent emitter backstop
    /// in [`Backend::lower`](crate::backend::Backend::lower).
    pub views: &'a [crate::ir::View],
    /// Per-input **data-dependent read roles** ([`crate::ir::ReadIndex`], increment
    /// 4, GATHER) — index `i` ↔ `Input(i)`. **Empty for every index-free op**
    /// (every pre-increment-4 constructor), so the strided emitter's per-operand
    /// offset is byte-identical; a non-empty slice has length `n_inputs`. Only a
    /// [`crate::ir::ReadIndex::Indexed`] entry changes emission (the axis
    /// value-substitution in `cuda::emit_strided`); `Direct` reads at the
    /// iteration coordinate. Validated at the top of [`build_plan`]
    /// (`assert_valid_gather`) with an independent emitter backstop in
    /// [`Backend::lower`](crate::backend::Backend::lower).
    pub read_index: &'a [crate::ir::ReadIndex],
    /// The output's **data-dependent write role** ([`crate::ir::WriteIndex`],
    /// increment 5, SCATTER) — the write-side mirror of [`Self::read_index`].
    /// [`crate::ir::WriteIndex::Direct`] for every non-scatter op (byte-identical
    /// output offset); a [`crate::ir::WriteIndex::ScatterIndexed`] role
    /// substitutes a runtime index value for one OUTPUT-axis coordinate and turns
    /// the store into a [`crate::ir::WriteCombine`] op. Validated at the top of
    /// [`build_plan`] (`assert_valid_scatter`) with an independent emitter
    /// backstop in [`Backend::lower`](crate::backend::Backend::lower).
    pub write_index: &'a crate::ir::WriteIndex,
    /// Per-input **runtime base element offsets** ([`crate::ir::BaseOffset`], the
    /// BASE_OFFSET SLICE increment) — index `i` ↔ `Input(i)`. **Empty for every
    /// offset-free op** (every pre-increment constructor), so the strided emitter's
    /// per-operand address math is byte-identical; a non-empty slice has length
    /// `n_inputs`. Only a [`crate::ir::BaseOffset::Runtime`] entry changes emission
    /// (a `long long off{i}` launch arg bumped onto the operand base at kernel
    /// entry). Presence forces [`Schedule::Strided`] (a runtime offset invalidates
    /// the keyed alignment fact the vectorized path relies on). Validated at the top
    /// of [`build_plan`] (`assert_valid_offsets`) with an independent emitter
    /// backstop in the backend.
    pub base_offsets: &'a [BaseOffset],
    /// The **single output's** runtime base element offset
    /// ([`crate::ir::BaseOffset`]) — the output-side mirror of [`Self::write_index`].
    /// [`crate::ir::BaseOffset::Zero`] for every non-offset op (byte-identical
    /// output address); [`crate::ir::BaseOffset::Runtime`] adds a `long long offo`
    /// launch arg bumped onto the output base at kernel entry.
    pub out_base_offset: BaseOffset,
}

/// Whether to compute 16-bit floats (f16, bf16) with **native** 16-bit
/// arithmetic on the plan's target, or promote them to f32.
///
/// This is a per-target **speed** decision, not a numerics one. An emitter
/// whose native 16-bit spelling is bit-identical to its f32 round-trip (as
/// baracuda-cuda-emit's packed pair path is designed to be) produces the same
/// bits either way. What changes is the clock cost: on sm_61 (the P40), NVIDIA
/// tabulates 2 fp16 results per clock per SM against fp32's 128.
///
/// Returned by [`KernelPlan::half_arith`] and [`half_arith_for`].
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum HalfArith {
    /// Native 16-bit arithmetic is allowed. This is the answer whenever the
    /// rates are unknown, because it is what every emitter did before this
    /// decision existed, so no emitted byte moves for an unsourced target.
    Native,
    /// The target's sourced fp16 rate is below its fp32 rate: compute 16-bit
    /// floats in f32 and skip any native 16-bit (packed) arithmetic.
    ViaF32,
}

/// The [`HalfArith`] decision for `target`.
///
/// Returns [`HalfArith::ViaF32`] **only** when both arithmetic rates are sourced
/// ([`crate::capability::TargetCapabilities::fp16_results_per_clk_per_sm`] and
/// its fp32 sibling) and the fp16 rate is lower. An unknown target, an unsourced
/// column, or a non-CUDA namespace returns [`HalfArith::Native`], today's
/// behaviour.
///
/// bf16 follows the same answer. NVIDIA's table has no bf16 row except
/// footnote 3 on 8.0/8.6 (128). Below sm_80, bf16 has no native arithmetic
/// for a promotion to lose.
#[must_use]
pub fn half_arith_for(target: unpopped_vocab::TargetId) -> HalfArith {
    let Some(caps) = crate::capability::capabilities_for(target) else {
        return HalfArith::Native;
    };
    match (
        caps.fp16_results_per_clk_per_sm,
        caps.fp32_results_per_clk_per_sm,
    ) {
        (Some(f16), Some(f32)) if f16 < f32 => HalfArith::ViaF32,
        _ => HalfArith::Native,
    }
}

impl KernelPlan<'_> {
    /// The per-target 16-bit float arithmetic decision for this plan's
    /// `key.target`. See [`HalfArith`] and [`half_arith_for`].
    ///
    /// A method rather than a field: `KernelPlan` is not `#[non_exhaustive]`,
    /// and emitters build it as a struct literal, so a new field would break
    /// every one of them. The target is already in `key`.
    #[must_use]
    pub fn half_arith(&self) -> HalfArith {
        half_arith_for(self.key.target)
    }

    /// All output bodies in order — `body` (output 0) then `extra_out_bodies`.
    /// One element for a single-output plan; the multi-output emitter interns
    /// these together for cross-body CSE, and the backstop walks gate every one.
    #[must_use]
    pub fn output_bodies(&self) -> Vec<&ScalarExpr> {
        std::iter::once(self.body)
            .chain(self.extra_out_bodies.iter())
            .collect()
    }

    /// The resolved element dtype of output `j` (the hetero multi-output /
    /// dropout-class increment). Output 0 is [`Self::out_dtype`] (already resolved
    /// `op.out_dtype.unwrap_or(key.dtype)`). Output `j > 0` reads
    /// [`Self::extra_out_dtypes`]`[j-1]`, resolving `None` **and the empty-slice
    /// case** (a uniform multi-output op stores an empty `extra_out_dtypes`) to the
    /// compute dtype [`Self::dtype`]. So for every uniform-multi/single-output plan
    /// `out_dtype_of(j) == dtype` and the store is byte-identical; a `Some(U8)`
    /// entry is the only hetero case. Backends use this to pick each output
    /// pointer's ctype + store conversion.
    #[must_use]
    pub fn out_dtype_of(&self, j: usize) -> ElementKind {
        if j == 0 {
            self.out_dtype
        } else {
            self.extra_out_dtypes
                .get(j - 1)
                .copied()
                .flatten()
                .unwrap_or(self.dtype)
        }
    }
}

/// Why a plan could not be built for an op/cell pair.
///
/// # Why this exists alongside the panicking gates
///
/// `build_plan`'s gates are `assert!`s, and that is right for AOT authoring: an
/// op an author wrote by hand that violates an IR rule is a program error, and
/// panicking names it loudly at the call site.
///
/// It is wrong at a **trust boundary**. `jit::synthesize` builds a plan from a
/// region another process chose, and `jit.rs` requires that the Synthesizer never
/// unwind into its caller. Historically that was held by pre-screening with a
/// second predicate (`dtype_compatible`) that mirrored these gates — two
/// implementations of one rule, which is a drift hazard by construction. The
/// mirror had already lost the 8-bit composition pin.
///
/// So the rules live once, in the gate, and are reachable in two shapes:
/// [`try_build_plan`] returns this, and [`build_plan`] panics with its `Display`.
///
/// **The two shapes are not yet equivalent** — `try_build_plan` types four of
/// the thirteen gates and then calls `build_plan`, which re-runs all thirteen as
/// asserts. See its `# Honest scope`.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum PlanError {
    /// The op's body uses a construct that has no admissible lowering at this
    /// dtype — an integer body reaching a float-only op, a `Const` at an integer
    /// dtype (spelled as an `f64` literal), a half `Nextafter`, and the rest of
    /// the §6 admissibility rules.
    InadmissibleOpAtDtype {
        /// The compute dtype the op was keyed at.
        dtype: ElementKind,
        /// The gate's own explanation, verbatim — for diagnostics, never matching.
        detail: String,
    },
    /// The op is inadmissible in this cell for a reason that is **not** the
    /// dtype: a construct used in an [`Access`] arm that does not admit it, or a
    /// term naming something the cell does not have (an axis past `key.rank`).
    ///
    /// Separate from [`Self::InadmissibleOpAtDtype`] because "this dtype cannot
    /// carry this construct" and "this cell has no such axis" are different
    /// facts with different fixes — one is answered by re-keying the dtype, the
    /// other never is. Collapsing them would leave the `detail` string as the
    /// only way to tell, and `detail` is diagnostics and must never be matched
    /// on.
    InadmissibleOpInCell {
        /// The gate's own explanation, verbatim — for diagnostics, never matching.
        detail: String,
    },
    /// A contraction cell this crate's plan cannot serve: the op's axis roles, the
    /// key's contraction facts, or both together say something v1 cannot lower.
    /// Match on the [`ContractionDecline`], never on the text.
    ///
    /// Reachable by *requesting* a cell: a key derived with declared M/N/K and
    /// batch roles (`unpopped_vocab::ContractionRoles`) can name a rank-4 cell, a
    /// rank-3 cell with no batch axis, or a batch axis stored inside the matrix
    /// axes, none of which an `OpDef` for the canonical matmul/batched matmul can
    /// pair with.
    UnservableContraction(ContractionDecline),
}

/// Why a contraction `(OpDef, structure_key)` pair is not servable
/// ([`PlanError::UnservableContraction`]). Every case is a fact about the pair,
/// checked once in [`try_build_plan`] so every emitter inherits it.
///
/// What this does **not** cover is each backend's own limits (a tile schedule, an
/// accumulator spelling, a dtype list): those are the backend's decline.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ContractionDecline {
    /// The op's role vectors are not a legal v1 contraction for the key's rank:
    /// wrong length, not exactly one free/contracted role per operand, or rank
    /// outside 2..=3 (v1 lowers one batch coordinate, so rank 4 and above, i.e. two
    /// or more batch axes, cannot be emitted). The text is for diagnostics.
    RolesInvalid {
        /// The validator's own explanation, verbatim.
        detail: String,
    },
    /// The key carries no contraction facts: it is not a `ctr` cell.
    KeyHasNoContractionFacts,
    /// The epilogue reads something other than the K-sum, constants and the fused
    /// bias inputs.
    EpilogueInadmissible,
    /// The op's accumulation policy does not fit the key: a TF32 or bf16x3 policy needs `F32` operands, an `f16`
    /// accumulator needs `F16`, and `k_step` must be at least 1 (`docs/accum-policies.md`).
    AccumPolicyMismatch {
        /// The key's operand dtype.
        dtype: ElementKind,
        /// The policy's `k_step` (zero is refused).
        k_step: u32,
    },
    /// An integer accumulation policy (`Int32`, `Int32Sat`) does not fit the cell: it needs `I8`/`U8` operands, an `I32`
    /// accumulator and output in the key, `out_dtype = Some(I32)` on the op, the identity epilogue, and (for the
    /// saturating form) a `k_step` of at least 1 (`docs/accum-policies.md`).
    IntegerPolicyMismatch {
        /// What is wrong, verbatim.
        detail: String,
    },
    /// The op has a `Batch` axis and the key declares no batch (or the reverse): the
    /// key and the op describe different cells. A rank-3 key whose caller declared
    /// no batch axis is the usual way to reach this.
    BatchDisagrees {
        /// The op's roles carry a `Batch` axis.
        op_batched: bool,
        /// The key's contraction facts carry a batch class.
        key_batched: bool,
    },
    /// A real (non-broadcast) batch axis of this input is stored inside the matrix
    /// axes. v1 binds the batch as the outermost storage stride, so no emitter can
    /// address it.
    BatchNotOutermost {
        /// The input operand, 0 = lhs, 1 = rhs.
        operand: usize,
    },
}

impl std::fmt::Display for ContractionDecline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RolesInvalid { detail } => write!(f, "contraction roles: {detail}"),
            Self::KeyHasNoContractionFacts => write!(
                f,
                "contraction cell must carry ContractionKey facts (a `ctr` key)"
            ),
            Self::EpilogueInadmissible => write!(
                f,
                "contraction: epilogue may read only Reduced(0), constants, and the \
                 fused bias Input(2..n_inputs); got a disallowed leaf"
            ),
            Self::AccumPolicyMismatch { dtype, k_step } => write!(
                f,
                "contraction: the accumulation policy needs F32 operands (TF32, 3xTF32, bf16x3) or F16 (an f16                  accumulator) and a k_step of at least 1; got operands {dtype:?}, k_step {k_step}"
            ),
            Self::IntegerPolicyMismatch { detail } => {
                write!(
                    f,
                    "contraction: the integer accumulation policy does not fit: {detail}"
                )
            }
            Self::BatchDisagrees {
                op_batched,
                key_batched,
            } => write!(
                f,
                "contraction: the op {} a Batch axis but the key {} a batch class",
                if *op_batched { "has" } else { "has no" },
                if *key_batched { "has" } else { "has no" }
            ),
            Self::BatchNotOutermost { operand } => write!(
                f,
                "contraction: input {operand}'s batch axis is stored inside the matrix axes \
                 (v1 binds batch as the outermost storage stride)"
            ),
        }
    }
}

impl std::fmt::Display for PlanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InadmissibleOpAtDtype { detail, .. } | Self::InadmissibleOpInCell { detail } => {
                write!(f, "{detail}")
            }
            Self::UnservableContraction(d) => write!(f, "{d}"),
        }
    }
}

impl std::error::Error for PlanError {}
