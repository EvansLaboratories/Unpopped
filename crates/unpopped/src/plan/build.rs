// SPDX-License-Identifier: MIT OR Apache-2.0
//! Section of the plan gate; split from `plan.rs` (ROADMAP U30). Paths unchanged.

use super::*;

/// [`build_plan`], but returns the gate's refusal instead of panicking.
///
/// # Errors
///
/// [`PlanError`] when an admissibility gate rejects the op/dtype pair.
///
/// # Honest scope
///
/// (0.20.1: the contraction cell's gates are typed too, as [`ContractionDecline`]; the count
/// below is the 0.6.0 reading.)
///
/// This types **four** of the thirteen gates: the op/dtype admissibility ones a
/// caller can trip by requesting a cell, and the ones the JIT's old pre-screen
/// mirrored. It then calls [`build_plan`], which **re-runs all thirteen as
/// asserts** — so a rejection outside those four still panics, from inside a
/// function whose name says otherwise.
///
/// The intended line is that the remaining gates are *structural* invariants
/// about how an `OpDef` was constructed (operand arity, view validity,
/// multi-output shape), which panic on purpose: the JIT builds its `OpDef`
/// itself via `region_to_op`, which already validates the region and returns
/// `Result`, so reaching one means a bug in construction rather than an
/// unserveable request.
///
/// # Two gates are on the wrong side of that line, measured
///
/// `tests/adversarial_input_panic_census.rs` feeds 1280 `(OpDef,
/// structure_key)` pairs through the emitter. Seven of the nine untyped gates
/// never fire — the classification holds for them. **Two do:**
///
/// - `check_coord_admissibility` — `coord(1)` at `i32`, or a coordinate whose
///   axis exceeds the cell rank. A caller trips this by *requesting a cell*, not
///   by mis-constructing an op.
/// - the RowReduce validator — likewise reachable from an ordinary request.
///
/// Both are op/dtype admissibility wearing a structural gate's clothes, and both
/// are KISS-EMIT §6.8-0004 violations when reached through [`crate::try_generate`].
/// They gain `check_*` twins in 0.6.0; until then this scope note is the honest
/// description rather than the intended one.
pub fn try_build_plan<'a>(
    op: &'a OpDef,
    key: &'a StructureKey,
) -> Result<KernelPlan<'a>, PlanError> {
    try_build_plan_with(op, key, false)
}

/// [`try_build_plan`] for a backend that declares [`Backend::total_integer_ops`](crate::backend::Backend::total_integer_ops): integer `Div`
/// and an integer-typed constant are admitted at an integer compute dtype, because that backend
/// spells them by the total rule (`docs/int-div-shift-semantics.md`). A backend that does not
/// override it gets exactly [`try_build_plan`]'s refusals. [`crate::try_generate`] calls this.
///
/// # Errors
/// The one fact the plan gate reads of a backend: whether it declares the total integer rule. Every
/// [`Backend`](crate::backend::Backend) implements it (`backend.rs`), so a `&dyn Backend` and a concrete backend both
/// satisfy [`try_build_plan_for`]; `plan` names no backend type (ROADMAP U27).
pub trait IntegerOpsGate {
    /// The backend's `total_integer_ops`.
    fn declares_total_integer_ops(&self) -> bool;
}

/// As [`try_build_plan`].
pub fn try_build_plan_for<'a, B: IntegerOpsGate + ?Sized>(
    op: &'a OpDef,
    key: &'a StructureKey,
    backend: &B,
) -> Result<KernelPlan<'a>, PlanError> {
    try_build_plan_with(op, key, backend.declares_total_integer_ops())
}

/// [`build_plan`] for a backend that declares [`Backend::total_integer_ops`](crate::backend::Backend::total_integer_ops); see
/// [`try_build_plan_for`].
///
/// # Panics
/// Where [`build_plan`] does.
#[must_use]
pub fn build_plan_for<'a, B: IntegerOpsGate + ?Sized>(
    op: &'a OpDef,
    key: &'a StructureKey,
    backend: &B,
) -> KernelPlan<'a> {
    try_build_plan_for(op, key, backend).unwrap_or_else(|e| panic!("{e}"))
}

pub(super) fn try_build_plan_with<'a>(
    op: &'a OpDef,
    key: &'a StructureKey,
    total_ints: bool,
) -> Result<KernelPlan<'a>, PlanError> {
    check_dtype_is_a_compute_dtype(key.dtype)?;
    check_no_half_nextafter(op, key.dtype)?;
    check_int_op_admissibility_for(op, key.dtype, total_ints)?;
    check_complex_op_admissibility(op, key.dtype)?;
    check_bool_op_admissibility(op, key.dtype)?;
    check_coord_admissibility(op, key)?;
    check_fma_admissibility(op, key.dtype)?;
    check_typed_admissibility_for(op, key.dtype, total_ints)?;
    check_qualified_admissibility(op, key.dtype)?;
    if let Access::RowReduce {
        ref stages,
        ref epilogue,
    } = op.access
    {
        check_row_reduce(stages, epilogue, op.n_inputs, key)?;
    }
    check_contraction(op, key)?;
    Ok(build_plan_core(op, key))
}

/// Choose the schedule for `op` at structure cell `key` and return a neutral
/// [`KernelPlan`].
///
/// Elementwise ops vectorize when every operand is `Contig` + `V4`, scalar/strided
/// otherwise. A reduction op maps straight to [`Schedule::Reduction`] (the fold is
/// the schedule). (Whether a backend can lower the chosen dtype is the backend's
/// call, not this function's.)
#[must_use]
pub fn build_plan<'a>(op: &'a OpDef, key: &'a StructureKey) -> KernelPlan<'a> {
    try_build_plan(op, key).unwrap_or_else(|e| panic!("{e}"))
}

/// [`build_plan`] minus the admissibility gates [`try_build_plan`] already ran.
///
/// The gates left here are **structural invariants about how an `OpDef` was
/// constructed** — arity, view validity, multi-output shape — which a caller
/// cannot trip by requesting a cell. They panic on purpose: reaching one means a
/// bug in construction, not an unserveable request, and an internal invariant
/// that returns `Err` teaches its caller to handle a case that cannot happen.
///
/// Measured, not assumed: `tests/adversarial_input_panic_census.rs` fires none
/// of them across 1280 `(OpDef, structure_key)` pairs.
pub(super) fn build_plan_core<'a>(op: &'a OpDef, key: &'a StructureKey) -> KernelPlan<'a> {
    assert_valid_out_dtype(op);
    assert_valid_multi_output(op, key);
    assert_valid_reduction_post(op);
    assert_valid_views(op, key);
    assert_valid_gather(op, key);
    assert_valid_scatter(op, key);
    assert_valid_offsets(op, key);
    let schedule = match op.access {
        Access::Reduction {
            op: rop,
            axes,
            keepdim,
            post: _,
            accum: _,
        } => {
            // `class`/`keepdim` are consumed by the emitter in step 3; today all
            // classes lower to the same sequential fold, so the legacy last-axis
            // path (empty mask ⇒ `InnerContig`) stays byte-identical.
            let input0_contig = key.n_operands > 0 && key.operands[0].contig == Contiguity::Contig;
            Schedule::Reduction {
                op: rop,
                class: classify_reduce_axes(axes, key.rank, input0_contig),
                keepdim,
            }
        }
        // `ref` borrows (the Vec/expr can't move out of the borrowed `op.access`);
        // v1 always routes RowReduce to the block-parallel tree reduce.
        Access::RowReduce {
            ref stages,
            epilogue: _,
        } => {
            // Validation moved to `check_row_reduce`, hoisted into
            // `try_build_plan` (0.6.0): it is op/dtype admissibility a caller
            // trips by REQUESTING a cell, so it owes a typed decline rather
            // than a panic. Left here it could only abort — this arm has no
            // error channel.
            Schedule::RowReduce {
                n_stages: stages.len() as u8,
                block: true,
            }
        }
        // v1 admissibility (role-validated, keyed with contraction facts, an epilogue
        // over the K-sum and the fused bias only, a batch the key and the op agree on and
        // that is stored outermost) is `check_contraction`, which `try_build_plan` ran:
        // the pair is a request, so its refusal is typed and lives once.
        Access::Contraction { .. } => Schedule::Contraction,
        // Increment 6 SCAN: validate admissibility (mirrors the RowReduce arm's
        // `check_row_reduce` call), then derive the serial-fold BASE schedule
        // (`block: false`). The cooperative block-scan is produced separately by
        // `cuda::scan_blockscan_variant` (a `lower_variants` filter), never here.
        Access::Scan {
            op: sop,
            axis,
            reverse,
            exclusive,
            ..
        } => {
            validate_scan(op, key, axis, reverse, exclusive);
            Schedule::Scan {
                op: sop,
                axis,
                reverse,
                exclusive,
                block: false,
            }
        }
        // Increment 7 WINDOW: validate admissibility (mirrors the Scan arm's
        // `validate_scan` call), then derive the one-thread-per-output pooling
        // schedule. The window geometry is Copy, so it rides the schedule; `pre`/
        // `post` ride on `KernelPlan::access` (Scan/RowReduce precedent).
        Access::Window {
            op: wop,
            axis,
            size,
            stride,
            dilation,
            pad_lo,
            pad_hi,
            count_include_pad,
            ..
        } => {
            validate_window(op, key, axis, size, stride, dilation, pad_lo, pad_hi);
            Schedule::Window {
                op: wop,
                axis,
                size,
                stride,
                dilation,
                pad_lo,
                pad_hi,
                count_include_pad,
            }
        }
        // Increment 8 SORT_PERM: validate admissibility (mirrors the Scan/Window
        // arms), then derive the per-output RANK-sort BASE schedule. The
        // cooperative smem bitonic pair-sort is produced separately by
        // `cuda::row_sort_bitonic_variant` (a `lower_variants` filter), never
        // here. All payload fields are Copy — no `ref` borrow.
        Access::RowSort {
            order,
            stable,
            out,
            limit,
        } => {
            validate_row_sort(op, key, order, stable, out, limit);
            Schedule::RowSort {
                order,
                stable,
                out,
                limit,
            }
        }
        // Increment 11 IM2COL: validate admissibility (mirrors the Scan/Window/Sort
        // arms), then thread the conv geometry into the one-thread-per-output-cell
        // expanding-gather schedule. All geometry fields are Copy — no `ref` borrow;
        // `body == Input(0)` (no pre/post), and im2col has no vectorize/variant gate.
        Access::Im2Col {
            kernel,
            stride,
            pad,
            dilation,
        } => {
            validate_im2col(op, key, kernel, stride, pad, dilation);
            Schedule::Im2Col {
                kernel,
                stride,
                pad,
                dilation,
            }
        }
        Access::Elementwise => {
            let n = key.n_operands as usize;
            let all_contig = n > 0 && (0..n).all(|k| key.operands[k].contig == Contiguity::Contig);
            // The kernel vectorizes at the *narrowest* width every operand supports.
            let min_width = (0..n)
                .map(|k| vec_width_elems(key.operands[k].vec_width))
                .min()
                .unwrap_or(1);
            // Every output stores its compute-dtype value raw (no U8/hetero store):
            // output 0 uniform (`out_dtype` None) AND every extra output uniform
            // (`extra_out_dtypes` all None / empty). A hetero output has no packed
            // vector store, so it must fall to Scalar/Strided (G4). This is the
            // per-output generalization of the single-output 0b `out_dtype.is_none()`
            // vectorize gate — byte-identical for every uniform op.
            let all_outputs_uniform =
                op.out_dtype.is_none() && op.extra_out_dtypes.iter().all(Option::is_none);
            if op_has_gather(op) || op_has_scatter(op) {
                // Increment 4/5: a GATHERED input (read) or a SCATTERED output
                // (write) resolves a DATA-DEPENDENT address (one axis coordinate
                // is a runtime index value), which only the strided emitter folds
                // (it substitutes `idx·stride[axis]` for `c{axis}·stride[axis]`, on
                // the input side for gather / the output side for scatter). NEVER
                // vectorized/packed/scalar — a data-dependent address cannot
                // coalesce into a vector load/store, and the vector/packed emitters
                // iterate a bare linear index that would ignore the index operand
                // entirely. Pinned by the force-strided tests + the independent
                // `assert_gather_lowerable` / `assert_scatter_lowerable` backstops.
                // Index-free / write-Direct ops never reach here, so emission stays
                // byte-identical.
                Schedule::Strided
            } else if op_has_offset(op) {
                // BASE_OFFSET SLICE (G4): a Runtime base offset shifts the effective
                // base pointer at kernel entry, so the keyed `align_bytes → VecWidth`
                // fact the float4/ld.128 vectorized path relies on becomes a LIE —
                // even for a width-multiple offset VALUE (the gate keys on PRESENCE, a
                // compile-time mask, never the runtime value). Force Strided: only its
                // single per-pointer entry-bump lowering is proven, and `emit_vectorized`
                // indexes width-element vectors that would silently ignore the bump.
                // This cannot be patched in the emitter alone. An offset-free op never
                // reaches here (op_has_offset false), so emission stays byte-identical.
                // Pinned by the force-strided positive test + the independent
                // `assert_offsets_lowerable` backstop.
                Schedule::Strided
            } else if op_has_addressing_view(op) {
                // Item 01: a viewed INPUT (a `Permute`/transpose or a `Broadcast`
                // read-through) reads the producer through a layout change, which
                // only the strided emitter folds into address math (`offset_expr`
                // remaps `c{d}·stride[perm[d]]`). NEVER vectorized/packed/scalar —
                // a transposed read is non-contiguous, and the vector/packed
                // emitters iterate a bare linear index that would ignore the view
                // (silently reading the un-transposed operand). `Identity` and a
                // same-rank `Reshape` (identity linear map) are NOT addressing
                // views, so a view-free or all-identity op is unaffected here —
                // byte-identical. Pinned by the vectorize-never view test in
                // `cuda` and the independent `assert_views_lowerable` backstop.
                Schedule::Strided
            } else if expr_contains_coord(&op.body) {
                // A Coord body always takes the STRIDED schedule (increment
                // 0d): the strided emitter is the only one that materializes
                // the per-axis output coordinates `c{d}` a Coord leaf reads —
                // the Vectorized/Scalar emitters iterate a bare linear index.
                // Contiguous cells are still CORRECT under strided emission
                // (the unravel + stride dot-product reproduces the linear
                // offset exactly), just unoptimized — a coordinate-aware
                // vectorized variant is a follow-up. Pinned by the
                // vectorize-never test in `cuda`.
                Schedule::Strided
            } else if !all_contig {
                Schedule::Strided
            } else if min_width >= 2 && all_outputs_uniform {
                // A hetero-output (u8 predicate) kernel takes the SCALAR path
                // in v1 — never Vectorized: the vector/packed emitters load and
                // STORE one vector type, and a u8-mask output has no packed
                // store (a contiguous u8 output even keys V8, which would
                // otherwise widen `min_width` past the inputs'). `all_outputs_uniform`
                // (G4) covers BOTH the single-output predicate (`out_dtype` Some)
                // AND a hetero MULTI-output (any `extra_out_dtypes` Some, e.g. the
                // dropout U8 keep-mask): any U8 output forces Scalar/Strided so
                // `emit_vectorized_multi` never receives a hetero plan. Pinned by
                // the packed-fallback golden + `dropout_hetero_never_vectorizes`.
                Schedule::Vectorized { width: min_width }
            } else {
                Schedule::Scalar
            }
        }
    };
    KernelPlan {
        op_name: &op.name,
        n_inputs: op.n_inputs,
        dtype: key.dtype,
        out_dtype: op.out_dtype.unwrap_or(key.dtype),
        schedule,
        key,
        body: &op.body,
        n_outputs: op.n_outputs(),
        extra_out_bodies: &op.extra_out_bodies,
        extra_out_dtypes: &op.extra_out_dtypes,
        access: &op.access,
        views: &op.views,
        read_index: &op.read_index,
        write_index: &op.write_index,
        base_offsets: &op.base_offsets,
        out_base_offset: op.out_base_offset,
    }
}

/// Classify a reduction's axis geometry (design-doc predicate #9) from its
/// reduced-axis mask + the input's contiguity. An empty mask is the legacy
/// last-axis sentinel; a single trailing axis over a contiguous input is the
/// existing fast path. All classes lower to the same sequential fold in v2 — the
/// class only reserves the dispatch token for a future block-parallel kernel.
pub(super) fn classify_reduce_axes(
    axes: AxisMask,
    rank: u8,
    input0_contig: bool,
) -> ReduceAxisClass {
    match axes.0.count_ones() {
        0 => ReduceAxisClass::InnerContig, // empty ⇒ legacy last-axis default
        1 => {
            let d = axes.0.trailing_zeros() as u8;
            if rank > 0 && d == rank - 1 && input0_contig {
                ReduceAxisClass::InnerContig // contiguous trailing axis = fast path
            } else if d == 0 {
                ReduceAxisClass::Outer
            } else {
                ReduceAxisClass::Middle
            }
        }
        _ => ReduceAxisClass::Multi,
    }
}
