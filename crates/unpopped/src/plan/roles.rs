// SPDX-License-Identifier: MIT OR Apache-2.0
//! Section of the plan gate; split from `plan.rs` (ROADMAP U30). Paths unchanged.

use super::*;

/// The access role of a [`Access::RowReduce`] input operand, from its layout.
///
/// The three roles are the three broadcast geometries a row-reduce operand can
/// take, and they map one-to-one onto the emitter's load index (`last` = the
/// feature/reduced axis, `rank-1`):
///
/// | role | broadcast mask | varies along | index |
/// |---|---|---|---|
/// | [`RowStreamed`](RrRole::RowStreamed) | empty | row **and** feature | `in_i[base+j]` |
/// | [`ColBroadcast`](RrRole::ColBroadcast) | every outer axis, **not** `last` | feature only | `in_i[j]` |
/// | [`RowScalar`](RrRole::RowScalar) | `last` set, **no** outer axis | row only | `in_i[row]` (hoisted) |
///
/// `RowScalar` is the exact inverse of `ColBroadcast`: a `ColBroadcast` weight is
/// constant *across rows* and varies *along the feature axis*; a `RowScalar`
/// (a saved per-row statistic — μ, rstd, lse) is constant *along the feature
/// axis* and varies *across rows*.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum RrRole {
    /// A reduced/streamed tensor ([n_out, k], full / empty bcast) — `in_i[base+j]`.
    /// Input 0 (`x`) is always this; a second streamed input (softmax-bw's `dy`
    /// beside `y`) is now also legal (the increment-2 lift).
    RowStreamed,
    /// A per-column `[k]` weight/bias, broadcast over the row axis — `in_i[j]`.
    ColBroadcast,
    /// A per-row scalar (saved stat), broadcast over the feature axis —
    /// `in_i[row]`, loaded **once** per row (hoisted outside the feature loop).
    RowScalar,
}

/// Classify a RowReduce input by its broadcast mask, given the feature axis
/// `last` (`rank-1`). **Total / non-panicking** — the emitter calls this for the
/// load index and must never crash; all *rejection* of malformed shapes lives in
/// `check_row_reduce` (one source of truth, no drift). The three-way split:
///
/// - empty bcast ⇒ [`RrRole::RowStreamed`] (the reduced/streamed tensor);
/// - `last` axis broadcast ⇒ [`RrRole::RowScalar`] (constant along the feature
///   axis ⇒ one value per row). An all-broadcast operand also lands here (last is
///   set) and is then *rejected* by validate's outer-axis-clear check — a true
///   scalar is a `Const`, not an operand;
/// - otherwise (some axis broadcast, but not `last`) ⇒ [`RrRole::ColBroadcast`]
///   (varies along the feature axis ⇒ a per-column weight/bias).
///
/// Which *specific* broadcast masks are legal for each role (every outer axis for
/// a column, no outer axis for a row-scalar, contiguity for a streamed input) is
/// validate's job, so the classification here is deliberately coarse and total.
pub fn rr_role(o: OperandKey, last: u8) -> RrRole {
    if o.bcast.is_empty() {
        RrRole::RowStreamed
    } else if o.bcast.is_set(last) {
        RrRole::RowScalar
    } else {
        RrRole::ColBroadcast
    }
}

/// Validate a [`Access::RowReduce`] op at build time (AOT — RowReduce never crosses
/// the JIT trust boundary, so a panic here is an author-error backstop, like
/// `emit_reduction`'s asserts). Catches expression errors (a `Reduced(s)` not yet
/// produced, out-of-range `Input`, a `Param`, a non-finite `Const`, a column input
/// inside a reduction stage) **and** operand-layout errors that would mis-index or
/// read out of bounds. Input 0 (`x`) must be row-streamed + contiguous. Each other
/// input takes one of three [`RrRole`]s, classified by broadcast mask ([`rr_role`])
/// and validated for the load index the emitter uses:
///
/// - **`RowStreamed`** (empty bcast, `in_i[base+j]`) — a second reduced/streamed
///   tensor (softmax-bw's `dy` beside `y`); must be contiguous. This is the
///   increment-2 lift of the former "inputs>0 must be column-broadcast" guard.
/// - **`ColBroadcast`** (every outer axis bcast, `last` not, `in_i[j]`) — a
///   per-column `[k]` weight/bias; not reversed; rank ≥ 2.
/// - **`RowScalar`** (`last` bcast, no outer axis, `in_i[row]`) — a saved per-row
///   scalar (μ, rstd, lse), the inverse of `ColBroadcast`; not reversed; rank ≥ 2.
///   An all-broadcast operand (a true scalar → `Const`) is rejected here.
///
/// The output is full-width contiguous.
///
/// v1 assumes a **uniform operand dtype** (the structure key carries one dtype) — a
/// mixed-dtype LayerNorm (fp16 `x` + fp32 weight) is unrepresentable here and must
/// be refused upstream by the caller.
///
/// **Caller pre-conditions this cannot check** (the structure key carries broadcast
/// masks but **no numeric extents** — specialize on structure, not extents), each at
/// the same trust level as the `n_out`/`k` launch args, asserted by the layer still
/// holding the `OperandDesc` extents (an AOT op author, or the live seam caller once
/// `region_to_op` wires RowReduce):
/// - a `ColBroadcast` weight's feature-axis extent must equal `x`'s `k` (else the
///   emitter reads `in_i[j]` past its buffer — a confirmed on-device OOB);
/// - a second `RowStreamed` input must be full `[n_out,k]` dense (else `in_i[base+j]`
///   over-reads — the identical trust as input 0; a bare rank-1 `[k]` has the same
///   key as a full `[n_out,k]`, so this can no longer be a key-visible rejection);
/// - a `RowScalar` must be `[n_out]`-shaped with a dense outer layout so its linear
///   offset equals `row` (else `in_i[row]` mis-indexes).
///
/// Nextafter is declared f32/f64-only at the IR level (its half lowering via
/// promote-to-f32 would step the f32 lattice — ~2^13 steps inside one half
/// step, so the demote rounds straight back: a silently wrong no-op). The
/// CUDA emitter's `cuda_binary` panic only guards the elementwise path; the
/// reduction pre-body, RowReduce stages/epilogue, and contraction epilogues
/// lower through accumulator-width helpers that never pass through it. This
/// plan-level walk covers EVERY Access arm, so no lowering path — present or
/// future backend — can bypass the honest miss.
/// Increment-1 **multi-output** admissibility gate — runs at the top of
/// [`build_plan`] (the house pattern), with an independent emitter backstop in
/// [`Backend::lower`](crate::backend::Backend::lower). A single-output op (`extra_out_bodies` empty,
/// every pre-increment-1 op) returns immediately, so nothing about the
/// established path changes and emission stays byte-identical. For a multi-output
/// op (built by [`OpDef::elementwise_multi`] or [`OpDef::elementwise_multi_hetero`])
/// the v1 rules, all honest AOT panics:
///
/// 1. **Access**: `Access::Elementwise` only. Multi-output is meaningful only for
///    an elementwise map; the reduction-class arms reject `extra_out_bodies`
///    (a fused reduction/contraction stores one accumulator, not N bodies).
/// 2. **Per-output dtype legality** (G1): every output `j` resolves a declared
///    dtype `d_j` (`out_dtype` for `j = 0`, `extra_out_dtypes[j-1]` for `j > 0`,
///    each `unwrap_or(key dtype)`). A UNIFORM output (`d_j == key dtype`) is
///    unconstrained (any elementwise body — the established path). A HETERO output
///    must be `U8` **and** have a `Cmp*`-root body (the FKC "comparison → U8 mask"
///    convention — only a predicate yields the exact `0.0/1.0` a u8 store
///    round-trips; the per-output generalization of `assert_valid_out_dtype`'s
///    Elementwise arm). The dropout fw U8 keep-mask is the v1 vehicle.
///    `extra_out_dtypes` is EMPTY (every extra output uniform) or one entry per
///    extra output. (A key-side cross-check of `d_j` vs the caller's OperandDesc
///    dtype is NOT possible — the structure key carries no per-operand dtype, only
///    the operand-0 primary; that agreement is an honest caller precondition, like
///    buffer aliasing below. G1 authored legality + the G5 emitter backstop hold
///    regardless.)
/// 3. **Operand budget**: `1 ≤ n_outputs` and `n_inputs + n_outputs ≤
///    MAX_OPERANDS`, and the key must carry exactly `n_inputs + n_outputs`
///    operands (inputs then outputs) — the caller's `OperandDesc` list.
/// 4. **Body legality** (every output body): `Input(i) < n_inputs`; NO `Reduced`
///    (there is no reduction here); NO `Coord` (Elementwise-map only in v1 — a
///    multi-output coordinate kernel is deferred, same rejection the Coord gate
///    would give); `Const` finite. `Param` f32-only is enforced by the emitter's
///    param assert over all output bodies (same rule as the single-output path).
///    (This gate rejecting `Coord` here is also what lets the downstream
///    `check_coord_admissibility` keep its `Access::Elementwise => {}` arm — a
///    multi-output `Coord` never reaches it.)
/// 5. **Output operands**: each of the last `n_outputs` operands must be
///    **non-broadcast** (a stride-0 output would alias its own writes across
///    iteration coordinates — a write race, and not the full output shape) and
///    **not flipped**. This is the key-visible slice of "outputs must not alias
///    and must match the output shape".
///
/// **Caller preconditions the key cannot see** (documented honestly, the same
/// trust level as the RowReduce `n_out`/`k` and `Coord` extent preconditions):
/// true buffer aliasing — an output buffer pointer equal to an input's (in-place)
/// — and exact per-output extent agreement are abstracted away by the structure
/// key (buffer identity and numeric extents are not keyed). The AOT op author (or
/// a future seam caller, once a multi-output region envelope exists) must ensure
/// distinct, correctly-shaped output buffers; v1 defers in-place entirely.
pub(super) fn assert_valid_multi_output(op: &OpDef, key: &StructureKey) {
    if op.extra_out_bodies.is_empty() {
        return; // single-output — the established path, unchanged.
    }
    let name = &op.name;
    assert!(
        matches!(op.access, Access::Elementwise),
        "OpDef '{name}': multi-output is Access::Elementwise-only in v1 — a fused \
         reduction/contraction stores a single accumulator, not N bodies; \
         extra_out_bodies is rejected on a {}-class op",
        access_tag(&op.access)
    );
    // Per-output dtype length invariant: `extra_out_dtypes` is EMPTY (every extra
    // output uniform — the byte-identical uniform-multi path) or carries exactly
    // one dtype per extra output.
    assert!(
        op.extra_out_dtypes.is_empty() || op.extra_out_dtypes.len() == op.extra_out_bodies.len(),
        "OpDef '{name}': extra_out_dtypes (len {}) must be empty or match \
         extra_out_bodies (len {})",
        op.extra_out_dtypes.len(),
        op.extra_out_bodies.len()
    );
    let n_inputs = op.n_inputs as usize;
    let n_outputs = op.n_outputs() as usize;
    assert!(
        n_inputs + n_outputs <= MAX_OPERANDS,
        "OpDef '{name}': n_inputs ({n_inputs}) + n_outputs ({n_outputs}) exceeds \
         MAX_OPERANDS ({MAX_OPERANDS})"
    );
    assert!(
        key.n_operands as usize == n_inputs + n_outputs,
        "OpDef '{name}': multi-output key must carry n_inputs+n_outputs operands \
         (inputs then outputs) = {}, got {} — the caller's OperandDesc list is a \
         shape mismatch",
        n_inputs + n_outputs,
        key.n_operands
    );

    // Body legality — walk every output body.
    fn check_body(e: &ScalarExpr, n_inputs: u8, name: &str) {
        match e {
            ScalarExpr::Input(i) => assert!(
                *i < n_inputs,
                "OpDef '{name}': multi-output body Input({i}) >= n_inputs {n_inputs}"
            ),
            ScalarExpr::Const(v) => assert!(
                v.is_finite(),
                "OpDef '{name}': multi-output body Const must be finite, got {v}"
            ),
            ScalarExpr::Param(_) => {}
            ScalarExpr::Reduced(s) => panic!(
                "OpDef '{name}': multi-output body must not read Reduced({s}) — there \
                 is no reduction in an Elementwise multi-output op"
            ),
            ScalarExpr::Coord(d) => panic!(
                "OpDef '{name}': multi-output body must not read Coord({d}) — v1 is \
                 elementwise-map only (a multi-output coordinate kernel is deferred)"
            ),
            ScalarExpr::CoordInt(d, _) => panic!(
                "OpDef '{name}': multi-output body must not read CoordInt({d}) — v1 is                  elementwise-map only (a multi-output coordinate kernel is deferred)"
            ),
            ScalarExpr::ConstInt(..) | ScalarExpr::Local(_) | ScalarExpr::Exec(_) => {}
            ScalarExpr::Unary(_, x) => check_body(x, n_inputs, name),
            ScalarExpr::Cast(x, _)
            | ScalarExpr::Bitcast(x, _)
            | ScalarExpr::Qualified(x, _)
            | ScalarExpr::Extract(x, _)
            | ScalarExpr::BitcastPacked(x, ..) => check_body(x, n_inputs, name),
            ScalarExpr::Pack(xs) => {
                for x in xs {
                    check_body(x, n_inputs, name);
                }
            }
            ScalarExpr::Add(a, b)
            | ScalarExpr::Sub(a, b)
            | ScalarExpr::Mul(a, b)
            | ScalarExpr::Div(a, b)
            | ScalarExpr::Binary(_, a, b) => {
                check_body(a, n_inputs, name);
                check_body(b, n_inputs, name);
            }
            ScalarExpr::Select(c, a, b) | ScalarExpr::Fma(c, a, b) => {
                check_body(c, n_inputs, name);
                check_body(a, n_inputs, name);
                check_body(b, n_inputs, name);
            }
        }
    }
    let bodies = op.output_bodies();
    for &e in &bodies {
        check_body(e, op.n_inputs, name);
    }

    // Output operands: the last n_outputs entries. Per output j we check (a) the
    // write shape (non-broadcast, non-flipped) and (b) the per-output dtype
    // legality (G1: uniform == key dtype, or a U8 keep-mask with a Cmp* root).
    //
    // The authored per-output dtype vs the caller's OperandDesc dtype (the "G2
    // wrong-bind cross-check" the brief called for) is NOT checkable here: the
    // structure key does NOT carry a per-operand dtype (`OperandKey` is
    // contig/bcast/vec_width/inner_div/flipped only; the caller's `OperandDesc`
    // dtype is consumed to compute `vec_width` and then discarded — `StructureKey`
    // keeps only the operand-0 primary `dtype`). Storing a per-operand output dtype
    // would need a `baracuda-kernels-types` schema change, which is out of scope
    // (kernels-types untouched, no `STRUCTURE_KEY_VERSION` bump). So per-output
    // dtype/caller agreement joins the honest CALLER PRECONDITIONS the key cannot
    // see (documented below with buffer aliasing and exact extents): the AOT op
    // author keys each hetero output's `OperandDesc` at the authored dtype (U8 for
    // a keep-mask). G1 (authored legality) + G5 (emitter backstop) hold regardless.
    for (j, &body) in bodies.iter().enumerate() {
        let o = key.operands[n_inputs + j];
        assert!(
            o.bcast.is_empty(),
            "OpDef '{name}': multi-output output {j} is broadcast (mask {:#04x}) — a \
             stride-0 output aliases its own writes across iteration coordinates \
             (a write race) and is not the full output shape",
            o.bcast.0
        );
        assert!(
            !o.flipped,
            "OpDef '{name}': multi-output output {j} is flipped (negative stride) — \
             a reversed output view is deferred"
        );
        // G1: resolve the authored per-output dtype (`out_dtype` for output 0,
        // `extra_out_dtypes[j-1]` else; each `unwrap_or(key dtype)`).
        let d_j = if j == 0 {
            op.out_dtype.unwrap_or(key.dtype)
        } else {
            op.extra_out_dtypes
                .get(j - 1)
                .copied()
                .flatten()
                .unwrap_or(key.dtype)
        };
        if d_j != key.dtype {
            // A HETERO output is a U8 keep-mask ONLY, and only when its body ROOT
            // is a `Cmp*` predicate (the exact 0.0/1.0 the u8 store round-trips —
            // the per-output analog of `assert_valid_out_dtype`'s Elementwise arm).
            assert!(
                d_j == ElementKind::U8,
                "OpDef '{name}': multi-output output {j} declares hetero dtype \
                 {d_j:?} — the only legal per-output hetero dtype is U8 (a \
                 comparison-predicate keep-mask); a non-U8 / wider-than-compute \
                 side-output has no exact store conversion and is out of scope"
            );
            assert!(
                matches!(body, ScalarExpr::Binary(bop, _, _) if bop.is_cmp()),
                "OpDef '{name}': multi-output output {j} is U8 but its body ROOT is \
                 not a comparison (ScalarExpr::Binary with a Cmp* op) — only a \
                 predicate yields exactly 0.0/1.0, so any other body would truncate \
                 silently under the u8 store"
            );
        }
    }
}
