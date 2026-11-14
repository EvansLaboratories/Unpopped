// SPDX-License-Identifier: MIT OR Apache-2.0
//! Section of the plan gate; split from `plan.rs` (ROADMAP U30). Paths unchanged.

use super::*;

/// `assert!`, but returning the plan gate's typed refusal.
///
/// Exists so converting a gate from panicking to declining is a change of ONE
/// TOKEN per site: the condition, the message and its arguments are carried
/// through untouched. That matters more than it looks — `build_plan` panics with
/// `PlanError`'s `Display`, which is this `detail`, so every
/// `#[should_panic(expected = ...)]` written against the panicking path keeps
/// matching. An adopter has 34 of those and this crate cannot run them.
macro_rules! require {
    ($cond:expr, $($msg:tt)*) => {
        if !($cond) {
            return Err(PlanError::InadmissibleOpInCell {
                detail: format!($($msg)*),
            });
        }
    };
}
pub(crate) fn check_row_reduce(
    stages: &[ReduceStage],
    epilogue: &ScalarExpr,
    n_inputs: u8,
    key: &StructureKey,
) -> Result<(), PlanError> {
    let dtype = key.dtype;
    require!(
        matches!(
            dtype,
            ElementKind::F16
                | ElementKind::Bf16
                | ElementKind::F32
                | ElementKind::F32Strict
                | ElementKind::F64
        ),
        "RowReduce requires a float dtype, got {dtype:?}"
    );
    let n = n_inputs as usize;
    require!(
        (1..MAX_OPERANDS).contains(&n),
        "RowReduce n_inputs {n_inputs} out of [1, MAX_OPERANDS)"
    );
    require!(
        key.n_operands as usize == n + 1,
        "RowReduce expects n_inputs+1 operands (inputs then output); got {}",
        key.n_operands
    );
    let rank = key.rank as usize;
    require!(rank >= 1, "RowReduce needs a last (reduced) axis");
    let last = (rank - 1) as u8;

    // Operand roles + layout legality (the OOB / mis-index guards). Parallel index
    // over key.operands, so a range loop is the natural form. `is_col` feeds the
    // epilogue-only check below (reducing a per-column weight is rejected); a
    // RowScalar is legal in BOTH a stage `pre` and the epilogue (it is constant
    // along the reduced axis — layer-norm-bw's x_hat reads μ/rstd inside a fold),
    // so it is deliberately NOT tracked as a column.
    let mut is_col = [false; MAX_OPERANDS];
    let mut input0_streamed = false;
    #[allow(
        clippy::needless_range_loop,
        reason = "`i` is reported in the error and compared with 0 as well as indexing `key.operands`"
    )]
    for i in 0..n {
        let o = key.operands[i];
        match rr_role(o, last) {
            RrRole::RowStreamed => {
                require!(
                    o.contig == Contiguity::Contig,
                    "RowReduce row-streamed input {i} must be contiguous (base = row*k assumes a dense last axis)"
                );
                // `Contiguity::Contig` is |stride|-based (classify_contiguity uses
                // strides[d].abs()), so a dense-but-REVERSED view passes the contig
                // check while carrying flipped=true — and the emitter walks memory
                // FORWARD (idx = row*k + j), reading the tensor mirrored / off the
                // end. `flipped` is a key-visible axis (the 'r'/'f' token field), so
                // reject it here as the ColBroadcast/RowScalar branches do. (Review
                // #2: the pre-lift inputs>0-must-be-column guard flip-checked every
                // extra input; lifting it to allow a 2nd row-streamed input newly
                // exposed this path.)
                require!(
                    !o.flipped,
                    "RowReduce row-streamed input {i} must not be reversed along an axis (base = row*k reads forward-dense; a flipped view would read mirrored/OOB)"
                );
                input0_streamed |= i == 0;
            }
            RrRole::ColBroadcast => {
                require!(
                    !o.bcast.is_set(last),
                    "RowReduce input {i}: the feature (last) axis is broadcast (mask {:#04x}) — a column weight/bias must vary along it; bake a true scalar as Const",
                    o.bcast.0
                );
                // Must broadcast EVERY outer axis (a per-column [k] vector), else
                // in_i[j] silently drops an outer-axis dependence.
                require!(
                    (0..last).all(|d| o.bcast.is_set(d)),
                    "RowReduce column input {i} must broadcast every outer (row) axis — a per-column [k] weight/bias"
                );
                require!(
                    !o.flipped,
                    "RowReduce column input {i} must not be reversed along the feature axis"
                );
                is_col[i] = true;
            }
            RrRole::RowScalar => {
                // Per-row scalar (a saved stat: μ, rstd, lse). The feature (last)
                // axis is broadcast (guaranteed by rr_role); NO outer (row) axis may
                // be — else it is either all-broadcast (a true scalar → bake as
                // Const) or drops a row dependence. Indexed `in_i[row]`, so it needs
                // an outer axis (rank >= 2) laid out dense (offset == row), the
                // latter a caller precondition at the same trust level as `x`'s
                // base = row*k (see the module note).
                require!(
                    (0..last).all(|d| !o.bcast.is_set(d)),
                    "RowReduce row-scalar input {i}: an outer (row) axis is broadcast (mask {:#04x}) — a per-row scalar varies across rows and is constant only along the feature axis; an all-broadcast operand is a true scalar (bake as Const)",
                    o.bcast.0
                );
                require!(
                    !o.flipped,
                    "RowReduce row-scalar input {i} must not be reversed"
                );
                require!(
                    rank >= 2,
                    "RowReduce row-scalar input {i} needs rank >= 2 (an outer row axis to index by `row`)"
                );
            }
        }
    }
    require!(
        input0_streamed,
        "RowReduce Input0 (x) must be the row-streamed reduced tensor, not a column-broadcast weight or a per-row scalar"
    );
    // Inputs 1.. may now be a second **row-streamed** tensor (softmax-bw's `dy`
    // beside `y`), a per-column weight/bias, OR a per-row scalar (layer-norm-bw's
    // μ/rstd). The former "inputs>0 must be column-broadcast" guard is LIFTED
    // (increment 2): a second row-streamed input full [n_out,k] is the point. A
    // bare rank-1 [k] passed as input>0 has an empty bcast and so is now accepted
    // as row-streamed — its full extent [n_out,k] is a caller precondition (the key
    // cannot see n_out/k), the identical trust level as input 0. See the module note.
    if n > 1 {
        require!(
            rank >= 2,
            "RowReduce with a multi-operand epilogue needs rank >= 2"
        );
    }
    let out = key.operands[n];
    require!(
        out.bcast.is_empty() && out.contig == Contiguity::Contig,
        "RowReduce output must be full-width contiguous (empty bcast)"
    );

    // Expression legality. `max_reduced` = stages already produced (stage `i` may
    // read `Reduced(0..i)`; the epilogue may read all). `in_stage` forbids a column
    // input inside a reduction `pre` (reducing a per-column operand is nonsense).
    fn check(
        e: &ScalarExpr,
        n_inputs: u8,
        max_reduced: u8,
        in_stage: bool,
        is_col: &[bool],
    ) -> Result<(), PlanError> {
        match e {
            ScalarExpr::Input(i) => {
                require!(*i < n_inputs, "RowReduce Input({i}) >= n_inputs {n_inputs}");
                if in_stage {
                    require!(
                        !is_col[*i as usize],
                        "RowReduce column input {i} used inside a reduction stage.pre — column weight/bias are epilogue-only"
                    );
                }
                Ok(())
            }
            ScalarExpr::Reduced(s) => {
                require!(
                    *s < max_reduced,
                    "RowReduce Reduced({s}) references a stage not yet produced (have {max_reduced})"
                );
                Ok(())
            }
            ScalarExpr::Param(i) => Err(PlanError::InadmissibleOpInCell {
                detail: format!("RowReduce v1 forbids Param({i}) — bake scalars (eps) as Const"),
            }),
            ScalarExpr::Coord(d) => Err(PlanError::InadmissibleOpInCell {
                detail: format!(
                    "RowReduce forbids Coord({d}) — the RowReduce stages/epilogue iterate                      the (row, j) space, not an elementwise output coordinate space;                      Coord is Elementwise-only in 0d"
                ),
            }),
            ScalarExpr::CoordInt(..) | ScalarExpr::ConstInt(..) | ScalarExpr::Local(_) | ScalarExpr::Exec(_) => {
                Err(PlanError::InadmissibleOpInCell {
                    detail: "RowReduce forbids integer-typed leaves (CoordInt/ConstInt) — they                              are Elementwise-only in v1"
                        .to_string(),
                })
            }
            ScalarExpr::Const(v) => {
                require!(v.is_finite(), "RowReduce Const must be finite, got {v}");
                Ok(())
            }
            ScalarExpr::Unary(_, x) => check(x, n_inputs, max_reduced, in_stage, is_col),
            ScalarExpr::Cast(x, _)
            | ScalarExpr::Bitcast(x, _)
            | ScalarExpr::Qualified(x, _)
            | ScalarExpr::Extract(x, _) | ScalarExpr::BitcastPacked(x, ..) => {
                check(x, n_inputs, max_reduced, in_stage, is_col)
            }
            ScalarExpr::Pack(xs) => xs.iter().try_for_each(|x| check(x, n_inputs, max_reduced, in_stage, is_col)),
            ScalarExpr::Add(a, b)
            | ScalarExpr::Sub(a, b)
            | ScalarExpr::Mul(a, b)
            | ScalarExpr::Div(a, b)
            | ScalarExpr::Binary(_, a, b) => {
                check(a, n_inputs, max_reduced, in_stage, is_col)?;
                check(b, n_inputs, max_reduced, in_stage, is_col)
            }
            ScalarExpr::Select(c, a, b) | ScalarExpr::Fma(c, a, b) => {
                check(c, n_inputs, max_reduced, in_stage, is_col)?;
                check(a, n_inputs, max_reduced, in_stage, is_col)?;
                check(b, n_inputs, max_reduced, in_stage, is_col)
            }
        }
    }
    for (i, st) in stages.iter().enumerate() {
        check(&st.pre, n_inputs, i as u8, true, &is_col)?;
    }
    check(epilogue, n_inputs, stages.len() as u8, false, &is_col)
}

/// Validate an [`Access::Scan`] op at build time (AOT — a scan never crosses the
/// JIT trust boundary, so a panic here is an author-error backstop). Mirrors
/// [`check_row_reduce`]'s operand-role + layout checks, with three DELIBERATE
/// differences:
///
/// - **ADMITS `Prod`** — a scan explicitly wants cumprod, and the
///   serial base folds it directly (RowReduce admits a `Prod` stage too since 0.42.6; only its launched lift declines). Only the block-scan VARIANT emitter (§4)
///   declines integer Sum/Prod; the serial base is `BitIdentical` for every
///   admitted dtype (integer Sum/Prod wraps exactly; Max/Min is exactly associative).
/// - **NO float-only gate** — integer `Sum`/`Prod`/`Max`/`Min` on the serial base
///   are legal and bit-exact, so (unlike RowReduce) no `float dtype` assert.
/// - **`axis < rank`** — any axis; a non-innermost axis (since 0.42.11) needs every
///   input full-width dense, the oracle walking it with the axis stride.
///
/// Admits `Mean` at a float dtype only (since 0.42.10, an Unpopped extension; see below). `exclusive` and `reverse` are independently legal
/// and composable — there is no illegal combination, so nothing extra is asserted
/// for them (the on-device validator covers all four cells).
pub(super) fn validate_scan(
    op: &OpDef,
    key: &StructureKey,
    axis: u8,
    reverse: bool,
    exclusive: bool,
) {
    let Access::Scan {
        op: scan_op,
        pre,
        post,
        ..
    } = &op.access
    else {
        unreachable!("validate_scan on a non-Scan op");
    };
    let name = &op.name;
    // Both flags are independently legal and composable — asserted-consumed so a
    // future reader sees they were considered (no illegal (exclusive, reverse) cell).
    let _ = (reverse, exclusive);

    // Mean (a running mean) is not a monoid and not a KISS scan (KISS-OPS-6.11-0003 takes sum/prod/max/min; the question
    // is ThinkersJournal/KISS#766). Unpopped admits it as an extension, float only: an integer running mean rounds.
    // Prod is admitted too (a scan folds cumprod serially).
    if matches!(scan_op, ReduceOp::Mean) {
        assert!(
            matches!(
                key.dtype,
                ElementKind::F16
                    | ElementKind::Bf16
                    | ElementKind::F32
                    | ElementKind::F32Strict
                    | ElementKind::F64
            ),
            "OpDef '{name}': Scan Mean (a running mean) requires a float dtype, got {:?} - an integer mean has              rounding semantics (miss honestly)",
            key.dtype
        );
    }

    let n = op.n_inputs as usize;
    assert!(
        (1..MAX_OPERANDS).contains(&n),
        "Scan n_inputs {} out of [1, MAX_OPERANDS)",
        op.n_inputs
    );
    assert!(
        key.n_operands as usize == n + 1,
        "Scan expects n_inputs+1 operands (inputs then output); got {}",
        key.n_operands
    );
    let rank = key.rank as usize;
    assert!(rank >= 1, "Scan needs a scanned axis (rank >= 1)");
    let last = (rank - 1) as u8;

    // The scanned axis is any axis of the rank (since 0.42.11). The innermost axis keeps the row-iteration skeleton
    // (rows of a dense inner dimension, with column-broadcast and per-row operands); another axis is strided: the
    // oracle walks `(outer, axis, inner)` with the axis stride `inner`, and every input must then be full-width
    // dense (no broadcast), so there is no per-column or per-row operand to index differently.
    assert!(
        axis <= last,
        "Scan axis {axis} is outside the rank {rank} (axes 0..={last})"
    );
    if axis != last {
        for i in 0..n {
            let o = key.operands[i];
            assert!(
                o.bcast.is_empty() && o.contig == Contiguity::Contig && !o.flipped,
                "Scan over a non-innermost axis needs input {i} full-width dense (no broadcast, contiguous, not reversed)"
            );
        }
    }

    // Operand roles + layout legality (mirrors check_row_reduce). Input 0 is the
    // row-streamed scanned tensor: `base = row*k` + the forward `idx = base+j` walk
    // assume a dense, forward last axis, so it must be Contig and NOT flipped (a
    // reversed operand keys |stride|-Contig + flipped and would read mirrored/OOB —
    // the reverse SCAN is the `reverse` flag, never a flipped operand).
    let mut input0_streamed = false;
    #[allow(
        clippy::needless_range_loop,
        reason = "`i` is reported in the error and compared with 0 as well as indexing `key.operands`"
    )]
    for i in 0..n {
        let o = key.operands[i];
        match rr_role(o, last) {
            RrRole::RowStreamed => {
                assert!(
                    o.contig == Contiguity::Contig,
                    "Scan streamed input {i} must be contiguous (base = row*k assumes a dense scanned axis)"
                );
                assert!(
                    !o.flipped,
                    "Scan streamed input {i} must not be reversed along an axis (idx = base+j reads forward-dense; a flipped view reads mirrored/OOB — use the `reverse` scan flag, not a flipped operand)"
                );
                input0_streamed |= i == 0;
            }
            RrRole::ColBroadcast => {
                assert!(
                    !o.bcast.is_set(last) && (0..last).all(|d| o.bcast.is_set(d)),
                    "Scan column input {i} must broadcast every outer axis and vary along the scanned axis (in_i[j])"
                );
                assert!(!o.flipped, "Scan column input {i} must not be reversed");
            }
            RrRole::RowScalar => {
                assert!(
                    rank >= 2 && (0..last).all(|d| !o.bcast.is_set(d)),
                    "Scan row-scalar input {i} needs rank >= 2 and no outer-axis broadcast (in_i[row]); an all-broadcast operand is a true scalar (bake as Const)"
                );
                assert!(!o.flipped, "Scan row-scalar input {i} must not be reversed");
            }
        }
    }
    assert!(
        input0_streamed,
        "Scan Input0 must be the row-streamed scanned tensor, not a column-broadcast weight or a per-row scalar"
    );
    let out = key.operands[n];
    assert!(
        out.bcast.is_empty() && out.contig == Contiguity::Contig && !out.flipped,
        "Scan output must be full-width forward-dense contiguous (empty bcast, not flipped)"
    );

    // Expression legality. `pre` (the per-element pre-map) runs BEFORE the fold, so
    // it must NOT read the running prefix (`Reduced` is rejected in `pre`); `post`
    // (the per-element epilogue) reads the running prefix as the single `Reduced(0)`
    // leaf. Coord is rejected upstream by `check_coord_admissibility` (non-
    // elementwise); Param is f32-only (emitter). Input indices must be in range.
    fn check(e: &ScalarExpr, n_inputs: u8, allow_reduced: bool, ctx: &str, name: &str) {
        match e {
            ScalarExpr::Input(i) => assert!(
                *i < n_inputs,
                "Scan '{name}' {ctx} Input({i}) >= n_inputs {n_inputs}"
            ),
            ScalarExpr::Reduced(s) => {
                assert!(
                    allow_reduced,
                    "Scan '{name}' {ctx} must not read Reduced({s}) — the running prefix does not exist in the pre-map (it reads inputs only)"
                );
                assert!(
                    *s == 0,
                    "Scan '{name}' {ctx} Reduced({s}) — the running prefix is the single Reduced(0) leaf"
                );
            }
            ScalarExpr::Const(v) => assert!(
                v.is_finite(),
                "Scan '{name}' {ctx} Const must be finite, got {v}"
            ),
            ScalarExpr::Param(_)
            | ScalarExpr::Coord(_)
            | ScalarExpr::CoordInt(..)
            | ScalarExpr::ConstInt(..)
            | ScalarExpr::Local(_)
            | ScalarExpr::Exec(_) => {}
            ScalarExpr::Unary(_, x) => check(x, n_inputs, allow_reduced, ctx, name),
            ScalarExpr::Cast(x, _)
            | ScalarExpr::Bitcast(x, _)
            | ScalarExpr::Qualified(x, _)
            | ScalarExpr::Extract(x, _)
            | ScalarExpr::BitcastPacked(x, ..) => check(x, n_inputs, allow_reduced, ctx, name),
            ScalarExpr::Pack(xs) => {
                for x in xs {
                    check(x, n_inputs, allow_reduced, ctx, name);
                }
            }
            ScalarExpr::Add(a, b)
            | ScalarExpr::Sub(a, b)
            | ScalarExpr::Mul(a, b)
            | ScalarExpr::Div(a, b)
            | ScalarExpr::Binary(_, a, b) => {
                check(a, n_inputs, allow_reduced, ctx, name);
                check(b, n_inputs, allow_reduced, ctx, name);
            }
            ScalarExpr::Select(c, a, b) | ScalarExpr::Fma(c, a, b) => {
                check(c, n_inputs, allow_reduced, ctx, name);
                check(a, n_inputs, allow_reduced, ctx, name);
                check(b, n_inputs, allow_reduced, ctx, name);
            }
        }
    }
    check(pre, op.n_inputs, false, "pre-map", name);
    check(post, op.n_inputs, true, "post-epilogue", name);
}

/// Validate an [`Access::Window`] op at build time (increment 7; AOT — a window
/// never crosses the JIT trust boundary, so a panic here is an author-error
/// backstop). Mirrors [`validate_scan`]'s operand-role + layout checks, with the
/// pooling-specific window-parameter gate:
///
/// - **`Prod`** — a windowed product is not in the pool family (KISS has `avg_pool` and
///   `max_pool` only); since 0.42.9 it is admitted as an Unpopped extension, the product of the
///   in-bounds taps from the identity `1`. `Max`/`Min`/`Sum`/`Mean` are admitted.
/// - **`Mean` requires a float dtype** — an integer average has rounding
///   semantics (i32 `sum/count` truncates); avg_pool is float-only. `Max`/`Min`/
///   `Sum` are legal on the integer base too (max/min-pool select; sum-pool
///   wraps, matching the reduction Sum contract).
/// - **`axis < rank`** — any axis; a non-innermost axis (since 0.42.12) needs every
///   input full-width dense, the oracle walking it with the axis stride.
/// - **`size`/`stride`/`dilation >= 1`** — a zero window / stride / dilation is a
///   degenerate (empty-window) config.
/// - **`2*pad_lo <= span` and `2*pad_hi <= span`** where the tap footprint is
///   `span = dilation*(size-1) + 1` — each edge window must overlap the input by
///   at least 1 tap (the bespoke `pool1d` `pad*2 <= window` constraint,
///   generalized to dilation). A pad exceeding half the window would place an
///   entire edge window in padding.
///
/// The `in_len → out_len` window arithmetic (`out_len = floor((in_len + pad_lo +
/// pad_hi - dilation*(size-1) - 1)/stride) + 1`) is a **runtime-launch-arg caller
/// precondition**, NOT a plan-time check: [`StructureKey`] deliberately abstracts
/// numeric extents away (it carries per-operand contiguity/broadcast/flip, never
/// shapes), so the plan gate cannot see `in_len`/`out_len` — the same trust level
/// as RowReduce's `k`/`n_out` and `Coord`'s exact-integer extent bound. The output
/// operand's LAYOUT (forward-dense contiguous, downsampled extent) IS keyed and is
/// checked here.
#[allow(
    clippy::too_many_arguments,
    reason = "one argument per declared window parameter"
)]
pub(super) fn validate_window(
    op: &OpDef,
    key: &StructureKey,
    axis: u8,
    size: u8,
    stride: u8,
    dilation: u8,
    pad_lo: u8,
    pad_hi: u8,
) {
    let Access::Window {
        op: wop, pre, post, ..
    } = &op.access
    else {
        unreachable!("validate_window on a non-Window op");
    };
    let name = &op.name;

    // Mean (avg_pool) is float-only: an integer average rounds (i32 sum/count
    // truncates). Max/Min/Sum ride the integer base too.
    if matches!(wop, ReduceOp::Mean) {
        assert!(
            matches!(
                key.dtype,
                ElementKind::F16
                    | ElementKind::Bf16
                    | ElementKind::F32
                    | ElementKind::F32Strict
                    | ElementKind::F64
            ),
            "OpDef '{name}': Window Mean (avg_pool) requires a float dtype, got \
             {:?} — an integer average has rounding semantics (miss honestly; \
             integer max/min/sum-pool are legal)",
            key.dtype
        );
    }

    let n = op.n_inputs as usize;
    assert!(
        (1..MAX_OPERANDS).contains(&n),
        "Window n_inputs {} out of [1, MAX_OPERANDS)",
        op.n_inputs
    );
    assert!(
        key.n_operands as usize == n + 1,
        "Window expects n_inputs+1 operands (inputs then output); got {}",
        key.n_operands
    );
    let rank = key.rank as usize;
    assert!(rank >= 1, "Window needs a pooled axis (rank >= 1)");
    let last = (rank - 1) as u8;

    // Any axis of the rank (since 0.42.12). The innermost axis keeps the row-iteration skeleton (with column-broadcast
    // and per-row operands); another axis is strided, and every input must then be full-width dense.
    assert!(
        axis <= last,
        "Window axis {axis} is outside the rank {rank} (axes 0..={last})"
    );
    if axis != last {
        for i in 0..n {
            let o = key.operands[i];
            assert!(
                o.bcast.is_empty() && o.contig == Contiguity::Contig && !o.flipped,
                "Window over a non-innermost axis needs input {i} full-width dense (no broadcast, contiguous, not reversed)"
            );
        }
    }

    // Window-parameter legality — a degenerate (empty-window) config is a reject.
    assert!(
        size >= 1,
        "Window size must be >= 1 (an empty window has no taps)"
    );
    assert!(stride >= 1, "Window stride must be >= 1");
    assert!(dilation >= 1, "Window dilation must be >= 1");
    // span = the tap footprint (dilation*(size-1)+1); each edge window must overlap
    // the input by >= 1 tap, i.e. 2*pad <= span (bespoke `pool1d` pad*2 <= window,
    // generalized to dilation). u32 arithmetic avoids u8 overflow for large params.
    let span = u32::from(dilation) * (u32::from(size) - 1) + 1;
    assert!(
        2 * u32::from(pad_lo) <= span,
        "Window pad_lo {pad_lo} exceeds half the window span {span} \
         (dilation*(size-1)+1) — an entire low-edge window would fall in padding; \
         2*pad_lo <= span (bespoke pool1d pad*2 <= window)"
    );
    assert!(
        2 * u32::from(pad_hi) <= span,
        "Window pad_hi {pad_hi} exceeds half the window span {span} \
         (dilation*(size-1)+1) — an entire high-edge window would fall in padding; \
         2*pad_hi <= span (bespoke pool1d pad*2 <= window)"
    );

    // Operand roles + layout legality (mirrors validate_scan). Input 0 is the
    // row-streamed pooled tensor: `base = row*k_in` + the tap walk `idx = base+p`
    // assume a dense, forward inner axis, so it must be Contig and NOT flipped.
    let mut input0_streamed = false;
    #[allow(
        clippy::needless_range_loop,
        reason = "`i` is reported in the error and compared with 0 as well as indexing `key.operands`"
    )]
    for i in 0..n {
        let o = key.operands[i];
        match rr_role(o, last) {
            RrRole::RowStreamed => {
                assert!(
                    o.contig == Contiguity::Contig,
                    "Window streamed input {i} must be contiguous (base = row*k_in assumes a dense pooled axis)"
                );
                assert!(
                    !o.flipped,
                    "Window streamed input {i} must not be reversed along an axis (idx = base+p reads forward-dense; a flipped view reads mirrored/OOB)"
                );
                input0_streamed |= i == 0;
            }
            RrRole::ColBroadcast => {
                assert!(
                    !o.bcast.is_set(last) && (0..last).all(|d| o.bcast.is_set(d)),
                    "Window column input {i} must broadcast every outer axis and vary along the pooled axis (in_i[p])"
                );
                assert!(!o.flipped, "Window column input {i} must not be reversed");
            }
            RrRole::RowScalar => {
                assert!(
                    rank >= 2 && (0..last).all(|d| !o.bcast.is_set(d)),
                    "Window row-scalar input {i} needs rank >= 2 and no outer-axis broadcast (in_i[row]); an all-broadcast operand is a true scalar (bake as Const)"
                );
                assert!(
                    !o.flipped,
                    "Window row-scalar input {i} must not be reversed"
                );
            }
        }
    }
    assert!(
        input0_streamed,
        "Window Input0 must be the row-streamed pooled tensor, not a column-broadcast weight or a per-row scalar"
    );
    // The output is full-width forward-dense contiguous (a DOWNSAMPLED extent — the
    // caller sizes it via the window formula, a runtime precondition — but the same
    // layout class as the input's inner axis).
    // Not `out.bcast`: the key's mask is over the iteration frame, so a size-1 own axis
    // (a global pool, a top-1) beside a wider input reads as broadcast. `contig` is
    // `Broadcast` for a genuine stride-0 axis of extent > 1, so it alone rejects those.
    let out = key.operands[n];
    assert!(
        out.contig == Contiguity::Contig && !out.flipped,
        "Window output must be full-width forward-dense contiguous (empty bcast, not flipped)"
    );

    // Expression legality (mirrors validate_scan): `pre` (per-tap pre-map) runs
    // BEFORE the fold, so it must NOT read the window result (`Reduced` rejected in
    // `pre`); `post` (per-output epilogue) reads the result as the single
    // `Reduced(0)` leaf. Coord is rejected upstream (non-elementwise); Param is
    // f32-only (emitter). Input indices must be in range.
    fn check(e: &ScalarExpr, n_inputs: u8, allow_reduced: bool, ctx: &str, name: &str) {
        match e {
            ScalarExpr::Input(i) => assert!(
                *i < n_inputs,
                "Window '{name}' {ctx} Input({i}) >= n_inputs {n_inputs}"
            ),
            ScalarExpr::Reduced(s) => {
                assert!(
                    allow_reduced,
                    "Window '{name}' {ctx} must not read Reduced({s}) — the window result does not exist in the pre-map (it reads taps only)"
                );
                assert!(
                    *s == 0,
                    "Window '{name}' {ctx} Reduced({s}) — the window result is the single Reduced(0) leaf"
                );
            }
            ScalarExpr::Const(v) => assert!(
                v.is_finite(),
                "Window '{name}' {ctx} Const must be finite, got {v}"
            ),
            ScalarExpr::Param(_)
            | ScalarExpr::Coord(_)
            | ScalarExpr::CoordInt(..)
            | ScalarExpr::ConstInt(..)
            | ScalarExpr::Local(_)
            | ScalarExpr::Exec(_) => {}
            ScalarExpr::Unary(_, x) => check(x, n_inputs, allow_reduced, ctx, name),
            ScalarExpr::Cast(x, _)
            | ScalarExpr::Bitcast(x, _)
            | ScalarExpr::Qualified(x, _)
            | ScalarExpr::Extract(x, _)
            | ScalarExpr::BitcastPacked(x, ..) => check(x, n_inputs, allow_reduced, ctx, name),
            ScalarExpr::Pack(xs) => {
                for x in xs {
                    check(x, n_inputs, allow_reduced, ctx, name);
                }
            }
            ScalarExpr::Add(a, b)
            | ScalarExpr::Sub(a, b)
            | ScalarExpr::Mul(a, b)
            | ScalarExpr::Div(a, b)
            | ScalarExpr::Binary(_, a, b) => {
                check(a, n_inputs, allow_reduced, ctx, name);
                check(b, n_inputs, allow_reduced, ctx, name);
            }
            ScalarExpr::Select(c, a, b) | ScalarExpr::Fma(c, a, b) => {
                check(c, n_inputs, allow_reduced, ctx, name);
                check(a, n_inputs, allow_reduced, ctx, name);
                check(b, n_inputs, allow_reduced, ctx, name);
            }
        }
    }
    check(pre, op.n_inputs, false, "pre-map", name);
    check(post, op.n_inputs, true, "post-epilogue", name);
}

/// Validate an [`Access::Im2Col`] op at build time (increment 11; AOT — an im2col
/// never crosses the JIT trust boundary, so a panic here is an author-error
/// backstop). Unlike the rank-agnostic-innermost emitters (Scan/Window/RowSort
/// pin `axis == rank-1`), im2col has a FIXED rank-4 -> rank-3 conv layout, so this
/// gate PINS the operand ranks + layout the way `validate_window` pins the pooled
/// axis:
///
/// - **single input** (`n_inputs == 1`, no weight operand — a weighted window is
///   the deferred windowed-contraction), `n_operands == 2` (input then output).
/// - **input rank-4 forward-dense NCHW** — `key.rank == 4` (the input is also the
///   widest operand, so the shared `key.rank` is the input rank); input0 `Contig`,
///   `!flipped`, empty broadcast. The emitter's `(((n*C+c)*H_in+in_h)*W_in+in_w)`
///   address math assumes a dense row-major NCHW input.
/// - **output rank-3 forward-dense** — `Contig`, `!flipped`, empty broadcast. The
///   EXPANDED extent `[N, C*kh*kw, oH*oW]` is a runtime caller precondition (only
///   LAYOUT is keyed — the layout-only gate admits the expansion).
/// - **geometry legality** — `kh,kw >= 1`, `stride.* >= 1`, `dilation.* >= 1`
///   (`pad.*` is a `u8`, always `>= 0`). A zero kernel/stride/dilation is a
///   degenerate config.
/// - **`out_dtype == None`** (a pure gather preserves dtype;
///   `assert_valid_out_dtype` double-gates this) and **`body == Input(0)`** (a pure
///   raw-bit copy — no pre/post in v1).
///
/// The `(H_in,W_in) -> (oH,oW)` conv arithmetic
/// (`oH = (H_in + 2*pad_h - dilation_h*(kh-1) - 1)/stride_h + 1`, `oW` symmetric)
/// is a **runtime-launch-arg caller precondition**, NOT a plan-time check:
/// [`StructureKey`] deliberately abstracts numeric extents away (it carries
/// per-operand contiguity/broadcast/flip, never shapes), so the plan gate cannot
/// see `H_in`/`oH` — the same trust level as Window's `k_in -> k_out` and RowReduce's
/// `k`/`n_out`. It is on-device-validated via `initcheck` (full write + in-bounds
/// source reads); see the increment-11 brief §3/§6.
pub(super) fn validate_im2col(
    op: &OpDef,
    key: &StructureKey,
    kernel: (u8, u8),
    stride: (u8, u8),
    pad: (u8, u8),
    dilation: (u8, u8),
) {
    let Access::Im2Col { .. } = &op.access else {
        unreachable!("validate_im2col on a non-Im2Col op");
    };
    let name = &op.name;
    let _ = pad; // pad.* is u8 (>= 0); baked into the caller's oH/oW, checked on-device.

    // G1 — single input (no weight operand), input + output operands only.
    assert!(
        op.n_inputs == 1,
        "Im2Col '{name}': v1 takes a SINGLE input (n_inputs {} != 1) — a weighted \
         window is the deferred windowed-contraction, not an im2col",
        op.n_inputs
    );
    assert!(
        key.n_operands == 2,
        "Im2Col '{name}': expects n_inputs+1 = 2 operands (input then output); got {}",
        key.n_operands
    );

    // G2 — input rank-4 forward-dense NCHW. The shared `key.rank` is the widest
    // operand rank; the rank-4 NCHW input is the widest (output is rank-3), so
    // `key.rank == 4` pins the input rank. A rank-3 (or other) input drives the max
    // below 4 and rejects here (the miss is honest, not silently wrong).
    assert!(
        key.rank == 4,
        "Im2Col '{name}': v1 pins a rank-4 NCHW input [N,C,H_in,W_in] (key.rank {} \
         != 4) — 1-D/3-D im2col are deferred rank variants",
        key.rank
    );
    let in0 = key.operands[0];
    assert!(
        in0.contig == Contiguity::Contig,
        "Im2Col '{name}': input 0 must be dense contiguous NCHW (the \
         (((n*C+c)*H_in+in_h)*W_in+in_w) address math assumes row-major)"
    );
    assert!(
        !in0.flipped,
        "Im2Col '{name}': input 0 must not be reversed along an axis (a flipped view \
         reads mirrored/OOB source coords)"
    );

    // G3 — output rank-3 forward-dense. Only LAYOUT is keyed (the expanded extent
    // [N, C*kh*kw, oH*oW] is a runtime precondition), so the gate admits the
    // expansion so long as the output is forward-dense contiguous.
    //
    // The key's `bcast` mask is NOT consulted. It is over the iteration frame
    // (KISS-CLASSIFY §6.5-0014 as amended by KISS#519, §6.6-0008), so it also records
    // "absent from this frame axis" (the rank-3 output never spans frame axis 0) and
    // "own axis of extent 1 beside a wider frame" (N = 1 beside C > 1, or a 1x1 output
    // beside a wider input). Neither is a broadcast, and a mask test rejected both.
    // A genuine own-axis broadcast (extent > 1, stride 0) makes `contig` read
    // `Broadcast`, and a strided output reads `Strided`/`InnerContig`, so the
    // `contig == Contig` clause alone rejects every output that is not dense.
    let out = key.operands[1];
    assert!(
        out.contig == Contiguity::Contig && !out.flipped,
        "Im2Col '{name}': output must be forward-dense contiguous (empty broadcast, \
         not flipped) — the [N, C*kh*kw, oH*oW] column matrix is written densely"
    );

    // G4 — geometry legality (a zero kernel/stride/dilation is degenerate).
    assert!(
        kernel.0 >= 1 && kernel.1 >= 1,
        "Im2Col '{name}': kernel (kh,kw) = ({},{}) must be >= 1 on each axis",
        kernel.0,
        kernel.1
    );
    assert!(
        stride.0 >= 1 && stride.1 >= 1,
        "Im2Col '{name}': stride (sh,sw) = ({},{}) must be >= 1 on each axis",
        stride.0,
        stride.1
    );
    assert!(
        dilation.0 >= 1 && dilation.1 >= 1,
        "Im2Col '{name}': dilation (dh,dw) = ({},{}) must be >= 1 on each axis",
        dilation.0,
        dilation.1
    );

    // G5 — dtype-preserving pure gather: out_dtype None (double-gated in
    // assert_valid_out_dtype) and body pinned to a raw Input(0) (no pre/post — a
    // composed body would need arithmetic the raw-bit copy does not emit).
    assert!(
        op.out_dtype.is_none(),
        "Im2Col '{name}': a pure gather preserves dtype — out_dtype must be None, got \
         {:?}",
        op.out_dtype
    );
    assert!(
        matches!(op.body, ScalarExpr::Input(0)),
        "Im2Col '{name}': body must be exactly Input(0) (a raw-bit copy) — v1 has no \
         pre/post map"
    );
}

/// Validate an [`Access::RowSort`] op at build time (increment 8; AOT — a sort
/// never crosses the JIT trust boundary, so a panic here is an author-error
/// backstop). Mirrors [`validate_scan`]/[`validate_window`]'s operand-role +
/// layout checks, with the sort-specific gates:
///
/// - **`stable == true` only** — v1 ALWAYS emits the stable pair-sort (`(key,
///   original-index)` with index tie-break), so stability is free; an unstable
///   network would emit byte-identical code under a different symbol (dead
///   keying) and admitting-then-ignoring the flag would be dishonest keying.
///   `stable: false` rejects.
/// - **dtype** — admit `F32|F32Strict|F64|F16|Bf16|I32|I64`; reject `U32`
///   (index-only, no value dtype) and `S8|U8` (v1 de-scope; the bespoke argsort
///   covers small ints; liftable by widening this gate + the validator cells).
/// - **out_dtype ↔ [`SortOut`] state coupling** — `Indices ⇒ out_dtype ==
///   Some(I32)`; `Values | Both ⇒ out_dtype == None` (Both's I32 out_idx is
///   emitter-hardwired, off the symbol, not out_dtype). Double-gated (also
///   `assert_valid_out_dtype`).
/// - **`body` must be exactly `Input(0)`** — v1 has no pre/post; this single
///   equality replaces the recursive `check` machinery of validate_scan/window
///   (it rejects Param/Coord/Reduced/Const/composed bodies in one stroke).
/// - **`extra_out_bodies` empty** — multi-output does not ride RowSort (also
///   caught by `assert_valid_multi_output`; double-gated). `Both` carries its
///   second buffer via the `SortOut` state + the 3-operand key, NEVER a body.
/// - **operand count** — `Values | Indices` carry `n_inputs + 1` operands
///   (`[in0, out]`); `Both` carries `n_inputs + 2` (`[in0, out_val, out_idx]`).
/// - **Input 0 layout** — `RowStreamed` + `Contig` + `!flipped` (the emitters
///   read `in0[base+j]` stride-free; a flipped/strided input reads mirrored/OOB).
/// - **Output layout** — every output (out / out_val AND out_idx for `Both`) is
///   empty bcast + `Contig` + `!flipped`. This gate is LAYOUT-only (it does NOT
///   assert the output WIDTH), so a `TopK` cell's narrower `[batch, k_out]` output
///   passes as written — no relaxation needed.
///
/// There is no axis field: RowSort is innermost-axis by definition (matching
/// Scan/Window's `axis == rank-1` posture). The bitonic variant's `k <= 1024`
/// bound is NOT checkable here (the structure key carries no numeric extents) —
/// it is a `launch_note` precondition + on-device-validated contract, the same
/// trust level as smemrow/blockscan; the base rank sort has no length limit.
///
/// - **`limit` (increment 10 TOPK/BOTTOMK)** — `Full` is today's whole-row sort;
///   `TopK` caps the writeback to the first `k_out` ranks. `k_out` is the OUT
///   operand's inner extent, a `long long` launch arg (the Window `(n_out, k_in,
///   k_out)` precedent) — the structure key carries NO numeric extent, so the
///   `k_out <= k_in` precondition is NOT expressible as a plan assert. It is a
///   runtime launch precondition, on-device-validated by `initcheck` (proves all
///   `k_out` slots written AND no over-write past `k_out`), the same trust tier as
///   the bitonic `k <= 1024`. `Indices`/`Both` under `TopK` inherit argsort's
///   `k_in <= 2^31-1` cap (the `I32` index).
pub(super) fn validate_row_sort(
    op: &OpDef,
    key: &StructureKey,
    _order: SortOrder,
    stable: bool,
    out: SortOut,
    _limit: SortLimit,
) {
    let name = &op.name;

    // v1 emits only the stable pair-sort. Reject stable=false before anything else
    // so the message is unambiguous.
    assert!(
        stable,
        "OpDef '{name}': row_sort v1 emits only the stable pair-sort (stable=true); \
         unstable declined (an unstable network would emit byte-identical code under \
         a different symbol — dead keying)"
    );

    // dtype gate: the v1 comparable value set. U32 is index-only (no value dtype);
    // S8/U8 are a v1 de-scope (bespoke argsort covers small ints).
    assert!(
        matches!(
            key.dtype,
            ElementKind::F32
                | ElementKind::F32Strict
                | ElementKind::F64
                | ElementKind::F16
                | ElementKind::Bf16
                | ElementKind::I32
                | ElementKind::I64
        ),
        "OpDef '{name}': row_sort dtype {:?} is out of the v1 set \
         (F32/F32Strict/F64/F16/Bf16/I32/I64) — U32 is index-only and S8/U8 are a \
         v1 de-scope (miss honestly)",
        key.dtype
    );

    // out_dtype ↔ SortOut state coupling (double-gated with assert_valid_out_dtype).
    // G1: Indices ⇒ Some(I32); Values | Both ⇒ None (Both's I32 out_idx is
    // emitter-hardwired off the symbol, not carried by out_dtype).
    match out {
        SortOut::Indices => assert!(
            op.out_dtype == Some(ElementKind::I32),
            "OpDef '{name}': a row argsort (SortOut::Indices) produces an I32 index \
             output — out_dtype must be Some(I32), got {:?}",
            op.out_dtype
        ),
        SortOut::Values | SortOut::Both => assert!(
            op.out_dtype.is_none(),
            "OpDef '{name}': a values-sort (Values) / fused two-output (Both) is \
             dtype-preserving on output 0 — out_dtype must be None (Both's I32 \
             out_idx is emitter-hardwired), got {:?}",
            op.out_dtype
        ),
    }

    // body must be exactly Input(0): v1 has no pre/post, so a single equality check
    // rejects Param/Coord/Reduced/Const/composed bodies in one stroke.
    assert!(
        matches!(op.body, ScalarExpr::Input(0)),
        "OpDef '{name}': row_sort body must be exactly Input(0) (v1 has no pre/post \
         map); got {:?}",
        op.body
    );

    // Multi-output does not ride RowSort (also caught by assert_valid_multi_output).
    assert!(
        op.extra_out_bodies.is_empty(),
        "OpDef '{name}': row_sort is single-output (a permutation is not a ScalarExpr \
         body) — extra_out_bodies must be empty"
    );

    // Operand count + rank. G2: Values/Indices write one output ([in0, out]);
    // Both writes two ([in0, out_val, out_idx]).
    let n = op.n_inputs as usize;
    assert!(
        n == 1,
        "OpDef '{name}': row_sort streams exactly one row operand (n_inputs=1), got {n}"
    );
    let n_outputs = if matches!(out, SortOut::Both) { 2 } else { 1 };
    assert!(
        key.n_operands as usize == n + n_outputs,
        "row_sort expects n_inputs+{n_outputs} operands (input then {n_outputs} \
         output buffer(s)); got {}",
        key.n_operands
    );
    let rank = key.rank as usize;
    assert!(rank >= 1, "row_sort needs a sorted axis (rank >= 1)");
    let last = (rank - 1) as u8;

    // Input 0 layout: the row-streamed sorted tensor. The emitters read
    // `in0[base+j]` stride-free, so it must be Contig and NOT flipped (a
    // flipped/strided operand reads mirrored/OOB).
    let in0 = key.operands[0];
    assert!(
        rr_role(in0, last) == RrRole::RowStreamed,
        "OpDef '{name}': row_sort Input0 must be the row-streamed sorted tensor, not \
         a column-broadcast or per-row-scalar operand"
    );
    assert!(
        in0.contig == Contiguity::Contig,
        "OpDef '{name}': row_sort input must be contiguous (base = row*k assumes a \
         dense sorted axis)"
    );
    assert!(
        !in0.flipped,
        "OpDef '{name}': row_sort input must not be reversed along an axis (idx = \
         base+j reads forward-dense; a flipped view reads mirrored/OOB — use the \
         Desc order, not a flipped operand)"
    );

    // Output layout: full-width forward-dense contiguous. G3: output 0 (out for
    // Values/Indices, out_val for Both) always; and for Both, operand 2 (out_idx)
    // too — it is written forward-dense out_idx[base + r/p].
    // Not `out.bcast`: the key's mask is over the iteration frame, so a size-1 own axis
    // (a global pool, a top-1) beside a wider input reads as broadcast. `contig` is
    // `Broadcast` for a genuine stride-0 axis of extent > 1, so it alone rejects those.
    let out0 = key.operands[n];
    assert!(
        out0.contig == Contiguity::Contig && !out0.flipped,
        "OpDef '{name}': row_sort output must be forward-dense contiguous (empty \
         bcast, not flipped); the width may be < the input for a TopK cap"
    );
    if matches!(out, SortOut::Both) {
        let out_idx = key.operands[n + 1];
        assert!(
            out_idx.contig == Contiguity::Contig && !out_idx.flipped,
            "OpDef '{name}': row_sort Both out_idx (operand 2) must be forward-dense \
             contiguous (empty bcast, not flipped); width may be < input for a TopK cap"
        );
    }
}

/// Vector width in elements for a [`VecWidth`] bucket.
pub(super) fn vec_width_elems(v: VecWidth) -> u32 {
    match v {
        VecWidth::V8 => 8,
        VecWidth::V4 => 4,
        VecWidth::V2 => 2,
        VecWidth::Scalar => 1,
    }
}
