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

/// Validate an [`Access::Scan`] op at build time. The OPERAND-LAYOUT checks (contiguity, reversal, broadcast, the output)
/// are typed declines (`PlanError::InadmissibleOpInCell`), because a caller's `(OpDef, structure_key)` pair reaches them
/// (KISS-EMIT 6.8-0004); the checks on the `OpDef` itself stay author-error backstops. Mirrors
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
) -> Result<(), PlanError> {
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
            require!(
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
                require!(
                    o.contig == Contiguity::Contig,
                    "Scan streamed input {i} must be contiguous (base = row*k assumes a dense scanned axis)"
                );
                require!(
                    !o.flipped,
                    "Scan streamed input {i} must not be reversed along an axis (idx = base+j reads forward-dense; a flipped view reads mirrored/OOB — use the `reverse` scan flag, not a flipped operand)"
                );
                input0_streamed |= i == 0;
            }
            RrRole::ColBroadcast => {
                require!(
                    !o.bcast.is_set(last) && (0..last).all(|d| o.bcast.is_set(d)),
                    "Scan column input {i} must broadcast every outer axis and vary along the scanned axis (in_i[j])"
                );
                require!(!o.flipped, "Scan column input {i} must not be reversed");
            }
            RrRole::RowScalar => {
                require!(
                    rank >= 2 && (0..last).all(|d| !o.bcast.is_set(d)),
                    "Scan row-scalar input {i} needs rank >= 2 and no outer-axis broadcast (in_i[row]); an all-broadcast operand is a true scalar (bake as Const)"
                );
                require!(!o.flipped, "Scan row-scalar input {i} must not be reversed");
            }
        }
    }
    require!(
        input0_streamed,
        "Scan Input0 must be the row-streamed scanned tensor, not a column-broadcast weight or a per-row scalar"
    );
    let out = key.operands[n];
    require!(
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
    Ok(())
}
