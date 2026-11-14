// SPDX-License-Identifier: MIT OR Apache-2.0
//! Section of the plan gate; split from `plan.rs` (ROADMAP U30). Paths unchanged.

use super::*;

/// [`check_int_op_admissibility`] with `total_ints`: the backend spells integer `Div` and
/// `ConstInt` by the total rule (`Backend::total_integer_ops`). It admits an Elementwise
/// `Div`, `Select` and comparison at `I32`/`I64`/`U32`/`U64` and a `ConstInt` leaf at an integer
/// compute dtype, and nothing else.
pub(crate) fn check_int_op_admissibility_for(
    op: &OpDef,
    dtype: ElementKind,
    total_ints: bool,
) -> Result<(), PlanError> {
    // Increment 5 — bincount exemption: a scatter with a bare `Const` body is the
    // integer-count histogram (`out[x[i]] += 1`). The `Const(1)` is NOT compute
    // (no int arithmetic, no double-math hazard) — it is a store literal the
    // scatter combine narrows EXACTLY to the count cell (`(int)(1.0)` = 1). The
    // input `x` is read only as an integer INDEX, never a value leaf. So the
    // int-Const rejection (which polices f64 literals inside int arithmetic) does
    // not apply; skip the walk for this one shape.
    if op_has_scatter(op) && matches!(op.body, ScalarExpr::Const(_)) {
        return Ok(());
    }
    let int_dt = is_int_dtype(dtype);
    check_reduce_op_admissibility(op, dtype)?;
    // PM ruling 2026-11-01: an integer `Mean` reduction has no defined semantics (the oracle and the
    // emitters assert it out), so decline it here, typed, rather than build a plan only they refuse.
    if int_dt
        && matches!(
            op.access,
            Access::Reduction {
                op: ReduceOp::Mean,
                ..
            }
        )
    {
        return Err(PlanError::InadmissibleOpAtDtype {
            dtype,
            detail: format!(
                "op '{}': an integer Mean reduction has no defined semantics (no rounding rule for the quotient), so {dtype:?} must miss honestly; use Sum and divide, or a float dtype",
                op.name
            ),
        });
    }
    let elementwise = matches!(op.access, Access::Elementwise);
    // Rule 4: true while walking an `Access::Reduction`'s `body`/`post` — the
    // any/all/count fused-predicate lift is scoped to exactly this Access arm
    // (NOT RowReduce/Contraction/Scan/Window, which stay float-only/out of
    // scope per the design spec), and covers BOTH `body` (the per-element
    // `count = in != 0` predicate) and `post` (the any/all boolean cast) —
    // both are pushed into `exprs` below and walked with the same flag.
    let in_reduction = matches!(op.access, Access::Reduction { .. });
    // `at_reduction_root` is `true` ONLY for the initial `walk` call on
    // `op.body` and on the reduction `post` (the `for e in exprs` loop
    // below), and `false` for every recursive descent (whole-branch-review
    // finding, closing the composed-predicate leak: rule 4's Cmp*/Const
    // lift must admit exactly what `cuda::emit_reduction`'s
    // `int_reduction_predicate` can integer-lower — the body/post ROOT
    // only. A `Cmp*` reached as a sub-node of Add/Sub/Mul (e.g.
    // `Add(Cmp(..), Cmp(..))`) is NOT the root, so without this flag the
    // `in_reduction && bop.is_cmp()` arm below would wrongly admit it here
    // while the emitter falls through to the FLOAT `binary_f32` speller for
    // that same non-root Cmp — reopening the `long long acc += float`
    // double-math hazard commit 6fdfe478 closed. Restricting admission to
    // the root keeps every SHIPPED catalog shape (`count`: Cmp at body
    // root; `any`/`all`: Cmp at body root AND post root) admitted while a
    // nested Cmp falls to the existing fail-closed `else` reject below —
    // the same behavior as before rule 4 existed.
    fn walk(
        e: &ScalarExpr,
        op_name: &str,
        dtype: ElementKind,
        int_dt: bool,
        elementwise: bool,
        in_reduction: bool,
        at_reduction_root: bool,
        total: bool,
    ) -> Result<(), String> {
        match e {
            ScalarExpr::Input(_) | ScalarExpr::Reduced(_) => {}
            // Coord's own gate (`check_coord_admissibility`, which also runs
            // at the top of build_plan) rejects EVERY int dtype — a Coord is
            // spelled as a float cast, the same double-math hazard this walk
            // polices for Const/Param — so this arm carries no second assert
            // (one source of truth for the message). It is also structurally
            // moot for rule 3: an int-only op's operands are pinned to leaf
            // Inputs at 8-bit before Coord could ever appear there.
            ScalarExpr::Coord(_) => {}
            ScalarExpr::ConstInt(..) if total && elementwise => {}
            ScalarExpr::CoordInt(..) | ScalarExpr::ConstInt(..) => {
                if int_dt {
                    return Err(format!(
                        "op '{op_name}': an integer-typed leaf (CoordInt/ConstInt) at int compute                          dtype {dtype:?} is rejected, integer index expressions are float-compute                          only in v1"
                    ));
                }
            }
            ScalarExpr::Qualified(x, n) => {
                if n.round == crate::ir::RoundMode::NearestAway {
                    return Err(format!(
                        "op '{op_name}': Qualified under RoundMode::NearestAway is rejected, an arithmetic operation takes only the four IEEE directions (ties-away is a Cast's mode)"
                    ));
                }
                if int_dt {
                    return Err(format!(
                        "op '{op_name}': Qualified at int compute dtype {dtype:?} is                          rejected, rounding/ftz/approx qualifiers are f32-compute only in v1"
                    ));
                }
                walk(
                    x,
                    op_name,
                    dtype,
                    int_dt,
                    elementwise,
                    in_reduction,
                    false,
                    total,
                )?;
            }
            ScalarExpr::Pack(_)
            | ScalarExpr::Extract(..)
            | ScalarExpr::BitcastPacked(..)
            | ScalarExpr::Local(_)
            | ScalarExpr::Exec(_) => {
                return Err(format!(
                    "op '{op_name}': Pack/Extract are not lowerable yet (packed values have no emitter seam)"
                ));
            }
            ScalarExpr::Cast(x, _) | ScalarExpr::Bitcast(x, _) => {
                if int_dt {
                    return Err(format!(
                        "op '{op_name}': Cast/Bitcast at int compute dtype {dtype:?} is                          rejected — typed expressions are f32/f32s/f64-compute only in v1"
                    ));
                }
                walk(
                    x,
                    op_name,
                    dtype,
                    int_dt,
                    elementwise,
                    in_reduction,
                    false,
                    total,
                )?;
            }
            ScalarExpr::Const(_) => {
                if !(!int_dt) {
                    return Err(format!(
                        "op '{op_name}': Const at int dtype {dtype:?} is rejected — a Const \
                 is spelled as an f64 C literal, which would silently run double \
                 math in an integer kernel (and f64 cannot represent all i64); \
                 int-literal Const spelling is a follow-up"
                    ));
                }
            }
            ScalarExpr::Param(_) => {
                if !(!int_dt) {
                    return Err(format!(
                        "op '{op_name}': scalar params are f32-only (int dtype {dtype:?})"
                    ));
                }
            }
            ScalarExpr::Unary(uop, x) => {
                if !(!int_dt) {
                    return Err(format!(
                        "op '{op_name}': {uop:?} has no integer lowering — the bespoke \
                     unary elementwise surface is float-only, so int dtype {dtype:?} \
                     must miss honestly"
                    ));
                }
                walk(
                    x,
                    op_name,
                    dtype,
                    int_dt,
                    elementwise,
                    in_reduction,
                    false,
                    total,
                )?;
            }
            ScalarExpr::Div(a, b) => {
                let total_int_div = total && elementwise && is_total_int_dtype(dtype);
                if !(!int_dt) && !total_int_div {
                    return Err(format!(
                        "op '{op_name}': integer division is rejected at {dtype:?} — the \
                     bespoke elementwise surface has no int div (binary_div_fp.cu is \
                     float-only) and C `/` division by zero is device-undefined; \
                     miss honestly"
                    ));
                }
                walk(
                    a,
                    op_name,
                    dtype,
                    int_dt,
                    elementwise,
                    in_reduction,
                    false,
                    total,
                )?;
                walk(
                    b,
                    op_name,
                    dtype,
                    int_dt,
                    elementwise,
                    in_reduction,
                    false,
                    total,
                )?;
            }
            ScalarExpr::Binary(bop, a, b) => {
                if bop.is_int_only() {
                    if !(elementwise) {
                        return Err(format!(
                            "op '{op_name}': {bop:?} is Elementwise-only in 0c — the \
                         reduction-class paths lower through the float accumulator \
                         spellers, which have no integer arms"
                        ));
                    }
                    // `Bool` is admitted for the LOGICAL ops specifically. It is
                    // not an int dtype (§6.2-0001 makes it its own numeric kind)
                    // but it IS the dtype the bespoke logical surface exists for.
                    // Treating it as a float — which this branch used to do,
                    // reporting "float dtype Bool" — refused the one op family
                    // `bool` is actually for, while `Add` (meaningless on a truth
                    // value) sailed through. `check_bool_op_admissibility` owns
                    // the rest of that surface.
                    let bool_logical = bop.is_logical() && dtype == ElementKind::Bool;
                    if !(int_dt || bool_logical) {
                        return Err(format!(
                            "op '{op_name}': {bop:?} is int-only (I32/I64/S8/U8), or `Bool` \
                         for the logical ops — dtype {dtype:?} must miss honestly (the \
                         bespoke bitwise/logical kernels have no instantiation for it)"
                        ));
                    }
                    if !(!bop.is_logical() || matches!(dtype, ElementKind::U8 | ElementKind::Bool))
                    {
                        return Err(format!(
                            "op '{op_name}': {bop:?} is the bespoke BOOL surface — `Bool` \
                         itself or its `U8` representation, which instantiates exactly \
                         uint8_t; so {dtype:?} must miss honestly"
                        ));
                    }
                    // Rule 3 (8-bit composition pin, v1): every operand of an
                    // int-only op at U8/S8 must be a LEAF Input — a composed
                    // operand's value differs between the inlined (un-truncated
                    // promoted-int) and hoisted (8-bit tmp, truncated) spellings,
                    // so admitting it would make the result depend on DAG
                    // sharing. See the doc comment above for the full rationale.
                    // The pin is about SUB-`int` widths, not about 8 bits: C
                    // promotes `short` to `int` exactly as it promotes `char`,
                    // so a 16-bit composed operand observes the same
                    // un-truncated promoted value. `(a+b)>>c` at s16 with
                    // (30000, 30000, 1) is 30000 inlined and -2768 hoisted —
                    // the 8-bit worked example one width up.
                    //
                    // `U32` is deliberately ABSENT, and the reason is the same
                    // rule read the other way: `unsigned int` has the same rank
                    // as `int`, so it does not promote, so a composed operand
                    // observes the *same* value inlined or hoisted. u32 is safe
                    // for composition precisely because of what makes its
                    // arithmetic unsigned. Adding it here would forbid a legal
                    // and useful shape for no reason.
                    if matches!(
                        dtype,
                        ElementKind::U8 | ElementKind::I8 | ElementKind::I16 | ElementKind::U16
                    ) {
                        for (side, operand) in [("lhs", &**a), ("rhs", &**b)] {
                            if !(matches!(operand, ScalarExpr::Input(_))) {
                                return Err(format!(
                                    "op '{op_name}': {bop:?} at {dtype:?} requires LEAF \
                                 Input operands ({side} is a composed expression) — \
                                 at 8-bit dtypes a composed operand observes the \
                                 un-truncated promoted-int value when inlined but \
                                 the truncated 8-bit value when hoisted to a shared \
                                 tmp (one body, two results); v1 pins all int-op \
                                 operands at U8/S8 to leaves. Compose at I32/I64, \
                                 or wait for the dtype-aware truncating speller"
                                ));
                            }
                        }
                    }
                    walk(
                        a,
                        op_name,
                        dtype,
                        int_dt,
                        elementwise,
                        in_reduction,
                        false,
                        total,
                    )?;
                    walk(
                        b,
                        op_name,
                        dtype,
                        int_dt,
                        elementwise,
                        in_reduction,
                        false,
                        total,
                    )?;
                } else if matches!(bop, crate::ir::BinaryOp::Max | crate::ir::BinaryOp::Min)
                    && total
                    && elementwise
                    && is_total_int_dtype(dtype)
                {
                    // `Backend::total_integer_ops` (ROADMAP U33): an Elementwise integer `Max`/`Min` at a 32/64-bit
                    // dtype. It is exact and total (the larger or smaller operand, no NaN, no overflow), the plan
                    // oracle evaluates it exactly (0.40.10), and an emitter spells it as a compare-select.
                    for operand in [&**a, &**b] {
                        walk(
                            operand,
                            op_name,
                            dtype,
                            int_dt,
                            elementwise,
                            in_reduction,
                            false,
                            total,
                        )?;
                    }
                } else if bop.is_cmp() && total && elementwise && is_total_int_dtype(dtype) {
                    // `Backend::total_integer_ops` (0.40.10): an Elementwise comparison at a 32/64-bit
                    // integer dtype, which an integer `Rem` (`Select(b == 0, 0, ..)`) needs.
                    for operand in [&**a, &**b] {
                        walk(
                            operand,
                            op_name,
                            dtype,
                            int_dt,
                            elementwise,
                            in_reduction,
                            false,
                            total,
                        )?;
                    }
                } else if int_dt && in_reduction && at_reduction_root && bop.is_cmp() {
                    // Rule 4: an int-dtype Cmp* is admitted HERE ONLY — the
                    // reduction body/post predicate position of any/all/count
                    // (`in_reduction` is false for Access::Elementwise, so the
                    // elementwise int Cmp* rejection below is untouched), AND
                    // only when this Cmp* IS the body/post ROOT
                    // (`at_reduction_root`) — a Cmp* reached as a sub-node of
                    // Add/Sub/Mul (a composed predicate like `Add(Cmp(..),
                    // Cmp(..))`) falls through to the fail-closed `else` below,
                    // because `cuda::emit_reduction`'s `int_reduction_predicate`
                    // only integer-lowers a Cmp* at that same root position —
                    // admitting a non-root Cmp here would let it fall through to
                    // the FLOAT `binary_f32` speller at emit time (whole-branch-
                    // review finding: the composed-predicate float-launder).
                    //
                    // Double-math hazard close-out: `ScalarExpr::Const` always
                    // lowers via `backend::const_lit` as an f64 C literal
                    // (`format!("{v:?}")`) REGARDLESS of dtype — that spelling
                    // is dtype-oblivious by construction and cannot be fixed
                    // from this gate alone. So every operand admitted here must
                    // be a LEAF `Input`/`Reduced` (the real per-element or
                    // folded value) or a `Const` whose value is EXACTLY `0.0`
                    // or `1.0` — the only values any/all/count ever compare
                    // against (the fixed zero threshold; the implicit 0/1
                    // keep-mask), and the only ones that round-trip losslessly
                    // through the f64 spelling into int compute (no precision
                    // loss is possible for a 1-bit value, unlike an arbitrary
                    // literal). Any other Const value, or any composed
                    // sub-expression, is a genuine authoring error smuggling
                    // float math into an int kernel and is rejected outright —
                    // this leaf-or-{0,1} pin is what keeps the lift surgical.
                    // The admission test itself is centralized in
                    // `ir::is_admissible_int_reduction_operand` — the SAME
                    // helper `cuda::assert_no_int_div_or_const` and
                    // `cuda::emit_reduction`'s `int_reduction_predicate`/
                    // `int_cmp_operand` call, so this shape cannot drift
                    // between the gate and the emitter again. This match only
                    // survives to pick the right panic message for the two
                    // ways an operand can be inadmissible.
                    for (side, operand) in [("lhs", &**a), ("rhs", &**b)] {
                        if is_admissible_int_reduction_operand(operand) {
                            continue;
                        }
                        match operand {
                            ScalarExpr::Const(v) => {
                                return Err(format!(
                                    "op '{op_name}': int reduction-predicate {bop:?} \
                                 Const({v}) at {side} must be exactly 0 or 1 — any \
                                 other value risks the f64-literal double-math \
                                 hazard this gate exists to prevent (Const always \
                                 lowers as an f64 C literal; 0/1 are the only \
                                 values that round-trip exactly into int compute)"
                                ));
                            }
                            other => {
                                return Err(format!(
                                    "op '{op_name}': int reduction-predicate {bop:?} at \
                                 {side} requires a leaf Input/Reduced or a 0/1 \
                                 Const, got {other:?} — composed operands are out \
                                 of scope for the any/all/count fused-predicate \
                                 shape"
                                ));
                            }
                        }
                    }
                } else {
                    // Fail-closed default — also the landing spot for a
                    // reduction-position Cmp* that is NOT at the body/post root
                    // (`in_reduction` true but `at_reduction_root` false): same
                    // reject as pre-rule-4, so a composed predicate misses
                    // honestly instead of being laundered through the float
                    // speller.
                    if !(!int_dt) {
                        return Err(format!(
                            "op '{op_name}': {bop:?} has no integer lowering — the \
                         bespoke elementwise surface instantiates it for float \
                         dtypes only, so int dtype {dtype:?} must miss honestly \
                         (a reduction-predicate Cmp* only lowers to integer at \
                         the body/post ROOT — nested inside Add/Sub/Mul it is \
                         out of scope for the any/all/count fused-predicate lift)"
                        ));
                    }
                    walk(
                        a,
                        op_name,
                        dtype,
                        int_dt,
                        elementwise,
                        in_reduction,
                        false,
                        total,
                    )?;
                    walk(
                        b,
                        op_name,
                        dtype,
                        int_dt,
                        elementwise,
                        in_reduction,
                        false,
                        total,
                    )?;
                }
            }
            ScalarExpr::Add(a, b) | ScalarExpr::Sub(a, b) | ScalarExpr::Mul(a, b) => {
                // Wrapping two's-complement at int dtypes — the audited-legal
                // set. Neither operand is the reduction body/post root anymore
                // (this node IS the root, if anything is), so both recurse with
                // `at_reduction_root: false` — this is precisely what closes the
                // composed-predicate leak: a Cmp* nested here (e.g. `Add(Cmp(..),
                // Cmp(..))`) now falls to the fail-closed reject above instead of
                // being admitted.
                walk(
                    a,
                    op_name,
                    dtype,
                    int_dt,
                    elementwise,
                    in_reduction,
                    false,
                    total,
                )?;
                walk(
                    b,
                    op_name,
                    dtype,
                    int_dt,
                    elementwise,
                    in_reduction,
                    false,
                    total,
                )?;
            }
            ScalarExpr::Fma(a, b, c) => {
                if int_dt {
                    return Err(format!(
                        "op '{op_name}': Fma has no integer meaning (a fused multiply-add \
                         rounds once; integer multiply-add wraps), so int dtype {dtype:?} \
                         must miss honestly"
                    ));
                }
                walk(
                    a,
                    op_name,
                    dtype,
                    int_dt,
                    elementwise,
                    in_reduction,
                    false,
                    total,
                )?;
                walk(
                    b,
                    op_name,
                    dtype,
                    int_dt,
                    elementwise,
                    in_reduction,
                    false,
                    total,
                )?;
                walk(
                    c,
                    op_name,
                    dtype,
                    int_dt,
                    elementwise,
                    in_reduction,
                    false,
                    total,
                )?;
            }
            ScalarExpr::Select(c, a, b) => {
                // G1 (WHERE/SELECT): select is rejected OUTRIGHT at every int
                // dtype — v1 select is float-only (f32/f32s/f64/f16/bf16).
                // Rationale: bespoke cmp is `_fp`-only already, so a cmp-cond
                // select is float-only by transitivity, and admitting an int
                // select would drag in the 0c U8/I8 observer problem — the
                // cond's `!= 0` test OBSERVES the un-truncated promoted-int
                // value of a composed cond (value 256: nonzero inlined, 0 when
                // hoisted to an 8-bit tmp — one body, two results) while the
                // arms are value-transparent. Rejecting outright sidesteps it
                // with zero bespoke-parity loss (`where_dtype_fanout` int
                // coverage is a later increment). Legal in EVERY Access arm at
                // float dtypes (a select in a Reduction pre-expr is the
                // masked-sum shape), so no elementwise-only assert here.
                if !(!int_dt) && !(total && elementwise && is_total_int_dtype(dtype)) {
                    return Err(format!(
                        "op '{op_name}': Select has no integer lowering — v1 select is \
                     float-only (f32/f32s/f64/f16/bf16), so int dtype {dtype:?} must \
                     miss honestly (the 0c U8/I8 cond-observer question is unresolved \
                     and bespoke where int coverage is a later increment)"
                    ));
                }
                walk(
                    c,
                    op_name,
                    dtype,
                    int_dt,
                    elementwise,
                    in_reduction,
                    false,
                    total,
                )?;
                walk(
                    a,
                    op_name,
                    dtype,
                    int_dt,
                    elementwise,
                    in_reduction,
                    false,
                    total,
                )?;
                walk(
                    b,
                    op_name,
                    dtype,
                    int_dt,
                    elementwise,
                    in_reduction,
                    false,
                    total,
                )?;
            }
        }
        Ok(())
    }
    let mut exprs: Vec<&ScalarExpr> = vec![&op.body];
    match &op.access {
        Access::RowReduce { stages, epilogue } => {
            exprs.extend(stages.iter().map(|s| &s.pre));
            exprs.push(epilogue);
        }
        Access::Contraction { epilogue, .. } => exprs.push(epilogue),
        // The reduction post-expr (0e) lowers at the accumulator dtype — a
        // Const/Param/Div/unary there hits the same int-dtype hazards, so gate it.
        Access::Reduction { post, .. } => exprs.push(post),
        // Increment 6 SCAN: `pre`/`post` lower at the accumulator dtype (an int
        // cumsum/cummax rides the serial base), so a Const/Param/Div/unary/int-only
        // op there hits the same int hazards — gate both.
        Access::Scan { pre, post, .. } => {
            exprs.push(pre);
            exprs.push(post);
        }
        // Increment 7 WINDOW: `pre`/`post` lower at the accumulator dtype (an int
        // sum/max pool rides the same fold), so a Const/Param/Div/unary/int-only
        // op there hits the same int hazards — gate both.
        Access::Window { pre, post, .. } => {
            exprs.push(pre);
            exprs.push(post);
        }
        // Increment 8 SORT_PERM: `body` is pinned `Input(0)` (already in `exprs`,
        // trivially int-clean) and RowSort has no pre/post; an int sort just
        // permutes storage bits (no int arithmetic), so nothing extra to gate.
        Access::RowSort { .. } => {}
        // Increment 11 IM2COL: `body` is pinned `Input(0)` (already in `exprs`,
        // trivially int-clean) and im2col has no pre/post; a raw-bit gather does no
        // int arithmetic (no Div/Const/unary), so nothing extra to gate.
        Access::Im2Col { .. } => {}
        // Review-caught gate asymmetry (increment 1): walk the EXTRA output bodies
        // too. Without this, an int-only op with a COMPOSED operand hides in a
        // multi-output extra body at U8/S8 and bypasses the 8-bit leaf-operand pin
        // (rule 3) — cross-body CSE then hoists it into a truncated 8-bit tmp, the
        // exact 0c value-divergence ((200+100)>>1 = 22 hoisted vs 150 inlined).
        // Non-elementwise multi-output is already rejected by
        // `assert_valid_multi_output` (runs first), so this is the only place
        // `extra_out_bodies` is legal; empty for every single-output op.
        Access::Elementwise => exprs.extend(op.extra_out_bodies.iter()),
    }
    for e in exprs {
        // `at_reduction_root: in_reduction` — true here exactly because
        // `exprs` is `[body, post]` precisely when `op.access` is
        // `Access::Reduction` (the only branch that pushes `post` instead of
        // an epilogue/stage; see the match above), i.e. this IS the initial
        // call on the reduction body/post root. For every other Access arm
        // `in_reduction` is already `false`, so the value is moot there (the
        // Cmp* admission arm requires `in_reduction` regardless).
        walk(
            e,
            &op.name,
            dtype,
            int_dt,
            elementwise,
            in_reduction,
            in_reduction,
            total_ints,
        )
        .map_err(|detail| PlanError::InadmissibleOpAtDtype { dtype, detail })?;
    }
    Ok(())
}
