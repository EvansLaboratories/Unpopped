// SPDX-License-Identifier: MIT OR Apache-2.0
//! Section of the plan gate; split from `plan.rs` (ROADMAP U30). Paths unchanged.

use super::*;

/// Validate [`crate::ir::OpDef::out_dtype`] at plan time (AOT — like
/// `check_no_half_nextafter`, this runs at the top of [`build_plan`] so EVERY
/// Access arm and every lowering path is covered; a panic here is an
/// author-error backstop, and the JIT never constructs a `Some` out_dtype).
///
/// Two admitted hetero-output shapes, both with an EXACT store conversion:
///
/// 1. **[`Access::Elementwise`] predicate → `Some(U8)`** (increment 0b): the
///    body ROOT must be a `Cmp*` — the value is exactly 0.0/1.0 and
///    `(unsigned char)` of that is exactly 1/0 (`OpDef::elementwise_pred`).
///
/// 2. **[`Access::Reduction`] hetero-out** (increment 0e — the roadmap "any/all
///    → U8, count → I64" reduction):
///    - `Some(U8)`: the POST-expr ROOT must be a `Cmp*`, so the stored value is
///      exactly 0.0/1.0 regardless of the fold magnitude — the honest
///      boolean-reduce (`any` = `Sum(x≠0)` with post `Reduced(0) > 0`; `all` =
///      `Sum(x=0)` with post `Reduced(0) = 0`; or `Max`/`Min` of a predicate
///      wrapped in a redundant cmp post). A non-cmp post would truncate the raw
///      accumulator silently (a count of 300 → `44` at u8), so it rejects.
///    - `Some(I64)`: the combine must be `Sum` and the post the identity
///      `Reduced(0)` — a **count** (`Sum(x≠0)`) or a sum-widening. The store is
///      `(long long)` of the accumulator: exact for an int input (i32→i64
///      widening) and exact for a float accumulator while the count ≤ 2²⁴ (a
///      documented CALLER precondition, the same trust level as `Coord`'s
///      exact-integer bound — the key abstracts extents away). `Mean`/`Max`/
///      `Min` → I64 reject (fractional / not the count shape).
///
/// Everything else panics honestly: `Some(non-U8/I64)` has no store conversion;
/// a `RowReduce`/`Contraction` stores its accumulator, not a predicate, so a
/// hetero store there would truncate silently.
///
/// A `Cmp*` NESTED inside a float body (mask-multiply `dy * (x > 0)`) is legal
/// with `out_dtype = None` — it is an inline 0.0/1.0 float, no u8 store — and a
/// top-level cmp with `out_dtype = None` (a float mask) is likewise legal.
pub(super) fn assert_valid_out_dtype(op: &OpDef) {
    let Some(od) = op.out_dtype else { return };
    // A typed body's store type is checked by `check_typed_admissibility` (the body's
    // inferred type must equal this dtype), which `try_build_plan` ran first; the
    // predicate-mask and bincount shapes below are about untyped bodies.
    if matches!(&op.access, Access::Elementwise) && crate::types::is_typed(&op.body) {
        return;
    }
    match &op.access {
        // Increment 5 — a SCATTER may write an `I32` counts output (bincount:
        // `out[x[i]] += 1`), a hetero store distinct from the U8 mask. The body is
        // a `Const` increment (no input value read), narrowed exactly to the count
        // cell; the write role's own gate (`assert_valid_scatter`) covers the rest.
        Access::Elementwise if op_has_scatter(op) => {
            assert!(
                od == ElementKind::I32 && matches!(&op.body, ScalarExpr::Const(_)),
                "OpDef '{}': a hetero-output scatter admits only the bincount shape \
                 (out_dtype Some(I32) with a Const increment body), got out_dtype \
                 Some({od:?})",
                op.name
            );
        }
        Access::Elementwise => {
            assert!(
                od == ElementKind::U8,
                "OpDef '{}': out_dtype Some({od:?}) is unsupported for an \
                 Elementwise op — the only hetero output dtype there is U8 (the \
                 comparison-predicate mask; use OpDef::elementwise_pred)",
                op.name
            );
            assert!(
                matches!(&op.body, ScalarExpr::Binary(bop, _, _) if bop.is_cmp()),
                "OpDef '{}': out_dtype = Some(U8) requires the body ROOT to be a \
                 comparison (ScalarExpr::Binary with a Cmp* op) — only a predicate \
                 yields exactly 0.0/1.0, so any other body would truncate silently \
                 under the u8 store; nested comparisons in a float body take \
                 out_dtype = None instead",
                op.name
            );
        }
        Access::Reduction { op: rop, post, .. } => match od {
            ElementKind::U8 => assert!(
                matches!(post, ScalarExpr::Binary(bop, _, _) if bop.is_cmp()),
                "OpDef '{}': a U8-output reduction requires the POST-expr ROOT to \
                 be a comparison (Cmp*) — only then is the stored value exactly \
                 0.0/1.0 (the honest any/all boolean-reduce); a non-cmp post would \
                 truncate the raw accumulator silently under the u8 store",
                op.name
            ),
            ElementKind::I64 => assert!(
                matches!(rop, ReduceOp::Sum | ReduceOp::ArgMax | ReduceOp::ArgMin)
                    && matches!(post, ScalarExpr::Reduced(0)),
                "OpDef '{}': an I64-output reduction is the count/sum-widening shape \
                 — it requires op = Sum, ArgMax or ArgMin and the identity post (Reduced(0)); \
                 Mean/Max/Min → I64 or a non-identity post is out of scope (would \
                 not be an exact integer store)",
                op.name
            ),
            other => panic!(
                "OpDef '{}': out_dtype Some({other:?}) is unsupported for a \
                 reduction — v1 admits U8 (boolean any/all, via a Cmp* post) and \
                 I64 (count, via Sum + identity post)",
                op.name
            ),
        },
        // An integer accumulation policy stores its `i32` accumulator (`try_build_plan` has checked the rest).
        Access::Contraction { accum, .. }
            if od == ElementKind::I32 && accum_integer_step(accum).is_some() => {}
        Access::RowReduce { .. } | Access::Contraction { .. } => panic!(
            "OpDef '{}': out_dtype = Some({od:?}) is rejected under \
             RowReduce/Contraction — a fused reduction/contraction stores its \
             accumulator, not a 0/1 predicate, so a hetero store would truncate \
             silently; only Access::Elementwise (predicate → U8) and \
             Access::Reduction (any/all → U8, count → I64) carry a hetero output",
            op.name
        ),
        // Increment 6 SCAN: a cumulative op does not change dtype (the output is
        // same-shape, same-dtype as Input(0)), so a hetero out_dtype has no exact
        // store — reject it, exactly like RowReduce/Contraction.
        Access::Scan { .. } => panic!(
            "OpDef '{}': out_dtype = Some({od:?}) is rejected under Scan — a prefix \
             scan is same-dtype as its input (a cumulative op does not change \
             dtype), so there is no hetero store; use out_dtype = None",
            op.name
        ),
        // Increment 7 WINDOW: a pool preserves the input dtype (max/avg pool output
        // is same-dtype as the pooled tensor), so a hetero out_dtype has no exact
        // store — reject it, exactly like Scan/RowReduce/Contraction.
        Access::Window { .. } => panic!(
            "OpDef '{}': out_dtype = Some({od:?}) is rejected under Window — a \
             pooling reduction is same-dtype as its input (a pool does not change \
             dtype), so there is no hetero store; use out_dtype = None",
            op.name
        ),
        // Increment 8/9 SORT_PERM: the ONLY hetero output under RowSort is the
        // ARGSORT index output — `SortOut::Indices` with `Some(I32)` (the sort
        // permutation, riding the single-output out_dtype precedent). A values-sort
        // (`Values`) and the fused two-output (`Both`) are dtype-preserving on
        // output 0 (out_dtype None ⇒ this fn returned early; for `Both` the I32
        // out_idx is emitter-hardwired, NOT off out_dtype). Reject any other
        // Some(_), and reject Some(I32) on a Values/Both sort. Double-gated: also
        // enforced (state ⇔ out_dtype) in `validate_row_sort`.
        Access::RowSort { out, .. } => assert!(
            matches!(out, SortOut::Indices) && od == ElementKind::I32,
            "OpDef '{}': out_dtype Some({od:?}) under RowSort is admitted only as \
             the argsort index output (SortOut::Indices requires Some(I32)); a \
             values-sort and the fused Both are dtype-preserving (out_dtype None)",
            op.name
        ),
        // Increment 11 IM2COL: a pure gather preserves the input dtype (the column
        // matrix is same-dtype as the NCHW input), so a hetero out_dtype has no exact
        // store — reject it, exactly like Scan/Window/RowReduce/Contraction.
        Access::Im2Col { .. } => panic!(
            "OpDef '{}': out_dtype = Some({od:?}) is rejected under Im2Col — a 2-D \
             im2col is a dtype-preserving raw-bit gather (the column matrix keeps the \
             input dtype), so there is no hetero store; use out_dtype = None",
            op.name
        ),
    }
}

/// Validate the increment-0e reduction **post-expression** at plan time (AOT).
/// Runs at the top of [`build_plan`] (like the other honest-miss gates), with an
/// independent emitter backstop in `cuda::emit_reduction` (the post's `leaf`
/// closure panics if an `Input` ever reaches it). The post references the fold
/// result as `Reduced(0)` and MAY read `Const`/`Param`; it must NOT read:
///
/// - `Input(_)` — the reduced axis is gone, so an input at the output coordinate
///   is a different, ambiguous tensor (this mirrors the contraction epilogue's
///   `epilogue_reads_only_reduced0`, generalized to also admit `Param`);
/// - `Coord(_)` — reduction-class, Elementwise-only (also caught upstream);
/// - `Reduced(s)` for `s ≥ 1` — a single-fold reduction produces only
///   `Reduced(0)`.
pub(super) fn assert_valid_reduction_post(op: &OpDef) {
    let Access::Reduction { post, .. } = &op.access else {
        return;
    };
    fn walk(e: &ScalarExpr, name: &str) {
        match e {
            ScalarExpr::Reduced(0) | ScalarExpr::Const(_) | ScalarExpr::Param(_) => {}
            ScalarExpr::Reduced(s) => panic!(
                "OpDef '{name}': reduction post-expr Reduced({s}) — a single-fold \
                 reduction produces only Reduced(0)"
            ),
            ScalarExpr::Input(i) => panic!(
                "OpDef '{name}': reduction post-expr must not read Input({i}) — the \
                 reduced axis is gone, so an input at the output coordinate is a \
                 different, ambiguous tensor; the post reads Reduced(0)/Const/Param"
            ),
            ScalarExpr::Coord(d) => panic!(
                "OpDef '{name}': reduction post-expr must not read Coord({d}) — Coord \
                 is Elementwise-only (a coordinate along a folded axis is ambiguous)"
            ),
            ScalarExpr::CoordInt(d, _) => panic!(
                "OpDef '{name}': reduction post-expr must not read CoordInt({d}) — Coord                  is Elementwise-only (a coordinate along a folded axis is ambiguous)"
            ),
            ScalarExpr::ConstInt(..) | ScalarExpr::Local(_) | ScalarExpr::Exec(_) => {}
            ScalarExpr::Unary(_, x) => walk(x, name),
            ScalarExpr::Cast(x, _)
            | ScalarExpr::Bitcast(x, _)
            | ScalarExpr::Qualified(x, _)
            | ScalarExpr::Extract(x, _)
            | ScalarExpr::BitcastPacked(x, ..) => walk(x, name),
            ScalarExpr::Pack(xs) => {
                for x in xs {
                    walk(x, name);
                }
            }
            ScalarExpr::Add(a, b)
            | ScalarExpr::Sub(a, b)
            | ScalarExpr::Mul(a, b)
            | ScalarExpr::Div(a, b)
            | ScalarExpr::Binary(_, a, b) => {
                walk(a, name);
                walk(b, name);
            }
            ScalarExpr::Select(c, a, b) | ScalarExpr::Fma(c, a, b) => {
                walk(c, name);
                walk(a, name);
                walk(b, name);
            }
        }
    }
    walk(post, &op.name);
}
