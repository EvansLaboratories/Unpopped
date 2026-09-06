//! **All FOUR fold-shaped `Access` variants, not the two I happened to repair.**
//!
//! # Why this exists
//!
//! Repairing the §6.16-0011 classification for `RowReduce` I checked `Reduction`
//! and `RowReduce` — **two of four** — and then wrote that the structural-safety
//! claim was repaired, without saying which shapes the evidence covered.
//! `Scan` and `Window` I never looked at.
//!
//! ⚠️ **The claim antecedent to the repair was itself per-instance**: that
//! `is_bit_move_fold_output`'s signature "cannot be satisfied without naming the
//! fold" lapsed for the shape with more folds. **Fixing the shape that broke it
//! and declaring the class repaired is the same move, one level up.** The Claim
//! Auditor's question — *is the repaired claim true for EVERY shape, or the two
//! you looked at?* — is what forced the disclosure.
//!
//! # The measured answer
//!
//! `Scan` and `Window` each carry `(op, pre, post)`, which maps **directly** onto
//! `is_bit_move_fold_output(fold, element, post)`. So they were always covered —
//! **but that is a fact I had not checked when I asserted it**, and the point of
//! this file is that it is now checked rather than that the answer is reassuring.
//!
//! | shape | fold | element | epilogue | correct call |
//! |---|---|---|---|---|
//! | `Reduction` | `access.op` | `plan.body` | `access.post` | `is_bit_move_fold_output` |
//! | `Scan` | `access.op` | `access.pre` | `access.post` | `is_bit_move_fold_output` |
//! | `Window` | `access.op` | `access.pre` | `access.post` | `is_bit_move_fold_output` |
//! | `RowReduce` | per stage | `stages[i].pre` | `body` (the epilogue) | `is_bit_move_row_reduce_output` |

use unpopped::ir::{
    ReduceOp, ReduceStage, UnaryOp, input, is_bit_move_fold_output, is_bit_move_row_reduce_output,
    reduced,
};

/// Every shape whose fields are `(op, element, post)` classifies identically,
/// because §6.16-0011 traces inputs→output and does not care which access
/// variant produced the fold.
#[test]
fn the_three_single_fold_shapes_classify_identically() {
    let element = input(0).0;
    let identity = reduced(0).0;
    let moving = reduced(0).unary(UnaryOp::Neg).0;
    let arithmetic = reduced(0).unary(UnaryOp::Sqrt).0;

    // Reduction, Scan and Window all present the same triple to the predicate,
    // so one table covers three shapes -- which is the property that makes the
    // single helper correct for all of them rather than a lucky fit.
    for (post, want, why) in [
        (&identity, true, "identity epilogue"),
        (&moving, true, "pure-move epilogue"),
        (&arithmetic, false, "arithmetic epilogue"),
    ] {
        assert_eq!(
            is_bit_move_fold_output(ReduceOp::Max, &element, post),
            want,
            "Max fold, {why}: §6.16-0011 traces inputs->output and is blind to \
             the access variant, so Reduction/Scan/Window must agree"
        );
    }

    // And the fold half still binds on every one of them.
    assert!(
        !is_bit_move_fold_output(ReduceOp::Sum, &element, &identity),
        "a Sum fold computes on ANY of the three shapes"
    );
}

/// ⚠️ `RowReduce` is the one shape the single-fold helper cannot serve, and this
/// pins WHY rather than restating that it cannot.
#[test]
fn only_row_reduce_needs_the_multi_stage_helper() {
    let stage = |op| ReduceStage {
        pre: input(0).0,
        op,
    };
    let identity = reduced(0).0;

    // The distinguishing case: one move-fold beside one arithmetic fold. No
    // (op, element, post) triple can express it, because there is no single
    // `op` -- which is exactly why RowReduce needed its own helper.
    assert!(
        !is_bit_move_row_reduce_output(&[stage(ReduceOp::Max), stage(ReduceOp::Sum)], &identity),
        "every stage is traced"
    );
    assert!(
        is_bit_move_row_reduce_output(&[stage(ReduceOp::Max), stage(ReduceOp::Min)], &identity),
        "control: two move-folds ARE a move, or the assertion above is satisfied \
         by rejecting every multi-stage op"
    );

    // A single-stage RowReduce must agree with the single-fold helper, or the
    // two predicates disagree on the shape where they overlap -- which is the
    // seam a consumer would trip on when refactoring between them.
    for (post, why) in [
        (&identity, "identity"),
        (&reduced(0).unary(UnaryOp::Neg).0, "moving"),
    ] {
        assert_eq!(
            is_bit_move_row_reduce_output(&[stage(ReduceOp::Max)], post),
            is_bit_move_fold_output(ReduceOp::Max, &input(0).0, post),
            "a ONE-stage RowReduce and a Reduction are the same trace ({why} \
             epilogue); the two helpers must not disagree where they overlap"
        );
    }
}
