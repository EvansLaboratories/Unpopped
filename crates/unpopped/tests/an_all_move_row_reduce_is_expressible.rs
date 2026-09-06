//! **Is a RowReduce whose stages AND epilogue are all moves CONSTRUCTIBLE?**
//!
//! Asked by baracuda 2026-09-06, and the answer decides whether a gap in their
//! shipped CUDA emitter is live or unreachable-by-construction: their
//! `emit_row_reduce_impl` promotes the accumulator unconditionally
//! (`let acc = if dbl { "double" } else { "float" }`), with **no bit-move path
//! at all**.
//!
//! Under KISS-OPS §6.16-0011 (KISS `origin/main` `3db1f994`, `spec/ops.md:1536`)
//! an op every one of whose transformations is a move — **the fold included** —
//! is governed by §6.16-0009 and its output bits MUST be preserved. So if such a
//! RowReduce is expressible, promoting its accumulator quiets an sNaN the clause
//! preserves, and that is a live conformance defect rather than a latent one.
//!
//! ⚠️ **They could not answer this from their side and said so.** Every RowReduce
//! they build is softmax or rmsnorm, whose epilogues are arithmetic
//! (`exp(x - m)`, `x * rsqrt(..)`) — **but "every one I build" is a fact about
//! their recipes, not about the IR.** I own the IR, so the question is mine.

use unpopped::ir::{OpDef, ReduceOp, ReduceStage, UnaryOp, input, reduced};
use unpopped::plan::build_plan;
use unpopped_vocab::{ArchSku, ElementKind, OpCategory, OperandDesc, structure_key};

/// A `Max` row-fold over raw inputs, under an epilogue that is itself a pure
/// move. Every transformation on the input→output path is a move.
#[test]
fn an_all_move_row_reduce_is_expressible_and_survives_planning() {
    let dt = ElementKind::F16;
    let d = OperandDesc::new(2, &[4, 4], &[4, 1], dt, 4);
    let key = structure_key(OpCategory::Reduction, &[d, d], ArchSku::Sm89);

    let op = OpDef::row_reduce(
        "all_move_row_reduce",
        1,
        &[dt],
        vec![ReduceStage {
            pre: input(0).0,
            op: ReduceOp::Max,
        }],
        reduced(0).unary(UnaryOp::Neg),
    );

    let plan = build_plan(&op, &key);

    // The epilogue IS `plan.body` for this shape -- the opposite field from
    // `Access::Reduction`, which is the trap the ir.rs field table exists for.
    assert!(
        matches!(plan.body, unpopped::ir::ScalarExpr::Unary(UnaryOp::Neg, _)),
        "the epilogue must reach plan.body for RowReduce, or this test is \
         measuring the wrong field: {:?}",
        plan.body
    );

    // Control: the same shape with an ARITHMETIC epilogue must also plan, or
    // "it planned" says nothing -- both arms have to be reachable for the
    // affirmative to carry information.
    let arithmetic = OpDef::row_reduce(
        "arithmetic_epilogue_row_reduce",
        1,
        &[dt],
        vec![ReduceStage {
            pre: input(0).0,
            op: ReduceOp::Max,
        }],
        reduced(0).unary(UnaryOp::Sqrt),
    );
    let _ = build_plan(&arithmetic, &key);
}

/// The predicate says this op is a move, so §6.16-0009 governs its output.
///
/// ⚠️ **Stated as its own test because the two facts are separable and only the
/// PAIR is the finding:** that the op is constructible is about my IR; that it
/// classifies as a move is about the clause. **A consumer needs both to know
/// their emitter must carry a bit-move path.**
#[test]
fn and_the_predicate_classifies_it_as_a_move() {
    use unpopped::ir::is_bit_move_fold_output;

    let element = input(0).0;
    let epilogue = reduced(0).unary(UnaryOp::Neg).0;

    assert!(
        is_bit_move_fold_output(ReduceOp::Max, &element, &epilogue),
        "a Max row-fold over raw inputs under a sign-edit epilogue is a move on \
         every transformation, so §6.16-0011 puts -0009 on the whole op"
    );

    // Control, and it is the row that catches a fold-blind reading: make the
    // FOLD arithmetic while every VISIBLE transformation stays a move.
    assert!(
        !is_bit_move_fold_output(ReduceOp::Sum, &element, &epilogue),
        "a Sum fold computes even under a pure-move epilogue -- the fold is \
         itself one of the transformations traced (§6.16-0011)"
    );
}

/// The RowReduce-shape predicate, against the clause's own four rows plus the
/// two refusals that only this shape can express.
///
/// ⚠️ **The multi-stage rows are the point.** `Access::Reduction` has one fold,
/// so no test on that shape can distinguish "the fold is a move" from "every
/// fold is a move" — and §6.16-0011 says *every* transformation, the folds
/// included.
#[test]
fn the_row_reduce_predicate_traces_every_stage() {
    use unpopped::ir::{ReduceStage, is_bit_move_row_reduce_output};

    let mv = |op| ReduceStage {
        pre: input(0).0,
        op,
    };
    let identity = reduced(0).0;
    let moving = reduced(0).unary(UnaryOp::Neg).0;
    let arithmetic = reduced(0).unary(UnaryOp::Sqrt).0;

    // The four clause rows, on this shape.
    assert!(is_bit_move_row_reduce_output(
        &[mv(ReduceOp::Max)],
        &identity
    ));
    assert!(is_bit_move_row_reduce_output(&[mv(ReduceOp::Max)], &moving));
    assert!(!is_bit_move_row_reduce_output(
        &[mv(ReduceOp::Max)],
        &arithmetic
    ));
    assert!(
        !is_bit_move_row_reduce_output(&[mv(ReduceOp::Sum)], &moving),
        "a Sum fold computes even under a pure-move epilogue (§6.16-0011)"
    );

    // ⚠️ The row `Access::Reduction` CANNOT express: one move-fold and one
    // arithmetic fold. Every visible transformation on the epilogue is a move
    // and the op is still computed, because a traced fold is arithmetic.
    assert!(
        !is_bit_move_row_reduce_output(&[mv(ReduceOp::Max), mv(ReduceOp::Sum)], &identity),
        "EVERY stage is traced -- one arithmetic fold among moves makes the \
         whole op computed, and a predicate checking only the first or last \
         stage would pass this"
    );
    assert!(
        is_bit_move_row_reduce_output(&[mv(ReduceOp::Max), mv(ReduceOp::Min)], &identity),
        "control: two move-folds ARE a move, or the assertion above passes by \
         rejecting every multi-stage op"
    );

    // A later stage referencing an earlier one is a MOVED value, not an opaque
    // leaf -- the leaf policy this shape needs and `Reduction` never exercises.
    assert!(
        is_bit_move_row_reduce_output(
            &[
                mv(ReduceOp::Max),
                ReduceStage {
                    pre: reduced(0).unary(UnaryOp::Abs).0,
                    op: ReduceOp::Min,
                },
            ],
            &identity
        ),
        "stage 1's `pre` may reference Reduced(0), which is itself a move by \
         induction"
    );

    // No fold is not an affirmative answer.
    assert!(
        !is_bit_move_row_reduce_output(&[], &identity),
        "an empty stage list has no fold to trace -- the question is malformed, \
         and `true` would hand a caller a bit-move route for an op with no \
         reduction in it"
    );
}

/// ⚠️ **The prescription this module's field table used to give was WRONG for
/// stages after the first, and wrong in the direction nobody audits.**
///
/// It said: `Access::RowReduce  is_bit_move_reduce(stage.op, &stage.pre)` per
/// stage. But `is_bit_move_reduce` runs the leaf policy `false`, so a stage whose
/// `pre` references an earlier `Reduced(_)` scores FALSE — and a legitimately
/// all-move multi-stage RowReduce is classified computed.
///
/// **It fails SAFE — quieting where it could have preserved — which is why it
/// could sit in a doc block for months.** A consumer following it emits
/// conforming-but-pessimised code and has no symptom to report.
///
/// This pins the difference so the prescription cannot silently come back.
#[test]
fn the_old_per_stage_prescription_disagrees_with_the_shape_predicate() {
    #[allow(deprecated)]
    use unpopped::ir::is_bit_move_reduce;
    use unpopped::ir::{ReduceStage, is_bit_move_row_reduce_output};

    // Stage 1 refers to stage 0's result — the ordinary multi-stage shape.
    let stage0 = ReduceStage {
        pre: input(0).0,
        op: ReduceOp::Max,
    };
    let stage1 = ReduceStage {
        pre: reduced(0).unary(UnaryOp::Abs).0,
        op: ReduceOp::Min,
    };

    #[allow(deprecated)]
    let per_stage_says =
        is_bit_move_reduce(stage0.op, &stage0.pre) && is_bit_move_reduce(stage1.op, &stage1.pre);

    let shape_says = is_bit_move_row_reduce_output(&[stage0, stage1], &reduced(0).0);

    assert!(
        shape_says,
        "every fold is Max/Min and every transformation is a move, so \
         §6.16-0011 governs this as a move"
    );
    assert!(
        !per_stage_says,
        "the OLD prescription must disagree here -- if it ever agrees, this \
         test has stopped measuring the hazard it was written for"
    );
}
