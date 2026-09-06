//! The driver declares a hoisted temporary through a seam, not in C.
//!
//! # The leak this closes, and why nothing could see it
//!
//! `lower_node` — the walker every backend reaches through [`lower_dag`],
//! [`lower_dag_multi`] and [`lower_dag_all`] — contained exactly one
//! target-specific line:
//!
//! ```text
//! prelude.push(format!("{ctype} {name} = {rhs};"));
//! ```
//!
//! `type name = rhs;` is C-family syntax sitting in the language-neutral core.
//! **Both in-tree emitters are C-family** (CpuC is C, Slang extends C++/HLSL),
//! so every existing test agreed with it, and a test that agrees with a bug is
//! indistinguishable from one that agrees with a fix. The neutrality claim had
//! no way to fail — the same shape as an in-crate implementation that cannot
//! exercise the extension API it defines.
//!
//! A target whose declaration form is not `type name = rhs;` — an SSA or
//! three-address ISA, a `let`-binding language — could not use the shared
//! walker at all, and the core asserted neutrality anyway.
//!
//! # Why the gate needs a second formatter rather than an assertion on C
//!
//! A test that asserts the C output is correct **passes identically with the
//! seam wired and with it ignored**. It re-verifies C against C. The arm that
//! discriminates is *the same DAG lowered through two formatters produces two
//! different declaration forms* — which requires a non-C-family formatter to
//! exist, even a stub nobody would ship.
//!
//! **Measured, not argued.** With the driver reverted to the hardcoded
//! `format!("{ctype} {name} = {rhs};")`, this file runs **2 passed, 1 failed**:
//! `the_default_still_emits_the_c_form_byte_for_byte` and
//! `the_public_c_temp_is_the_default_the_driver_uses` both **certify the bug as
//! fixed**. Only the two-formatter arm goes red. Two thirds of this file would
//! have signed off on the leak.
//!
//! # Scope, stated because a half-closed leak described as closed is worse
//!
//! This makes the **declaration** neutral, not the whole lowering. `rhs` is
//! still whatever the `binary`/`arith` seams spelled, and those compose
//! *expressions* (`(a + b)`). A real three-address emitter also needs its
//! operand seams to emit instructions and return register names, which this
//! seam does not reach.

use unpopped::backend::{LowerError, Lowering, Spelling, c_temp, lower_dag_all};
use unpopped::ir::{ExprDag, ScalarExpr};

/// `(in0 + in1) + in2` — two non-leaf nodes, so `lower_dag_all` hoists **twice**.
/// More than one temp on purpose: a seam consulted for the first and hardcoded
/// for the rest would pass a single-temp test.
fn two_temp_dag() -> ExprDag {
    let inner = ScalarExpr::Add(
        Box::new(ScalarExpr::Input(0)),
        Box::new(ScalarExpr::Input(1)),
    );
    let outer = ScalarExpr::Add(Box::new(inner), Box::new(ScalarExpr::Input(2)));
    ExprDag::from_expr(&outer)
}

/// Lower the shared corpus, optionally overriding the declaration seam.
fn prelude(temp: Option<&dyn Fn(&str, &str, &str) -> String>) -> Vec<String> {
    let leaf = |i: u8| Ok(Spelling::Spelled(format!("in{i}")));
    let never_un = |_, _| -> Result<Spelling, LowerError> { panic!("no unary in this corpus") };
    let never_bin =
        |_, _, _| -> Result<Spelling, LowerError> { panic!("no binary in this corpus") };
    let arith = |_op, a: String, b: String| Ok(Spelling::Spelled(format!("({a} + {b})")));

    // Through the builder, not a struct literal: `Lowering` is `#[non_exhaustive]`,
    // so this reaches the seam the way an out-of-crate backend must.
    let b = Lowering::builder(&leaf, &never_un, &never_bin).arith(&arith);
    let lo = match temp {
        Some(f) => b.temp(f).build(),
        None => b.build(),
    };
    let (prelude, _root) =
        lower_dag_all(&two_temp_dag(), "float", &lo).expect("this corpus lowers");
    prelude
}

/// A declaration form that is **not** `type name = rhs;`.
///
/// PTX-flavoured: a register declaration and a move, two statements per temp.
/// It is a stub, not shippable PTX — its only job is to be unmistakably not-C
/// so the discriminating arm has somewhere else to come from.
fn ssa_temp(ctype: &str, name: &str, rhs: &str) -> String {
    format!(".reg .{ctype} {name}; mov.{ctype} {name}, {rhs};")
}

#[test]
fn the_declaration_form_comes_from_the_seam_and_not_from_the_driver() {
    let c = prelude(None);
    let ssa = prelude(Some(&ssa_temp));

    // VACUITY CONTROL. Two empty preludes compare equal to each other and
    // unequal to nothing, so an expression that never hoisted would make every
    // assertion below vacuously true. This is the assertion that makes the rest
    // mean something.
    assert_eq!(
        c.len(),
        2,
        "the corpus must hoist exactly twice for this gate to test anything; got {c:?}"
    );
    assert_eq!(ssa.len(), c.len(), "both formatters see the same hoists");

    // THE DISCRIMINATING ARM. Without a non-C formatter this comparison cannot
    // exist, and the test degrades to re-verifying C against C.
    assert_ne!(
        c, ssa,
        "the driver produced the same declarations through two different \
         formatters, which means it ignored the seam and is still spelling C"
    );

    for line in &ssa {
        assert!(
            line.starts_with(".reg ") && !line.contains(" = "),
            "a declaration came from the driver rather than the supplied seam: {line}"
        );
    }
}

/// Opening the seam changed no emitted byte.
///
/// Pinned as exact strings rather than a shape: this is the output every
/// in-tree emitter and every downstream consumer already depends on, and the
/// point of an additive seam is that the default path is untouched.
#[test]
fn the_default_still_emits_the_c_form_byte_for_byte() {
    assert_eq!(
        prelude(None),
        vec![
            "float tmp0 = (in0 + in1);".to_string(),
            "float tmp1 = (tmp0 + in2);".to_string(),
        ]
    );
}

/// The default is [`c_temp`], and `c_temp` is what it claims to be.
///
/// Without this, `c_temp` could drift from the built-in default and the test
/// above would still pass by describing whatever the default became.
#[test]
fn the_public_c_temp_is_the_default_the_driver_uses() {
    assert_eq!(
        c_temp("float", "tmp0", "(in0 + in1)"),
        "float tmp0 = (in0 + in1);"
    );
    assert_eq!(prelude(None), prelude(Some(&c_temp)));
}

/// `is_bit_or_sign_move`'s SUBJECT is one expression, and at a reduction the fold
/// is not in it — pinned so the doc comment cannot drift from the behaviour.
///
/// `plan.body` is safe because a reduction's body is the epilogue, whose identity
/// is `Reduced(0)` and which is not an arm of the predicate. `ReduceStage::pre` is
/// the hazard: its identity is `Input(0)`, which answers TRUE for a sum-fold
/// exactly as for a max-fold, so a caller routing on it alone would send an
/// arithmetic reduction down a bit-move path.
#[test]
fn the_move_predicate_cannot_see_a_reduction_fold() {
    use unpopped::ir::{OpDef, ReduceOp, ReduceStage, is_bit_or_sign_move, reduced};
    use unpopped_vocab::ElementKind;

    for rop in [ReduceOp::Max, ReduceOp::Sum] {
        let stage = ReduceStage {
            pre: unpopped::ir::input(0).0,
            op: rop,
        };
        let op = OpDef::row_reduce("r", 1, &[ElementKind::F32], vec![stage.clone()], reduced(0));

        // ⚠️ ONLY for RowReduce. `Access::Reduction` puts the ELEMENT expr in
        // `body`, so the same assertion there would be false — see the second
        // loop below, which is the half the first version of this test missed.
        assert!(
            !is_bit_or_sign_move(&op.body),
            "{rop:?}: a reduction's body is the EPILOGUE (identity `Reduced(0)`), \
             which must not read as a move — this is why routing on `plan.body` \
             is safe today"
        );
        assert!(
            is_bit_or_sign_move(&stage.pre),
            "{rop:?}: `pre` is `Input(0)` and DOES read as a move. If this ever \
             returns false the hazard is gone and the doc comment on \
             `is_bit_or_sign_move` should be updated rather than left claiming it"
        );
    }
}

/// `is_bit_move_reduce` discriminates the FOLD, which is the whole reason it
/// exists — `is_bit_or_sign_move` alone cannot, because `pre` is `Input(0)` for
/// every fold (KISS #416).
#[test]
fn the_reduce_predicate_separates_a_max_fold_from_a_sum_fold() {
    use unpopped::ir::{ReduceOp, input, is_bit_move_reduce, is_bit_or_sign_move};

    let pre = input(0).0;
    // The control: the expression half says "move" for every one of these.
    assert!(
        is_bit_or_sign_move(&pre),
        "control: `pre` must read as a move, or this test proves nothing about \
         the FOLD half"
    );

    for op in [ReduceOp::Max, ReduceOp::Min] {
        assert!(
            is_bit_move_reduce(op, &pre),
            "{op:?} folds by comparison and select, so the result IS one of the \
             input elements — a move under KISS-OPS-6.16-0009"
        );
    }
    for op in [ReduceOp::Sum, ReduceOp::Prod, ReduceOp::Mean] {
        assert!(
            !is_bit_move_reduce(op, &pre),
            "{op:?} produces a value that is not any input element, so it is \
             COMPUTED and must keep its rounding step (§6.16-0010). Routing it as \
             a move is the breakage this predicate exists to prevent"
        );
    }

    // And the expression half still has teeth: an arithmetic `pre` under a Max
    // fold is not a move, because the elements were computed before the fold.
    let arith = (input(0) + input(1)).0;
    assert!(
        !is_bit_move_reduce(ReduceOp::Max, &arith),
        "a Max fold over COMPUTED elements is not a bit move — both halves must \
         hold, and this is the half `matches!(op, ..)` alone would miss"
    );
}

/// The OTHER reduction shape, where `plan.body` IS the element expression — the
/// half the first version of this test did not cover, and the reason its doc
/// comment claimed a safety that does not hold generally.
///
/// `OpDef::reduction_axes` sets `body` to the per-element expr and keeps the
/// epilogue in `Access::Reduction::post`. `OpDef::row_reduce` does the opposite.
/// **There is no single field that is safe across both**, which is why
/// `is_bit_move_reduce` takes the fold explicitly.
#[test]
fn the_element_expression_lives_in_a_different_field_per_access_shape() {
    use unpopped::ir::{OpDef, ReduceOp, input, is_bit_move_reduce, is_bit_or_sign_move};
    use unpopped_vocab::ElementKind;

    for rop in [ReduceOp::Max, ReduceOp::Sum] {
        let op = OpDef::reduction("r", 1, &[ElementKind::F32], input(0), rop);
        assert!(
            is_bit_or_sign_move(&op.body),
            "{rop:?}: Access::Reduction carries the ELEMENT expr in `body`, so the \
             expression half reads TRUE here — for a Sum fold as much as a Max one. \
             This is exactly why `is_bit_or_sign_move` alone is not a safe guard"
        );
    }

    // The fold half is what separates them, and it must, since the expression
    // half cannot.
    let sum = OpDef::reduction("s", 1, &[ElementKind::F32], input(0), ReduceOp::Sum);
    assert!(
        !is_bit_move_reduce(ReduceOp::Sum, &sum.body),
        "a Sum fold over Access::Reduction must NOT read as a move — this is the \
         mis-route the predicate exists to prevent, on the shape where `body` is \
         the element expression"
    );
    let max = OpDef::reduction("m", 1, &[ElementKind::F32], input(0), ReduceOp::Max);
    assert!(
        is_bit_move_reduce(ReduceOp::Max, &max.body),
        "a Max fold over a moved element IS a move (KISS #416)"
    );
}

/// The truth table about which field holds a reduction's element expression
/// exists in exactly ONE place, and no doc restates it.
///
/// # Why a test reads prose here
///
/// The behavioural claim is already pinned by
/// `the_element_expression_lives_in_a_different_field_per_access_shape`. **That
/// test cannot read doc comments**, and the same false sentence survived a
/// correction precisely by living in prose on a neighbouring function — the
/// correct table was written once while the wrong instruction stayed on the
/// function callers actually read.
///
/// So this pins the DELETION, not the fact: a second copy is what produced the
/// defect, and a second copy is what this forbids.
#[test]
fn the_reduction_field_table_is_not_duplicated_in_prose() {
    let src = include_str!("../src/ir.rs");

    // The exact unqualified form that was false for `Access::Reduction`.
    assert!(
        !src.contains("which for a reduction is the *epilogue*"),
        "the unqualified 'plan.body is the epilogue' claim is back. It is true \
         for RowReduce and FALSE for Access::Reduction, where `body` IS the \
         element expression. Point at the table on `is_bit_or_sign_move` instead \
         of restating it"
    );

    // Exactly one table. Its header row is the marker.
    let tables = src.matches("element expr lives in").count();
    assert_eq!(
        tables, 1,
        "expected exactly ONE reduction-field table in ir.rs, found {tables}. Two \
         copies of this fact is how the correction landed in one doc while the \
         falsehood survived in the other"
    );

    // Control: the marker IS present, so a zero above would be a real finding
    // rather than a renamed table silently passing both assertions.
    assert!(
        src.contains("Access::RowReduce        Reduced(0)"),
        "control: the table itself must be present and spelled as expected, or \
         this test passes by finding nothing rather than by finding one"
    );

    // ⚠️ AND THE SAME CLAIM ESCAPED INTO A DOC THIS TEST DID NOT READ.
    //
    // Round 3 was two copies in `ir.rs`. Round 4 was `docs/normative-seams.md`
    // saying "`plan.body` is the epilogue (Reduced(0) -> false, safe)" — the
    // RowReduce-only claim, unqualified, in the file whose whole subject is
    // stating things accurately. A guard scoped to one file cannot see that.
    //
    // The invariant is not a phrase, because each round spelled it differently.
    // It is: you may not discuss what `plan.body` means at a reduction without
    // naming BOTH shapes, since every wrong version was true of exactly one.
    for (name, text) in [
        (
            "docs/normative-seams.md",
            include_str!("../../../docs/normative-seams.md"),
        ),
        ("src/ir.rs", src),
    ] {
        if !(text.contains("plan.body") && text.contains("epilogue")) {
            continue;
        }
        assert!(
            text.contains("Access::Reduction") && text.contains("Access::RowReduce"),
            "{name} discusses `plan.body` and `epilogue` together but does not name \
             BOTH access shapes. Every wrong version of this claim was true of one \
             shape and stated unqualified — naming both is what makes it checkable"
        );
    }
}

/// The four cases of KISS #416's attachment rule, and the one that was
/// inexpressible until `is_bit_move_fold_output` existed.
///
/// §6.16-0009 attaches to the value reaching the OBSERVABLE OUTPUT: trace fold →
/// output, and if every transformation is a move the whole is a move.
///
/// # ⚠️ These four rows are the CLAUSE's own enumeration, not mine
///
/// Read at KISS `origin/main` `3db1f994`, `spec/ops.md:1536`, KISS-OPS
/// §6.16-0011 — which does not merely state the rule, it names the
/// discriminating cases in both directions:
///
/// > *"trace the whole path from the op's inputs to that output, **the fold
/// > itself included** … An implementation MUST NOT classify such an op by its
/// > access variant, by its fold operator alone, or by its epilogue alone —
/// > **both** directions fail. A max-reduction under an arithmetic epilogue is a
/// > computed op, though its fold is a move; and a sum-reduction is a computed
/// > op even when its epilogue is a pure move."*
///
/// **The provenance is the point.** A reader cannot otherwise tell whether these
/// cases were derived from the clause or read off my own predicate — and a table
/// derived from the implementation it checks proves only that the implementation
/// is self-consistent. **KISS-CONSUME §8.2-0001: the question is provenance, not
/// equality.**
///
/// **Case 4 (`Sum` + pure-move post → COMPUTED) is the one to keep.** It is the
/// trap in the opposite direction, the clause spends a whole sentence on it, and
/// it is the row least likely to be written by anyone reasoning forward from
/// "which folds preserve bits". ⚠️ **A `pub` predicate answering the fold
/// question WITHOUT the post is what makes that row reachable as a mistake** —
/// see the note on `is_bit_move_reduce`'s exposure.
///
/// Measured 2026-09-06: baracuda's CUDA emitter — the only consumer — gates on
/// `is_bit_move_fold_output` and agrees with all four rows in code, while a
/// comment beside that gate described case 2 as deliberately unrouted. **The
/// clause says routing it is required, so the code conforms and the prose does
/// not.** The disagreement was in prose on both sides of the repo boundary and
/// no test on either side could see it.
#[test]
fn the_output_attachment_rule_covers_a_moving_post() {
    use unpopped::ir::{ReduceOp, UnaryOp, input, is_bit_move_fold_output, reduced};

    let elem = input(0).0;
    let identity = reduced(0).0;
    let moving = reduced(0).unary(UnaryOp::Neg).0;
    let arithmetic = reduced(0).unary(UnaryOp::Sqrt).0;

    // 1 — move fold, identity post.
    assert!(is_bit_move_fold_output(ReduceOp::Max, &elem, &identity));

    // 4 — move fold, MOVING post. THE CASE THAT WAS UNREACHABLE: an epilogue's
    // leaf is `Reduced(0)`, which the public predicate scores false, so
    // `Neg(Reduced(0))` was not merely unimplemented — it was inexpressible.
    assert!(
        is_bit_move_fold_output(ReduceOp::Max, &elem, &moving),
        "a sign edit applied to a moved fold result is still a move, and this is \
         the case the leaf policy exists for"
    );

    // 2 — move fold, ARITHMETIC post.
    assert!(
        !is_bit_move_fold_output(ReduceOp::Max, &elem, &arithmetic),
        "sqrt of the fold result computes, so §6.16-0010 governs the output no \
         matter how bit-preserving the fold was"
    );

    // 3 — arithmetic fold, any post.
    for post in [&identity, &moving] {
        assert!(
            !is_bit_move_fold_output(ReduceOp::Sum, &elem, post),
            "a Sum fold computes, so no post can make the output a move"
        );
    }

    // And the element half still has teeth.
    assert!(
        !is_bit_move_fold_output(ReduceOp::Max, &(input(0) + input(1)).0, &identity),
        "a Max fold over COMPUTED elements is not a move — all three halves must \
         hold, not two"
    );
}

/// Adding the leaf policy must NOT have changed the public predicate, because a
/// bare `Reduced(0)` reading `true` there is precisely the dangerous answer.
///
/// An identity epilogue IS bare `Reduced(0)`. If this ever returns true, a caller
/// checking only the epilogue routes an identity post over a SUM fold as a move.
#[test]
fn the_public_move_predicate_still_scores_a_fold_result_false() {
    use unpopped::ir::{UnaryOp, input, is_bit_or_sign_move, reduced};

    assert!(
        !is_bit_or_sign_move(&reduced(0).0),
        "a bare fold result must stay FALSE here — the safe answer. The leaf \
         policy lives in `is_bit_move_fold_output`, whose signature cannot \
         be satisfied without naming the fold"
    );
    assert!(!is_bit_or_sign_move(&reduced(0).unary(UnaryOp::Neg).0));
    // Control: the walk still works for the leaf it does admit.
    assert!(
        is_bit_or_sign_move(&input(0).unary(UnaryOp::Neg).0),
        "control: `Neg(Input(0))` must stay true, or the two assertions above \
         pass because the walk broke rather than because the policy holds"
    );
}
