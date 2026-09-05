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
