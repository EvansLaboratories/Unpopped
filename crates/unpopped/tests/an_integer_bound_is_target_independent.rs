//! **A bitwise expression has no rounding step on any hardware, so its zero ULP
//! bound is not CUDA's to lend.**
//!
//! Raised by baracuda 2026-09-06 while adopting `unpopped 0.10.0`. `ulp_bound`
//! returned `INFINITY` for a non-CUDA target **before walking the expression**,
//! so a bitwise AND — exact on every target that has ever existed — reported an
//! unknown bound and `precision_of` downgraded it to `("approximate", None)`.
//!
//! ⚠️ **They applied this workspace's own rule: a prescription that errs
//! conservatively has no complainant.** A non-CUDA backend got a weaker contract
//! than it could prove, emitted nothing wrong, and had nothing to file. **They
//! found it only because a signature change forced them to read the function.**

use unpopped::contract::AccuracyKey;
use unpopped::contract::{precision_of, ulp_bound};
use unpopped::ir::{BinaryOp, ScalarExpr, UnaryOp, input};
use unpopped_vocab::{ArchSku, TargetId};

fn cuda() -> AccuracyKey {
    AccuracyKey::for_target(ArchSku::Sm89.into())
}

fn non_cuda() -> AccuracyKey {
    AccuracyKey::for_target(
        TargetId::parse("vulkan:sg64.ops-abr.arith-f16-i8.cm-none").expect("a valid target token"),
    )
}

fn bitwise() -> ScalarExpr {
    ScalarExpr::Binary(BinaryOp::BitAnd, Box::new(input(0).0), Box::new(input(1).0))
}

#[test]
fn an_integer_expression_keeps_its_zero_on_an_unmeasured_target() {
    assert_eq!(
        ulp_bound(&bitwise(), &non_cuda()),
        0.0,
        "BitAnd has no rounding step on any hardware -- its zero is by \
         construction, not borrowed from CUDA's table"
    );
    assert_eq!(
        ulp_bound(&bitwise(), &cuda()),
        0.0,
        "and the CUDA answer is unchanged, or this 'fix' moved the measured path"
    );
}

/// ⚠️ **THE DISCRIMINATING TEST.** `ulp_sum` rates `Sqrt` at 0.0 — but that zero
/// is an **IEEE claim about a target's float unit**, exactly the borrowed
/// assertion the namespace gate exists to refuse.
///
/// **If this ever returns 0.0, the exception was keyed on `ulp_sum(e) == 0`
/// instead of on `is_int_only`, and every float op rated exact has been
/// re-admitted through the gate.**
#[test]
fn a_float_op_rated_exact_is_still_declined_on_an_unmeasured_target() {
    let sqrt = ScalarExpr::Unary(UnaryOp::Sqrt, Box::new(input(0).0));

    // Control first: `ulp_sum` really does rate this 0 on CUDA, or the test
    // below passes for want of a zero rather than because the gate held.
    assert_eq!(
        ulp_bound(&sqrt, &cuda()),
        0.0,
        "control: Sqrt IS rated exact on the measured target -- without this, \
         the assertion below is trivially satisfied"
    );
    assert!(
        ulp_bound(&sqrt, &non_cuda()).is_infinite(),
        "Sqrt's zero asserts IEEE semantics without contraction, which an \
         unmeasured target has not told us. It must still decline"
    );

    // And a genuinely inexact op is declined too, so the gate is not simply
    // passing everything now.
    let exp = ScalarExpr::Unary(UnaryOp::Exp, Box::new(input(0).0));
    assert!(ulp_bound(&exp, &non_cuda()).is_infinite());
}

/// A mixed expression is only as exact as its worst operator.
#[test]
fn one_float_operator_forfeits_the_whole_expression() {
    let mixed = ScalarExpr::Binary(
        BinaryOp::BitAnd,
        Box::new(input(0).0),
        Box::new(ScalarExpr::Unary(UnaryOp::Exp, Box::new(input(1).0))),
    );
    assert!(
        ulp_bound(&mixed, &non_cuda()).is_infinite(),
        "the walk must reach EVERY node -- a predicate checking only the root \
         operator passes this"
    );
}

/// The consumer-visible effect, which is what baracuda actually observed.
#[test]
fn precision_of_now_rates_an_integer_op_correctly_on_an_unmeasured_target() {
    let (mode, ulp) = precision_of(&bitwise(), &non_cuda());
    assert_eq!(
        (mode, ulp),
        ("correctly_rounded", Some(0)),
        "an unmeasured target must not downgrade a bitwise op to approximate"
    );
}
