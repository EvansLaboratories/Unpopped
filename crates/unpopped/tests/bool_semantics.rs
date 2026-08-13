//! `bool` is its own numeric kind, not a spelling of `u8`.
//!
//! KISS-CLASSIFY §6.1: a 1-byte truth value where `0` is false, **any non-zero
//! byte is true**, and **ops normalize to 0/1**. §6.2-0001 makes `bool` one of
//! the five numeric kinds in its own right. Same storage width as `u8`,
//! different semantics — which is the entire reason they are two dtypes.
//!
//! # The inversion this file pins
//!
//! Measured before any of it was written: `Add` at `Bool` **built a plan**, while
//! `LogicalAnd` was **refused** — with a message calling `Bool` "a float dtype",
//! because the int gate classified every non-int dtype as one. So the dtype
//! admitted arithmetic that is meaningless on a truth value (`true + true` is
//! `2`, not a value of the dtype) and refused the one op family it exists for.
//!
//! The bespoke logical surface had been pinned to `U8` — the *representation* —
//! and `Bool`, the dtype that surface is named after, fell through the wrong
//! branch.

use unpopped::cpu_c::CpuC;
use unpopped::ir::{BinaryOp, OpDef, input};
use unpopped::plan::try_build_plan;
use unpopped::try_generate;
use unpopped_vocab::{ArchSku, ElementKind, OpCategory, OperandDesc, structure_key};

const B: ElementKind = ElementKind::Bool;

fn cell(op: &OpDef) -> Result<String, String> {
    let d = OperandDesc::new(1, &[7], &[1], B, 1);
    let key = structure_key(OpCategory::BinaryElementwise, &[d, d, d], ArchSku::Sm89);
    try_build_plan(op, &key).map_err(|e| e.to_string())?;
    try_generate(op, &key, &CpuC)
        .map(|k| k.source)
        .map_err(|e| format!("{e:?}"))
}

/// The logical ops lower, and their C normalizes to 0/1.
#[test]
fn the_logical_ops_lower_and_normalize() {
    for (name, bop) in [
        ("and", BinaryOp::LogicalAnd),
        ("or", BinaryOp::LogicalOr),
        ("xor", BinaryOp::LogicalXor),
    ] {
        let op = OpDef::elementwise("l", 2, &[B], input(0).binary(bop, input(1)));
        let src = cell(&op).unwrap_or_else(|e| panic!("{name} at Bool must lower: {e}"));
        assert!(
            src.contains("unsigned char"),
            "{name}: bool stores as a byte:\n{src}"
        );
        assert!(
            src.contains("? 1 : 0"),
            "{name}: the result must be NORMALIZED to 0/1 (§6.1), not left as \
             whatever the comparison produced:\n{src}"
        );
        assert!(
            src.contains("!= 0"),
            "{name}: any NON-ZERO byte is true (§6.1), so operands must be tested \
             against zero rather than assumed to be exactly 1:\n{src}"
        );
    }
}

/// Arithmetic is refused, and the reason is the dtype's semantics rather than a
/// missing implementation.
///
/// `true + true` is `2`. Normalizing it silently would make `+` mean `or` — a
/// coincidence that holds at one addition and stops holding at three.
#[test]
fn arithmetic_is_refused_on_a_truth_value() {
    for (name, op) in [
        ("add", OpDef::elementwise("a", 2, &[B], input(0) + input(1))),
        ("sub", OpDef::elementwise("s", 2, &[B], input(0) - input(1))),
        ("mul", OpDef::elementwise("m", 2, &[B], input(0) * input(1))),
    ] {
        let err = cell(&op).expect_err(&format!("{name} at Bool must be refused"));
        assert!(
            err.contains("normalize") || err.contains("truth value"),
            "{name}: the refusal must cite the dtype's semantics, not a missing \
             lowering: {err}"
        );
    }
}

/// `Bool` and `U8` share a storage width and are still different dtypes.
///
/// Both spell `unsigned char`; only `Bool` refuses arithmetic. If the two ever
/// collapse, this is the test that says which property was lost.
#[test]
fn bool_and_u8_share_storage_but_not_semantics() {
    let add_u8 = OpDef::elementwise("a", 2, &[ElementKind::U8], input(0) + input(1));
    let d = OperandDesc::new(1, &[7], &[1], ElementKind::U8, 1);
    let key = structure_key(OpCategory::BinaryElementwise, &[d, d, d], ArchSku::Sm89);
    let u8_src = try_generate(&add_u8, &key, &CpuC)
        .expect("u8 add lowers")
        .source;
    assert!(u8_src.contains("unsigned char"), "u8 also stores as a byte");

    let add_bool = OpDef::elementwise("a", 2, &[B], input(0) + input(1));
    assert!(
        cell(&add_bool).is_err(),
        "the same op at Bool must be refused — same storage, different dtype"
    );
}
