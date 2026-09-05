//! KISS-OPS-6.16-0009: a narrow-float body that only MOVES must not round.
//!
//! # The defect this pins, as it shipped
//!
//! `unpopped-cpu-c 0.5.0` emitted, for `max(in0, in1)` at `f8e5m2`:
//!
//! ```text
//! out[i] = unpopped_f8e5m2_store(
//!   load(in0[i]) != load(in0[i]) ? load(in0[i]) : ( ... ) );
//! ```
//!
//! The body between the codecs is `a != a ? a : (b != b ? b : (a >= b ? a : b))`
//! — **comparison and select, no arithmetic at all**, the §6.13 decomposition
//! emitted literally. And it was still wrapped in `_store(...)`: **a rounding
//! applied to a value that was selected, never computed.**
//!
//! The harm is broader than "a signalling NaN is quieted". The E5M2 store codec
//! is `if (x != x) { return 0x7F; }`, and E5M2 has several NaN encodings —
//! `0x7D`, `0x7E`, `0x7F` and the negatives. **They all collapse to one
//! constant, so a moved QUIET NaN loses its payload and sign too**, where
//! §6.8-0010(a) requires the moved operand's bits exactly.
//!
//! # ⚠️ Why the fix is not "stop rounding narrow floats"
//!
//! **KISS-OPS-6.16-0010 requires arithmetic to quiet a signalling NaN.** Removing
//! the round-trip globally would satisfy 0009 and *break* 0010. The decomposition
//! decides, per body — which is what `unpopped::ir::is_bit_move` computes, and
//! why the second half of this file matters as much as the first.

use unpopped::ir::{BinaryOp, OpDef, UnaryOp, input};
use unpopped::try_generate;
use unpopped_cpu_c::CpuC;
use unpopped_vocab::{ArchSku, ElementKind, OpCategory, OperandDesc, structure_key};

/// 7 elements at 4-byte alignment elects the scalar schedule. A divisible extent
/// at a wide alignment picks a vectorized one and every assertion below reports
/// `UnsupportedSchedule` instead — green for the wrong reason.
fn emit(op: &OpDef, dt: ElementKind, n_in: usize) -> String {
    let d = OperandDesc::new(1, &[7], &[1], dt, 4);
    let mut ops: Vec<OperandDesc> = vec![d; n_in];
    ops.push(d);
    let k = structure_key(OpCategory::BinaryElementwise, &ops, ArchSku::Sm89);
    try_generate(op, &k, &CpuC)
        .expect("this corpus lowers")
        .source
}

fn body(src: &str) -> String {
    src.lines()
        .filter(|l| l.contains("tmp") || l.contains("out["))
        .map(str::trim)
        .collect::<Vec<_>>()
        .join(" ")
}

#[test]
fn a_pure_move_selects_raw_storage_and_never_re_encodes() {
    for dt in [ElementKind::Fp8E5M2, ElementKind::Fp8E4M3FN] {
        let tag = if dt == ElementKind::Fp8E5M2 {
            "f8e5m2"
        } else {
            "f8e4m3fn"
        };
        let op = OpDef::elementwise("mx", 2, &[dt], input(0).binary(BinaryOp::Max, input(1)));
        let b = body(&emit(&op, dt, 2));

        assert!(
            !b.contains(&format!("unpopped_{tag}_store")),
            "{dt:?}: a moved operand was RE-ENCODED — this is the 6.16-0009 \
             defect exactly:\n{b}"
        );
        // The selected values are the raw stored bytes, not decoded ones.
        assert!(
            b.contains("? in0[i] :") && b.contains(": in1[i])"),
            "{dt:?}: the arms are not raw storage operands:\n{b}"
        );
        // The comparison still decodes — a bitwise compare would order NaN and
        // signed zero wrongly, so the promote must survive on the TEST side.
        assert!(
            b.contains(&format!("unpopped_{tag}_load(in0[i]) >=")),
            "{dt:?}: the comparison is not performed on decoded values:\n{b}"
        );
        // And the temporary carries the STORAGE type, not `float`.
        assert!(
            b.contains("unsigned char tmp0"),
            "{dt:?}: the hoisted temporary is not the storage type:\n{b}"
        );
    }
}

/// ⚠️ The arm that keeps 0009's fix from breaking 0010.
///
/// A body containing arithmetic must STILL round-trip, because arithmetic is
/// required to quiet a signalling NaN. A fix that removed the round-trip globally
/// passes the test above and silently violates the complementary clause.
#[test]
fn a_body_containing_arithmetic_still_rounds() {
    let dt = ElementKind::Fp8E5M2;
    for (nm, op, n) in [
        (
            "max feeding an add",
            OpDef::elementwise(
                "ma",
                2,
                &[dt],
                input(0).binary(BinaryOp::Max, input(1)) + input(0),
            ),
            2,
        ),
        (
            "plain add",
            OpDef::elementwise("ad", 2, &[dt], input(0) + input(1)),
            2,
        ),
    ] {
        let b = body(&emit(&op, dt, n));
        assert!(
            b.contains("unpopped_f8e5m2_store"),
            "{nm}: arithmetic must still round-trip — removing it globally would \
             break KISS-OPS-6.16-0010:\n{b}"
        );
    }
}

/// A wide float is untouched by any of this: nothing to round in the first place.
#[test]
fn a_wide_float_move_is_unaffected() {
    let dt = ElementKind::F32;
    let op = OpDef::elementwise("mx", 2, &[dt], input(0).binary(BinaryOp::Max, input(1)));
    let b = body(&emit(&op, dt, 2));
    assert!(
        !b.contains("_store"),
        "f32 gained a codec it never had:\n{b}"
    );
    assert!(
        b.contains("in0[i] >= in1[i]") || b.contains("in0[i] != in0[i]"),
        "f32 max is no longer the plain compare-select:\n{b}"
    );
}

/// `neg`/`abs`/`copysign` edit ONE BIT and compute nothing, so a store codec has
/// nothing to round — and applying one destroys what the op is defined to do.
///
/// # The defect, as it shipped
///
/// `is_bit_move` admitted `Input`/`Max`/`Min`/`Select` — the **no-sign-edit**
/// subset — so these three fell through to the arithmetic path and emitted
/// `store(op(load(x)))`. The E5M2 store codec is `if (x != x) { return 0x7F; }`
/// and E5M2 has **six** NaN encodings, so `neg(NaN)` returned a fixed constant:
/// the sign flip was discarded along with the payload.
///
/// `f8e4m3fn` is the sharper case and the reason this is not merely about
/// payloads. It has **two** NaN encodings, `0x7F` and `0xFF`, so a NaN's SIGN is
/// representable in it — and `neg` losing that is a loss of information the
/// format can hold. (KISS `#402` corrects §6.16-0004, whose "a single NaN
/// encoding" wording invited exactly this reading; OCP OFP8 spells e4m3fn NaN as
/// `S.1111.111`, with the sign free.)
#[test]
fn a_sign_edit_is_a_mask_and_never_a_codec_round_trip() {
    for dt in [ElementKind::Fp8E5M2, ElementKind::Fp8E4M3FN] {
        let cases = [
            (
                "neg",
                OpDef::elementwise("n", 1, &[dt], input(0).unary(UnaryOp::Neg)),
                1,
                "^ 0x80u",
            ),
            (
                "abs",
                OpDef::elementwise("a", 1, &[dt], input(0).unary(UnaryOp::Abs)),
                1,
                "& 0x7Fu",
            ),
            (
                "copysign",
                OpDef::elementwise("c", 2, &[dt], input(0).binary(BinaryOp::Copysign, input(1))),
                2,
                "| ((in1[i]) & 0x80u)",
            ),
        ];
        for (nm, op, n, mask) in cases {
            let b = body(&emit(&op, dt, n));
            assert!(
                b.contains(mask),
                "{dt:?} {nm}: expected the byte mask `{mask}`, which is exact for \
                 every input including NaN:\n{b}"
            );
            // The load codec is what promotes to f32; its absence is what makes
            // the mask exact. Asserted separately from the store because a body
            // that loaded and then masked would be wrong in a subtler way.
            assert!(
                !b.contains("_load("),
                "{dt:?} {nm}: promoted through f32, so a NaN reaches the store \
                 codec and collapses:\n{b}"
            );
            assert!(
                !b.contains("_store("),
                "{dt:?} {nm}: re-encoded a value that was never computed — the \
                 exact step KISS-OPS-6.16-0009 forbids:\n{b}"
            );
        }
    }
}

/// The complement, and the reason the predicate is a predicate rather than a
/// blanket rule: a sign edit FEEDING arithmetic is arithmetic, and must round.
///
/// Without this, "stop rounding sign edits" would satisfy §6.16-0009 by breaking
/// §6.16-0010, which requires arithmetic to quiet a signalling NaN.
#[test]
fn a_sign_edit_feeding_arithmetic_still_rounds() {
    let dt = ElementKind::Fp8E5M2;
    let op = OpDef::elementwise("na", 2, &[dt], input(0).unary(UnaryOp::Neg) + input(1));
    let b = body(&emit(&op, dt, 2));
    assert!(
        b.contains("unpopped_f8e5m2_store"),
        "`neg(a) + b` computes, so it must round-trip — the sign edit does not \
         make the addition a move:\n{b}"
    );
}

/// A sign edit nested INSIDE a move still bypasses: `max(neg(a), b)` yields one
/// operand's bytes, possibly sign-flipped, and computes nothing.
///
/// This is the composition case, and it is why the predicate recurses through
/// `Max`/`Min`/`Select` arms rather than only matching at the root.
#[test]
fn a_sign_edit_inside_a_move_is_still_a_move() {
    let dt = ElementKind::Fp8E5M2;
    let op = OpDef::elementwise(
        "mn",
        2,
        &[dt],
        input(0).unary(UnaryOp::Neg).binary(BinaryOp::Max, input(1)),
    );
    let b = body(&emit(&op, dt, 2));
    assert!(
        !b.contains("_store("),
        "`max(neg(a), b)` selects bytes and computes nothing, so it must not \
         re-encode:\n{b}"
    );
    assert!(
        b.contains("^ 0x80u"),
        "the negated arm must still be spelled as a mask:\n{b}"
    );
}
