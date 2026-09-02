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

use unpopped::ir::{BinaryOp, OpDef, input};
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
