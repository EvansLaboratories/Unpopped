// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The oracle's `F64` `atanh` is accurate near `|x| = 1`, on both signs** (pm-kernels `[FINDING]` oracle-f64-atanh, from
//! openclownfish, 2026-11-19, confirmed by two methods and again here by a third: an 80-digit `Decimal`
//! `ln((1+x)/(1-x))/2` against Rust's own `f64::atanh`).
//!
//! `f64::atanh` is `0.5 * ln_1p(2x / (1 - x))`. For `x` near -1 the argument is near -1 and `ln_1p` cancels: 1290 ULP off at
//! `x = -0.99995695`, 37618 at `-0.999999`, while the same `|x|` positive was exact. Every emitter's ULP claim for `F64`
//! `atanh` is measured against this oracle, so the error was inherited. Fixed by evaluating on `|x|` and restoring the sign
//! (`atanh` is odd). The expected bits below are the correctly rounded `Decimal` values.

use unpopped::ir::{OpDef, UnaryOp, input};
use unpopped::oracle::{TypedBuffer, evaluate};
use unpopped::plan::try_build_plan;
use unpopped_vocab::{ArchSku, ElementKind, OpCategory, OperandDesc, structure_key};

const F64: ElementKind = ElementKind::F64;

fn atanh_of(xs: &[f64]) -> Vec<f64> {
    let op = OpDef::elementwise("atanh", 1, &[F64], input(0).unary(UnaryOp::Atanh));
    let n = xs.len() as i64;
    let bytes = xs.iter().flat_map(|v| v.to_le_bytes()).collect();
    let src = TypedBuffer::new(F64, vec![n], vec![1], bytes);
    let a = OperandDesc::new(1, &src.shape, &src.strides, F64, 256);
    let key = structure_key(OpCategory::UnaryElementwise, &[a, a], ArchSku::Sm89);
    let plan = try_build_plan(&op, &key).expect("plans");
    evaluate(&plan, &[a, a], std::slice::from_ref(&src), &[])
        .remove(0)
        .bytes
        .chunks(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn ulps(a: f64, b: f64) -> u64 {
    let key = |f: f64| {
        let i = f.to_bits() as i64;
        if i < 0 { i64::MIN - i } else { i }
    };
    key(a).abs_diff(key(b))
}

/// `(x, correctly rounded atanh(x))`, both signs of each magnitude.
const CASES: [(f64, u64); 16] = [
    (0.5, 0x3fe1_93ea_7aad_030b),
    (-0.5, 0xbfe1_93ea_7aad_030b),
    (0.9, 0x3ff7_8e36_0604_b32d),
    (-0.9, 0xbff7_8e36_0604_b32d),
    (0.999_956_949_180_724, 0x4015_7e15_2105_61a7),
    (-0.999_956_949_180_724, 0xc015_7e15_2105_61a7),
    (0.999_956_95, 0x4015_7e17_9f98_ae1b),
    (-0.999_956_95, 0xc015_7e17_9f98_ae1b),
    (0.999_999, 0x401d_046e_b8b8_6c1c),
    (-0.999_999, 0xc01d_046e_b8b8_6c1c),
    (0.999_999_999_9, 0x4027_b810_4136_e732),
    (-0.999_999_999_9, 0xc027_b810_4136_e732),
    (1e-5, 0x3ee4_f8b5_88e6_698e),
    (-1e-5, 0xbee4_f8b5_88e6_698e),
    (0.3, 0x3fd3_cf2b_5061_7c95),
    (-0.3, 0xbfd3_cf2b_5061_7c95),
];

#[test]
fn atanh_is_within_two_ulp_of_the_exact_value_on_both_signs() {
    let xs: Vec<f64> = CASES.iter().map(|c| c.0).collect();
    let got = atanh_of(&xs);
    for ((x, want), g) in CASES.iter().zip(got) {
        let want = f64::from_bits(*want);
        assert!(
            ulps(g, want) <= 2,
            "atanh({x}) = {g:?}, exact {want:?}, {} ulp",
            ulps(g, want)
        );
    }
}

#[test]
fn atanh_is_odd_bit_for_bit() {
    let mags = [0.25, 0.9, 0.999_956_95, 0.999_999, 0.999_999_999_9];
    let pos = atanh_of(&mags);
    let neg = atanh_of(&mags.map(|m| -m));
    for ((m, p), n) in mags.iter().zip(&pos).zip(&neg) {
        assert_eq!(p.to_bits() ^ n.to_bits(), 1 << 63, "atanh(+-{m})");
    }
}

#[test]
fn the_special_values_are_unchanged() {
    let got = atanh_of(&[0.0, -0.0, 1.0, -1.0, 2.0, -2.0, f64::NAN]);
    assert_eq!(got[0].to_bits(), 0.0f64.to_bits());
    assert_eq!(got[1].to_bits(), (-0.0f64).to_bits(), "atanh(-0) = -0");
    assert_eq!(got[2], f64::INFINITY);
    assert_eq!(got[3], f64::NEG_INFINITY);
    assert!(got[4].is_nan() && got[5].is_nan() && got[6].is_nan());
}
