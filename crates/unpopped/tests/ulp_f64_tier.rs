// SPDX-License-Identifier: MIT OR Apache-2.0
//! ROADMAP U42: the unary ULP rating at `F64` is `max(CUDA f32 count, the vendor's double-precision bound)`. Only `erfc` moves
//! (4 at `F32`, 5 at `F64`); every other op is unchanged at both dtypes (the `f32` count is already at or above the double bound,
//! and the bounds are "not guaranteed", so none is tightened).
//!
//! Vendor source: NVIDIA CUDA Programming Guide v13.5, "Mathematical Functions" appendix, tables 54-58 and 62, page dated
//! 2026-10-31. Both tables below are written out here independently of `contract.rs`.

use unpopped::contract::{AccuracyKey, ulp_bound, ulp_bound_measured_for};
use unpopped::ir::{ScalarExpr, UnaryOp, input};
use unpopped::oracle::required_fidelity;
use unpopped::{build_plan, ir::OpDef};
use unpopped_vocab::{ArchSku, ElementKind, OpCategory, OperandDesc, structure_key};

/// `(op, the f32 rating that was in the table before U42)`.
const F32_TABLE: [(UnaryOp, f64); 30] = [
    (UnaryOp::Neg, 0.0),
    (UnaryOp::Abs, 0.0),
    (UnaryOp::Sqr, 0.0),
    (UnaryOp::Sqrt, 0.0),
    (UnaryOp::Recip, 0.0),
    (UnaryOp::Relu, 0.0),
    (UnaryOp::Floor, 0.0),
    (UnaryOp::Ceil, 0.0),
    (UnaryOp::Round, 0.0),
    (UnaryOp::Sign, 0.0),
    (UnaryOp::Step, 0.0),
    (UnaryOp::Trunc, 0.0),
    (UnaryOp::Log, 1.0),
    (UnaryOp::Expm1, 1.0),
    (UnaryOp::Log1p, 1.0),
    (UnaryOp::Cbrt, 1.0),
    (UnaryOp::Log2, 1.0),
    (UnaryOp::Exp, 2.0),
    (UnaryOp::Tanh, 2.0),
    (UnaryOp::Erf, 2.0),
    (UnaryOp::Sin, 2.0),
    (UnaryOp::Cos, 2.0),
    (UnaryOp::Rsqrt, 2.0),
    (UnaryOp::Exp2, 2.0),
    (UnaryOp::Log10, 2.0),
    (UnaryOp::Cosh, 2.0),
    (UnaryOp::Atan, 2.0),
    (UnaryOp::Tan, 4.0),
    (UnaryOp::Erfc, 4.0),
    (UnaryOp::Lgamma, 6.0),
];

/// The vendor's `double` bound for the ops the page lists (verbatim from tables 54-58 and 62).
const VENDOR_F64: [(UnaryOp, f64); 23] = [
    (UnaryOp::Exp, 1.0),
    (UnaryOp::Expm1, 1.0),
    (UnaryOp::Log, 1.0),
    (UnaryOp::Log10, 1.0),
    (UnaryOp::Log2, 1.0),
    (UnaryOp::Log1p, 1.0),
    (UnaryOp::Cbrt, 1.0),
    (UnaryOp::Cosh, 1.0),
    (UnaryOp::Tanh, 1.0),
    (UnaryOp::Rsqrt, 1.0),
    (UnaryOp::Sin, 2.0),
    (UnaryOp::Cos, 2.0),
    (UnaryOp::Tan, 2.0),
    (UnaryOp::Asin, 2.0),
    (UnaryOp::Acos, 2.0),
    (UnaryOp::Atan, 2.0),
    (UnaryOp::Sinh, 2.0),
    (UnaryOp::Atanh, 2.0),
    (UnaryOp::Erf, 2.0),
    (UnaryOp::Asinh, 3.0),
    (UnaryOp::Acosh, 3.0),
    (UnaryOp::Lgamma, 4.0),
    (UnaryOp::Erfc, 5.0),
];

/// The f32 ratings of the ops the first table does not list (3 and 4 tiers).
const F32_REST: [(UnaryOp, f64); 5] = [
    (UnaryOp::Sigmoid, 3.0),
    (UnaryOp::Silu, 3.0),
    (UnaryOp::Gelu, 3.0),
    (UnaryOp::Sinh, 3.0),
    (UnaryOp::Acos, 3.0),
];
const F32_REST2: [(UnaryOp, f64); 4] = [
    (UnaryOp::Asinh, 3.0),
    (UnaryOp::Atanh, 3.0),
    (UnaryOp::Asin, 4.0),
    (UnaryOp::Acosh, 4.0),
];

fn key() -> AccuracyKey {
    AccuracyKey::for_target(ArchSku::Sm89.into())
}

fn at(op: UnaryOp, dt: ElementKind) -> f64 {
    let e = ScalarExpr::Unary(op, Box::new(input(0).0));
    ulp_bound_measured_for(&e, &key(), None, dt)
}

#[test]
fn only_erfc_moves_at_f64_and_every_other_op_is_unchanged_at_both_dtypes() {
    let all: Vec<(UnaryOp, f64)> = F32_TABLE
        .iter()
        .chain(&F32_REST)
        .chain(&F32_REST2)
        .copied()
        .collect();
    assert_eq!(all.len(), 39, "every unary op of this version is rated");
    for (op, f32_count) in all {
        let vendor = VENDOR_F64.iter().find(|v| v.0 == op).map(|v| v.1);
        let want_f64 = vendor.map_or(f32_count, |v| v.max(f32_count));
        assert_eq!(at(op, ElementKind::F32), f32_count, "{op:?} at F32");
        assert_eq!(at(op, ElementKind::F64), want_f64, "{op:?} at F64");
        // the dtype-less entry point is the f32 table
        let e = ScalarExpr::Unary(op, Box::new(input(0).0));
        assert_eq!(ulp_bound(&e, &key()), f32_count, "{op:?} via ulp_bound");
        if op == UnaryOp::Erfc {
            assert_eq!((f32_count, want_f64), (4.0, 5.0));
        } else {
            assert_eq!(want_f64, f32_count, "{op:?} must not move");
        }
    }
}

#[test]
fn the_f64_fidelity_band_of_erfc_uses_five_ulp_and_the_f32_band_four() {
    // the band as the oracle states it, read through its `Debug` text (this file hand-writes no tolerant band)
    let band = |dt: ElementKind| {
        let d = OperandDesc::new(
            1,
            &[4],
            &[1],
            dt,
            if dt == ElementKind::F64 { 8 } else { 4 },
        );
        let ops = vec![d, d];
        let def = OpDef::elementwise("e", 1, &[dt], input(0).unary(UnaryOp::Erfc));
        let k = structure_key(OpCategory::UnaryElementwise, &ops, ArchSku::Sm89);
        let plan = build_plan(&def, &k);
        format!("{:?}", required_fidelity(&plan, &ops).expect("a band"))
    };
    // rel = (arith steps (1) + 2 * ulp) * unit roundoff
    let want =
        |ulp: f64, u: f64| format!("Tolerant {{ rel: {:?}, abs: 0.0 }}", (1.0 + 2.0 * ulp) * u);
    assert_eq!(band(ElementKind::F64), want(5.0, 2f64.powi(-53)));
    assert_eq!(band(ElementKind::F32), want(4.0, 2f64.powi(-24)));
}
