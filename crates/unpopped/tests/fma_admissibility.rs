// SPDX-License-Identifier: MIT OR Apache-2.0
//! **`ScalarExpr::Fma` is admissible at f32/f32s/f64 and nowhere else.**
//!
//! A fused multiply-add rounds ONCE. At f16/bf16 the only spelling this crate has
//! is "widen to f32, fuse, round to half", which rounds twice, so the result is
//! not the fused one; at integer, complex and bool dtypes there is no
//! single-rounding meaning at all. The plan gate must say so as a typed
//! [`PlanError`], not by panicking and not by quietly lowering `Mul` then `Add`.
//!
//! The positive control is in this file too: the same op at f32 and f64 plans.
//! Without it, a gate that rejected *everything* would satisfy every negative
//! assertion below.

use unpopped::ir::{OpDef, ReduceOp, input};
use unpopped::plan::{PlanError, try_build_plan};
use unpopped_vocab::{ArchSku, ElementKind, OpCategory, OperandDesc, structure_key};

fn fma_op(dtype: ElementKind) -> OpDef {
    OpDef::elementwise("fma3", 3, &[dtype], input(0).fma(input(1), input(2)))
}

fn plan_at(op: &OpDef, dtype: ElementKind) -> Result<(), PlanError> {
    let d = OperandDesc::new(1, &[7], &[1], dtype, 256);
    let key = structure_key(OpCategory::BinaryElementwise, &[d, d, d, d], ArchSku::Sm89);
    try_build_plan(op, &key).map(|_| ())
}

#[test]
fn fma_plans_at_every_float_compute_dtype_it_claims() {
    for dt in [ElementKind::F32, ElementKind::F32Strict, ElementKind::F64] {
        plan_at(&fma_op(dt), dt).unwrap_or_else(|e| panic!("Fma must plan at {dt:?}: {e:?}"));
    }
}

#[test]
fn fma_misses_honestly_at_half_precision() {
    for dt in [ElementKind::F16, ElementKind::Bf16] {
        match plan_at(&fma_op(dt), dt) {
            Err(PlanError::InadmissibleOpAtDtype { dtype, detail }) => {
                assert_eq!(dtype, dt);
                assert!(
                    detail.contains("Fma"),
                    "the decline must name the construct: {detail}"
                );
            }
            other => panic!("Fma at {dt:?} must be InadmissibleOpAtDtype, got {other:?}"),
        }
    }
}

#[test]
fn fma_inside_a_reduction_pre_expression_is_gated_too() {
    // The half gate must walk the Access arm's own expressions, not only `body`:
    // a dot-product-shaped `sum(a*b + c)` is the place an FMA is most wanted.
    let dt = ElementKind::F16;
    let op = OpDef::reduction(
        "sum_fma",
        3,
        &[dt],
        input(0).fma(input(1), input(2)),
        ReduceOp::Sum,
    );
    let d = OperandDesc::new(2, &[4, 8], &[8, 1], dt, 256);
    let o = OperandDesc::new(1, &[4], &[1], dt, 256);
    let key = structure_key(OpCategory::Reduction, &[d, d, d, o], ArchSku::Sm89);
    match try_build_plan(&op, &key) {
        Err(PlanError::InadmissibleOpAtDtype { .. }) => {}
        other => panic!("Fma in a half reduction must be gated, got {other:?}"),
    }
}

#[test]
fn fma_has_no_integer_meaning() {
    for dt in [ElementKind::I32, ElementKind::U8] {
        assert!(
            plan_at(&fma_op(dt), dt).is_err(),
            "Fma at integer dtype {dt:?} must be rejected"
        );
    }
}
