// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Typed expressions: `Cast` and `Bitcast`** (`docs/ir-typed-expressions.md`).
//!
//! Four things must hold, each with a control that proves the check can fail:
//!
//! 1. an expression without the new nodes types to the compute dtype everywhere, so
//!    every pre-existing op is unchanged (the back-compat obligation);
//! 2. the type rules and the conversion legality table accept exactly what they
//!    claim and name what they refuse;
//! 3. the optimizer leaves a typed expression exactly as authored, while still
//!    rewriting its untyped sibling (so "unchanged" is not "the optimizer is off");
//! 4. the plan gate admits the shapes v1 supports and refuses the rest as typed
//!    [`PlanError`]s, never a panic.

use unpopped::ir::{CastSpec, OpDef, ReduceOp, RoundMode, Saturation, ScalarExpr, input, konst};
use unpopped::optimize::optimize;
use unpopped::plan::{PlanError, try_build_plan};
use unpopped::types::{TypeError, check_bitcast, check_cast, infer, is_typed, rounding_matters};
use unpopped_vocab::{ArchSku, ElementKind, OpCategory, OperandDesc, structure_key};

const F32: ElementKind = ElementKind::F32;

fn cast(to: ElementKind) -> CastSpec {
    CastSpec {
        to,
        round: RoundMode::NearestEven,
        sat: Saturation::Wrap,
    }
}

// ---------------------------------------------------------------- 1. back-compat

#[test]
fn untyped_expressions_infer_to_the_compute_dtype_at_every_node() {
    // One of each pre-existing node kind, nested, at several compute dtypes.
    let e = (input(0) * konst(2.0) + input(1)).fma(input(2), input(0));
    let e = e.select(input(1) / input(2), input(0));
    for dt in [F32, ElementKind::F64, ElementKind::F16, ElementKind::I32] {
        assert!(!is_typed(&e.0));
        assert_eq!(infer(&e.0, dt), Ok(dt), "an untyped body is compute-typed");
    }
    // The control: a typed body is NOT compute-typed, so the assertion above can fail.
    let t = input(0).cast(ElementKind::I32, RoundMode::TowardZero, Saturation::Clamp);
    assert!(is_typed(&t.0));
    assert_eq!(infer(&t.0, F32), Ok(ElementKind::I32));
}

// ---------------------------------------------------------------- 2. rules

#[test]
fn operators_need_identical_operand_types_and_never_promote_implicitly() {
    let i = input(0).cast(ElementKind::I32, RoundMode::TowardZero, Saturation::Clamp);
    let sum = ScalarExpr::Add(Box::new(i.0.clone()), Box::new(input(1).0));
    assert_eq!(
        infer(&sum, F32),
        Err(TypeError::Mismatch {
            node: "Add",
            left: ElementKind::I32,
            right: F32
        })
    );
    // Control: two operands of the same type are fine.
    let ok = ScalarExpr::Add(Box::new(i.0.clone()), Box::new(i.0));
    assert_eq!(infer(&ok, F32), Ok(ElementKind::I32));
}

#[test]
fn the_conversion_table_accepts_what_it_claims() {
    let float = [F32, ElementKind::F32Strict, ElementKind::F64];
    let int = [
        ElementKind::I8,
        ElementKind::I16,
        ElementKind::I32,
        ElementKind::I64,
        ElementKind::U8,
        ElementKind::U16,
        ElementKind::U32,
    ];
    let spec = |to, sat| CastSpec {
        to,
        round: RoundMode::NearestEven,
        sat,
    };
    for &f in &float {
        for &t in &float {
            assert!(
                check_cast(f, spec(t, Saturation::Wrap)).is_ok(),
                "{f:?}->{t:?}"
            );
            assert!(check_cast(f, spec(t, Saturation::ClampFinite)).is_ok());
            assert!(
                check_cast(f, spec(t, Saturation::Clamp)).is_err(),
                "float->float Clamp"
            );
        }
        for &t in &int {
            assert!(
                check_cast(f, spec(t, Saturation::Clamp)).is_ok(),
                "{f:?}->{t:?}"
            );
            assert!(
                check_cast(f, spec(t, Saturation::Wrap)).is_err(),
                "float->int must be Clamp: C's out-of-range cast is undefined"
            );
        }
    }
    for &i in &int {
        for &t in &float {
            assert!(
                check_cast(i, spec(t, Saturation::Wrap)).is_ok(),
                "{i:?}->{t:?}"
            );
            assert!(check_cast(i, spec(t, Saturation::Clamp)).is_err());
        }
        for &t in &int {
            assert!(check_cast(i, spec(t, Saturation::Wrap)).is_ok());
            assert!(check_cast(i, spec(t, Saturation::Clamp)).is_ok());
            assert!(check_cast(i, spec(t, Saturation::ClampFinite)).is_err());
        }
    }
}

#[test]
fn endpoints_outside_the_table_are_refused_by_name() {
    for bad in [
        ElementKind::F16,
        ElementKind::Bf16,
        ElementKind::Fp8E4M3FN,
        ElementKind::Bool,
    ] {
        match check_cast(F32, cast(bad)) {
            Err(TypeError::IllegalCast { why, .. }) => {
                assert!(why.contains("outside"), "the reason must say why: {why}");
            }
            other => panic!("F32 -> {bad:?} must be IllegalCast, got {other:?}"),
        }
        assert!(check_cast(bad, cast(F32)).is_err());
        assert!(check_bitcast(bad, F32).is_err());
    }
}

#[test]
fn bitcast_needs_equal_widths() {
    assert!(check_bitcast(F32, ElementKind::I32).is_ok());
    assert!(check_bitcast(F32, ElementKind::U32).is_ok());
    assert!(check_bitcast(ElementKind::F64, ElementKind::I64).is_ok());
    assert!(check_bitcast(ElementKind::I16, ElementKind::U16).is_ok());
    assert!(check_bitcast(F32, ElementKind::I64).is_err());
    assert!(check_bitcast(ElementKind::I8, ElementKind::I16).is_err());
}

#[test]
fn rounding_matters_only_where_a_conversion_can_be_inexact() {
    assert!(
        !rounding_matters(F32, ElementKind::F64),
        "widening is exact"
    );
    assert!(rounding_matters(ElementKind::F64, F32), "narrowing rounds");
    assert!(
        rounding_matters(F32, ElementKind::I32),
        "float to int rounds"
    );
    assert!(
        !rounding_matters(ElementKind::I16, F32),
        "i16 fits f32's significand"
    );
    assert!(
        rounding_matters(ElementKind::I32, F32),
        "i32 does not fit f32's 24 bits"
    );
    assert!(
        !rounding_matters(ElementKind::I32, ElementKind::F64),
        "i32 fits f64's 53 bits"
    );
    assert!(rounding_matters(ElementKind::I64, ElementKind::F64));
    assert!(
        !rounding_matters(ElementKind::I32, ElementKind::I8),
        "int to int never rounds"
    );
}

// ---------------------------------------------------------------- 3. optimizer

#[test]
fn the_optimizer_leaves_a_typed_expression_exactly_as_authored() {
    // `x * 1.0` is rewritten to `x` in the untyped world ...
    let plain = input(0) * konst(1.0);
    assert_ne!(
        optimize(&plain.0, F32),
        plain.0,
        "control: the optimizer does rewrite x*1"
    );
    // ... and the identical shape under a Cast is not touched.
    let typed =
        (input(0) * konst(1.0)).cast(ElementKind::I32, RoundMode::TowardZero, Saturation::Clamp);
    assert_eq!(optimize(&typed.0, F32), typed.0);
    // Cast(Cast(x)) is lossy in general, so it is not collapsed.
    let double = input(0)
        .cast(ElementKind::I8, RoundMode::TowardZero, Saturation::Clamp)
        .cast(F32, RoundMode::NearestEven, Saturation::Wrap);
    assert_eq!(optimize(&double.0, F32), double.0);
    assert_eq!(
        optimize(&optimize(&double.0, F32), F32),
        double.0,
        "idempotent"
    );
}

// ---------------------------------------------------------------- 4. plan gate

fn plan(op: &OpDef, dtype: ElementKind, out: ElementKind) -> Result<(), PlanError> {
    let a = OperandDesc::new(1, &[7], &[1], dtype, 256);
    let o = OperandDesc::new(1, &[7], &[1], out, 256);
    let key = structure_key(OpCategory::UnaryElementwise, &[a, o], ArchSku::Sm89);
    try_build_plan(op, &key).map(|_| ())
}

fn to_i32() -> unpopped::ir::Expr {
    input(0).cast(ElementKind::I32, RoundMode::TowardZero, Saturation::Clamp)
}

#[test]
fn a_cast_root_with_a_matching_out_dtype_plans() {
    let mut op = OpDef::elementwise("to_i32", 1, &[F32], to_i32());
    op.out_dtype = Some(ElementKind::I32);
    plan(&op, F32, ElementKind::I32).expect("a typed root with the matching out_dtype must plan");
}

#[test]
fn a_store_type_mismatch_is_a_typed_decline_naming_both_types() {
    // out_dtype left at the compute dtype, but the body is I32.
    let op = OpDef::elementwise("to_i32_no_out", 1, &[F32], to_i32());
    match plan(&op, F32, F32) {
        Err(PlanError::InadmissibleOpAtDtype { detail, .. }) => {
            assert!(detail.contains("I32") && detail.contains("F32"), "{detail}");
        }
        other => panic!("expected a store-type decline, got {other:?}"),
    }
}

#[test]
fn a_typed_value_may_not_feed_arithmetic_in_v1() {
    let mut op = OpDef::elementwise(
        "i32_plus_one",
        1,
        &[F32],
        to_i32().cast(F32, RoundMode::NearestEven, Saturation::Wrap) + konst(1.0),
    );
    // Control: the round trip back to the compute type, then arithmetic, is fine.
    op.out_dtype = None;
    plan(&op, F32, F32).expect("Cast back to the compute type then arithmetic must plan");
    // The refused shape: arithmetic directly on the I32-typed value.
    let bad = OpDef::elementwise(
        "i32_arith",
        1,
        &[F32],
        unpopped::ir::Expr(ScalarExpr::Add(Box::new(to_i32().0), Box::new(to_i32().0))),
    );
    assert!(
        matches!(
            plan(&bad, F32, F32),
            Err(PlanError::InadmissibleOpAtDtype { .. })
        ),
        "arithmetic on a non-compute-typed value must be a typed decline"
    );
}

#[test]
fn only_f32_f32s_f64_compute_admits_typed_nodes() {
    let op = OpDef::elementwise("h", 1, &[ElementKind::F16], to_i32());
    match plan(&op, ElementKind::F16, ElementKind::F16) {
        Err(PlanError::InadmissibleOpAtDtype { dtype, .. }) => {
            assert_eq!(dtype, ElementKind::F16);
        }
        other => panic!("typed nodes at f16 compute must decline, got {other:?}"),
    }
}

#[test]
fn typed_nodes_are_elementwise_only_in_v1() {
    let op = OpDef::reduction("sum_cast", 1, &[F32], to_i32(), ReduceOp::Sum);
    let a = OperandDesc::new(2, &[4, 8], &[8, 1], F32, 256);
    let o = OperandDesc::new(1, &[4], &[1], F32, 256);
    let key = structure_key(OpCategory::Reduction, &[a, o], ArchSku::Sm89);
    assert!(
        matches!(
            try_build_plan(&op, &key),
            Err(PlanError::InadmissibleOpInCell { .. })
        ),
        "a Cast inside a reduction must be a typed decline"
    );
}
