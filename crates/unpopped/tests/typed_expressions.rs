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
    for bad in [ElementKind::Fp8E4M3FN, ElementKind::Bool] {
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
fn the_16_bit_floats_are_cast_endpoints_with_the_float_rules() {
    use ElementKind::{Bf16, F16, F64, I8, I32, U16};
    for half in [F16, Bf16] {
        for other in [F32, F64, I8, I32, U16, F16, Bf16] {
            let wrap = cast(half);
            // Wrap is the legal saturation into a float from a float or an integer.
            assert!(check_cast(other, wrap).is_ok(), "{other:?} -> {half:?}");
            if matches!(other, F32 | F64 | F16 | Bf16) {
                assert!(
                    check_cast(
                        other,
                        CastSpec {
                            sat: Saturation::ClampFinite,
                            ..wrap
                        }
                    )
                    .is_ok()
                );
                assert!(
                    check_cast(
                        other,
                        CastSpec {
                            sat: Saturation::Clamp,
                            ..wrap
                        }
                    )
                    .is_err()
                );
            }
        }
        // float to int saturates: Clamp only
        let to_int = |sat| CastSpec {
            to: I8,
            round: RoundMode::TowardZero,
            sat,
        };
        assert!(check_cast(half, to_int(Saturation::Clamp)).is_ok());
        assert!(check_cast(half, to_int(Saturation::Wrap)).is_err());
    }
    // A narrow float is an endpoint, never a compute dtype.
    let mut op = OpDef::elementwise(
        "f16_compute",
        1,
        &[F16],
        input(0).cast(F32, RoundMode::NearestEven, Saturation::Wrap),
    );
    op.out_dtype = Some(F32);
    assert!(
        plan(&op, F16, F32).is_err(),
        "F16 compute must still be refused by the typed gate"
    );
}

#[test]
fn rounding_matters_between_16_bit_floats_by_range_and_significand_not_width() {
    use ElementKind::{Bf16, F16, F64, I8, I16};
    assert!(!rounding_matters(F16, F32), "f16 widens to f32 exactly");
    assert!(!rounding_matters(Bf16, F32), "bf16 widens to f32 exactly");
    assert!(rounding_matters(F32, F16));
    assert!(rounding_matters(F64, Bf16));
    // Equal width, yet neither direction is exact: f16 has more precision, bf16 more range.
    assert!(
        rounding_matters(F16, Bf16),
        "f16 -> bf16 drops mantissa bits"
    );
    assert!(rounding_matters(Bf16, F16), "bf16 -> f16 drops range");
    assert!(!rounding_matters(F16, F16));
    assert!(
        !rounding_matters(I8, F16) && !rounding_matters(I8, Bf16),
        "i8 fits both"
    );
    assert!(rounding_matters(I16, F16), "i16 does not fit f16's 11 bits");
    assert!(rounding_matters(I16, Bf16));
}

#[test]
fn a_16_bit_float_bitcasts_to_the_integer_of_its_width_only() {
    use ElementKind::{Bf16, F16, I16, U16};
    for half in [F16, Bf16] {
        assert!(check_bitcast(half, U16).is_ok() && check_bitcast(I16, half).is_ok());
        assert!(check_bitcast(half, F32).is_err());
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
fn a_non_integer_typed_value_may_not_feed_arithmetic_in_v1() {
    let mut op = OpDef::elementwise(
        "i32_plus_one",
        1,
        &[F32],
        to_i32().cast(F32, RoundMode::NearestEven, Saturation::Wrap) + konst(1.0),
    );
    // Control: the round trip back to the compute type, then arithmetic, is fine.
    op.out_dtype = None;
    plan(&op, F32, F32).expect("Cast back to the compute type then arithmetic must plan");
    // The refused shape: arithmetic directly on a value typed f64 under f32 compute. (An
    // INTEGER-typed value is admitted by `integer_index_expressions.rs`: integer arithmetic
    // is the one typed arithmetic with a spelling.)
    let wide = || input(0).cast(ElementKind::F64, RoundMode::NearestEven, Saturation::Wrap);
    let bad = OpDef::elementwise(
        "f64_arith",
        1,
        &[F32],
        unpopped::ir::Expr(ScalarExpr::Add(Box::new(wide().0), Box::new(wide().0))),
    );
    assert!(
        matches!(
            plan(&bad, F32, F32),
            Err(PlanError::InadmissibleOpAtDtype { .. })
        ),
        "arithmetic on a non-compute-typed float value must be a typed decline"
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

// ---------------------------------------------------------------- 6. the oracle at 16 bits

fn run_cast(to: ElementKind, spec: CastSpec, xs: &[f32]) -> unpopped::oracle::TypedBuffer {
    use unpopped::oracle::{TypedBuffer, evaluate};
    let n = xs.len() as i64;
    let mut op = OpDef::elementwise(
        "to_half",
        1,
        &[F32],
        input(0).cast(to, spec.round, spec.sat),
    );
    op.out_dtype = Some(to);
    let a = OperandDesc::new(1, &[n], &[1], F32, 256);
    let o = OperandDesc::new(1, &[n], &[1], to, 256);
    let operands = vec![a, o];
    let key = structure_key(OpCategory::UnaryElementwise, &operands, ArchSku::Sm89);
    let plan = try_build_plan(&op, &key).expect("f32 to a 16-bit float must plan");
    evaluate(&plan, &operands, &[TypedBuffer::from_f32(&[n], xs)], &[])
        .into_iter()
        .next()
        .expect("one output")
}

#[test]
fn the_oracle_narrows_f32_to_f16_and_bf16_with_the_stated_rounding() {
    let xs = [
        1.0f32,
        65519.0,
        65520.0,
        1.0e10,
        -1.0e10,
        f32::INFINITY,
        f32::NAN,
        1.000_488_3,
    ];
    let nearest = run_cast(ElementKind::F16, cast(ElementKind::F16), &xs).to_f64_vec();
    assert_eq!(nearest[0], 1.0);
    assert_eq!(nearest[1], 65504.0, "just under the overflow midpoint");
    assert_eq!(
        nearest[2],
        f64::INFINITY,
        "the midpoint ties to even, which is infinity"
    );
    assert_eq!(nearest[3], f64::INFINITY);
    assert_eq!(
        nearest[5],
        f64::INFINITY,
        "an infinite input stays infinite"
    );
    assert!(nearest[6].is_nan());
    let clamped = CastSpec {
        sat: Saturation::ClampFinite,
        ..cast(ElementKind::F16)
    };
    let c = run_cast(ElementKind::F16, clamped, &xs).to_f64_vec();
    assert_eq!(&c[1..6], &[65504.0, 65504.0, 65504.0, -65504.0, 65504.0]);
    assert!(c[6].is_nan(), "ClampFinite leaves NaN");
    let down = CastSpec {
        round: RoundMode::TowardZero,
        ..cast(ElementKind::Bf16)
    };
    let up = CastSpec {
        round: RoundMode::TowardPosInf,
        ..cast(ElementKind::Bf16)
    };
    let (lo, hi) = (
        run_cast(ElementKind::Bf16, down, &[1.000_488_3]).to_f64_vec()[0],
        run_cast(ElementKind::Bf16, up, &[1.000_488_3]).to_f64_vec()[0],
    );
    assert_eq!(lo, 1.0);
    assert_eq!(
        hi,
        1.0 + 2f64.powi(-7),
        "directed rounding sees the dropped bits"
    );
}

#[test]
fn the_oracle_widens_a_16_bit_float_back_exactly_and_casts_it_to_an_integer() {
    use unpopped::oracle::{TypedBuffer, evaluate};
    for half in [ElementKind::F16, ElementKind::Bf16] {
        let mut op = OpDef::elementwise(
            "half_to_i8",
            1,
            &[F32],
            input(0)
                .cast(half, RoundMode::NearestEven, Saturation::Wrap)
                .cast(ElementKind::I8, RoundMode::TowardZero, Saturation::Clamp),
        );
        op.out_dtype = Some(ElementKind::I8);
        let n = 4;
        let a = OperandDesc::new(1, &[n], &[1], F32, 256);
        let o = OperandDesc::new(1, &[n], &[1], ElementKind::I8, 256);
        let operands = vec![a, o];
        let key = structure_key(OpCategory::UnaryElementwise, &operands, ArchSku::Sm89);
        // The root is I8; the inner F16 value is a typed non-compute value under a Cast.
        let Ok(plan) = try_build_plan(&op, &key) else {
            panic!("{half:?} under an integer Cast must plan");
        };
        let out = evaluate(
            &plan,
            &operands,
            &[TypedBuffer::from_f32(&[n], &[3.9, -3.9, 1000.0, f32::NAN])],
            &[],
        )
        .into_iter()
        .next()
        .expect("one output");
        assert_eq!(out.to_f64_vec(), vec![3.0, -3.0, 127.0, 0.0], "{half:?}");
    }
}
