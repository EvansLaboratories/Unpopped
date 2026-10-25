// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Integer index expressions: `CoordInt`, `ConstInt` and integer arithmetic**
//! (`docs/ir-typed-expressions.md` section 8).
//!
//! Each rule has a control that proves the check can fail:
//!
//! 1. the type rules: which leaves and operators exist at an integer type, and the typed
//!    error naming each refusal;
//! 2. an integer COMPUTE dtype is untouched by those refusals (the back-compat control);
//! 3. the optimizer returns an integer-typed body exactly as authored;
//! 4. the plan gate admits what v1 supports and declines the rest as typed errors;
//! 5. the oracle evaluates the integer operations from their definition, wrapping at the
//!    type's width, and `CoordInt` is the exact coordinate with no float round-trip.

use unpopped::ir::{
    BinaryOp, CastSpec, Expr, OpDef, RoundMode, Saturation, ScalarExpr, coord_int, input, konst,
    konst_int,
};
use unpopped::optimize::optimize;
use unpopped::oracle::{TypedBuffer, evaluate};
use unpopped::plan::{PlanError, build_plan, try_build_plan};
use unpopped::types::{TypeError, infer, int_spine_type, is_typed};
use unpopped_vocab::{ArchSku, ElementKind, OpCategory, OperandDesc, structure_key};

const F32: ElementKind = ElementKind::F32;
const I32: ElementKind = ElementKind::I32;
const I64: ElementKind = ElementKind::I64;
const U32: ElementKind = ElementKind::U32;

fn b(e: Expr) -> Box<ScalarExpr> {
    Box::new(e.0)
}

// ---------------------------------------------------------------- 1. the type rules

#[test]
fn integer_leaves_have_their_declared_type() {
    assert_eq!(infer(&coord_int(0, I32).0, F32), Ok(I32));
    assert_eq!(infer(&coord_int(1, U32).0, F32), Ok(U32));
    assert_eq!(infer(&konst_int(-5, I64).0, F32), Ok(I64));
    assert!(is_typed(&coord_int(0, I32).0) && is_typed(&konst_int(1, I32).0));
}

#[test]
fn an_integer_leaf_outside_the_index_types_or_range_is_a_typed_error() {
    for t in [F32, ElementKind::I8, ElementKind::U16, ElementKind::F64] {
        assert!(
            matches!(
                infer(&coord_int(0, t).0, F32),
                Err(TypeError::IllegalIntLeaf { ty, .. }) if ty == t
            ),
            "CoordInt({t:?}) must be refused"
        );
    }
    // The value must fit the type: -1 is not a U32, 2^31 is not an I32.
    for (v, t) in [(-1, U32), (1 << 31, I32), (1 << 32, U32)] {
        assert!(
            matches!(
                infer(&konst_int(v, t).0, F32),
                Err(TypeError::IllegalIntLeaf { .. })
            ),
            "ConstInt({v}, {t:?}) is out of range"
        );
    }
    // Controls: the extremes themselves are fine.
    for (v, t) in [
        (i64::from(i32::MAX), I32),
        (i64::from(i32::MIN), I32),
        (i64::from(u32::MAX), U32),
        (i64::MAX, I64),
        (i64::MIN, I64),
    ] {
        assert_eq!(infer(&konst_int(v, t).0, F32), Ok(t), "{v} fits {t:?}");
    }
}

#[test]
fn integer_add_sub_mul_keep_the_integer_type() {
    let c = coord_int(0, I32);
    let k = || konst_int(3, I32);
    for e in [c.clone() + k(), c.clone() - k(), c.clone() * k()] {
        assert_eq!(infer(&e.0, F32), Ok(I32));
    }
    // Min / Max keep it; a comparison is the COMPUTE type's 1/0.
    assert_eq!(infer(&c.clone().max(k()).0, F32), Ok(I32));
    assert_eq!(infer(&c.clone().min(k()).0, F32), Ok(I32));
    for cmp in [
        BinaryOp::CmpEq,
        BinaryOp::CmpNe,
        BinaryOp::CmpLt,
        BinaryOp::CmpLe,
        BinaryOp::CmpGt,
        BinaryOp::CmpGe,
    ] {
        assert_eq!(
            infer(&c.clone().binary(cmp, k()).0, F32),
            Ok(F32),
            "{cmp:?}"
        );
    }
}

#[test]
fn there_is_no_implicit_promotion_between_integer_and_float() {
    let sum = ScalarExpr::Add(b(coord_int(0, I32)), b(input(0)));
    assert_eq!(
        infer(&sum, F32),
        Err(TypeError::Mismatch {
            node: "Add",
            left: I32,
            right: F32
        })
    );
    // Nor between two integer types.
    let wide = ScalarExpr::Add(b(coord_int(0, I32)), b(konst_int(1, I64)));
    assert!(matches!(
        infer(&wide, F32),
        Err(TypeError::Mismatch { node: "Add", .. })
    ));
}

#[test]
fn operators_with_no_integer_form_refuse_an_integer_operand_by_name() {
    let c = || coord_int(0, I32);
    let k = || konst_int(2, I32);
    let cases: [(&str, ScalarExpr); 4] = [
        ("Div", ScalarExpr::Div(b(c()), b(k()))),
        ("Unary", c().sqrt().0),
        ("Fma", c().fma(k(), k()).0),
        ("Binary", c().binary(BinaryOp::Pow, k()).0),
    ];
    for (node, e) in cases {
        assert_eq!(
            infer(&e, F32),
            Err(TypeError::IntegerOperand { node, ty: I32 }),
            "{node}"
        );
    }
    // Controls: the same four over FLOATS are fine, so the refusals above are about the
    // operand type and not the operator.
    assert!(infer(&(input(0) / konst(2.0)).0, F32).is_ok());
    assert!(infer(&input(0).sqrt().0, F32).is_ok());
    assert!(infer(&input(0).fma(konst(2.0), konst(1.0)).0, F32).is_ok());
    assert!(infer(&input(0).binary(BinaryOp::Pow, konst(2.0)).0, F32).is_ok());
}

#[test]
fn select_takes_any_condition_and_needs_matching_integer_arms() {
    let c = || coord_int(0, I32);
    let k = || konst_int(7, I32);
    // An integer condition, a float condition and a comparison condition all pick an
    // integer arm.
    assert_eq!(infer(&c().select(k(), c()).0, F32), Ok(I32));
    assert_eq!(infer(&input(0).select(k(), c()).0, F32), Ok(I32));
    assert_eq!(
        infer(&c().binary(BinaryOp::CmpLt, k()).select(k(), c()).0, F32),
        Ok(I32)
    );
    // Arms of different types do not.
    assert!(matches!(
        infer(&c().select(k(), konst_int(7, I64)).0, F32),
        Err(TypeError::Mismatch { node: "Select", .. })
    ));
    assert!(matches!(
        infer(&c().select(k(), input(0)).0, F32),
        Err(TypeError::Mismatch { node: "Select", .. })
    ));
}

#[test]
fn the_spine_type_agrees_with_inference_on_well_typed_expressions() {
    let c = || coord_int(0, I32);
    let k = || konst_int(7, I32);
    let casted = input(0).cast(U32, RoundMode::TowardZero, Saturation::Clamp);
    let exprs: Vec<Expr> = vec![
        c(),
        k(),
        c() + k(),
        (c() * k() - k()).max(k()),
        c().binary(BinaryOp::CmpLt, k()).select(c(), k()),
        casted.clone() + konst_int(1, U32),
        casted.bitcast(I32) * k(),
    ];
    for e in exprs {
        assert_eq!(
            int_spine_type(&e.0),
            infer(&e.0, F32).ok(),
            "spine and inference disagree on {:?}",
            e.0
        );
    }
    // Controls: a float expression and a comparison have no integer spine type.
    assert_eq!(int_spine_type(&(input(0) + konst(1.0)).0), None);
    assert_eq!(
        int_spine_type(&c().binary(BinaryOp::CmpLt, k()).0),
        None,
        "a comparison is the compute type"
    );
}

// ---------------------------------------------------------------- 2. back-compat

#[test]
fn an_integer_compute_dtype_still_infers_every_operator() {
    // `Div`, `Unary`, `Fma` and `Pow` refuse an integer operand only when it differs from
    // the compute dtype; at an integer compute dtype the operand IS the compute type.
    for t in [I32, I64, U32] {
        assert_eq!(infer(&(input(0) / input(1)).0, t), Ok(t));
        assert_eq!(infer(&input(0).sqrt().0, t), Ok(t));
        assert_eq!(infer(&input(0).fma(input(1), input(0)).0, t), Ok(t));
        assert_eq!(
            infer(&input(0).binary(BinaryOp::CmpLt, input(1)).0, t),
            Ok(t)
        );
    }
}

// ---------------------------------------------------------------- 3. optimizer

#[test]
fn the_optimizer_returns_an_integer_body_exactly_as_authored() {
    // `x * 1` is rewritten for floats ...
    let plain = input(0) * konst(1.0);
    assert_ne!(
        optimize(&plain.0, F32),
        plain.0,
        "control: x*1 is rewritten"
    );
    // ... and the integer-typed twin is not (a rewrite could introduce a compute-typed
    // constant where an integer stood).
    let int = coord_int(0, I32) * konst_int(1, I32) + konst_int(0, I32);
    assert_eq!(optimize(&int.0, F32), int.0);
}

// ---------------------------------------------------------------- 4. plan gate

fn plan(op: &OpDef, out: ElementKind) -> Result<(), PlanError> {
    let a = OperandDesc::new(1, &[7], &[1], F32, 256);
    let o = OperandDesc::new(1, &[7], &[1], out, 256);
    let key = structure_key(OpCategory::UnaryElementwise, &[a, o], ArchSku::Sm89);
    try_build_plan(op, &key).map(|_| ())
}

fn int_op(name: &str, body: Expr, out: ElementKind) -> OpDef {
    let mut op = OpDef::elementwise(name, 1, &[F32], body);
    op.out_dtype = Some(out);
    op
}

#[test]
fn an_integer_index_body_plans_when_the_store_type_matches() {
    let body = coord_int(0, I32) * konst_int(3, I32) + konst_int(1, I32);
    plan(&int_op("idx", body.clone(), I32), I32).expect("CoordInt arithmetic must plan");
    // Control: the same body stored at the wrong type is a typed decline naming both.
    match plan(&OpDef::elementwise("idx_bad", 1, &[F32], body), F32) {
        Err(PlanError::InadmissibleOpAtDtype { detail, .. }) => {
            assert!(detail.contains("I32") && detail.contains("F32"), "{detail}");
        }
        other => panic!("expected a store-type decline, got {other:?}"),
    }
}

#[test]
fn integer_arithmetic_on_a_cast_input_plans() {
    let c = input(0).cast(U32, RoundMode::TowardZero, Saturation::Clamp);
    let body = (c + konst_int(5, U32)).max(konst_int(9, U32));
    plan(&int_op("u32_chain", body, U32), U32).expect("integer arithmetic on a Cast must plan");
}

#[test]
fn an_ill_typed_integer_body_is_a_typed_decline_never_a_panic() {
    // Div has no integer form.
    let div = ScalarExpr::Div(b(coord_int(0, I32)), b(konst_int(2, I32)));
    let op = int_op("idiv", Expr(div), I32);
    assert!(matches!(
        plan(&op, I32),
        Err(PlanError::InadmissibleOpAtDtype { .. })
    ));
    // Mixed int and float arithmetic.
    let mixed = ScalarExpr::Add(b(coord_int(0, I32)), b(input(0)));
    assert!(matches!(
        plan(&int_op("mixed", Expr(mixed), I32), I32),
        Err(PlanError::InadmissibleOpAtDtype { .. })
    ));
    // An out-of-range constant.
    let bad = konst_int(1 << 40, I32) + konst_int(1, I32);
    assert!(matches!(
        plan(&int_op("range", bad, I32), I32),
        Err(PlanError::InadmissibleOpAtDtype { .. })
    ));
}

#[test]
fn an_integer_leaf_is_refused_at_non_float_compute_and_outside_elementwise() {
    // Half compute.
    let op = int_op("h", konst_int(1, I32) + konst_int(2, I32), I32);
    let a = OperandDesc::new(1, &[7], &[1], ElementKind::F16, 256);
    let o = OperandDesc::new(1, &[7], &[1], I32, 256);
    let key = structure_key(OpCategory::UnaryElementwise, &[a, o], ArchSku::Sm89);
    assert!(matches!(
        try_build_plan(&op, &key),
        Err(PlanError::InadmissibleOpAtDtype { .. })
    ));
    // Reduction epilogue.
    let red = OpDef::reduction(
        "red",
        1,
        &[F32],
        input(0) + konst_int(1, I32).cast(F32, RoundMode::NearestEven, Saturation::Wrap),
        unpopped::ir::ReduceOp::Sum,
    );
    let a = OperandDesc::new(2, &[4, 8], &[8, 1], F32, 256);
    let o = OperandDesc::new(1, &[4], &[1], F32, 256);
    let key = structure_key(OpCategory::Reduction, &[a, o], ArchSku::Sm89);
    assert!(matches!(
        try_build_plan(&red, &key),
        Err(PlanError::InadmissibleOpInCell { .. })
    ));
}

#[test]
fn coord_int_axis_must_be_inside_the_rank() {
    let op = int_op("axis", coord_int(3, I32) + konst_int(1, I32), I32);
    assert!(matches!(
        plan(&op, I32),
        Err(PlanError::InadmissibleOpInCell { .. })
    ));
    // Control: axis 0 of a rank-1 cell is in range.
    plan(
        &int_op("axis0", coord_int(0, I32) + konst_int(1, I32), I32),
        I32,
    )
    .expect("axis 0");
}

// ---------------------------------------------------------------- 5. the oracle

fn run(op: &OpDef, n: i64, out: ElementKind, xs: &[f32]) -> TypedBuffer {
    let a = OperandDesc::new(1, &[n], &[1], F32, 256);
    let o = OperandDesc::new(1, &[n], &[1], out, 256);
    let operands = vec![a, o];
    let key = structure_key(OpCategory::UnaryElementwise, &operands, ArchSku::Sm89);
    let plan = build_plan(op, &key);
    evaluate(&plan, &operands, &[TypedBuffer::from_f32(&[n], xs)], &[])
        .into_iter()
        .next()
        .expect("one output")
}

#[test]
fn coord_int_is_the_exact_coordinate_with_no_float_round_trip() {
    let n = 9;
    let op = int_op(
        "affine_index",
        coord_int(0, I32) * konst_int(3, I32) + konst_int(1, I32),
        I32,
    );
    let got = run(&op, n, I32, &vec![0.0; n as usize]);
    for i in 0..n as usize {
        assert_eq!(
            got.bits_at(i) as u32 as i32,
            3 * i as i32 + 1,
            "element {i}"
        );
    }
}

#[test]
fn integer_arithmetic_wraps_at_the_types_width() {
    let xs = [0.0_f32; 4];
    let wrap = |body: Expr, t: ElementKind| -> u128 {
        run(&int_op("wrap", body, t), 4, t, &xs).bits_at(0)
    };
    // i32::MAX + 1 wraps to i32::MIN; a signed overflow must not be undefined here.
    assert_eq!(
        wrap(konst_int(i64::from(i32::MAX), I32) + konst_int(1, I32), I32) as u32,
        i32::MIN as u32
    );
    // u32 0 - 1 wraps to u32::MAX (zero-extended, not read as -1).
    assert_eq!(
        wrap(konst_int(0, U32) - konst_int(1, U32), U32) as u32,
        u32::MAX
    );
    // i64::MIN * -1 wraps to i64::MIN.
    assert_eq!(
        wrap(konst_int(i64::MIN, I64) * konst_int(-1, I64), I64) as u64,
        i64::MIN as u64
    );
    // Control: a non-overflowing sum is the sum.
    assert_eq!(wrap(konst_int(40, I32) + konst_int(2, I32), I32) as u32, 42);
}

#[test]
fn comparison_and_select_over_integers_follow_the_exact_values() {
    // Select(coord < 4, coord * 10, -1) over eight elements: the condition is a float 1/0
    // from an INTEGER comparison, the arms are integers.
    let cond = coord_int(0, I32).binary(BinaryOp::CmpLt, konst_int(4, I32));
    let body = cond.select(coord_int(0, I32) * konst_int(10, I32), konst_int(-1, I32));
    let got = run(&int_op("sel", body, I32), 8, I32, &[0.0; 8]);
    for i in 0..8usize {
        let want: i32 = if i < 4 { 10 * i as i32 } else { -1 };
        assert_eq!(got.bits_at(i) as u32 as i32, want, "element {i}");
    }
    // An integer condition is tested != 0: coord - 3 is zero only at element 3.
    let body = (coord_int(0, I32) - konst_int(3, I32)).select(konst_int(1, I32), konst_int(0, I32));
    let got = run(&int_op("sel_int", body, I32), 8, I32, &[0.0; 8]);
    for i in 0..8usize {
        assert_eq!(
            got.bits_at(i) as u32 as i32,
            i32::from(i != 3),
            "element {i}"
        );
    }
}

#[test]
fn a_cast_spec_round_trip_through_an_integer_body_keeps_the_value() {
    // float -> i32 (saturating) -> + 1 -> f32: the integer step is exact.
    let body = (input(0).cast(I32, RoundMode::TowardZero, Saturation::Clamp) + konst_int(1, I32))
        .cast(F32, RoundMode::NearestEven, Saturation::Wrap);
    let op = OpDef::elementwise("rt", 1, &[F32], body);
    let _ = CastSpec {
        to: F32,
        round: RoundMode::NearestEven,
        sat: Saturation::Wrap,
    };
    let got = run(&op, 4, F32, &[1.9, -1.9, 2.0e9, f32::NAN]);
    let f = |i: usize| f32::from_bits(got.bits_at(i) as u32);
    assert_eq!(f(0), 2.0);
    assert_eq!(f(1), 0.0);
    // 2.0e9 + 1 as an i32 is 2000000001, which rounds to f32 2.0e9 + 64 or the nearest.
    assert_eq!(f(2), (2_000_000_001_i64 as f32));
    assert_eq!(f(3), 1.0, "NaN saturates to 0, then + 1");
}

// ------------------------------------- 6. an emitter that skips the plan gate (Pip, #130)

/// `lower_expr` on its own, with an `int_op` seam that records the call: the shape of an
/// out-of-tree emitter that never ran `build_plan`. A mixed integer/float node is ill-typed,
/// so it must decline BY NAME rather than reach `arith`/`select` (or `int_op` at the wrong
/// type) and be spelled.
#[test]
fn a_mixed_integer_and_float_node_declines_even_when_the_plan_gate_was_skipped() {
    use unpopped::backend::{Decline, DeclinedOp, Lowering, Spelling, lower_expr};
    let leaf = |i: u8| Ok(Spelling::Spelled(format!("x{i}")));
    let un = |_o, x: String| Ok(Spelling::Spelled(x));
    let bin = |_o, a: String, b: String| Ok(Spelling::Spelled(format!("bin({a},{b})")));
    let ci = |d: u8, _t| Ok(Spelling::Spelled(format!("c{d}")));
    let io = |op, t, v: Vec<String>| {
        Ok(Spelling::Spelled(format!(
            "int{op:?}{t:?}({})",
            v.join(",")
        )))
    };
    let lo = Lowering::builder(&leaf, &un, &bin)
        .compute_dtype(F32)
        .coord_int(&ci)
        .int_op(&io)
        .build();
    let int = || coord_int(0, I32);
    let mixed = [
        ("int + float", int() + input(0)),
        ("float + int", input(0) + int()),
        ("int * float", int() * input(0)),
        ("select(int, float)", input(1).select(int(), input(0))),
        ("select(float, int)", input(1).select(input(0), int())),
        (
            "max(float, int)",
            Expr(ScalarExpr::Binary(BinaryOp::Max, b(input(0)), b(int()))),
        ),
        (
            "cmp(int, float)",
            Expr(ScalarExpr::Binary(BinaryOp::CmpLt, b(int()), b(input(0)))),
        ),
    ];
    for (name, e) in mixed {
        match lower_expr(&e.0, &lo).expect("a decline is a value, not an error") {
            Spelling::Declined(Decline::UnsupportedOp { op, why }) => {
                assert_eq!(op, DeclinedOp::Integer, "{name}");
                assert!(why.contains("mixed"), "{name}: {why}");
            }
            other => panic!("{name}: expected a decline, got {other:?}"),
        }
    }
    // Control: the same operators over matching integers are spelled through `int_op`.
    for e in [
        int() + int(),
        input(1).select(int(), int()),
        Expr(ScalarExpr::Binary(BinaryOp::CmpLt, b(int()), b(int()))),
    ] {
        let got = lower_expr(&e.0, &lo).unwrap();
        assert!(
            matches!(&got, Spelling::Spelled(s) if s.starts_with("int")),
            "{got:?}"
        );
    }
    // Control: an all-float node is still spelled by the default infix `arith`.
    let f = lower_expr(&(input(0) + input(1)).0, &lo).unwrap();
    assert!(
        matches!(&f, Spelling::Spelled(s) if s == "(x0 + x1)"),
        "{f:?}"
    );
}
