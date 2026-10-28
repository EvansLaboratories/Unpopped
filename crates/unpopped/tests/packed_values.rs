// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Packed values: `Pack` and `Extract`** (`docs/ir-typed-expressions.md` section 9).
//!
//! The type rules, with a control for each refusal: `infer_ty` accepts exactly the
//! shapes the doc names, and `infer` (which returns a bare `ElementKind`) refuses a
//! packed result rather than reporting one lane's type.

use unpopped::ir::{ScalarExpr, input};
use unpopped::types::{Ty, TypeError, infer, infer_ty};
use unpopped_vocab::ElementKind;

const F32: ElementKind = ElementKind::F32;

fn pack(xs: Vec<ScalarExpr>) -> ScalarExpr {
    ScalarExpr::Pack(xs)
}

fn extract(x: ScalarExpr, i: u8) -> ScalarExpr {
    ScalarExpr::Extract(Box::new(x), i)
}

#[test]
fn a_pack_of_two_or_four_scalars_is_packed_of_their_type() {
    let two = pack(vec![input(0).0, input(1).0]);
    assert_eq!(infer_ty(&two, F32), Ok(Ty::Packed(F32, 2)));
    let four = pack(vec![input(0).0, input(1).0, input(0).0, input(1).0]);
    assert_eq!(infer_ty(&four, F32), Ok(Ty::Packed(F32, 4)));
}

#[test]
fn a_scalar_expression_has_the_scalar_ty() {
    // Control: infer_ty agrees with infer where there is no packed value.
    assert_eq!(infer_ty(&input(0).0, F32), Ok(Ty::Scalar(F32)));
}

#[test]
fn lane_counts_other_than_two_and_four_are_refused() {
    for n in [0usize, 1, 3, 5] {
        let p = pack((0..n).map(|_| input(0).0).collect());
        assert!(
            matches!(infer_ty(&p, F32), Err(TypeError::BadLanes { lanes }) if usize::from(lanes) == n.min(255)),
            "{n} lanes"
        );
    }
}

#[test]
fn pack_children_must_share_one_scalar_type() {
    let mixed = pack(vec![input(0).0, ScalarExpr::ConstInt(1, ElementKind::I32)]);
    assert!(matches!(
        infer_ty(&mixed, F32),
        Err(TypeError::Mismatch { node: "Pack", .. })
    ));
    // Nested packs are not defined: a lane is a scalar.
    let nested = pack(vec![pack(vec![input(0).0, input(1).0]), input(0).0]);
    assert!(matches!(
        infer_ty(&nested, F32),
        Err(TypeError::PackedOperand { node: "Pack" })
    ));
}

#[test]
fn extract_returns_the_lane_type_and_checks_the_index() {
    let p = pack(vec![input(0).0, input(1).0]);
    assert_eq!(infer_ty(&extract(p.clone(), 1), F32), Ok(Ty::Scalar(F32)));
    assert!(matches!(
        infer_ty(&extract(p, 2), F32),
        Err(TypeError::BadLane { index: 2, lanes: 2 })
    ));
    // Extract of a scalar is not an Extract.
    assert!(matches!(
        infer_ty(&extract(input(0).0, 0), F32),
        Err(TypeError::NotPacked { node: "Extract" })
    ));
}

#[test]
fn extract_of_pack_is_the_lane_so_a_packed_body_can_be_scalar_again() {
    let p = pack(vec![input(0).0, input(1).0]);
    assert_eq!(infer(&extract(p, 0), F32), Ok(F32));
}

#[test]
fn infer_refuses_a_packed_result() {
    let p = pack(vec![input(0).0, input(1).0]);
    assert_eq!(
        infer(&p, F32),
        Err(TypeError::PackedResult { lanes: 2 }),
        "a bare ElementKind cannot describe a packed value"
    );
}

#[test]
fn operators_with_no_packed_form_refuse_a_packed_operand() {
    let p = pack(vec![input(0).0, input(1).0]);
    let e = ScalarExpr::Div(Box::new(p.clone()), Box::new(p));
    assert!(matches!(
        infer_ty(&e, F32),
        Err(TypeError::PackedOperand { node: "Div" })
    ));
}

// ---------------------------------------------------------------- the carriers

use unpopped::ir::{ExprDag, OpDef};
use unpopped::optimize::optimize;
use unpopped::plan::{PlanError, try_build_plan};
use unpopped::types::{has_packed, is_typed};
use unpopped::{op_from_text, op_to_text};
use unpopped_vocab::{ArchSku, OpCategory, OperandDesc, structure_key};

fn packed_body() -> ScalarExpr {
    extract(pack(vec![input(0).0, ScalarExpr::Const(2.0)]), 0)
}

#[test]
fn packed_nodes_are_typed_and_found_by_has_packed() {
    let b = packed_body();
    assert!(is_typed(&b) && has_packed(&b));
    // Control: a plain body is neither.
    assert!(!is_typed(&input(0).0) && !has_packed(&input(0).0));
}

#[test]
fn the_dag_round_trips_a_packed_body_and_interns_equal_packs_once() {
    let b = packed_body();
    assert_eq!(ExprDag::from_expr(&b).to_expr(), b);
    // Two equal Packs are one node: Add(Extract(P,0), Extract(P,1)) has P once.
    let p = pack(vec![input(0).0, input(1).0]);
    let e = ScalarExpr::Add(Box::new(extract(p.clone(), 0)), Box::new(extract(p, 1)));
    let dag = ExprDag::from_expr(&e);
    let packs = (0..dag.len() as u32)
        .filter(|&i| matches!(dag.node(i), unpopped::ir::DagNode::Pack(_)))
        .count();
    assert_eq!(packs, 1);
}

#[test]
fn the_text_format_round_trips_a_packed_body() {
    let op = OpDef::elementwise("packed", 1, &[F32], unpopped::ir::Expr(packed_body()));
    assert_eq!(op_from_text(&op_to_text(&op)).expect("parses"), op);
}

#[test]
fn the_optimizer_leaves_a_packed_body_as_authored() {
    // The same shape unpacked IS rewritten (x * 1 -> x), so "unchanged" is not "off".
    let times_one = ScalarExpr::Mul(Box::new(input(0).0), Box::new(ScalarExpr::Const(1.0)));
    assert_ne!(optimize(&times_one, F32), times_one);
    let packed = extract(pack(vec![times_one.clone(), input(1).0]), 0);
    assert_eq!(optimize(&packed, F32), packed);
}

#[test]
fn the_plan_gate_declines_a_packed_body_with_a_typed_error() {
    let op = OpDef::elementwise("packed", 1, &[F32], unpopped::ir::Expr(packed_body()));
    let a = OperandDesc::new(1, &[7], &[1], F32, 256);
    let o = OperandDesc::new(1, &[7], &[1], F32, 256);
    let key = structure_key(OpCategory::UnaryElementwise, &[a, o], ArchSku::Sm89);
    assert!(
        matches!(
            try_build_plan(&op, &key),
            Err(PlanError::InadmissibleOpInCell { .. } | PlanError::InadmissibleOpAtDtype { .. })
        ),
        "no emitter has a packed seam yet, so the gate declines: {:?}",
        try_build_plan(&op, &key).map(|_| ())
    );
    // Control: the same op without the pack plans.
    let plain = OpDef::elementwise("plain", 1, &[F32], input(0) * unpopped::ir::konst(2.0));
    assert!(try_build_plan(&plain, &key).is_ok());
}

// ---------------------------------------------------------------- lane-wise operators (step 2)

use unpopped::ir::{BinaryOp, CastSpec, RoundMode, Saturation};
use unpopped::types::lane_of;

fn p2(a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    pack(vec![a, b])
}

fn bx(e: ScalarExpr) -> Box<ScalarExpr> {
    Box::new(e)
}

#[test]
fn lane_wise_arithmetic_keeps_the_packed_type() {
    let a = p2(input(0).0, input(1).0);
    let b = p2(input(1).0, input(0).0);
    for e in [
        ScalarExpr::Add(bx(a.clone()), bx(b.clone())),
        ScalarExpr::Sub(bx(a.clone()), bx(b.clone())),
        ScalarExpr::Mul(bx(a.clone()), bx(b.clone())),
        ScalarExpr::Binary(BinaryOp::Max, bx(a.clone()), bx(b.clone())),
        ScalarExpr::Binary(BinaryOp::Min, bx(a.clone()), bx(b.clone())),
        ScalarExpr::Fma(bx(a.clone()), bx(b.clone()), bx(a.clone())),
    ] {
        assert_eq!(infer_ty(&e, F32), Ok(Ty::Packed(F32, 2)), "{e:?}");
    }
}

#[test]
fn a_packed_operand_beside_a_scalar_or_a_different_width_is_refused() {
    let a = p2(input(0).0, input(1).0);
    let four = pack(vec![input(0).0, input(1).0, input(0).0, input(1).0]);
    // No broadcast: a scalar beside a packed value is a typed error, never implicit.
    assert!(matches!(
        infer_ty(&ScalarExpr::Add(bx(a.clone()), bx(input(0).0)), F32),
        Err(TypeError::PackedOperand { node: "Add" })
    ));
    assert!(matches!(
        infer_ty(&ScalarExpr::Add(bx(a.clone()), bx(four)), F32),
        Err(TypeError::PackedOperand { node: "Add" })
    ));
    // Different lane element types.
    let ints = p2(
        ScalarExpr::ConstInt(1, ElementKind::I32),
        ScalarExpr::ConstInt(2, ElementKind::I32),
    );
    assert!(matches!(
        infer_ty(&ScalarExpr::Add(bx(a), bx(ints)), F32),
        Err(TypeError::Mismatch { node: "Add", .. })
    ));
}

#[test]
fn operators_with_no_packed_form_still_decline() {
    let a = p2(input(0).0, input(1).0);
    for (node, e) in [
        ("Div", ScalarExpr::Div(bx(a.clone()), bx(a.clone()))),
        (
            "Unary",
            ScalarExpr::Unary(unpopped::ir::UnaryOp::Neg, bx(a.clone())),
        ),
        (
            "Binary",
            ScalarExpr::Binary(BinaryOp::Pow, bx(a.clone()), bx(a.clone())),
        ),
    ] {
        assert!(
            matches!(infer_ty(&e, F32), Err(TypeError::PackedOperand { node: n }) if n == node),
            "{node}"
        );
    }
}

#[test]
fn integer_lane_arithmetic_follows_the_scalar_integer_rules() {
    let i = |v| ScalarExpr::ConstInt(v, ElementKind::I32);
    let ints = p2(i(1), i(2));
    let sum = ScalarExpr::Add(bx(ints.clone()), bx(ints.clone()));
    assert_eq!(infer_ty(&sum, F32), Ok(Ty::Packed(ElementKind::I32, 2)));
    // Integer Div has no form, packed or not.
    assert!(infer_ty(&ScalarExpr::Div(bx(ints.clone()), bx(ints)), F32).is_err());
}

#[test]
fn a_whole_value_select_takes_a_scalar_condition_and_packed_arms() {
    let a = p2(input(0).0, input(1).0);
    let b = p2(input(1).0, input(0).0);
    let sel = ScalarExpr::Select(bx(input(0).0), bx(a.clone()), bx(b.clone()));
    assert_eq!(infer_ty(&sel, F32), Ok(Ty::Packed(F32, 2)));
    // A packed condition is a later slice.
    let bad = ScalarExpr::Select(bx(a.clone()), bx(a), bx(b));
    assert!(matches!(
        infer_ty(&bad, F32),
        Err(TypeError::PackedOperand { node: "Select" })
    ));
}

#[test]
fn cast_converts_each_lane() {
    let a = p2(input(0).0, input(1).0);
    let spec = CastSpec {
        to: ElementKind::F64,
        round: RoundMode::NearestEven,
        sat: Saturation::Wrap,
    };
    assert_eq!(
        infer_ty(&ScalarExpr::Cast(bx(a), spec), F32),
        Ok(Ty::Packed(ElementKind::F64, 2))
    );
}

#[test]
fn extract_through_an_operator_is_that_lane_of_the_operands() {
    let a = p2(input(0).0, ScalarExpr::Const(2.0));
    let b = p2(input(1).0, ScalarExpr::Const(3.0));
    let sum = ScalarExpr::Add(bx(a.clone()), bx(b.clone()));
    assert_eq!(
        lane_of(&sum, 1),
        Ok(ScalarExpr::Add(
            bx(ScalarExpr::Const(2.0)),
            bx(ScalarExpr::Const(3.0))
        ))
    );
    assert_eq!(
        lane_of(&sum, 0),
        Ok(ScalarExpr::Add(bx(input(0).0), bx(input(1).0)))
    );
    // Typed: Extract(Add(a, b), i) infers to the lane type.
    assert_eq!(infer_ty(&extract(sum, 1), F32), Ok(Ty::Scalar(F32)));
    // Out-of-range and non-lane-wise are errors, not panics.
    assert!(lane_of(&a, 2).is_err());
    assert!(lane_of(&ScalarExpr::Div(bx(a.clone()), bx(a)), 0).is_err());
}

#[test]
fn lane_of_distributes_every_lane_wise_operator() {
    let a = p2(input(0).0, ScalarExpr::Const(2.0));
    let b = p2(input(1).0, ScalarExpr::Const(3.0));
    let c = p2(ScalarExpr::Const(5.0), input(0).0);
    let (a0, b0, c0) = (input(0).0, input(1).0, ScalarExpr::Const(5.0));
    let spec = CastSpec {
        to: ElementKind::F64,
        round: RoundMode::NearestEven,
        sat: Saturation::Wrap,
    };
    let cases = [
        (
            ScalarExpr::Sub(bx(a.clone()), bx(b.clone())),
            ScalarExpr::Sub(bx(a0.clone()), bx(b0.clone())),
        ),
        (
            ScalarExpr::Mul(bx(a.clone()), bx(b.clone())),
            ScalarExpr::Mul(bx(a0.clone()), bx(b0.clone())),
        ),
        (
            ScalarExpr::Binary(BinaryOp::Max, bx(a.clone()), bx(b.clone())),
            ScalarExpr::Binary(BinaryOp::Max, bx(a0.clone()), bx(b0.clone())),
        ),
        (
            ScalarExpr::Binary(BinaryOp::Min, bx(a.clone()), bx(b.clone())),
            ScalarExpr::Binary(BinaryOp::Min, bx(a0.clone()), bx(b0.clone())),
        ),
        (
            ScalarExpr::Fma(bx(a.clone()), bx(b.clone()), bx(c.clone())),
            ScalarExpr::Fma(bx(a0.clone()), bx(b0.clone()), bx(c0)),
        ),
        (
            // The condition is a whole-value scalar and is not lane-selected.
            ScalarExpr::Select(bx(input(1).0), bx(a.clone()), bx(b.clone())),
            ScalarExpr::Select(bx(input(1).0), bx(a0.clone()), bx(b0)),
        ),
        (
            ScalarExpr::Cast(bx(a), spec),
            ScalarExpr::Cast(bx(a0), spec),
        ),
    ];
    for (packed, lane0) in cases {
        assert_eq!(lane_of(&packed, 0), Ok(lane0), "{packed:?}");
    }
}

/// A lane of integer type `t` (integer leaves are I32/I64/U32, so narrower lanes are cast).
fn ci(v: i64, t: ElementKind) -> ScalarExpr {
    let spec = CastSpec {
        to: t,
        round: RoundMode::NearestEven,
        sat: Saturation::Wrap,
    };
    ScalarExpr::Cast(bx(ScalarExpr::ConstInt(v, ElementKind::I32)), spec)
}

fn bitcast(x: ScalarExpr, to: ElementKind) -> ScalarExpr {
    ScalarExpr::Bitcast(Box::new(x), to)
}

#[test]
fn bitcast_of_a_packed_value_to_a_scalar_of_equal_total_bits_is_that_scalar() {
    let i8x4 = pack((0..4).map(|v| ci(v, ElementKind::I8)).collect());
    assert_eq!(
        infer_ty(&bitcast(i8x4, ElementKind::U32), F32),
        Ok(Ty::Scalar(ElementKind::U32))
    );
    let i16x2 = pack(vec![ci(1, ElementKind::I16), ci(2, ElementKind::I16)]);
    assert_eq!(
        infer_ty(&bitcast(i16x2, ElementKind::I32), F32),
        Ok(Ty::Scalar(ElementKind::I32))
    );
    let f32x2 = pack(vec![input(0).0, input(1).0]);
    assert_eq!(
        infer_ty(&bitcast(f32x2, ElementKind::I64), F32),
        Ok(Ty::Scalar(ElementKind::I64))
    );
}

#[test]
fn bitcast_of_a_packed_value_refuses_a_width_mismatch() {
    // Control for the test above: the same shapes, one width off.
    let i8x4 = pack((0..4).map(|v| ci(v, ElementKind::I8)).collect());
    for to in [ElementKind::U16, ElementKind::I64] {
        assert!(
            matches!(
                infer_ty(&bitcast(i8x4.clone(), to), F32),
                Err(TypeError::IllegalBitcast { .. })
            ),
            "{to:?}"
        );
    }
}
