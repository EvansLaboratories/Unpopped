// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The layers agree** (`docs/ir-kernel-layer.md` sections 5 and 7): an Elementwise `OpDef`
//! lifted into the kernel layer and run by the kernel interpreter produces, byte for byte, what
//! `oracle::evaluate` produces for the `OpDef`.
//!
//! Special values (signed zero, NaN, infinities, subnormals, extremes) are in every corpus: an
//! agreement over friendly numbers would pass a lift that dropped them. A broadcast input
//! (row stride 0) proves the view survives behind `Place::Logical`.

use unpopped::ir::{BinaryOp, Expr, OpDef, ScalarExpr, UnaryOp, input};
use unpopped::kernel::check_kernel;
use unpopped::kernel_lift::{LiftDecline, lift_elementwise_opdef};
use unpopped::oracle::kernel_eval::{KernelArg, run_kernel};
use unpopped::oracle::{TypedBuffer, evaluate};
use unpopped::plan::try_build_plan;
use unpopped_vocab::{ArchSku, ElementKind, OpCategory, OperandDesc, structure_key};

const ROWS: i64 = 3;
const COLS: i64 = 5;

fn bx(e: ScalarExpr) -> Box<ScalarExpr> {
    Box::new(e)
}

fn un(op: UnaryOp) -> ScalarExpr {
    ScalarExpr::Unary(op, bx(input(0).0))
}

fn bin(op: BinaryOp) -> ScalarExpr {
    ScalarExpr::Binary(op, bx(input(0).0), bx(input(1).0))
}

fn f32_bytes(vs: &[f32]) -> Vec<u8> {
    vs.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn f64_bytes(vs: &[f64]) -> Vec<u8> {
    vs.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn i32_bytes(vs: &[i32]) -> Vec<u8> {
    vs.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// 15 f32 values with the awkward ones in.
fn f32_corpus(seed: usize) -> Vec<f32> {
    let base = [
        0.0,
        -0.0,
        1.0,
        -1.0,
        0.5,
        2.0,
        -2.5,
        f32::MIN_POSITIVE,
        1.0e-40,
        3.0e38,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        0.1,
        -7.25,
    ];
    (0..15).map(|i| base[(i + seed) % base.len()]).collect()
}

/// Run `op` through the oracle and through the lifted kernel; return both output byte images.
fn both(
    op: &OpDef,
    cat: OpCategory,
    dtype: ElementKind,
    inputs: &[TypedBuffer],
) -> (Vec<u8>, Vec<u8>) {
    let rank = 2usize;
    let desc = |tb: &TypedBuffer| OperandDesc::new(rank, &tb.shape, &tb.strides, dtype, 256);
    let out_shape = [ROWS, COLS];
    let out_strides = [COLS, 1];
    let out_desc = OperandDesc::new(rank, &out_shape, &out_strides, dtype, 256);
    let mut descs: Vec<OperandDesc> = inputs.iter().map(desc).collect();
    descs.push(out_desc);
    let key = structure_key(cat, &descs, ArchSku::Sm89);
    let plan = try_build_plan(op, &key).expect("the op plans");
    let want = evaluate(&plan, &descs, inputs, &[]).remove(0).bytes;

    let kernel = lift_elementwise_opdef(op, dtype, 2).expect("the op lifts");
    assert_eq!(
        check_kernel(&kernel),
        Ok(()),
        "a lifted kernel is well formed"
    );
    let n_bytes = (ROWS * COLS) as usize * 16;
    let mut args: Vec<KernelArg> = inputs.iter().cloned().map(KernelArg::Buffer).collect();
    args.push(KernelArg::Buffer(TypedBuffer::new(
        dtype,
        out_shape.to_vec(),
        out_strides.to_vec(),
        vec![0u8; n_bytes],
    )));
    args.push(KernelArg::Int(i128::from(ROWS)));
    args.push(KernelArg::Int(i128::from(COLS)));
    run_kernel(&kernel, &mut args).expect("the kernel runs");
    let KernelArg::Buffer(out) = &args[inputs.len()] else {
        unreachable!()
    };
    let size = want.len();
    (want, out.bytes[..size].to_vec())
}

fn dense_f32(seed: usize) -> TypedBuffer {
    TypedBuffer::new(
        ElementKind::F32,
        vec![ROWS, COLS],
        vec![COLS, 1],
        f32_bytes(&f32_corpus(seed)),
    )
}

/// A row vector broadcast down the rows (row stride 0).
fn broadcast_f32(seed: usize) -> TypedBuffer {
    TypedBuffer::new(
        ElementKind::F32,
        vec![ROWS, COLS],
        vec![0, 1],
        f32_bytes(&f32_corpus(seed)[..COLS as usize]),
    )
}

#[test]
fn every_unary_op_agrees_on_f32_special_values() {
    let ops = [
        UnaryOp::Neg,
        UnaryOp::Abs,
        UnaryOp::Sqr,
        UnaryOp::Sqrt,
        UnaryOp::Rsqrt,
        UnaryOp::Recip,
        UnaryOp::Exp,
        UnaryOp::Log,
        UnaryOp::Tanh,
        UnaryOp::Sigmoid,
        UnaryOp::Relu,
        UnaryOp::Erf,
    ];
    for u in ops {
        let op = OpDef::elementwise("u", 1, &[ElementKind::F32], Expr(un(u)));
        let (want, got) = both(
            &op,
            OpCategory::UnaryElementwise,
            ElementKind::F32,
            &[dense_f32(0)],
        );
        assert_eq!(got, want, "{u:?}");
    }
}

#[test]
fn every_binary_op_agrees_on_f32_special_values_with_a_broadcast_input() {
    let ops = [
        BinaryOp::Max,
        BinaryOp::Min,
        BinaryOp::Pow,
        BinaryOp::Rem,
        BinaryOp::Atan2,
        BinaryOp::Copysign,
        BinaryOp::Nextafter,
        BinaryOp::FmaxIeee,
        BinaryOp::FminIeee,
        BinaryOp::CmpEq,
        BinaryOp::CmpNe,
        BinaryOp::CmpLt,
        BinaryOp::CmpLe,
    ];
    for b in ops {
        let op = OpDef::elementwise("b", 2, &[ElementKind::F32], Expr(bin(b)));
        let (want, got) = both(
            &op,
            OpCategory::BinaryElementwise,
            ElementKind::F32,
            &[dense_f32(3), broadcast_f32(7)],
        );
        assert_eq!(got, want, "{b:?}");
    }
}

#[test]
fn arithmetic_select_fma_and_a_cast_agree() {
    let a = || input(0).0;
    let b = || input(1).0;
    let bodies: Vec<(&str, ScalarExpr)> = vec![
        ("add", ScalarExpr::Add(bx(a()), bx(b()))),
        ("sub", ScalarExpr::Sub(bx(a()), bx(b()))),
        ("mul", ScalarExpr::Mul(bx(a()), bx(b()))),
        ("div", ScalarExpr::Div(bx(a()), bx(b()))),
        (
            "fma",
            ScalarExpr::Fma(bx(a()), bx(b()), bx(ScalarExpr::Const(0.1))),
        ),
        (
            "select",
            ScalarExpr::Select(
                bx(ScalarExpr::Binary(BinaryOp::CmpLt, bx(a()), bx(b()))),
                bx(a()),
                bx(b()),
            ),
        ),
        (
            "const mix",
            ScalarExpr::Add(
                bx(ScalarExpr::Const(1.5)),
                bx(ScalarExpr::Mul(bx(a()), bx(ScalarExpr::Const(-0.0)))),
            ),
        ),
    ];
    for (name, body) in bodies {
        let op = OpDef::elementwise(name, 2, &[ElementKind::F32], Expr(body));
        let (want, got) = both(
            &op,
            OpCategory::BinaryElementwise,
            ElementKind::F32,
            &[dense_f32(1), broadcast_f32(4)],
        );
        assert_eq!(got, want, "{name}");
    }
}

#[test]
fn f64_compute_agrees() {
    let vals: Vec<f64> = vec![
        0.0,
        -0.0,
        1.0,
        -1.0,
        1.0e-310,
        1.7e308,
        f64::INFINITY,
        f64::NAN,
        0.1,
        -7.25,
        3.0,
        1.0e-5,
        2.5,
        -2.5,
        123456.789,
    ];
    let tb = TypedBuffer::new(
        ElementKind::F64,
        vec![ROWS, COLS],
        vec![COLS, 1],
        f64_bytes(&vals),
    );
    let op = OpDef::elementwise(
        "f64",
        1,
        &[ElementKind::F64],
        Expr(ScalarExpr::Mul(
            bx(un(UnaryOp::Sqr)),
            bx(ScalarExpr::Const(3.0)),
        )),
    );
    let (want, got) = both(&op, OpCategory::UnaryElementwise, ElementKind::F64, &[tb]);
    assert_eq!(got, want);
}

#[test]
fn integer_compute_wraps_exactly_as_the_oracle_does() {
    let vals = [
        i32::MAX,
        i32::MIN,
        -1,
        0,
        1,
        2,
        -2,
        65_536,
        -65_536,
        i32::MAX - 1,
        7,
        -7,
        1_000_000,
        -1_000_000,
        46_341,
    ];
    let tb = |s: usize| {
        let v: Vec<i32> = (0..15).map(|i| vals[(i + s) % vals.len()]).collect();
        TypedBuffer::new(
            ElementKind::I32,
            vec![ROWS, COLS],
            vec![COLS, 1],
            i32_bytes(&v),
        )
    };
    for (name, body) in [
        ("add", ScalarExpr::Add(bx(input(0).0), bx(input(1).0))),
        ("sub", ScalarExpr::Sub(bx(input(0).0), bx(input(1).0))),
        ("mul", ScalarExpr::Mul(bx(input(0).0), bx(input(1).0))),
    ] {
        let op = OpDef::elementwise(name, 2, &[ElementKind::I32], Expr(body));
        let (want, got) = both(
            &op,
            OpCategory::BinaryElementwise,
            ElementKind::I32,
            &[tb(0), tb(5)],
        );
        assert_eq!(got, want, "{name}");
    }
}

#[test]
fn forms_the_lift_does_not_cover_decline_with_a_reason() {
    let f32k = [ElementKind::F32];
    // Control: the plain op lifts.
    let plain = OpDef::elementwise("p", 1, &f32k, Expr(input(0).0));
    assert!(lift_elementwise_opdef(&plain, ElementKind::F32, 2).is_ok());
    // A body that reads Coord.
    let coord = OpDef::elementwise("c", 1, &f32k, Expr(ScalarExpr::Coord(0)));
    assert_eq!(
        lift_elementwise_opdef(&coord, ElementKind::F32, 2),
        Err(LiftDecline::UnsupportedLeaf("Coord"))
    );
    // A body that reads a runtime Param.
    let param = OpDef::elementwise("q", 1, &f32k, Expr(ScalarExpr::Param(0)));
    assert_eq!(
        lift_elementwise_opdef(&param, ElementKind::F32, 2),
        Err(LiftDecline::UnsupportedLeaf("Param"))
    );
    // Rank 0.
    assert_eq!(
        lift_elementwise_opdef(&plain, ElementKind::F32, 0),
        Err(LiftDecline::BadRank)
    );
    // Dtype mismatch between the op's declared dtypes and the compute dtype.
    assert_eq!(
        lift_elementwise_opdef(&plain, ElementKind::F64, 2),
        Err(LiftDecline::MixedDtypes)
    );
}
