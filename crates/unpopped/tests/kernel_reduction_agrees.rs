// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The layers agree on reductions** (`docs/ir-kernel-layer.md` section 7): a float
//! `Access::Reduction` lifted into the kernel layer and run by the kernel interpreter produces,
//! byte for byte, what `oracle::evaluate` produces.
//!
//! **Why the interpreter has a mixed-type fallback** (see `kernel_eval_semantics.rs`): the oracle
//! rounds every float operation to the plan's compute dtype, but a reduction accumulates in `f64`
//! inside an `f32` kernel, so the kernel mixes widths and each node must be evaluated at its
//! operands' type or the sum would be rounded to `f32` after every addition.
//!
//! The oracle accumulates in `f64` and rounds once at the store; `Max` / `Min` keep a NaN once
//! seen; `Mean` divides by the reduced extents. Every corpus here has NaN, signed zeros,
//! infinities and cancelling magnitudes in it, because an agreement over friendly numbers would
//! pass a lift that accumulated in `f32`, dropped the NaN rule, or reordered the fold.

use unpopped::ir::{Expr, OpDef, ReduceOp, ScalarExpr, input};
use unpopped::kernel::check_kernel;
use unpopped::kernel_lift::{LiftDecline, lift_reduction_opdef};
use unpopped::oracle::kernel_eval::{KernelArg, run_kernel};
use unpopped::oracle::{TypedBuffer, evaluate};
use unpopped::plan::try_build_plan;
use unpopped_vocab::{ArchSku, AxisMask, ElementKind, OpCategory, OperandDesc, structure_key};

const ROWS: i64 = 5;
const COLS: i64 = 6;

fn f32_bytes(vs: &[f32]) -> Vec<u8> {
    vs.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn f64_bytes(vs: &[f64]) -> Vec<u8> {
    vs.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// 30 values, five rows. Row 0 cancels in f32 (1e8 + 1 - 1e8) but not in f64; row 1 is all
/// negative (a `Max` that forgot its first element would answer 0); row 2 has a NaN first and row 3
/// a NaN in the middle (a `Max` / `Min` that forgot the NaN rule loses the second); row 4 is signed
/// zeros.
fn corpus() -> Vec<f32> {
    vec![
        1.0e8,
        1.0,
        -1.0e8,
        0.5,
        2.0,
        -3.0, //
        -1.0,
        -2.0,
        -3.0,
        -4.0,
        -5.0,
        -6.0, //
        f32::NAN,
        1.0,
        2.0,
        3.0,
        -4.0,
        5.0, //
        f32::INFINITY,
        f32::NAN,
        f32::NEG_INFINITY,
        2.0,
        0.1,
        0.2, //
        0.0,
        -0.0,
        0.0,
        -0.0,
        0.0,
        -0.0,
    ]
}

fn input_buffer(vals: &[f32]) -> TypedBuffer {
    TypedBuffer::new(
        ElementKind::F32,
        vec![ROWS, COLS],
        vec![COLS, 1],
        f32_bytes(vals),
    )
}

/// Reduce the last axis (`axes` empty) or axis 0, with or without keepdim.
fn both(
    rop: ReduceOp,
    reduce_axis0: bool,
    keepdim: bool,
    inp: &TypedBuffer,
    dtype: ElementKind,
) -> (Vec<u8>, Vec<u8>) {
    let mut axes = AxisMask::EMPTY;
    if reduce_axis0 {
        axes.set(0);
    }
    let op = OpDef::reduction_axes("r", 1, &[dtype], Expr(input(0).0), rop, axes, keepdim);
    let (out_shape, out_strides): (Vec<i64>, Vec<i64>) = match (reduce_axis0, keepdim) {
        (false, false) => (vec![ROWS], vec![1]),
        (false, true) => (vec![ROWS, 1], vec![1, 1]),
        (true, false) => (vec![COLS], vec![1]),
        (true, true) => (vec![1, COLS], vec![COLS, 1]),
    };
    let a = OperandDesc::new(2, &inp.shape, &inp.strides, dtype, 256);
    let o = OperandDesc::new(out_shape.len(), &out_shape, &out_strides, dtype, 256);
    let key = structure_key(OpCategory::Reduction, &[a, o], ArchSku::Sm89);
    let plan = try_build_plan(&op, &key).expect("the reduction plans");
    let want = evaluate(&plan, &[a, o], std::slice::from_ref(inp), &[])
        .remove(0)
        .bytes;

    let kernel = lift_reduction_opdef(&op, dtype, 2).expect("the reduction lifts");
    assert_eq!(
        check_kernel(&kernel),
        Ok(()),
        "a lifted kernel is well formed"
    );
    let mut args = vec![
        KernelArg::Buffer(inp.clone()),
        KernelArg::Buffer(TypedBuffer::new(
            dtype,
            out_shape,
            out_strides,
            vec![0u8; 8 * 64],
        )),
        KernelArg::Int(i128::from(ROWS)),
        KernelArg::Int(i128::from(COLS)),
    ];
    run_kernel(&kernel, &mut args).expect("the kernel runs");
    let KernelArg::Buffer(out) = &args[1] else {
        unreachable!()
    };
    let got = out.bytes[..want.len()].to_vec();
    (want, got)
}

const OPS: [ReduceOp; 5] = [
    ReduceOp::Sum,
    ReduceOp::Prod,
    ReduceOp::Max,
    ReduceOp::Min,
    ReduceOp::Mean,
];

#[test]
fn every_reduce_op_agrees_over_the_last_axis() {
    for rop in OPS {
        let (want, got) = both(
            rop,
            false,
            false,
            &input_buffer(&corpus()),
            ElementKind::F32,
        );
        assert_eq!(got, want, "{rop:?}");
    }
}

#[test]
fn keepdim_and_the_outer_axis_agree() {
    for rop in OPS {
        for (axis0, keep) in [(false, true), (true, false), (true, true)] {
            let (want, got) = both(rop, axis0, keep, &input_buffer(&corpus()), ElementKind::F32);
            assert_eq!(got, want, "{rop:?} axis0={axis0} keepdim={keep}");
        }
    }
}

#[test]
fn the_accumulator_is_f64_not_f32() {
    // Row 0 is (1e8, 1, -1e8, 0.5, 2, -3). The f64 fold gives 0.5; an f32 fold loses the 1 next to
    // 1e8 and gives -0.5. The kernel and the oracle must agree on the f64 answer.
    let inp = input_buffer(&corpus());
    let (want, got) = both(ReduceOp::Sum, false, false, &inp, ElementKind::F32);
    assert_eq!(got, want);
    let row0 = f32::from_le_bytes([got[0], got[1], got[2], got[3]]);
    let f32_fold = ((((1.0e8_f32 + 1.0) + -1.0e8) + 0.5) + 2.0) + -3.0;
    assert_eq!(f64::from(row0), 0.5, "the f64 fold of row 0");
    assert_ne!(row0, f32_fold, "an f32 fold gives a different answer");
}

#[test]
fn max_and_min_keep_a_nan_once_seen() {
    // Row 2 starts with a NaN, row 3 has one in the middle: both reductions return NaN for both
    // rows (the oracle's rule), and an all-negative row 1 keeps its true maximum.
    let inp = input_buffer(&corpus());
    for rop in [ReduceOp::Max, ReduceOp::Min] {
        let (want, got) = both(rop, false, false, &inp, ElementKind::F32);
        assert_eq!(got, want, "{rop:?}");
        let at = |i: usize| {
            f32::from_le_bytes([got[4 * i], got[4 * i + 1], got[4 * i + 2], got[4 * i + 3]])
        };
        assert!(
            at(2).is_nan() && at(3).is_nan(),
            "{rop:?} over a row holding a NaN is NaN"
        );
    }
    let (_, max) = both(ReduceOp::Max, false, false, &inp, ElementKind::F32);
    assert_eq!(
        f32::from_le_bytes([max[4], max[5], max[6], max[7]]),
        -1.0,
        "all-negative row"
    );
}

#[test]
fn f64_compute_agrees() {
    let vals: Vec<f64> = corpus().into_iter().map(f64::from).collect();
    let inp = TypedBuffer::new(
        ElementKind::F64,
        vec![ROWS, COLS],
        vec![COLS, 1],
        f64_bytes(&vals),
    );
    for rop in OPS {
        let (want, got) = both(rop, false, false, &inp, ElementKind::F64);
        assert_eq!(got, want, "{rop:?}");
    }
}

#[test]
fn a_body_over_the_input_agrees() {
    // sum of squares: the body is evaluated at the compute dtype, the fold at f64.
    let body = ScalarExpr::Mul(Box::new(input(0).0), Box::new(input(0).0));
    let op = OpDef::reduction("ss", 1, &[ElementKind::F32], Expr(body), ReduceOp::Sum);
    let inp = input_buffer(&corpus());
    let a = OperandDesc::new(2, &inp.shape, &inp.strides, ElementKind::F32, 256);
    let o = OperandDesc::new(1, &[ROWS], &[1], ElementKind::F32, 256);
    let key = structure_key(OpCategory::Reduction, &[a, o], ArchSku::Sm89);
    let plan = try_build_plan(&op, &key).expect("plans");
    let want = evaluate(&plan, &[a, o], std::slice::from_ref(&inp), &[])
        .remove(0)
        .bytes;
    let kernel = lift_reduction_opdef(&op, ElementKind::F32, 2).expect("lifts");
    let mut args = vec![
        KernelArg::Buffer(inp),
        KernelArg::Buffer(TypedBuffer::new(
            ElementKind::F32,
            vec![ROWS],
            vec![1],
            vec![0u8; 64],
        )),
        KernelArg::Int(i128::from(ROWS)),
        KernelArg::Int(i128::from(COLS)),
    ];
    run_kernel(&kernel, &mut args).expect("runs");
    let KernelArg::Buffer(out) = &args[1] else {
        unreachable!()
    };
    assert_eq!(out.bytes[..want.len()], want[..]);
}

#[test]
fn forms_the_lift_does_not_cover_decline_with_a_reason() {
    let f32k = [ElementKind::F32];
    // Control: the plain sum lifts.
    let plain = OpDef::reduction("p", 1, &f32k, Expr(input(0).0), ReduceOp::Sum);
    assert!(lift_reduction_opdef(&plain, ElementKind::F32, 2).is_ok());
    // An elementwise op is not a reduction.
    let ew = OpDef::elementwise("e", 1, &f32k, Expr(input(0).0));
    assert_eq!(
        lift_reduction_opdef(&ew, ElementKind::F32, 2),
        Err(LiftDecline::NotReduction)
    );
    // A fused post-expression lifts in tolerance mode (`kernel_reduction_post_agrees.rs`).
    let post = OpDef::reduction_post(
        "q",
        1,
        &f32k,
        Expr(input(0).0),
        ReduceOp::Sum,
        Expr(ScalarExpr::Unary(
            unpopped::ir::UnaryOp::Sqrt,
            Box::new(ScalarExpr::Reduced(0)),
        )),
    );
    assert!(lift_reduction_opdef(&post, ElementKind::F32, 2).is_ok());
    // A narrow integer reduction (`I32` and `I64` lift: `kernel_int_reduce_scan_agrees.rs`).
    let int = OpDef::reduction("i", 1, &[ElementKind::I16], Expr(input(0).0), ReduceOp::Sum);
    assert_eq!(
        lift_reduction_opdef(&int, ElementKind::I16, 2),
        Err(LiftDecline::UnsupportedForm(
            "a reduction outside F32, F64, I32 and I64"
        ))
    );
}
