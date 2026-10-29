// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The layers agree on integer reductions and scans** (`docs/ir-kernel-layer.md` section 7):
//! `lift_reduction_opdef` and `lift_scan_opdef` at `I32` and `I64`, run by the kernel interpreter,
//! produce byte for byte what `oracle::evaluate` produces.
//!
//! The oracle folds an integer exactly and wraps, so the corpus holds wrap-around (`MAX + 1`, a
//! product past the width), an all-negative row (a `Max` whose accumulator started at `0` instead
//! of the type minimum would answer `0`), an all-positive row (the `Min` mirror) and ties.

use unpopped::ir::{Expr, OpDef, ReduceOp, ScalarExpr, input};
use unpopped::kernel::check_kernel;
use unpopped::kernel_lift::{LiftDecline, lift_reduction_opdef, lift_scan_opdef};
use unpopped::oracle::kernel_eval::{KernelArg, run_kernel};
use unpopped::oracle::{TypedBuffer, evaluate};
use unpopped::plan::try_build_plan;
use unpopped::types::int_range;
use unpopped_vocab::{ArchSku, ElementKind, OpCategory, OperandDesc, structure_key};

const ROWS: i64 = 5;
const COLS: i64 = 6;
const INTS: [ElementKind; 2] = [ElementKind::I32, ElementKind::I64];
const OPS: [ReduceOp; 4] = [ReduceOp::Sum, ReduceOp::Prod, ReduceOp::Max, ReduceOp::Min];

fn width(dtype: ElementKind) -> usize {
    if dtype == ElementKind::I64 { 8 } else { 4 }
}

fn corpus(dtype: ElementKind) -> Vec<i128> {
    let (lo, hi) = int_range(dtype).expect("an integer dtype");
    let big = if dtype == ElementKind::I64 {
        1 << 40
    } else {
        100_000
    };
    vec![
        hi, 1, -1, 2, hi, 3, //
        -1, -2, -3, -4, -5, -6, //
        7, 7, 7, -7, lo, 0, //
        big, big, 3, 1, -1, 2, //
        4, 9, 2, 8, 6, 5, //
    ]
}

fn input_buffer(dtype: ElementKind) -> TypedBuffer {
    let bytes: Vec<u8> = corpus(dtype)
        .iter()
        .flat_map(|v| v.to_le_bytes()[..width(dtype)].to_vec())
        .collect();
    TypedBuffer::new(dtype, vec![ROWS, COLS], vec![COLS, 1], bytes)
}

fn run(
    kernel: &unpopped::kernel::KernelFn,
    inp: &TypedBuffer,
    out_shape: Vec<i64>,
    out_strides: Vec<i64>,
    len: usize,
) -> Vec<u8> {
    assert_eq!(
        check_kernel(kernel),
        Ok(()),
        "a lifted kernel is well formed"
    );
    let out = TypedBuffer::new(inp.dtype, out_shape, out_strides, vec![0u8; len]);
    let mut args = vec![
        KernelArg::Buffer(inp.clone()),
        KernelArg::Buffer(out),
        KernelArg::Int(i128::from(ROWS)),
        KernelArg::Int(i128::from(COLS)),
    ];
    run_kernel(kernel, &mut args).expect("the kernel runs");
    let KernelArg::Buffer(out) = &args[1] else {
        unreachable!()
    };
    out.bytes[..len].to_vec()
}

fn reduce_both(dtype: ElementKind, rop: ReduceOp, axis0: bool) -> (Vec<u8>, Vec<u8>) {
    let mut axes = unpopped_vocab::AxisMask::EMPTY;
    if axis0 {
        axes.set(0);
    }
    let op = OpDef::reduction_axes("r", 1, &[dtype], Expr(input(0).0), rop, axes, false);
    let inp = input_buffer(dtype);
    let n = if axis0 { COLS } else { ROWS };
    let a = OperandDesc::new(2, &inp.shape, &inp.strides, dtype, 256);
    let o = OperandDesc::new(1, &[n], &[1], dtype, 256);
    let key = structure_key(OpCategory::Reduction, &[a, o], ArchSku::Sm89);
    let plan = try_build_plan(&op, &key).expect("the integer reduction plans");
    let want = evaluate(&plan, &[a, o], std::slice::from_ref(&inp), &[])
        .remove(0)
        .bytes;
    let len = n as usize * width(dtype);
    let kernel = lift_reduction_opdef(&op, dtype, 2).expect("the integer reduction lifts");
    let got = run(&kernel, &inp, vec![n], vec![1], len);
    (want[..len].to_vec(), got)
}

fn scan_both(dtype: ElementKind, op: &OpDef) -> (Vec<u8>, Vec<u8>) {
    let inp = input_buffer(dtype);
    let a = OperandDesc::new(2, &inp.shape, &inp.strides, dtype, 256);
    let key = structure_key(OpCategory::Scan, &[a, a], ArchSku::Sm89);
    let plan = try_build_plan(op, &key).expect("the integer scan plans");
    let want = evaluate(&plan, &[a, a], std::slice::from_ref(&inp), &[])
        .remove(0)
        .bytes;
    let len = (ROWS * COLS) as usize * width(dtype);
    let kernel = lift_scan_opdef(op, dtype, 2).expect("the integer scan lifts");
    let got = run(&kernel, &inp, vec![ROWS, COLS], vec![COLS, 1], len);
    (want[..len].to_vec(), got)
}

#[test]
fn integer_reductions_agree_over_each_axis() {
    for dtype in INTS {
        for rop in OPS {
            for axis0 in [false, true] {
                let (want, got) = reduce_both(dtype, rop, axis0);
                assert_eq!(got, want, "{dtype:?} {rop:?} axis0={axis0}");
            }
        }
    }
}

#[test]
fn integer_scans_agree_in_every_direction_and_exclusivity() {
    for dtype in INTS {
        for rop in OPS {
            for reverse in [false, true] {
                for exclusive in [false, true] {
                    let op = OpDef::scan_simple("s", &[dtype], rop, 1, reverse, exclusive);
                    let (want, got) = scan_both(dtype, &op);
                    assert_eq!(
                        got, want,
                        "{dtype:?} {rop:?} reverse={reverse} exclusive={exclusive}"
                    );
                }
            }
        }
    }
}

#[test]
fn a_scan_pre_map_agrees() {
    let body = ScalarExpr::Mul(Box::new(input(0).0), Box::new(input(0).0));
    for dtype in INTS {
        let op = OpDef::scan(
            "sq",
            1,
            &[dtype],
            ReduceOp::Sum,
            1,
            false,
            false,
            Expr(body.clone()),
            Expr(ScalarExpr::Reduced(0)),
        );
        let (want, got) = scan_both(dtype, &op);
        assert_eq!(got, want, "{dtype:?}");
    }
}

fn read(dtype: ElementKind, b: &[u8], i: usize) -> i128 {
    let w = width(dtype);
    let mut x = [0u8; 16];
    x[..w].copy_from_slice(&b[i * w..(i + 1) * w]);
    if x[w - 1] & 0x80 != 0 {
        x[w..].fill(0xff);
    }
    i128::from_le_bytes(x)
}

#[test]
fn the_corpus_exercises_what_the_lift_must_get_right() {
    for dtype in INTS {
        let (lo, hi) = int_range(dtype).unwrap();
        // Controls, read from the ORACLE's answer: a wrapped sum, an all-negative Max row and an
        // all-positive Min row (an accumulator started at 0 would answer 0 for both).
        let (sum, _) = reduce_both(dtype, ReduceOp::Sum, false);
        let exact_sum: i128 = corpus(dtype)[..6].iter().sum();
        assert!(exact_sum > hi, "{dtype:?} the sum row overflows the width");
        assert_eq!(
            read(dtype, &sum, 0),
            wrap(dtype, exact_sum),
            "{dtype:?} wrapped sum"
        );
        let (max, _) = reduce_both(dtype, ReduceOp::Max, false);
        assert_eq!(read(dtype, &max, 1), -1, "{dtype:?} all-negative Max");
        let (min, _) = reduce_both(dtype, ReduceOp::Min, false);
        assert_eq!(read(dtype, &min, 4), 2, "{dtype:?} all-positive Min");
        let (prod, _) = reduce_both(dtype, ReduceOp::Prod, false);
        let exact: i128 = corpus(dtype)[18..24].iter().product();
        assert!(
            !(lo..=hi).contains(&exact),
            "{dtype:?} the product row overflows the width"
        );
        assert_eq!(
            read(dtype, &prod, 3),
            wrap(dtype, exact),
            "{dtype:?} wrapped product"
        );
    }
}

/// `v` wrapped into `dtype`'s range (two's complement).
fn wrap(dtype: ElementKind, v: i128) -> i128 {
    let (lo, hi) = int_range(dtype).unwrap();
    (v - lo).rem_euclid(hi - lo + 1) + lo
}

#[test]
fn forms_the_lift_does_not_cover_decline_with_a_reason() {
    // Control: the plain integer forms lift.
    let sum = OpDef::reduction("r", 1, &[ElementKind::I32], Expr(input(0).0), ReduceOp::Sum);
    assert!(lift_reduction_opdef(&sum, ElementKind::I32, 2).is_ok());
    let mean = OpDef::reduction(
        "r",
        1,
        &[ElementKind::I32],
        Expr(input(0).0),
        ReduceOp::Mean,
    );
    assert_eq!(
        lift_reduction_opdef(&mean, ElementKind::I32, 2),
        Err(LiftDecline::UnsupportedForm("an integer Mean"))
    );
    let narrow = OpDef::scan_simple("s", &[ElementKind::I16], ReduceOp::Sum, 1, false, false);
    assert!(matches!(
        lift_scan_opdef(&narrow, ElementKind::I16, 2),
        Err(LiftDecline::UnsupportedForm(_))
    ));
}
