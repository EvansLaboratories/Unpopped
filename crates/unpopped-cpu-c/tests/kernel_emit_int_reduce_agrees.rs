// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The emitted C agrees with the kernel interpreter on integer reductions, scans and windows**
//! (`docs/e1-emit-from-the-kernel-layer.md`, stage E1b-int): a lifted `I32`/`I64` reduction or
//! scan is lowered by `CpuC::lower_kernel`, compiled, run, and its output bytes compared with
//! `run_kernel`.
//!
//! The corpus wraps (`MAX + 1`, a product past the width), has an all-negative row (a `Max` whose
//! accumulator started at `0` would answer `0`) and an all-positive row (the `Min` mirror). The
//! serial C is a correctness reference; nothing here times anything.

use unpopped::backend::Backend;
use unpopped::ir::{Expr, OpDef, ReduceOp, ScalarExpr, input};
use unpopped::kernel::{KernelFn, check_kernel};
use unpopped::kernel_lift::{lift_reduction_opdef, lift_scan_opdef, lift_window_opdef};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::types::int_range;
use unpopped_cpu_c::CpuC;
use unpopped_vocab::{AxisMask, ElementKind};

#[path = "support/cc.rs"]
mod cc;
#[path = "support/kernel_data.rs"]
mod kernel_data;
#[path = "support/kernel_run.rs"]
mod kernel_run;
use kernel_data::zeroed;
use kernel_run::{agree, run_c_mutated, width};

const ROWS: i64 = 5;
const COLS: i64 = 6;
const I32: ElementKind = ElementKind::I32;
const I64: ElementKind = ElementKind::I64;
const INTS: [ElementKind; 2] = [I32, I64];
const OPS: [ReduceOp; 4] = [ReduceOp::Sum, ReduceOp::Prod, ReduceOp::Max, ReduceOp::Min];

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

fn args(dtype: ElementKind, out_shape: &[i64]) -> Vec<KernelArg> {
    vec![
        KernelArg::Buffer(input_buffer(dtype)),
        KernelArg::Buffer(zeroed(dtype, out_shape)),
        KernelArg::Int(i128::from(ROWS)),
        KernelArg::Int(i128::from(COLS)),
    ]
}

fn reduction(dtype: ElementKind, rop: ReduceOp, axis0: bool) -> KernelFn {
    let mut axes = AxisMask::EMPTY;
    if axis0 {
        axes.set(0);
    }
    let op = OpDef::reduction_axes("r", 1, &[dtype], Expr(input(0).0), rop, axes, false);
    let k = lift_reduction_opdef(&op, dtype, 2).expect("the integer reduction lifts");
    assert_eq!(check_kernel(&k), Ok(()));
    k
}

fn scan(dtype: ElementKind, op: &OpDef) -> KernelFn {
    let k = lift_scan_opdef(op, dtype, 2).expect("the integer scan lifts");
    assert_eq!(check_kernel(&k), Ok(()));
    k
}

#[test]
fn integer_reductions_agree_over_each_axis() {
    // I32 over both axes; I64 over the last axis (each compile is seconds).
    for (dtype, axes) in [(I32, &[false, true][..]), (I64, &[false][..])] {
        for rop in OPS {
            for &axis0 in axes {
                let n = if axis0 { COLS } else { ROWS };
                agree(
                    &format!("ired_{dtype:?}_{rop:?}_{axis0}"),
                    &reduction(dtype, rop, axis0),
                    args(dtype, &[n]),
                );
            }
        }
    }
}

#[test]
fn integer_scans_agree_in_both_directions_and_exclusivities() {
    // Forward-inclusive and reverse-exclusive bracket the two directions and both write orders.
    let cases = [
        (I32, ReduceOp::Sum, false, false),
        (I32, ReduceOp::Prod, true, true),
        (I32, ReduceOp::Max, false, false),
        (I32, ReduceOp::Max, true, true),
        (I32, ReduceOp::Min, false, true),
        (I32, ReduceOp::Min, true, false),
        (I64, ReduceOp::Sum, false, false),
        (I64, ReduceOp::Max, true, true),
        (I64, ReduceOp::Prod, false, true),
    ];
    for (dtype, rop, reverse, exclusive) in cases {
        let op = OpDef::scan_simple("s", &[dtype], rop, 1, reverse, exclusive);
        agree(
            &format!("iscan_{dtype:?}_{rop:?}_{reverse}_{exclusive}"),
            &scan(dtype, &op),
            args(dtype, &[ROWS, COLS]),
        );
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
        agree(
            &format!("iscanmap_{dtype:?}"),
            &scan(dtype, &op),
            args(dtype, &[ROWS, COLS]),
        );
    }
}

/// Row `row` of the first output after `from` is replaced by `to` in the emitted C (anchor
/// asserted present first); an empty `from` runs the C as emitted.
fn row_i32(tag: &str, k: &KernelFn, row: usize, from: &str, to: &str) -> i32 {
    let ran = run_c_mutated(tag, k, &args(I32, &[ROWS]), |s| {
        if from.is_empty() {
            return s;
        }
        assert_eq!(
            s.matches(from).count(),
            1,
            "{tag}: anchor `{from}` in:
{s}"
        );
        s.replace(from, to)
    })
    .expect("a host C compiler");
    i32::from_le_bytes(ran.outs[0][row * 4..row * 4 + 4].try_into().unwrap())
}

#[test]
fn max_compares_with_greater_and_starts_from_the_first_element() {
    let k = reduction(I32, ReduceOp::Max, false);
    // Row 1 is all negative, its Max is -1; row 4 is (4, 9, 2, 8, 6, 5), its Max 9.
    assert_eq!(row_i32("imax_ok1", &k, 1, "", ""), -1);
    assert_eq!(row_i32("imax_ok4", &k, 4, "", ""), 9);
    // `>` flipped to `<`: the all-negative row now answers its minimum, -6.
    assert_eq!(
        row_i32(
            "imax_lt",
            &k,
            1,
            "((l7) > (l5) ? 1 : 0)",
            "((l7) < (l5) ? 1 : 0)"
        ),
        -6
    );
    // Without the first-element rule the accumulator stays 0, which beats every negative.
    assert_eq!(
        row_i32("imax_have", &k, 1, "if ((l6) != 0) {", "if (1) {"),
        0
    );
}

#[test]
fn the_sum_wraps_at_the_type_width_in_the_emitted_c() {
    // Row 0 is (MAX, 1, -1, 2, MAX, 3), which wraps. The add is spelled through `unsigned int`;
    // narrowing the accumulator's cast to `unsigned short` loses the high half, so the width is observable.
    let k = reduction(I32, ReduceOp::Sum, false);
    let src = CpuC.lower_kernel(&k).unwrap().source;
    assert!(
        src.contains("(unsigned int)"),
        "an I32 add goes through unsigned:
{src}"
    );
    let exact: i64 = corpus(I32)[..6].iter().map(|&v| v as i64).sum();
    assert!(exact > i64::from(i32::MAX), "the row overflows I32");
    let want = exact as i32;
    assert_eq!(row_i32("iwrap_ok", &k, 0, "", ""), want);
    // Row 0 happens to cancel under the truncation, so the mutant is judged on row 3
    // (100000, 100000, 3, 1, -1, 2), whose sum needs more than 16 bits.
    let row3 = |t: &str, from: &str, to: &str| row_i32(t, &k, 3, from, to);
    assert_eq!(row3("iwrap_ok3", "", ""), 200_005);
    let narrowed = row3("iwrap_short", "(unsigned int)(l5)", "(unsigned short)(l5)");
    assert_ne!(narrowed, 200_005, "the mutant must change the answer");
}

/// A 1-D pool over `rows x k_in`; the geometry is `[size, stride, dilation, pad]`.
fn window(dtype: ElementKind, wop: ReduceOp, geo: [u8; 4]) -> KernelFn {
    let [size, stride, dilation, pad] = geo;
    let op = OpDef::window_simple(
        "win",
        &[dtype],
        wop,
        1,
        size,
        stride,
        dilation,
        pad,
        pad,
        false,
    );
    let k = lift_window_opdef(&op, dtype, 2).expect("the integer window lifts");
    assert_eq!(check_kernel(&k), Ok(()));
    k
}

fn window_args(dtype: ElementKind, rows: i64, k_in: i64, k_out: i64) -> Vec<KernelArg> {
    let (lo, hi) = int_range(dtype).expect("an integer dtype");
    let vals = [hi, 1, -1, 2, hi, 3, lo, -5, -7, -7, 4];
    let n = (rows * k_in) as usize;
    let bytes: Vec<u8> = (0..n)
        .flat_map(|i| vals[i % vals.len()].to_le_bytes()[..width(dtype)].to_vec())
        .collect();
    vec![
        KernelArg::Buffer(TypedBuffer::new(
            dtype,
            vec![rows, k_in],
            vec![k_in, 1],
            bytes,
        )),
        KernelArg::Buffer(zeroed(dtype, &[rows, k_out])),
        KernelArg::Int(i128::from(rows)),
        KernelArg::Int(i128::from(k_in)),
        KernelArg::Int(i128::from(k_out)),
    ]
}

#[test]
fn integer_pools_agree_in_padded_strided_and_dilated_geometries() {
    // Padded and strided: k_out = (9 + 2 - 3) / 2 + 1 = 5, edge windows hold fewer valid taps.
    // Dilated: k_out = (11 + 4 - 5) / 1 + 1 = 11.
    for (dtype, geo, k_in, k_out) in [
        (I32, [3, 2, 1, 1], 9, 5),
        (I32, [3, 1, 2, 2], 11, 11),
        (I64, [3, 2, 1, 1], 9, 5),
    ] {
        for wop in [ReduceOp::Sum, ReduceOp::Max, ReduceOp::Min] {
            agree(
                &format!("iwin_{dtype:?}_{wop:?}_{}_{}", geo[1], geo[2]),
                &window(dtype, wop, geo),
                window_args(dtype, 3, k_in, k_out),
            );
        }
    }
}

#[test]
fn a_window_max_starts_from_its_first_valid_tap_not_zero() {
    // Row 0, windows of taps 2o-1..=2o+1: window 4 holds only negatives (-5, -7, -7), so its Max
    // is -5; an accumulator started at 0 would answer 0 there.
    let k = window(I32, ReduceOp::Max, [3, 2, 1, 1]);
    let ran = run_c_mutated("iwin_max", &k, &window_args(I32, 3, 9, 5), |s| s).expect("cc");
    let at = |i: usize| i32::from_le_bytes(ran.outs[0][i * 4..i * 4 + 4].try_into().unwrap());
    assert_eq!(at(4), -5);
}
