// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The emitted C agrees with the kernel interpreter on NaN-ignoring extrema** (ROADMAP U5 P1; PM ruling
//! 2026-11-02 (5)): a lifted `F32`/`F64` `MaxNum` / `MinNum` reduction is lowered by `CpuC::lower_kernel`, compiled
//! with a host C compiler, run, and its output bytes compared with `run_kernel` (a NaN is compared as a NaN, any
//! other value bit for bit, so `0.0` and `-0.0` differ).
//!
//! The corpus has an all-NaN row (the answer is NaN, not an identity), a NaN first, a NaN in the middle, an
//! all-negative row (a seed of `0` is wrong) and signed zeros (the first of two equal zeros is kept).

use unpopped::backend::Backend;
use unpopped::ir::{Expr, OpDef, ReduceOp, input};
use unpopped::kernel::{KernelFn, check_kernel};
use unpopped::kernel_lift::lift_reduction_opdef;
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped_cpu_c::CpuC;
use unpopped_vocab::{AxisMask, ElementKind};

#[path = "support/cc.rs"]
mod cc;
#[path = "support/kernel_data.rs"]
mod kernel_data;
#[path = "support/kernel_run.rs"]
mod kernel_run;
use kernel_data::zeroed;
use kernel_run::{agree, run_c_mutated};

const ROWS: i64 = 5;
const COLS: i64 = 6;
const NUM: [ReduceOp; 2] = [ReduceOp::MaxNum, ReduceOp::MinNum];
const FLOATS: [ElementKind; 2] = [ElementKind::F32, ElementKind::F64];

fn corpus() -> Vec<f64> {
    let n = f64::NAN;
    vec![
        n, n, n, n, n, n, //
        n, 1.0, 2.0, 3.0, -4.0, 5.0, //
        1.0, n, -7.0, 3.0, n, 2.0, //
        -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, //
        0.0, -0.0, 0.0, -0.0, 0.0, -0.0, //
    ]
}

fn args(dtype: ElementKind, out_shape: &[i64]) -> Vec<KernelArg> {
    let bytes: Vec<u8> = corpus()
        .iter()
        .flat_map(|v| {
            if dtype == ElementKind::F64 {
                v.to_le_bytes().to_vec()
            } else {
                (*v as f32).to_le_bytes().to_vec()
            }
        })
        .collect();
    vec![
        KernelArg::Buffer(TypedBuffer::new(
            dtype,
            vec![ROWS, COLS],
            vec![COLS, 1],
            bytes,
        )),
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
    let k = lift_reduction_opdef(&op, dtype, 2).expect("the NaN-ignoring reduction lifts");
    assert_eq!(check_kernel(&k), Ok(()));
    k
}

#[test]
fn nan_ignoring_extrema_agree_over_each_axis_and_width() {
    for dtype in FLOATS {
        for rop in NUM {
            for axis0 in [false, true] {
                let n = if axis0 { COLS } else { ROWS };
                let k = reduction(dtype, rop, axis0);
                let tag = format!("num_{dtype:?}_{rop:?}_{axis0}");
                agree(&tag, &k, args(dtype, &[n]));
            }
        }
    }
}

#[test]
fn the_c_nan_probe_is_what_the_differential_is_sensitive_to() {
    // a seeded mutant of the generated C: turning the accumulator's NaN probe around must change an answer
    let k = reduction(ElementKind::F32, ReduceOp::MaxNum, false);
    let g = CpuC.lower_kernel(&k).expect("the reduction lowers");
    assert!(
        g.source.contains(" != "),
        "the NaN probe `x != x` is spelled"
    );
    let inputs = || args(ElementKind::F32, &[ROWS]);
    let honest = run_c_mutated("num_honest", &k, &inputs(), |s| s).expect("a C compiler");
    let flipped = run_c_mutated("num_flipped", &k, &inputs(), |s| s.replace(" != ", " == "));
    if let Some(m) = flipped {
        assert_ne!(
            m.outs, honest.outs,
            "the answer must not survive the NaN probe being flipped"
        );
    }
}
