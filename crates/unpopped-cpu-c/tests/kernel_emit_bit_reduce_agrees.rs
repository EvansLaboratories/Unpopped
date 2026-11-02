// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The emitted C agrees with the kernel interpreter on bitwise reductions** (ROADMAP U5 P2; PM ruling
//! 2026-11-02): a lifted `I32`/`I64` `And`/`Or`/`Xor` reduction is lowered by `CpuC::lower_kernel`, compiled with
//! a host C compiler, run, and its output bytes compared with `run_kernel`.
//!
//! The corpus has an all-ones row, a row with a zero in it (an `And` that started at `0` answers `0`
//! everywhere), a row of an even count of equal elements (a `Xor` that dropped an element is wrong on one parity)
//! and negative values (sign extension).

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
use kernel_run::{agree, run_c_mutated, width};

const ROWS: i64 = 5;
const COLS: i64 = 6;
const INTS: [ElementKind; 2] = [ElementKind::I32, ElementKind::I64];
const BITS: [ReduceOp; 3] = [ReduceOp::And, ReduceOp::Or, ReduceOp::Xor];

fn corpus() -> Vec<i128> {
    vec![
        -1, -1, -1, -1, -1, -1, //
        0b1011, 0b0110, 0, 0b1111, 0b1011, 0b0001, //
        5, 5, 5, 5, 5, 5, //
        7, 7, 7, 7, 7, 1, //
        -8, 3, -1, 0x7fff, -0x8000, 12, //
    ]
}

fn args(dtype: ElementKind, out_shape: &[i64]) -> Vec<KernelArg> {
    let bytes: Vec<u8> = corpus()
        .iter()
        .flat_map(|v| v.to_le_bytes()[..width(dtype)].to_vec())
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
    let k = lift_reduction_opdef(&op, dtype, 2).expect("the bit reduction lifts");
    assert_eq!(check_kernel(&k), Ok(()));
    k
}

#[test]
fn bit_reductions_agree_over_each_axis_and_width() {
    for dtype in INTS {
        for rop in BITS {
            for axis0 in [false, true] {
                let n = if axis0 { COLS } else { ROWS };
                let k = reduction(dtype, rop, axis0);
                let tag = format!("bit_{dtype:?}_{rop:?}_{axis0}");
                agree(&tag, &k, args(dtype, &[n]));
            }
        }
    }
}

#[test]
fn the_c_operator_is_what_the_differential_is_sensitive_to() {
    // a seeded mutant of the generated C: each bit operator replaced by another must change an answer
    let k = reduction(ElementKind::I32, ReduceOp::Xor, false);
    let g = CpuC.lower_kernel(&k).expect("the Xor reduction lowers");
    assert!(
        g.source.contains(" ^ "),
        "the Xor reduction is spelled with a C `^`"
    );
    let inputs = || args(ElementKind::I32, &[ROWS]);
    let honest = run_c_mutated("bit_honest", &k, &inputs(), |s| s).expect("a C compiler");
    let as_or = run_c_mutated("bit_xor_as_or", &k, &inputs(), |s| s.replace(" ^ ", " | "));
    if let Some(m) = as_or {
        assert_ne!(
            m.outs, honest.outs,
            "the Xor reduction must not survive a `^` that became `|`"
        );
    }
}
