// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Arg-reductions** (ROADMAP U5 P3; `docs/options-u5-collectives-and-richer-reductions.md`): `ReduceOp::ArgMax` /
//! `ArgMin` in an `Access::Reduction` give the **index** of the extreme, as an `I64` (`out_dtype = Some(I64)`), by
//! KISS-OPS-6.11-0007's order: NaN is the **greatest** value and ties go to the lower index. So `ArgMax` of a run with a
//! NaN is the first NaN's index, and `ArgMin` never selects a NaN while a non-NaN exists (index 0 when every element is
//! NaN). That differs from NumPy and PyTorch, whose `argmin` returns the NaN's index; 0.42.2 had the PyTorch rule, a
//! defect against KISS fixed in 0.42.5. The index is the position in the row-major order of the reduced axes: one axis
//! gives that axis's coordinate, all axes give the flat index.
//!
//! The oracle is held to a fold written here in Rust from the total order (not from the oracle's own steps). The
//! corpus has ties (the first index must win, not the last), a NaN first (it decides `ArgMax`, and `ArgMin` must skip
//! it), a NaN in the middle with a larger number after it, an all-NaN row (index 0 for both), an all-negative row (a
//! seed of `0` is wrong) and signed zeros (equal, so the first).

use unpopped::ir::{Expr, OpDef, ReduceOp, input};
use unpopped::kernel::check_kernel;
use unpopped::kernel_lift::{LiftDecline, lift_reduction_opdef};
use unpopped::oracle::kernel_eval::{KernelArg, run_kernel};
use unpopped::oracle::{TypedBuffer, evaluate};
use unpopped::plan::{PlanError, try_build_plan};
use unpopped_vocab::{ArchSku, AxisMask, ElementKind, OpCategory, OperandDesc, structure_key};

#[path = "support/arg_loop.rs"]
mod arg_loop;
use arg_loop::loop_kernel;

const ROWS: i64 = 6;
const COLS: i64 = 6;
const ARGS: [ReduceOp; 2] = [ReduceOp::ArgMax, ReduceOp::ArgMin];

fn corpus() -> Vec<f64> {
    let n = f64::NAN;
    vec![
        3.0, 9.0, 9.0, 1.0, 1.0, 2.0, //  ties: the first 9 and the first 1
        n, 1.0, 2.0, 3.0, -4.0, 5.0, //  a NaN first
        1.0, 4.0, n, 7.0, n, 2.0, //  a NaN in the middle, larger numbers after it
        -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, //  all negative
        0.0, -0.0, 0.0, -0.0, 0.0, -0.0, //  equal zeros
        n, n, n, n, n, n, //  all NaN
    ]
}

fn input_buffer(dtype: ElementKind) -> TypedBuffer {
    let bytes: Vec<u8> = corpus()
        .iter()
        .flat_map(|v| match dtype {
            ElementKind::F64 => v.to_le_bytes().to_vec(),
            ElementKind::I32 => (if v.is_nan() { 0 } else { *v as i32 })
                .to_le_bytes()
                .to_vec(),
            _ => (*v as f32).to_le_bytes().to_vec(),
        })
        .collect();
    TypedBuffer::new(dtype, vec![ROWS, COLS], vec![COLS, 1], bytes)
}

/// The corpus as the dtype holds it (an `I32` truncates and has no NaN).
fn values(dtype: ElementKind) -> Vec<f64> {
    corpus()
        .into_iter()
        .map(|v| match dtype {
            ElementKind::I32 => {
                if v.is_nan() {
                    0.0
                } else {
                    f64::from(v as i32)
                }
            }
            ElementKind::F32 => f64::from(v as f32),
            _ => v,
        })
        .collect()
}

/// `x` is greater than `y` in KISS-OPS-6.11-0007's total order: NaN is the greatest value (and equal to NaN).
fn greater(x: f64, y: f64) -> bool {
    (x.is_nan() && !y.is_nan()) || (!x.is_nan() && !y.is_nan() && x > y)
}

/// The first index of the extreme in that order: only a strictly greater (or, for the minimum, strictly smaller)
/// element replaces, so ties keep the lower index.
fn arg_of(xs: &[f64], max: bool) -> i64 {
    let mut best = 0usize;
    for (i, &x) in xs.iter().enumerate().skip(1) {
        let better = if max {
            greater(x, xs[best])
        } else {
            greater(xs[best], x)
        };
        if better {
            best = i;
        }
    }
    best as i64
}

/// The reduced axes (`axes` empty is the last axis) and the reference answer, row-major over the reduced set.
fn reference(dtype: ElementKind, rop: ReduceOp, axes: &[u8]) -> Vec<i64> {
    let v = values(dtype);
    let max = rop == ReduceOp::ArgMax;
    let cell = |r: i64, k: i64| v[(r * COLS + k) as usize];
    match axes {
        [] | [1] => (0..ROWS)
            .map(|r| arg_of(&(0..COLS).map(|k| cell(r, k)).collect::<Vec<_>>(), max))
            .collect(),
        [0] => (0..COLS)
            .map(|k| arg_of(&(0..ROWS).map(|r| cell(r, k)).collect::<Vec<_>>(), max))
            .collect(),
        _ => vec![arg_of(&v, max)],
    }
}

fn reduction(dtype: ElementKind, rop: ReduceOp, axes: &[u8], with_out: bool) -> OpDef {
    let mut mask = AxisMask::EMPTY;
    for a in axes {
        mask.set(*a);
    }
    let mut op = OpDef::reduction_axes("r", 1, &[dtype], Expr(input(0).0), rop, mask, false);
    if with_out {
        op.out_dtype = Some(ElementKind::I64);
    }
    op
}

fn oracle(
    dtype: ElementKind,
    rop: ReduceOp,
    axes: &[u8],
    with_out: bool,
) -> Result<Vec<i64>, PlanError> {
    let op = reduction(dtype, rop, axes, with_out);
    let inp = input_buffer(dtype);
    let n: i64 = match axes {
        [] | [1] => ROWS,
        [0] => COLS,
        _ => 1,
    };
    let a = OperandDesc::new(2, &inp.shape, &inp.strides, dtype, 256);
    let o = OperandDesc::new(1, &[n], &[1], ElementKind::I64, 256);
    let key = structure_key(OpCategory::Reduction, &[a, o], ArchSku::Sm89);
    let plan = try_build_plan(&op, &key)?;
    let out = evaluate(&plan, &[a, o], std::slice::from_ref(&inp), &[])
        .remove(0)
        .bytes;
    Ok(out[..n as usize * 8]
        .chunks(8)
        .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
        .collect())
}

#[test]
fn the_oracle_matches_an_independent_kiss_order_fold() {
    for dtype in [ElementKind::F32, ElementKind::F64, ElementKind::I32] {
        for rop in ARGS {
            for axes in [&[][..], &[0][..], &[0, 1][..]] {
                let got = oracle(dtype, rop, axes, true).expect("an arg-reduction plans");
                assert_eq!(
                    got,
                    reference(dtype, rop, axes),
                    "{dtype:?} {rop:?} axes={axes:?}"
                );
            }
        }
    }
}

#[test]
fn the_corpus_pins_the_rule() {
    // ties: the FIRST 9 (index 1) and the first 1 (index 3); a NaN first wins ArgMax and is skipped by ArgMin (which
    // finds the smallest number, index 4); a NaN in the middle wins ArgMax; an all-negative row; equal zeros and an
    // all-NaN row give index 0
    let max = oracle(ElementKind::F32, ReduceOp::ArgMax, &[], true).unwrap();
    let min = oracle(ElementKind::F32, ReduceOp::ArgMin, &[], true).unwrap();
    assert_eq!(max, vec![1, 0, 2, 0, 0, 0]);
    assert_eq!(min, vec![3, 4, 0, 5, 0, 0]);
}

#[test]
fn an_arg_reduction_needs_its_i64_output_and_a_reduction() {
    let d = match oracle(ElementKind::F32, ReduceOp::ArgMax, &[], false) {
        Err(PlanError::InadmissibleOpAtDtype { detail, .. }) => detail,
        other => panic!("expected a decline, got {other:?}"),
    };
    assert!(d.contains("I64"), "{d}");
    // a scan declines (Reduction only)
    let dtype = ElementKind::F32;
    let scan = OpDef::scan_simple("s", &[dtype], ReduceOp::ArgMax, 1, false, false);
    let inp = input_buffer(dtype);
    let a = OperandDesc::new(2, &inp.shape, &inp.strides, dtype, 256);
    let key = structure_key(OpCategory::Scan, &[a, a], ArchSku::Sm89);
    match try_build_plan(&scan, &key) {
        Err(PlanError::InadmissibleOpAtDtype { detail, .. }) => {
            assert!(detail.contains("Reduction only"), "{detail}");
        }
        other => panic!("expected a decline, got {:?}", other.map(|_| ())),
    }
}

#[test]
fn the_kernel_lift_declines_an_arg_reduction() {
    // a kernel states an arg-reduction as a loop over locals (the sequential form needs no lift)
    let op = reduction(ElementKind::F32, ReduceOp::ArgMax, &[], true);
    assert!(matches!(
        lift_reduction_opdef(&op, ElementKind::F32, 2),
        Err(LiftDecline::UnsupportedForm(_))
    ));
}

#[test]
fn a_kernel_loop_states_the_same_rule_as_the_oracle() {
    for rop in ARGS {
        let k = loop_kernel(rop == ReduceOp::ArgMax);
        assert_eq!(check_kernel(&k), Ok(()));
        let out = TypedBuffer::new(
            ElementKind::I64,
            vec![ROWS],
            vec![1],
            vec![0u8; 8 * ROWS as usize],
        );
        let mut args = vec![
            KernelArg::Buffer(input_buffer(ElementKind::F32)),
            KernelArg::Buffer(out),
            KernelArg::Int(i128::from(ROWS)),
            KernelArg::Int(i128::from(COLS)),
        ];
        run_kernel(&k, &mut args).expect("the loop runs");
        let KernelArg::Buffer(out) = &args[1] else {
            unreachable!()
        };
        let got: Vec<i64> = out
            .bytes
            .chunks(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let want = oracle(ElementKind::F32, rop, &[], true).expect("plans");
        assert_eq!(got, want, "{rop:?}");
    }
}
