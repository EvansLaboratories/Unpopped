// SPDX-License-Identifier: MIT OR Apache-2.0
//! **NaN-ignoring extrema** (ROADMAP U5 P1, PM ruling 2026-11-02 (5); `docs/options-u5-collectives-and-richer-reductions.md`):
//! `ReduceOp::MaxNum` / `MinNum` fold a run ignoring NaN (IEEE `maxNum` / `minNum`); a run of only NaN gives NaN. Unlike
//! `Max` / `Min`, which keep a NaN once seen. They reduce in an `Access::Reduction` (scan, window and row reduction
//! decline). Three things agree: the oracle, a fold written here in Rust, and the lifted kernel run by the interpreter.
//!
//! The corpus has an all-NaN row (the answer must be NaN, not an identity), a NaN first (a fold seeded by the first
//! element must replace a NaN seed), a NaN in the middle (a fold that kept the NaN rule of `Max` loses the rest), an
//! all-negative row (a seed of `0` is wrong) and signed zeros (the first of two equal zeros is kept, as `Max` keeps it).

use unpopped::ir::{Expr, OpDef, ReduceOp, input};
use unpopped::kernel::check_kernel;
use unpopped::kernel_lift::{LiftDecline, lift_reduction_opdef, lift_scan_opdef};
use unpopped::oracle::kernel_eval::{KernelArg, run_kernel};
use unpopped::oracle::{TypedBuffer, evaluate};
use unpopped::plan::{PlanError, try_build_plan};
use unpopped_vocab::{ArchSku, AxisMask, ElementKind, OpCategory, OperandDesc, structure_key};

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

fn width(dtype: ElementKind) -> usize {
    if dtype == ElementKind::F64 { 8 } else { 4 }
}

fn encode(dtype: ElementKind, vals: &[f64]) -> Vec<u8> {
    vals.iter()
        .flat_map(|v| {
            if dtype == ElementKind::F64 {
                v.to_le_bytes().to_vec()
            } else {
                (*v as f32).to_le_bytes().to_vec()
            }
        })
        .collect()
}

fn decode(dtype: ElementKind, bytes: &[u8]) -> Vec<f64> {
    bytes
        .chunks(width(dtype))
        .map(|c| {
            if dtype == ElementKind::F64 {
                f64::from_le_bytes(c.try_into().unwrap())
            } else {
                f64::from(f32::from_le_bytes(c.try_into().unwrap()))
            }
        })
        .collect()
}

fn input_buffer(dtype: ElementKind) -> TypedBuffer {
    TypedBuffer::new(
        dtype,
        vec![ROWS, COLS],
        vec![COLS, 1],
        encode(dtype, &corpus()),
    )
}

/// The independent reference: the first element seeds, a NaN accumulator is replaced by anything, a NaN element is
/// skipped, otherwise a strictly better element wins (so the first of equal values is kept).
fn fold_num(xs: &[f64], max: bool) -> f64 {
    let mut it = xs.iter().copied();
    let mut acc = it.next().expect("a non-empty run");
    for x in it {
        let better = if max { x > acc } else { x < acc };
        if acc.is_nan() || (!x.is_nan() && better) {
            acc = x;
        }
    }
    acc
}

fn reference(rop: ReduceOp, axis0: bool) -> Vec<f64> {
    let c = corpus();
    let max = rop == ReduceOp::MaxNum;
    let cell = |r: i64, k: i64| c[(r * COLS + k) as usize];
    if axis0 {
        (0..COLS)
            .map(|k| fold_num(&(0..ROWS).map(|r| cell(r, k)).collect::<Vec<_>>(), max))
            .collect()
    } else {
        (0..ROWS)
            .map(|r| fold_num(&(0..COLS).map(|k| cell(r, k)).collect::<Vec<_>>(), max))
            .collect()
    }
}

/// Equal as the oracle means it: both NaN, or the same bits (so `0.0` and `-0.0` differ).
fn same(a: &[f64], b: &[f64], dtype: ElementKind) -> bool {
    let bits = |v: f64| {
        if dtype == ElementKind::F64 {
            v.to_bits()
        } else {
            u64::from((v as f32).to_bits())
        }
    };
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(x, y)| (x.is_nan() && y.is_nan()) || bits(*x) == bits(*y))
}

fn reduction(dtype: ElementKind, rop: ReduceOp, axis0: bool) -> OpDef {
    let mut axes = AxisMask::EMPTY;
    if axis0 {
        axes.set(0);
    }
    OpDef::reduction_axes("r", 1, &[dtype], Expr(input(0).0), rop, axes, false)
}

fn oracle(dtype: ElementKind, rop: ReduceOp, axis0: bool) -> Result<Vec<f64>, PlanError> {
    let op = reduction(dtype, rop, axis0);
    let inp = input_buffer(dtype);
    let n = if axis0 { COLS } else { ROWS };
    let a = OperandDesc::new(2, &inp.shape, &inp.strides, dtype, 256);
    let o = OperandDesc::new(1, &[n], &[1], dtype, 256);
    let key = structure_key(OpCategory::Reduction, &[a, o], ArchSku::Sm89);
    let plan = try_build_plan(&op, &key)?;
    let out = evaluate(&plan, &[a, o], std::slice::from_ref(&inp), &[])
        .remove(0)
        .bytes;
    Ok(decode(dtype, &out[..n as usize * width(dtype)]))
}

fn lifted(dtype: ElementKind, rop: ReduceOp, axis0: bool) -> Vec<f64> {
    let op = reduction(dtype, rop, axis0);
    let kernel = lift_reduction_opdef(&op, dtype, 2).expect("the NaN-ignoring reduction lifts");
    assert_eq!(check_kernel(&kernel), Ok(()));
    let n = if axis0 { COLS } else { ROWS };
    let len = n as usize * width(dtype);
    let out = TypedBuffer::new(dtype, vec![n], vec![1], vec![0u8; len]);
    let mut args = vec![
        KernelArg::Buffer(input_buffer(dtype)),
        KernelArg::Buffer(out),
        KernelArg::Int(i128::from(ROWS)),
        KernelArg::Int(i128::from(COLS)),
    ];
    run_kernel(&kernel, &mut args).expect("the kernel runs");
    let KernelArg::Buffer(out) = &args[1] else {
        unreachable!()
    };
    decode(dtype, &out.bytes[..len])
}

#[test]
fn the_oracle_matches_an_independent_fold() {
    for dtype in FLOATS {
        for rop in NUM {
            for axis0 in [false, true] {
                let got = oracle(dtype, rop, axis0).expect("plans");
                let want = reference(rop, axis0);
                assert!(
                    same(&got, &want, dtype),
                    "{dtype:?} {rop:?} axis0={axis0}: {got:?} vs {want:?}"
                );
            }
        }
    }
}

#[test]
fn the_lifted_kernel_agrees_with_the_oracle() {
    for dtype in FLOATS {
        for rop in NUM {
            for axis0 in [false, true] {
                let want = oracle(dtype, rop, axis0).expect("plans");
                let got = lifted(dtype, rop, axis0);
                assert!(
                    same(&got, &want, dtype),
                    "{dtype:?} {rop:?} axis0={axis0}: {got:?} vs {want:?}"
                );
            }
        }
    }
}

#[test]
fn the_corpus_tells_a_nan_ignoring_extremum_from_a_nan_propagating_one() {
    // Max keeps a NaN once seen: rows 1 and 2 give NaN, MaxNum gives 5 and 3. A fold that ignored the difference
    // would pass an agreement test on a corpus where the two coincide.
    let num = oracle(ElementKind::F32, ReduceOp::MaxNum, false).unwrap();
    let prop = oracle(ElementKind::F32, ReduceOp::Max, false).unwrap();
    assert_eq!((num[1], num[2]), (5.0, 3.0));
    assert!(prop[1].is_nan() && prop[2].is_nan());
    assert!(num[0].is_nan(), "a row of only NaN gives NaN");
    assert_eq!(
        oracle(ElementKind::F32, ReduceOp::MinNum, false).unwrap()[1..4],
        [-4.0, -7.0, -6.0]
    );
}

#[test]
fn an_integer_extremum_has_no_nan_so_it_equals_max_and_min() {
    let dtype = ElementKind::I32;
    let ints: Vec<u8> = (0..ROWS * COLS)
        .flat_map(|i| (((i * 7) % 11 - 5) as i32).to_le_bytes())
        .collect();
    let inp = TypedBuffer::new(dtype, vec![ROWS, COLS], vec![COLS, 1], ints);
    let run = |rop| {
        let op = reduction(dtype, rop, false);
        let a = OperandDesc::new(2, &inp.shape, &inp.strides, dtype, 256);
        let o = OperandDesc::new(1, &[ROWS], &[1], dtype, 256);
        let key = structure_key(OpCategory::Reduction, &[a, o], ArchSku::Sm89);
        let plan = try_build_plan(&op, &key).expect("plans");
        evaluate(&plan, &[a, o], std::slice::from_ref(&inp), &[])
            .remove(0)
            .bytes
    };
    assert_eq!(run(ReduceOp::MaxNum), run(ReduceOp::Max));
    assert_eq!(run(ReduceOp::MinNum), run(ReduceOp::Min));
}

#[test]
fn a_scan_declines_so_does_the_gate_for_anything_but_a_reduction() {
    let dtype = ElementKind::F32;
    let scan = OpDef::scan_simple("s", &[dtype], ReduceOp::MaxNum, 1, false, false);
    let inp = input_buffer(dtype);
    let a = OperandDesc::new(2, &inp.shape, &inp.strides, dtype, 256);
    let key = structure_key(OpCategory::Scan, &[a, a], ArchSku::Sm89);
    match try_build_plan(&scan, &key) {
        Err(PlanError::InadmissibleOpAtDtype { detail, .. }) => {
            assert!(detail.contains("Reduction only"), "{detail}");
        }
        other => panic!("expected a decline, got {:?}", other.map(|_| ())),
    }
    // and the kernel lift of a scan declines too
    assert!(matches!(
        lift_scan_opdef(&scan, dtype, 2),
        Err(LiftDecline::UnsupportedForm(_))
    ));
}
