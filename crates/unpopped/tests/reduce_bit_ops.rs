// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Bitwise reductions** (ROADMAP U5 P2, PM ruling 2026-11-02; `docs/options-u5-collectives-and-richer-reductions.md`):
//! `ReduceOp::{And, Or, Xor}` fold integers in an `Access::Reduction`. Three things must agree byte for byte:
//! the oracle, the lifted kernel run by the interpreter (`I32`, `I64`), and a fold written here in Rust with no
//! identity literal (the first element seeds it). The corpus has an all-ones row, a row with a zero (an `And`
//! that started at `0` instead of the first element answers `0` everywhere), negative values (sign), and an
//! odd and an even count of equal elements (a `Xor` that dropped an element is wrong on exactly one of them).
//!
//! The ops that are shapes only (`MaxNum`, `MinNum`) and the forms that do not fold bits (a float input, a
//! scan, a window) decline at the plan gate, typed.

use unpopped::ir::{Access, Expr, OpDef, ReduceOp, input};
use unpopped::kernel::check_kernel;
use unpopped::kernel_lift::{LiftDecline, lift_reduction_opdef};
use unpopped::oracle::kernel_eval::{KernelArg, run_kernel};
use unpopped::oracle::{TypedBuffer, evaluate};
use unpopped::plan::{PlanError, try_build_plan};
use unpopped_vocab::{ArchSku, AxisMask, ElementKind, OpCategory, OperandDesc, structure_key};

const ROWS: i64 = 5;
const COLS: i64 = 6;
const BITS: [ReduceOp; 3] = [ReduceOp::And, ReduceOp::Or, ReduceOp::Xor];

fn width(dtype: ElementKind) -> usize {
    match dtype {
        ElementKind::I64 | ElementKind::U64 => 8,
        _ => 4,
    }
}

/// Five rows of six, the same bit patterns at every width (as `i128`, truncated to the width on the way in).
fn corpus() -> Vec<i128> {
    vec![
        -1, -1, -1, -1, -1, -1, //  all ones
        0b1011, 0b0110, 0, 0b1111, 0b1011, 0b0001, //  a zero in the middle
        5, 5, 5, 5, 5, 5, //  an even count of equal elements
        7, 7, 7, 7, 7, 1, //  a mix
        -8, 3, -1, 0x7fff, -0x8000, 12, //  signs
    ]
}

fn input_buffer(dtype: ElementKind) -> TypedBuffer {
    let w = width(dtype);
    let bytes: Vec<u8> = corpus()
        .iter()
        .flat_map(|v| v.to_le_bytes()[..w].to_vec())
        .collect();
    TypedBuffer::new(dtype, vec![ROWS, COLS], vec![COLS, 1], bytes)
}

/// The independent reference: row `r` (last axis) or column `c` (axis 0), folded with the first element as seed.
fn reference(dtype: ElementKind, rop: ReduceOp, axis0: bool) -> Vec<u8> {
    let c = corpus();
    let fold = |xs: Vec<i128>| {
        xs.into_iter()
            .reduce(|a, b| match rop {
                ReduceOp::And => a & b,
                ReduceOp::Or => a | b,
                _ => a ^ b,
            })
            .expect("a non-empty run")
    };
    let cell = |r: i64, k: i64| c[(r * COLS + k) as usize];
    let runs: Vec<Vec<i128>> = if axis0 {
        (0..COLS)
            .map(|k| (0..ROWS).map(|r| cell(r, k)).collect())
            .collect()
    } else {
        (0..ROWS)
            .map(|r| (0..COLS).map(|k| cell(r, k)).collect())
            .collect()
    };
    runs.into_iter()
        .flat_map(|run| fold(run).to_le_bytes()[..width(dtype)].to_vec())
        .collect()
}

fn reduction(dtype: ElementKind, rop: ReduceOp, axis0: bool) -> OpDef {
    let mut axes = AxisMask::EMPTY;
    if axis0 {
        axes.set(0);
    }
    OpDef::reduction_axes("r", 1, &[dtype], Expr(input(0).0), rop, axes, false)
}

fn plan_and_run(dtype: ElementKind, rop: ReduceOp, axis0: bool) -> Result<Vec<u8>, PlanError> {
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
    Ok(out[..n as usize * width(dtype)].to_vec())
}

fn lifted(dtype: ElementKind, rop: ReduceOp, axis0: bool) -> Vec<u8> {
    let op = reduction(dtype, rop, axis0);
    let kernel = lift_reduction_opdef(&op, dtype, 2).expect("the bit reduction lifts");
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
    out.bytes[..len].to_vec()
}

#[test]
fn the_oracle_matches_an_independent_fold_at_every_integer_width() {
    for dtype in [
        ElementKind::I32,
        ElementKind::I64,
        ElementKind::U32,
        ElementKind::U64,
    ] {
        for rop in BITS {
            for axis0 in [false, true] {
                let got = plan_and_run(dtype, rop, axis0).expect("a bit reduction plans");
                assert_eq!(
                    got,
                    reference(dtype, rop, axis0),
                    "{dtype:?} {rop:?} axis0={axis0}"
                );
            }
        }
    }
}

#[test]
fn the_lifted_kernel_agrees_with_the_oracle() {
    for dtype in [ElementKind::I32, ElementKind::I64] {
        for rop in BITS {
            for axis0 in [false, true] {
                let want = plan_and_run(dtype, rop, axis0).expect("plans");
                assert_eq!(
                    lifted(dtype, rop, axis0),
                    want,
                    "{dtype:?} {rop:?} axis0={axis0}"
                );
            }
        }
    }
}

#[test]
fn the_corpus_tells_the_three_ops_apart() {
    // a fold that ignored its op would pass an agreement test on a corpus where the ops coincide
    let all: Vec<Vec<u8>> = BITS
        .iter()
        .map(|rop| reference(ElementKind::I32, *rop, false))
        .collect();
    assert!(all[0] != all[1] && all[1] != all[2] && all[0] != all[2]);
}

fn declined(result: Result<Vec<u8>, PlanError>) -> String {
    match result {
        Err(PlanError::InadmissibleOpAtDtype { detail, .. }) => detail,
        other => panic!("expected an InadmissibleOpAtDtype decline, got {other:?}"),
    }
}

#[test]
fn a_float_input_declines_at_the_plan_gate() {
    for rop in BITS {
        let d = declined(plan_and_run(ElementKind::F32, rop, false));
        assert!(d.contains("integer dtype"), "{d}");
    }
}

#[test]
fn a_bit_op_in_a_scan_or_a_window_declines() {
    let dtype = ElementKind::I32;
    let scan = OpDef::scan_simple("s", &[dtype], ReduceOp::Xor, 1, false, false);
    assert!(matches!(scan.access, Access::Scan { .. }));
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
fn the_kernel_lift_declines_what_it_does_not_fold() {
    // a float bit reduction never reaches a kernel
    for (dtype, rop) in [
        (ElementKind::F32, ReduceOp::And),
        (ElementKind::F32, ReduceOp::Xor),
    ] {
        let op = reduction(dtype, rop, false);
        assert!(
            matches!(
                lift_reduction_opdef(&op, dtype, 2),
                Err(LiftDecline::UnsupportedForm(_))
            ),
            "{dtype:?} {rop:?}"
        );
    }
}
