// SPDX-License-Identifier: MIT OR Apache-2.0
//! **A running mean** (ROADMAP U2 D1; `docs/options-u2-lift-declines.md`): an `Access::Scan` whose combine is `Mean`.
//! Lane `j` of an inclusive scan is `(x_0 + ... + x_j) / (j + 1)`; lane `j` of an exclusive scan is the sum of the lanes
//! before it divided by `j`, so **lane 0 is `0 / 0`, NaN** (the mean of nothing). The sum is an `f64` fold from `0`, the
//! quotient is taken in `f64` and rounded once at the store. A reverse scan counts from the far end. Float only.
//!
//! This is an **Unpopped extension, not KISS**: `prefix_scan` takes the monoids `{sum, prod, max, min}`
//! (KISS-OPS-6.11-0003) and a running mean is not one; whether KISS accepts, declines or decomposes it is
//! ThinkersJournal/KISS#766, open, and the PM approved building it as an extension in the meantime.
//!
//! The oracle is held to a fold written here in Rust; the lifted kernel is held to the oracle byte for byte (NaN equal to
//! NaN); the launched lift and the integer scan decline, typed.

use unpopped::ir::{OpDef, ReduceOp};
use unpopped::kernel::check_kernel;
use unpopped::kernel_lift::{LiftDecline, lift_scan_opdef, lift_scan_opdef_launched};
use unpopped::oracle::kernel_eval::{KernelArg, run_kernel};
use unpopped::oracle::{TypedBuffer, evaluate};
use unpopped::plan::try_build_plan;
use unpopped_vocab::{ArchSku, ElementKind, OpCategory, OperandDesc, structure_key};

const ROWS: usize = 3;
const COLS: usize = 6;

/// Three rows: a plain run, an f32-cancelling run, a run with a NaN and an infinity.
const DATA: [[f64; COLS]; ROWS] = [
    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    [1.0e8, 1.0, -1.0e8, 0.5, 0.25, -2.0],
    [4.0, f64::NAN, 2.0, f64::INFINITY, -1.0, 3.0],
];

fn buffer(dtype: ElementKind) -> TypedBuffer {
    let bytes = DATA
        .iter()
        .flatten()
        .flat_map(|&v| match dtype {
            ElementKind::F32 => (v as f32).to_le_bytes().to_vec(),
            _ => v.to_le_bytes().to_vec(),
        })
        .collect();
    TypedBuffer::new(
        dtype,
        vec![ROWS as i64, COLS as i64],
        vec![COLS as i64, 1],
        bytes,
    )
}

fn op(dtype: ElementKind, reverse: bool, exclusive: bool) -> OpDef {
    OpDef::scan_simple("cummean", &[dtype], ReduceOp::Mean, 1, reverse, exclusive)
}

fn oracle(dtype: ElementKind, reverse: bool, exclusive: bool) -> Vec<u8> {
    let src = buffer(dtype);
    let a = OperandDesc::new(2, &src.shape, &src.strides, dtype, 256);
    let key = structure_key(OpCategory::Scan, &[a, a], ArchSku::Sm89);
    let o = op(dtype, reverse, exclusive);
    let plan = try_build_plan(&o, &key).expect("a Mean scan plans");
    evaluate(&plan, &[a, a], std::slice::from_ref(&src), &[])
        .remove(0)
        .bytes
}

fn lifted(dtype: ElementKind, reverse: bool, exclusive: bool, len: usize) -> Vec<u8> {
    let kernel = lift_scan_opdef(&op(dtype, reverse, exclusive), dtype, 2).expect("lifts");
    assert_eq!(check_kernel(&kernel), Ok(()));
    let out = TypedBuffer::new(
        dtype,
        vec![ROWS as i64, COLS as i64],
        vec![COLS as i64, 1],
        vec![0xAB; len],
    );
    let mut args = vec![
        KernelArg::Buffer(buffer(dtype)),
        KernelArg::Buffer(out),
        KernelArg::Int(ROWS as i128),
        KernelArg::Int(COLS as i128),
    ];
    run_kernel(&kernel, &mut args).expect("the kernel runs");
    let KernelArg::Buffer(got) = &args[1] else {
        unreachable!()
    };
    got.bytes[..len].to_vec()
}

/// The reference: per row an `f64` running sum in scan order, divided by the count so far.
fn reference(dtype: ElementKind, reverse: bool, exclusive: bool) -> Vec<f64> {
    let mut out = vec![0.0; ROWS * COLS];
    for (r, row) in DATA.iter().enumerate() {
        let vals: Vec<f64> = row
            .iter()
            .map(|&v| match dtype {
                ElementKind::F32 => f64::from(v as f32),
                _ => v,
            })
            .collect();
        let mut sum = 0.0f64;
        for step in 0..COLS {
            let j = if reverse { COLS - 1 - step } else { step };
            let count = if exclusive { step } else { step + 1 } as f64;
            if exclusive {
                out[r * COLS + j] = sum / count;
                sum += vals[j];
            } else {
                sum += vals[j];
                out[r * COLS + j] = sum / count;
            }
        }
    }
    out
}

fn as_f64(dtype: ElementKind, b: &[u8]) -> Vec<f64> {
    match dtype {
        ElementKind::F32 => b
            .chunks(4)
            .map(|c| f64::from(f32::from_le_bytes(c.try_into().unwrap())))
            .collect(),
        _ => b
            .chunks(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
            .collect(),
    }
}

fn same(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(x, y)| x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan()))
}

#[test]
fn the_oracle_matches_a_running_mean_written_in_rust() {
    for dtype in [ElementKind::F64, ElementKind::F32] {
        for (reverse, exclusive) in [(false, false), (false, true), (true, false), (true, true)] {
            let got = as_f64(dtype, &oracle(dtype, reverse, exclusive));
            let mut want = reference(dtype, reverse, exclusive);
            if dtype == ElementKind::F32 {
                // rounded once at the store
                want = want.iter().map(|&v| f64::from(v as f32)).collect();
            }
            assert!(
                same(&got, &want),
                "{dtype:?} reverse={reverse} exclusive={exclusive}: {got:?} vs {want:?}"
            );
        }
    }
}

#[test]
fn the_corpus_pins_the_rule() {
    // hand-computed: row 0 inclusive means 1, 1.5, 2, 2.5, 3, 3.5; exclusive lane 0 is NaN then 1, 1.5, 2, 2.5, 3
    let inc = as_f64(ElementKind::F64, &oracle(ElementKind::F64, false, false));
    assert_eq!(&inc[..6], [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]);
    let exc = as_f64(ElementKind::F64, &oracle(ElementKind::F64, false, true));
    assert!(exc[0].is_nan());
    assert_eq!(&exc[1..6], [1.0, 1.5, 2.0, 2.5, 3.0]);
    // reverse inclusive counts from the far end: 6, 5.5, 5, 4.5, 4, 3.5
    let rev = as_f64(ElementKind::F64, &oracle(ElementKind::F64, true, false));
    assert_eq!(&rev[..6], [3.5, 4.0, 4.5, 5.0, 5.5, 6.0]);
}

#[test]
fn the_lifted_kernel_agrees_with_the_oracle_byte_for_byte() {
    for dtype in [ElementKind::F64, ElementKind::F32] {
        for (reverse, exclusive) in [(false, false), (false, true), (true, false), (true, true)] {
            let want = oracle(dtype, reverse, exclusive);
            let got = lifted(dtype, reverse, exclusive, want.len());
            assert!(
                same(&as_f64(dtype, &got), &as_f64(dtype, &want)),
                "{dtype:?} reverse={reverse} exclusive={exclusive}"
            );
        }
    }
}

#[test]
fn an_integer_mean_scan_and_a_launched_one_decline() {
    let int = op(ElementKind::I32, false, false);
    assert_eq!(
        lift_scan_opdef(&int, ElementKind::I32, 2),
        Err(LiftDecline::UnsupportedForm("an integer Mean"))
    );
    let float = op(ElementKind::F32, false, false);
    assert!(matches!(
        lift_scan_opdef_launched(&float, ElementKind::F32, 2, 8, 2),
        Err(LiftDecline::UnsupportedForm(m)) if m.contains("Mean")
    ));
}
