// SPDX-License-Identifier: MIT OR Apache-2.0
//! **A `Prod` stage in `Access::RowReduce`** (ROADMAP U2 D3; `docs/options-u2-lift-declines.md`): the stage multiplies
//! the per-element `pre` values along the row from the multiplicative identity `1` (KISS-OPS-6.11-0002), in an `F64`
//! accumulator, and rounds once to the compute dtype, as the other stages do. The oracle is held to a product written
//! here in Rust; the serial kernel lift is held to the oracle (byte for byte); a launched row reduction still
//! declines a `Prod` stage, typed, because a float product is not order-independent.
//!
//! The corpus has a zero (the product is `-0`), a zero beside an infinity (`NaN`), an odd count of negatives, all
//! ones, a one-element row (the identity must not leak in) and an `F32` overflow (the `F64` accumulator holds it, the
//! single rounding to `F32` makes it infinite).

use unpopped::ir::{Expr, OpDef, ReduceOp, ReduceStage, input, reduced};
use unpopped::kernel::check_kernel;
use unpopped::kernel_lift::{LiftDecline, lift_row_reduce_opdef, lift_row_reduce_opdef_launched};
use unpopped::oracle::kernel_eval::{KernelArg, run_kernel};
use unpopped::oracle::{TypedBuffer, evaluate};
use unpopped::plan::try_build_plan;
use unpopped_vocab::{ArchSku, ElementKind, OpCategory, OperandDesc, try_structure_key};

const F32: ElementKind = ElementKind::F32;
const F64: ElementKind = ElementKind::F64;
const K: i64 = 5;

const ROWS: [[f64; 5]; 6] = [
    [2.0, -3.0, 0.5, 4.0, -1.5],
    [2.0, 0.0, 3.0, -4.0, 5.0],
    [f64::INFINITY, 0.0, 1.0, 2.0, 3.0],
    [-1.0, -2.0, -3.0, -4.0, -5.0],
    [1.0, 1.0, 1.0, 1.0, 1.0],
    [1.0e20, 1.0e20, 1.0e3, 1.0, 1.0],
];

fn buffer(dtype: ElementKind, rows: i64, k: i64, vals: &[f64]) -> TypedBuffer {
    let bytes = vals
        .iter()
        .flat_map(|&v| match dtype {
            F32 => (v as f32).to_le_bytes().to_vec(),
            _ => v.to_le_bytes().to_vec(),
        })
        .collect();
    TypedBuffer::new(dtype, vec![rows, k], vec![k, 1], bytes)
}

fn corpus(dtype: ElementKind) -> TypedBuffer {
    let vals: Vec<f64> = ROWS.iter().flatten().copied().collect();
    buffer(dtype, ROWS.len() as i64, K, &vals)
}

fn prod_op(dtype: ElementKind, epilogue: Expr) -> OpDef {
    let stage = ReduceStage {
        pre: input(0).0,
        op: ReduceOp::Prod,
    };
    OpDef::row_reduce("rr", 1, &[dtype], vec![stage], epilogue)
}

fn oracle(op: &OpDef, dtype: ElementKind, src: &TypedBuffer) -> Vec<u8> {
    let (rows, k) = (src.shape[0], src.shape[1]);
    let desc = OperandDesc::new(2, &[rows, k], &[k, 1], dtype, 256);
    let ops = [desc, desc];
    let key = try_structure_key(OpCategory::Softmax, &ops, ArchSku::Sm89).expect("a key");
    let plan = try_build_plan(op, &key).expect("a Prod stage plans");
    evaluate(&plan, &ops, std::slice::from_ref(src), &[])
        .remove(0)
        .bytes
}

fn lifted(op: &OpDef, dtype: ElementKind, src: &TypedBuffer, len: usize) -> Vec<u8> {
    let (rows, k) = (src.shape[0], src.shape[1]);
    let kernel = lift_row_reduce_opdef(op, dtype, 2).expect("a Prod stage lifts");
    assert_eq!(check_kernel(&kernel), Ok(()));
    let out = TypedBuffer::new(dtype, vec![rows, k], vec![k, 1], vec![0xAB; len]);
    let mut args = vec![
        KernelArg::Buffer(src.clone()),
        KernelArg::Buffer(out),
        KernelArg::Int(i128::from(rows)),
        KernelArg::Int(i128::from(k)),
    ];
    run_kernel(&kernel, &mut args).expect("the kernel runs");
    let KernelArg::Buffer(got) = &args[1] else {
        unreachable!()
    };
    got.bytes[..len].to_vec()
}

/// The product of each row, left to right in `f64` from `1`, rounded once to `dtype`, broadcast over the row.
fn reference(dtype: ElementKind) -> Vec<u8> {
    let mut out = Vec::new();
    for row in &ROWS {
        let p = row.iter().fold(1.0f64, |acc, &x| acc * x);
        for _ in 0..K {
            match dtype {
                F32 => out.extend((p as f32).to_le_bytes()),
                _ => out.extend(p.to_le_bytes()),
            }
        }
    }
    out
}

fn f32s(b: &[u8]) -> Vec<f32> {
    b.chunks(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn f64s(b: &[u8]) -> Vec<f64> {
    b.chunks(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

#[test]
fn the_oracle_matches_a_product_written_in_rust() {
    // 0 * inf is a NaN whose payload is the host's in both, so the bytes are compared
    for dtype in [F64, F32] {
        let got = oracle(&prod_op(dtype, reduced(0)), dtype, &corpus(dtype));
        let want = reference(dtype);
        if dtype == F64 {
            let (g, w) = (f64s(&got), f64s(&want));
            for (i, (a, b)) in g.iter().zip(&w).enumerate() {
                assert!(
                    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()),
                    "{dtype:?} element {i}: {a} vs {b}"
                );
            }
        } else {
            let (g, w) = (f32s(&got), f32s(&want));
            for (i, (a, b)) in g.iter().zip(&w).enumerate() {
                assert!(
                    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()),
                    "{dtype:?} element {i}: {a} vs {b}"
                );
            }
        }
    }
}

#[test]
fn the_corpus_pins_the_product() {
    // hand-computed: 2*-3*0.5*4*-1.5 = 18; a zero row 2*0*3*-4*5 = -0; 0*inf = NaN; five negatives = -120; ones = 1;
    // the overflow row is 1e43 in F64 and infinite once rounded to F32
    let w = f64s(&oracle(&prod_op(F64, reduced(0)), F64, &corpus(F64)));
    let row = |r: usize| w[r * K as usize];
    assert_eq!(row(0), 18.0);
    assert_eq!(row(1).to_bits(), (-0.0f64).to_bits());
    assert!(row(2).is_nan());
    assert_eq!(row(3), -120.0);
    assert_eq!(row(4), 1.0);
    assert_eq!(row(5), 1.0e43);
    let w32 = f32s(&oracle(&prod_op(F32, reduced(0)), F32, &corpus(F32)));
    assert_eq!(w32[5 * K as usize], f32::INFINITY);
}

#[test]
fn a_one_element_row_is_that_element() {
    // the identity 1 must not leak into the answer
    let src = buffer(F64, 3, 1, &[7.5, -2.0, 0.0]);
    assert_eq!(
        f64s(&oracle(&prod_op(F64, reduced(0)), F64, &src)),
        [7.5, -2.0, 0.0]
    );
}

#[test]
fn the_serial_lift_agrees_with_the_oracle_byte_for_byte() {
    for dtype in [F64, F32] {
        let src = corpus(dtype);
        let op = prod_op(dtype, reduced(0));
        let want = oracle(&op, dtype, &src);
        assert_eq!(lifted(&op, dtype, &src, want.len()), want, "{dtype:?}");
    }
}

#[test]
fn a_product_stage_feeds_a_later_epilogue() {
    // x / prod(row) in F64: the first row's product is 18
    let op = prod_op(F64, input(0) / reduced(0));
    let vals = [2.0, -3.0, 0.5, 4.0, -1.5, 1.0, 2.0, 3.0, 4.0, 5.0];
    let src = buffer(F64, 2, K, &vals);
    let want = oracle(&op, F64, &src);
    assert_eq!(lifted(&op, F64, &src, want.len()), want);
    assert_eq!(f64s(&want)[0], 2.0 / 18.0);
    assert_eq!(f64s(&want)[5], 1.0 / 120.0);
}

#[test]
fn a_launched_row_reduction_still_declines_a_product() {
    let op = prod_op(F32, reduced(0));
    assert!(matches!(
        lift_row_reduce_opdef_launched(&op, F32, 2, 8, 2),
        Err(LiftDecline::UnsupportedForm(m)) if m.contains("Prod")
    ));
}
