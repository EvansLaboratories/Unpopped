// SPDX-License-Identifier: MIT OR Apache-2.0
//! ROADMAP E19 tail: the heterogeneous-output `count`, `any` and `all` reductions at a narrow integer
//! dtype through `oracle::evaluate`. 0.40.16 let `oracle::evaluate` past its dtype check for `I8`/`U8`/
//! `I16`/`U16` but only tested the same-dtype folds (`tests/oracle_narrow_int_reduction.rs`); these are
//! the shapes the plan gate's rule 4 admits (`count` is `Sum` of `in != 0` with an `I64` output; `any`
//! and `all` fold `in != 0` with `Max`/`Min` and a `Reduced(0) != 0` post, `U8` output). The references
//! are computed on the host from the same values, sharing no code with the oracle.

use unpopped::ir::{BinaryOp, OpDef, ReduceOp, input, konst, reduced};
use unpopped::oracle::{TypedBuffer, evaluate};
use unpopped::plan::try_build_plan;
use unpopped_vocab::{ArchSku, ElementKind, OpCategory, OperandDesc, structure_key};

#[derive(Clone, Copy, Debug)]
enum Shape {
    Count,
    Any,
    All,
}

fn op(shape: Shape, dt: ElementKind) -> OpDef {
    let nz = || input(0).binary(BinaryOp::CmpNe, konst(0.0));
    match shape {
        Shape::Count => {
            let mut o = OpDef::reduction("count", 1, &[dt], nz(), ReduceOp::Sum);
            o.out_dtype = Some(ElementKind::I64);
            o
        }
        Shape::Any | Shape::All => {
            let rop = if matches!(shape, Shape::Any) {
                ReduceOp::Max
            } else {
                ReduceOp::Min
            };
            let post = reduced(0).binary(BinaryOp::CmpNe, konst(0.0));
            let mut o = OpDef::reduction_post("pred", 1, &[dt], nz(), rop, post);
            o.out_dtype = Some(ElementKind::U8);
            o
        }
    }
}

/// The oracle's answer for one row of `vals` (little-endian, `width` bytes each).
fn run(shape: Shape, dt: ElementKind, vals: &[i128], width: usize) -> i128 {
    let n = vals.len() as i64;
    let bytes: Vec<u8> = vals
        .iter()
        .flat_map(|v| v.to_le_bytes()[..width].to_vec())
        .collect();
    let inp = TypedBuffer::new(dt, vec![1, n], vec![n, 1], bytes);
    let out_dt = if matches!(shape, Shape::Count) {
        ElementKind::I64
    } else {
        ElementKind::U8
    };
    let d_in = OperandDesc::new(2, &[1, n], &[n, 1], dt, 256);
    let d_out = OperandDesc::new(1, &[1], &[1], out_dt, 256);
    let descs = [d_in, d_out];
    let key = structure_key(OpCategory::Reduction, &descs, ArchSku::Sm89);
    let def = op(shape, dt);
    let plan = try_build_plan(&def, &key).expect("a narrow predicate reduction plans");
    let out = evaluate(&plan, &descs, &[inp], &[]).remove(0).bytes;
    let mut buf = [0u8; 16];
    let w = if matches!(shape, Shape::Count) { 8 } else { 1 };
    buf[..w].copy_from_slice(&out[..w]);
    i128::from_le_bytes(buf)
}

fn host(shape: Shape, vals: &[i128]) -> i128 {
    let nonzero = vals.iter().filter(|v| **v != 0).count();
    match shape {
        Shape::Count => nonzero as i128,
        Shape::Any => i128::from(nonzero > 0),
        Shape::All => i128::from(nonzero == vals.len()),
    }
}

#[test]
fn count_any_all_agree_with_the_host_at_every_narrow_dtype() {
    let cases: [(ElementKind, usize); 4] = [
        (ElementKind::I8, 1),
        (ElementKind::U8, 1),
        (ElementKind::I16, 2),
        (ElementKind::U16, 2),
    ];
    // Rows: all zero, one nonzero, all nonzero, and values that are nonzero only in a high byte.
    let rows: [&[i128]; 5] = [&[0, 0, 0], &[0, 5, 0], &[1, 2, 3], &[0], &[7]];
    for (dt, width) in cases {
        for shape in [Shape::Count, Shape::Any, Shape::All] {
            for row in rows {
                assert_eq!(
                    run(shape, dt, row, width),
                    host(shape, row),
                    "{shape:?} at {dt:?} over {row:?}"
                );
            }
        }
    }
}

#[test]
fn a_value_nonzero_only_above_the_low_byte_is_still_nonzero_at_16_bits() {
    for dt in [ElementKind::I16, ElementKind::U16] {
        let row: &[i128] = &[0x0100, 0, 0x4000];
        assert_eq!(run(Shape::Count, dt, row, 2), 2, "{dt:?}");
        assert_eq!(run(Shape::Any, dt, row, 2), 1, "{dt:?}");
        assert_eq!(run(Shape::All, dt, row, 2), 0, "{dt:?}");
    }
}
