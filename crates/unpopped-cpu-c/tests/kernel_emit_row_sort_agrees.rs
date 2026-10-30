// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The emitted C agrees with the kernel interpreter on row sorts**
//! (`docs/e1-emit-from-the-kernel-layer.md`, stage E1b): lifted `Access::RowSort` kernels (every
//! `SortOut`, full and capped, both directions, `F32` and `F64`, rank 2 and 3) are lowered by
//! `CpuC::lower_kernel`, compiled, run, and compared with `run_kernel`: values bit-exact except
//! NaN sign/payload, indices exactly.
//!
//! The corpus has ties (a guaranteed one), both zeros, infinities and NaNs of several signs and
//! payloads, because the order is a NaN-greatest total order with a stable tie-break and a
//! friendly corpus would pass a comparator that got any of those wrong. The serial C is a
//! correctness reference; nothing is timed.

use unpopped::backend::Backend;
use unpopped::ir::{Access, OpDef, SortLimit, SortOrder, SortOut};
use unpopped::kernel::{KernelFn, check_kernel};
use unpopped::kernel_lift::lift_row_sort_opdef;
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped_cpu_c::CpuC;
use unpopped_vocab::ElementKind;

#[path = "support/cc.rs"]
mod cc;
#[path = "support/kernel_data.rs"]
mod kernel_data;
#[path = "support/kernel_run.rs"]
mod kernel_run;
use kernel_data::zeroed;
use kernel_run::{F32, F64, agree, run_c_mutated};

const I32: ElementKind = ElementKind::I32;

/// Ties, both zeros, infinities, and NaNs with distinct payloads and signs.
const POOL: [u32; 11] = [
    0x3f80_0000, // 1.0
    0x4000_0000, // 2.0
    0xc000_0000, // -2.0
    0x0000_0000, // +0.0
    0x8000_0000, // -0.0
    0x7f80_0000, // +inf
    0xff80_0000, // -inf
    0x7fc0_0001, // NaN, quiet, payload 1
    0x7f80_0001, // NaN, signalling
    0xffc0_0000, // NaN, negative
    0x3f80_0000, // 1.0 again: a guaranteed tie
];

fn input(dtype: ElementKind, shape: &[i64]) -> TypedBuffer {
    let n = shape.iter().product::<i64>() as usize;
    let bytes: Vec<u8> = (0..n)
        .flat_map(|i| {
            let bits = POOL[(i * 7 + (i / 5) * 3 + i * i) % POOL.len()];
            if dtype == F64 {
                f64::from(f32::from_bits(bits)).to_le_bytes().to_vec()
            } else {
                bits.to_le_bytes().to_vec()
            }
        })
        .collect();
    let strides: Vec<i64> = (0..shape.len())
        .map(|d| shape[d + 1..].iter().product())
        .collect();
    TypedBuffer::new(dtype, shape.to_vec(), strides, bytes)
}

fn sort_op(dtype: ElementKind, order: SortOrder, out: SortOut, limit: SortLimit) -> OpDef {
    let mut op = match (out, limit) {
        (SortOut::Values, _) => OpDef::row_sort("s", dtype, order),
        (SortOut::Indices, _) => OpDef::row_argsort("s", dtype, order),
        (SortOut::Both, SortLimit::Full) => OpDef::row_sort_indices("s", dtype, order),
        (SortOut::Both, SortLimit::TopK) if order == SortOrder::Desc => OpDef::row_topk("s", dtype),
        (SortOut::Both, SortLimit::TopK) => OpDef::row_bottomk("s", dtype),
    };
    if let Access::RowSort { limit: l, .. } = &mut op.access {
        *l = limit;
    }
    op
}

/// The lifted kernel and its arguments. `k_out` caps the output rows' length.
fn build(
    dtype: ElementKind,
    order: SortOrder,
    out: SortOut,
    k_out: Option<i64>,
    shape: &[i64],
) -> (KernelFn, Vec<KernelArg>) {
    let limit = if k_out.is_some() {
        SortLimit::TopK
    } else {
        SortLimit::Full
    };
    let op = sort_op(dtype, order, out, limit);
    let k = lift_row_sort_opdef(&op, dtype, shape.len() as u8).expect("the row sort lifts");
    assert_eq!(check_kernel(&k), Ok(()));
    let mut out_shape = shape.to_vec();
    *out_shape.last_mut().unwrap() = k_out.unwrap_or(*shape.last().unwrap());

    let mut args = vec![KernelArg::Buffer(input(dtype, shape))];
    if out != SortOut::Indices {
        args.push(KernelArg::Buffer(zeroed(dtype, &out_shape)));
    }
    if out != SortOut::Values {
        args.push(KernelArg::Buffer(zeroed(I32, &out_shape)));
    }
    args.extend(shape.iter().map(|&e| KernelArg::Int(i128::from(e))));
    args.extend(k_out.map(|k| KernelArg::Int(i128::from(k))));
    (k, args)
}

/// Lift the sort and compare C with the interpreter.
fn check(dtype: ElementKind, order: SortOrder, out: SortOut, k_out: Option<i64>, shape: &[i64]) {
    let (k, args) = build(dtype, order, out, k_out, shape);
    let tag = format!(
        "sort_{dtype:?}_{order:?}_{out:?}_{}_{}",
        k_out.unwrap_or(0),
        shape.len()
    );
    agree(&tag, &k, args);
}

#[test]
fn full_sorts_agree_in_every_form() {
    for dtype in [F32, F64] {
        for order in [SortOrder::Asc, SortOrder::Desc] {
            for out in [SortOut::Values, SortOut::Indices, SortOut::Both] {
                check(dtype, order, out, None, &[4, 11]);
            }
        }
    }
    check(F32, SortOrder::Asc, SortOut::Both, None, &[2, 3, 7]);
}

#[test]
fn capped_sorts_agree() {
    for order in [SortOrder::Asc, SortOrder::Desc] {
        for k_out in [1, 3, 11] {
            check(F32, order, SortOut::Both, Some(k_out), &[4, 11]);
        }
    }
    check(F64, SortOrder::Desc, SortOut::Values, Some(4), &[3, 9]);
}

#[test]
fn degenerate_rows_agree() {
    check(F32, SortOrder::Asc, SortOut::Both, None, &[3, 1]);
    check(F32, SortOrder::Desc, SortOut::Both, None, &[1, 2]);
}

#[test]
fn the_corpus_catches_a_wrong_comparator() {
    // Seeded mutants of the emitted comparator of an ascending value+index sort: each must change
    // the output bytes of the corpus (the unmutated C agrees with the interpreter above), or the
    // corpus could not tell a wrong tie-break or NaN rule from a right one.
    let (k, args) = build(F32, SortOrder::Asc, SortOut::Both, None, &[4, 11]);
    let src = CpuC.lower_kernel(&k).expect("the sort lowers").source;
    let outs = |tag: &str, from: &str, to: &str| {
        run_c_mutated(tag, &k, &args, |s| {
            if from.is_empty() {
                return s;
            }
            assert!(s.contains(from), "the anchor `{from}` is in the emitted C");
            assert_eq!(s, src, "the mutation starts from the emitter's own output");
            s.replace(from, to)
        })
        .expect("a host C compiler")
        .outs
    };
    let base = outs("sort_mut_base", "", "");
    for (tag, from, to) in [
        (
            "sort_mut_le",
            "(float)l9 <= (float)l8",
            "(float)l9 < (float)l8",
        ),
        (
            "sort_mut_lt",
            "(float)l9 < (float)l8",
            "(float)l9 <= (float)l8",
        ),
        (
            "sort_mut_nan",
            "(float)l8 != (float)l8",
            "(float)l8 == (float)l8",
        ),
    ] {
        assert_ne!(
            outs(tag, from, to),
            base,
            "{tag}: the mutant went unnoticed"
        );
    }
}
