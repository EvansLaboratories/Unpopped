// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The layers agree on row sorts, byte for byte** (`docs/ir-kernel-layer.md` section 7). The
//! oracle sorts each row with a stable `sort_by` under a NaN-greatest total order; the lifted
//! kernel counts, for each element, how many elements precede it under that order and stores the
//! element at that rank (`lift_row_sort_opdef`). The two share no algorithm, so a byte-identical
//! answer on a corpus of ties, NaNs (with payloads), signed zeros and infinities is evidence.
//!
//! Every `SortOut` and `SortLimit`, both directions, `F32` and `F64`, rank 2 and rank 3.

use unpopped::ir::{Access, OpDef, SortLimit, SortOrder, SortOut};
use unpopped::kernel::check_kernel;
use unpopped::kernel_lift::{LiftDecline, lift_row_sort_opdef};
use unpopped::oracle::kernel_eval::{KernelArg, run_kernel};
use unpopped::oracle::{TypedBuffer, evaluate};
use unpopped::plan::try_build_plan;
use unpopped::types::int_range;
use unpopped_vocab::{ArchSku, ElementKind, OpCategory, OperandDesc, try_structure_key};

/// Raw `f32` bit patterns: ties, both zeros, infinities, and NaNs with distinct payloads and signs
/// (a quiet one, a signalling one, a negative one).
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

/// The `i`th corpus element as raw bits of `dtype`. An `F64` element is the `F32` value widened,
/// except that a NaN keeps a payload of its own so the bits being moved are distinguishable.
fn element(dtype: ElementKind, i: usize) -> Vec<u8> {
    let slot = (i * 7 + (i / 5) * 3 + i * i) % POOL.len();
    let bits = POOL[slot];
    match dtype {
        ElementKind::F32 => bits.to_le_bytes().to_vec(),
        ElementKind::F64 => {
            let f = f32::from_bits(bits);
            let wide = if f.is_nan() {
                0x7ff0_0000_0000_0000_u64
                    | u64::from(bits & 0x7f_ffff)
                    | (u64::from(bits >> 31) << 63)
            } else {
                f64::from(f).to_bits()
            };
            wide.to_le_bytes().to_vec()
        }
        _ => {
            let (lo, hi) = int_range(dtype).expect("an integer dtype");
            // Ties, both extremes, zero, and (signed) minus one: the values where a signed read of
            // an unsigned key, or the reverse, changes the order.
            let pool = [
                hi,
                lo,
                0,
                7,
                hi - 1,
                lo + 1,
                7,
                if lo < 0 { -1 } else { hi / 2 },
                0,
                lo,
                1,
            ];
            let v = pool[slot];
            v.to_le_bytes()[..width(dtype)].to_vec()
        }
    }
}

fn width(dtype: ElementKind) -> usize {
    match dtype {
        ElementKind::I8 | ElementKind::U8 => 1,
        ElementKind::I16 | ElementKind::U16 => 2,
        ElementKind::F64 => 8,
        _ => 4,
    }
}

fn src(dtype: ElementKind, shape: &[i64]) -> TypedBuffer {
    let n: i64 = shape.iter().product();
    let bytes: Vec<u8> = (0..n as usize).flat_map(|i| element(dtype, i)).collect();
    let strides: Vec<i64> = (0..shape.len())
        .map(|d| shape[d + 1..].iter().product())
        .collect();
    TypedBuffer::new(dtype, shape.to_vec(), strides, bytes)
}

fn desc(dtype: ElementKind, shape: &[i64]) -> OperandDesc {
    let strides: Vec<i64> = (0..shape.len())
        .map(|d| shape[d + 1..].iter().product())
        .collect();
    OperandDesc::new(shape.len(), shape, &strides, dtype, 256)
}

fn contiguous(dtype: ElementKind, shape: &[i64], fill: u8) -> TypedBuffer {
    let n: i64 = shape.iter().product();
    let size = width(dtype);
    let strides: Vec<i64> = (0..shape.len())
        .map(|d| shape[d + 1..].iter().product())
        .collect();
    TypedBuffer::new(
        dtype,
        shape.to_vec(),
        strides,
        vec![fill; n as usize * size],
    )
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

/// The operand list the plan expects (input, then the value and/or index outputs) and the
/// pre-filled output buffers the kernel writes.
fn operands(
    dtype: ElementKind,
    out: SortOut,
    shape: &[i64],
    out_shape: &[i64],
) -> (Vec<OperandDesc>, Vec<TypedBuffer>) {
    let mut ops = vec![desc(dtype, shape)];
    let mut out_bufs: Vec<TypedBuffer> = Vec::new();
    if out != SortOut::Indices {
        ops.push(desc(dtype, out_shape));
        out_bufs.push(contiguous(dtype, out_shape, 0xAB));
    }
    if out != SortOut::Values {
        ops.push(desc(ElementKind::I32, out_shape));
        out_bufs.push(contiguous(ElementKind::I32, out_shape, 0xAB));
    }
    (ops, out_bufs)
}

/// Lift `op`, run it on `input` and the pre-filled outputs, and return every argument.
fn run_lifted(
    op: &OpDef,
    input: TypedBuffer,
    out_bufs: Vec<TypedBuffer>,
    shape: &[i64],
    k_out: Option<i64>,
) -> Vec<KernelArg> {
    let dtype = input.dtype;
    let kernel = lift_row_sort_opdef(op, dtype, shape.len() as u8).expect("the row sort lifts");
    assert_eq!(
        check_kernel(&kernel),
        Ok(()),
        "a lifted kernel is well formed"
    );
    let mut args = vec![KernelArg::Buffer(input)];
    args.extend(out_bufs.into_iter().map(KernelArg::Buffer));
    args.extend(shape.iter().map(|&e| KernelArg::Int(i128::from(e))));
    if let Some(k) = k_out {
        args.push(KernelArg::Int(i128::from(k)));
    }
    run_kernel(&kernel, &mut args).expect("the kernel runs");
    args
}

/// Run the oracle and the lifted kernel on `shape` and return each output's bytes from both.
fn both(
    dtype: ElementKind,
    order: SortOrder,
    out: SortOut,
    k_out: Option<i64>,
    shape: &[i64],
) -> Vec<(Vec<u8>, Vec<u8>)> {
    let limit = if k_out.is_some() {
        SortLimit::TopK
    } else {
        SortLimit::Full
    };
    let op = sort_op(dtype, order, out, limit);
    let input = src(dtype, shape);
    let rank = shape.len();
    let mut out_shape = shape.to_vec();
    out_shape[rank - 1] = k_out.unwrap_or(shape[rank - 1]);

    let (ops, out_bufs) = operands(dtype, out, shape, &out_shape);
    let key = try_structure_key(OpCategory::Sorting, &ops, ArchSku::Sm89).expect("a key");
    let plan = try_build_plan(&op, &key).expect("the row sort plans");
    let want = evaluate(&plan, &ops, std::slice::from_ref(&input), &[]);

    let args = run_lifted(&op, input, out_bufs, shape, k_out);
    // The oracle sizes every output at the plan dtype's width; only the real elements count.
    let n_out: usize = out_shape.iter().product::<i64>() as usize;
    let widths: Vec<usize> = ops[1..].iter().map(|o| width(o.dtype)).collect();
    want.into_iter()
        .enumerate()
        .map(|(j, w)| {
            let KernelArg::Buffer(g) = &args[1 + j] else {
                unreachable!()
            };
            let len = n_out * widths[j];
            (w.bytes[..len].to_vec(), g.bytes[..len].to_vec())
        })
        .collect()
}

fn assert_agree(
    dtype: ElementKind,
    order: SortOrder,
    out: SortOut,
    k_out: Option<i64>,
    shape: &[i64],
) {
    for (j, (want, got)) in both(dtype, order, out, k_out, shape).iter().enumerate() {
        assert_eq!(
            got, want,
            "{dtype:?} {order:?} {out:?} k_out={k_out:?} shape={shape:?} output {j}"
        );
    }
}

#[test]
fn full_sorts_agree_in_every_form() {
    for dtype in [ElementKind::F32, ElementKind::F64] {
        for order in [SortOrder::Asc, SortOrder::Desc] {
            for out in [SortOut::Values, SortOut::Indices, SortOut::Both] {
                assert_agree(dtype, order, out, None, &[4, 11]);
                assert_agree(dtype, order, out, None, &[2, 3, 7]);
            }
        }
    }
}

#[test]
fn capped_sorts_agree() {
    for dtype in [ElementKind::F32, ElementKind::F64] {
        for order in [SortOrder::Asc, SortOrder::Desc] {
            for k_out in [1, 3, 10, 11] {
                assert_agree(dtype, order, SortOut::Both, Some(k_out), &[4, 11]);
            }
            assert_agree(dtype, order, SortOut::Values, Some(4), &[3, 9]);
            assert_agree(dtype, order, SortOut::Indices, Some(4), &[3, 9]);
        }
    }
}

#[test]
fn integer_sorts_agree_in_every_form() {
    for dtype in [ElementKind::I32] {
        for order in [SortOrder::Asc, SortOrder::Desc] {
            for out in [SortOut::Values, SortOut::Indices, SortOut::Both] {
                assert_agree(dtype, order, out, None, &[4, 11]);
                assert_agree(dtype, order, out, None, &[2, 3, 7]);
            }
            for k_out in [1, 3, 11] {
                assert_agree(dtype, order, SortOut::Both, Some(k_out), &[4, 11]);
            }
        }
    }
}

#[test]
fn the_integer_corpus_has_both_extremes_and_ties() {
    // Control: the oracle's own answer shows both extremes, each at its end of a row, and a tie.
    let io = both(
        ElementKind::I32,
        SortOrder::Asc,
        SortOut::Values,
        None,
        &[4, 11],
    );
    let v = i32s(&io[0].0);
    let rows: Vec<&[i32]> = v.chunks(11).collect();
    assert!(
        rows.iter().any(|r| r[0] == i32::MIN),
        "a minimum sorts first"
    );
    assert!(
        rows.iter().any(|r| r[10] == i32::MAX),
        "a maximum sorts last"
    );
    assert!(v.windows(2).any(|w| w[0] == w[1]), "a tie is sorted");
}

#[test]
fn degenerate_rows_agree() {
    for order in [SortOrder::Asc, SortOrder::Desc] {
        assert_agree(ElementKind::F32, order, SortOut::Both, None, &[3, 1]);
        assert_agree(ElementKind::F32, order, SortOut::Both, None, &[1, 2]);
    }
}

fn f32s(b: &[u8]) -> Vec<f32> {
    b.chunks(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn i32s(b: &[u8]) -> Vec<i32> {
    b.chunks(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

#[test]
fn the_corpus_exercises_what_the_lift_must_get_right() {
    // Controls: the agreement above would be vacuous on a corpus without these, so the oracle's
    // own answer on it must show each one. Row 0 of the [4, 11] corpus.
    let io = both(
        ElementKind::F32,
        SortOrder::Asc,
        SortOut::Both,
        None,
        &[4, 11],
    );
    let (vals, idx) = (f32s(&io[0].0), i32s(&io[1].0));
    let row = &vals[..11];
    assert!(row.iter().any(|v| v.is_nan()), "a NaN is sorted");
    assert!(row.last().unwrap().is_nan(), "ascending puts NaN last");
    let zeros: Vec<u32> = vals
        .iter()
        .filter(|v| **v == 0.0)
        .map(|v| v.to_bits())
        .collect();
    assert!(zeros.len() >= 2 && zeros.contains(&0) && zeros.contains(&0x8000_0000));
    let tied = idx[..11]
        .windows(2)
        .zip(row.windows(2))
        .any(|(i, v)| v[0] == v[1] && i[0] < i[1]);
    assert!(tied, "a tie keeps ascending original index");

    let d = both(
        ElementKind::F32,
        SortOrder::Desc,
        SortOut::Values,
        None,
        &[4, 11],
    );
    assert!(f32s(&d[0].0)[0].is_nan(), "descending puts NaN first");

    let payloads: std::collections::BTreeSet<u32> = f32s(&io[0].0)
        .iter()
        .filter(|v| v.is_nan())
        .map(|v| v.to_bits())
        .collect();
    assert!(
        payloads.len() >= 2,
        "NaNs with different payloads are moved"
    );
}

#[test]
fn declines_what_it_does_not_lift() {
    let f = |op: &OpDef, dt| lift_row_sort_opdef(op, dt, 2).err();
    let sort = OpDef::row_sort("s", ElementKind::F32, SortOrder::Asc);
    assert_eq!(
        f(
            &OpDef::elementwise("e", 1, &[ElementKind::F32], unpopped::ir::input(0)),
            ElementKind::F32
        ),
        Some(LiftDecline::NotRowSort)
    );
    assert!(matches!(
        f(&sort, ElementKind::I64),
        Some(LiftDecline::UnsupportedForm(_))
    ));
    let mut unstable = sort;
    if let Access::RowSort { stable, .. } = &mut unstable.access {
        *stable = false;
    }
    assert!(matches!(
        f(&unstable, ElementKind::F32),
        Some(LiftDecline::UnsupportedForm(_))
    ));
}
