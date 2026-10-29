// SPDX-License-Identifier: MIT OR Apache-2.0
//! **A view on a contraction operand is recognition metadata** (`docs/ir-kernel-layer.md` section
//! 7). The plan admits `View::Permute` on a matmul's operands, and a planner emits one for a
//! transposed factor, but the oracle never reads it: it reads each factor through the bound
//! operand's own (physical) strides, so a transposed operand is a transposed binding. The lift
//! therefore accepts the view and builds the **view-free kernel**; reordering the coordinates as
//! the Elementwise lift does for a `Permute` would transpose twice.
//!
//! This file checks that the op carrying the views agrees with the oracle on transposed bindings
//! (plain and batched, either factor or both), that its kernel is *identical* to the view-free
//! op's, and that a view on a fused bias, which the oracle does read, still declines.

use unpopped::ir::{ContractionAxes, OpDef, View, input, reduced};
use unpopped::kernel::check_kernel;
use unpopped::kernel_lift::{LiftDecline, lift_contraction_opdef};
use unpopped::oracle::kernel_eval::{KernelArg, run_kernel};
use unpopped::oracle::{TypedBuffer, evaluate};
use unpopped::plan::try_build_plan;
use unpopped_vocab::{
    ArchSku, CellFacts, ContractionRoles, ElementKind, OperandDesc, derive_structure_key,
};

/// A buffer of `shape` stored with `strides` whose element at logical row-major index `lin` is
/// `value(lin)`; small integers, so every product and sum is exact in `f32`.
fn buffer(shape: &[i64], strides: &[i64], value: impl Fn(usize) -> f32) -> TypedBuffer {
    let extent: i64 = shape
        .iter()
        .zip(strides)
        .map(|(s, st)| (s - 1) * st)
        .sum::<i64>()
        + 1;
    let mut data = vec![0f32; extent as usize];
    for lin in 0..shape.iter().product::<i64>() {
        let (mut rem, mut off) = (lin, 0i64);
        for (s, st) in shape.iter().zip(strides).rev() {
            off += (rem % s) * st;
            rem /= s;
        }
        data[off as usize] = value(lin as usize);
    }
    let bytes = data.into_iter().flat_map(f32::to_le_bytes).collect();
    TypedBuffer::new(ElementKind::F32, shape.to_vec(), strides.to_vec(), bytes)
}

fn lhs_value(i: usize) -> f32 {
    (i % 7) as f32 - 3.0
}

fn rhs_value(i: usize) -> f32 {
    (i * 3 % 11) as f32 - 5.0
}

/// A logical `[rows, cols]` (or `[b, rows, cols]`) factor, stored dense or transposed.
fn factor(
    batch: Option<i64>,
    rows: i64,
    cols: i64,
    transposed: bool,
    f: fn(usize) -> f32,
) -> TypedBuffer {
    let (row_st, col_st) = if transposed { (1, rows) } else { (cols, 1) };
    match batch {
        None => buffer(&[rows, cols], &[row_st, col_st], f),
        Some(b) => buffer(&[b, rows, cols], &[rows * cols, row_st, col_st], f),
    }
}

fn matmul(batched: bool, views: Vec<View>) -> OpDef {
    let axes = if batched {
        ContractionAxes::batched_matmul()
    } else {
        ContractionAxes::matmul()
    };
    let op = OpDef::contraction("mm", &[ElementKind::F32], axes, reduced(0));
    if views.is_empty() {
        op
    } else {
        op.with_views(views)
    }
}

/// The oracle's output and the lifted kernel's for `op` on `lhs`, `rhs`.
fn both(op: &OpDef, lhs: &TypedBuffer, rhs: &TypedBuffer) -> (Vec<u8>, Vec<u8>) {
    let rank = lhs.shape.len();
    let (m, n) = (lhs.shape[rank - 2], rhs.shape[rank - 1]);
    let out_shape: Vec<i64> = lhs.shape[..rank - 2]
        .iter()
        .copied()
        .chain([m, n])
        .collect();
    let out_strides = dense_strides(&out_shape);
    let desc = |t: &TypedBuffer| OperandDesc::new(rank, &t.shape, &t.strides, t.dtype, 256);
    let ops = [
        desc(lhs),
        desc(rhs),
        OperandDesc::new(rank, &out_shape, &out_strides, ElementKind::F32, 256),
    ];
    let mut facts = CellFacts::new();
    facts.contraction_declared = true;
    facts.weight_operand = Some(1);
    facts.contraction_roles = Some(ContractionRoles::gemm(rank, usize::from(rank == 3)));
    let key = derive_structure_key(&ops, &facts, ArchSku::Sm89).expect("the key derives");
    let plan = try_build_plan(op, &key).expect("the viewed contraction plans");
    let want = evaluate(&plan, &ops, &[lhs.clone(), rhs.clone()], &[])
        .remove(0)
        .bytes;

    let got = kernel_out(op, lhs, rhs, &out_shape, out_strides, want.len());
    (want, got)
}

/// What the lifted kernel writes for `op` on `lhs`, `rhs`: the first `len` bytes of its output.
fn kernel_out(
    op: &OpDef,
    lhs: &TypedBuffer,
    rhs: &TypedBuffer,
    out_shape: &[i64],
    out_strides: Vec<i64>,
    len: usize,
) -> Vec<u8> {
    let rank = lhs.shape.len();
    let kernel = lift_contraction_opdef(op, ElementKind::F32).expect("the contraction lifts");
    assert_eq!(
        check_kernel(&kernel),
        Ok(()),
        "a lifted kernel is well formed"
    );
    let out = TypedBuffer::new(
        ElementKind::F32,
        out_shape.to_vec(),
        out_strides,
        vec![0; len],
    );
    let mut args = vec![
        KernelArg::Buffer(lhs.clone()),
        KernelArg::Buffer(rhs.clone()),
        KernelArg::Buffer(out),
    ];
    let (m, k) = (lhs.shape[rank - 2], lhs.shape[rank - 1]);
    let extents = lhs.shape[..rank - 2]
        .iter()
        .copied()
        .chain([m, rhs.shape[rank - 1], k]);
    args.extend(extents.map(|e| KernelArg::Int(i128::from(e))));
    run_kernel(&kernel, &mut args).expect("the kernel runs");
    let KernelArg::Buffer(got) = &args[2] else {
        unreachable!()
    };
    got.bytes[..len].to_vec()
}

fn dense_strides(shape: &[i64]) -> Vec<i64> {
    (0..shape.len())
        .map(|d| shape[d + 1..].iter().product())
        .collect()
}

fn permute(batched: bool, transposed: bool) -> View {
    match (transposed, batched) {
        (false, _) => View::Identity,
        (true, false) => View::Permute { perm: vec![1, 0] },
        (true, true) => View::Permute {
            perm: vec![0, 2, 1],
        },
    }
}

#[test]
fn transposed_factors_declared_by_a_permute_view_agree() {
    for batched in [false, true] {
        for (lhs_t, rhs_t) in [(false, true), (true, false), (true, true)] {
            let b = batched.then_some(2);
            let lhs = factor(b, 3, 5, lhs_t, lhs_value);
            let rhs = factor(b, 5, 4, rhs_t, rhs_value);
            let op = matmul(
                batched,
                vec![permute(batched, lhs_t), permute(batched, rhs_t)],
            );
            let (want, got) = both(&op, &lhs, &rhs);
            assert_eq!(got, want, "batched={batched} lhs_t={lhs_t} rhs_t={rhs_t}");
        }
    }
}

#[test]
fn the_kernel_is_the_view_free_one() {
    // A reorder of the coordinates here would transpose a factor twice (the binding already
    // carries the transpose), so the views must change nothing in the kernel.
    for batched in [false, true] {
        let viewed = matmul(
            batched,
            vec![permute(batched, true), permute(batched, true)],
        );
        let plain = matmul(batched, vec![]);
        assert_eq!(
            lift_contraction_opdef(&viewed, ElementKind::F32),
            lift_contraction_opdef(&plain, ElementKind::F32),
            "batched={batched}"
        );
    }
}

#[test]
fn the_binding_is_what_the_agreement_depends_on() {
    // Control, read from the ORACLE: the same storage bound with the dense strides instead of the
    // transposed ones is a different matrix, so the strides, not the bytes, carry the transpose.
    let op = matmul(false, vec![View::Identity, permute(false, true)]);
    let lhs = factor(None, 3, 5, false, lhs_value);
    let transposed = factor(None, 5, 4, true, rhs_value);
    let mut misbound = transposed.clone();
    misbound.strides = vec![4, 1];
    assert_ne!(both(&op, &lhs, &transposed).0, both(&op, &lhs, &misbound).0);
}

#[test]
fn a_view_on_a_fused_bias_still_declines() {
    let biased = OpDef::contraction_bias("mb", &[ElementKind::F32], (reduced(0) + input(2)).relu());
    assert!(
        lift_contraction_opdef(&biased, ElementKind::F32).is_ok(),
        "control"
    );
    let viewed = biased.with_views(vec![
        View::Identity,
        View::Identity,
        View::Broadcast {
            bcast: unpopped_vocab::AxisMask::EMPTY,
        },
    ]);
    assert_eq!(
        lift_contraction_opdef(&viewed, ElementKind::F32),
        Err(LiftDecline::UnsupportedForm("a view on a fused bias"))
    );
}
