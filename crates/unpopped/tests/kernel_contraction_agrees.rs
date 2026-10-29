// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The layers agree on contractions** (`docs/ir-kernel-layer.md` section 7): a float matmul
//! (plain or batched) with the identity epilogue, lifted into the kernel layer and run by the
//! kernel interpreter, produces byte for byte what `oracle::evaluate` produces.
//!
//! The oracle reads each operand by its PHYSICAL strides at the logical coordinate, widens both
//! factors to `f64`, multiplies and accumulates in `f64` with `k` ascending, and rounds once at
//! the store. The corpus has an f32-cancelling column (1e8, 1, -1e8), NaN, infinities and signed
//! zeros, and the operands are also run transposed (column-major strides), because a lift that
//! assumed dense row-major would agree on every dense case and still be wrong.

use unpopped::ir::{ContractionAxes, Expr, OpDef, ScalarExpr, input, reduced};
use unpopped::kernel::check_kernel;
use unpopped::kernel_lift::{LiftDecline, lift_contraction_opdef};
use unpopped::oracle::kernel_eval::{KernelArg, run_kernel};
use unpopped::oracle::{TypedBuffer, evaluate};
use unpopped::plan::try_build_plan;
use unpopped_vocab::{
    ArchSku, CellFacts, ContractionRoles, ElementKind, OperandDesc, derive_structure_key,
};

const LHS_VALUES: [f32; 12] = [
    1.0e8,
    1.0,
    -1.0e8,
    0.5,
    2.0,
    f32::NAN,
    1.0,
    2.0,
    f32::INFINITY,
    -0.0,
    0.1,
    -3.0,
];
const RHS_VALUES: [f32; 8] = [1.0, -1.0, 0.5, 2.0, 0.0, -0.0, 3.0, 0.25];

fn bytes_f32(vs: impl Iterator<Item = f32>) -> Vec<u8> {
    vs.flat_map(f32::to_le_bytes).collect()
}

/// A buffer of `shape` whose element at logical `coords` is `value(linear)` (linear in logical
/// row-major order), stored with `strides`.
fn buffer(shape: &[i64], strides: &[i64], value: impl Fn(usize) -> f32) -> TypedBuffer {
    let n: i64 = shape.iter().product();
    let extent: i64 = shape
        .iter()
        .zip(strides)
        .map(|(s, st)| (s - 1) * st)
        .sum::<i64>()
        + 1;
    let mut data = vec![0f32; extent as usize];
    for lin in 0..n {
        let (mut rem, mut off) = (lin, 0i64);
        for (s, st) in shape.iter().zip(strides).rev() {
            off += (rem % s) * st;
            rem /= s;
        }
        data[off as usize] = value(lin as usize);
    }
    TypedBuffer::new(
        ElementKind::F32,
        shape.to_vec(),
        strides.to_vec(),
        bytes_f32(data.into_iter()),
    )
}

fn matmul() -> OpDef {
    OpDef::contraction(
        "mm",
        &[ElementKind::F32],
        ContractionAxes::matmul(),
        reduced(0),
    )
}

fn batched() -> OpDef {
    OpDef::contraction(
        "bmm",
        &[ElementKind::F32],
        ContractionAxes::batched_matmul(),
        reduced(0),
    )
}

/// Run `op` through the oracle and through the lifted kernel on `lhs`, `rhs`.
fn both(op: &OpDef, lhs: &TypedBuffer, rhs: &TypedBuffer, out_shape: &[i64]) -> (Vec<u8>, Vec<u8>) {
    let out_strides: Vec<i64> = out_shape
        .iter()
        .enumerate()
        .map(|(i, _)| out_shape[i + 1..].iter().product())
        .collect();
    let dtype = lhs.dtype;
    let desc =
        |tb: &TypedBuffer| OperandDesc::new(tb.shape.len(), &tb.shape, &tb.strides, dtype, 256);
    let out = OperandDesc::new(out_shape.len(), out_shape, &out_strides, dtype, 256);
    let ops = [desc(lhs), desc(rhs), out];
    let rank = lhs.shape.len();
    let mut facts = CellFacts::new();
    facts.contraction_declared = true;
    facts.weight_operand = Some(1);
    facts.contraction_roles = Some(ContractionRoles::gemm(rank, usize::from(rank == 3)));
    let key = derive_structure_key(&ops, &facts, ArchSku::Sm89).expect("the key derives");
    let plan = try_build_plan(op, &key).expect("the contraction plans");
    let want = evaluate(&plan, &ops, &[lhs.clone(), rhs.clone()], &[])
        .remove(0)
        .bytes;

    let got = run_lifted(op, lhs, rhs, out_shape, out_strides, want.len());
    (want, got)
}

/// Lift `op`, run the kernel on `lhs`, `rhs`, and return the first `len` bytes of its output.
fn run_lifted(
    op: &OpDef,
    lhs: &TypedBuffer,
    rhs: &TypedBuffer,
    out_shape: &[i64],
    out_strides: Vec<i64>,
    len: usize,
) -> Vec<u8> {
    let dtype = lhs.dtype;
    let rank = lhs.shape.len();
    let kernel = lift_contraction_opdef(op, dtype).expect("the contraction lifts");
    assert_eq!(
        check_kernel(&kernel),
        Ok(()),
        "a lifted kernel is well formed"
    );
    let n_out: i64 = out_shape.iter().product();
    let mut args = vec![
        KernelArg::Buffer(lhs.clone()),
        KernelArg::Buffer(rhs.clone()),
        KernelArg::Buffer(TypedBuffer::new(
            dtype,
            out_shape.to_vec(),
            out_strides,
            vec![0u8; 8 * n_out as usize],
        )),
    ];
    // Extents: [B,] M, N, K.
    let (b, m, n, k) = if rank == 3 {
        (Some(lhs.shape[0]), lhs.shape[1], rhs.shape[2], lhs.shape[2])
    } else {
        (None, lhs.shape[0], rhs.shape[1], lhs.shape[1])
    };
    for e in b.into_iter().chain([m, n, k]) {
        args.push(KernelArg::Int(i128::from(e)));
    }
    run_kernel(&kernel, &mut args).expect("the kernel runs");
    let KernelArg::Buffer(got) = &args[2] else {
        unreachable!()
    };
    got.bytes[..len].to_vec()
}

fn lhs_value(i: usize) -> f32 {
    LHS_VALUES[i % LHS_VALUES.len()]
}

fn rhs_value(i: usize) -> f32 {
    RHS_VALUES[i % RHS_VALUES.len()]
}

#[test]
fn a_plain_matmul_agrees() {
    let lhs = buffer(&[3, 5], &[5, 1], lhs_value);
    let rhs = buffer(&[5, 4], &[4, 1], rhs_value);
    let (want, got) = both(&matmul(), &lhs, &rhs, &[3, 4]);
    assert_eq!(got, want);
}

#[test]
fn transposed_operands_agree() {
    // Column-major lhs ([3,5] with strides [1,3]) and rhs ([5,4] with strides [1,5]).
    let lhs = buffer(&[3, 5], &[1, 3], lhs_value);
    let rhs = buffer(&[5, 4], &[1, 5], rhs_value);
    let (want, got) = both(&matmul(), &lhs, &rhs, &[3, 4]);
    assert_eq!(got, want);
    // Control: the same logical values stored dense give the same answer, so the strided case is
    // exercising the view and not different data.
    let dense = both(
        &matmul(),
        &buffer(&[3, 5], &[5, 1], lhs_value),
        &buffer(&[5, 4], &[4, 1], rhs_value),
        &[3, 4],
    );
    assert_eq!(want, dense.0);
}

#[test]
fn a_batched_matmul_agrees() {
    let lhs = buffer(&[2, 3, 5], &[15, 5, 1], lhs_value);
    let rhs = buffer(&[2, 5, 4], &[20, 4, 1], rhs_value);
    let (want, got) = both(&batched(), &lhs, &rhs, &[2, 3, 4]);
    assert_eq!(got, want);
}

#[test]
fn the_accumulator_is_f64() {
    // Row 0 of the lhs is (1e8, 1, -1e8, 0.5, 2) and column 0 of the rhs is all 1.0 for the
    // values used here: the f64 sum is 3.5; an f32 accumulator would lose the 1 next to 1e8.
    let lhs = buffer(&[1, 5], &[5, 1], lhs_value);
    let rhs = buffer(&[5, 1], &[1, 1], |_| 1.0);
    let (want, got) = both(&matmul(), &lhs, &rhs, &[1, 1]);
    assert_eq!(got, want);
    let v = f32::from_le_bytes([got[0], got[1], got[2], got[3]]);
    assert_eq!(v, 3.5, "1e8 + 1 - 1e8 + 0.5 + 2 in f64");
}

#[test]
fn forms_the_lift_does_not_cover_decline_with_a_reason() {
    // Control: the plain matmul lifts.
    assert!(lift_contraction_opdef(&matmul(), ElementKind::F32).is_ok());
    let ew = OpDef::elementwise("e", 1, &[ElementKind::F32], Expr(input(0).0));
    assert_eq!(
        lift_contraction_opdef(&ew, ElementKind::F32),
        Err(LiftDecline::NotContraction)
    );
    // A fused epilogue (here a scale of the accumulator).
    let scaled = OpDef::contraction(
        "s",
        &[ElementKind::F32],
        ContractionAxes::matmul(),
        Expr(ScalarExpr::Mul(
            Box::new(ScalarExpr::Reduced(0)),
            Box::new(ScalarExpr::Const(2.0)),
        )),
    );
    assert_eq!(
        lift_contraction_opdef(&scaled, ElementKind::F32),
        Err(LiftDecline::UnsupportedForm("a fused epilogue or bias"))
    );
    // An integer contraction.
    let int = OpDef::contraction(
        "i",
        &[ElementKind::I32],
        ContractionAxes::matmul(),
        reduced(0),
    );
    assert_eq!(
        lift_contraction_opdef(&int, ElementKind::I32),
        Err(LiftDecline::UnsupportedForm("a non-float contraction"))
    );
}

#[test]
fn the_k_order_is_ascending_and_the_accumulator_f64_when_compute_is_f64() {
    // 1e16 + 1 - 1e16 + 1 + 0.1 + 0.2 + 0.3: the f64 fold is order-sensitive (1e16 swallows the 1),
    // so a descending or reordered k loop gives a different last-place answer. At f32 compute the
    // final rounding hides the order, which is why this case is f64.
    let vals = [1.0e16, 1.0, -1.0e16, 1.0, 0.1, 0.2, 0.3];
    let f64_bytes = |vs: &[f64]| vs.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>();
    let lhs = TypedBuffer::new(ElementKind::F64, vec![1, 7], vec![7, 1], f64_bytes(&vals));
    let rhs = TypedBuffer::new(
        ElementKind::F64,
        vec![7, 1],
        vec![1, 1],
        f64_bytes(&[1.0; 7]),
    );
    let op = OpDef::contraction(
        "mm64",
        &[ElementKind::F64],
        ContractionAxes::matmul(),
        reduced(0),
    );
    let (want, got) = both(&op, &lhs, &rhs, &[1, 1]);
    assert_eq!(got, want);
    // Control: the descending fold really is different, so the case discriminates.
    let desc: f64 = vals.iter().rev().fold(0.0, |a, v| a + v);
    let asc: f64 = vals.iter().fold(0.0, |a, v| a + v);
    assert_ne!(desc.to_bits(), asc.to_bits());
    assert_eq!(
        f64::from_le_bytes(got[..8].try_into().unwrap()).to_bits(),
        asc.to_bits()
    );
}
