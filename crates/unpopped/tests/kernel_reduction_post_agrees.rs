// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Fused reduction posts, in the mode they allow** (`docs/ir-kernel-layer.md` 7.1). The oracle
//! hands the finalised fold to the post as an unrounded `f64`; the lift sets
//! `Reduced(0) := Cast(f64 acc -> compute)`. So:
//!
//! * `F64` compute is byte-exact for every post (the cast is the identity);
//! * `F32` compute is compared to within [`MAX_ULPS`] ulps (NaN equal, infinities exact), and the
//!   oracle is a tolerance referee. `the_tolerance_is_not_vacuous` proves the rounded and
//!   unrounded values really differ on the corpus.

use unpopped::ir::{Expr, OpDef, ReduceOp, input, konst, reduced};
use unpopped::kernel::check_kernel;
use unpopped::kernel_lift::{LiftDecline, lift_reduction_opdef};
use unpopped::oracle::kernel_eval::{KernelArg, run_kernel};
use unpopped::oracle::{TypedBuffer, evaluate};
use unpopped::plan::try_build_plan;
use unpopped_vocab::{ArchSku, ElementKind, OpCategory, OperandDesc, structure_key};

/// The PM-approved bound for tolerance mode (2026-10-29).
const MAX_ULPS: u64 = 2;
const COLS: i64 = 7;

fn buffer(dtype: ElementKind, rows: i64, f: impl Fn(usize) -> f64) -> TypedBuffer {
    let bytes: Vec<u8> = (0..(rows * COLS) as usize)
        .flat_map(|i| match dtype {
            ElementKind::F32 => (f(i) as f32).to_le_bytes().to_vec(),
            _ => f(i).to_le_bytes().to_vec(),
        })
        .collect();
    TypedBuffer::new(dtype, vec![rows, COLS], vec![COLS, 1], bytes)
}

/// Mixed magnitudes and signs, so a sum of squares is not a round number.
fn corpus(i: usize) -> f64 {
    [0.1, -2.7, 3.3, 0.05, 11.0, -0.9, 1.7, 100.5, -33.3, 0.77][i % 10]
        * (1.0 + (i / 7) as f64 * 0.11)
}

fn both(op: &OpDef, dtype: ElementKind, src: &TypedBuffer) -> (Vec<u8>, Vec<u8>) {
    let rows = src.shape[0];
    let a = OperandDesc::new(2, &src.shape, &src.strides, dtype, 256);
    let o = OperandDesc::new(1, &[rows], &[1], dtype, 256);
    let key = structure_key(OpCategory::Reduction, &[a, o], ArchSku::Sm89);
    let plan = try_build_plan(op, &key).expect("the reduction plans");
    let want = evaluate(&plan, &[a, o], std::slice::from_ref(src), &[])
        .remove(0)
        .bytes;
    let kernel = lift_reduction_opdef(op, dtype, 2).expect("the reduction with a post lifts");
    assert_eq!(
        check_kernel(&kernel),
        Ok(()),
        "a lifted kernel is well formed"
    );
    let mut args = vec![
        KernelArg::Buffer(src.clone()),
        KernelArg::Buffer(TypedBuffer::new(
            dtype,
            vec![rows],
            vec![1],
            vec![0xAB; want.len()],
        )),
        KernelArg::Int(i128::from(rows)),
        KernelArg::Int(i128::from(COLS)),
    ];
    run_kernel(&kernel, &mut args).expect("the kernel runs");
    let KernelArg::Buffer(got) = &args[1] else {
        unreachable!()
    };
    (want.clone(), got.bytes[..want.len()].to_vec())
}

fn ulps_f32(a: f32, b: f32) -> u64 {
    if a.is_nan() || b.is_nan() {
        return if a.is_nan() && b.is_nan() {
            0
        } else {
            u64::MAX
        };
    }
    let key = |x: f32| {
        let bits = i64::from(x.to_bits() as i32);
        if bits < 0 {
            i64::from(i32::MIN) - bits
        } else {
            bits
        }
    };
    key(a).abs_diff(key(b))
}

fn max_ulps_f32(want: &[u8], got: &[u8]) -> u64 {
    let f = |b: &[u8]| -> Vec<f32> {
        b.chunks(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    };
    f(want)
        .into_iter()
        .zip(f(got))
        .map(|(a, b)| ulps_f32(a, b))
        .max()
        .unwrap_or(0)
}

fn norm2(dtype: ElementKind) -> OpDef {
    OpDef::reduction_post(
        "norm2",
        1,
        &[dtype],
        input(0) * input(0),
        ReduceOp::Sum,
        reduced(0).sqrt(),
    )
}

fn rms(dtype: ElementKind) -> OpDef {
    OpDef::reduction_post(
        "rms",
        1,
        &[dtype],
        input(0) * input(0),
        ReduceOp::Mean,
        (reduced(0) + konst(1.0e-3)).sqrt(),
    )
}

fn scaled_max(dtype: ElementKind) -> OpDef {
    OpDef::reduction_post(
        "smax",
        1,
        &[dtype],
        Expr(input(0).0),
        ReduceOp::Max,
        reduced(0) * konst(0.3) - konst(1.0),
    )
}

#[test]
fn f64_compute_is_byte_exact_for_every_post() {
    for op in [
        norm2(ElementKind::F64),
        rms(ElementKind::F64),
        scaled_max(ElementKind::F64),
    ] {
        let (want, got) = both(&op, ElementKind::F64, &buffer(ElementKind::F64, 6, corpus));
        assert_eq!(got, want, "{}", op.name);
    }
}

#[test]
fn f32_posts_are_within_the_tolerance() {
    for op in [
        norm2(ElementKind::F32),
        rms(ElementKind::F32),
        scaled_max(ElementKind::F32),
    ] {
        let (want, got) = both(&op, ElementKind::F32, &buffer(ElementKind::F32, 9, corpus));
        let worst = max_ulps_f32(&want, &got);
        assert!(
            worst <= MAX_ULPS,
            "{}: {worst} ulps apart (bound {MAX_ULPS})",
            op.name
        );
    }
}

#[test]
fn the_tolerance_is_not_vacuous() {
    // Control: rounding the Mean to f32 before the sqrt really moves some row's answer, so the
    // tolerance cases are not secretly byte-exact.
    let mut moved = false;
    for rows in 1..60 {
        let src = buffer(ElementKind::F32, rows, |i| {
            ((i * 13 + 5) % 23) as f64 * 0.317 - 3.1
        });
        let (want, got) = both(&rms(ElementKind::F32), ElementKind::F32, &src);
        assert!(max_ulps_f32(&want, &got) <= MAX_ULPS);
        moved |= want != got;
    }
    assert!(
        moved,
        "some row differs between the unrounded and rounded finalised fold"
    );
}

#[test]
fn a_nan_row_agrees() {
    let src = buffer(ElementKind::F32, 3, |i| match i {
        1 => f64::NAN,
        9 => f64::INFINITY,
        _ => corpus(i),
    });
    for op in [norm2(ElementKind::F32), scaled_max(ElementKind::F32)] {
        let (want, got) = both(&op, ElementKind::F32, &src);
        assert!(max_ulps_f32(&want, &got) <= MAX_ULPS, "{}", op.name);
    }
}

#[test]
fn a_post_that_reads_an_input_declines() {
    // The oracle panics on this form and the plan gate rejects it; the lift must not guess.
    let mut op = norm2(ElementKind::F32);
    if let unpopped::ir::Access::Reduction { post, .. } = &mut op.access {
        *post = (reduced(0) + input(0)).0;
    }
    assert_eq!(
        lift_reduction_opdef(&op, ElementKind::F32, 2),
        Err(LiftDecline::UnsupportedLeaf("an Input in a reduction post"))
    );
}

#[test]
fn a_wide_accumulator_is_required_under_a_post() {
    // 1e8 + 1 - 1e8 + 0.5 + 2 = 3.5 in f64; an f32 accumulator loses the 1 and gets 2.5.
    let op = OpDef::reduction_post(
        "dbl",
        1,
        &[ElementKind::F32],
        Expr(input(0).0),
        ReduceOp::Sum,
        reduced(0) * konst(2.0),
    );
    let src = buffer(ElementKind::F32, 1, |i| {
        [1.0e8, 1.0, -1.0e8, 0.5, 2.0, 0.0, 0.0][i]
    });
    let (want, got) = both(&op, ElementKind::F32, &src);
    assert!(max_ulps_f32(&want, &got) <= MAX_ULPS);
    assert_eq!(f32::from_le_bytes(got[..4].try_into().unwrap()), 7.0);
}
