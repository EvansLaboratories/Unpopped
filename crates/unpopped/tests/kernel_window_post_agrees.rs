// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Fused window posts, in the mode they allow** (`docs/ir-kernel-layer.md` 7.1). The oracle hands
//! the finalised pool value to the post as an unrounded `f64`; the lift sets
//! `Reduced(0) := Cast(f64 value -> compute)`. So `F64` compute is byte-exact and `F32` compute is
//! compared to within [`MAX_ULPS`] ulps (NaN equal, infinities exact), the oracle being a tolerance
//! referee. A post that reads an `Input` declines.

use unpopped::ir::{Expr, OpDef, ReduceOp, input, konst, reduced};
use unpopped::kernel::check_kernel;
use unpopped::kernel_lift::{LiftDecline, lift_window_opdef};
use unpopped::oracle::kernel_eval::{KernelArg, run_kernel};
use unpopped::oracle::{TypedBuffer, evaluate};
use unpopped::plan::try_build_plan;
use unpopped_vocab::{ArchSku, ElementKind, OpCategory, OperandDesc, try_structure_key};

/// The PM-approved bound for tolerance mode (2026-10-29).
const MAX_ULPS: u64 = 2;
const K_IN: i64 = 9;
const K_OUT: i64 = 5; // size 3, stride 2, pad 1 on both sides

fn buffer(dtype: ElementKind, rows: i64, f: impl Fn(usize) -> f64) -> TypedBuffer {
    let bytes: Vec<u8> = (0..(rows * K_IN) as usize)
        .flat_map(|i| match dtype {
            ElementKind::F32 => (f(i) as f32).to_le_bytes().to_vec(),
            _ => f(i).to_le_bytes().to_vec(),
        })
        .collect();
    TypedBuffer::new(dtype, vec![rows, K_IN], vec![K_IN, 1], bytes)
}

/// Positive, varied values, so a square root of a pooled sum is defined and not a round number.
fn corpus(i: usize) -> f64 {
    [0.1, 2.7, 3.3, 0.05, 11.0, 0.9, 1.7, 100.5, 33.3, 0.77][i % 10] * (1.0 + (i / 9) as f64 * 0.13)
}

fn window(dtype: ElementKind, wop: ReduceOp, include_pad: bool, post: Expr) -> OpDef {
    OpDef::window(
        "wp",
        1,
        &[dtype],
        wop,
        1,
        3,
        2,
        1,
        1,
        1,
        include_pad,
        input(0),
        post,
    )
}

fn both(op: &OpDef, dtype: ElementKind, src: &TypedBuffer) -> (Vec<u8>, Vec<u8>) {
    let rows = src.shape[0];
    let ops = [
        OperandDesc::new(2, &src.shape, &src.strides, dtype, 256),
        OperandDesc::new(2, &[rows, K_OUT], &[K_OUT, 1], dtype, 256),
    ];
    let key = try_structure_key(OpCategory::Pooling, &ops, ArchSku::Sm89).expect("a key");
    let plan = try_build_plan(op, &key).expect("the window plans");
    let want = evaluate(&plan, &ops, std::slice::from_ref(src), &[])
        .remove(0)
        .bytes;
    let kernel = lift_window_opdef(op, dtype, 2).expect("the window with a post lifts");
    assert_eq!(
        check_kernel(&kernel),
        Ok(()),
        "a lifted kernel is well formed"
    );
    let mut args = vec![
        KernelArg::Buffer(src.clone()),
        KernelArg::Buffer(TypedBuffer::new(
            dtype,
            vec![rows, K_OUT],
            vec![K_OUT, 1],
            vec![0xAB; want.len()],
        )),
    ];
    for e in [rows, K_IN, K_OUT] {
        args.push(KernelArg::Int(i128::from(e)));
    }
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

/// The posts under test: a square root of a pooled sum, a mean scaled and shifted (both divisors),
/// and a halved maximum.
fn posts() -> Vec<(&'static str, ReduceOp, bool, Expr)> {
    vec![
        (
            "sqrt of a pooled sum",
            ReduceOp::Sum,
            false,
            reduced(0).sqrt(),
        ),
        (
            "scaled and shifted mean, excluding pad",
            ReduceOp::Mean,
            false,
            reduced(0) * konst(1.7) + konst(0.3),
        ),
        (
            "scaled and shifted mean, including pad",
            ReduceOp::Mean,
            true,
            reduced(0) * konst(1.7) + konst(0.3),
        ),
        ("halved max", ReduceOp::Max, false, reduced(0) * konst(0.5)),
        (
            "reciprocal of a min",
            ReduceOp::Min,
            false,
            konst(1.0) / reduced(0),
        ),
    ]
}

#[test]
fn f64_compute_is_byte_exact_for_every_post() {
    for (name, wop, inc, post) in posts() {
        let op = window(ElementKind::F64, wop, inc, post);
        let (want, got) = both(&op, ElementKind::F64, &buffer(ElementKind::F64, 4, corpus));
        assert_eq!(got, want, "{name}");
    }
}

#[test]
fn f32_posts_are_within_the_tolerance() {
    for (name, wop, inc, post) in posts() {
        let op = window(ElementKind::F32, wop, inc, post);
        let (want, got) = both(&op, ElementKind::F32, &buffer(ElementKind::F32, 6, corpus));
        let worst = max_ulps_f32(&want, &got);
        assert!(worst <= MAX_ULPS, "{name}: {worst} ulps (bound {MAX_ULPS})");
    }
}

#[test]
fn the_tolerance_is_not_vacuous() {
    // Control: rounding the pooled sum to f32 before the sqrt really moves some cell.
    let mut moved = false;
    for rows in 1..40 {
        let src = buffer(ElementKind::F32, rows, |i| {
            ((i * 11 + 3) % 17) as f64 * 0.413 + 0.2
        });
        let op = window(ElementKind::F32, ReduceOp::Sum, false, reduced(0).sqrt());
        let (want, got) = both(&op, ElementKind::F32, &src);
        assert!(max_ulps_f32(&want, &got) <= MAX_ULPS);
        moved |= want != got;
    }
    assert!(
        moved,
        "some cell differs between the unrounded and rounded value"
    );
}

#[test]
fn a_wide_accumulator_is_required_under_a_post() {
    // Window 1 (pad 1, stride 2, size 3) covers positions 1..=3: 1e8, 1, -1e8. The f64 fold gives
    // 1; an f32 accumulator loses the 1 against 1e8 and gives 0.
    let src = buffer(ElementKind::F32, 1, |i| {
        [0.0, 1.0e8, 1.0, -1.0e8, 0.0, 0.0, 0.0, 0.0, 0.0][i]
    });
    let op = window(
        ElementKind::F32,
        ReduceOp::Sum,
        false,
        reduced(0) * konst(2.0),
    );
    let (want, got) = both(&op, ElementKind::F32, &src);
    assert!(max_ulps_f32(&want, &got) <= MAX_ULPS);
    let cell = |i: usize| f32::from_le_bytes(got[4 * i..4 * i + 4].try_into().unwrap());
    assert_eq!(cell(1), 2.0, "twice the f64 sum 1.0");
}

#[test]
fn nan_and_infinity_cells_agree() {
    let src = buffer(ElementKind::F32, 3, |i| match i {
        2 => f64::NAN,
        13 => f64::INFINITY,
        _ => corpus(i),
    });
    for (name, wop, inc, post) in posts() {
        let op = window(ElementKind::F32, wop, inc, post);
        let (want, got) = both(&op, ElementKind::F32, &src);
        assert!(max_ulps_f32(&want, &got) <= MAX_ULPS, "{name}");
    }
}

#[test]
fn a_post_that_reads_an_input_declines() {
    let op = window(
        ElementKind::F32,
        ReduceOp::Sum,
        false,
        reduced(0) + input(0),
    );
    assert!(matches!(
        lift_window_opdef(&op, ElementKind::F32, 2),
        Err(LiftDecline::UnsupportedLeaf(_))
    ));
}
