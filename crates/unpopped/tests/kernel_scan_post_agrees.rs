// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Fused scan posts, in the mode they allow** (`docs/ir-kernel-layer.md` 7.1). The oracle hands
//! the running prefix to the post as an unrounded `f64`; the lift sets
//! `Reduced(0) := Cast(f64 prefix -> compute)` and gives the post the elements loaded for the
//! position as its `Input(k)`. So `F64` compute is byte-exact and `F32` compute is compared to
//! within [`MAX_ULPS`] ulps (NaN equal, infinities exact), the oracle being a tolerance referee.

use unpopped::ir::{Expr, OpDef, ReduceOp, input, konst, reduced};
use unpopped::kernel::check_kernel;
use unpopped::kernel_lift::lift_scan_opdef;
use unpopped::oracle::kernel_eval::{KernelArg, run_kernel};
use unpopped::oracle::{TypedBuffer, evaluate};
use unpopped::plan::try_build_plan;
use unpopped_vocab::{ArchSku, ElementKind, OpCategory, OperandDesc, structure_key};

/// The PM-approved bound for tolerance mode (2026-10-29).
const MAX_ULPS: u64 = 2;
const COLS: i64 = 8;

fn buffer(dtype: ElementKind, rows: i64, f: impl Fn(usize) -> f64) -> TypedBuffer {
    let bytes: Vec<u8> = (0..(rows * COLS) as usize)
        .flat_map(|i| match dtype {
            ElementKind::F32 => (f(i) as f32).to_le_bytes().to_vec(),
            _ => f(i).to_le_bytes().to_vec(),
        })
        .collect();
    TypedBuffer::new(dtype, vec![rows, COLS], vec![COLS, 1], bytes)
}

/// Positive, varied values, so a square root of a prefix sum is defined and not a round number.
fn corpus(i: usize) -> f64 {
    [0.1, 2.7, 3.3, 0.05, 11.0, 0.9, 1.7, 100.5, 33.3, 0.77][i % 10] * (1.0 + (i / 8) as f64 * 0.13)
}

fn both(op: &OpDef, dtype: ElementKind, src: &TypedBuffer) -> (Vec<u8>, Vec<u8>) {
    let rows = src.shape[0];
    let a = OperandDesc::new(2, &src.shape, &src.strides, dtype, 256);
    let key = structure_key(OpCategory::Scan, &[a, a], ArchSku::Sm89);
    let plan = try_build_plan(op, &key).expect("the scan plans");
    let want = evaluate(&plan, &[a, a], std::slice::from_ref(src), &[])
        .remove(0)
        .bytes;
    let kernel = lift_scan_opdef(op, dtype, 2).expect("the scan with a post lifts");
    assert_eq!(
        check_kernel(&kernel),
        Ok(()),
        "a lifted kernel is well formed"
    );
    let mut args = vec![
        KernelArg::Buffer(src.clone()),
        KernelArg::Buffer(TypedBuffer::new(
            dtype,
            vec![rows, COLS],
            vec![COLS, 1],
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

fn scan(dtype: ElementKind, rop: ReduceOp, reverse: bool, exclusive: bool, post: Expr) -> OpDef {
    OpDef::scan(
        "sp",
        1,
        &[dtype],
        rop,
        1,
        reverse,
        exclusive,
        input(0),
        post,
    )
}

/// The posts under test: the prefix alone, the prefix with the element, and a constant fold.
fn posts() -> Vec<(&'static str, ReduceOp, Expr)> {
    vec![
        ("sqrt of a running sum", ReduceOp::Sum, reduced(0).sqrt()),
        (
            "prefix over one plus element squared",
            ReduceOp::Sum,
            reduced(0) / (konst(1.0) + input(0) * input(0)),
        ),
        ("halved running max", ReduceOp::Max, reduced(0) * konst(0.5)),
    ]
}

const DIRECTIONS: [(bool, bool); 4] = [(false, false), (true, false), (false, true), (true, true)];

#[test]
fn f64_compute_is_byte_exact_for_every_post() {
    for (name, rop, post) in posts() {
        for (reverse, exclusive) in DIRECTIONS {
            let op = scan(ElementKind::F64, rop, reverse, exclusive, post.clone());
            let (want, got) = both(&op, ElementKind::F64, &buffer(ElementKind::F64, 4, corpus));
            assert_eq!(got, want, "{name} reverse={reverse} exclusive={exclusive}");
        }
    }
}

#[test]
fn f32_posts_are_within_the_tolerance() {
    for (name, rop, post) in posts() {
        for (reverse, exclusive) in DIRECTIONS {
            let op = scan(ElementKind::F32, rop, reverse, exclusive, post.clone());
            let (want, got) = both(&op, ElementKind::F32, &buffer(ElementKind::F32, 6, corpus));
            let worst = max_ulps_f32(&want, &got);
            assert!(
                worst <= MAX_ULPS,
                "{name} reverse={reverse} exclusive={exclusive}: {worst} ulps (bound {MAX_ULPS})"
            );
        }
    }
}

#[test]
fn the_tolerance_is_not_vacuous() {
    // Control: rounding the running sum to f32 before the sqrt really moves some cell.
    let mut moved = false;
    for rows in 1..40 {
        let src = buffer(ElementKind::F32, rows, |i| {
            ((i * 11 + 3) % 17) as f64 * 0.413 + 0.2
        });
        let op = scan(
            ElementKind::F32,
            ReduceOp::Sum,
            false,
            false,
            reduced(0).sqrt(),
        );
        let (want, got) = both(&op, ElementKind::F32, &src);
        assert!(max_ulps_f32(&want, &got) <= MAX_ULPS);
        moved |= want != got;
    }
    assert!(
        moved,
        "some cell differs between the unrounded and rounded prefix"
    );
}

#[test]
fn a_wide_prefix_is_required_under_a_post() {
    // 1e8 + 1 - 1e8 + 0.5: the f64 running sums are 1e8, 1e8+1, 1, 1.5; an f32 prefix loses the 1.
    let op = scan(
        ElementKind::F32,
        ReduceOp::Sum,
        false,
        false,
        reduced(0) * konst(2.0),
    );
    let src = buffer(ElementKind::F32, 1, |i| {
        [1.0e8, 1.0, -1.0e8, 0.5, 0.0, 0.0, 0.0, 0.0][i]
    });
    let (want, got) = both(&op, ElementKind::F32, &src);
    assert!(max_ulps_f32(&want, &got) <= MAX_ULPS);
    let third = f32::from_le_bytes(got[8..12].try_into().unwrap());
    assert_eq!(third, 2.0, "twice the f64 prefix 1.0");
}

#[test]
fn nan_and_infinity_cells_agree() {
    let src = buffer(ElementKind::F32, 3, |i| match i {
        2 => f64::NAN,
        11 => f64::INFINITY,
        _ => corpus(i),
    });
    for (name, rop, post) in posts() {
        let op = scan(ElementKind::F32, rop, false, false, post);
        let (want, got) = both(&op, ElementKind::F32, &src);
        assert!(max_ulps_f32(&want, &got) <= MAX_ULPS, "{name}");
    }
}
