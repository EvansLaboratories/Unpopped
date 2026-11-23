// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Census wall: an inclusive upward bound** (`for (int i = lo; i <= hi; ++i)`, 5 Baracuda kernels). The IR's limit is
//! exclusive, so `i <= b` is `i < b + 1` (the mirror of the existing `i >= b`). Oracle runs against Rust references.

use unpopped::exec::LaunchShape;
use unpopped::kernel::KernelFn;
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_convert::{Frontend, KernelLiftError, lift_kernel_launched};

fn parse(src: &str) -> Option<tree_sitter::Tree> {
    let mut p = tree_sitter::Parser::new();
    p.set_language(&tree_sitter_cpp::LANGUAGE.into()).ok()?;
    p.parse(src, None)
}

const C: Frontend = Frontend::new("c", "void", &[], "out", "in", parse);

fn lift(src: &str) -> Result<KernelFn, KernelLiftError> {
    lift_kernel_launched(&C, src, "k", LaunchShape::new([1, 1, 1], [1, 1, 1]))
}

fn out(k: &KernelFn, n: usize) -> Vec<f64> {
    let mut args = vec![KernelArg::Buffer(TypedBuffer::from_f32(
        &[n as i64],
        &vec![0.0; n],
    ))];
    run_launch(k, &mut args).expect("runs");
    let KernelArg::Buffer(o) = &args[0] else {
        panic!()
    };
    o.to_f64_vec()
}

#[test]
fn an_inclusive_bound_runs_the_last_value_too() {
    // 2..=5 is four trips (2+3+4+5), not three
    let src = "void k(float* out) { float s = 0.0f; for (int i = 2; i <= 5; i++) { s += (float)i; } out[0] = s; }";
    assert_eq!(out(&lift(src).unwrap(), 1), vec![14.0]);
}

#[test]
fn a_computed_bound_and_an_empty_range() {
    // the scan_cumprod_backward shape: bounds set by a branch, `i <= hi`; `hi < lo` is no trips
    let src = "void k(float* out) { for (int j = 0; j < 4; j++) { int lo; int hi; if (j < 2) { lo = 0; hi = j; } else { lo = j; hi = 3; } \
        float s = 0.0f; for (int i = lo; i <= hi; ++i) { s += (float)(i + 1); } out[j] = s; } }";
    let want: Vec<f64> = (0..4)
        .map(|j: i32| {
            let (lo, hi) = if j < 2 { (0, j) } else { (j, 3) };
            f64::from((lo..=hi).map(|i| i + 1).sum::<i32>())
        })
        .collect();
    assert_eq!(out(&lift(src).unwrap(), 4), want);
    let empty = "void k(float* out) { float s = 7.0f; for (int i = 3; i <= 2; i++) { s += 1.0f; } out[0] = s; }";
    assert_eq!(out(&lift(empty).unwrap(), 1), vec![7.0]);
}

#[test]
fn a_stride_and_the_other_directions_are_unchanged() {
    let strided = "void k(float* out) { float s = 0.0f; for (int i = 0; i <= 6; i += 3) { s += (float)i; } out[0] = s; }";
    assert_eq!(out(&lift(strided).unwrap(), 1), vec![9.0]); // 0, 3, 6
    let down = "void k(float* out) { float s = 0.0f; for (int i = 5; i >= 2; i--) { s += (float)i; } out[0] = s; }";
    assert_eq!(out(&lift(down).unwrap(), 1), vec![14.0]);
    // a counter on the right of `<=` is not the counted form
    assert!(
        lift("void k(float* out) { float s = 0.0f; for (int i = 0; 3 <= i; i++) { s += 1.0f; } out[0] = s; }")
            .is_err()
    );
}
