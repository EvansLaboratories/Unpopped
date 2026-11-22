// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Census wall: "a loop bound or step that names a variable the function assigns"** (13 Baracuda kernels). The IR reads a
//! `For` bound once; C reads it every iteration; they agree if nothing the bound names changes *while the loop runs*. The check
//! was function-wide, so a triangular loop (`j < i` inside `for i`, where the outer update assigns `i`) and a variable set
//! before or after a loop were refused. It now asks what the loop itself assigns. Oracle runs against Rust references.

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

fn declines(src: &str) {
    match lift(src) {
        Err(KernelLiftError::NotYetLifted { construct, .. }) => {
            assert!(construct.contains("assigns"), "{construct}");
        }
        other => panic!("expected the assigned-bound decline, got {other:?}"),
    }
}

#[test]
fn a_triangular_loop_bounds_the_inner_loop_by_the_outer_counter() {
    let src = "void k(float* out) { float t = 0.0f; \
        for (int i = 0; i < 5; i++) { float s = 0.0f; for (int j = 0; j < i; j++) { s += (float)(j + 1); } t += s; out[i] = s; } \
        out[5] = t; }";
    let want: Vec<f64> = (0..5)
        .map(|i: i32| f64::from((0..i).map(|j| j + 1).sum::<i32>()))
        .collect();
    let mut want = want;
    want.push(want.iter().sum());
    assert_eq!(out(&lift(src).unwrap(), 6), want);
}

#[test]
fn a_variable_set_between_loops_is_fixed_within_each() {
    let src = "void k(float* out) { int m = 2; float s = 0.0f; \
        for (int i = 0; i < m; i++) { s += 1.0f; } m = 5; \
        for (int i = 0; i < m; i++) { s += 10.0f; } out[0] = s; }";
    assert_eq!(out(&lift(src).unwrap(), 1), vec![52.0]);
}

#[test]
fn a_bound_the_loop_itself_changes_still_declines() {
    declines(
        "void k(float* out) { int m = 4; float s = 0.0f; for (int i = 0; i < m; i++) { m -= 1; s += 1.0f; } out[0] = s; }",
    );
    // the update, a nested loop's body, and the loop's own counter count too
    declines(
        "void k(float* out) { int m = 4; float s = 0.0f; for (int i = 0; i < m; i++) { for (int j = 0; j < 2; j++) { m = 3; } s += 1.0f; } out[0] = s; }",
    );
    declines(
        "void k(float* out) { int m = 4; float s = 0.0f; for (int i = 0; i < 9; i += m) { m = 1; s += 1.0f; } out[0] = s; }",
    );
}

#[test]
fn a_loop_that_calls_a_function_of_the_source_keeps_the_function_wide_rule() {
    // the callee could assign the caller's variable through a reference, so any assignment of `m` in the function counts
    declines(
        "float f(float x) { return x; } \
        void k(float* out) { int m = 2; float s = 0.0f; for (int i = 0; i < m; i++) { s += f(1.0f); } m = 3; out[0] = s; }",
    );
    // control: without the call, the same assignment after the loop lifts
    assert!(
        lift("void k(float* out) { int m = 2; float s = 0.0f; for (int i = 0; i < m; i++) { s += 1.0f; } m = 3; out[0] = s; }")
            .is_ok()
    );
}
