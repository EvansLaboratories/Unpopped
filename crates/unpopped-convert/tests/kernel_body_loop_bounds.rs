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
    // a bound that names the loop's own counter moves with it (only the update assigns it)
    declines(
        "void k(float* out) { float s = 0.0f; for (int i = 1; i < 10 - i; i++) { s += 1.0f; } out[0] = s; }",
    );
    declines(
        "void k(float* out) { int m = 4; float s = 0.0f; for (int i = 0; i < 9; i += m) { m = 1; s += 1.0f; } out[0] = s; }",
    );
}

#[test]
fn a_helper_that_takes_the_bound_by_reference_keeps_it_changing() {
    // the callee assigns the caller's `m` through the reference, so the bound moves while the loop runs
    declines(
        "void dec(int& v) { v -= 1; }         void k(float* out) { int m = 4; float s = 0.0f; for (int i = 0; i < m; i++) { dec(m); s += 1.0f; } out[0] = s; }",
    );
    // passed on through a second helper it is still the caller's variable
    declines(
        "void dec(int& v) { v -= 1; } void via(int& w) { dec(w); }         void k(float* out) { int m = 4; float s = 0.0f; for (int i = 0; i < m; i++) { via(m); s += 1.0f; } out[0] = s; }",
    );
}

#[test]
fn a_helper_called_by_value_leaves_the_bound_alone() {
    // triangular loop whose body calls helpers (the batched_ormqr shape): `k` is the outer counter, set only by the outer update
    let src = "float add(float a, float b) { return a + b; } void bump(float& acc, float v) { acc += v; }         void k(float* out) { float t = 0.0f;         for (int k2 = 0; k2 < 4; k2++) { float s = 0.0f; for (int j = 0; j < k2; j++) { s = add(s, (float)(j + 1)); bump(t, 1.0f); } out[k2] = s; }         out[4] = t; }";
    let mut want: Vec<f64> = (0..4)
        .map(|k: i32| f64::from((0..k).map(|j| j + 1).sum::<i32>()))
        .collect();
    want.push(f64::from((0..4).sum::<i32>()));
    assert_eq!(out(&lift(src).unwrap(), 5), want);
    // control: the same helper given the counter's bound by reference declines
    declines(
        "void bump(int& m) { m += 1; }         void k(float* out) { int m = 2; float s = 0.0f; for (int i = 0; i < m; i++) { bump(m); s += 1.0f; } out[0] = s; }",
    );
}

#[test]
fn a_bound_used_in_a_pointer_offset_argument_is_not_assigned_by_the_call() {
    // `put(out + n, ..)` gives the callee elements to write, never `n` itself (the decode kernel's shape)
    let src = "void put(float* p, float v) { p[0] = v; } \
        void k(float* out) { int n = 2; for (int i = 0; i < n; i++) { put(out + n + i, 1.0f); } }";
    assert_eq!(out(&lift(src).unwrap(), 4), vec![0.0, 0.0, 1.0, 1.0]);
}
