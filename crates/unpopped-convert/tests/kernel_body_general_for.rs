// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Census wall: a `for` that is not a counted loop** (`for (k = 2; k <= n; k <<= 1)`, `for (j = k >> 1; j > 0; j >>= 1)`, the bitonic
//! sort loops; a condition on a product). `T i = a; while (cond) { body; update }`, the counter an ordinary local. A `continue` in the
//! body would skip the update, so such a loop still declines (a `continue` inside a nested loop is that loop's own). Oracle runs.

use unpopped::exec::LaunchShape;
use unpopped::kernel::KernelFn;
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_convert::{Frontend, KernelLiftError, LiftContext, lift_kernel_in};

fn parse(src: &str) -> Option<tree_sitter::Tree> {
    let mut p = tree_sitter::Parser::new();
    p.set_language(&tree_sitter_cpp::LANGUAGE.into()).ok()?;
    p.parse(src, None)
}

const C: Frontend = Frontend::new("c", "void", &[], "out", "in", parse);

fn lift(src: &str) -> Result<KernelFn, KernelLiftError> {
    lift_kernel_in(
        &C,
        src,
        "k",
        &LiftContext::new(LaunchShape::new([1, 1, 1], [1, 1, 1])),
    )
}

fn run(k: &KernelFn, n: i128, outn: usize) -> Vec<f64> {
    let mut args = vec![
        KernelArg::Buffer(TypedBuffer::from_f32(&[outn as i64], &vec![0.0; outn])),
        KernelArg::Buffer(TypedBuffer::from_f32(&[4], &[1.0, 2.0, 3.0, 4.0])),
        KernelArg::Int(n),
    ];
    run_launch(k, &mut args).expect("runs");
    let KernelArg::Buffer(o) = &args[0] else {
        panic!()
    };
    o.to_f64_vec()
}

#[test]
fn doubling_and_halving_counters_and_a_condition_on_a_product_loop_like_c() {
    let k = lift(
        "void k(float* out, const float* in, int n) { int a = 0; int b = 0; int c = 0; \
         for (int k = 2; k <= n; k <<= 1) { a += k; } \
         for (int j = n >> 1; j > 0; j >>= 1) { b += j; } \
         for (int i = 1; i * i <= n; i += 2) { c += i; } \
         out[0] = (float)a; out[1] = (float)b; out[2] = (float)c; }",
    )
    .unwrap();
    // n = 20: k = 2, 4, 8, 16; j = 10, 5, 2, 1; i = 1, 3 (3 * 3 = 9 <= 20, 5 * 5 = 25 > 20)
    assert_eq!(run(&k, 20, 3), vec![30.0, 18.0, 4.0]);
    // n = 1: no pass of the first, none of the second (j = 0), one of the third
    assert_eq!(run(&k, 1, 3), vec![0.0, 0.0, 1.0]);
}

#[test]
fn a_break_ends_it_and_a_condition_that_reads_memory_is_re_read_each_pass() {
    let k = lift(
        "void k(float* out, const float* in, int n) { float s = 0.0f; int t = 0; \
         for (int k = 1; k < n; k <<= 1) { s += (float)k; if (k >= 4) break; } \
         for (int m = 0; in[m] < 3.5f; m += 1) { t += 1; } \
         out[0] = s; out[1] = (float)t; }",
    )
    .unwrap();
    // n = 100: 1 + 2 + 4 then the break; in = 1, 2, 3, 4: three entries below 3.5
    assert_eq!(run(&k, 100, 2), vec![7.0, 3.0]);
}

#[test]
fn a_continue_that_would_skip_the_update_declines_but_a_nested_loops_own_does_not() {
    let r = lift(
        "void k(float* out, const float* in, int n) { for (int k = 1; k < n; k <<= 1) { if (k == 2) continue; out[0] = 1.0f; } }",
    );
    assert!(
        matches!(r, Err(KernelLiftError::NotYetLifted { .. })),
        "{r:?}"
    );
    let k = lift(
        "void k(float* out, const float* in, int n) { int a = 0; for (int k = 1; k < n; k <<= 1) { \
         for (int q = 0; q < 3; ++q) { if (q == 1) continue; a += 1; } } out[0] = (float)a; }",
    )
    .unwrap();
    // n = 9: k = 1, 2, 4, 8 (four passes), each adding q = 0 and q = 2
    assert_eq!(run(&k, 9, 1), vec![8.0]);
}
