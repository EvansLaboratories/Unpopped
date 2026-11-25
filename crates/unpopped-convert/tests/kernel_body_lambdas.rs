// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Census wall: a named lambda `auto f = [&](args) -> T { body };`** (33 real instantiations: ctc `ext_t`, grid_sample `fetch`,
//! contiguize `src_off_for`, ormqr `step`). A lambda captured by reference is a function that sees the caller's locals, so a call
//! inlines the body in the caller's scope; what it assigns reaches the caller. An explicit capture list, a capture by value, a
//! deduced non-`void` return type, and a call that sees a captured name shadowed decline by name. Oracle runs against Rust.

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

const IN: [f32; 6] = [1.5, -2.0, 4.0, 0.5, 8.0, -1.0];

fn run(k: &KernelFn, n: i32) -> Vec<f64> {
    let mut args = vec![
        KernelArg::Buffer(TypedBuffer::from_f32(&[4], &[0.0; 4])),
        KernelArg::Buffer(TypedBuffer::from_f32(&[6], &IN)),
        KernelArg::Int(i128::from(n)),
    ];
    run_launch(k, &mut args).expect("runs");
    let KernelArg::Buffer(o) = &args[0] else {
        panic!()
    };
    o.to_f64_vec()
}

fn declines(src: &str, needle: &str) {
    match lift(src) {
        Err(KernelLiftError::NotYetLifted { construct, .. }) => {
            assert!(construct.contains(needle), "`{construct}` lacks `{needle}`");
        }
        other => panic!("expected NotYetLifted({needle}), got {other:?}"),
    }
}

#[test]
fn a_value_lambda_reads_the_callers_locals_and_a_void_lambda_writes_them() {
    let k = lift(
        "void k(float* out, const float* in, int n) { float scale = 2.0f; int bias = n;\n\
           auto fetch = [&](int i) -> float { if (i < 0 || i >= 6) return 0.0f; return in[i] * scale + (float)bias; };\n\
           auto bump = [&](int by) { scale = scale + (float)by; };\n\
           auto pick = [&](int i) -> int { return i + 1; };\n\
           out[0] = fetch(1); bump(3); out[1] = fetch(2); out[2] = fetch(9); out[3] = fetch(pick(pick(3))); }",
    )
    .expect("lifts (was a decline)");
    for n in [0, 7] {
        let b = n as f32;
        let r = run(&k, n);
        assert_eq!(r[0], f64::from(IN[1] * 2.0 + b));
        assert_eq!(r[1], f64::from(IN[2] * 5.0 + b));
        assert_eq!(r[2], 0.0);
        assert_eq!(
            r[3],
            f64::from(IN[5] * 5.0 + b),
            "the argument is a call of the same lambda"
        );
    }
}

#[test]
fn a_lambda_without_parameters_and_one_with_no_capture_lift() {
    let k = lift(
        "void k(float* out, const float* in, int n) { int c = 0; int x = 3; auto tick = [&] { c = c + 2; };\n\
           auto sq = [](float x) -> float { return x * x; };\n\
           tick(); tick(); out[0] = (float)c; out[1] = sq(in[2]); out[2] = (float)x; }",
    )
    .expect("lifts");
    // the lambda's parameter `x` must not outlive the call: the caller's own `x` is still 3
    assert_eq!(run(&k, 0)[..3], [4.0, 16.0, 3.0]);
}

#[test]
fn a_lambda_advances_a_captured_pointer_for_the_caller() {
    let k = lift(
        "void k(float* out, const float* in, int n) { const float* p = in;\n\
           auto next = [&]() -> float { float v = p[0]; p += 1; return v; };\n\
           out[0] = next(); out[1] = next(); out[2] = p[0]; }",
    )
    .expect("lifts");
    assert_eq!(
        run(&k, 0)[..3],
        [f64::from(IN[0]), f64::from(IN[1]), f64::from(IN[2])]
    );
}

#[test]
fn what_cannot_be_a_lambda_declines_by_name() {
    declines(
        "void k(float* out, const float* in, int n) { float s = 1.0f; auto f = [s](int i) -> float { return s; }; out[0] = f(1); }",
        "explicit capture list",
    );
    declines(
        "void k(float* out, const float* in, int n) { float s = 1.0f; auto f = [=](int i) -> float { return s; }; out[0] = f(1); }",
        "capture by value",
    );
    declines(
        "void k(float* out, const float* in, int n) { auto f = [&](int i) { return in[i]; }; out[0] = f(1); }",
        "deduced",
    );
    // the call sees `s` as another variable than the one the lambda captured
    declines(
        "void k(float* out, const float* in, int n) { float s = 1.0f; auto f = [&](int i) -> float { return s; }; { float s = 5.0f; out[0] = f(1); out[1] = s; } }",
        "shadowed",
    );
}
