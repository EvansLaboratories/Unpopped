// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Local arrays with a brace initialiser** (`int a[4] = {1, 2};`; census: 14 Baracuda kernels stopped on it). The listed elements are
//! stored in order and the rest are zero, as C does for a short list. Each lifted kernel runs on the oracle against a Rust reference.

use unpopped::exec::LaunchShape;
use unpopped::kernel::KernelFn;
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_convert::{Frontend, KernelLiftError, LiftContext, lift_kernel_in};

fn parse(src: &str) -> Option<tree_sitter::Tree> {
    let mut b = src.to_string();
    b = b.replace("__shared__", &" ".repeat(10));
    let mut p = tree_sitter::Parser::new();
    p.set_language(&tree_sitter_cpp::LANGUAGE.into()).ok()?;
    p.parse(b, None)
}

const C: Frontend = Frontend::new("c", "void", &[], "out", "in", parse);

fn ctx() -> LiftContext {
    LiftContext::new(LaunchShape::new([1, 1, 1], [1, 1, 1]))
}

fn lift(src: &str) -> Result<KernelFn, KernelLiftError> {
    lift_kernel_in(&C, src, "k", &ctx())
}

fn run(k: &KernelFn, n: usize, extra: Vec<KernelArg>) -> Vec<f64> {
    let mut args = vec![KernelArg::Buffer(TypedBuffer::from_f32(
        &[n as i64],
        &vec![-1.0; n],
    ))];
    args.extend(extra);
    run_launch(k, &mut args).expect("runs");
    let KernelArg::Buffer(o) = &args[0] else {
        panic!()
    };
    o.to_f64_vec()
}

fn not_yet(r: Result<KernelFn, KernelLiftError>, needle: &str) {
    match r {
        Err(KernelLiftError::NotYetLifted { construct, .. }) => {
            assert!(construct.contains(needle), "`{construct}` lacks `{needle}`");
        }
        other => panic!("expected NotYetLifted({needle}), got {other:?}"),
    }
}

#[test]
fn a_full_a_short_and_an_empty_brace_list_fill_the_array_as_c_does() {
    let src = "void k(float* out) { int a[4] = {1, 2, 3, 4}; float b[4] = {1.5f, 2.5f}; float c[3] = {0}; float d[2] = {}; \
        for (int i = 0; i < 4; ++i) { out[i] = (float)a[i]; out[4 + i] = b[i]; } \
        for (int i = 0; i < 3; ++i) { out[8 + i] = c[i]; } out[11] = d[0]; out[12] = d[1]; }";
    let got = run(&lift(src).unwrap(), 13, vec![]);
    assert_eq!(
        got,
        vec![
            1.0, 2.0, 3.0, 4.0, 1.5, 2.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ]
    );
}

#[test]
fn an_initialised_array_is_an_ordinary_array_afterwards() {
    // a table looked up at a runtime index, then written and read back
    let src = "void k(float* out, const float* in) { int t[4] = {10, 20, 30, 40}; int idx = (int)in[0]; out[0] = (float)t[idx]; \
        t[idx] = 7; out[1] = (float)t[idx]; out[2] = (float)t[(idx + 1) % 4]; }";
    let k = lift(src).unwrap();
    let extra = vec![KernelArg::Buffer(TypedBuffer::from_f32(&[1], &[2.0]))];
    assert_eq!(run(&k, 3, extra), vec![30.0, 7.0, 40.0]);
}

#[test]
fn the_size_may_be_a_template_argument_and_the_initialiser_declines_by_name_when_it_is_wrong() {
    let src = "template <int N>\nvoid k(float* out) { float a[N] = {2.0f}; out[0] = a[0]; out[1] = a[N - 1]; }";
    let c = ctx().with_const("N", 3);
    let k = lift_kernel_in(&C, src, "k", &c).unwrap();
    assert_eq!(run(&k, 2, vec![]), vec![2.0, 0.0]);
    // too many initialisers, a nested brace, a non-brace initialiser (a `__shared__` array with one also declines, but the test parser blanks `__shared__`)
    not_yet(
        lift("void k(float* out) { int a[2] = {1, 2, 3}; out[0] = 1.0f; }"),
        "3 initialisers for an array of 2 elements",
    );
    not_yet(
        lift("void k(float* out) { int a[2] = {{1}, 2}; out[0] = 1.0f; }"),
        "initializer_list",
    );
    not_yet(
        lift("void k(float* out, int b) { int a[2] = b; out[0] = 1.0f; }"),
        "not a brace list",
    );
}
