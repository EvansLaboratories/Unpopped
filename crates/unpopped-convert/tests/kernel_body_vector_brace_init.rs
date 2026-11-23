// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Census wall: a brace-initialised vector** (`float4 s = {0.f, 0.f, 0.f, 0.f};`, 12 gguf mmvq kernels). The list is the lanes in
//! order; fewer initialisers than lanes zero the rest (aggregate initialisation); more decline. A list of the wrong kind of
//! element converts as an assignment would. Oracle runs against Rust references.

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

fn run(src: &str, n: usize) -> Vec<f64> {
    let k = lift(src).unwrap();
    let mut args = vec![
        KernelArg::Buffer(TypedBuffer::from_f32(&[n as i64], &vec![9.0; n])),
        KernelArg::Buffer(TypedBuffer::from_f32(&[2], &[10.0, 20.0])),
    ];
    run_launch(&k, &mut args).expect("runs");
    let KernelArg::Buffer(o) = &args[0] else {
        panic!()
    };
    o.to_f64_vec()
}

#[test]
fn a_full_list_is_the_lanes_and_a_short_list_zero_fills() {
    let src = "void k(float* out, const float* in) { float4 a = {1.0f, in[0], 3.0f, in[1] * 2.0f}; float4 b = {5.0f, 6.0f}; \
        float2 c = {7.0f, 8.0f}; \
        out[0] = a.x; out[1] = a.y; out[2] = a.z; out[3] = a.w; out[4] = b.x; out[5] = b.y; out[6] = b.z; out[7] = b.w; \
        out[8] = c.x + c.y; }";
    assert_eq!(
        run(src, 9),
        vec![1.0, 10.0, 3.0, 40.0, 5.0, 6.0, 0.0, 0.0, 15.0]
    );
}

#[test]
fn an_accumulator_vector_starts_at_zero_and_sums_like_the_gguf_kernels_do() {
    let src = "void k(float* out, const float* in) { float4 s = {0.f, 0.f, 0.f, 0.f}; \
        for (int i = 0; i < 2; ++i) { s.x += in[i]; s.w += in[i] * 2.0f; } out[0] = s.x; out[1] = s.y; out[2] = s.w; }";
    assert_eq!(run(src, 3), vec![30.0, 0.0, 60.0]);
}

#[test]
fn more_initialisers_than_lanes_decline() {
    let r = lift(
        "void k(float* out, const float* in) { float2 a = {1.0f, 2.0f, 3.0f}; out[0] = a.x; }",
    );
    assert!(
        matches!(r, Err(KernelLiftError::NotYetLifted { .. })),
        "{r:?}"
    );
}
