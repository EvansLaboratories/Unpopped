// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Census wall: a namespace-qualified struct type** (`baracuda::elementwise::DimsI32 shape` as a kernel parameter, `ns::P p;` as a
//! local). The walker resolved only a bare `type_identifier`; the struct definition sits in a namespace, so the qualified spelling
//! resolves by its last segment. Each lift runs on the oracle against a Rust reference.

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

fn launched(src: &str) -> Result<KernelFn, KernelLiftError> {
    lift_kernel_launched(&C, src, "k", LaunchShape::new([1, 1, 1], [1, 1, 1]))
}

fn out(k: &KernelFn, input: &[f32]) -> Vec<f64> {
    let mut args = vec![
        KernelArg::Buffer(TypedBuffer::from_f32(&[1], &[0.0])),
        KernelArg::Buffer(TypedBuffer::from_f32(&[input.len() as i64], input)),
    ];
    run_launch(k, &mut args).expect("runs");
    let KernelArg::Buffer(o) = &args[0] else {
        panic!()
    };
    o.to_f64_vec()
}

#[test]
fn a_qualified_struct_value_parameter_lifts() {
    let src = "namespace a { namespace b { struct P { float x; float y; }; } }\n\
        void k(float* out, const float* in) { }\n";
    // the parameter is the struct, so the kernel takes it as a struct parameter; the lift must at least resolve the type
    let with_param = src.replace(
        "void k(float* out, const float* in) { }",
        "void k(float* out, const float* in, a::b::P p) { out[0] = in[0] * p.x + p.y; }",
    );
    assert!(launched(&with_param).is_ok(), "{:?}", launched(&with_param));
}

#[test]
fn a_qualified_struct_local_lifts_and_runs() {
    let src = "namespace a { namespace b { struct P { float x; float y; }; } }\n\
        void k(float* out, const float* in) { a::b::P p; p.x = in[0]; p.y = 4.0f; out[0] = p.x * 2.0f + p.y; }";
    assert_eq!(out(&launched(src).unwrap(), &[1.5]), vec![7.0]);
}

#[test]
fn a_qualified_name_that_is_not_a_struct_still_declines() {
    let src = "void k(float* out, const float* in) { std::mystery m; out[0] = in[0]; }";
    assert!(launched(src).is_err());
}
