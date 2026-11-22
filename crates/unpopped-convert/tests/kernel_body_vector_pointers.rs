// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Census wall: a pointer to a vector type** (`float4* out_vec = reinterpret_cast<float4*>(out_row);`, 4 Baracuda kernels). The
//! declaration read `float4*` as a vector local and declined the pointer declarator; a pointer to a CUDA vector type is a pointer
//! whose pointee is the vector (lane type and lane count), indexed in whole vectors. Oracle runs against Rust references.

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

fn lift(src: &str, block: u32) -> Result<KernelFn, KernelLiftError> {
    lift_kernel_launched(&C, src, "k", LaunchShape::new([block, 1, 1], [1, 1, 1]))
}

fn run(k: &KernelFn, input: &[f32]) -> Vec<f64> {
    let mut args = vec![
        KernelArg::Buffer(TypedBuffer::from_f32(
            &[input.len() as i64],
            &vec![0.0; input.len()],
        )),
        KernelArg::Buffer(TypedBuffer::from_f32(&[input.len() as i64], input)),
    ];
    run_launch(k, &mut args).expect("runs");
    let KernelArg::Buffer(o) = &args[0] else {
        panic!()
    };
    o.to_f64_vec()
}

#[test]
fn a_float4_pointer_scales_whole_vectors_in_place() {
    // the divide_by_rms shape: each thread scales one float4 of the row by r
    let src = "void k(float* out, const float* in) { const float4* in_vec = reinterpret_cast<const float4*>(in); \
        float4* out_vec = reinterpret_cast<float4*>(out); float r = 2.0f; \
        float4 v = in_vec[threadIdx.x]; v.x *= r; v.y *= r; v.z *= r; v.w *= r; out_vec[threadIdx.x] = v; }";
    let k = lift(src, 2).unwrap();
    let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let want: Vec<f64> = input.iter().map(|v| f64::from(v * 2.0)).collect();
    assert_eq!(run(&k, &input), want);
}

#[test]
fn a_pointer_of_the_wrong_vector_shape_still_declines() {
    // `float2*` initialised from a `float4*` is not the same pointee
    let src = "void k(float* out, const float* in) { float4* a = reinterpret_cast<float4*>(out); float2* b = a; out[0] = 1.0f; }";
    assert!(matches!(
        lift(src, 1),
        Err(KernelLiftError::NotYetLifted { .. })
    ));
}
