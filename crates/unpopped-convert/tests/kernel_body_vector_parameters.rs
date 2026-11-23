// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Census wall: a vector-typed parameter of an inlined function** (`dfloat2 & v` with `typedef float2 dfloat2;`, the gguf
//! `dequantize_q4_0(vx, ib, iqs, v)` out-parameter; 30 kernels). `V& v` aliases the caller's vector local (it must have the
//! same shape); `V v` and `const V& v` copy its lanes. Oracle runs against Rust references.

use unpopped::exec::LaunchShape;
use unpopped::kernel::KernelFn;
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_convert::{Frontend, KernelLiftError, lift_kernel_launched};

fn parse(src: &str) -> Option<tree_sitter::Tree> {
    let mut p = tree_sitter::Parser::new();
    p.set_language(&tree_sitter_cpp::LANGUAGE.into()).ok()?;
    p.parse(src.replace("__device__", "          "), None)
}

const C: Frontend = Frontend::new("c", "void", &[], "out", "in", parse);

fn lift(src: &str) -> Result<KernelFn, KernelLiftError> {
    lift_kernel_launched(&C, src, "k", LaunchShape::new([1, 1, 1], [1, 1, 1]))
}

fn run(src: &str, n: usize) -> Vec<f64> {
    let k = lift(src).unwrap();
    let mut args = vec![
        KernelArg::Buffer(TypedBuffer::from_f32(&[n as i64], &vec![0.0; n])),
        KernelArg::Buffer(TypedBuffer::from_f32(&[2], &[3.0, 4.0])),
    ];
    run_launch(&k, &mut args).expect("runs");
    let KernelArg::Buffer(o) = &args[0] else {
        panic!()
    };
    o.to_f64_vec()
}

const DEFS: &str = "typedef float2 dfloat2;\n\
    __device__ void fill(const float* in, int i, dfloat2 & v) { v.x = in[i] * 2.0f; v.y = in[i] + 1.0f; }\n\
    __device__ float sum(const dfloat2 w) { return w.x + w.y; }\n\
    __device__ float byval(dfloat2 w) { w.x = 100.0f; return w.x; }\n";

#[test]
fn a_reference_vector_is_filled_in_the_callers_local_and_a_value_vector_is_a_copy() {
    let src = format!(
        "{DEFS}void k(float* out, const float* in) {{ dfloat2 v; fill(in, 1, v); out[0] = v.x; out[1] = v.y; \
         out[2] = sum(v); out[3] = byval(v); out[4] = v.x; }}"
    );
    // fill: v = (8, 5); sum = 13; byval writes its own copy (100) and leaves v.x = 8
    assert_eq!(run(&src, 5), vec![8.0, 5.0, 13.0, 100.0, 8.0]);
}

#[test]
fn a_reference_to_the_wrong_shape_or_a_non_vector_declines() {
    let wrong = format!(
        "{DEFS}void k(float* out, const float* in) {{ float4 v; fill(in, 0, v); out[0] = v.x; }}"
    );
    let scalar = format!(
        "{DEFS}void k(float* out, const float* in) {{ float v = 1.0f; fill(in, 0, v); out[0] = v; }}"
    );
    for src in [wrong, scalar] {
        assert!(
            matches!(lift(&src), Err(KernelLiftError::NotYetLifted { .. })),
            "{src}"
        );
    }
}

#[test]
fn a_const_reference_vector_takes_any_vector_value_not_only_a_local() {
    let src = "typedef float2 dfloat2;\n__device__ float sum(const dfloat2 & w) { return w.x + w.y; }\n\
        void k(float* out, const float* in) { out[0] = sum(make_float2(in[0], in[1])); }";
    assert_eq!(run(src, 1), vec![7.0]);
}
