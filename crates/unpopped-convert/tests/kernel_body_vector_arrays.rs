// SPDX-License-Identifier: MIT OR Apache-2.0
//! **U38b-3: arrays of CUDA vector types** (`float4 d[N];`). N vectors are one allocation of `N * lanes` scalars, read and written
//! as vectors through the same pointer-to-vector machinery as `float4* p` (`d[i]`, `d[i].x`), and viewable as scalars
//! (`((float*)d)[k]`: the lanes of element `i` are scalars `4 * i ..`, no padding). Oracle runs against Rust references.

use unpopped::exec::LaunchShape;
use unpopped::kernel::KernelFn;
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_convert::{Frontend, KernelLiftError, lift_kernel_launched};

fn parse(src: &str) -> Option<tree_sitter::Tree> {
    let b = src.replace("__shared__", "volatile  ");
    let mut p = tree_sitter::Parser::new();
    p.set_language(&tree_sitter_cpp::LANGUAGE.into()).ok()?;
    p.parse(b, None)
}

const C: Frontend = Frontend::new("c", "void", &[], "out", "in", parse);

fn lift(src: &str) -> Result<KernelFn, KernelLiftError> {
    lift_kernel_launched(&C, src, "k", LaunchShape::new([1, 1, 1], [1, 1, 1]))
}

fn run(src: &str, n: usize) -> Vec<f64> {
    let k = lift(src).unwrap();
    let mut args = vec![
        KernelArg::Buffer(TypedBuffer::from_f32(&[n as i64], &vec![0.0; n])),
        KernelArg::Buffer(TypedBuffer::from_f32(&[2], &[10.0, 20.0])),
    ];
    run_launch(&k, &mut args).expect("runs");
    let KernelArg::Buffer(o) = &args[0] else {
        panic!()
    };
    o.to_f64_vec()
}

#[test]
fn elements_are_vectors_written_and_read_whole_and_by_lane() {
    let src = "void k(float* out, const float* in) { float4 d[3]; \
        d[0] = make_float4(1.0f, 2.0f, 3.0f, 4.0f); d[2] = make_float4(in[0], in[1], 7.0f, 8.0f); \
        float4 v = d[2]; out[0] = v.x; out[1] = v.w; out[2] = d[0].z; out[3] = d[2].y; }";
    assert_eq!(run(src, 4), vec![10.0, 8.0, 3.0, 20.0]);
}

#[test]
fn the_scalar_view_is_the_flat_lane_array() {
    // `((float*)d)[k]`: element 1's lanes are scalars 4..8, element 0's are 0..4, with no padding between vectors
    let src = "void k(float* out, const float* in) { float4 d[2]; \
        d[0] = make_float4(1.0f, 2.0f, 3.0f, 4.0f); d[1] = make_float4(5.0f, 6.0f, 7.0f, 8.0f); \
        float* f = (float*)d; out[0] = f[0]; out[1] = f[3]; out[2] = f[4]; out[3] = f[7]; out[4] = ((float*)d)[5]; }";
    assert_eq!(run(src, 5), vec![1.0, 4.0, 5.0, 8.0, 6.0]);
}

#[test]
fn a_runtime_index_and_a_shared_vector_array() {
    let src = "void k(float* out, const float* in) { __shared__ float2 s[4]; \
        for (int i = 0; i < 4; i++) { s[i] = make_float2((float)i, (float)(i * 10)); } __syncthreads(); \
        float2 a = s[3]; out[0] = a.x; out[1] = a.y; out[2] = s[1].y; }";
    assert_eq!(run(src, 3), vec![3.0, 30.0, 10.0]);
}

#[test]
fn an_initialiser_or_two_dimensions_still_decline() {
    for src in [
        "void k(float* out) { float4 d[2] = {make_float4(1.0f, 2.0f, 3.0f, 4.0f), make_float4(1.0f, 2.0f, 3.0f, 4.0f)}; out[0] = d[0].x; }",
        "void k(float* out) { float4 d[2][2]; out[0] = 1.0f; }",
    ] {
        assert!(
            matches!(lift(src), Err(KernelLiftError::NotYetLifted { .. })),
            "{src}"
        );
    }
}

#[test]
fn a_lane_of_an_element_is_assigned_in_place() {
    let src = "void k(float* out, const float* in) { float4 d[2]; \
        d[1] = make_float4(1.0f, 2.0f, 3.0f, 4.0f); d[1].y = in[1]; d[1].z += 5.0f; \
        out[0] = d[1].x; out[1] = d[1].y; out[2] = d[1].z; out[3] = d[1].w; }";
    assert_eq!(run(src, 4), vec![1.0, 20.0, 8.0, 4.0]);
}

#[test]
fn a_struct_with_a_vector_field_and_a_vector_array_field() {
    // the vec_t shape: `float4 data[N / 4]` with a scalar view `((float*)data)[i]`, a fill and a load/store by vectors
    let src = "struct V { float4 head; float4 data[2];\n\
        float at(int i) { return ((float*)data)[i]; }\n\
        void fill(float x) { for (int i = 0; i < 2; ++i) { data[i] = make_float4(x, x, x, x); } head = make_float4(x, 0.0f, 0.0f, 0.0f); } };\n\
        void k(float* out, const float* in) { V v; v.fill(in[1]); v.data[1].w = 1.0f; v.head.y = in[0]; \
        out[0] = v.at(0); out[1] = v.at(7); out[2] = v.at(6); out[3] = v.head.x; out[4] = v.head.y; }";
    // data lanes all 20; element 1's w (scalar 7) is overwritten with 1; head = (20, 10, 0, 0)
    assert_eq!(run(src, 5), vec![20.0, 1.0, 20.0, 20.0, 10.0]);
}

#[test]
fn copying_a_struct_that_holds_an_array_of_vectors_declines() {
    let src = "struct V { float4 data[2]; };\nvoid k(float* out, const float* in) { V a; V b = a; out[0] = b.data[0].x; }";
    assert!(matches!(
        lift(src),
        Err(KernelLiftError::NotYetLifted { .. })
    ));
}
