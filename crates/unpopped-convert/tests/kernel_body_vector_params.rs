// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Census wall: a pointer parameter of a vector type** (`const T* A` with `T = cuFloatComplex`, the batched ormqr kernels: 12 real
//! instantiations stopped at "the type `float2`"). The parameter is a buffer of the lane type (`float`), read and written two (or four)
//! lanes at a time, so element `i` of a `float2*` is the lanes `2i` and `2i + 1` of the buffer. A vector passed by value
//! (`float2 v` as a parameter) still declines. Oracle runs against Rust references.

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

const IN: [f32; 8] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

fn run(k: &KernelFn, outs: usize) -> Vec<f64> {
    let mut args = vec![
        KernelArg::Buffer(TypedBuffer::from_f32(&[outs as i64], &vec![0.0; outs])),
        KernelArg::Buffer(TypedBuffer::from_f32(&[8], &IN)),
    ];
    run_launch(k, &mut args).expect("runs");
    let KernelArg::Buffer(o) = &args[0] else {
        panic!()
    };
    o.to_f64_vec()
}

#[test]
fn a_float2_pointer_parameter_reads_and_writes_lane_pairs() {
    let k = lift(
        "void k(float2* out, const float2* in) { for (int i = 0; i < 4; ++i) { float2 v = in[i]; out[i] = make_float2(v.y, v.x + 10.0f); } }",
    )
    .expect("lifts (was a decline: the type `float2`)");
    // element i is the pair (IN[2i], IN[2i+1]); the result pair is (y, x + 10)
    let want: Vec<f64> = (0..4)
        .flat_map(|i| [f64::from(IN[2 * i + 1]), f64::from(IN[2 * i]) + 10.0])
        .collect();
    assert_eq!(run(&k, 8), want);
}

#[test]
fn a_complex_pointer_parameter_through_the_cu_complex_names() {
    let k = lift(
        "void k(cuFloatComplex* z, const cuFloatComplex* a) { for (int i = 0; i < 4; ++i) { z[i] = cuCmulf(a[i], a[i]); } }",
    )
    .expect("lifts");
    // (x + iy)^2 = (x^2 - y^2, 2xy)
    let want: Vec<f64> = (0..4)
        .flat_map(|i| {
            let (x, y) = (f64::from(IN[2 * i]), f64::from(IN[2 * i + 1]));
            [x * x - y * y, 2.0 * x * y]
        })
        .collect();
    assert_eq!(run(&k, 8), want);
}

#[test]
fn a_pointer_local_into_a_vector_parameter_moves_by_whole_elements() {
    let k = lift(
        "void k(float2* out, const float2* in) { const float2* q = in + 2; float2 a = q[0]; float2 b = q[1]; out[0] = make_float2(a.x, b.y); }",
    )
    .expect("lifts");
    // q[0] = (5, 6), q[1] = (7, 8)
    assert_eq!(run(&k, 2)[..2], [5.0, 8.0]);
}

#[test]
fn address_of_an_element_and_a_moving_pointer_step_whole_vectors() {
    let k = lift(
        "void k(float2* out, const float2* in) { const float2* r = &in[1]; float2 a = r[0]; const float2* q = in; q += 3; float2 b = q[0];          out[0] = make_float2(a.x, b.y); }",
    )
    .expect("lifts");
    // &in[1] = (3, 4); in + 3 = (7, 8)
    assert_eq!(run(&k, 2)[..2], [3.0, 8.0]);
}

#[test]
fn a_lane_of_an_element_is_read_and_written_in_place() {
    // `out[i].z = v` writes lane z of element i: byte offset i * 16 + 8 in a float4 buffer; `in[i].y` reads the second lane
    let k = lift(
        "void k(float4* out, const float4* in) { out[0].x = 1.0f; out[1].z = in[1].y; out[1].w = in[0].w + 7.0f; }",
    )
    .expect("lifts (was a decline: the type `float4`)");
    // in = [1,2,3,4, 5,6,7,8]: in[1].y = 6, in[0].w = 4
    assert_eq!(run(&k, 8), vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6.0, 11.0]);
}

#[test]
fn a_vector_by_value_parameter_still_declines() {
    match lift("void k(float* out, float2 v) { out[0] = 1.0f; }") {
        Err(KernelLiftError::NotYetLifted { construct, .. }) => {
            assert!(
                construct.contains("vector type passed by value"),
                "{construct}"
            );
        }
        other => panic!("expected a decline, got {other:?}"),
    }
}
