// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Census wall: a namespace-qualified template call** (`baracuda::atomic::add<float>(&dx[i], v)`, 10 Baracuda kernels). The
//! explicit-argument call expansion matched only a bare `f<T>(..)`; a helper is found by its short name (the walker is
//! namespace-blind), so the qualified spelling now expands to the same plain instance. Oracle runs against Rust references.

use unpopped::exec::LaunchShape;
use unpopped::kernel::KernelFn;
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_convert::{Frontend, KernelLiftError, lift_kernel_launched};

fn parse(src: &str) -> Option<tree_sitter::Tree> {
    let mut b = src.to_string();
    for q in ["__device__", "__forceinline__", "__shared__"] {
        b = b.replace(q, &" ".repeat(q.len()));
    }
    let mut p = tree_sitter::Parser::new();
    p.set_language(&tree_sitter_cpp::LANGUAGE.into()).ok()?;
    p.parse(b, None)
}

const C: Frontend = Frontend::new("c", "void", &[], "out", "in", parse);

fn lift(src: &str, block: u32) -> Result<KernelFn, KernelLiftError> {
    lift_kernel_launched(&C, src, "k", LaunchShape::new([block, 1, 1], [1, 1, 1]))
}

fn run(k: &KernelFn, input: &[f32], n_out: usize) -> Vec<f64> {
    let mut args = vec![
        KernelArg::Buffer(TypedBuffer::from_f32(&[n_out as i64], &vec![0.0; n_out])),
        KernelArg::Buffer(TypedBuffer::from_f32(&[input.len() as i64], input)),
    ];
    run_launch(k, &mut args).expect("runs");
    let KernelArg::Buffer(o) = &args[0] else {
        panic!()
    };
    o.to_f64_vec()
}

const NS: &str = "namespace ns { namespace inner {\n\
    template <typename T, int K>\n__device__ __forceinline__ T scale(T v) { return v * (T)K; }\n\
    template <typename T>\n__device__ __forceinline__ void add(T* addr, T val) { atomicAdd(addr, val); }\n\
    } }\n";

#[test]
fn a_qualified_template_call_expands_like_a_bare_one() {
    let src = format!(
        "{NS}void k(float* out, const float* in) {{ out[0] = ns::inner::scale<float, 3>(in[0]) + scale<float, 5>(in[0]); }}"
    );
    // `scale<float, 5>` is the bare spelling of the same helper: both forms resolve
    let k = lift(&src, 1).unwrap();
    assert_eq!(run(&k, &[2.0], 1), vec![6.0 + 10.0]);
    // control: distinct arguments give distinct instances
    let src3 = src.replace("scale<float, 5>", "scale<float, 7>");
    assert_eq!(run(&lift(&src3, 1).unwrap(), &[2.0], 1), vec![6.0 + 14.0]);
}

#[test]
fn a_qualified_atomic_add_forwarder_accumulates_across_threads() {
    let src = format!(
        "{NS}void k(float* out, const float* in) {{ ns::inner::add<float>(&out[0], in[threadIdx.x]); }}"
    );
    let k = lift(&src, 4).unwrap();
    assert_eq!(run(&k, &[1.0, 2.0, 4.0, 8.0], 1), vec![15.0]);
}

#[test]
fn a_qualified_call_that_is_not_a_template_in_the_file_still_declines() {
    let src = "void k(float* out, const float* in) { out[0] = other::f<float>(in[0]); }";
    assert!(lift(src, 1).is_err());
}

#[test]
fn an_atomic_through_an_offset_pointer_hits_the_element_it_names() {
    // `p = out + 2` is byte 8 of a float buffer: the atomic must land on element 2, not 8 or 0
    let src = "void k(float* out, const float* in) { float* p = out + 2; atomicAdd(p, in[threadIdx.x]); atomicAdd(&p[1], in[1]); }";
    let k = lift(src, 3).unwrap();
    assert_eq!(run(&k, &[1.0, 2.0, 4.0], 5), vec![0.0, 0.0, 7.0, 6.0, 0.0]);
}

#[test]
fn an_atomic_through_a_shared_array_pointer_lands_on_the_element() {
    let src = "void k(float* out, const float* in) { extern __shared__ float s[]; float* q = s + 1; \
        if (threadIdx.x == 0) { s[0] = 0.0f; s[1] = 0.0f; } __syncthreads(); atomicAdd(q, in[threadIdx.x]); \
        __syncthreads(); if (threadIdx.x == 0) { out[0] = s[0]; out[1] = s[1]; } }";
    let ctx = unpopped_convert::LiftContext::new(LaunchShape::new([3, 1, 1], [1, 1, 1]))
        .with_dynamic_shared_bytes(8);
    let k = unpopped_convert::lift_kernel_in(&C, src, "k", &ctx).unwrap();
    assert_eq!(run(&k, &[1.0, 2.0, 4.0], 2), vec![0.0, 7.0]);
}

#[test]
fn an_atomic_through_a_cast_pointer_still_declines() {
    // the pointer is read at another type than the buffer's: no element of the buffer is named
    let src = "void k(float* out, const float* in) { int* p = (int*)out; atomicAdd(p, 1); }";
    match lift(src, 1) {
        Err(KernelLiftError::NotYetLifted { construct, .. }) => {
            assert!(
                construct.contains("atomic through a pointer"),
                "{construct}"
            );
        }
        other => panic!("{other:?}"),
    }
}
