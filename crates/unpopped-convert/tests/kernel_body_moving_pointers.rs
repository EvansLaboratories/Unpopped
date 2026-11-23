// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Census wall: a pointer local that moves** (`unsigned char* sp = smem_raw; T* a = (T*)sp; sp += n * sizeof(T);`, the carve-up
//! of one shared buffer; 8 Baracuda kernels). `p += n` / `p -= n` moves the pointer's byte offset by `n` elements; a pointer
//! made from `p` before the move keeps the old position. Oracle runs against Rust references.

use unpopped::exec::LaunchShape;
use unpopped::kernel::KernelFn;
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_convert::{Frontend, KernelLiftError, LiftContext, lift_kernel_in};

fn parse(src: &str) -> Option<tree_sitter::Tree> {
    let b = src.replace("__shared__", &" ".repeat("__shared__".len()));
    let mut p = tree_sitter::Parser::new();
    p.set_language(&tree_sitter_cpp::LANGUAGE.into()).ok()?;
    p.parse(b, None)
}

const C: Frontend = Frontend::new("c", "void", &[], "out", "in", parse);

fn lift(src: &str, block: u32, shared: u32) -> Result<KernelFn, KernelLiftError> {
    let ctx = LiftContext::new(LaunchShape::new([block, 1, 1], [1, 1, 1]))
        .with_dynamic_shared_bytes(shared);
    lift_kernel_in(&C, src, "k", &ctx)
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

#[test]
fn a_shared_buffer_carved_into_arrays_by_a_moving_byte_pointer() {
    // a: 4 floats at byte 0, b: 2 floats at byte 16. If `sp` did not move, b would alias a and corrupt it.
    let src = "void k(float* out, const float* in) { extern __shared__ unsigned char raw[]; unsigned char* sp = raw; \
        float* a = reinterpret_cast<float*>(sp); sp += 4 * sizeof(float); \
        float* b = reinterpret_cast<float*>(sp); sp += 2 * sizeof(float); \
        a[threadIdx.x] = in[threadIdx.x]; if (threadIdx.x < 2) { b[threadIdx.x] = in[threadIdx.x] * 10.0f; } __syncthreads(); \
        out[threadIdx.x] = a[threadIdx.x] + 100.0f * b[threadIdx.x % 2]; }";
    let k = lift(src, 4, 32).unwrap();
    let input = [1.0, 2.0, 3.0, 4.0];
    let want: Vec<f64> = (0..4usize)
        .map(|t| f64::from(input[t]) + 100.0 * f64::from(input[t % 2] * 10.0))
        .collect();
    assert_eq!(run(&k, &input, 4), want);
}

#[test]
fn a_pointer_that_moves_in_a_loop_reads_the_next_element_each_trip() {
    let src = "void k(float* out, const float* in) { extern __shared__ float s[]; \
        if (threadIdx.x == 0) { s[0] = in[0]; s[1] = in[1]; s[2] = in[2]; s[3] = in[3]; } __syncthreads(); \
        float* p = s; float acc = 0.0f; for (int i = 0; i < 3; i++) { acc = acc * 10.0f + p[0]; p += 1; } \
        out[threadIdx.x] = acc + p[0] * 1000.0f; p -= 2; out[threadIdx.x] = out[threadIdx.x] + p[0] * 100000.0f; }";
    let k = lift(src, 1, 16).unwrap();
    // 1,2,4 gives acc = 124; p then points at s[3] = 8 (8000); p -= 2 points at s[1] = 2 (200000)
    assert_eq!(
        run(&k, &[1.0, 2.0, 4.0, 8.0], 1),
        vec![124.0 + 8000.0 + 200_000.0]
    );
}

#[test]
fn any_other_assignment_to_a_pointer_local_declines() {
    let reseat = "void k(float* out, const float* in) { extern __shared__ float s[]; float* p = s; float* q = s + 1; p = q; out[0] = p[0]; }";
    let times = "void k(float* out, const float* in) { extern __shared__ float s[]; float* p = s; p *= 2; out[0] = p[0]; }";
    for src in [reseat, times] {
        assert!(
            matches!(lift(src, 1, 16), Err(KernelLiftError::NotYetLifted { .. })),
            "{src}"
        );
    }
}
