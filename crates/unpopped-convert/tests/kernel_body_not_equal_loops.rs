// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Census wall: a `!=` loop test** (`for (int ii = i_start; ii != i_end; ii += i_step)`, the scan kernels' forward/reverse
//! walk; 6 Baracuda kernels). The IR's `For` runs while below the end (above it, for a negative step), so a `!=` test is the same
//! loop wherever the C loop ends at all: it ends only on reaching the bound, which the IR also stops at. Oracle runs against
//! Rust references.

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

fn run(src: &str, block: u32, n: usize) -> Vec<f64> {
    let k = lift(src, block).unwrap();
    let mut args = vec![KernelArg::Buffer(TypedBuffer::from_f32(
        &[n as i64],
        &vec![0.0; n],
    ))];
    run_launch(&k, &mut args).expect("runs");
    let KernelArg::Buffer(o) = &args[0] else {
        panic!()
    };
    o.to_f64_vec()
}

#[test]
fn counting_up_and_down_to_a_bound() {
    // `z` is 0 at run time (the compiler cannot see it), so these are not constant loops
    let z = "int z = (int)threadIdx.x;";
    // digits appended in visit order, so a wrong order or count shows
    let up = format!(
        "void k(float* out) {{ {z} float a = 0.0f; for (int i = z; i != z + 4; i++) {{ a = a * 10.0f + (float)i; }} out[0] = a; }}"
    );
    assert_eq!(run(&up, 1, 1), vec![123.0]);
    let down = format!(
        "void k(float* out) {{ {z} float a = 0.0f; for (int i = z + 3; i != z - 1; i--) {{ a = a * 10.0f + (float)i; }} out[0] = a; }}"
    );
    assert_eq!(run(&down, 1, 1), vec![3210.0]);
    let stride = format!(
        "void k(float* out) {{ {z} float a = 0.0f; for (int i = z; i != z + 6; i += 2) {{ a = a * 10.0f + (float)i; }} out[0] = a; }}"
    );
    assert_eq!(run(&stride, 1, 1), vec![24.0]);
    let empty = format!(
        "void k(float* out) {{ {z} float a = 7.0f; for (int i = z + 3; i != z + 3; i++) {{ a = 0.0f; }} out[0] = a; }}"
    );
    assert_eq!(run(&empty, 1, 1), vec![7.0]);
}

#[test]
fn a_run_time_step_sign_walks_forward_or_backward() {
    // thread 0 walks 0..3 forward, thread 1 walks 3..0 backward (the scan kernels' `reverse` flag)
    let src = "void k(float* out) { int s; int e; int st; \
        if (threadIdx.x == 0) { s = 0; e = 4; st = 1; } else { s = 3; e = -1; st = -1; } \
        float a = 0.0f; for (int i = s; i != e; i += st) { a = a * 10.0f + (float)i; } out[threadIdx.x] = a; }";
    assert_eq!(run(src, 2, 2), vec![123.0, 3210.0]);
}

#[test]
fn an_unsigned_counter_that_counts_down_still_declines() {
    // `i != z; i--` on an unsigned counter has no direction the IR can test
    let src = "void k(float* out) { unsigned int z = threadIdx.x; float a = 0.0f; for (unsigned int i = z + 3u; i != z; i--) { a += 1.0f; } out[0] = a; }";
    // a clean decline, not an IR the checker then refuses (a negative step constant in an unsigned type)
    assert!(matches!(
        lift(src, 1),
        Err(KernelLiftError::NotYetLifted { .. })
    ));
    // control: the same loop counting up lifts
    let up = "void k(float* out) { unsigned int z = threadIdx.x; float a = 0.0f; for (unsigned int i = z; i != z + 3u; i++) { a += 1.0f; } out[0] = a; }";
    assert_eq!(run(up, 1, 1), vec![3.0]);
}
