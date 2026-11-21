// SPDX-License-Identifier: MIT OR Apache-2.0
//! **U19 slice 2: thread builtins, `__syncthreads`, `__shared__`** (`docs/options-u19-cfamily-body-frontend.md`). Launched kernels
//! run on the oracle's launch machine against a Rust reference. `tree-sitter-cpp` does not parse `__shared__` (a CUDA
//! qualifier), so the test parser blanks it to `static` **of the same length**: byte offsets are unchanged, and the walker reads
//! the original text, as it will behind Baracuda's grammar.

use unpopped::exec::LaunchShape;
use unpopped::kernel::KernelFn;
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::{KernelArg, KernelTrap};
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_convert::{Frontend, KernelLiftError, lift_kernel, lift_kernel_launched};

fn parse(src: &str) -> Option<tree_sitter::Tree> {
    let blanked = src.replace("__shared__", "static    ");
    let mut p = tree_sitter::Parser::new();
    p.set_language(&tree_sitter_cpp::LANGUAGE.into()).ok()?;
    p.parse(blanked, None)
}

const C: Frontend = Frontend::new("c", "void", &[], "out", "in", parse);

fn launched(src: &str, block: u32, grid: u32) -> Result<KernelFn, KernelLiftError> {
    lift_kernel_launched(&C, src, "k", LaunchShape::new([block, 1, 1], [grid, 1, 1]))
}

fn f32s(v: &[f32]) -> KernelArg {
    KernelArg::Buffer(TypedBuffer::from_f32(&[v.len() as i64], v))
}

fn out_of(args: &[KernelArg]) -> Vec<f64> {
    let KernelArg::Buffer(o) = &args[0] else {
        panic!("out first")
    };
    o.to_f64_vec()
}

#[test]
fn the_global_thread_index_with_a_bounds_guard() {
    let k = launched(
        "void k(float* out, const float* a, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) { out[i] = a[i] * 2.0f + (float)gridDim.x; } }",
        4,
        2,
    )
    .unwrap();
    let a = [1.5f32, -2.0, 3.25, 4.0, 0.5, -7.0, 99.0, 99.0];
    let mut args = vec![f32s(&[-1.0; 8]), f32s(&a), KernelArg::Int(6)];
    run_launch(&k, &mut args).expect("runs");
    let want: Vec<f64> = (0..8)
        .map(|i| {
            if i < 6 {
                f64::from(a[i] * 2.0 + 2.0)
            } else {
                -1.0
            }
        })
        .collect();
    assert_eq!(out_of(&args), want);
}

const REVERSE: &str = "void k(float* out, const float* x) { __shared__ float tile[8]; tile[threadIdx.x] = x[threadIdx.x]; __syncthreads(); out[threadIdx.x] = tile[7 - threadIdx.x]; }";

#[test]
fn a_shared_tile_and_a_barrier_reverse_a_block() {
    let k = launched(REVERSE, 8, 1).unwrap();
    let x = [10.0f32, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0];
    let mut args = vec![f32s(&[0.0; 8]), f32s(&x)];
    run_launch(&k, &mut args).expect("runs");
    let want: Vec<f64> = x.iter().rev().map(|&v| f64::from(v)).collect();
    assert_eq!(out_of(&args), want);
}

#[test]
fn without_the_barrier_the_oracle_sees_the_race() {
    // Control: the same kernel minus `__syncthreads()` reads a tile slots other threads have not written yet, which the oracle traps as an uninitialized shared read. So the
    // barrier in the lifted kernel is what makes the run above legal.
    let src = REVERSE.replace("__syncthreads(); ", "");
    let k = launched(&src, 8, 1).unwrap();
    let mut args = vec![f32s(&[0.0; 8]), f32s(&[1.0; 8])];
    let r = run_launch(&k, &mut args);
    assert!(
        matches!(r, Err(KernelTrap::UninitializedShared { .. })),
        "{r:?}"
    );
}

fn declined(src: &str, launch: bool) -> KernelLiftError {
    let r = if launch {
        launched(src, 8, 1)
    } else {
        lift_kernel(&C, src, "k")
    };
    r.expect_err("must decline")
}

fn not_yet(src: &str, launch: bool, needle: &str) {
    match declined(src, launch) {
        KernelLiftError::NotYetLifted { construct, .. } => {
            assert!(construct.contains(needle), "`{construct}` lacks `{needle}`");
        }
        other => panic!("expected NotYetLifted({needle}), got {other:?}"),
    }
}

#[test]
fn every_builtin_needs_the_callers_launch() {
    not_yet(
        "void k(float* out) { out[threadIdx.x] = 1.0f; }",
        false,
        "without a launch",
    );
    not_yet(
        "void k(float* out) { __syncthreads(); out[0] = 1.0f; }",
        false,
        "without a launch",
    );
    not_yet(
        "void k(float* out) { __shared__ float t[4]; out[0] = 1.0f; }",
        false,
        "without a launch",
    );
    // Control: the same sources lift with a launch.
    assert!(launched("void k(float* out) { out[threadIdx.x] = 1.0f; }", 8, 1).is_ok());
    assert!(
        launched(
            "void k(float* out) { __syncthreads(); out[0] = 1.0f; }",
            8,
            1
        )
        .is_ok()
    );
}

#[test]
fn the_rest_of_the_cuda_surface_declines_by_name() {
    not_yet(
        "void k(float* out) { out[threadIdx.w] = 1.0f; }",
        true,
        "member `w`",
    );
    not_yet(
        "void k(float* out) { out[warpSize] = 1.0f; }",
        true,
        "warpSize",
    );
    not_yet(
        "void k(float* out, int n) { __shared__ float t[n]; out[0] = 1.0f; }",
        true,
        "literal",
    );
    not_yet(
        "void k(float* out) { extern __shared__ float t[]; out[0] = 1.0f; }",
        true,
        "storage class",
    );
    not_yet(
        "void k(float* out) { __syncwarp(); out[0] = 1.0f; }",
        true,
        "`__syncwarp`",
    );
    not_yet(
        "void k(float* out) { __shared__ float t[4]; out[0] = t; }",
        true,
        "array or pointer",
    );
}

#[test]
fn all_three_axes_map_to_their_own_ids() {
    // A distinct weight per id, so a swapped axis changes the sum: thread (x,y,z) of block (bx,by) in a 2x3x2 block, 3x2 grid.
    let src = "void k(float* out) { int t = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z); \
        int b = blockIdx.x + gridDim.x * blockIdx.y; \
        out[b * 12 + t] = (float)(threadIdx.x + 10 * threadIdx.y + 100 * threadIdx.z + 1000 * blockIdx.x + 10000 * blockIdx.y + 7 * gridDim.y + 70 * blockDim.z); }";
    let k = lift_kernel_launched(&C, src, "k", LaunchShape::new([2, 3, 2], [3, 2, 1])).unwrap();
    let mut args = vec![f32s(&[0.0; 72])];
    run_launch(&k, &mut args).expect("runs");
    let mut want = vec![0.0f64; 72];
    for by in 0..2u32 {
        for bx in 0..3u32 {
            for tz in 0..2u32 {
                for ty in 0..3u32 {
                    for tx in 0..2u32 {
                        let t = tx + 2 * (ty + 3 * tz);
                        let b = bx + 3 * by;
                        want[(b * 12 + t) as usize] = f64::from(
                            tx + 10 * ty + 100 * tz + 1000 * bx + 10000 * by + 7 * 2 + 70 * 2,
                        );
                    }
                }
            }
        }
    }
    assert_eq!(out_of(&args), want);
}

#[test]
fn a_shared_array_has_exactly_the_declared_size() {
    // `tile[8]` is declared with 8 elements: index 7 is legal, index 8 is out of range. A lifted size that is off by one in
    // either direction changes one of the two runs.
    let src = |i: u32| {
        format!(
            "void k(float* out) {{ __shared__ float tile[8]; tile[{i}] = 1.0f; out[0] = tile[{i}]; }}"
        )
    };
    let ok = launched(&src(7), 1, 1).unwrap();
    let mut args = vec![f32s(&[0.0])];
    run_launch(&ok, &mut args).expect("index 7 is in range");
    assert_eq!(out_of(&args), vec![1.0]);
    let bad = launched(&src(8), 1, 1).unwrap();
    let mut args = vec![f32s(&[0.0])];
    let r = run_launch(&bad, &mut args);
    assert!(r.is_err(), "index 8 of 8 must trap, got {r:?}");
}
