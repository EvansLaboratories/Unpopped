// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Census wall: a halving loop over a launch-derived constant** (`const int block_size = blockDim.x; for (int stride =
//! block_size / 2; stride > 0; stride >>= 1)`, 11 Baracuda kernels). The loop is not the counted form, so it is unrolled over
//! its induction variable; its start and bound only needed to be known at lift time, and a `const` integer local initialised from
//! the launch's block extent (or a constant) is. The IR is the unrolled body: the launch it was lifted for is the one it is
//! valid for. Oracle runs against Rust references.

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

fn lift(src: &str, block: [u32; 3], shared: u32) -> Result<KernelFn, KernelLiftError> {
    let ctx = LiftContext::new(LaunchShape::new(block, [1, 1, 1])).with_dynamic_shared_bytes(shared);
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

const TREE: &str = "void k(float* out, const float* in) { extern __shared__ float s[]; const int bs = blockDim.x; \
    s[threadIdx.x] = in[threadIdx.x]; __syncthreads(); \
    for (int stride = bs / 2; stride > 0; stride >>= 1) { if (threadIdx.x < stride) { s[threadIdx.x] += s[threadIdx.x + stride]; } __syncthreads(); } \
    if (threadIdx.x == 0) { out[0] = s[0]; } }";

#[test]
fn a_tree_reduction_over_the_block_extent_sums_the_block() {
    for block in [8u32, 16] {
        let k = lift(TREE, [block, 1, 1], block * 4).unwrap();
        let input: Vec<f32> = (0..block).map(|i| (i * i % 7) as f32 + 1.0).collect();
        let want: f32 = input.iter().sum();
        assert_eq!(run(&k, &input, 1), vec![f64::from(want)], "block {block}");
    }
}

#[test]
fn the_trip_count_follows_the_launch_axis_it_names() {
    // one `acc += 1` per halving trip: blockDim.y = 8 gives 3 trips (8, 4, 2), blockDim.x = 2 gives 1
    let src = |axis: &str| {
        format!(
            "void k(float* out, const float* in) {{ float acc = 0.0f; \
            for (int st = blockDim.{axis}; st > 1; st >>= 1) {{ acc += 1.0f; }} if (threadIdx.x == 0 && threadIdx.y == 0) {{ out[0] = acc; }} }}"
        )
    };
    let k = |axis: &str| lift(&src(axis), [2, 8, 1], 0).unwrap();
    assert_eq!(run(&k("y"), &[0.0], 1), vec![3.0]);
    assert_eq!(run(&k("x"), &[0.0], 1), vec![1.0]);
}

#[test]
fn a_local_that_is_not_known_at_lift_time_still_declines() {
    fn declines(r: Result<KernelFn, KernelLiftError>) {
        match r {
            Err(KernelLiftError::NotYetLifted { construct, .. }) => {
                assert!(construct.contains("not a counted loop"), "{construct}");
            }
            other => panic!("expected the counted-loop decline, got {other:?}"),
        }
    }
    // not `const`: it may have changed before the loop
    declines(lift(
        "void k(float* out) { int bs = blockDim.x; bs = 4; for (int st = bs / 2; st > 0; st >>= 1) { out[0] = 1.0f; } }",
        [8, 1, 1],
        0,
    ));
    // `const`, but from a runtime value
    declines(lift(
        "void k(float* out, int n) { const int bs = n; for (int st = bs / 2; st > 0; st >>= 1) { out[0] = 1.0f; } }",
        [8, 1, 1],
        0,
    ));
    // control: the same loop from a `const` of the launch lifts
    assert!(
        lift(
            "void k(float* out) { const int bs = blockDim.x; for (int st = bs / 2; st > 0; st >>= 1) { out[0] = 1.0f; } }",
            [8, 1, 1],
            0
        )
        .is_ok()
    );
}
