// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Census wall: the `__shfl_*_sync` intrinsics** (`partial += __shfl_xor_sync(0xffffffffu, partial, offset)`, 7 Baracuda
//! kernels and the warp-reduce idiom everywhere). A full-mask, 32-wide shuffle is the IR's subgroup `Collective::Shuffle`, the
//! same one a `thread_block_tile` shuffle lifts to. Oracle runs against Rust references.

use unpopped::exec::LaunchShape;
use unpopped::kernel::KernelFn;
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch_with_subgroup;
use unpopped_convert::{Frontend, KernelLiftError, lift_kernel_launched};

fn parse(src: &str) -> Option<tree_sitter::Tree> {
    let mut p = tree_sitter::Parser::new();
    p.set_language(&tree_sitter_cpp::LANGUAGE.into()).ok()?;
    p.parse(src, None)
}

const C: Frontend = Frontend::new("c", "void", &[], "out", "in", parse);

fn lift(src: &str, sub: Option<u32>) -> Result<KernelFn, KernelLiftError> {
    let mut l = LaunchShape::new([32, 1, 1], [1, 1, 1]);
    if let Some(w) = sub {
        l = l.with_subgroup(w, w);
    }
    lift_kernel_launched(&C, src, "k", l)
}

fn run(k: &KernelFn, input: &[f32]) -> Vec<f64> {
    let mut args = vec![
        KernelArg::Buffer(TypedBuffer::from_f32(&[32], &[0.0; 32])),
        KernelArg::Buffer(TypedBuffer::from_f32(&[32], input)),
    ];
    run_launch_with_subgroup(k, &mut args, Some(32)).expect("runs");
    let KernelArg::Buffer(o) = &args[0] else {
        panic!()
    };
    o.to_f64_vec()
}

fn not_yet(r: Result<KernelFn, KernelLiftError>, needle: &str) {
    match r {
        Err(KernelLiftError::NotYetLifted { construct, .. }) => {
            assert!(construct.contains(needle), "`{construct}` lacks `{needle}`");
        }
        other => panic!("expected NotYetLifted({needle}), got {other:?}"),
    }
}

fn input() -> Vec<f32> {
    (0..32).map(|i| ((i * 7) % 13) as f32 + 0.5).collect()
}

#[test]
fn a_butterfly_warp_reduce_gives_every_lane_the_sum() {
    let src = "void k(float* out, const float* in) { float partial = in[threadIdx.x]; \
        for (int offset = 16; offset > 0; offset >>= 1) { partial += __shfl_xor_sync(0xffffffffu, partial, offset); } out[threadIdx.x] = partial; }";
    let k = lift(src, Some(32)).unwrap();
    let want: f32 = input().iter().sum();
    assert_eq!(run(&k, &input()), vec![f64::from(want); 32]);
}

#[test]
fn idx_up_and_down_read_the_lane_they_name() {
    let src = "void k(float* out, const float* in) { float v = in[threadIdx.x]; \
        float a = __shfl_sync(0xffffffffu, v, 3); float u = __shfl_up_sync(0xffffffffu, v, 2, 32); float d = __shfl_down_sync(0xffffffffu, v, 1); \
        out[threadIdx.x] = a * 10000.0f + u * 100.0f + d; }";
    let inp = input();
    let want: Vec<f64> = (0..32usize)
        .map(|i| {
            let u = if i >= 2 { i - 2 } else { i };
            let d = if i + 1 < 32 { i + 1 } else { i };
            f64::from(inp[3]) * 10000.0 + f64::from(inp[u]) * 100.0 + f64::from(inp[d])
        })
        .collect();
    assert_eq!(run(&lift(src, None).unwrap(), &inp), want);
}

#[test]
fn a_partial_mask_a_narrow_width_or_another_subgroup_declines() {
    let body = |call: &str| {
        format!("void k(float* out, const float* in) {{ out[threadIdx.x] = {call}; }}")
    };
    not_yet(
        lift(&body("__shfl_xor_sync(0x0000ffff, in[threadIdx.x], 1)"), Some(32)),
        "member mask",
    );
    not_yet(
        lift(&body("__shfl_down_sync(0xffffffffu, in[threadIdx.x], 1, 16)"), Some(32)),
        "width other than 32",
    );
    not_yet(
        lift(&body("__shfl_xor_sync(0xffffffffu, in[threadIdx.x], 1)"), Some(16)),
        "not exactly 32",
    );
    not_yet(
        lift(&body("__shfl_xor_sync(0xffffffffu, in[threadIdx.x])"), Some(32)),
        "2 argument",
    );
    // control: the full form lifts
    assert!(lift(&body("__shfl_xor_sync(0xffffffffu, in[threadIdx.x], 1)"), Some(32)).is_ok());
}
