// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Census wall: warp tiles and `cg::reduce`** (14 + 8 Baracuda kernels). `cg::thread_block_tile<32> w =
//! cg::tiled_partition<32>(block)` is a name for the launch's subgroup (the launch must name exactly that width), `w.sync()` a
//! subgroup barrier, `w.thread_rank()` the lane id, `w.shfl*` a shuffle collective and `cg::reduce(w, v, cg::plus<float>())` a
//! reduce collective: the IR's existing `Stmt::Collective`, no widening. Oracle runs against Rust references.

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

fn lift(src: &str, block: u32, sub: Option<u32>) -> Result<KernelFn, KernelLiftError> {
    let mut l = LaunchShape::new([block, 1, 1], [1, 1, 1]);
    if let Some(w) = sub {
        l = l.with_subgroup(w, w);
    }
    lift_kernel_launched(&C, src, "k", l)
}

fn run(k: &KernelFn, input: &[f32], n_out: usize) -> Vec<f64> {
    let mut args = vec![
        KernelArg::Buffer(TypedBuffer::from_f32(&[n_out as i64], &vec![0.0; n_out])),
        KernelArg::Buffer(TypedBuffer::from_f32(&[input.len() as i64], input)),
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

const TILE: &str = "cg::thread_block block = cg::this_thread_block(); \
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);";

#[test]
fn a_warp_reduce_gives_every_lane_the_sum_of_its_warp() {
    // 64 threads = two warps; each lane writes its warp's sum
    let src = format!(
        "void k(float* out, const float* in) {{ {TILE} \
         float s = cg::reduce(warp, in[threadIdx.x], cg::plus<float>()); out[threadIdx.x] = s + (float)warp.thread_rank(); }}"
    );
    let k = lift(&src, 64, Some(32)).unwrap();
    let input: Vec<f32> = (0..64).map(|i| (i % 7) as f32 + 0.5).collect();
    let want: Vec<f64> = (0..64)
        .map(|i| {
            let w = i / 32;
            let s: f32 = input[w * 32..w * 32 + 32].iter().sum();
            f64::from(s) + (i % 32) as f64
        })
        .collect();
    assert_eq!(run(&k, &input, 64), want);
}

#[test]
fn max_and_min_reductions_and_the_auto_spelling() {
    let src = "void k(float* out, const float* in) { auto b = cg::this_thread_block(); auto w = cg::tiled_partition<32>(b); \
        float hi = cg::reduce(w, in[threadIdx.x], cg::greater<float>()); float lo = cg::reduce(w, in[threadIdx.x], cg::less<float>()); \
        out[threadIdx.x] = hi - lo; }";
    let k = lift(src, 32, Some(32)).unwrap();
    let input: Vec<f32> = (0..32).map(|i| ((i * 5) % 11) as f32).collect();
    let spread = f64::from(input.iter().copied().fold(f32::MIN, f32::max))
        - f64::from(input.iter().copied().fold(f32::MAX, f32::min));
    assert_eq!(run(&k, &input, 32), vec![spread; 32]);
}

#[test]
fn shuffles_and_the_tile_barrier() {
    // shfl_down by 1: lane i reads lane i + 1 (the top lane keeps its own value)
    let src = format!(
        "void k(float* out, const float* in) {{ {TILE} float v = in[threadIdx.x]; warp.sync(); \
         float d = warp.shfl_down(v, 1); float x = warp.shfl_xor(v, 2); out[threadIdx.x] = d * 100.0f + x + (float)warp.size(); }}"
    );
    let k = lift(&src, 32, Some(32)).unwrap();
    let input: Vec<f32> = (0..32).map(|i| i as f32).collect();
    let want: Vec<f64> = (0..32usize)
        .map(|i| {
            let d = if i + 1 < 32 { i + 1 } else { i };
            (d as f64) * 100.0 + ((i ^ 2) as f64) + 32.0
        })
        .collect();
    assert_eq!(run(&k, &input, 32), want);
}

#[test]
fn a_tile_needs_a_launch_that_names_that_subgroup() {
    let src = format!("void k(float* out) {{ {TILE} out[0] = 1.0f; }}");
    not_yet(lift(&src, 32, None), "subgroup is exactly 32");
    // a subgroup of another width is a different tile
    let mut l = LaunchShape::new([32, 1, 1], [1, 1, 1]).with_subgroup(16, 16);
    l.grid = [1, 1, 1];
    not_yet(
        lift_kernel_launched(&C, &src, "k", l),
        "subgroup is exactly 32",
    );
}

#[test]
fn what_the_tile_subset_does_not_cover_declines_by_name() {
    let reduce_block = "void k(float* out) { auto b = cg::this_thread_block(); out[0] = cg::reduce(b, 1.0f, cg::plus<float>()); }";
    not_yet(lift(reduce_block, 32, Some(32)), "not a warp tile");
    let op = format!(
        "void k(float* out, const float* in) {{ {TILE} out[0] = cg::reduce(warp, in[0], cg::bit_and<int>()); }}"
    );
    not_yet(lift(&op, 32, Some(32)), "bit_and");
    let method = format!("void k(float* out) {{ {TILE} out[0] = (float)warp.mystery(); }}");
    not_yet(lift(&method, 32, Some(32)), "mystery");
    let tile16 = "void k(float* out) { auto b = cg::this_thread_block(); cg::thread_block_tile<16> w = cg::tiled_partition<16>(b); out[0] = 1.0f; }";
    not_yet(lift(tile16, 32, Some(32)), "subgroup is exactly 16");
}
