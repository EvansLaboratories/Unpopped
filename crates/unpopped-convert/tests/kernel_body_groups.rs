// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Census wall: `cg::thread_block`** (13 Baracuda kernels). The block-wide subset of cooperative groups: the handle, `sync()`,
//! `thread_rank()` and `size()`. Each runs on the oracle against a Rust reference; a block barrier is observable because the
//! threads read what their neighbours wrote before it.

use unpopped::exec::LaunchShape;
use unpopped::kernel::KernelFn;
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_convert::{
    Frontend, KernelLiftError, LiftContext, lift_kernel_in, lift_kernel_launched,
};

fn parse(src: &str) -> Option<tree_sitter::Tree> {
    let b = src.replace("__shared__", &" ".repeat("__shared__".len()));
    let mut p = tree_sitter::Parser::new();
    p.set_language(&tree_sitter_cpp::LANGUAGE.into()).ok()?;
    p.parse(b, None)
}

const C: Frontend = Frontend::new("c", "void", &[], "out", "in", parse);

fn launched(src: &str, block: [u32; 3]) -> Result<KernelFn, KernelLiftError> {
    lift_kernel_launched(&C, src, "k", LaunchShape::new(block, [1, 1, 1]))
}

fn run(k: &KernelFn, n: usize) -> Vec<f64> {
    let mut args = vec![KernelArg::Buffer(TypedBuffer::from_f32(
        &[n as i64],
        &vec![0.0; n],
    ))];
    run_launch(k, &mut args).expect("runs");
    let KernelArg::Buffer(o) = &args[0] else {
        panic!()
    };
    o.to_f64_vec()
}

#[test]
fn sync_is_a_block_barrier_and_thread_rank_is_the_linear_index() {
    // each thread writes its rank to shared, the barrier, then reads its right neighbour's slot: wrong without the barrier
    let src = "void k(float* out) { extern __shared__ float s[]; cg::thread_block block = cg::this_thread_block(); \
        s[block.thread_rank()] = (float)block.thread_rank() + 10.0f; block.sync(); \
        out[block.thread_rank()] = s[(block.thread_rank() + 1) % 4]; }";
    let ctx =
        LiftContext::new(LaunchShape::new([4, 1, 1], [1, 1, 1])).with_dynamic_shared_bytes(16);
    let k = lift_kernel_in(&C, src, "k", &ctx).unwrap();
    assert_eq!(run(&k, 4), vec![11.0, 12.0, 13.0, 10.0]);
}

#[test]
fn auto_declaration_and_a_two_dimensional_block() {
    let src = "void k(float* out) { auto b = cg::this_thread_block(); \
        out[b.thread_rank()] = (float)(b.size() * 100u + b.thread_rank()); }";
    let k = launched(src, [2, 3, 1]).unwrap();
    assert_eq!(run(&k, 6), vec![600.0, 601.0, 602.0, 603.0, 604.0, 605.0]);
}

#[test]
fn what_is_not_the_block_subset_declines_by_name() {
    let tile = "void k(float* out) { cg::thread_block b = cg::this_thread_block(); \
        cg::thread_block_tile<32> w = cg::tiled_partition<32>(b); out[0] = 1.0f; }";
    assert!(launched(tile, [32, 1, 1]).is_err());
    let method =
        "void k(float* out) { auto b = cg::this_thread_block(); b.mystery(); out[0] = 1.0f; }";
    match launched(method, [1, 1, 1]) {
        Err(KernelLiftError::NotYetLifted { construct, .. }) => {
            assert!(construct.contains("mystery"), "{construct}");
        }
        other => panic!("{other:?}"),
    }
    // control: the same group with a supported call lifts
    assert!(
        launched(
            "void k(float* out) { auto b = cg::this_thread_block(); b.sync(); out[0] = 1.0f; }",
            [1, 1, 1]
        )
        .is_ok()
    );
}
