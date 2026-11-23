// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Census wall: a half / bfloat16 value through a warp collective** (12 real instantiations stopped at the checker's
//! `CollectiveUnsupported`: collective values are integers, F32 and F64). A shuffle only moves a value, so a half travels as an F32
//! (widening and narrowing back are exact) and the IR is unchanged. `cg::reduce` with `max`/`min` picks one operand, so it is
//! exact the same way; with `plus` C rounds to half at every step and one F32 sum would not, so it declines by name. Oracle runs
//! against Rust references over exactly representable halves.

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

fn lift(src: &str) -> Result<KernelFn, KernelLiftError> {
    lift_kernel_launched(
        &C,
        src,
        "k",
        LaunchShape::new([32, 1, 1], [1, 1, 1]).with_subgroup(32, 32),
    )
}

/// The f16 bits of a small non-negative integer (exact below 2048).
fn half_bits(v: u32) -> u16 {
    if v == 0 {
        return 0;
    }
    let e = 31 - v.leading_zeros();
    let mant = ((v << 10) >> e) & 0x3ff;
    (((e + 15) << 10) | mant) as u16
}

/// Lane `i` holds the half `(i * 5 + 3) % 29`.
fn lane_value(i: usize) -> u32 {
    ((i * 5 + 3) % 29) as u32
}

fn run(k: &KernelFn) -> Vec<f64> {
    let bits: Vec<u16> = (0..32).map(|i| half_bits(lane_value(i))).collect();
    let mut args = vec![
        KernelArg::Buffer(TypedBuffer::from_f32(&[32], &[0.0; 32])),
        KernelArg::Buffer(TypedBuffer::from_f16_bits(&[32], &bits)),
    ];
    run_launch_with_subgroup(k, &mut args, Some(32)).expect("runs");
    let KernelArg::Buffer(o) = &args[0] else {
        panic!()
    };
    o.to_f64_vec()
}

#[test]
fn a_half_moves_through_every_shuffle_flavour_unchanged() {
    let k = lift(
        "void k(float* out, const half* in) { half v = in[threadIdx.x]; \
         half x = __shfl_xor_sync(0xffffffff, v, 1); half d = __shfl_down_sync(0xffffffff, v, 2, 32); \
         half i0 = __shfl_sync(0xffffffff, v, 7); \
         out[threadIdx.x] = __half2float(x) * 10000.0f + __half2float(d) * 100.0f + __half2float(i0); }",
    )
    .expect("lifts (was Invalid(CollectiveUnsupported))");
    let want: Vec<f64> = (0..32)
        .map(|i| {
            let x = f64::from(lane_value(i ^ 1));
            // a lane whose source is past the end keeps its own value
            let d = f64::from(lane_value(if i + 2 < 32 { i + 2 } else { i }));
            x * 10000.0 + d * 100.0 + f64::from(lane_value(7))
        })
        .collect();
    assert_eq!(run(&k), want);
}

#[test]
fn a_cooperative_groups_half_max_and_min_are_exact_and_a_half_sum_declines() {
    let tile = "cg::thread_block block = cg::this_thread_block(); cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);";
    let k = lift(&format!(
        "void k(float* out, const half* in) {{ {tile} half v = in[threadIdx.x]; \
         half hi = cg::reduce(warp, v, cg::greater<half>()); half lo = cg::reduce(warp, v, cg::less<half>()); \
         out[threadIdx.x] = __half2float(hi) * 100.0f + __half2float(lo); }}"
    ))
    .expect("lifts");
    let hi = (0..32).map(lane_value).max().unwrap();
    let lo = (0..32).map(lane_value).min().unwrap();
    assert_eq!(run(&k), vec![f64::from(hi * 100 + lo); 32]);
    match lift(&format!(
        "void k(float* out, const half* in) {{ {tile} half s = cg::reduce(warp, in[threadIdx.x], cg::plus<half>()); out[threadIdx.x] = __half2float(s); }}"
    )) {
        Err(KernelLiftError::NotYetLifted { construct, .. }) => {
            assert!(construct.contains("over a half value"), "{construct}");
        }
        other => panic!("expected a decline, got {other:?}"),
    }
}

#[test]
fn a_bfloat16_shuffle_moves_the_rounded_value() {
    // bfloat16 holds 8 significand bits: 257 rounds to 256 before it moves, and the shuffle must not change what it holds
    let k = lift(
        "void k(float* out, const float* in) { __nv_bfloat16 v = __float2bfloat16(in[threadIdx.x]); \
         __nv_bfloat16 x = __shfl_xor_sync(0xffffffff, v, 1); out[threadIdx.x] = __bfloat162float(x); }",
    )
    .expect("lifts");
    let input: Vec<f32> = (0..32).map(|i| 257.0 + 2.0 * i as f32).collect();
    let mut args = vec![
        KernelArg::Buffer(TypedBuffer::from_f32(&[32], &[0.0; 32])),
        KernelArg::Buffer(TypedBuffer::from_f32(&[32], &input)),
    ];
    run_launch_with_subgroup(&k, &mut args, Some(32)).expect("runs");
    let KernelArg::Buffer(o) = &args[0] else {
        panic!()
    };
    // the bfloat16 rounding of a value v >= 256: the nearest multiple of 2 (ties to even) below 512, of 4 from 512
    let bf = |v: f32| {
        f64::from(f32::from_bits(
            (v.to_bits() + 0x7fff + ((v.to_bits() >> 16) & 1)) & 0xffff_0000,
        ))
    };
    let want: Vec<f64> = (0..32).map(|i| bf(input[i ^ 1])).collect();
    assert_eq!(o.to_f64_vec(), want);
}
