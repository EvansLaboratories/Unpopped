// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Census wall: the CUDA conversion intrinsics** (`__float2int_rn`, `__double2int_rn`, `__int2float_rz`, ...; 14 Baracuda
//! kernels stopped at one). Each is one IR `Cast` with the rounding its suffix names; float to integer clamps to the integer's
//! range and maps NaN to 0. Oracle runs against Rust references.

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

fn run(src: &str, input: &[f32], n: usize) -> Vec<f64> {
    let k = lift(src).unwrap();
    let mut args = vec![
        KernelArg::Buffer(TypedBuffer::from_f32(&[n as i64], &vec![0.0; n])),
        KernelArg::Buffer(TypedBuffer::from_f32(&[input.len() as i64], input)),
    ];
    run_launch(&k, &mut args).expect("runs");
    let KernelArg::Buffer(o) = &args[0] else {
        panic!()
    };
    o.to_f64_vec()
}

#[test]
fn each_rounding_suffix_rounds_its_way() {
    let src = "void k(float* out, const float* in) { \
        out[0] = (float)__float2int_rn(in[0]); out[1] = (float)__float2int_rn(in[1]); out[2] = (float)__float2int_rn(in[2]); \
        out[3] = (float)__float2int_rz(in[3]); out[4] = (float)__float2int_rz(in[4]); \
        out[5] = (float)__float2int_rd(in[5]); out[6] = (float)__float2int_rd(in[6]); \
        out[7] = (float)__float2int_ru(in[7]); out[8] = (float)__float2int_ru(in[8]); }";
    // 2.5 -> 2 and 3.5 -> 4 (ties to even), -2.5 -> -2; truncation; floor; ceiling
    let input = [2.5, 3.5, -2.5, 2.7, -2.7, -2.1, 2.9, 2.1, -2.9];
    assert_eq!(
        run(src, &input, 9),
        vec![2.0, 4.0, -2.0, 2.0, -2.0, -3.0, 2.0, 3.0, -2.0]
    );
}

#[test]
fn float_to_integer_clamps_and_maps_nan_to_zero() {
    let src = "void k(float* out, const float* in) { \
        out[0] = (float)(__float2int_rn(in[0]) == 2147483647); out[1] = (float)(__float2int_rn(in[1]) == -2147483647 - 1); \
        out[2] = (float)__float2int_rn(in[2]); out[3] = (float)__float2uint_rz(in[3]); \
        out[4] = (float)(__float2ll_rz(in[4]) == 10000000000ll); out[5] = (float)(__double2int_rn((double)in[5]) == 2); }";
    let input = [3.0e9, -3.0e9, f32::NAN, -5.0, 1.0e10, 2.5];
    assert_eq!(run(src, &input, 6), vec![1.0, 1.0, 0.0, 0.0, 1.0, 1.0]);
}

#[test]
fn integer_to_float_rounds_the_inexact_values() {
    // 16777217 = 2^24 + 1 is not an f32: nearest-even gives 2^24, round-up gives 2^24 + 2
    let src = "void k(float* out, const float* in) { \
        out[0] = __int2float_rn(16777217) - 16777216.0f; out[1] = __int2float_ru(16777217) - 16777216.0f; \
        out[2] = __int2float_rz(16777217) - 16777216.0f; out[3] = __uint2float_rn(7u); }";
    assert_eq!(run(src, &[0.0], 4), vec![0.0, 2.0, 0.0, 7.0]);
}

#[test]
fn what_is_not_a_conversion_still_declines() {
    // an unknown suffix, two float sides, and a wrong argument count
    for call in [
        "__float2int_rx(in[0])",
        "__float2double_rn(in[0])",
        "__float2int_rn(in[0], in[1])",
    ] {
        let src = format!("void k(float* out, const float* in) {{ out[0] = (float){call}; }}");
        assert!(lift(&src).is_err(), "{call}");
    }
}
