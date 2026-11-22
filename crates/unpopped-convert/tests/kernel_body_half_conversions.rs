// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Census wall: `__half2float` and friends** (`__half2float`, `__bfloat162float`, `__float2half`, `__float2bfloat16`; 9+ Baracuda
//! kernels stopped at one). Widening is exact; narrowing rounds to nearest even and overflows to infinity. Oracle runs against
//! Rust references (the `half` crate's rounding is the reference for f16: here, hand-computed bit patterns).

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

fn run(src: &str, input: &[f32], half_bits: &[u16], n: usize) -> Vec<f64> {
    let k = lift(src).unwrap();
    let mut args = vec![
        KernelArg::Buffer(TypedBuffer::from_f32(&[n as i64], &vec![0.0; n])),
        KernelArg::Buffer(TypedBuffer::from_f32(&[input.len() as i64], input)),
        KernelArg::Buffer(TypedBuffer::from_f16_bits(&[half_bits.len() as i64], half_bits)),
    ];
    run_launch(&k, &mut args).expect("runs");
    let KernelArg::Buffer(o) = &args[0] else {
        panic!()
    };
    o.to_f64_vec()
}

#[test]
fn widening_a_half_is_exact() {
    // 0x3e00 = 1.5, 0x2e66 = 0.0999755859375, 0xc000 = -2, 0x7bff = 65504 (the largest finite half)
    let src = "void k(float* out, const float* in, const half* h) { \
        out[0] = __half2float(h[0]); out[1] = __half2float(h[1]); out[2] = __half2float(h[2]); out[3] = __half2float(h[3]); }";
    let got = run(src, &[0.0], &[0x3e00, 0x2e66, 0xc000, 0x7bff], 4);
    assert_eq!(got, vec![1.5, 0.099_975_585_937_5, -2.0, 65504.0]);
}

#[test]
fn narrowing_to_half_rounds_to_nearest_even_and_overflows_to_infinity() {
    let src = "void k(float* out, const float* in, const half* h) { \
        for (int i = 0; i < 6; i++) { out[i] = __half2float(__float2half_rn(in[i])); } }";
    // 0.1 -> 0.0999755859375; 1 + 2^-11 is a tie between 1 and 1 + 2^-10 -> 1 (even); 1 + 3 * 2^-11 is a tie -> 1 + 2^-9;
    // 70000 -> inf; -70000 -> -inf; 2^-26 underflows to 0
    let input = [
        0.1,
        1.0 + 2f32.powi(-11),
        1.0 + 3.0 * 2f32.powi(-11),
        70000.0,
        -70000.0,
        2f32.powi(-26),
    ];
    let got = run(src, &input, &[0], 6);
    assert_eq!(
        got,
        vec![
            0.099_975_585_937_5,
            1.0,
            1.0 + 2f64.powi(-9),
            f64::INFINITY,
            f64::NEG_INFINITY,
            0.0
        ]
    );
}

#[test]
fn bfloat16_rounds_to_nearest_even_with_an_eight_bit_significand() {
    let src = "void k(float* out, const float* in, const half* h) { \
        for (int i = 0; i < 3; i++) { out[i] = __bfloat162float(__float2bfloat16(in[i])); } }";
    // 1 + 2^-8 is a tie between 1 and 1 + 2^-7 -> 1; pi -> 3.140625; 1 + 3 * 2^-8 is a tie -> 1 + 2^-6
    let got = run(
        src,
        &[1.0 + 2f32.powi(-8), std::f32::consts::PI, 1.0 + 3.0 * 2f32.powi(-8)],
        &[0],
        3,
    );
    assert_eq!(got, vec![1.0, 3.140_625, 1.0 + 2f64.powi(-6)]);
}

#[test]
fn the_wrong_argument_type_or_count_declines() {
    // a float where `__half2float` takes a half, and `__bfloat162float` of a half
    for call in [
        "__half2float(in[0])",
        "__bfloat162float(h[0])",
        "__half2float(h[0], h[1])",
    ] {
        let src = format!("void k(float* out, const float* in, const half* h) {{ out[0] = {call}; }}");
        assert!(
            matches!(lift(&src), Err(KernelLiftError::NotYetLifted { .. })),
            "{call}"
        );
    }
}
