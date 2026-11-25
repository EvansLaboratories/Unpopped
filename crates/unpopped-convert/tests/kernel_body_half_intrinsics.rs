// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Census wall: the scalar half intrinsics** (`__habs` 4, `__hisnan` 2 real instantiations; `__hadd`, `__hsub`, `__hmul` and the
//! comparisons `__heq`... come with them). They work at f32 (widening a half is exact) and narrow the result back; for `+ - *` the f32
//! result rounded to half is the half operation's (24 >= 2 * 11 + 2: no double-rounding difference), shown below on a tie that
//! must round to even. `__hfma` is not lifted (an f32 fma then a narrowing is not it). Oracle runs against hand-computed half bit
//! patterns.

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

fn run(k: &KernelFn, bits: &[u16], outs: usize) -> Vec<f64> {
    let mut args = vec![
        KernelArg::Buffer(TypedBuffer::from_f32(&[outs as i64], &vec![0.0; outs])),
        KernelArg::Buffer(TypedBuffer::from_f16_bits(&[bits.len() as i64], bits)),
    ];
    run_launch(k, &mut args).expect("runs");
    let KernelArg::Buffer(o) = &args[0] else {
        panic!()
    };
    o.to_f64_vec()
}

// half bit patterns: 1.0 = 0x3c00, 1 + 2^-10 = 0x3c01, 2048 = 0x6800, -3.5 = 0xc300, 2.0 = 0x4000, NaN = 0x7e00
const ONE: u16 = 0x3c00;
const ONE_EPS: u16 = 0x3c01;
const TWO_K: u16 = 0x6800;
const NEG_3_5: u16 = 0xc300;
const TWO: u16 = 0x4000;
const NAN: u16 = 0x7e00;

#[test]
fn abs_and_isnan() {
    let k = lift(
        "void k(float* out, const half* in) { half a = in[0]; half n = in[1]; \
         out[0] = __half2float(__habs(a)); out[1] = (float)__hisnan(n); out[2] = (float)__hisnan(a); }",
    )
    .expect("lifts (was a decline)");
    assert_eq!(run(&k, &[NEG_3_5, NAN], 3), vec![3.5, 1.0, 0.0]);
}

#[test]
fn add_sub_mul_round_to_half_and_a_tie_goes_to_even() {
    let k = lift(
        "void k(float* out, const half* in) { half a = in[0]; half b = in[1]; half c = in[2]; \
         out[0] = __half2float(__hadd(a, b)); out[1] = __half2float(__hsub(a, b)); \
         out[2] = __half2float(__hmul(b, b)); out[3] = __half2float(__hmul(c, c)); }",
    )
    .expect("lifts");
    // a = 2048, b = 1, c = 1 + 2^-10
    let r = run(&k, &[TWO_K, ONE, ONE_EPS], 4);
    // 2048 + 1 = 2049 is the tie between 2048 and 2050 (the spacing at 2048 is 2): to even, 2048; 2048 - 1 = 2047 is exact
    assert_eq!(r[0], 2048.0);
    assert_eq!(r[1], 2047.0);
    assert_eq!(r[2], 1.0);
    // (1 + 2^-10)^2 = 1 + 2^-9 + 2^-20 rounds to 1 + 2^-9
    assert_eq!(r[3], 1.0 + 2f64.powi(-9));
}

#[test]
fn comparisons_and_what_is_declined() {
    let k = lift(
        "void k(float* out, const half* in) { half a = in[0]; half b = in[1]; \
         out[0] = (float)__hlt(a, b); out[1] = (float)__hgt(a, b); out[2] = (float)__heq(a, a); out[3] = (float)__hne(a, b); \
         out[4] = (float)__hle(a, b); out[5] = (float)__hge(a, b); \n         out[6] = (float)__hlt(a, a); out[7] = (float)__hgt(a, a); out[8] = (float)__hle(a, a); out[9] = (float)__hge(a, a); }",
    )
    .expect("lifts");
    // the last four compare a value with itself: strict is false, non-strict is true
    assert_eq!(
        run(&k, &[ONE, TWO], 10),
        vec![1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0]
    );
    // an f32 argument is not a half; `__hfma` is not lifted
    for src in [
        "void k(float* out, const float* in) { out[0] = __habs(in[0]); }",
        "void k(float* out, const half* in) { half a = in[0]; out[0] = __half2float(__hfma(a, a, a)); }",
    ] {
        assert!(
            matches!(lift(src), Err(KernelLiftError::NotYetLifted { .. })),
            "{src}"
        );
    }
}
