// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Census wall: `fmodf`/`fmod`, `atan2`, `copysign`, `nextafter` and the bit-cast intrinsics** (`__float_as_int` 5, `__double_as_longlong` 5,
//! `fmodf` 9, `fmod` 3, `atan2` 2, `copysign` 2 real instantiations). The two-argument names map to the IR's exact-spelled
//! `RemTrunc` (sign of the dividend, exact), `Atan2`, `Copysign` and `Nextafter`, and the bit-casts to `ScalarExpr::Bitcast`. `hypot` and
//! `pow` have no exact spelling here and stay declined. Oracle runs against Rust references (`%` on floats is C's `fmod`).

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

const IN: [f32; 4] = [-5.5, 2.0, 0.75, -0.0];

fn run(k: &KernelFn, n: usize) -> Vec<f64> {
    let mut args = vec![
        KernelArg::Buffer(TypedBuffer::from_f32(&[n as i64], &vec![0.0; n])),
        KernelArg::Buffer(TypedBuffer::from_f32(&[4], &IN)),
    ];
    run_launch(k, &mut args).expect("runs");
    let KernelArg::Buffer(o) = &args[0] else {
        panic!()
    };
    o.to_f64_vec()
}

#[test]
fn fmod_takes_the_sign_of_the_dividend_and_copysign_and_nextafter_are_bit_exact() {
    let k = lift(
        "void k(float* out, const float* in) { \
         out[0] = fmodf(in[0], in[1]); out[1] = fmodf(5.5f, in[0]); out[2] = fmodf(in[2], 0.5f); \
         out[3] = copysignf(in[2], in[0]); out[4] = copysignf(in[1], in[3]); \
         out[5] = nextafterf(in[1], 3.0f) - in[1]; out[6] = in[1] - nextafterf(in[1], 0.0f); }",
    )
    .expect("lifts (was a decline)");
    let r = run(&k, 7);
    assert_eq!(r[0], f64::from(IN[0] % IN[1]));
    assert_eq!(r[0], -1.5, "sign of the dividend");
    assert_eq!(r[1], f64::from(5.5f32 % IN[0]));
    assert_eq!(r[2], f64::from(IN[2] % 0.5));
    assert_eq!(r[3], f64::from(IN[2].copysign(IN[0])));
    assert_eq!(r[4], -2.0, "the sign bit of -0.0 transfers");
    assert_eq!(r[5], f64::from(2.0f32.next_up() - 2.0));
    assert_eq!(r[6], f64::from(2.0 - 2.0f32.next_down()));
}

#[test]
fn atan2_lifts_in_both_spellings_within_the_contract_tolerance() {
    let k = lift("void k(float* out, const float* in) { out[0] = atan2f(in[1], in[2]); out[1] = atan2(in[0], in[1]); }")
        .expect("lifts");
    let r = run(&k, 2);
    for (got, want) in [(r[0], IN[1].atan2(IN[2])), (r[1], IN[0].atan2(IN[1]))] {
        let want = f64::from(want);
        assert!((got - want).abs() <= want.abs() * 1e-6, "{got} vs {want}");
    }
}

#[test]
fn the_bit_cast_intrinsics_reinterpret_without_converting() {
    let k = lift(
        "void k(float* out, const float* in) { int b = __float_as_int(in[0]); unsigned int u = __float_as_uint(in[2]); \
         out[0] = (float)(b >> 23 & 255); out[1] = __int_as_float(b); out[2] = __uint_as_float(u + 0x800000u); \
         out[3] = (float)(u >> 23); }",
    )
    .expect("lifts");
    let b = IN[0].to_bits() as i32;
    let u = IN[2].to_bits();
    let r = run(&k, 4);
    assert_eq!(r[0], f64::from((b >> 23) & 255));
    assert_eq!(r[1], f64::from(IN[0]));
    // adding one to the exponent field doubles the value
    assert_eq!(r[2], f64::from(f32::from_bits(u + 0x80_0000)));
    assert_eq!(r[2], 1.5);
    assert_eq!(r[3], f64::from(u >> 23));
}

#[test]
fn double_bit_casts_and_what_stays_declined() {
    let k = lift(
        "void k(double* out, const double* in) { long long b = __double_as_longlong(in[0]); out[0] = __longlong_as_double(b + 0x10000000000000LL); }",
    );
    // a double kernel with a double buffer: runs below only if the oracle supports it; the lift is the claim
    assert!(k.is_ok(), "{k:?}");
    for src in [
        "void k(float* out, const float* in) { out[0] = hypotf(in[0], in[1]); }",
        "void k(float* out, const float* in) { out[0] = __float_as_int(in[0], in[1]); }",
    ] {
        assert!(
            matches!(lift(src), Err(KernelLiftError::NotYetLifted { .. })),
            "{src}"
        );
    }
}
