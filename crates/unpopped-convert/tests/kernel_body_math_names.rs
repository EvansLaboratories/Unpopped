// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Census wall: C library math names the frontend's table does not list** (`exp(x)` in the `double` kernels, `log1pf`, `rintf`; 20+
//! Baracuda kernels). The walker now also reads the `double` spellings of the table's functions and the exact `f`/plain spellings of
//! `expm1`, `log2`, `log10`, `log1p`, the inverse and hyperbolic trigonometry, `cbrt`, `erf`, `erfc`, `lgamma`, and `rint`/`nearbyint`
//! (round to nearest even, as `UnaryOp::Round`). The `f` spellings compute at `float`, the others at the argument's type (an integer
//! argument of a `double` function is promoted to `double`). `round`/`roundf` (ties away from zero) still decline, so does `__powf`.
//! Oracle runs against Rust references.

use unpopped::exec::LaunchShape;
use unpopped::kernel::KernelFn;
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_convert::{Frontend, KernelLiftError, LiftContext, lift_kernel_in};

fn parse(src: &str) -> Option<tree_sitter::Tree> {
    let mut p = tree_sitter::Parser::new();
    p.set_language(&tree_sitter_cpp::LANGUAGE.into()).ok()?;
    p.parse(src, None)
}

const C: Frontend = Frontend::new("c", "void", &[], "out", "in", parse);

fn lift(src: &str) -> Result<KernelFn, KernelLiftError> {
    lift_kernel_in(
        &C,
        src,
        "k",
        &LiftContext::new(LaunchShape::new([1, 1, 1], [1, 1, 1])),
    )
}

fn run_f32(src: &str, input: &[f32]) -> Vec<f64> {
    let k = lift(src).expect("lifts");
    let n = input.len();
    let mut args = vec![
        KernelArg::Buffer(TypedBuffer::from_f32(&[n as i64], &vec![0.0; n])),
        KernelArg::Buffer(TypedBuffer::from_f32(&[n as i64], input)),
    ];
    run_launch(&k, &mut args).expect("runs");
    let KernelArg::Buffer(o) = &args[0] else {
        panic!()
    };
    o.to_f64_vec()
}

fn near(got: &[f64], want: &[f64]) {
    assert_eq!(got.len(), want.len());
    for (g, w) in got.iter().zip(want) {
        assert!(
            (g - w).abs() <= 1e-5 * w.abs().max(1.0),
            "{got:?} vs {want:?}"
        );
    }
}

#[test]
fn the_plain_and_f_spellings_compute_the_named_function() {
    let x = [0.5f32, 1.5, 2.0];
    let src = |f: &str| {
        format!(
            "void k(float* out, const float* in) {{ for (int i = 0; i < 3; ++i) {{ out[i] = {f}(in[i]); }} }}"
        )
    };
    let want = |g: fn(f64) -> f64| x.iter().map(|v| g(f64::from(*v))).collect::<Vec<_>>();
    near(&run_f32(&src("exp"), &x), &want(f64::exp));
    near(&run_f32(&src("log1pf"), &x), &want(f64::ln_1p));
    near(&run_f32(&src("log1p"), &x), &want(f64::ln_1p));
    near(&run_f32(&src("expm1f"), &x), &want(f64::exp_m1));
    near(&run_f32(&src("log2"), &x), &want(f64::log2));
    near(&run_f32(&src("log10f"), &x), &want(f64::log10));
    near(&run_f32(&src("sqrt"), &x), &want(f64::sqrt));
    near(&run_f32(&src("tan"), &x), &want(f64::tan));
    near(&run_f32(&src("atanf"), &x), &want(f64::atan));
    near(&run_f32(&src("asinh"), &x), &want(f64::asinh));
    near(&run_f32(&src("cbrtf"), &x), &want(f64::cbrt));
}

#[test]
fn rint_rounds_half_to_even_and_round_still_declines() {
    let x = [0.5f32, 1.5, 2.5];
    let src = |f: &str| {
        format!(
            "void k(float* out, const float* in) {{ for (int i = 0; i < 3; ++i) {{ out[i] = {f}(in[i]); }} }}"
        )
    };
    // ties to even: 0, 2, 2 (not 1, 2, 3)
    assert_eq!(run_f32(&src("rintf"), &x), vec![0.0, 2.0, 2.0]);
    assert_eq!(run_f32(&src("nearbyint"), &x), vec![0.0, 2.0, 2.0]);
    for name in ["roundf", "round", "__powf", "erfx"] {
        let r = lift(&src(name));
        assert!(
            matches!(r, Err(KernelLiftError::NotYetLifted { .. })),
            "{name}: {r:?}"
        );
    }
}

#[test]
fn a_double_function_takes_the_argument_type_and_an_f_function_takes_float() {
    // `erf` is a `double` function that merely ends in `f`: erf(1.0) at the argument's type; an integer argument is promoted
    let k = lift("void k(double* out, const double* in) { int n = 2; out[0] = erf(in[0]); out[1] = sqrt(n); out[2] = exp(in[1]); }")
        .expect("lifts");
    let mut args = vec![
        KernelArg::Buffer(TypedBuffer::from_f64(&[3], &[0.0; 3])),
        KernelArg::Buffer(TypedBuffer::from_f64(&[2], &[1.0, 0.25])),
    ];
    run_launch(&k, &mut args).expect("runs");
    let KernelArg::Buffer(o) = &args[0] else {
        panic!()
    };
    let got = o.to_f64_vec();
    // erf(1) = 0.8427007929497149 to 1e-12 (a table value, not a model of the lowering)
    assert!((got[0] - 0.842_700_792_949_714_9).abs() < 1e-12, "{got:?}");
    assert!((got[1] - 2f64.sqrt()).abs() < 1e-12, "{got:?}");
    assert!((got[2] - 0.25f64.exp()).abs() < 1e-12, "{got:?}");
}
