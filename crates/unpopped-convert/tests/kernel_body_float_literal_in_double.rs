// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Census wall found by real instantiations: a `float` literal in a `double` kernel** (`0.5f` in the `double` instance of a templated
//! kernel; 40 kernels). The literal is a `float` (its decimal rounded to `f32`), widened where C widens it: `x * 0.1f` with a `double`
//! `x` multiplies by the `f32` value of 0.1, not by the double 0.1, and two `float` literals combine in `float`. A `double` literal in
//! an `f32` kernel still declines. Oracle runs against Rust references.

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

#[test]
fn a_float_literal_in_a_double_kernel_has_its_f32_value() {
    let k = lift("void k(double* out, const double* in) { out[0] = in[0] * 0.1f; out[1] = in[0] + 0.5f; out[2] = (double)(0.1f + 0.2f); }")
        .unwrap();
    let mut args = vec![
        KernelArg::Buffer(TypedBuffer::from_f64(&[3], &[0.0; 3])),
        KernelArg::Buffer(TypedBuffer::from_f64(&[1], &[3.0])),
    ];
    run_launch(&k, &mut args).expect("runs");
    let KernelArg::Buffer(o) = &args[0] else {
        panic!()
    };
    let got = o.to_f64_vec();
    // the f32 value of 0.1 is 0.100000001490116..., so 3 * 0.1f differs from 3 * 0.1 at the 9th digit
    assert_eq!(got[0], 3.0 * f64::from(0.1f32));
    assert_ne!(got[0], 3.0 * 0.1f64);
    assert_eq!(got[1], 3.5);
    // two float literals add in float: 0.1f + 0.2f rounded to f32
    assert_eq!(got[2], f64::from(0.1f32 + 0.2f32));
}

#[test]
fn a_double_literal_in_a_float_kernel_still_declines() {
    let r = lift("void k(float* out, const float* in) { out[0] = in[0] * 0.1; }");
    assert!(
        matches!(r, Err(KernelLiftError::NotYetLifted { .. })),
        "{r:?}"
    );
}
