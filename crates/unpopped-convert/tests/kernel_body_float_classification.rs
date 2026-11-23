// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Census wall: `isnan`, `isinf`, `isfinite`** (`isfinite(x)`, 7 Baracuda kernels). `x != x`; `|x| == inf`; `x - x == 0`.
//! Oracle runs against Rust references on NaN, both infinities, zero and an ordinary value.

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

#[test]
fn each_class_test_agrees_with_rust_on_the_special_values() {
    let src = "void k(float* out, const float* in) { for (int i = 0; i < 6; i++) { \
        out[i * 3] = (float)isnan(in[i]); out[i * 3 + 1] = (float)isinf(in[i]); out[i * 3 + 2] = (float)isfinite(in[i]); } }";
    let input = [
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        0.0,
        -0.0,
        3.0e38,
    ];
    let k = lift(src).unwrap();
    let mut args = vec![
        KernelArg::Buffer(TypedBuffer::from_f32(&[18], &[0.0; 18])),
        KernelArg::Buffer(TypedBuffer::from_f32(&[6], &input)),
    ];
    run_launch(&k, &mut args).expect("runs");
    let KernelArg::Buffer(o) = &args[0] else {
        panic!()
    };
    let want: Vec<f64> = input
        .iter()
        .flat_map(|v| {
            [v.is_nan(), v.is_infinite(), v.is_finite()].map(|b| if b { 1.0 } else { 0.0 })
        })
        .collect();
    assert_eq!(o.to_f64_vec(), want);
}

#[test]
fn an_integer_argument_or_a_wrong_count_declines() {
    for call in ["isnan(1)", "isfinite(in[0], in[1])"] {
        let src = format!("void k(float* out, const float* in) {{ out[0] = (float){call}; }}");
        assert!(
            matches!(lift(&src), Err(KernelLiftError::NotYetLifted { .. })),
            "{call}"
        );
    }
}
