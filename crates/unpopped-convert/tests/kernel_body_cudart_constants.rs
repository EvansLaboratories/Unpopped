// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Census wall: CUDA's `math_constants.h`** (`CUDART_INF_F`, `CUDART_NAN_F`, `CUDART_PI_F`, `CUDART_ZERO_F`, `CUDART_ONE_F`;
//! 3 Baracuda kernels stopped at `CUDART_INF_F`). Single-precision constants, like `FLT_MAX`. Oracle runs against Rust references.

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
fn each_constant_has_its_value() {
    let src = "void k(float* out) { out[0] = (float)(CUDART_INF_F > 3.0e38f); out[1] = (float)isnan(CUDART_NAN_F); \
        out[2] = CUDART_PI_F; out[3] = CUDART_ONE_F + CUDART_ZERO_F * 5.0f; out[4] = -CUDART_INF_F; }";
    let k = lift(src).unwrap();
    let mut args = vec![KernelArg::Buffer(TypedBuffer::from_f32(&[5], &[0.0; 5]))];
    run_launch(&k, &mut args).expect("runs");
    let KernelArg::Buffer(o) = &args[0] else {
        panic!()
    };
    assert_eq!(
        o.to_f64_vec(),
        vec![
            1.0,
            1.0,
            f64::from(std::f32::consts::PI),
            1.0,
            f64::NEG_INFINITY
        ]
    );
}

#[test]
fn a_name_that_is_not_one_of_them_still_declines() {
    assert!(lift("void k(float* out) { out[0] = CUDART_NOTHING_F; }").is_err());
}
