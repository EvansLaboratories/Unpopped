// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Census wall: the rounding-named arithmetic intrinsics** (`__frcp_rn`, `__fadd_rz`, `__fmaf_rn`, `__fmul_rd`, ...; 6+ Baracuda
//! kernels stopped at `__frcp_rn`). Each is the plain operation wrapped in the IR's `Qualified` node with the rounding its suffix
//! names. Directed rounding is observable as a one-ulp difference from round-to-nearest. Oracle runs against Rust references.

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

fn run(src: &str, n: usize) -> Vec<f64> {
    let k = lift(src).unwrap();
    let mut args = vec![KernelArg::Buffer(TypedBuffer::from_f32(
        &[n as i64],
        &vec![0.0; n],
    ))];
    run_launch(&k, &mut args).expect("runs");
    let KernelArg::Buffer(o) = &args[0] else {
        panic!()
    };
    o.to_f64_vec()
}

#[test]
fn directed_division_and_square_root_differ_from_nearest_by_one_ulp() {
    // 1/3 in f32 rounds up (0.33333334 > 1/3), so round-down is one ulp (2^-25) below it; sqrt(2) rounds down, so round-up is
    // one ulp (2^-23) above it
    let src = "void k(float* out) { float one = 1.0f; float three = 3.0f; float two = 2.0f; \
        out[0] = __fdiv_rd(one, three) - __fdiv_rn(one, three); out[1] = __fdiv_ru(one, three) - __fdiv_rn(one, three); \
        out[2] = __fsqrt_ru(two) - __fsqrt_rn(two); out[3] = __fsqrt_rd(two) - __fsqrt_rn(two); }";
    let got = run(src, 4);
    assert_eq!(got, vec![-(2f64.powi(-25)), 0.0, 2f64.powi(-23), 0.0]);
}

#[test]
fn directed_addition_moves_with_the_sign() {
    let src = "void k(float* out) { float one = 1.0f; float tiny = 1.0e-10f; float m = -1.0f; float mt = -1.0e-10f; \
        out[0] = __fadd_ru(one, tiny) - one; out[1] = __fadd_rn(one, tiny) - one; out[2] = __fadd_rz(m, mt) - m; \
        out[3] = __fadd_rd(m, mt) - m; out[4] = __fsub_rd(one, tiny) - one; out[5] = __fmul_ru(one, one) - one; }";
    let got = run(src, 6);
    let ulp = 2f64.powi(-23);
    // 1 + tiny: up is 1 + 2^-23; -1 - tiny: toward zero stays -1, down is -1 - 2^-23; 1 - tiny: down is 1 - 2^-24
    assert_eq!(got, vec![ulp, 0.0, 0.0, -ulp, -(2f64.powi(-24)), 0.0]);
}

#[test]
fn a_fused_multiply_add_rounds_once_and_reciprocal_is_exact() {
    let src = "void k(float* out) { float a = 0.1f; float b = 10.0f; float c = -1.0f; \
        out[0] = __fmaf_rn(a, b, c); out[1] = __fmul_rn(a, b) + c; out[2] = __frcp_rn(4.0f); out[3] = __fdiv_rn(1.0f, 8.0f); }";
    let got = run(src, 4);
    let fused = 0.1f32.mul_add(10.0, -1.0);
    assert_ne!(
        fused, 0.0,
        "the reference must tell the fused result from the unfused one"
    );
    assert_eq!(got, vec![f64::from(fused), 0.0, 0.25, 0.125]);
}

#[test]
fn what_does_not_exist_or_takes_no_direction_declines() {
    for call in [
        "__frcp_rz(2.0f)",
        "__fadd_rx(1.0f, 2.0f)",
        "__fadd_rn(1.0f)",
        "__fmaf_rn(1.0f, 2.0f)",
        "__dmul_rn(1.0f, 2.0f)",
    ] {
        let src = format!("void k(float* out) {{ out[0] = {call}; }}");
        assert!(lift(&src).is_err(), "{call}");
    }
}
