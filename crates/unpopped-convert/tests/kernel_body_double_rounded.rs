// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Census wall: the double `CUDART_*` constants** (`CUDART_INF` 4 real instantiations). `CUDART_INF` and `CUDART_NAN` are exact at
//! either float compute type, `CUDART_PI` only at F64 (a double constant in an F32 kernel declines like a double literal). The double
//! rounding intrinsics (`__dadd_rn` ...) are NOT lifted: see the pinned declines below.

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

fn run(k: &KernelFn, input: &[f64], outs: usize) -> Vec<f64> {
    let mut args = vec![
        KernelArg::Buffer(TypedBuffer::from_f64(&[outs as i64], &vec![0.0; outs])),
        KernelArg::Buffer(TypedBuffer::from_f64(&[input.len() as i64], input)),
    ];
    run_launch(k, &mut args).expect("runs");
    let KernelArg::Buffer(o) = &args[0] else {
        panic!()
    };
    o.to_f64_vec()
}

#[test]
fn the_double_rounding_intrinsics_stay_declined_until_the_oracle_can_run_qualified_f64() {
    // `Qualified` at F64 passes `check_kernel` but the oracle refuses to evaluate it (`expr_eval.rs`: "Qualified at F64 (the plan gate
    // rejects this)"), so a kernel lifted with one would panic when run: they decline until that is settled (reported to pm-kernels)
    for call in [
        "__dadd_rn(in[0], in[1])",
        "__dmul_rz(in[0], in[1])",
        "__ddiv_ru(1.0, in[1])",
        "__dsqrt_rn(in[1])",
        "__fma_rn(in[0], in[1], 0.5)",
    ] {
        let src = format!("void k(double* out, const double* in) {{ out[0] = {call}; }}");
        assert!(
            matches!(lift(&src), Err(KernelLiftError::NotYetLifted { .. })),
            "{src}"
        );
    }
}

#[test]
fn the_double_constants_infinity_and_nan_and_pi_at_double_compute() {
    let k = lift(
        "void k(double* out, const double* in) { out[0] = CUDART_INF; out[1] = -CUDART_INF; out[2] = CUDART_NAN; out[3] = CUDART_PI; }",
    )
    .expect("lifts");
    let r = run(&k, &[0.0], 4);
    assert_eq!(r[0], f64::INFINITY);
    assert_eq!(r[1], f64::NEG_INFINITY);
    assert!(r[2].is_nan());
    assert_eq!(r[3], std::f64::consts::PI);
    // infinity is exact in a float kernel too; a double pi is not
    assert!(lift("void k(float* out, const float* in) { out[0] = CUDART_INF; }").is_ok());
    assert!(matches!(
        lift("void k(float* out, const float* in) { out[0] = CUDART_PI; }"),
        Err(KernelLiftError::NotYetLifted { .. })
    ));
}
