// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Census wall: the CUDA fast-math intrinsics** (`__expf`, `__logf`, `__fdividef`, ...; 13+ Baracuda kernels stopped at
//! `__expf`). Each is written as the hardware sequence it is (`__expf(x)` = `ex2.approx(x * log2 e)`), an `approx` node the
//! oracle evaluates within a few ulp. Oracle runs against Rust references; the form is asserted from the lifted IR.

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

fn run(k: &KernelFn, input: &[f32]) -> Vec<f64> {
    let mut args = vec![
        KernelArg::Buffer(TypedBuffer::from_f32(&[1], &[0.0])),
        KernelArg::Buffer(TypedBuffer::from_f32(&[input.len() as i64], input)),
    ];
    run_launch(k, &mut args).expect("runs");
    let KernelArg::Buffer(o) = &args[0] else {
        panic!()
    };
    o.to_f64_vec()
}

fn one(call: &str) -> String {
    format!("void k(float* out, const float* in) {{ out[0] = {call}; }}")
}

#[test]
fn each_intrinsic_matches_the_exact_function_within_its_tolerance() {
    type Case = (&'static str, fn(f32) -> f32, [f32; 3]);
    let cases: [Case; 5] = [
        ("__expf(in[0])", f32::exp, [0.5, -2.25, 3.0]),
        ("__exp10f(in[0])", |x| 10f32.powf(x), [0.5, -1.5, 2.0]),
        ("__log2f(in[0])", f32::log2, [0.5, 3.5, 100.0]),
        ("__logf(in[0])", f32::ln, [0.5, 3.5, 100.0]),
        ("__log10f(in[0])", f32::log10, [0.5, 3.5, 100.0]),
    ];
    for (call, f, xs) in cases {
        let k = lift(&one(call)).unwrap();
        for x in xs {
            let got = run(&k, &[x])[0];
            let want = f64::from(f(x));
            assert!(
                (got - want).abs() <= 1e-5 * want.abs().max(1e-3),
                "{call} at {x}: {got} vs {want}"
            );
        }
    }
    let k = lift(&one("__fdividef(in[0], in[1])")).unwrap();
    let got = run(&k, &[7.0, 4.0])[0];
    assert!((got - 1.75).abs() < 1e-5, "{got}");
}

#[test]
fn the_lowering_is_the_approximate_hardware_form() {
    for call in ["__expf(in[0])", "__logf(in[0])", "__fdividef(in[0], in[1])"] {
        let k = lift(&one(call)).unwrap();
        let ir = format!("{:?}", k.body);
        assert!(ir.contains("approx: true"), "{call}: {ir}");
    }
}

#[test]
fn a_wrong_argument_count_or_a_name_not_in_the_set_declines() {
    assert!(lift(&one("__expf(in[0], in[1])")).is_err());
    assert!(lift(&one("__fdividef(in[0])")).is_err());
    assert!(lift(&one("__sinf(in[0])")).is_err());
}
