// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Census wall found by real instantiations: a trait reached through a local alias** (`using DT = LseDtype<T>; using C = typename
//! DT::Compute; C m = DT::neg_infinity(); m = DT::gmax(m, DT::load(x[i]));`; the logsumexp kernels, 31). `typename Alias::member`
//! follows the alias to the trait, and `Alias::f(..)` calls the trait instance's static member. The explicit specialisation picked
//! follows the template argument. Oracle runs against Rust references.

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

const SRC: &str = "template <typename T> struct Lse;\n\
    template <> struct Lse<float> { using Compute = float; static float load(float v) { return v * 2.0f; } \
      static float gmax(float a, float b) { return fmaxf(a, b); } static float neg() { return -100.0f; } };\n\
    template <> struct Lse<double> { using Compute = double; static double load(double v) { return v * 3.0; } \
      static double gmax(double a, double b) { return fmax(a, b); } static double neg() { return -200.0; } };\n\
    template <typename T> void k(T* out, const T* in) { using DT = Lse<T>; using C = typename DT::Compute; \
      C m = DT::neg(); for (int i = 0; i < 3; ++i) { m = DT::gmax(m, DT::load(in[i])); } out[0] = m; }";

fn lift(ty: &str) -> Result<KernelFn, KernelLiftError> {
    lift_kernel_in(
        &C,
        SRC,
        "k",
        &LiftContext::new(LaunchShape::new([1, 1, 1], [1, 1, 1])).with_type("T", ty),
    )
}

#[test]
fn the_alias_reaches_the_specialisation_of_its_template_argument() {
    let k = lift("float").unwrap();
    let mut args = vec![
        KernelArg::Buffer(TypedBuffer::from_f32(&[1], &[0.0])),
        KernelArg::Buffer(TypedBuffer::from_f32(&[3], &[1.0, 4.0, -7.0])),
    ];
    run_launch(&k, &mut args).expect("runs");
    let KernelArg::Buffer(o) = &args[0] else {
        panic!()
    };
    // float: max(-100, 2, 8, -14) = 8
    assert_eq!(o.to_f64_vec(), vec![8.0]);
    let k = lift("double").unwrap();
    let mut args = vec![
        KernelArg::Buffer(TypedBuffer::from_f64(&[1], &[0.0])),
        KernelArg::Buffer(TypedBuffer::from_f64(&[3], &[1.0, 4.0, -7.0])),
    ];
    run_launch(&k, &mut args).expect("runs");
    let KernelArg::Buffer(o) = &args[0] else {
        panic!()
    };
    // double: max(-200, 3, 12, -21) = 12
    assert_eq!(o.to_f64_vec(), vec![12.0]);
}

#[test]
fn an_alias_of_something_else_has_no_static_members() {
    let r = lift_kernel_in(
        &C,
        "template <typename T> void k(T* out, const T* in) { using F = float; out[0] = F::load(in[0]); }",
        "k",
        &LiftContext::new(LaunchShape::new([1, 1, 1], [1, 1, 1])).with_type("T", "float"),
    );
    assert!(
        matches!(r, Err(KernelLiftError::NotYetLifted { .. })),
        "{r:?}"
    );
}
