// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Census wall found by real instantiations: a functor parameter called like a function** (`F op` as a kernel parameter, `y[i] =
//! op(a[i], b[i]);` with `F` = `AddFunctor<float>`; the elementwise kernels, ~34). `op(args)` where `op` names a struct instance
//! that has an `operator()` inlines that method with `this` bound to the instance (a functor with data members works the same way:
//! its fields are in scope). A call on a struct with no `operator()` still declines. Oracle runs against Rust references.

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

const FUNCTORS: &str = "template <typename T> struct AddFunctor { T operator()(T a, T b) const { return a + b; } };\n\
    template <typename T> struct ScaleAdd { T k; T operator()(T a, T b) const { return a * k + b; } };\n\
    struct NoCall { int z; };\n";

fn lift(kernel: &str, f: &str) -> Result<KernelFn, KernelLiftError> {
    lift_kernel_in(
        &C,
        &format!("{FUNCTORS}{kernel}"),
        "k",
        &LiftContext::new(LaunchShape::new([1, 1, 1], [1, 1, 1]))
            .with_type("T", "float")
            .with_type("F", f),
    )
}

const KERNEL: &str = "template <typename T, typename F> void k(const T* a, const T* b, T* y, long long numel, F op) \
    { for (long long i = 0; i < numel; i++) { y[i] = op(a[i], b[i]); } }";

fn run(k: &KernelFn, functor: Vec<KernelArg>) -> Vec<f64> {
    let mut args = vec![
        KernelArg::Buffer(TypedBuffer::from_f32(&[3], &[1.0, 2.0, 3.0])),
        KernelArg::Buffer(TypedBuffer::from_f32(&[3], &[10.0, 20.0, 30.0])),
        KernelArg::Buffer(TypedBuffer::from_f32(&[3], &[0.0; 3])),
        KernelArg::Int(3),
    ];
    args.extend(functor);
    run_launch(k, &mut args).expect("runs");
    let KernelArg::Buffer(o) = &args[2] else {
        panic!()
    };
    o.to_f64_vec()
}

#[test]
fn an_empty_functor_parameter_is_called_like_a_function() {
    let k = lift(KERNEL, "AddFunctor<float>").unwrap();
    assert_eq!(
        run(&k, vec![KernelArg::Struct(vec![])]),
        vec![11.0, 22.0, 33.0]
    );
}

#[test]
fn a_functor_with_a_member_reads_it_in_operator_call() {
    let k = lift(KERNEL, "ScaleAdd<float>").unwrap();
    // y = a * k + b with k = 2
    assert_eq!(
        run(&k, vec![KernelArg::Struct(vec![KernelArg::Float(2.0)])]),
        vec![12.0, 24.0, 36.0]
    );
}

#[test]
fn a_call_on_a_struct_without_operator_call_declines() {
    let r = lift(KERNEL, "NoCall");
    assert!(
        matches!(r, Err(KernelLiftError::NotYetLifted { .. })),
        "{r:?}"
    );
}
