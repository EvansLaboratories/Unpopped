// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Census wall: out-of-class member specialisations of a class template** (`template <typename T> struct AccMin { static T
//! identity(); ... }; template <> float AccMin<float>::identity() { return FLT_MAX; }`: the Baracuda reduce-to and reduce-axis functors,
//! 12 real instantiations stopped at "the field `init()`" and "a call to `AccMin__t0::identity`"). The instance of the class for the
//! arguments in use takes the matching out-of-class definition as an in-class one (a `static` member stays `static`); the in-class
//! prototype is a method declaration, not data. The definition of ANOTHER specialisation is never used. Oracle runs.

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

const SRC: &str = "template <typename T> struct MinR { static T init(); T operator()(T a, T b) const { return b < a ? b : a; } };\n\
    template <> double MinR<double>::init() { return 77.0; }\n\
    template <> float MinR<float>::init() { return 3.5f; }\n\
    template <typename T, typename F> void k(T* out, const T* in, int n) { F op{}; T acc = F::init(); \
        for (int i = 0; i < n; ++i) { acc = op(acc, in[i]); } out[0] = acc; }";

fn lift(src: &str, t: &str, f: &str) -> Result<KernelFn, KernelLiftError> {
    let ctx = LiftContext::new(LaunchShape::new([1, 1, 1], [1, 1, 1]))
        .with_type("T", t)
        .with_type("F", f);
    lift_kernel_in(&C, src, "k", &ctx)
}

fn run(k: &KernelFn, input: &[f32]) -> f64 {
    let mut args = vec![
        KernelArg::Buffer(TypedBuffer::from_f32(&[1], &[0.0])),
        KernelArg::Buffer(TypedBuffer::from_f32(&[input.len() as i64], input)),
        KernelArg::Int(input.len() as i128),
    ];
    run_launch(k, &mut args).expect("runs");
    let KernelArg::Buffer(o) = &args[0] else {
        panic!()
    };
    o.to_f64_vec()[0]
}

#[test]
fn the_instance_takes_its_own_out_of_class_member_definition() {
    let k = lift(SRC, "float", "MinR<float>").expect("lifts (was a decline: the field `init()`)");
    // the fold starts at init() = 3.5, so a value below it wins and a value above it does not
    assert_eq!(run(&k, &[5.0, 2.0, 7.0]), 2.0);
    assert_eq!(
        run(&k, &[5.0, 6.0, 7.0]),
        3.5,
        "init() is the float specialisation's 3.5, not the double one's 77 and not 0"
    );
}

#[test]
fn a_class_with_no_matching_member_specialisation_keeps_a_prototype_only_and_declines_at_the_call()
{
    // MinR<int> has no `init` definition anywhere: the prototype is not data and not a body, so the call declines by name
    match lift(SRC, "int", "MinR<int>") {
        Err(KernelLiftError::NotYetLifted { construct, .. }) => {
            assert!(construct.contains("init"), "{construct}");
        }
        other => panic!("expected a decline, got {other:?}"),
    }
}
