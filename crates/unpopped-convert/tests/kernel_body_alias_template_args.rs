// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Census wall: an explicit template call whose argument is a local `using` alias** (`using Acc = typename WelfordAcc<T>::type; ...
//! welford_sqrt<Acc>(v)`: 8 real instantiations stopped at "a call to `welford_sqrt<Acc>`"). The argument is read as the type the alias
//! names: a plain type as written, `typename C<args>::member` through the class template `C` for those arguments (the matching
//! specialisation, else the primary) and its own `using member = X;`. Oracle runs.

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

// the primary accumulates in T, the double specialisation in float; `fin` has one definition per accumulator type
const SRC: &str = "template <typename T> struct Acc { using type = T; };\n\
    template <> struct Acc<double> { using type = float; };\n\
    template <typename A> A fin(A v);\n\
    template <> float fin<float>(float v) { return v + 1.0f; }\n\
    template <> double fin<double>(double v) { return v * 2.0; }\n\
    template <typename T> void k(T* out, const T* in) { using AccT = typename Acc<T>::type; AccT a = (AccT)in[0]; out[0] = (T)fin<AccT>(a); }";

fn lift(t: &str) -> Result<KernelFn, KernelLiftError> {
    let ctx = LiftContext::new(LaunchShape::new([1, 1, 1], [1, 1, 1])).with_type("T", t);
    lift_kernel_in(&C, SRC, "k", &ctx)
}

#[test]
fn the_primary_class_gives_the_argument_type() {
    let k = lift("float").expect("lifts (was a decline: a call to `fin<AccT>`)");
    let mut args = vec![
        KernelArg::Buffer(TypedBuffer::from_f32(&[1], &[0.0])),
        KernelArg::Buffer(TypedBuffer::from_f32(&[1], &[3.0])),
    ];
    run_launch(&k, &mut args).expect("runs");
    let KernelArg::Buffer(o) = &args[0] else {
        panic!()
    };
    // Acc<float>::type is float: fin<float> adds 1
    assert_eq!(o.to_f64_vec(), vec![4.0]);
}

#[test]
fn a_matching_specialisation_wins_over_the_primary() {
    let k = lift("double").expect("lifts");
    let mut args = vec![
        KernelArg::Buffer(TypedBuffer::from_f64(&[1], &[0.0])),
        KernelArg::Buffer(TypedBuffer::from_f64(&[1], &[3.0])),
    ];
    run_launch(&k, &mut args).expect("runs");
    let KernelArg::Buffer(o) = &args[0] else {
        panic!()
    };
    // Acc<double>::type is float (the specialisation): fin<float> adds 1 (the primary would have given fin<double>: 6)
    assert_eq!(o.to_f64_vec(), vec![4.0]);
}
