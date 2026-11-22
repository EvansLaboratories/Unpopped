// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Census wall: a cast to an alias spelled `(Acc)(x)`** (`sx = (Acc)((double)ow * scale_w);`, 2 Baracuda kernels). The grammar
//! does not know `Acc` is a type and reads `(Acc)(x)` as a call of the parenthesised name; when the name is a type alias and no
//! variable has it, it is the cast. Oracle runs against Rust references.

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

fn run(k: &KernelFn) -> Vec<f64> {
    let mut args = vec![
        KernelArg::Buffer(TypedBuffer::from_f32(&[2], &[0.0, 0.0])),
        KernelArg::Buffer(TypedBuffer::from_f32(&[1], &[7.9])),
    ];
    run_launch(k, &mut args).expect("runs");
    let KernelArg::Buffer(o) = &args[0] else {
        panic!()
    };
    o.to_f64_vec()
}

#[test]
fn a_cast_to_an_alias_converts_like_the_c_cast() {
    // `Idx` is int: the cast truncates 7.9 to 7; `Acc` is float: it keeps the fraction
    let src = "using Idx = int; using Acc = float;\n\
        void k(float* out, const float* in) { Acc a = (Acc)(in[0] * 2.0f); out[0] = a; out[1] = (float)((Idx)(in[0])); }";
    let got = run(&lift(src).unwrap());
    assert_eq!(got, vec![f64::from(7.9f32 * 2.0), 7.0]);
}

#[test]
fn a_variable_of_that_name_or_a_non_type_is_not_a_cast() {
    // a local named like a type alias is a variable: `(i)(x)` is not a cast
    let a = "using Idx = int;\nvoid k(float* out, const float* in) { int Idx = 3; out[0] = (float)(Idx)(in[0]); }";
    // a name that is not a type at all
    let b = "void k(float* out, const float* in) { out[0] = (nothing)(in[0]); }";
    for src in [a, b] {
        assert!(lift(src).is_err(), "{src}");
    }
}
