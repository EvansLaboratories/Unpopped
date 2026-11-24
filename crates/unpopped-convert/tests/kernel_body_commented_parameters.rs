// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Census wall found by real instantiations: a comment among the parameters** (`T operator()(T x, float p0, float /*p1*/)`, the
//! PowI functors: the comment counted as a parameter and the call "took 4 arguments"). The unused unnamed parameter takes its
//! argument and binds nothing. Oracle runs.

use unpopped::exec::LaunchShape;
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_convert::{Frontend, LiftContext, lift_kernel_in};

fn parse(src: &str) -> Option<tree_sitter::Tree> {
    let mut p = tree_sitter::Parser::new();
    p.set_language(&tree_sitter_cpp::LANGUAGE.into()).ok()?;
    p.parse(src, None)
}

const C: Frontend = Frontend::new("c", "void", &[], "out", "in", parse);

#[test]
fn a_commented_out_parameter_name_still_takes_its_argument() {
    let k = lift_kernel_in(
        &C,
        "float scale(float x, float p0, float /*p1*/) { return x * p0; }\n\
         struct F { float operator()(float x, float p0, float /* p1 */) const { return x + p0; } };\n\
         void k(float* out, const float* in, F op) { out[0] = scale(in[0], 3.0f, 9.0f); out[1] = op(in[0], 10.0f, 9.0f); }",
        "k",
        &LiftContext::new(LaunchShape::new([1, 1, 1], [1, 1, 1])),
    )
    .expect("lifts");
    let mut args = vec![
        KernelArg::Buffer(TypedBuffer::from_f32(&[2], &[0.0; 2])),
        KernelArg::Buffer(TypedBuffer::from_f32(&[1], &[2.0])),
        KernelArg::Struct(vec![]),
    ];
    run_launch(&k, &mut args).expect("runs");
    let KernelArg::Buffer(o) = &args[0] else {
        panic!()
    };
    assert_eq!(o.to_f64_vec(), vec![6.0, 12.0]);
}
