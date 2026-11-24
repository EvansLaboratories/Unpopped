// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Census wall: a parenthesised callee** (`(dequantize_q4_0)(vx, ib, iqs, v)`, a function passed as a template argument and
//! substituted for the parameter; 30 gguf mmvq kernels). `(f)(x)` is the call `f(x)`; `(a + b)(x)` and `(*f)(x)` are not names and
//! still decline. Oracle runs.

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

fn lift(src: &str) -> Result<KernelFn, KernelLiftError> {
    lift_kernel_in(
        &C,
        src,
        "k",
        &LiftContext::new(LaunchShape::new([1, 1, 1], [1, 1, 1])),
    )
}

fn first(k: &KernelFn) -> f64 {
    let mut args = vec![KernelArg::Buffer(TypedBuffer::from_f32(&[1], &[0.0]))];
    run_launch(k, &mut args).expect("runs");
    let KernelArg::Buffer(o) = &args[0] else {
        panic!()
    };
    o.to_f64_vec()[0]
}

#[test]
fn a_parenthesised_function_name_is_called_and_a_function_template_argument_reaches_it() {
    let k = lift(
        "float twice(float x) { return x * 2.0f; }\n\
         typedef float (*fn_t)(float);
\n         template <fn_t f> float apply(float v) { return (f)(v) + 1.0f; }\n\
         extern \"C\" void k(float* out) { out[0] = (twice)(5.0f) + apply<twice>(1.0f); }",
    )
    .unwrap();
    // 10 + (2 + 1)
    assert_eq!(first(&k), 13.0);
}

#[test]
fn a_parenthesised_expression_or_dereference_is_not_a_callee_name() {
    for callee in ["(twice + 1)", "(*twice)"] {
        let r = lift(&format!(
            "float twice(float x) {{ return x * 2.0f; }}\nextern \"C\" void k(float* out) {{ out[0] = {callee}(5.0f); }}"
        ));
        assert!(
            matches!(r, Err(KernelLiftError::NotYetLifted { .. })),
            "{callee}: {r:?}"
        );
    }
}
