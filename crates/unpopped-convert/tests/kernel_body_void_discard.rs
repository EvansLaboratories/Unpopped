// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Census wall: `(void)x;`** (a statement that silences an unused-parameter warning; 7 Baracuda pooling kernels). It does nothing, so
//! it lifts to nothing, but only for a plain name or literal: `(void)f(x);` and `(void)(a = 1);` carry effects and decline, and a
//! cast to another type as a statement still declines. Oracle runs.

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

#[test]
fn a_void_cast_of_a_name_or_literal_is_a_statement_that_does_nothing() {
    let k = lift("void k(float* out, int n, int spatial) { out[0] = (float)n; (void)spatial; (void)n; (void)0; }").unwrap();
    let mut args = vec![
        KernelArg::Buffer(TypedBuffer::from_f32(&[1], &[0.0])),
        KernelArg::Int(7),
        KernelArg::Int(3),
    ];
    run_launch(&k, &mut args).expect("runs");
    let KernelArg::Buffer(o) = &args[0] else {
        panic!()
    };
    assert_eq!(o.to_f64_vec(), vec![7.0]);
}

#[test]
fn a_void_cast_with_an_effect_or_another_cast_statement_declines() {
    for body in [
        "(void)(out[0] = 1.0f);",
        "(void)sqrtf(2.0f);",
        "(float)n;",
        "(int)n;",
    ] {
        let r = lift(&format!("void k(float* out, int n) {{ {body} }}"));
        assert!(
            matches!(r, Err(KernelLiftError::NotYetLifted { .. })),
            "{body}: {r:?}"
        );
    }
}
