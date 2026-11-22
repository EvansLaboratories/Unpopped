// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Census wall: a trait's member type** (`using Acc = typename AccOf<T>::type;`, 8 Baracuda kernels). A struct template with
//! `using type = ...;` members and explicit specialisations answers `typename Trait<Args>::type`: the specialisation with exactly
//! those arguments, else the primary template. Oracle runs against Rust references.

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

fn run(src: &str) -> Vec<f64> {
    let k = lift(src).unwrap();
    let mut args = vec![
        KernelArg::Buffer(TypedBuffer::from_f32(&[2], &[0.0, 0.0])),
        KernelArg::Buffer(TypedBuffer::from_f32(&[1], &[16_777_216.0])),
    ];
    run_launch(&k, &mut args).expect("runs");
    let KernelArg::Buffer(o) = &args[0] else {
        panic!()
    };
    o.to_f64_vec()
}

const TRAIT: &str = "template <typename T> struct AccOf { using type = float; };\n\
    template <> struct AccOf<double> { using type = double; };\n";

#[test]
fn the_primary_template_and_a_specialisation_give_different_member_types() {
    // 2^24 + 1 is exact in a double and rounds back to 2^24 in a float: the difference tells which type the member named
    let src = format!(
        "{TRAIT}void k(float* out, const float* in) {{ \
         typename AccOf<float>::type f = in[0]; f = f + 1.0f; out[0] = (float)(f - in[0]); \
         using D = typename AccOf<double>::type; D d = (D)in[0]; d = d + (D)1.0f; out[1] = (float)(d - (D)in[0]); }}"
    );
    assert_eq!(run(&src), vec![0.0, 1.0]);
}

#[test]
fn a_type_without_that_member_still_declines() {
    let src = format!(
        "{TRAIT}void k(float* out, const float* in) {{ typename AccOf<float>::nothing x = in[0]; out[0] = (float)x; }}"
    );
    assert!(matches!(
        lift(&src),
        Err(KernelLiftError::NotYetLifted { .. })
    ));
}

#[test]
fn a_nested_member_is_not_read_as_the_outer_traits_member() {
    // `AccOf<double>::inner::type` names a type nested in the trait, which has none: it must not fall back to `AccOf`'s own `type`
    let src = format!(
        "{TRAIT}void k(float* out, const float* in) {{ typename AccOf<double>::inner::type x = in[0]; out[0] = (float)x; }}"
    );
    assert!(matches!(
        lift(&src),
        Err(KernelLiftError::NotYetLifted { .. })
    ));
}
