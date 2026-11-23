// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Census wall: a static member function called through its struct** (`mmvq_io<ActT>::load(&y[off])`, `mmvq_io<DstT>::store(&dst[row], v)`;
//! the gguf mmvq kernels). A `static` member of a struct, or of a class-template instance, is inlined like a free function under
//! `Struct::name`; a non-static member called that way still declines (it has no `this`). Oracle runs against Rust references.

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

fn run(k: &KernelFn, input: &[f32], n: usize) -> Vec<f64> {
    let mut args = vec![
        KernelArg::Buffer(TypedBuffer::from_f32(&[n as i64], &vec![0.0; n])),
        KernelArg::Buffer(TypedBuffer::from_f32(&[input.len() as i64], input)),
    ];
    run_launch(k, &mut args).expect("runs");
    let KernelArg::Buffer(o) = &args[0] else {
        panic!()
    };
    o.to_f64_vec()
}

#[test]
fn a_static_member_loads_and_stores_through_pointers() {
    let k = lift(
        "struct Io { static float load(const float* p) { return *p * 2.0f; } static void store(float* p, float v) { *p = v + 1.0f; } };\n\
         extern \"C\" void k(float* out, const float* in) { float s = 0.0f; for (int i = 0; i < 3; ++i) { s += Io::load(&in[i]); } Io::store(&out[1], s); }",
    )
    .unwrap();
    // 2 * (1 + 2 + 3) + 1 stored at out[1]
    assert_eq!(run(&k, &[1.0, 2.0, 3.0], 2), vec![0.0, 13.0]);
}

#[test]
fn a_class_template_instance_picks_its_specialisation_and_a_macro_can_hide_the_call() {
    // the primary doubles; the int specialisation adds 100: the instance chosen follows the template argument
    let k = lift(
        "template <typename T> struct Io { static float load(const T* p) { return *p * 2.0f; } };\n\
         template <> struct Io<int> { static float load(const int* p) { return (float)*p + 100.0f; } };\n\
         extern \"C\" void k(float* out, const float* in) {\n\
         #define Y(off) Io<float>::load(&in[off])\n\
         out[0] = Y(1) + Y(2);\n\
         #undef Y\n\
         }",
    )
    .unwrap();
    assert_eq!(run(&k, &[1.0, 2.0, 3.0], 1), vec![10.0]);
    let k = lift(
        "template <typename T> struct Io { static float load(const T* p) { return *p * 2.0f; } };\n\
         template <> struct Io<float> { static float load(const float* p) { return *p + 100.0f; } };\n\
         extern \"C\" void k(float* out, const float* in) { out[0] = Io<float>::load(&in[1]); }",
    )
    .unwrap();
    assert_eq!(run(&k, &[1.0, 2.0, 3.0], 1), vec![102.0]);
}

#[test]
fn a_non_static_member_called_through_the_struct_name_declines() {
    let r = lift(
        "struct S { float f(float x) { return x; } };\n\
         extern \"C\" void k(float* out, const float* in) { out[0] = S::f(in[0]); }",
    );
    assert!(
        matches!(r, Err(KernelLiftError::NotYetLifted { .. })),
        "{r:?}"
    );
}
