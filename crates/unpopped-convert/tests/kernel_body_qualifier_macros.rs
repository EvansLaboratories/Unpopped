// SPDX-License-Identifier: MIT OR Apache-2.0
//! **A macro that is only function qualifiers** (`#define FLASHINFER_INLINE __forceinline__ __device__`, the flashinfer
//! `vec_t` members). Its uses are blanked, as the qualifiers themselves are; a macro with any other body, or with parameters,
//! is not one. Oracle runs against Rust references.

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

#[test]
fn a_qualifier_only_macro_disappears_where_it_is_used() {
    let src = "#define INL __forceinline__ __device__\n\
        struct S { float x; INL float twice() { return x * 2.0f; } };\n\
        INL float add1(float v) { return v + 1.0f; }\n\
        void k(float* out, const float* in) { S s; s.x = in[0]; out[0] = add1(s.twice()); }";
    let k = lift(src).unwrap();
    let mut args = vec![
        KernelArg::Buffer(TypedBuffer::from_f32(&[1], &[0.0])),
        KernelArg::Buffer(TypedBuffer::from_f32(&[1], &[4.0])),
    ];
    run_launch(&k, &mut args).expect("runs");
    let KernelArg::Buffer(o) = &args[0] else {
        panic!()
    };
    assert_eq!(o.to_f64_vec(), vec![9.0]);
}

#[test]
fn a_macro_with_another_body_is_not_blanked() {
    // `float` is not a qualifier: `RET f()` stays a use of an unexpanded macro and declines
    let src = "#define RET float\nRET add1(float v) { return v + 1.0f; }\n\
        void k(float* out, const float* in) { out[0] = add1(in[0]); }";
    assert!(lift(src).is_err());
}
