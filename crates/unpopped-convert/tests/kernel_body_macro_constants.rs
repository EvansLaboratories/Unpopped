// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Census wall: an integer macro used in code** (`#define QR4_0 2` in a header, `dequantize<QK4_0, QR4_0>(...)`; 30+ Baracuda
//! kernels). The preprocessor kept macros for `#if` only; a use in code read as an unknown name. An object-like macro whose body
//! is an integer literal, or one parenthesised integer expression over earlier macros, with a single value over the whole file,
//! is that integer where it is used. A redefined, removed, empty or unparenthesised-expression macro declines. Oracle runs
//! against Rust references.

use unpopped::exec::LaunchShape;
use unpopped::kernel::KernelFn;
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_convert::{Frontend, KernelLiftError, LiftContext, lift_kernel_in};

fn parse(src: &str) -> Option<tree_sitter::Tree> {
    let b = src.replace("__device__", "          ");
    let mut p = tree_sitter::Parser::new();
    p.set_language(&tree_sitter_cpp::LANGUAGE.into()).ok()?;
    p.parse(b, None)
}

const C: Frontend = Frontend::new("c", "void", &[], "out", "in", parse);

fn lift(src: &str, header: Option<&str>) -> Result<KernelFn, KernelLiftError> {
    let mut ctx = LiftContext::new(LaunchShape::new([1, 1, 1], [1, 1, 1]));
    if let Some(h) = header {
        ctx = ctx.with_include("h.cuh", h);
    }
    lift_kernel_in(&C, src, "k", &ctx)
}

fn out(k: &KernelFn, n: usize) -> Vec<f64> {
    let mut args = vec![KernelArg::Buffer(TypedBuffer::from_f32(
        &[n as i64],
        &vec![0.0; n],
    ))];
    run_launch(k, &mut args).expect("runs");
    let KernelArg::Buffer(o) = &args[0] else {
        panic!()
    };
    o.to_f64_vec()
}

fn declines(r: Result<KernelFn, KernelLiftError>) {
    assert!(
        matches!(r, Err(KernelLiftError::NotYetLifted { .. })),
        "{r:?}"
    );
}

#[test]
fn a_header_macro_is_its_value_as_a_template_argument_an_extent_and_an_operand() {
    let hdr = "#define QK4_0 32\n#define QR4_0 2\n#define QI4_0 (QK4_0 / (4 * QR4_0))\n\
        template <int qk, int qr> __device__ float dmm(float v) { return v * (float)(qk * 100 + qr); }\n";
    let src = "#include \"h.cuh\"\nextern \"C\" void k(float* out) { float priv[QR4_0]; priv[1] = 1.0f; \
        out[0] = dmm<QK4_0, QR4_0>(1.0f); out[1] = (float)QI4_0; out[2] = priv[QR4_0 - 1] + (float)(QK4_0 - 5); }";
    let k = lift(src, Some(hdr)).unwrap();
    // 32 * 100 + 2; 32 / 8 = 4; 1 + (32 - 5)
    assert_eq!(out(&k, 3), vec![3202.0, 4.0, 28.0]);
}

#[test]
fn a_macro_in_the_source_itself_and_a_local_that_shadows_it() {
    let src = "#define N 3\nextern \"C\" void k(float* out) { out[0] = (float)(N * 2); int N = 5; out[1] = (float)N; }";
    assert_eq!(out(&lift(src, None).unwrap(), 2), vec![6.0, 5.0]);
}

#[test]
fn a_macro_with_two_values_or_removed_or_ambiguous_declines() {
    declines(lift(
        "#define A 1\n#define A 2\nextern \"C\" void k(float* out) { out[0] = (float)A; }",
        None,
    ));
    declines(lift(
        "#define A 1\n#undef A\nextern \"C\" void k(float* out) { out[0] = (float)A; }",
        None,
    ));
    // an empty flag is not the integer 1 where it is used
    declines(lift(
        "#define FLAG\nextern \"C\" void k(float* out) { out[0] = (float)FLAG; }",
        None,
    ));
    // `1 + 2` would regroup in `N * 2`
    declines(lift(
        "#define N 1 + 2\nextern \"C\" void k(float* out) { out[0] = (float)(N * 2); }",
        None,
    ));
    // a name that is not a macro is not read as 0
    declines(lift(
        "#define N (UNKNOWN + 1)\nextern \"C\" void k(float* out) { out[0] = (float)N; }",
        None,
    ));
    // the same value twice is one value; a parenthesised expression is its value
    let ok = "#define A 7\n#define A 7\n#define B (A * 2)\nextern \"C\" void k(float* out) { out[0] = (float)(A + B); }";
    assert_eq!(out(&lift(ok, None).unwrap(), 1), vec![21.0]);
}
