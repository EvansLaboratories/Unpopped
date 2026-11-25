// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Census wall: a function-like macro in code** (`#define Y(off) load(&y[off])` redefined and `#undef`ed inside a loop; the
//! gguf mmvq kernels). The preprocessor recorded such a macro as unexpandable and the use declined. A macro with a plain body
//! (no `#`, `##`, variadic) is now expanded textually at each use whose arguments close on the line: parameters replaced by the
//! arguments as written, the result rescanned, the definition in force at the use (a redefinition or `#undef` changes later uses).
//! A call with the wrong argument count or arguments that do not close on the line still declines. Oracle runs.

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
fn a_macro_with_arguments_expands_with_the_argument_as_written() {
    // SQ(a + 1) is a + 1 * a + 1 (no added parentheses): with a = 3, 3 + 3 + 1 = 7; SQ2 parenthesises: 16
    let k = lift(
        "#define SQ(x) x * x\n#define SQ2(x) ((x) * (x))\n\
         extern \"C\" void k(float* out) { float a = 3.0f; out[0] = SQ(a + 1.0f); out[1] = SQ2(a + 1.0f); }",
    )
    .unwrap();
    assert_eq!(out(&k, 2), vec![7.0, 16.0]);
}

#[test]
fn a_macro_redefined_and_undefined_in_a_loop_expands_per_the_definition_in_force() {
    let k = lift(
        "extern \"C\" void k(float* out) {\n\
         float y[4]; y[0] = 1.0f; y[1] = 2.0f; y[2] = 3.0f; y[3] = 4.0f;\n\
         float s = 0.0f;\n\
         for (int l = 0; l < 2; ++l) {\n\
         #define Y(off) y[off]\n\
         s += Y(l + 0) + Y(l + 2);\n\
         #undef Y\n\
         #define Y(off) (y[off] * 10.0f)\n\
         s += Y(l);\n\
         #undef Y\n\
         }\n\
         out[0] = s; }",
    )
    .unwrap();
    // l=0: 1+3 + 10 = 14; l=1: 2+4 + 20 = 26
    assert_eq!(out(&k, 1), vec![40.0]);
}

#[test]
fn a_macro_calling_another_is_rescanned_and_commas_inside_parentheses_stay_in_one_argument() {
    let k = lift(
        "#define ADD(a, b) ((a) + (b))\n#define TWICE(v) ADD(v, v)\n\
         float f2(float p, float q) { return p * 100.0f + q; }\n\
         extern \"C\" void k(float* out) { out[0] = TWICE(2.0f); out[1] = TWICE(f2(1.0f, 2.0f)); }",
    )
    .unwrap();
    assert_eq!(out(&k, 2), vec![4.0, 204.0]);
}

#[test]
fn a_wrong_argument_count_or_an_unclosed_call_or_stringising_still_declines() {
    declines(lift(
        "#define ADD(a, b) ((a) + (b))\nextern \"C\" void k(float* out) { out[0] = ADD(1.0f); }",
    ));
    declines(lift(
        "#define ADD(a, b) ((a) + (b))\nextern \"C\" void k(float* out) { out[0] = ADD(1.0f,\n 2.0f); }",
    ));
    declines(lift(
        "#define S(a) #a\nextern \"C\" void k(float* out) { out[0] = S(1.0f); }",
    ));
    // after #undef the name is a plain unknown call again
    declines(lift(
        "#define ADD(a, b) ((a) + (b))
#undef ADD
extern \"C\" void k(float* out) { out[0] = ADD(1.0f, 2.0f); }",
    ));
}

#[test]
fn a_block_comment_inside_a_macro_body_does_not_cut_the_body_short() {
    // The body was cut at the FIRST `/*`, so everything after a mid-body comment was lost. Found in the flce header: a continued macro
    // body `void* count_out, /* [1] int64_t; caller pre-zeroes */ \ void* stream_ptr)` expanded to a function cut off after `count_out,`,
    // and that unterminated text at the end of the file swallowed every template instance appended after it (12 real instantiations
    // failed with "a call to `store_from_acc__t0`").
    let k = lift(
        "#define ADD(a, b) ((a) /* the left */ + (b) /* the right; text */)\n\
         #define TWO (1 /* one */ + 1)\n\
         #define NOTE(a) (a) // runs to the end of the logical line\n\
         extern \"C\" void k(float* out) { out[0] = ADD(1.0f, 2.5f); out[1] = TWO; out[2] = NOTE(4.0f); }",
    )
    .expect("lifts");
    assert_eq!(out(&k, 3), vec![3.5, 2.0, 4.0]);
}

#[test]
fn a_continued_definition_macro_with_a_comment_before_the_backslash_keeps_its_whole_body() {
    // the body is a whole function; a comment sits at the end of a continued line (before the backslash); a template function after it
    // is instantiated from the kernel, and its appended instance must still parse
    let k = lift(
        "#define DEFINE_PAIR_SUM()                           \\\n\
         static float pair_sum(float a, /* [1] first; note */ \\\n\
                               float b)                       \\\n\
         { return a + b; }\n\
         DEFINE_PAIR_SUM()\n\
         template <typename T> T store(float v) { return (T)v; }\n\
         extern \"C\" void k(float* out) { out[0] = pair_sum(1.0f, 2.0f) + store<float>(3.0f); }",
    )
    .expect("lifts");
    assert_eq!(out(&k, 1), vec![6.0]);
}

#[test]
fn a_closed_comment_inside_an_if_condition_keeps_the_rest_of_the_condition() {
    // `#if A /* c */ && B` was read as `#if A`: the comment cut the condition, so the wrong branch was taken
    let k = lift(
        "#define A 1\n#define B 0\n#if A /* both */ && B\nextern \"C\" void k(float* out) { out[0] = 1.0f; }\n#else\nextern \"C\" void k(float* out) { out[0] = 2.0f; }\n#endif",
    )
    .expect("lifts");
    assert_eq!(out(&k, 1), vec![2.0]);
}
