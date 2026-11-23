// SPDX-License-Identifier: MIT OR Apache-2.0
//! **`LiftContext::with_include`** (pm-kernels `[TASK]` 2026-11-22; the census's largest wall: ~210 Baracuda kernels call a helper
//! defined in a header the per-file extraction never saw). The caller supplies include files by exact name; nothing is read from
//! disk. Each lift runs on the oracle against a Rust reference; the control kernel needs an include (`load_as_acc<float>`, the
//! shape of Baracuda's `baracuda_flce.cuh`), and without it the decline names the file.

use unpopped::exec::LaunchShape;
use unpopped::kernel::KernelFn;
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_convert::{Frontend, KernelLiftError, LiftContext, lift_kernel_in};

fn parse(src: &str) -> Option<tree_sitter::Tree> {
    let mut b = src.to_string();
    for q in ["__host__", "__device__", "__forceinline__", "__shared__"] {
        b = b.replace(q, &" ".repeat(q.len()));
    }
    let mut p = tree_sitter::Parser::new();
    p.set_language(&tree_sitter_cpp::LANGUAGE.into()).ok()?;
    p.parse(b, None)
}

const C: Frontend = Frontend::new("c", "void", &[], "out", "in", parse);

fn ctx() -> LiftContext {
    LiftContext::new(LaunchShape::new([1, 1, 1], [1, 1, 1]))
}

fn lift(src: &str, c: &LiftContext) -> Result<KernelFn, KernelLiftError> {
    lift_kernel_in(&C, src, "k", c)
}

fn run(k: &KernelFn, input: &[f32]) -> Vec<f64> {
    let mut args = vec![
        KernelArg::Buffer(TypedBuffer::from_f32(&[1], &[0.0])),
        KernelArg::Buffer(TypedBuffer::from_f32(&[input.len() as i64], input)),
    ];
    run_launch(k, &mut args).expect("runs");
    let KernelArg::Buffer(o) = &args[0] else {
        panic!()
    };
    o.to_f64_vec()
}

/// A header in the shape of `baracuda_dtype_promote.cuh`: a guarded template helper.
const PROMOTE: &str = "#ifndef PROMOTE_CUH\n#define PROMOTE_CUH\n\
    template <typename T>\n__device__ __forceinline__ float load_as_acc(T v) { return (float)v * 2.0f; }\n#endif\n";

/// A header whose helper holds a construct the walker declines (a `goto`), on the header's line 2.
const BAD: &str = "float bad(float v) {\n  goto done;\n  return v;\n}\n";

const KERNEL: &str = "#include \"promote.cuh\"\n#include <cuda_fp16.h>\n\
    template <typename T>\nvoid k(float* out, const T* in) { out[0] = load_as_acc<T>(in[0]) + 1.0f; }";

#[test]
fn a_template_helper_defined_in_an_include_lifts_when_the_caller_supplies_it() {
    let c = ctx()
        .with_type("T", "float")
        .with_include("promote.cuh", PROMOTE);
    assert_eq!(run(&lift(KERNEL, &c).unwrap(), &[2.5]), vec![6.0]);
}

#[test]
fn without_the_include_the_decline_names_the_file_it_needs() {
    let c = ctx().with_type("T", "float");
    match lift(KERNEL, &c) {
        Err(KernelLiftError::NotYetLifted { construct, line }) => {
            assert!(construct.contains("load_as_acc"), "{construct}");
            assert!(
                construct.contains("`promote.cuh`, which was not supplied"),
                "{construct}"
            );
            assert!(
                !construct.contains("cuda_fp16"),
                "a system header is not named: {construct}"
            );
            assert_eq!(line, 4, "the line in the source's own file");
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn a_decline_inside_an_include_is_reported_at_the_directive_with_the_file_and_line() {
    let header = BAD;
    let src = "void head() {}\n#include \"bad.cuh\"\nvoid k(float* out, const float* in) { out[0] = bad(in[0]); }";
    match lift(src, &ctx().with_include("bad.cuh", header)) {
        Err(KernelLiftError::NotYetLifted { construct, line }) => {
            assert!(
                construct.contains("in the include `bad.cuh`, line 2"),
                "{construct}"
            );
            assert_eq!(line, 2, "the `#include` line of the source");
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn an_include_supplies_constants_structs_and_a_nested_include() {
    let consts = "constexpr float SCALE = 3.0f;\nstruct Pair { float a; float b; };\n";
    let outer = "#pragma once\n#include \"consts.cuh\"\n__device__ float sum(Pair p) { return (p.a + p.b) * SCALE; }\n";
    let src = "#include \"outer.cuh\"\n#include \"outer.cuh\"\n\
        void k(float* out, const float* in) { Pair p; p.a = in[0]; p.b = 1.0f; out[0] = sum(p); }";
    let c = ctx()
        .with_include("outer.cuh", outer)
        .with_include("consts.cuh", consts);
    assert_eq!(run(&lift(src, &c).unwrap(), &[2.0]), vec![9.0]);
}

#[test]
fn the_name_is_matched_exactly_and_a_cycle_is_named() {
    let c = ctx()
        .with_type("T", "float")
        .with_include("promote.h", PROMOTE);
    assert!(
        matches!(lift(KERNEL, &c), Err(KernelLiftError::NotYetLifted { construct, .. }) if construct.contains("`promote.cuh`")),
        "a near name does not resolve"
    );
    let c = ctx()
        .with_include("a.h", "#include \"b.h\"\n")
        .with_include("b.h", "#include \"a.h\"\n");
    match lift(
        "#include \"a.h\"\nvoid k(float* out) { out[0] = 1.0f; }",
        &c,
    ) {
        Err(KernelLiftError::NotYetLifted { construct, .. }) => {
            assert!(
                construct.contains("depth limit") && construct.contains("a.h -> b.h -> a.h"),
                "{construct}"
            );
        }
        other => panic!("{other:?}"),
    }
}
