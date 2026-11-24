// SPDX-License-Identifier: MIT OR Apache-2.0
//! **U19 slice 4c-2: inlining the source's own functions** (`docs/options-u19-slice5-decode-census.md`). A call to a function
//! defined in the same source is replaced by its body: by-value and `T&` scalar parameters, one trailing `return`. The test
//! parser blanks the CUDA qualifiers (`__device__`, `__forceinline__`, `__host__`) to same-length spaces, as the walker reads
//! the original text.

use unpopped::kernel::KernelFn;
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::{KernelArg, run_kernel};
use unpopped_convert::{Frontend, KernelLiftError, lift_kernel};

fn parse(src: &str) -> Option<tree_sitter::Tree> {
    let mut blanked = src.to_string();
    for q in ["__device__", "__forceinline__", "__host__"] {
        blanked = blanked.replace(q, &" ".repeat(q.len()));
    }
    let mut p = tree_sitter::Parser::new();
    p.set_language(&tree_sitter_cpp::LANGUAGE.into()).ok()?;
    p.parse(blanked, None)
}

const C: Frontend = Frontend::new("c", "void", &[], "out", "in", parse);

fn lift(src: &str) -> Result<KernelFn, KernelLiftError> {
    lift_kernel(&C, src, "k")
}

fn run(k: &KernelFn, out: usize, a: &[f32]) -> Vec<f64> {
    let mut args = vec![KernelArg::Buffer(TypedBuffer::from_f32(
        &[out as i64],
        &vec![0.0; out],
    ))];
    if !a.is_empty() {
        args.push(KernelArg::Buffer(TypedBuffer::from_f32(
            &[a.len() as i64],
            a,
        )));
    }
    run_kernel(k, &mut args).expect("runs");
    let KernelArg::Buffer(o) = &args[0] else {
        panic!()
    };
    o.to_f64_vec()
}

#[test]
fn a_value_function_is_inlined_at_each_call_and_nests() {
    let k = lift(
        "__device__ __forceinline__ float sq(float x) { return x * x; }\n\
         __device__ float h(float x, int n) { float s = sq(x) + 1.0f; return s * (float)n; }\n\
         void k(float* out, const float* a) { for (int i = 0; i < 3; i++) { out[i] = h(a[i], i + 1) + sq(2.0f); } }",
    )
    .unwrap();
    let a = [1.5f32, -2.0, 3.25];
    let want: Vec<f64> = (0..3)
        .map(|i| {
            let s = a[i] * a[i] + 1.0;
            f64::from(s * (i as f32 + 1.0) + 4.0)
        })
        .collect();
    assert_eq!(run(&k, 3, &a), want);
}

#[test]
fn a_reference_parameter_writes_the_callers_local_and_a_value_parameter_does_not() {
    let k = lift(
        "__device__ void bump(int& c, int by) { c += by; by = 100; }\n\
         __device__ void inc(int c) { c = c + 1; }\n\
         void k(float* out) { int n = 1; bump(n, 5); bump(n, 7); int m = 3; inc(m); out[0] = (float)n; out[1] = (float)m; }",
    )
    .unwrap();
    assert_eq!(run(&k, 2, &[]), vec![13.0, 3.0]);
}

#[test]
fn an_argument_is_evaluated_once_and_a_load_in_it_comes_first() {
    let k = lift(
        "__device__ float twice(float x) { return x + x; }\n\
         void k(float* out, const float* a) { out[0] = twice(a[1]) + twice(a[2]); }",
    )
    .unwrap();
    let a = [0.0f32, 1.25, -3.5];
    assert_eq!(
        run(&k, 1, &a),
        vec![f64::from(1.25f32 + 1.25 + (-3.5 + -3.5))]
    );
}

#[test]
fn a_callee_loop_and_a_discarded_result() {
    let k = lift(
        "__device__ float sum_to(int n) { float s = 0.0f; for (int i = 0; i < n; i++) { s += (float)i; } return s; }\n\
         void k(float* out) { sum_to(3); out[0] = sum_to(5); }",
    )
    .unwrap();
    assert_eq!(run(&k, 1, &[]), vec![10.0]);
}

fn not_yet(src: &str, needle: &str) {
    match lift(src) {
        Err(KernelLiftError::NotYetLifted { construct, .. }) => {
            assert!(construct.contains(needle), "`{construct}` lacks `{needle}`");
        }
        other => panic!("expected NotYetLifted({needle}), got {other:?}"),
    }
}

#[test]
fn what_cannot_be_inlined_declines_by_name() {
    not_yet(
        "__device__ float f(float x) { return f(x); }\nvoid k(float* out) { out[0] = f(1.0f); }",
        "recursive call to `f`",
    );
    not_yet(
        "__device__ float f(float* p) { return 1.0f; }\nvoid k(float* out) { out[0] = f(1.0f); }",
        "not a pointer expression",
    );
    not_yet(
        "__device__ float f(float x) { return x; }\nvoid k(float* out) { out[0] = f(1.0f, 2.0f); }",
        "2 argument(s), it takes 1",
    );
    not_yet(
        "__device__ void g(float x) { }\nvoid k(float* out) { out[0] = g(1.0f); }",
        "returns void",
    );
    not_yet(
        "__device__ void bump(int& c) { c += 1; }\nvoid k(float* out, const float* a) { bump(3); out[0] = 1.0f; }",
        "reference argument",
    );
    // a `return` must match the function's type: a value from a `void` function, a bare `return` from a value function
    not_yet(
        "__device__ void g(float x) { return x; }
void k(float* out) { g(1.0f); out[0] = 1.0f; }",
        "does not match",
    );
    not_yet(
        "__device__ float g(float x) { return; }
void k(float* out) { out[0] = g(1.0f); }",
        "does not match",
    );
    // the callee does not see the caller's locals
    not_yet(
        "__device__ float f(float x) { return x + y; }\nvoid k(float* out) { float y = 1.0f; out[0] = f(2.0f); }",
        "name `y`",
    );
    // Control: the plain spelling lifts.
    assert!(
        lift("__device__ float f(float x) { return x; }\nvoid k(float* out) { out[0] = f(1.0f); }")
            .is_ok()
    );
}
