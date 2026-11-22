// SPDX-License-Identifier: MIT OR Apache-2.0
//! **U19 census slice 1: template instantiation from the caller** (`docs/options-u19-census.md`, 391 kernels stopped on
//! "template"). `LiftContext::with_type` / `with_const` give a `template <typename T, int N>` kernel its arguments; a parameter
//! with a default takes it; one with neither, an argument naming no parameter, and arguments on a non-template each decline by
//! name. Each lifted instantiation runs on the oracle against a Rust reference, and a control instantiation differs.

use unpopped::exec::LaunchShape;
use unpopped::kernel::KernelFn;
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::{KernelArg, KernelTrap};
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_convert::{Frontend, KernelLiftError, LiftContext, lift_kernel_in};

fn parse(src: &str) -> Option<tree_sitter::Tree> {
    let mut b = src.to_string();
    for q in [
        "__host__",
        "__device__",
        "__forceinline__",
        "__grid_constant__",
        "__shared__",
    ] {
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

fn f32s(v: &[f32]) -> KernelArg {
    KernelArg::Buffer(TypedBuffer::from_f32(&[v.len() as i64], v))
}

fn run(k: &KernelFn, out: usize, extra: Vec<KernelArg>) -> Result<Vec<f64>, KernelTrap> {
    let mut args = vec![f32s(&vec![0.0; out])];
    args.extend(extra);
    run_launch(k, &mut args)?;
    let KernelArg::Buffer(o) = &args[0] else {
        panic!()
    };
    Ok(o.to_f64_vec())
}

fn not_yet(r: Result<KernelFn, KernelLiftError>, needle: &str) {
    match r {
        Err(KernelLiftError::NotYetLifted { construct, .. }) => {
            assert!(construct.contains(needle), "`{construct}` lacks `{needle}`");
        }
        other => panic!("expected NotYetLifted({needle}), got {other:?}"),
    }
}

const K: &str = "template <typename T, int N>
void k(T* out, const T* in) { T Tx = (T)2; out[0] = in[0] * (T)N; out[1] = (T)N + Tx; }";

fn run_k(c: &LiftContext, src: &str) -> Vec<f64> {
    run(&lift(src, c).unwrap(), 2, vec![f32s(&[2.5])]).unwrap()
}

#[test]
fn a_template_kernel_lifts_with_the_callers_type_and_value_arguments() {
    let c3 = ctx().with_type("T", "float").with_const("N", 3);
    assert_eq!(run_k(&c3, K), vec![7.5, 5.0]); // 2.5 * 3, 3 + 2 (`Tx` is a different identifier from `T`)
    // control: another value gives another result, so the argument is what was used
    let c5 = ctx().with_type("T", "float").with_const("N", 5);
    assert_eq!(run_k(&c5, K), vec![12.5, 7.0]);
}

#[test]
fn a_parameter_with_a_default_takes_it_and_the_callers_argument_wins() {
    let src = "template <typename T = float, int N = 4>
void k(T* out, const T* in) { out[0] = in[0] * (T)N; out[1] = (T)N; }";
    assert_eq!(run_k(&ctx(), src), vec![10.0, 4.0]);
    assert_eq!(run_k(&ctx().with_const("N", 2), src), vec![5.0, 2.0]);
}

#[test]
fn a_non_type_argument_is_a_constant_everywhere_it_is_used() {
    let src = "template <int N>
void k(float* out) { float a[N]; for (int i = 0; i < N; ++i) { a[i] = (float)i; } out[0] = a[N - 1]; out[1] = (float)(N * 2); }";
    assert_eq!(
        run(&lift(src, &ctx().with_const("N", 4)).unwrap(), 2, vec![]).unwrap(),
        vec![3.0, 8.0]
    );
    assert_eq!(
        run(&lift(src, &ctx().with_const("N", 6)).unwrap(), 2, vec![]).unwrap(),
        vec![5.0, 12.0]
    );
}

#[test]
fn a_negative_value_and_a_bool_are_substituted_whole() {
    let src = "template <int N, bool B>
void k(float* out) { out[0] = (float)(N * 2); out[1] = B ? 1.0f : 2.0f; }";
    let c = |b| ctx().with_const("N", -3).with_const("B", b);
    assert_eq!(
        run(&lift(src, &c(1)).unwrap(), 2, vec![]).unwrap(),
        vec![-6.0, 1.0]
    );
    assert_eq!(
        run(&lift(src, &c(0)).unwrap(), 2, vec![]).unwrap(),
        vec![-6.0, 2.0]
    );
}

#[test]
fn a_missing_argument_a_stale_argument_and_a_non_template_each_decline_by_name() {
    not_yet(
        lift(K, &ctx().with_const("N", 3)),
        "the template parameter `T` of `k` has no value (give one with `LiftContext::with_type`)",
    );
    not_yet(
        lift(K, &ctx().with_type("T", "float")),
        "the template parameter `N` of `k` has no value (give one with `LiftContext::with_const`)",
    );
    not_yet(
        lift(
            K,
            &ctx()
                .with_type("T", "float")
                .with_const("N", 3)
                .with_const("Z", 1),
        ),
        "the template value argument `Z`",
    );
    not_yet(
        lift(
            K,
            &ctx()
                .with_type("T", "float")
                .with_type("N", "int")
                .with_const("N", 3),
        ),
        "the template type argument `N`",
    );
    not_yet(
        lift(
            "void k(float* out) { out[0] = 1.0f; }",
            &ctx().with_const("N", 3),
        ),
        "is not a template",
    );
    // a variadic pack is not modelled
    not_yet(
        lift(
            "template <typename... Ts>
void k(float* out) { out[0] = 1.0f; }",
            &ctx(),
        ),
        "the template parameter form",
    );
}

#[test]
fn only_the_kernels_own_template_is_substituted() {
    // another template in the file keeps its own `T`: the helper below would not parse as `float` for a `T` bound to a number
    let src = "template <typename T>
T other(T x) { return x; }
template <typename T>
void k(T* out) { out[0] = (T)1; }";
    assert!(lift(src, &ctx().with_type("T", "float")).is_ok());
}

#[test]
fn a_negative_value_is_parenthesised_so_a_unary_minus_cannot_fuse_with_it() {
    // `-N` with N = -3 must be `-(-3)`, not `--3` (a decrement, which does not lift)
    let src = "template <int N>\nvoid k(float* out) { out[0] = (float)(-N); }";
    assert_eq!(
        run(&lift(src, &ctx().with_const("N", -3)).unwrap(), 1, vec![]).unwrap(),
        vec![3.0]
    );
}

#[test]
fn the_substitution_stops_at_the_end_of_the_kernels_template() {
    // a later function's own parameter named `N` is not the template's `N`: substituting through the end of the file would turn
    // `int twice(int N)` into `int twice(int (7))`, which does not parse, and the kernel that calls it would decline
    let src = "template <int N>\nvoid k(float* out) { out[0] = (float)twice(5); out[1] = (float)N; }\nint twice(int N) { return N * 2; }";
    assert_eq!(
        run(&lift(src, &ctx().with_const("N", 7)).unwrap(), 2, vec![]).unwrap(),
        vec![10.0, 7.0]
    );
}

// ---- census slice 2: helper templates called with explicit arguments ----

const SCALE: &str = "template <typename T, int N>\nT scale(T x) { return x * (T)N; }\n";

fn run1(src: &str, c: &LiftContext, outs: usize) -> Vec<f64> {
    run(&lift(src, c).unwrap(), outs, vec![f32s(&[2.5])]).unwrap()
}

#[test]
fn a_helper_template_called_with_explicit_arguments_is_instantiated_per_distinct_arguments() {
    let src = format!(
        "{SCALE}void k(float* out, const float* in) {{ out[0] = scale<float, 3>(in[0]); out[1] = scale<float, 5>(in[0]); out[2] = scale<float, 3>(in[0]) + 1.0f; }}"
    );
    assert_eq!(run1(&src, &ctx(), 3), vec![7.5, 12.5, 8.5]);
}

#[test]
fn the_kernels_own_arguments_flow_into_the_helpers() {
    let src = format!(
        "{SCALE}template <int N>\nvoid k(float* out, const float* in) {{ out[0] = scale<float, N>(in[0]); out[1] = scale<float, N + 1>(in[0]); }}"
    );
    assert_eq!(run1(&src, &ctx().with_const("N", 4), 2), vec![10.0, 12.5]);
    assert_eq!(run1(&src, &ctx().with_const("N", 2), 2), vec![5.0, 7.5]); // control
}

#[test]
fn a_helper_template_that_calls_a_helper_template_and_a_default_argument() {
    let src = format!(
        "{SCALE}template <int M>\nfloat twice(float x) {{ return scale<float, M>(x) * 2.0f; }}\n\
         template <typename T, int N = 2>\nT sc(T x) {{ return x * (T)N; }}\n\
         void k(float* out, const float* in) {{ out[0] = twice<3>(in[0]); out[1] = sc<float>(in[0]); }}"
    );
    assert_eq!(run1(&src, &ctx(), 2), vec![15.0, 5.0]);
}

#[test]
fn a_call_that_cannot_be_instantiated_declines_at_its_use() {
    // not a template in the file
    let r = lift(
        "void k(float* out) { out[0] = std::max<float>(1.0f, 2.0f); }",
        &ctx(),
    );
    assert!(
        matches!(r, Err(KernelLiftError::NotYetLifted { .. })),
        "{r:?}"
    );
    // too many arguments for the helper
    let src = format!(
        "{SCALE}void k(float* out, const float* in) {{ out[0] = scale<float, 3, 9>(in[0]); }}"
    );
    assert!(matches!(
        lift(&src, &ctx()),
        Err(KernelLiftError::NotYetLifted { .. })
    ));
    // a parameter with no argument and no default
    let src =
        format!("{SCALE}void k(float* out, const float* in) {{ out[0] = scale<float>(in[0]); }}");
    assert!(matches!(
        lift(&src, &ctx()),
        Err(KernelLiftError::NotYetLifted { .. })
    ));
}
