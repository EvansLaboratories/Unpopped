// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Type aliases, `static_assert` and the C limit constants** (census: `using T = ...;` stopped 27 of Baracuda's kernels, `static_assert` 21,
//! `INFINITY` and kin 14, `typedef ... floatX` 11). Each lifted kernel runs on the oracle against a Rust reference.

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

fn ctx() -> LiftContext {
    LiftContext::new(LaunchShape::new([1, 1, 1], [1, 1, 1]))
}

fn lift(src: &str) -> Result<KernelFn, KernelLiftError> {
    lift_kernel_in(&C, src, "k", &ctx())
}

fn run(k: &KernelFn, n: usize) -> Vec<f64> {
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

fn not_yet(r: Result<KernelFn, KernelLiftError>, needle: &str) {
    match r {
        Err(KernelLiftError::NotYetLifted { construct, .. }) => {
            assert!(construct.contains(needle), "`{construct}` lacks `{needle}`");
        }
        other => panic!("expected NotYetLifted({needle}), got {other:?}"),
    }
}

#[test]
fn file_scope_and_function_local_aliases_name_the_type_they_alias() {
    let src = "using AccT = float;\ntypedef int Idx;\ntypedef Idx Index2;\n\
        void k(float* out) { using Half = float; AccT a = 1.5f; Half b = 2.0f; Index2 i = 3; out[0] = a + b; out[1] = (float)i; }";
    assert_eq!(run(&lift(src).unwrap(), 2), vec![3.5, 3.0]);
    // an alias of an alias resolves through the chain; control: an alias that names an unknown type declines naming the TYPE
    not_yet(
        lift("using Q = NotAType;\nvoid k(float* out) { Q q = 1.0f; out[0] = 1.0f; }"),
        "the type `NotAType`",
    );
}

#[test]
fn an_alias_of_a_template_parameter_follows_the_callers_argument() {
    // the common shape: `using AccT = T;` inside a templated kernel
    let src = "template <typename T>\nvoid k(float* out, const T* in) { using AccT = T; AccT a = in[0]; out[0] = a * 2.0f; }";
    let c = ctx().with_type("T", "float");
    let k = lift_kernel_in(&C, src, "k", &c).unwrap();
    let mut args = vec![
        KernelArg::Buffer(TypedBuffer::from_f32(&[1], &[0.0])),
        KernelArg::Buffer(TypedBuffer::from_f32(&[1], &[2.5])),
    ];
    run_launch(&k, &mut args).unwrap();
    let KernelArg::Buffer(o) = &args[0] else {
        panic!()
    };
    assert_eq!(o.to_f64_vec(), vec![5.0]);
}

#[test]
fn a_static_assert_is_nothing_unless_it_is_false_for_these_arguments() {
    let ok = "static_assert(sizeof(float) == 4, \"f32\");\n\
        void k(float* out) { static_assert(1 < 2, \"ok\"); static_assert(sizeof(UnknownStruct) == 16); out[0] = 1.0f; }";
    assert_eq!(run(&lift(ok).unwrap(), 1), vec![1.0]);
    // false under the caller's arguments: the instantiation is bad, and it is said so
    let bad = "template <int N>\nvoid k(float* out) { static_assert(N > 8, \"too small\"); out[0] = 1.0f; }";
    not_yet(
        lift_kernel_in(&C, bad, "k", &ctx().with_const("N", 4)),
        "that does not hold for these arguments",
    );
    // control: the same kernel with an argument that satisfies it lifts
    assert!(lift_kernel_in(&C, bad, "k", &ctx().with_const("N", 9)).is_ok());
}

#[test]
fn the_c_limit_constants_are_constants() {
    let src = "void k(float* out) { out[0] = INFINITY; out[1] = -INFINITY; out[2] = FLT_MAX; out[3] = FLT_EPSILON; \
        out[4] = (float)INT_MAX; out[5] = (float)(INT_MIN + 1); out[6] = (float)UINT_MAX; }";
    let got = run(&lift(src).unwrap(), 7);
    assert_eq!(got[0], f64::INFINITY);
    assert_eq!(got[1], f64::NEG_INFINITY);
    assert_eq!(got[2], f64::from(f32::MAX));
    assert_eq!(got[3], f64::from(f32::EPSILON));
    assert_eq!(got[4], f64::from(i32::MAX as f32));
    assert_eq!(got[5], f64::from((i32::MIN + 1) as f32));
    assert_eq!(got[6], f64::from(u32::MAX as f32));
    // a file's own constant of the same name wins over the library's
    let own = "constexpr float FLT_MAX = 7.0f;\nvoid k(float* out) { out[0] = FLT_MAX; }";
    assert_eq!(run(&lift(own).unwrap(), 1), vec![7.0]);
    // an unknown name still declines
    not_yet(
        lift("void k(float* out) { out[0] = FLT_BOGUS; }"),
        "FLT_BOGUS",
    );
}
