// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Census wall: a class template used with explicit arguments** (`vec_t<float, 128> v;`, U38b-1). The use becomes a plain
//! struct that is the template instantiated with those arguments, appended to the source: the explicit or partial
//! specialisation whose pattern matches (`template <size_t n> struct vec_t<float, n>` matches `vec_t<float, 128>`), else the
//! primary template. The instance is then an ordinary struct with its member functions. Oracle runs against Rust references.

use unpopped::exec::LaunchShape;
use unpopped::kernel::KernelFn;
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_convert::{
    Frontend, KernelLiftError, LiftContext, lift_kernel_in, lift_kernel_launched,
};

fn parse(src: &str) -> Option<tree_sitter::Tree> {
    let mut p = tree_sitter::Parser::new();
    p.set_language(&tree_sitter_cpp::LANGUAGE.into()).ok()?;
    p.parse(src.replace("__device__", "          "), None)
}

const C: Frontend = Frontend::new("c", "void", &[], "out", "in", parse);

fn lift(src: &str) -> Result<KernelFn, KernelLiftError> {
    lift_kernel_launched(&C, src, "k", LaunchShape::new([1, 1, 1], [1, 1, 1]))
}

fn run(k: &KernelFn, n_out: usize) -> Vec<f64> {
    let mut args = vec![
        KernelArg::Buffer(TypedBuffer::from_f32(&[n_out as i64], &vec![0.0; n_out])),
        KernelArg::Buffer(TypedBuffer::from_f32(&[4], &[1.5, 2.5, 3.5, 4.5])),
    ];
    run_launch(k, &mut args).expect("runs");
    let KernelArg::Buffer(o) = &args[0] else {
        panic!()
    };
    o.to_f64_vec()
}

const USE: &str = "void k(float* out, const float* in) { V<float, 4> v; v.fill(in[1]); v.put(3, in[3]); out[0] = v.at(2); out[1] = v.at(3); out[2] = (float)v.n(); }";

#[test]
fn the_primary_template_instantiates_with_its_arguments() {
    let src = format!(
        "template <typename T, int N> struct V {{ T data[N]; __device__ void fill(T x) {{ for (int i = 0; i < N; ++i) data[i] = x; }} \
         __device__ void put(int i, T x) {{ data[i] = x; }} __device__ T at(int i) {{ return data[i]; }} __device__ int n() {{ return N; }} }};\n{USE}"
    );
    assert_eq!(run(&lift(&src).unwrap(), 3), vec![2.5, 4.5, 4.0]);
}

#[test]
fn a_partial_specialisation_with_a_value_parameter_matches_and_binds_it() {
    // `V<float, n>` binds n = 4; the primary (declared only) would have no body at all
    let src = format!(
        "template <typename T, int N> struct V;\n\
         template <int N> struct V<float, N> {{ float data[N]; __device__ void fill(float x) {{ for (int i = 0; i < N; ++i) data[i] = x * 2.0f; }} \
         __device__ void put(int i, float x) {{ data[i] = x; }} __device__ float at(int i) {{ return data[i]; }} __device__ int n() {{ return N; }} }};\n{USE}"
    );
    // fill doubles (5.0), put overrides slot 3 with 4.5
    assert_eq!(run(&lift(&src).unwrap(), 3), vec![5.0, 4.5, 4.0]);
}

#[test]
fn the_more_specific_specialisation_wins_and_an_unmatched_use_declines() {
    let both = "template <typename T, int N> struct V { T data[N]; __device__ void fill(T x) { data[0] = x; } __device__ void put(int i, T x) { data[i] = x; } __device__ T at(int i) { return data[i]; } __device__ int n() { return 1; } };\n\
        template <int N> struct V<float, N> { float data[N]; __device__ void fill(float x) { data[0] = x; } __device__ void put(int i, float x) { data[i] = x; } __device__ float at(int i) { return data[i]; } __device__ int n() { return 2; } };\n\
        template <> struct V<float, 4> { float data[4]; __device__ void fill(float x) { data[0] = x; } __device__ void put(int i, float x) { data[i] = x; } __device__ float at(int i) { return data[i]; } __device__ int n() { return 3; } };\n";
    let src = format!("{both}{USE}");
    // the explicit V<float, 4> (n = 3) beats the partial (n = 2) and the primary (n = 1)
    let out = run(&lift(&src).unwrap(), 3);
    assert_eq!(out[2], 3.0);
    // a use that matches no definition (V<double, 4> with only a float partial and no primary) declines
    let only_partial = "template <typename T, int N> struct V;\ntemplate <int N> struct V<float, N> { float data[N]; };\n\
        void k(float* out, const float* in) { V<double, 4> v; out[0] = 1.0f; }";
    assert!(matches!(
        lift(only_partial),
        Err(KernelLiftError::NotYetLifted { .. })
    ));
}

#[test]
fn a_kernels_own_template_arguments_reach_the_class_template() {
    let src = "template <typename T, int N> struct V { T data[N]; __device__ void fill(T x) { for (int i = 0; i < N; ++i) data[i] = x; } __device__ T at(int i) { return data[i]; } };\n\
        template <int M> void k(float* out, const float* in) { V<float, M> v; v.fill(in[2]); out[0] = v.at(M - 1); }";
    let ctx = LiftContext::new(LaunchShape::new([1, 1, 1], [1, 1, 1])).with_const("M", 3);
    let k = lift_kernel_in(&C, src, "k", &ctx).unwrap();
    assert_eq!(run(&k, 1), vec![3.5]);
}
