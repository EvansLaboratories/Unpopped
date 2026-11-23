// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Census wall found by real instantiations: a trait or functor class reached through namespaces** (`typename
//! baracuda::reduce_to::AccSum<T>::AccT`, `baracuda::elementwise::AnyReduce<T>::init()`; ~100 instantiations). Namespaces are not
//! scopes the walker keeps: `a::b::Acc__t0::init` is the struct's static member, `typename a::b::Acc__t0::AccT` its member type,
//! and a local alias of the qualified name reaches both. A name that is not in the source still declines. Oracle runs.

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

const PRE: &str = "namespace a { namespace b { template <typename T> struct Acc { using AccT = float; \
    static AccT init() { return 1.5f; } static AccT comb(AccT x, T y) { return x + (AccT)y * 2.0f; } }; } }\n";

fn lift(body: &str) -> Result<KernelFn, KernelLiftError> {
    lift_kernel_in(
        &C,
        &format!("{PRE}{body}"),
        "k",
        &LiftContext::new(LaunchShape::new([1, 1, 1], [1, 1, 1])).with_type("T", "float"),
    )
}

fn run(k: &KernelFn) -> f64 {
    let mut args = vec![
        KernelArg::Buffer(TypedBuffer::from_f32(&[1], &[0.0])),
        KernelArg::Buffer(TypedBuffer::from_f32(&[1], &[4.0])),
    ];
    run_launch(k, &mut args).expect("runs");
    let KernelArg::Buffer(o) = &args[0] else {
        panic!()
    };
    o.to_f64_vec()[0]
}

#[test]
fn a_qualified_static_member_member_type_and_alias_reach_the_struct() {
    let k = lift("template <typename T> void k(T* out, const T* in) { float m = a::b::Acc<T>::init(); out[0] = m; }").unwrap();
    assert_eq!(run(&k), 1.5);
    let k = lift("template <typename T> void k(T* out, const T* in) { typename a::b::Acc<T>::AccT m = 2.0f; out[0] = m; }").unwrap();
    assert_eq!(run(&k), 2.0);
    let k = lift(
        "template <typename T> void k(T* out, const T* in) { using R = a::b::Acc<T>; typename R::AccT m = R::init(); \
         m = R::comb(m, in[0]); out[0] = m; }",
    )
    .unwrap();
    // 1.5 + 4 * 2
    assert_eq!(run(&k), 9.5);
}

#[test]
fn a_qualified_name_that_is_not_in_the_source_declines() {
    let r = lift(
        "template <typename T> void k(T* out, const T* in) { out[0] = a::b::Acc<T>::missing(); }",
    );
    assert!(
        matches!(r, Err(KernelLiftError::NotYetLifted { .. })),
        "{r:?}"
    );
}
