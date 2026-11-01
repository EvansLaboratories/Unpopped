// SPDX-License-Identifier: MIT OR Apache-2.0
//! ROADMAP E18 (openclownfish X1): the exact OpenCL C spellings its parser meets, as test input.
//! A `Mean` post, an exclusive scan and a reverse scan lift to the IR forms that mean the same
//! thing; a near-miss of each (a divisor that is not the loop bound, a store that is neither
//! before nor after the update) must still decline rather than lift as the wrong op.

#![cfg(feature = "convert")]

use unpopped::convert::{Frontend, lift_reduction_with, lift_scan_with};
use unpopped::artifact::{BindingRole, Packing};
use unpopped::ir::{OpDef, ReduceOp, input, reduced};
use unpopped::resource_map::{DeclaredBuffer, ResourceMap};
use unpopped::LiftError;
use unpopped_vocab::ElementKind as K;

const F32: &[K] = &[K::F32];

fn parse(src: &str) -> Option<tree_sitter::Tree> {
    let mut parser = tree_sitter::Parser::new();
    parser
        .set_language(&tree_sitter_cpp::LANGUAGE.into())
        .ok()?;
    parser.parse(src, None)
}

/// An OpenCL C front end for the tests: `in`/`out` are named by the resource map below.
const OPENCL: Frontend = Frontend::new("opencl-c", "__kernel", &[], "out", "in", parse);

fn map() -> ResourceMap {
    let b = |n: &str, i: u32, role| DeclaredBuffer::new(n, 0, i, role, K::F32, Packing::Unpacked);
    ResourceMap::new(
        vec![b("out", 0, BindingRole::Write), b("in", 1, BindingRole::Read)],
        None,
    )
}

fn kernel(body: &str) -> String {
    format!(
        "__kernel void f(__global const float* in, __global float* out, long n, long k) {{ {body} }}"
    )
}

fn scan(reverse: bool, exclusive: bool) -> OpDef {
    OpDef::scan("f", 1, F32, ReduceOp::Sum, 0, reverse, exclusive, input(0), reduced(0))
}

fn lift_scan(body: &str) -> Result<OpDef, LiftError> {
    lift_scan_with(&OPENCL, &kernel(body), "f", F32, Some(&map())).map(|l| l.op)
}

fn lift_reduce(body: &str) -> Result<OpDef, LiftError> {
    lift_reduction_with(&OPENCL, &kernel(body), "f", F32, Some(&map())).map(|l| l.op)
}

#[test]
fn a_mean_post_lifts_as_a_mean_reduction() {
    let op = lift_reduce(
        "float acc = 0.0f; for (long i = 0; i < n; ++i) { acc += in[i]; } out[0] = acc / (float)n;",
    )
    .expect("Sum over the loop, divided by the loop bound, is a Mean");
    assert_eq!(op, OpDef::reduction("f", 1, F32, input(0), ReduceOp::Mean));
}

#[test]
fn a_divisor_that_is_not_the_loop_bound_is_not_a_mean() {
    let src = "float acc = 0.0f; for (long i = 0; i < n; ++i) { acc += in[i]; } out[0] = acc / (float)k;";
    assert!(lift_reduce(src).is_err(), "k is not the reduced extent");
}

#[test]
fn an_exclusive_scan_lifts_as_exclusive() {
    let op = lift_scan(
        "float acc = 0.0f; for (long i = 0; i < k; ++i) { out[i] = acc; acc += in[i]; }",
    )
    .expect("the store before the update is exclusive");
    assert_eq!(op, scan(false, true));
}

#[test]
fn a_reverse_inclusive_scan_lifts_as_reverse() {
    let op = lift_scan(
        "float acc = 0.0f; for (long i = k - 1; i >= 0; --i) { acc += in[i]; out[i] = acc; }",
    )
    .expect("a descending loop is a reverse scan");
    assert_eq!(op, scan(true, false));
}

#[test]
fn the_forward_inclusive_scan_is_unchanged() {
    let op = lift_scan(
        "float acc = 0.0f; for (long i = 0; i < k; ++i) { acc += in[i]; out[i] = acc; }",
    )
    .expect("forward inclusive still lifts");
    assert_eq!(op, scan(false, false));
}
