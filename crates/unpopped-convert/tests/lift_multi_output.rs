// SPDX-License-Identifier: MIT OR Apache-2.0
//! ROADMAP U67: a kernel that stores several outputs (`output0`, `output1`, ...), one of them a `u8`
//! mask, lifts to one multi-output `OpDef` that the oracle runs against Rust references. The
//! fishstringer-slang-emit 0.10.1 spelling (`output0..`, a `u8` output on the plan path) is taken from
//! the lane brief and is UNVERIFIED against that crate's output; the source here is written to that shape.

use unpopped::oracle::{TypedBuffer, evaluate};
use unpopped::plan::try_build_plan;
use unpopped::{LiftError, Lifted};
use unpopped_convert::{Frontend, lift_elementwise, lift_elementwise_multi};
use unpopped_vocab::{ArchSku, ElementKind as K, OpCategory, OperandDesc, try_structure_key};

fn parse(src: &str) -> Option<tree_sitter::Tree> {
    let mut parser = tree_sitter::Parser::new();
    parser
        .set_language(&tree_sitter_cpp::LANGUAGE.into())
        .ok()?;
    parser.parse(src, None)
}

const FE: Frontend = Frontend::new("cuda-like", "__global__", &[], "output", "input", parse);
const F32: &[K] = &[K::F32];
const A: [f32; 6] = [1.0, -2.0, 0.5, 3.0, -0.25, 8.0];
const B: [f32; 6] = [2.0, -2.0, 0.25, 4.0, 1.0, -8.0];

fn kernel(stores: &str) -> String {
    format!(
        "__global__ void f(const float* input0, const float* input1, float* output0, \
         unsigned char* output1, float* output2, int n) {{ \
         int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) {{ {stores} }} }}"
    )
}

fn lift(stores: &str, outs: &[Option<K>]) -> Result<Lifted, LiftError> {
    lift_elementwise_multi(&FE, &kernel(stores), "f", F32, outs)
}

/// Run the lifted op over `A` and `B` on the oracle; one byte vector per output.
fn run(lifted: &Lifted, out_kinds: &[K]) -> Vec<Vec<u8>> {
    let d = |k| OperandDesc::new(1, &[6], &[1], k, 256);
    let mut ops = vec![d(K::F32), d(K::F32)];
    ops.extend(out_kinds.iter().map(|k| d(*k)));
    let key = try_structure_key(OpCategory::BinaryElementwise, &ops, ArchSku::Sm89).expect("a key");
    let plan = try_build_plan(&lifted.op, &key).expect("the lifted op plans");
    let input = |v: &[f32]| TypedBuffer::from_f32(&[6], v);
    evaluate(&plan, &ops, &[input(&A), input(&B)], &[])
        .into_iter()
        .map(|b| b.bytes)
        .collect()
}

fn f32s(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

#[test]
fn two_outputs_lift_and_match_the_rust_references() {
    let l = lift(
        "output0[i] = input0[i] * input1[i]; output1[i] = input0[i] + input1[i];",
        &[None, None],
    )
    .expect("two declared outputs, two stores");
    assert_eq!(l.op.n_outputs(), 2);
    let got = run(&l, &[K::F32, K::F32]);
    let prod: Vec<f32> = A.iter().zip(&B).map(|(a, b)| a * b).collect();
    let sum: Vec<f32> = A.iter().zip(&B).map(|(a, b)| a + b).collect();
    assert_eq!(f32s(&got[0]), prod, "output0");
    assert_eq!(f32s(&got[1]), sum, "output1 (the second store)");
}

#[test]
fn a_u8_mask_output_beside_a_float_output_matches_the_reference() {
    let l = lift(
        "output0[i] = input0[i] + input1[i]; output1[i] = input0[i] < input1[i];",
        &[None, Some(K::U8)],
    )
    .expect("a comparison root is a legal u8 output");
    let got = run(&l, &[K::F32, K::U8]);
    let sum: Vec<f32> = A.iter().zip(&B).map(|(a, b)| a + b).collect();
    let mask: Vec<u8> = A.iter().zip(&B).map(|(a, b)| u8::from(a < b)).collect();
    assert_eq!(f32s(&got[0]), sum);
    assert_eq!(got[1], mask, "the u8 mask has both 0 and 1 cells");
    assert!(mask.contains(&0) && mask.contains(&1), "corpus not blind");
}

#[test]
fn a_declared_output_with_no_store_is_refused() {
    let one = "output0[i] = input0[i] * input1[i];";
    assert!(lift(one, &[None]).is_ok(), "control: one store, one output");
    let e = lift(one, &[None, None]).expect_err("output1 is declared and never stored");
    assert!(matches!(e, LiftError::Unrecognized(m) if m.contains("output1")));
}

#[test]
fn a_store_to_an_undeclared_output_is_refused_not_dropped() {
    let two = "output0[i] = input0[i]; output1[i] = input1[i];";
    assert!(lift(two, &[None, None]).is_ok(), "control: all declared");
    let e = lift(&format!("{two} output2[i] = input0[i];"), &[None, None])
        .expect_err("output2 is stored but not declared");
    assert!(matches!(e, LiftError::Unrecognized(m) if m.contains("output2")));
}

#[test]
fn stores_at_different_indices_are_refused() {
    let ok = "output0[i] = input0[i]; output1[i] = input1[i];";
    assert!(lift(ok, &[None, None]).is_ok(), "control: one index");
    let e = lift(
        "output0[i] = input0[i]; output1[j] = input1[i];",
        &[None, None],
    )
    .expect_err("two indices");
    assert!(matches!(e, LiftError::Unrecognized(m) if m.contains("one index")));
}

#[test]
fn no_declared_outputs_is_refused_and_the_single_store_lift_is_unchanged() {
    assert!(lift("output0[i] = input0[i];", &[]).is_err());
    // the single-output lift still reads `output[i]` (no suffix)
    let src = "__global__ void f(const float* input0, float* output, int n) { int i = 0; output[i] = input0[i]; }";
    let l = lift_elementwise(&FE, src, "f", F32).expect("unchanged");
    assert_eq!(l.op.n_outputs(), 1);
}
