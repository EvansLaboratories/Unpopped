// SPDX-License-Identifier: MIT OR Apache-2.0
//! ROADMAP U67 against the REAL spelling: the two kernels below are the verbatim text that
//! `fishstringer-slang-emit` (checkout `46988a72`, origin/main of fishstringer, the multi-output path of
//! fishstringer#58) emits for `tests/lower_multi_output.rs`'s `sum_and_product` and `dropout_like`
//! (generated with `try_generate_with(.., &Slang, &CallerCaps::LEGACY.with_interface())`), header comments
//! dropped. Each is lifted, planned and run on the oracle against a Rust reference.

use unpopped::oracle::{TypedBuffer, evaluate};
use unpopped::plan::try_build_plan;
use unpopped::{LiftError, Lifted};
use unpopped_convert::{Frontend, lift_elementwise_multi};
use unpopped_vocab::{ArchSku, ElementKind as K, OpCategory, OperandDesc, try_structure_key};

fn parse(src: &str) -> Option<tree_sitter::Tree> {
    let mut parser = tree_sitter::Parser::new();
    parser
        .set_language(&tree_sitter_slang::LANGUAGE_SLANG.into())
        .ok()?;
    parser.parse(src, None)
}

/// fishstringer-slang-parse's `SLANG` frontend, without its entry-model reader (not needed to lift a body).
const SLANG: Frontend = Frontend::new(
    "slang",
    "numthreads",
    &["groupshared", "InterlockedAdd"],
    "output",
    "input",
    parse,
);

const SUM_AND_PRODUCT: &str = "\
[[vk::binding(0, 0)]] StructuredBuffer<float> input0;
[[vk::binding(1, 0)]] StructuredBuffer<float> input1;
[[vk::binding(2, 0)]] RWStructuredBuffer<float> output0;
[[vk::binding(3, 0)]] RWStructuredBuffer<float> output1;
struct Params_
{
    uint n;
};
[[vk::binding(4, 0)]] ConstantBuffer<Params_> pb_;
[numthreads(256, 1, 1)]
void unpopped_slang_sp_f32(uint3 tid : SV_DispatchThreadID) {
    uint i = tid.x;
    if (i >= pb_.n) {
        return;
    }
    float tmp0 = input0[i];
    float tmp1 = input1[i];
    output0[i] = (tmp0 + tmp1);
    output1[i] = (tmp0 * tmp1);
}
";

const DROPOUT_LIKE: &str = "\
[[vk::binding(0, 0)]] StructuredBuffer<float> input0;
[[vk::binding(1, 0)]] StructuredBuffer<float> input1;
[[vk::binding(2, 0)]] RWStructuredBuffer<float> output0;
[[vk::binding(3, 0)]] RWStructuredBuffer<uint8_t> output1;
struct Params_
{
    uint n;
};
[[vk::binding(4, 0)]] ConstantBuffer<Params_> pb_;
[numthreads(256, 1, 1)]
void unpopped_slang_dropout_like_f32(uint3 tid : SV_DispatchThreadID) {
    uint i = tid.x;
    if (i >= pb_.n) {
        return;
    }
    float tmp0 = ((float)input1[i] < (float)0.5 ? 1.0 : 0.0);
    output0[i] = (input0[i] * (((float)(tmp0)) != 0.0 ? (float)(2.0) : (float)(0.0)));
    output1[i] = (uint8_t)tmp0;
}
";

const A: [f32; 6] = [1.0, -2.0, 0.5, 3.0, -0.25, 8.0];
const B: [f32; 6] = [0.25, 0.75, 0.5, 0.1, 0.9, 0.0];

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

fn lift(src: &str, outs: &[Option<K>]) -> Result<Lifted, LiftError> {
    lift_elementwise_multi(&SLANG, src, "f", &[K::F32], outs)
}

#[test]
fn the_real_two_float_output_kernel_lifts_and_runs() {
    let l = lift(SUM_AND_PRODUCT, &[None, None]).expect("the emitted sum_and_product lifts");
    let got = run(&l, &[K::F32, K::F32]);
    let sum: Vec<f32> = A.iter().zip(&B).map(|(a, b)| a + b).collect();
    let prod: Vec<f32> = A.iter().zip(&B).map(|(a, b)| a * b).collect();
    assert_eq!(f32s(&got[0]), sum);
    assert_eq!(f32s(&got[1]), prod, "the second store");
}

#[test]
fn the_real_dropout_like_kernel_lifts_with_a_u8_mask_and_runs() {
    let l = lift(DROPOUT_LIKE, &[None, Some(K::U8)]).expect("the emitted dropout_like lifts");
    let got = run(&l, &[K::F32, K::U8]);
    let keep: Vec<bool> = B.iter().map(|b| *b < 0.5).collect();
    let value: Vec<f32> = A
        .iter()
        .zip(&keep)
        .map(|(a, k)| a * if *k { 2.0 } else { 0.0 })
        .collect();
    let mask: Vec<u8> = keep.iter().map(|k| u8::from(*k)).collect();
    assert_eq!(f32s(&got[0]), value);
    assert_eq!(got[1], mask);
    assert!(mask.contains(&0) && mask.contains(&1), "corpus not blind");
}

#[test]
fn dropping_the_second_store_of_the_real_kernel_is_refused() {
    let one = SUM_AND_PRODUCT.replace("    output1[i] = (tmp0 * tmp1);\n", "");
    assert_ne!(one, SUM_AND_PRODUCT, "the mutation applied");
    assert!(lift(&one, &[None, None]).is_err(), "output1 is declared");
    assert!(lift(&one, &[None]).is_ok(), "control: one declared output");
}

#[test]
fn only_a_cast_to_the_compute_type_is_the_identity() {
    let with = |ty: &str| {
        SUM_AND_PRODUCT.replace(
            "float tmp0 = input0[i];",
            &format!("float tmp0 = ({ty})input0[i];"),
        )
    };
    assert!(
        lift(&with("float"), &[None, None]).is_ok(),
        "control: (float) on f32"
    );
    let e = lift(&with("int"), &[None, None]).expect_err("(int) changes the value");
    assert!(matches!(e, LiftError::Unrecognized(m) if m.contains("cast to 'int'")));
    // a u8 store-site cast is stripped only on a declared u8 output
    let e =
        lift(DROPOUT_LIKE, &[None, None]).expect_err("(uint8_t) on a non-u8 output is a real cast");
    assert!(matches!(e, LiftError::Unrecognized(m) if m.contains("uint8_t")));
}
