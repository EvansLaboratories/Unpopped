//! Zero is divisible by everything, so a modulo ladder must not read it as
//! "maximally divisible". Two instances of that one defect, both reachable through
//! the public `structure_key`:
//!
//! 1. A contraction with `K = 0` derived the K-divisibility bucket `d16`, not `da`.
//! 2. An operand with `align_bytes = 0` (unspecified alignment) derived a packed
//!    vector width, not `v1`.
//!
//! # Instance 1: `K = 0`
//!
//! `div_bucket` tested `e % 16 == 0` first, and `0 % 16 == 0`, so `E = 0` landed
//! in `d16`, the *most* divisible bucket. KISS-CLASSIFY §6.5-0012 buckets
//! `E = 0` as `da` (*"covering odd `E`, `E = 1`, and `E = 0`"*), and §6.6-0016
//! says the contraction's K bucket comes from the K extent through that clause.
//!
//! The per-operand sub-key reaches `div_bucket` only through `inner_axis`, which
//! selects an axis of extent `> 1`, so it never passes `0`. That made the bug
//! look unreachable: KISS's report of it was retracted on exactly that ground.
//! The contraction path calls `div_bucket(k)` on the raw K extent, with no
//! `inner_axis` in between. These tests go through the public `structure_key`,
//! so they measure what a consumer gets rather than what a helper does in
//! isolation.
//!
//! Measured before the fix, on published 0.12.0: `K = 0` derived `Div16`.
//!
//! # Instance 2: `align_bytes = 0`
//!
//! `classify_vec_width` gated each width on `align % (L * bytes) == 0`, which is
//! true for `align = 0`. §6.5-0009: *"An operand with `alignment = 0`
//! (unspecified base-pointer alignment) cannot honor a packed load and MUST
//! derive `v1`."* `OperandDesc::new` accepts `0`, and nothing upstream guarded it.

use unpopped_vocab::{
    ArchSku, DivBucket, ElementKind, OpCategory, OperandDesc, VecWidth, structure_key,
};

/// Dense row-major descriptor with EXACT strides. For a zero extent the outer
/// stride is genuinely 0 (`[4, 0]` is `[0, 1]`), which `classify_mat_layout`
/// accepts; a `max(1)` shortcut would yield a non-packed layout and decline the
/// contraction, making the K tests vacuous. That happened while measuring this.
fn dense(shape: &[i64]) -> OperandDesc {
    let mut strides = vec![1i64; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    OperandDesc::new(shape.len(), shape, &strides, ElementKind::F32, 256)
}

fn k_div(m: i64, k: i64, n: i64) -> DivBucket {
    let key = structure_key(
        OpCategory::Gemm,
        &[dense(&[m, k]), dense(&[k, n]), dense(&[m, n])],
        ArchSku::Sm89,
    );
    key.contraction
        .expect("a dense rank-2 GEMM must derive contraction facts, or this test asserts nothing")
        .k_div
}

#[test]
fn k_zero_is_da_not_d16() {
    assert_eq!(k_div(4, 0, 8), DivBucket::Any);
}

#[test]
fn k_zero_batched_is_da_not_d16() {
    let key = structure_key(
        OpCategory::Gemm,
        &[dense(&[2, 4, 0]), dense(&[2, 0, 8]), dense(&[2, 4, 8])],
        ArchSku::Sm89,
    );
    assert_eq!(
        key.contraction
            .expect("a dense rank-3 GEMM must derive contraction facts")
            .k_div,
        DivBucket::Any
    );
}

/// Positive controls: the ladder is unchanged for every non-zero K, so a fix
/// that returned `Any` unconditionally would fail here.
#[test]
fn nonzero_k_ladder_is_unchanged() {
    assert_eq!(k_div(4, 16, 8), DivBucket::Div16);
    assert_eq!(k_div(4, 32, 8), DivBucket::Div16);
    assert_eq!(k_div(4, 8, 8), DivBucket::Div8);
    assert_eq!(k_div(4, 12, 8), DivBucket::Div4);
    assert_eq!(k_div(4, 6, 8), DivBucket::Div2);
    assert_eq!(k_div(4, 7, 8), DivBucket::Any);
    assert_eq!(k_div(4, 1, 8), DivBucket::Any);
}

fn vec_width_of(shape: &[i64], align: u32) -> VecWidth {
    let od = OperandDesc::new(shape.len(), shape, &[1], ElementKind::F32, align);
    structure_key(OpCategory::UnaryElementwise, &[od, od], ArchSku::Sm89).operands[0].vec_width
}

#[test]
fn zero_alignment_derives_v1() {
    assert_eq!(vec_width_of(&[256], 0), VecWidth::Scalar);
}

/// Positive control: the same operand with a real alignment still vectorizes,
/// so a fix that forced `Scalar` everywhere would fail here.
#[test]
fn real_alignment_still_vectorizes() {
    assert_eq!(vec_width_of(&[256], 256), VecWidth::V4);
    assert_eq!(vec_width_of(&[256], 8), VecWidth::V2);
}
