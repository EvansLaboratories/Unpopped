// SPDX-License-Identifier: MIT OR Apache-2.0
//! Does `StructureKey` separate a `[N]` bias from a full `[M,N]` C? **Yes.**
//!
//! Asked by the PM (2026-11-03) before accumulate-into-C (ROADMAP U7 S1a-2) may name an
//! extra contraction input by its rank. The per-operand sub-key has no rank field, but it
//! has `bcast`, the axes that broadcast (stride 0) in the frame: a `[N]` bias is broadcast
//! over the M axis (mask `1`), a full `[M,N]` C is not (mask `0`). The two therefore key
//! differently, so a kernel cached for one is never found for the other.
//!
//! The one case where they key alike is `M == 1`, where `[N]` and `[1,N]` read the same
//! element at the same address, so sharing the kernel is correct. Pinned too.
//!
//! The first version of this test predicted the OPPOSITE (that the sub-key carries no rank
//! and the two would collide). It failed on its first assertion, and the failure is where
//! the `bcast` difference was read.

use unpopped_vocab::{structure_key, ArchSku, ElementKind, OpCategory, OperandDesc};

fn desc(rank: usize, shape: &[i64], strides: &[i64]) -> OperandDesc {
    OperandDesc::new(rank, shape, strides, ElementKind::F32, 256)
}

#[test]
fn a_dense_bias_and_a_dense_full_c_have_different_operand_sub_keys() {
    let lhs = desc(2, &[8, 4096], &[4096, 1]);
    let rhs = desc(2, &[4096, 4096], &[4096, 1]);
    let out = desc(2, &[8, 4096], &[4096, 1]);
    let bias = desc(1, &[4096], &[1]);
    let full_c = desc(2, &[8, 4096], &[4096, 1]);

    let kb = structure_key(OpCategory::Gemm, &[lhs, rhs, bias, out], ArchSku::Sm89);
    let kc = structure_key(OpCategory::Gemm, &[lhs, rhs, full_c, out], ArchSku::Sm89);

    // The positive control: the bias cell derives contraction facts and a token.
    assert!(kb.contraction.is_some());
    // The answer: the extra operand's sub-key differs, through `bcast`.
    assert_ne!(kb.operands[2], kc.operands[2]);
    assert_ne!(kb.operands[2].bcast, kc.operands[2].bcast);
    // Today the derivation also declines a rank-2 extra operand outright; admitting it
    // (S1a-2) changes that, and the sub-key difference above is what keeps the tokens apart.
    assert!(kc.contraction.is_none());
}

#[test]
fn at_m_equal_one_a_bias_and_a_full_c_read_the_same_element_and_key_alike() {
    let lhs = desc(2, &[1, 4096], &[4096, 1]);
    let rhs = desc(2, &[4096, 4096], &[4096, 1]);
    let out = desc(2, &[1, 4096], &[4096, 1]);
    let kb = structure_key(
        OpCategory::Gemm,
        &[lhs, rhs, desc(1, &[4096], &[1]), out],
        ArchSku::Sm89,
    );
    let kc = structure_key(
        OpCategory::Gemm,
        &[lhs, rhs, desc(2, &[1, 4096], &[4096, 1]), out],
        ArchSku::Sm89,
    );
    // Semantically identical: row 0 is the only row, so `C[0,n] == bias[n]`.
    assert_eq!(kb.operands[2], kc.operands[2]);
}
