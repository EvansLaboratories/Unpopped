//! A block's shared scale is a **sibling operand**, and that is what closes the
//! quantization key collision (sk4 §3.2).
//!
//! # What this replaces
//!
//! `OperandDesc` used to carry `quant: Option<QuantFacts>` — family, sub-byte
//! bits, block extent, scale placement — hanging off the operand it described.
//! Nothing encoded it into the token, so two operands differing only in
//! quantization derived byte-identical keys. Under KISS-CLASSIFY §6.8-0002
//! (byte-exact matching, subset and implication logic forbidden) that collision
//! does not degrade into a slower correct match: the consumer serves whichever
//! kernel it holds under that token, and two operands with different scale
//! groupings are *different math*. A silent wrong answer.
//!
//! sk4 settles the model the other way. The **element dtype** and the **block
//! structure** are separate axes, and the shared scale is its own entry in the
//! operand list. A Q4 weight is therefore two operands; a bare `i4` tensor is
//! one. They differ in the operand list itself.
//!
//! # Why this needs no schema event
//!
//! Operands already reach the key — count, dtype, rank, strides, all of it. So
//! expressing quantization as an operand rather than as a field uses machinery
//! that already exists and already keys. Nothing about the token grammar
//! changes. That is the whole argument for doing it now rather than at sk5, and
//! these tests are what make it a measurement instead of a claim.

use unpopped_vocab::{ArchSku, ElementKind, OpCategory, OperandDesc, StructureKey, structure_key};

/// A packed 4-bit weight buffer: 1024 logical elements.
fn packed_i4(n: i64) -> OperandDesc {
    OperandDesc::new(1, &[n], &[1], ElementKind::I4, 256)
}

/// The shared-scale sibling for `n / block` blocks.
///
/// `f8e8m0` is the OCP microscaling scale type — KISS-CLASSIFY §6.1-0013 names
/// the MX scales as sibling operands, which is the same shape sk4 §3.2 settles
/// for quantization generally.
fn scale_sibling(blocks: i64) -> OperandDesc {
    OperandDesc::new(1, &[blocks], &[1], ElementKind::F8E8M0, 256)
}

fn token(operands: &[OperandDesc]) -> String {
    structure_key(OpCategory::Gemm, operands, ArchSku::Sm89).to_token()
}

/// **The collision is closed:** a quantized operand and a bare one differ.
///
/// This is the assertion the old `quant` field could not satisfy. Same element
/// dtype (`i4`), same shape, same strides — the *only* difference is that one
/// carries a scale sibling. Under the field model these produced byte-identical
/// tokens; under the sibling model they cannot, because the operand list itself
/// differs.
#[test]
fn a_scale_sibling_makes_a_quantized_operand_a_different_cell() {
    let bare = token(&[packed_i4(1024), packed_i4(1024)]);
    let quantized = token(&[packed_i4(1024), scale_sibling(32), packed_i4(1024)]);

    assert_ne!(
        bare, quantized,
        "an i4 tensor and a Q4 tensor are different math and must not share a token"
    );
    // And the difference is visible in the operand count, which is the mechanism
    // rather than an accident of some other coordinate moving.
    let bare_k = StructureKey::from_token(&bare).expect("round trip");
    let quant_k = StructureKey::from_token(&quantized).expect("round trip");
    assert_eq!(bare_k.n_operands, 2);
    assert_eq!(quant_k.n_operands, 3);
}

/// **The scale's DTYPE does not reach the key** — measured, not assumed.
///
/// An MX block scale (`f8e8m0`, a power-of-two exponent) and an NF4-style absmax
/// scale (`f16`) are different dequantization arithmetic, and they produce
/// byte-identical tokens. The per-operand sub-key carries contiguity, vector
/// width and a divisibility bucket — **not** the operand's dtype. Only operand
/// 0's dtype reaches the token, as the key's primary dtype field.
///
/// I wrote this test asserting the opposite and it failed, which is the reason
/// it is worth having: "the sibling model closes the collision" is true for
/// *presence* and false for *scale dtype*, and the difference is not visible
/// from the design description.
///
/// Measured sub-keys at 32 blocks over a 1024-element `i4` weight:
///
/// ```text
/// f8e8m0 -> co/00/v1/d16/f;co/00/v8/d16/f;co/00/v1/d16/f
/// f16    -> co/00/v1/d16/f;co/00/v8/d16/f;co/00/v1/d16/f   (identical)
/// f32    -> co/00/v1/d16/f;co/00/v4/d16/f;co/00/v1/d16/f   (differs — but see below)
/// ```
///
/// `f32` differing is **not** the dtype being keyed: it is the vector-width
/// bucket moving because a 4-byte element vectorizes differently at the same
/// alignment. `f8e8m0` (1 byte) and `f16` (2 bytes) both saturate at `v8`, so
/// even the width leak does not separate the pair that matters.
#[test]
fn the_scale_dtype_does_not_reach_the_key_sk5_item() {
    let mx = token(&[packed_i4(1024), scale_sibling(32), packed_i4(1024)]);
    let absmax = token(&[
        packed_i4(1024),
        OperandDesc::new(1, &[32], &[1], ElementKind::F16, 256),
        packed_i4(1024),
    ]);
    assert_eq!(
        mx, absmax,
        "KNOWN RESIDUAL (sk5): an e8m0 block scale and an f16 absmax scale are          different dequant math and must eventually differ. If this now FAILS,          the gap closed — update this test rather than deleting it, and say which          schema event closed it."
    );
}

/// **Block granularity collides once the counts are large enough**, which is the
/// realistic case.
///
/// Both buckets that could distinguish a scale sibling's extent **saturate**:
/// the divisibility bucket at `d16` and the vector-width bucket at `v8`. So any
/// two block counts ≥ 16 produce identical sub-keys, and the real granularities
/// (32, 64, 128, 256 blocks) are all in that range.
///
/// Measured over a 1024-element `i4` weight with an `f8e8m0` scale:
///
/// ```text
/// blocks=  4 -> co/00/v4/d4/f      distinct
/// blocks=  8 -> co/00/v8/d8/f      distinct
/// blocks= 16 -> co/00/v8/d16/f  ┐
/// blocks= 32 -> co/00/v8/d16/f  │
/// blocks= 64 -> co/00/v8/d16/f  ├ all identical
/// blocks=128 -> co/00/v8/d16/f  │
/// blocks=256 -> co/00/v8/d16/f  ┘
/// ```
///
/// My first version of this test compared 32 against 8 and PASSED — the one pair
/// in the realistic range that happens to straddle the saturation point. That is
/// exactly the shape of a test that certifies nothing while looking green, so
/// the comparison is now between two counts that both matter (32 and 128) and
/// the saturation is asserted directly rather than sampled.
#[test]
fn block_granularity_collides_above_the_bucket_saturation_sk5_item() {
    // 1024 elements at block 32 -> 32 scales; at block 8 -> 128 scales.
    let blk32 = token(&[packed_i4(1024), scale_sibling(32), packed_i4(1024)]);
    let blk128 = token(&[packed_i4(1024), scale_sibling(128), packed_i4(1024)]);
    assert_eq!(
        blk32, blk128,
        "KNOWN RESIDUAL (sk5): two block granularities are different math and          must eventually differ. If this now FAILS, the gap closed — update this          test rather than deleting it."
    );

    // The saturation itself, so the residual is characterised rather than
    // sampled: everything from 16 up is one bucket.
    let saturated: Vec<String> = [16i64, 32, 64, 128, 256]
        .iter()
        .map(|&b| token(&[packed_i4(1024), scale_sibling(b), packed_i4(1024)]))
        .collect();
    assert!(
        saturated.windows(2).all(|w| w[0] == w[1]),
        "the bucket saturation is the mechanism; if these ever differ, the          residual's shape changed and the doc above is stale"
    );

    // And the control: BELOW saturation they do differ, so the collision above
    // is saturation rather than the extent being ignored outright.
    let small = token(&[packed_i4(1024), scale_sibling(4), packed_i4(1024)]);
    assert_ne!(
        small, blk32,
        "4 and 32 blocks fall in different buckets and must differ — without          this, the equality above is consistent with extent being unkeyed"
    );
}

/// Positive control: the harness can tell tokens apart at all.
///
/// Without it, `block_granularity_still_collides` is equally consistent with
/// "every token is identical", which would be a broken harness rather than a
/// real residual — the same defect the `operand_facts_reach_the_key` tripwire
/// guards against.
#[test]
fn the_harness_distinguishes_tokens_it_should() {
    let a = token(&[packed_i4(1024), packed_i4(1024)]);
    let b = token(&[
        OperandDesc::new(1, &[1024], &[1], ElementKind::F32, 256),
        OperandDesc::new(1, &[1024], &[1], ElementKind::F32, 256),
    ]);
    assert_ne!(
        a, b,
        "different element dtypes must produce different tokens"
    );
}
