//! Tripwire: which [`OperandDesc`] fields actually reach the structure key.
//!
//! `OperandDesc` carries two optional fact bundles — `quant` and `symbolic` —
//! that **nothing in this workspace reads**, and that the key codec does not
//! encode. They are set to `None` by `OperandDesc::new`, no constructor sets
//! them, and a workspace-wide grep for `.quant` / `.symbolic` returns no reads.
//!
//! That makes them a trap rather than dead weight. The fields are public and
//! documented as part of "the minimal per-operand description `structure_key`
//! reads", so a consumer can populate them, get no error, and reasonably believe
//! the resulting key distinguishes what they described. It does not.
//!
//! Why that matters more here than it would elsewhere: KISS-CLASSIFY-6.8-0002
//! specifies byte-exact token matching with subset/implication logic **forbidden**.
//! A key collision therefore does not degrade into a slower-but-correct match —
//! the consumer serves whichever kernel it has under that token. Two operands
//! whose only difference is their quantization block size are *different math*
//! (different scale groupings), so the failure is a silent wrong answer.
//!
//! These tests assert the CURRENT behaviour, deliberately. They are not an
//! endorsement of it. Encoding these facts into the key changes the token
//! grammar, which is a `structure_key` schema event and goes through KISS
//! alongside sk4/sk5 — not a quiet patch here. When that lands, these tests fail
//! and whoever lands it updates them on purpose. That is the point: the gap is
//! visible and tripwired instead of discovered by a consumer in production.

use unpopped_vocab::{
    ArchSku, ElementKind, OpCategory, OperandDesc, QuantFacts, QuantFamily, ScalePlacement,
    SymExtent, SymKind, structure_key,
};

fn key_token(d: OperandDesc) -> String {
    structure_key(OpCategory::UnaryElementwise, &[d, d], ArchSku::Sm89).to_token()
}

/// Quantization facts do NOT reach the key — a known gap, pending a schema event.
///
/// All three of these describe genuinely different kernels: an unquantized S4
/// buffer, a Q4 buffer with 32-element scale blocks, and a Q4 buffer with
/// 128-element scale blocks. They share one token.
#[test]
fn quant_facts_do_not_reach_the_key_known_gap() {
    let mk = |q| {
        let mut d = OperandDesc::new(1, &[1024], &[1], ElementKind::I4, 256);
        d.quant = q;
        d
    };
    let blocked = |elems| {
        Some(QuantFacts {
            family: QuantFamily::AffineBlock,
            sub_byte_bits: 4,
            block_elems: elems,
            scale: ScalePlacement::SeparateBuffer,
        })
    };

    let plain = key_token(mk(None));
    let blk32 = key_token(mk(blocked(32)));
    let blk128 = key_token(mk(blocked(128)));

    assert_eq!(
        plain, blk32,
        "KNOWN GAP: unquantized and Q4/block-32 must eventually differ"
    );
    assert_eq!(
        blk32, blk128,
        "KNOWN GAP: two block sizes are different math and must eventually differ"
    );
}

/// Symbolic-extent facts do NOT reach the key — the same known gap.
///
/// A live-vs-capacity axis is exactly the kind of fact a kernel specializes on;
/// two operands differing only in which axis is symbolic share one token.
#[test]
fn symbolic_extent_does_not_reach_the_key_known_gap() {
    let mk = |s| {
        let mut d = OperandDesc::new(2, &[64, 128], &[128, 1], ElementKind::F32, 256);
        d.symbolic = s;
        d
    };

    let dense = key_token(mk(None));
    let sym_axis0 = key_token(mk(Some(SymExtent {
        axis: 0,
        kind: SymKind::Range,
    })));

    assert_eq!(
        dense, sym_axis0,
        "KNOWN GAP: a symbolic axis must eventually be distinguishable in the key"
    );
}

/// The fields that DO reach the key, as a positive control.
///
/// Without this, the two tests above are consistent with "the key ignores
/// everything", which would point at a broken harness rather than a real gap.
#[test]
fn layout_facts_do_reach_the_key() {
    let contiguous = OperandDesc::new(2, &[64, 128], &[128, 1], ElementKind::F32, 256);
    let broadcast = OperandDesc::new(2, &[64, 128], &[0, 1], ElementKind::F32, 256);
    let flipped = OperandDesc::new(2, &[64, 128], &[128, -1], ElementKind::F32, 256);

    assert_ne!(
        key_token(contiguous),
        key_token(broadcast),
        "a broadcast axis must change the key"
    );
    assert_ne!(
        key_token(contiguous),
        key_token(flipped),
        "a flipped (negative-stride) axis must change the key"
    );
}
