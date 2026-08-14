//! Tripwire: which [`OperandDesc`] fields actually reach the structure key.
//!
//! `OperandDesc` carries one optional fact bundle — `symbolic` — that **nothing
//! in this workspace reads**, and that the key codec does not encode. It is set
//! to `None` by `OperandDesc::new`, no constructor sets it, and a workspace-wide
//! grep for `.symbolic` returns no reads.
//!
//! There used to be a second, `quant`, tested here the same way. It is **gone**:
//! sk4 §3.2 settled that a block's shared scale is a sibling operand rather than
//! a field, so the field was not merely un-keyed but the wrong shape. The
//! correct model needs no schema event and is proven in
//! `tests/scale_sibling_model.rs`. `symbolic` has no such successor — no decided
//! design exists for keying a live-vs-capacity axis — so it stays, and stays
//! tripwired.
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
    ArchSku, ElementKind, OpCategory, OperandDesc, SymExtent, SymKind, structure_key,
};

fn key_token(d: OperandDesc) -> String {
    structure_key(OpCategory::UnaryElementwise, &[d, d], ArchSku::Sm89).to_token()
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
    let sym_axis0 = key_token(mk(Some(SymExtent::new(0, SymKind::Range))));

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
