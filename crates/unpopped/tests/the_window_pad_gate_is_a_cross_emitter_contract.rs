//! `2*pad <= span` is not an internal validation — it is a contract two emitters
//! rest on, and relaxing it makes dead code in another repository go live.
//!
//! # Why this needs a test at all
//!
//! `plan.rs` asserts every window overlaps the input by at least one tap. Pad taps
//! are SKIPPED for `Max`/`Min`, so that assertion is what guarantees a fold always
//! has a real element — which is why [`unpopped::ir::ReduceOp`] can say `Max`/`Min`
//! *"peel the first element, so no ±∞ literal"*.
//!
//! ⚠️ **baracuda pins this crate, so the assertion governs their CUDA emitter too.**
//! They measured it 2026-09-06: `window_simple(size=3, pad_lo=3)` panics on it, so
//! their all-pad branch is unreachable and their monoid-identity arm is dead.
//!
//! **Relaxing this constraint therefore un-deadens code in a repository this one
//! cannot see, silently — nothing in their tests would notice.** Until today the
//! gate had NO test here at all, so a relaxation would have been a one-line edit
//! with no signal anywhere.

use unpopped::ir::{OpDef, ReduceOp};
use unpopped::plan::build_plan;
use unpopped_vocab::{ArchSku, ElementKind, OpCategory, OperandDesc, structure_key};

/// `in_len = 4`, `size = 2`, `stride = 2`, `dilation = 1`.
fn window(pad_lo: u8, out_len: i64) -> (OpDef, [OperandDesc; 2]) {
    let op = OpDef::window_simple(
        "pool",
        &[ElementKind::F32],
        ReduceOp::Max,
        1,
        2,
        2,
        1,
        pad_lo,
        0,
        false,
    );
    let ind = OperandDesc::new(2, &[1, 4], &[4, 1], ElementKind::F32, 4);
    let outd = OperandDesc::new(2, &[1, out_len], &[out_len, 1], ElementKind::F32, 4);
    (op, [ind, outd])
}

#[test]
#[should_panic(expected = "exceeds half the window span")]
fn a_window_padded_past_half_its_span_is_rejected() {
    // 2*pad_lo = 4 > span 2. out_len = (4 + 2 - 1 - 1)/2 + 1 = 3.
    let (op, ops) = window(2, 3);
    let key = structure_key(OpCategory::Pooling, &ops, ArchSku::Sm89);
    let _ = build_plan(&op, &key);
}

/// The control. Without it the test above passes if `build_plan` panics for ANY
/// reason — a wrong extent, a bad category, an unrelated assertion — and the
/// `expected` string is the only thing standing between those and a false green.
#[test]
fn the_same_window_within_the_bound_builds() {
    // 2*pad_lo = 2 <= span 2. out_len = (4 + 1 - 1 - 1)/2 + 1 = 2.
    let (op, ops) = window(1, 2);
    let key = structure_key(OpCategory::Pooling, &ops, ArchSku::Sm89);
    let plan = build_plan(&op, &key);
    assert_eq!(
        plan.n_inputs, 1,
        "the control must actually BUILD — if this ever starts failing, the \
         should_panic above is passing for a reason that has nothing to do with \
         the pad bound"
    );
}
