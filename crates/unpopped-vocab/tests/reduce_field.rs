//! Field 8, the reduce spec — KISS-CLASSIFY-6.6-0009 / KISS-CLASSIFY-6.7-0005.
//!
//! # Why this file exists
//!
//! Until it did, **nothing pinned this field's spelling on the encode side.**
//! The decoder accepted all four forms and had a test saying so; the encoder
//! emitted `x<hh>` for every non-empty mask, including the two cases §6.7-0005
//! requires be spelled `rall` and `rlast`. The asymmetry was recorded in a
//! comment and excused as "byte-different but semantically identical", and the
//! whole suite stayed green — because a token that decodes to the right *mask*
//! round-trips through a decoder that also accepts the wrong *spelling*.
//!
//! That is the shape of the bug: semantic round-trip tests cannot see a spelling
//! divergence, because both spellings mean the same thing to the only reader
//! being asked. It takes an independently-authored byte to notice, which is what
//! a byte-match is for and what these goldens are.
//!
//! Every token below is KISS's, from `conformance/tests/structure_key_golden.rs`
//! at `origin/main` = `19c3ad7`.

use unpopped_vocab::{AxisMask, StructureKey};

/// KISS's four reduce-field goldens, re-encoded byte-for-byte.
///
/// §6.6-0009 requires four **distinctly-encoded** values and forbids overloading
/// one sentinel across two of them. The `x0a` case is the control: it proves the
/// fix added the two sentinels rather than replacing the bitmask form wholesale,
/// which would have been the same class of error pointing the other way.
#[test]
fn the_four_reduce_spellings_round_trip_byte_for_byte() {
    let goldens = [
        // rank 2, trailing axis only  → `rlast` (§6.6-0009 case 3)
        "sk4|red|f32|cuda:sm89|ix32|warp|r2|co/00/v1/d8/f;co/00/v1/da/f|rlast",
        // rank 2, every axis          → `rall`  (case 2)
        "sk4|red|f32|cuda:sm89|ix32|warp|r2|co/00/v1/d8/f;co/00/v1/da/f|rall",
        // rank 4, axes {1,3}          → `x0a`   (case 4 — neither all nor lone-trailing)
        "sk4|red|f32|cuda:sm89|ix32|block|r4|co/00/v1/da/f;co/00/v1/da/f|x0a",
        // not a reduction             → `-`     (case 1)
        "sk4|bin|f32|cuda:sm89|ix32|grid|r2|co/00/v4/d16/f;co/00/v4/d16/f;co/00/v4/d16/f|-",
    ];
    for g in goldens {
        let key = StructureKey::from_token(g).unwrap_or_else(|| panic!("golden must decode: {g}"));
        assert_eq!(key.to_token(), g, "byte-exact re-encode");
    }
}

/// The rank-1 tie-break: a rank-1 reduction's axis set is **simultaneously**
/// all-axes and the lone innermost axis, and §6.6-0009 pins `rall` as the winner
/// — "so two conforming implementations never disagree on the rank-1 encoding".
///
/// This is the single most common reduction shape there is, and the branch order
/// that decides it is invisible at every other rank. Its own test because a
/// swapped order is green everywhere else.
///
/// Golden: `a1_reduction_rank1_all_axes`.
#[test]
fn a_rank_1_reduction_spells_rall_not_rlast() {
    let g = "sk4|red|f32|cuda:sm89|ix32|warp|r1|co/00/v1/d8/f;co/00/v1/da/f|rall";
    let key = StructureKey::from_token(g).expect("golden must decode");
    assert_eq!(key.rank, 1);
    assert_eq!(
        key.reduce_axes,
        AxisMask(0b1),
        "rank-1: the one axis is reduced"
    );
    assert_eq!(
        key.to_token(),
        g,
        "§6.6-0009: `rall` takes precedence over `rlast` when the set is both"
    );
}

/// The encoder never emits the `x<hh>` form for a set that has a sentinel.
///
/// §6.6-0009: "the `x<hh>` form MUST NOT be used for those two cases". Driven
/// straight off the mask/rank rather than through a golden, so it covers ranks
/// the goldens do not reach — a spelling rule that only holds at the ranks
/// someone happened to write a vector for is not a spelling rule.
#[test]
fn no_rank_spells_a_sentinel_case_as_a_bitmask() {
    let base = "sk4|red|f32|cuda:sm89|ix32|warp|r2|co/00/v1/d8/f;co/00/v1/da/f|rall";
    let mut key = StructureKey::from_token(base).expect("stem must decode");

    for rank in 1..=8u8 {
        key.rank = rank;

        key.reduce_axes = AxisMask(((1u16 << rank) - 1) as u8);
        let all = key.to_token();
        assert!(
            all.ends_with("|rall"),
            "rank {rank}: all-axes must spell `rall`, got {all}"
        );

        key.reduce_axes = AxisMask(1u8 << (rank - 1));
        let last = key.to_token();
        let want_last = if rank == 1 { "|rall" } else { "|rlast" };
        assert!(
            last.ends_with(want_last),
            "rank {rank}: lone trailing axis must spell `{want_last}`, got {last}"
        );

        // Positive control at every rank that HAS a non-sentinel set: rank 1 and
        // rank 2 have none (every non-empty subset is all-axes or lone-trailing),
        // so the bitmask form must still be reachable from rank 3 up. Without
        // this the test would pass against an encoder that spells everything
        // `rall`.
        if rank >= 3 {
            key.reduce_axes = AxisMask(0b011);
            let subset = key.to_token();
            assert!(
                subset.ends_with("|x03"),
                "rank {rank}: a genuine subset must keep the bitmask form, got {subset}"
            );
        }
    }
}

/// Every spelling the encoder produces is one the decoder reads back to the same
/// mask — across all four forms and every rank the mask can express.
///
/// Encode and decode are separate code paths reading the same clause. A byte the
/// encoder emits and the decoder refuses would be a token this crate cannot
/// consume from its own producer, which is a worse failure than either half
/// being wrong alone.
#[test]
fn encode_and_decode_agree_on_every_reachable_mask() {
    let base = "sk4|red|f32|cuda:sm89|ix32|warp|r2|co/00/v1/d8/f;co/00/v1/da/f|rall";
    let mut key = StructureKey::from_token(base).expect("stem must decode");

    let mut saw = (false, false, false);
    for rank in 1..=8u8 {
        key.rank = rank;
        for mask in 0..=all_axes(rank) {
            key.reduce_axes = AxisMask(mask);
            let token = key.to_token();
            let back = StructureKey::from_token(&token)
                .unwrap_or_else(|| panic!("encoder emitted a token the decoder refuses: {token}"));
            assert_eq!(
                back.reduce_axes.0, mask,
                "rank {rank} mask {mask:#04x} decoded to a different set via {token}"
            );
            assert_eq!(back.to_token(), token, "re-encode must be stable");

            match token.rsplit('|').next().unwrap() {
                "rall" => saw.0 = true,
                "rlast" => saw.1 = true,
                s if s.starts_with('x') => saw.2 = true,
                _ => {}
            }
        }
    }
    // Coverage control: the loop above is only meaningful if it actually reached
    // all three non-empty forms.
    assert_eq!(
        saw,
        (true, true, true),
        "the sweep must exercise `rall`, `rlast` and `x<hh>`"
    );
}

fn all_axes(rank: u8) -> u8 {
    ((1u16 << rank) - 1) as u8
}
