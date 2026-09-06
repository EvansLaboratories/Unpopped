//! **Ranking a cell must not depend on the order the caller pushed candidates —
//! including on an exact tie, which is where it used to.**
//!
//! # The defect this was written against
//!
//! `dispatch.rs` asserted, at the sort:
//!
//! > *"`total_cmp` is a genuine total order with no NaN corner — the fastest
//! > wins, **independent of input order**."*
//!
//! ⚠️ **`total_cmp` is a total order on VALUES. It says nothing about which of
//! two EQUAL candidates lands at index 0** — and `sort_by` is stable, so the
//! winner, its `entry_point`, and the `margin` were decided by whichever
//! candidate the caller happened to push first.
//!
//! **Measured 2026-09-06 before the fix: swapping two candidates with identical
//! medians changed the winner.** The first version of this file asserted exactly
//! that (`assert_ne!`) to prove the defect was real; installing `rank_order`
//! reddened it, which is the intended life-cycle — **a born-red demonstration
//! becomes the permanent invariant once the fix lands.**
//!
//! # Provenance
//!
//! Relayed by baracuda 2026-09-06 from their own 0.4.2 producer, where an
//! omitted sort-key component made two tuples compare equal and the emitted
//! order was a **stable-sort accident** rather than a decision. ⚠️ **An omitted
//! sort-key component does not error. It silently defers to input order, and
//! input order is usually right by luck.**
//!
//! **What this does NOT assert:** which candidate *should* win a tie. The
//! tiebreak is arbitrary-but-stable and carries no performance meaning.
//! **Reproducible-but-arbitrary and arbitrary-and-irreproducible are different
//! defects, and only the second one was ever this crate's business.**

use unpopped_vocab::{CandidateResult, Implementor, winner_of};

fn candidate(implementor: Implementor, median_ns: f64, entry: Option<&str>) -> CandidateResult {
    CandidateResult {
        implementor,
        median_ns,
        entry_point: entry.map(str::to_string),
    }
}

#[test]
fn the_winner_does_not_depend_on_the_order_candidates_were_pushed() {
    let generated = candidate(Implementor::Generated, 1_000.0, Some("generated_kernel"));
    let vendor = candidate(Implementor::Cublas, 1_000.0, None);

    let forward = winner_of(
        "cell".to_string(),
        vec![generated.clone(), vendor.clone()],
        None,
    )
    .expect("two valid candidates must produce a route");
    let reversed = winner_of("cell".to_string(), vec![vendor, generated], None)
        .expect("two valid candidates must produce a route");

    assert_eq!(
        (forward.winner, &forward.winner_entry),
        (reversed.winner, &reversed.winner_entry),
        "an EXACT TIE must resolve the same way both directions. This is the \
         case that used to fall through to sort stability and be decided by the \
         caller's push order"
    );
    assert_eq!(
        forward.margin, reversed.margin,
        "the margin is derived from the ordering, so it moves with the winner"
    );
}

#[test]
fn the_control_a_distinct_median_was_already_order_independent() {
    let fast = candidate(Implementor::Generated, 1.0, Some("fast"));
    let slow = candidate(Implementor::Cublas, 2.0, None);

    let a = winner_of("c".to_string(), vec![fast.clone(), slow.clone()], None).unwrap();
    let b = winner_of("c".to_string(), vec![slow, fast], None).unwrap();

    assert_eq!(
        a.winner, b.winner,
        "control: with DISTINCT medians the winner never depended on input \
         order. Without this, the assertion above is also satisfied by a sort \
         that is broken in both directions"
    );
    assert_eq!(
        a.winner,
        Implementor::Generated,
        "and the FASTER candidate must still win -- a tiebreak that reorders \
         untied candidates would satisfy every equality above while destroying \
         the thing the ranking is for"
    );
}
