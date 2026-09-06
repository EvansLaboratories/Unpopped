//! ⚠️ **`MIN_FLIP_MARGIN` gates a FLIP of an existing decision. It does not gate
//! the FIRST decision for a cell.**
//!
//! Raised by baracuda 2026-09-06, whose `gate_cell` populates a routing table
//! from on-box timings, on a machine they measured as unable to produce a stable
//! one: **12 of 12 repeatedly-run cells vary by more than ±5%, three flip
//! direction, and one spanned 25.5 µs to 162.1 µs across ten runs.**
//!
//! `MIN_FLIP_MARGIN`'s own rationale cites `BENCHMARKS.md`'s "±5% as ≈" and
//! "20-30% timing variance at the smallest shapes". ⚠️ **A box with 2.34-5.55×
//! dispersion violates the model the constant was calibrated against by an order
//! of magnitude** — so 1.10 is not a conservative floor there, it is far below
//! the noise.
//!
//! **This test does not change that policy.** It pins where the floor applies, so
//! a consumer cannot read the constant's existence as protection it does not
//! give. **The name says "flip" and the name is accurate; the risk is that a
//! reader generalises it to "noise floor" and assumes every recorded decision
//! cleared it.**

use unpopped_vocab::{
    ArchSku, DispatchEntry, DispatchTable, ElementKind, Implementor, MIN_FLIP_MARGIN, OpCategory,
    OperandDesc, Provenance, merge, structure_key,
};

fn cell() -> String {
    let d = OperandDesc::new(1, &[7], &[1], ElementKind::F32, 4);
    structure_key(OpCategory::BinaryElementwise, &[d, d, d], ArchSku::Sm89).to_token()
}

fn entry_with(
    winner: Implementor,
    entry_point: Option<&str>,
    margin: f64,
    provenance: Provenance,
) -> DispatchEntry {
    DispatchEntry {
        structure_key: cell(),
        winner,
        winner_entry: entry_point.map(str::to_string),
        margin,
        ranked: Vec::new(),
        provenance,
        measured_on: None,
    }
}

fn entry(winner: Implementor, entry_point: Option<&str>, margin: f64) -> DispatchEntry {
    entry_with(winner, entry_point, margin, Provenance::Measured)
}

/// A first decision inside the noise floor is recorded with full authority.
#[test]
fn a_first_decision_is_not_gated_by_the_noise_floor() {
    let within_noise = 1.001_f64;
    assert!(
        within_noise < MIN_FLIP_MARGIN,
        "the fixture must be INSIDE the floor or this test proves nothing: \
         {within_noise} vs {MIN_FLIP_MARGIN}"
    );

    let mut table = DispatchTable::default();
    merge(
        &mut table,
        &[entry(Implementor::Generated, Some("k"), within_noise)],
    );

    assert_eq!(
        table.entries.len(),
        1,
        "a cell with NO incumbent takes the row regardless of margin -- the \
         floor is in the branch that has something to flip"
    );
    assert_eq!(table.entries[0].margin, within_noise);
}

/// The control: the same margin CANNOT flip an existing different route.
///
/// ⚠️ **The incumbent is `Seeded`, and that is load-bearing.** With two
/// `Measured` rows carrying no `measured_on`, `merge` refuses the second on
/// "only a newer capture may refresh" and never reaches the margin test —
/// **so the first assertion below would pass for the wrong reason.** Measured:
/// the first draft of this file did exactly that, and only the second assertion
/// (that a decisive win DOES flip) exposed it. A hand-seeded vendor route is
/// also the case `MIN_FLIP_MARGIN`'s own rationale names.
#[test]
fn the_control_that_same_margin_cannot_flip_an_incumbent() {
    let within_noise = 1.001_f64;

    let mut table = DispatchTable::default();
    merge(
        &mut table,
        &[entry_with(
            Implementor::Cublas,
            Some("vendor_route"),
            2.0,
            Provenance::Seeded,
        )],
    );
    merge(
        &mut table,
        &[entry(Implementor::Generated, Some("k"), within_noise)],
    );

    assert_eq!(
        table.entries[0].winner,
        Implementor::Cublas,
        "control: with an incumbent present the SAME margin is refused, so the          first-decision result above is about the ABSENCE of an incumbent and          not about the margin being accepted everywhere"
    );

    // And a decisive win does flip it, or the assertion above also passes on a
    // merge that is simply immovable -- which is exactly what happened on the
    // first draft, for an unrelated reason.
    merge(
        &mut table,
        &[entry(Implementor::Generated, Some("k"), MIN_FLIP_MARGIN)],
    );
    assert_eq!(
        table.entries[0].winner,
        Implementor::Generated,
        "a margin AT the floor must flip, or `merge` is refusing for some other          reason and neither assertion above means what it says"
    );
}
