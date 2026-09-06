//! **Renaming `ReassociatedDeterministic` → `DeterministicallyDivergent` must not
//! move a single byte a consumer parses.**
//!
//! The variant name and the FKC `determinism:` spelling are different strings —
//! `determinism_str` maps between them — so the rename is a source-level change
//! only. ⚠️ **That is a claim, and a claim about emitted artifacts is exactly the
//! kind this workspace spent 2026-09-06 discovering it could not see with a
//! signature diff.** So it is asserted rather than reasoned.

use unpopped::backend::VariantFidelity;

#[test]
fn the_determinism_spellings_are_unchanged_by_the_rename() {
    assert_eq!(
        VariantFidelity::DeterministicallyDivergent.determinism_str(),
        "same_hardware_bitwise",
        "the renamed variant must still emit the spelling Fuel's schema accepts \
         -- if this moved, the rename reached the wire"
    );

    // The whole mapping, because a rename that broke a NEIGHBOUR would leave the
    // assertion above green.
    assert_eq!(VariantFidelity::BitIdentical.determinism_str(), "bitwise");
    assert_eq!(
        VariantFidelity::Nondeterministic.determinism_str(),
        "nondeterministic"
    );
    assert_eq!(VariantFidelity::MorePrecise.determinism_str(), "bitwise");
}

/// ⚠️ The mapping is **not** injective, and that is deliberate — so a test
/// asserting "every fidelity has a distinct spelling" would be asserting a
/// property this enum does not have.
///
/// `BitIdentical` and `MorePrecise` both spell `bitwise`: one is bit-equal to the
/// default, the other is *better* than it, and **FKC's determinism axis does not
/// encode accuracy** — that is the precision block's job. The two axes are
/// orthogonal and the enum crosses them.
#[test]
fn two_fidelities_deliberately_share_a_determinism_spelling() {
    assert_eq!(
        VariantFidelity::BitIdentical.determinism_str(),
        VariantFidelity::MorePrecise.determinism_str(),
        "both are bitwise-reproducible; they differ in ACCURACY, which the \
         determinism axis does not carry"
    );
    assert_ne!(
        VariantFidelity::BitIdentical,
        VariantFidelity::MorePrecise,
        "control: they are still different fidelities -- if the enum ever \
         collapsed them, the assertion above would pass by identity"
    );
}
