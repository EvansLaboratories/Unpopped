//! `Complex64`'s storage width, pinned against a silent second flip.
//!
//! # Why this needs its own assertion
//!
//! `unpopped_vocab::Complex64` is `{ re: f32, im: f32 }` — 8 bytes, 64 bits
//! **total** — the sk4 respell (KISS-CLASSIFY §6.1-0001: `c64` is named by
//! total width, not component width). Before the sk4 cut this same type name
//! meant `{ re: f64, im: f64 }`, 16 bytes — the sk3 convention, named by
//! component width. That flip landed correctly: a major bump (0.1.0 →
//! 0.2.0), marked breaking, with a schema-version check so an sk3 token can
//! never silently decode under the sk4 vocabulary.
//!
//! But it DID happen once, which is the point of pinning it here: nothing
//! about the type or its name prevents it from happening again, silently,
//! the next time someone touches this struct. A test that would have caught
//! the first flip is worth having for the second one.
//!
//! # A live naming collision this crate does not control
//!
//! `baracuda-types` (a different crate, in Baracuda) defines its OWN,
//! independent `Complex64` (`crates/baracuda-types/src/numeric.rs:373`),
//! whose own test asserts `size_of::<Complex64>() == 16` — an `f64` pair,
//! the sk3/component-width convention. On this crate's (sk4/KISS) naming,
//! that type is a `Complex128`. Same identifier, double the width, no
//! shared lineage, and both reachable from Fuel's dependency graph. The
//! rename is a breaking call across two crates (a `baracuda-types` publish
//! event) and is explicitly NOT this test's job — see the board's item 70.
//! This test only pins THIS crate's own meaning so a future edit here
//! cannot narrow that collision's blast radius by accident, in either
//! direction, without a test failing to announce it.

use unpopped_vocab::Complex64;

#[test]
fn complex64_is_64_bits_total_an_f32_pair_not_an_f64_pair() {
    assert_eq!(
        core::mem::size_of::<Complex64>(),
        8,
        "Complex64 must stay an (f32, f32) pair, 64 bits TOTAL (KISS-CLASSIFY \
         §6.1-0001's sk4 naming) — 16 would mean it silently reverted to the \
         sk3 (f64, f64) meaning without a version bump, exactly the flip \
         0.2.0 made deliberately and loudly once already"
    );
    assert_eq!(
        core::mem::size_of::<f32>() * 2,
        core::mem::size_of::<Complex64>(),
        "Complex64 must be exactly a pair of its own re/im field type, no padding"
    );
}
