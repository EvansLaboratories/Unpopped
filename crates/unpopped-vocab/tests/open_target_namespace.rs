//! A non-CUDA target survives the whole `StructureKey` path.
//!
//! # Why this file exists separately from the target unit tests
//!
//! `target.rs`'s own tests prove the token grammar. They do **not** prove the
//! thing the change was for: that a `vulkan:`/`rocm:`/`metal:` target can be put
//! *into* a key, come back out of its token, and compare equal — which is the
//! path a real consumer takes and the path the closed `ArchSku` enum blocked.
//!
//! The distinction is not academic. Every test in this crate passed with the
//! enum in place; what they proved was that CUDA worked. A refactor that swapped
//! the type without opening the *path* would leave all of them green.

use unpopped_vocab::{
    ArchSku, ElementKind, OpCategory, OperandDesc, StructureKey, TargetId, structure_key,
    structure_key_token,
};

fn operands(dtype: ElementKind) -> Vec<OperandDesc> {
    let d = OperandDesc::new(2, &[128, 256], &[256, 1], dtype, 256);
    vec![d; 3]
}

/// The headline: a Vulkan cell builds, tokenizes, and parses back.
///
/// Before the open target model this was not expressible at all — there was no
/// `ArchSku` variant that could name it, so a conforming `vulkan:` reference
/// vector had to be **excluded** from the cross-project byte-match rather than
/// matched against.
#[test]
fn a_vulkan_cell_round_trips_through_the_key_and_its_token() {
    let ops = operands(ElementKind::F16);
    let target = TargetId::parse("vulkan:sg64.ops-abr.arith-f16.cm-none").expect("well-formed");
    let key = structure_key(OpCategory::BinaryElementwise, &ops, target);

    let token = key.to_token();
    assert!(
        token.contains("|vulkan:sg64.ops-abr.arith-f16.cm-none|"),
        "the target must appear as ONE field of the token: {token}"
    );

    let back = StructureKey::from_token(&token).expect("a key we emitted must parse");
    assert_eq!(back, key, "round trip must be lossless");
    assert_eq!(back.target, target);
    assert_eq!(back.target.namespace(), "vulkan");
}

/// Three namespaces, none of which this crate understands, all representable.
///
/// §6.8-0004 puts each capability-set vocabulary in its maintainer's hands, so
/// "understands" is the wrong bar — *carries faithfully* is the right one.
#[test]
fn foreign_namespaces_reach_the_token_unaltered() {
    for t in ["rocm:gfx942", "metal:apple9", "vulkan:spirv1.6"] {
        let target = TargetId::parse(t).unwrap();
        let token = structure_key_token(
            OpCategory::UnaryElementwise,
            &operands(ElementKind::F32),
            target,
        );
        assert!(
            token.contains(&format!("|{t}|")),
            "{t} missing from {token}"
        );
        let back = StructureKey::from_token(&token).expect("round trip");
        assert_eq!(back.target.as_str(), t);
    }
}

/// Two targets differing by one byte produce different keys.
///
/// §6.8-0002 is byte-exact matching with no prefix or feature-implication logic.
/// `sm90` and `sm90a` are the pair that makes this concrete — adjacent
/// spellings, genuinely different compilation targets, and a prefix-matching
/// implementation would collapse them into one cache entry that serves the wrong
/// kernel.
#[test]
fn one_byte_of_target_difference_is_a_different_cell() {
    let ops = operands(ElementKind::F32);
    let a = structure_key(OpCategory::BinaryElementwise, &ops, ArchSku::Sm90);
    let b = structure_key(OpCategory::BinaryElementwise, &ops, ArchSku::Sm90a);
    assert_ne!(a, b, "sm90 and sm90a are not the same target");
    assert_ne!(a.to_token(), b.to_token());

    // And the same holds for a namespace this crate does not own.
    let v1 = TargetId::parse("vulkan:spirv1.6").unwrap();
    let v2 = TargetId::parse("vulkan:spirv1.5").unwrap();
    assert_ne!(v1, v2);
    assert_ne!(
        structure_key(OpCategory::BinaryElementwise, &ops, v1),
        structure_key(OpCategory::BinaryElementwise, &ops, v2)
    );
}

/// The four CUDA tokens are byte-identical to what the closed enum emitted.
///
/// This is the regression that would invalidate every cross-project byte-match
/// vector simultaneously — and it would do so by emitting a **well-formed** key
/// with a different spelling, which no grammar check and no round-trip catches.
/// The literals are written out rather than derived, so the test cannot agree
/// with a mistake by computing the expectation the same wrong way.
#[test]
fn the_cuda_spelling_did_not_move() {
    for (sku, want) in [
        (ArchSku::Sm80, "cuda:sm80"),
        (ArchSku::Sm89, "cuda:sm89"),
        (ArchSku::Sm90, "cuda:sm90"),
        (ArchSku::Sm90a, "cuda:sm90a"),
    ] {
        let token = structure_key_token(
            OpCategory::BinaryElementwise,
            &operands(ElementKind::F32),
            sku,
        );
        assert!(
            token.contains(&format!("|{want}|")),
            "{sku:?} must still spell {want}: {token}"
        );
    }
}

/// A malformed target in a token is a typed decline, not a panic or a silent
/// mis-parse.
///
/// The key's own field separator is `|`, and §6.8-0005 forbids it inside a
/// target for exactly this reason: a target carrying one would **re-split the
/// token** into a different field set that still parses. So the interesting
/// malformed cases are the ones that keep the field count right.
#[test]
fn a_malformed_target_field_declines_rather_than_mis_parsing() {
    let ops = operands(ElementKind::F32);
    let good = structure_key_token(OpCategory::BinaryElementwise, &ops, ArchSku::Sm89);
    assert!(StructureKey::from_token(&good).is_some(), "control");

    for bad in ["sm89", "cuda:", ":sm89", "cuda:sm:89"] {
        let broken = good.replace("cuda:sm89", bad);
        assert!(
            StructureKey::from_token(&broken).is_none(),
            "target field {bad:?} must be refused, not accepted: {broken}"
        );
    }
}

/// **Forward-compatibility: a namespace changing its own field count is a
/// non-event here.**
///
/// KISS's `vulkan:` vocabulary is going from four `.`-separated fields to five
/// (`spec/namespaces/vulkan.md` v4, rule V-1, commit `ee2b730`), adding a
/// `<coopvec>` field so every token gains `cv-none` or equivalent. The published
/// reference vector will be regenerated to match.
///
/// This test runs both shapes through the whole key path and asserts nothing
/// about either one's structure. It is the evidence for a claim I would
/// otherwise be making on my own say-so — that a `vulkan:` vocabulary bump
/// cannot break this crate, because §6.8-0004 puts the capability-set vocabulary
/// in its maintainer's hands and this crate therefore validates the **grammar**
/// (§6.8-0001, -0005) and never the vocabulary.
///
/// Two failure modes it forecloses:
///
/// * A hardcoded field count, or any splitting on `.`, would reject the
///   five-field token. There is none, and this proves it rather than asserting
///   it.
/// * A consumer scoping a target by *literal string* rather than by namespace
///   would silently stop recognizing the regenerated token. This crate scopes by
///   nothing — it carries the token whole — so the regeneration is invisible.
#[test]
fn a_namespace_changing_its_field_count_is_invisible_here() {
    let ops = operands(ElementKind::F16);
    // The published four-field token, and its five-field successor.
    for t in [
        "vulkan:sg64.ops-abr.arith-f16.cm-none",
        "vulkan:sg64.ops-abr.arith-f16.cm-none.cv-none",
    ] {
        let target = TargetId::parse(t).unwrap_or_else(|e| panic!("{t}: {e}"));
        let key = structure_key(OpCategory::BinaryElementwise, &ops, target);
        let token = key.to_token();
        assert!(
            token.contains(&format!("|{t}|")),
            "{t} must appear verbatim as one field: {token}"
        );
        let back = StructureKey::from_token(&token).expect("round trip");
        assert_eq!(back, key);
        assert_eq!(back.target.as_str(), t, "the token must survive byte-exact");
    }

    // And the two are DIFFERENT cells, which is the correct reading: a device
    // with cooperative-vector support is not the same target as one without, so
    // a key that conflated them would serve the wrong kernel.
    let four = TargetId::parse("vulkan:sg64.ops-abr.arith-f16.cm-none").unwrap();
    let five = TargetId::parse("vulkan:sg64.ops-abr.arith-f16.cm-none.cv-none").unwrap();
    assert_ne!(four, five);
    assert_ne!(
        structure_key(OpCategory::BinaryElementwise, &ops, four),
        structure_key(OpCategory::BinaryElementwise, &ops, five)
    );
}
