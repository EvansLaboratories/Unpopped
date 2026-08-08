//! The `sk<N>` version field: canonical form, and three distinct verdicts.
//!
//! KISS-CLASSIFY-6.7-0015 splits what a reader must say about a version field:
//!
//! - a **canonical** decimal (`0` or `[1-9][0-9]*`) that this build does not
//!   accept is a *recognized token from another schema* — decline naming the
//!   version, the "re-derive" signpost;
//! - a **malformed** field — no `sk`, a non-numeric remainder, or a
//!   non-canonical spelling like `sk04` — is the *bad-version-prefix* wall.
//!
//! and requires the canonical-form check to run **before any numeric parse**.
//!
//! # What was actually wrong here
//!
//! All three of these were accepted as a valid `sk3` token before this guard:
//!
//! | token | parsed as | why |
//! |---|---|---|
//! | `sk03` | 3 | Rust's integer parse tolerates leading zeros |
//! | `sk+3` | 3 | …and a leading sign |
//! | `sk4` / `sk99` | 4 / 99 | the version was parsed but **never checked** |
//!
//! The third is the serious one and it is not a spelling nit. An `sk4` token fed
//! to an `sk3` build was accepted and decoded **under the sk3 dtype vocabulary**
//! — and `c64` denotes a pair of `f64` at sk3 but a pair of `f32` at sk4. So a
//! cross-vocabulary token would have been silently reinterpreted at half its
//! width, which is precisely the hazard §6.7-0014 cites when it makes the
//! version match exact. It was live in the published codec.

use unpopped_vocab::{
    ArchSku, ElementKind, OpCategory, OperandDesc, STRUCTURE_KEY_VERSION, StructureKey,
    TokenDecline, structure_key,
};

fn good_token() -> String {
    let d = OperandDesc::new(1, &[7], &[1], ElementKind::F32, 256);
    structure_key(OpCategory::BinaryElementwise, &[d, d, d], ArchSku::Sm89).to_token()
}

/// Swap only the version field, leaving the rest of the token intact.
fn with_version(v: &str) -> String {
    let good = good_token();
    let rest = good.split_once('|').expect("token has fields").1;
    format!("{v}|{rest}")
}

/// The positive control. Everything below asserts a refusal, so without this the
/// file would pass against a codec that rejects every token.
#[test]
fn the_current_schema_version_is_accepted() {
    let tok = good_token();
    assert!(
        tok.starts_with(&format!("sk{STRUCTURE_KEY_VERSION}|")),
        "harness precondition: emitted token must carry this build's version, got {tok}"
    );
    assert!(StructureKey::from_token(&tok).is_some());
    assert!(StructureKey::parse_token(&tok).is_ok());
}

/// A canonical version this build does not accept declines **naming the version**.
#[test]
fn another_schemas_version_declines_as_recognized_not_malformed() {
    for other in [0u16, 1, 2, 4, 99] {
        if other == STRUCTURE_KEY_VERSION {
            continue;
        }
        let tok = with_version(&format!("sk{other}"));
        assert_eq!(
            StructureKey::parse_token(&tok),
            Err(TokenDecline::UnsupportedSchemaVersion { version: other }),
            "sk{other} is a well-formed token from another schema — it must decline \
             naming the version so the caller knows to RE-DERIVE, not that it sent garbage"
        );
        assert_eq!(
            StructureKey::from_token(&tok),
            None,
            "sk{other} must not decode under this build's vocabulary — that is the \
             cross-vocabulary misdecode the exact version match exists to prevent"
        );
    }
}

/// A malformed version field is the distinct "not a token" wall.
///
/// `sk03` and `sk+3` are the two that a bare `parse()` accepts, and they are the
/// reason the canonical check must run first.
#[test]
fn a_malformed_version_field_declines_as_bad_prefix() {
    for bad in [
        "sk03", "sk+3", "sk-3", "skx", "sk", "3", "SK3", "sk3x", "sk 3",
    ] {
        let tok = with_version(bad);
        assert_eq!(
            StructureKey::parse_token(&tok),
            Err(TokenDecline::BadVersionPrefix),
            "{bad:?} is not a well-formed version field, so it must hit the \
             bad-prefix wall rather than report as a token from another schema"
        );
        assert_eq!(
            StructureKey::from_token(&tok),
            None,
            "{bad:?} must not decode"
        );
    }
}

/// The two verdicts must be **distinguishable**, which is the whole point of the
/// split — a consumer that cannot tell them apart cannot tell "you sent me
/// garbage" from "you sent me an older schema; re-derive it".
#[test]
fn the_two_version_verdicts_are_distinct() {
    let other = if STRUCTURE_KEY_VERSION == 3 { 4 } else { 3 };
    let from_other_schema = StructureKey::parse_token(&with_version(&format!("sk{other}")));
    let malformed = StructureKey::parse_token(&with_version("sk04"));

    assert_ne!(
        from_other_schema, malformed,
        "a token from another schema and a malformed one must not collapse to the \
         same decline"
    );
    assert!(matches!(
        from_other_schema,
        Err(TokenDecline::UnsupportedSchemaVersion { .. })
    ));
    assert_eq!(malformed, Err(TokenDecline::BadVersionPrefix));
}
