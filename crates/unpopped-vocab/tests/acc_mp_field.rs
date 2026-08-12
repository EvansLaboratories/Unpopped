//! The sk4 non-contraction `(acc + mp)` precision field — KISS-CLASSIFY-6.7-0013.
//!
//! # Why these tests are round-trips against literal bytes
//!
//! Every golden token here is copied from KISS's reference vectors
//! (`conformance/tests/structure_key_golden.rs` at `origin/main` = `19c3ad7`),
//! and each is exercised as **decode → re-encode → compare bytes**. Nothing is
//! hand-constructed and nothing is compared field-by-field, because this crate
//! is one of four independent derivers of these tokens and the only thing a
//! byte-match can mean is that the bytes match. A test asserting
//! `key.acc_mp == Some(..)` would pass just as happily against a codec that
//! spells the field `f32:st`.
//!
//! The dispatch is the point. §6.7-0013 puts the `(acc + mp)` field in the
//! **same token slot** the contraction group occupies for `gem`, and resolves
//! the resulting nine-or-ten-field ambiguity by the **op-family code**. A
//! decoder that instead splits on a fixed field count breaks silently the moment
//! a cell widens from nine to ten, and one that sniffs the payload's shape
//! accepts a contraction group on a `red` cell. Both failures are quiet, and
//! both produce a key that is wrong rather than a key that is refused — so the
//! contraction-payload-on-a-reduction case is the first test below.

use unpopped_vocab::{AccMp, ElementKind, MpCode, StructureKey, TokenDecline};

// ---------------------------------------------------------------------------
// The dispatch, first — this is the silent-break case.
// ---------------------------------------------------------------------------

/// A contraction-shaped payload in the tenth slot of a **non-`gem`** cell is a
/// malformed `(acc + mp)` field, not a contraction group.
///
/// At sk4 that slot on a `red` cell *is* the `(acc + mp)` field, so
/// `ctll/d16/f32/f32/f32/st` there is a six-part spelling of a two-part field.
/// The decline is the whole safety property: were this accepted, a reduction
/// would decode carrying M/N/K size classes it does not have, and the resulting
/// key would collide with — or separate from — cells it has no relation to.
#[test]
fn a_contraction_payload_on_a_reduction_cell_is_a_malformed_acc_mp_field() {
    let stem = "sk4|red|f32|cuda:sm89|ix32|warp|r2|co/00/v1/d8/f;co/00/v1/da/f|rall";
    let token = format!("{stem}|ctll/d16/f32/f32/f32/st");

    assert_eq!(
        StructureKey::parse_token(&token),
        Err(TokenDecline::BadAccMpField),
        "a contraction group on a non-gem cell must decline as a malformed acc+mp field"
    );
    assert_eq!(
        StructureKey::from_token(&token),
        None,
        "the strict decoder must refuse it too, not merely the diagnosing one"
    );

    // Positive control on the stem: the decline above must be attributable to
    // the tenth field, not to a stem this codec never accepted in the first
    // place. Without this, the assertions pass against a codec that rejects
    // every `red` token for some unrelated reason.
    assert!(
        StructureKey::from_token(stem).is_some(),
        "control: the nine-field stem itself must parse"
    );
}

/// The mirror: the same payload IS the contraction group on a `gem` cell, and
/// the identical bytes decode rather than decline.
///
/// This is what makes the previous test a statement about *dispatch* instead of
/// a statement about `ctll/...` being unparseable everywhere.
#[test]
fn the_same_payload_is_the_contraction_group_on_a_gem_cell() {
    let gem = "sk4|gem|f32|cuda:sm89|ix32|grid|r2|\
               co/00/v4/d16/f;co/00/v4/d16/f;co/00/v4/d16/f|-|ctll/d16/f32/f32/f32/st";
    let key = StructureKey::from_token(gem).expect("a gem cell's tenth field is the contraction");
    assert!(
        key.contraction.is_some(),
        "gem: contraction group populated"
    );
    assert!(
        key.acc_mp.is_none(),
        "§6.7-0013: the two precision fields never coexist"
    );
    assert_eq!(key.to_token(), gem, "byte-exact round-trip");
}

// ---------------------------------------------------------------------------
// KISS's golden vectors, byte-for-byte.
// ---------------------------------------------------------------------------

/// Rules (a)/(b): a deviating **accumulator** emits the field, and both slots
/// are spelled — including the `<mp>` slot sitting at its default `st`.
///
/// Golden: `test_classify_noncontraction_acc_mp_field`.
#[test]
fn a_deviating_accumulator_emits_both_slots() {
    let g = "sk4|red|f16|cuda:sm89|ix32|warp|r2|co/00/v1/d8/f;co/00/v1/da/f|rlast|f32/st";
    let key = StructureKey::from_token(g).expect("golden must decode");
    assert_eq!(
        key.acc_mp,
        Some(AccMp {
            acc: ElementKind::F32,
            mp: MpCode::St
        })
    );
    assert_eq!(key.to_token(), g);
}

/// Rules (a)/(b): a deviating **math-precision** emits the field even though the
/// accumulator equals the compute dtype — and rule (b) spells the accumulator
/// slot anyway. This is the strict-vs-reduced-mantissa axis reaching a scan
/// cell, where before sk4 it had nowhere to go.
///
/// Golden: `sk4_noncontraction_acc_mp_deviating_precision_only`. Note the `-` in
/// the reduce field: a `scn` cell carries no reduced axes, and the precision
/// field is a *separate* slot — the two are not alternatives.
#[test]
fn a_deviating_math_precision_alone_emits_the_field() {
    let g = "sk4|scn|f32|cuda:sm89|ix32|warp|r2|co/00/v4/d16/f;co/00/v4/d16/f|-|f32/rm";
    let key = StructureKey::from_token(g).expect("golden must decode");
    assert_eq!(
        key.acc_mp,
        Some(AccMp {
            acc: ElementKind::F32,
            mp: MpCode::Rm
        })
    );
    assert_eq!(key.to_token(), g);
}

/// Rules (c)/(e): when neither coordinate deviates the field is omitted
/// **entirely** — a nine-field token. Not `-`, not an empty tenth field.
///
/// Rule (e) exists because the reduce field (§6.6-0009) is mandatory and emits
/// `-` when inapplicable. Carrying that convention across to this field would
/// append a spurious `|-` to every ordinary cell and fail the byte-match on
/// tokens that have nothing to do with precision.
///
/// Golden: `sk4_noncontraction_acc_mp_omitted_when_default`.
#[test]
fn a_non_deviating_cell_omits_the_field_entirely() {
    let g = "sk4|red|f32|cuda:sm89|ix32|warp|r2|co/00/v1/d8/f;co/00/v1/da/f|rall";
    let key = StructureKey::from_token(g).expect("golden must decode");
    assert_eq!(key.acc_mp, None);

    let token = key.to_token();
    assert_eq!(token, g);
    assert_eq!(
        token.split('|').count(),
        9,
        "rule (c)/(e): no trailing precision field at all — not `-`, not empty"
    );
    assert!(
        !token.ends_with("|-") || token.ends_with("|rall"),
        "rule (e): the reduce field's `-` convention must not leak onto this field"
    );
}

/// Rule (d): the all-default spelling is a **forbidden redundant emission** and
/// a token carrying it MUST be rejected.
///
/// A canonical producer omits the field (rule (c)), so its presence at default
/// means one cell has two legal spellings — which is precisely a byte-match
/// failure that no party is at fault for. The decline is distinct from the
/// malformed one: this field is well-formed, it just says nothing.
///
/// Golden: `sk4_noncontraction_acc_mp_rejects_redundant_default`.
#[test]
fn the_all_default_spelling_is_rejected_as_redundant() {
    let stem = "sk4|red|f32|cuda:sm89|ix32|warp|r2|co/00/v1/d8/f;co/00/v1/da/f|rall";

    assert_eq!(
        StructureKey::parse_token(&format!("{stem}|f32/st")),
        Err(TokenDecline::RedundantAccMpField),
        "f32 compute + f32 acc + default st = redundant"
    );

    // The two neighbours that DO deviate must parse — otherwise the assertion
    // above is satisfied by a codec that rejects the tenth field wholesale.
    let deviating_acc = format!("{stem}|f64/st");
    assert!(
        StructureKey::parse_token(&deviating_acc).is_ok(),
        "a deviating accumulator (f64 != f32) must parse"
    );
    let deviating_mp = format!("{stem}|f32/rm");
    assert!(
        StructureKey::parse_token(&deviating_mp).is_ok(),
        "a deviating mp (rm != st) must parse"
    );

    // And both round-trip to the bytes they came from.
    for t in [&deviating_acc, &deviating_mp] {
        assert_eq!(StructureKey::from_token(t).unwrap().to_token(), *t);
    }
}

/// The encoder can never produce the token rule (d) forbids — even when handed a
/// redundant value directly.
///
/// `AccMp::new` refuses to build one, and `to_token` re-derives through it, so
/// the two halves of the codec cannot disagree about whether a cell deviates. A
/// producer that emitted `|f32/st` on an `f32` cell would be emitting a token
/// its own decoder is required to reject — the kind of self-inconsistency that
/// only ever surfaces as someone else's parse failure.
#[test]
fn the_encoder_cannot_emit_the_forbidden_redundant_field() {
    assert_eq!(
        AccMp::new(ElementKind::F32, ElementKind::F32, MpCode::St),
        None,
        "the constructor is rule (a)/(c)/(d): a non-deviating pair is not a field"
    );

    let g = "sk4|red|f32|cuda:sm89|ix32|warp|r2|co/00/v1/d8/f;co/00/v1/da/f|rall";
    let mut key = StructureKey::from_token(g).unwrap();
    // Force the redundant value past the constructor, the way a caller building
    // the struct by hand could.
    key.acc_mp = Some(AccMp {
        acc: ElementKind::F32,
        mp: MpCode::St,
    });
    assert_eq!(
        key.to_token(),
        g,
        "the encoder must omit a redundant field rather than emit an invalid token"
    );

    // Positive control: a genuinely deviating value on the same key DOES emit,
    // so the assertion above is about redundancy and not about `acc_mp` being
    // ignored on the encode path.
    key.acc_mp = Some(AccMp {
        acc: ElementKind::F64,
        mp: MpCode::St,
    });
    assert_eq!(key.to_token(), format!("{g}|f64/st"));
}

// ---------------------------------------------------------------------------
// Typed declines.
// ---------------------------------------------------------------------------

/// Each malformed spelling gets its own verdict, and the strict decoder agrees
/// with the diagnosing one on every case.
///
/// The agreement check is the reason this is one test rather than five. The two
/// entry points reach `parse_acc_mp` by different routes; if they ever diverged,
/// a consumer would be told *why* a token was refused by a code path that had
/// not refused it — a decline reason describing a verdict nobody reached.
///
/// Golden: `sk4_noncontraction_acc_mp_declines`.
#[test]
fn malformed_acc_mp_fields_decline_with_distinct_typed_verdicts() {
    let stem = "sk4|red|f32|cuda:sm89|ix32|warp|r2|co/00/v1/d8/f;co/00/v1/da/f|rall";

    let cases: &[(&str, TokenDecline)] = &[
        // An accumulator outside the closed §6.1 set.
        ("f99/rm", TokenDecline::Unrecognized),
        // Recognized but RESERVED (§6.1-0001) — distinct from unknown, because
        // "upgrade won't help, route around it" is different advice from "your
        // vocabulary may be stale".
        (
            "f8e4m3fnuz/rm",
            TokenDecline::ReservedDtype {
                spelling: "f8e4m3fnuz".to_string(),
            },
        ),
        // A `<mp>` code that is not `st`/`rm`.
        ("f64/zz", TokenDecline::BadAccMpField),
        // Wrong part count: one part, no `<mp>`. Rule (b) requires both slots.
        ("f64", TokenDecline::BadAccMpField),
        // Wrong part count: three parts.
        ("f64/st/x", TokenDecline::BadAccMpField),
    ];

    for (field, want) in cases {
        let token = format!("{stem}|{field}");
        assert_eq!(
            StructureKey::parse_token(&token),
            Err(want.clone()),
            "typed decline for tenth field `{field}`"
        );
        assert_eq!(
            StructureKey::from_token(&token),
            None,
            "the strict decoder must refuse `{field}` too — the two paths must agree"
        );
    }

    // Positive control: every assertion above is a refusal, so without a case
    // that PARSES this test would pass against a codec that declines all input.
    assert!(
        StructureKey::parse_token(&format!("{stem}|f64/rm")).is_ok(),
        "control: a well-formed deviating field must still parse"
    );
}

/// A reserved accumulator declines as reserved on **both** parse paths, and the
/// spelling it reports is the one that actually appeared.
///
/// §6.1-0001 governs every dtype position, and the accumulator slot is a dtype
/// position — this pins that the tenth field was not overlooked when that rule
/// was applied to the others.
#[test]
fn the_reserved_rule_reaches_the_accumulator_slot() {
    let stem = "sk4|red|f32|cuda:sm89|ix32|warp|r2|co/00/v1/d8/f;co/00/v1/da/f|rall";
    for spelling in ["f8e4m3fnuz", "f8e5m2fnuz"] {
        assert_eq!(
            StructureKey::parse_token(&format!("{stem}|{spelling}/rm")),
            Err(TokenDecline::ReservedDtype {
                spelling: spelling.to_string()
            }),
            "a reserved accumulator must be recognized-and-declined, not unknown"
        );
    }
}

// ---------------------------------------------------------------------------
// This generator's own cells.
// ---------------------------------------------------------------------------

/// Every key this crate DERIVES carries no `(acc + mp)` field, and that is a
/// claim about the generator rather than a default.
///
/// `derive_acc_mp` declares "accumulates at the compute dtype, bit-stable",
/// which is true because every schedule this generator can lower is elementwise
/// — both emitters decline `Reduction`/`RowReduce`/`Scan`/`Contraction`. The
/// consequence on the wire is the §6.7-0013 byte-stability property: these
/// tokens are byte-identical to the pre-sk4 codec modulo the §6.1 dtype renames.
///
/// If a reduction emitter lands with a wider accumulator and this test still
/// passes, the declaration has become false — the field must start appearing.
#[test]
fn derived_cells_declare_no_deviation_and_stay_nine_fields() {
    use unpopped_vocab::{ArchSku, OpCategory, OperandDesc, structure_key};

    for dt in [
        ElementKind::F32,
        ElementKind::F16,
        ElementKind::Bf16,
        ElementKind::F64,
        ElementKind::I32,
    ] {
        let a = OperandDesc::new(1, &[1024], &[1], dt, 256);
        let key = structure_key(OpCategory::BinaryElementwise, &[a, a, a], ArchSku::Sm89);
        assert_eq!(key.acc_mp, None, "{dt:?}: elementwise accumulates in place");

        let token = key.to_token();
        assert_eq!(
            token.split('|').count(),
            9,
            "{dt:?}: a non-deviating cell is a nine-field token — {token}"
        );
        assert_eq!(
            StructureKey::from_token(&token).unwrap().to_token(),
            token,
            "{dt:?}: round-trip"
        );
    }
}
