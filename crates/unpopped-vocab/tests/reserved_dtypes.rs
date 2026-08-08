//! Reserved dtypes: recognized, distinguished, and refused.
//!
//! KISS-Classify §6.1-0001 imposes three separate obligations on the `fnuz` FP8
//! pair, and it is easy to satisfy one while violating another:
//!
//! 1. **Recognize** the spelling — it is part of the closed vocabulary.
//! 2. **Distinguish** it from an unknown token — "a reader MUST recognize it and
//!    distinguish it from an unknown token".
//! 3. **Refuse** any `structure_key` that uses it in any dtype position — "no
//!    computation semantics at this schema version".
//!
//! Before this work `unpopped-vocab` had no `ElementKind` for either spelling,
//! so they parsed as **unknown**. That satisfied (3) by accident while violating
//! (1) and (2) — the key was refused, but for the wrong reason and with the
//! wrong information reaching the caller.
//!
//! The distinction in (2) is not bookkeeping. *Unrecognized* tells a consumer the
//! peer is speaking a vocabulary it does not have, so it may be out of date and
//! upgrading might help. *Reserved* tells it the vocabulary is shared and agreed
//! and this member is simply parked — upgrading will not help, and the right
//! response is to route around the dtype rather than suspect version skew. A
//! consumer that cannot tell these apart will chase the wrong problem.

use unpopped_vocab::{
    ArchSku, ElementKind, OpCategory, OperandDesc, StructureKey, TokenDecline, dtype_token,
    structure_key,
};

/// Obligation 1: the spellings are recognized as dtypes.
#[test]
fn the_reserved_spellings_are_in_the_vocabulary() {
    assert_eq!(dtype_token(ElementKind::Fp8E4M3FNUZ), "f8e4m3fnuz");
    assert_eq!(dtype_token(ElementKind::Fp8E5M2FNUZ), "f8e5m2fnuz");

    // And they are distinct from the ACTIVE FP8 variants, which is the whole
    // point of the variant-explicit spelling: `e4m3fn` and `e4m3fnuz` are
    // byte-incompatible (different bias, no infinities), so a token that
    // conflated them would licence silently wrong numerics.
    assert_ne!(
        dtype_token(ElementKind::Fp8E4M3FN),
        dtype_token(ElementKind::Fp8E4M3FNUZ)
    );
    assert_ne!(
        dtype_token(ElementKind::Fp8E5M2),
        dtype_token(ElementKind::Fp8E5M2FNUZ)
    );
}

/// Obligation 1, and the property that makes it checkable: `is_reserved` picks
/// out exactly the parked pair and nothing else.
#[test]
fn exactly_two_dtypes_are_reserved() {
    let all = [
        ElementKind::F16,
        ElementKind::Bf16,
        ElementKind::F32,
        ElementKind::F32Strict,
        ElementKind::F64,
        ElementKind::I8,
        ElementKind::I16,
        ElementKind::U8,
        ElementKind::U16,
        ElementKind::I32,
        ElementKind::I64,
        ElementKind::U32,
        ElementKind::U64,
        ElementKind::Bool,
        ElementKind::Fp8E4M3FN,
        ElementKind::Fp8E5M2,
        ElementKind::Fp8E4M3FNUZ,
        ElementKind::Fp8E5M2FNUZ,
        ElementKind::I4,
        ElementKind::U4,
        ElementKind::B1,
        ElementKind::Complex64,
        ElementKind::Complex128,
    ];
    let reserved: Vec<_> = all.into_iter().filter(|d| d.is_reserved()).collect();
    assert_eq!(
        reserved,
        vec![ElementKind::Fp8E4M3FNUZ, ElementKind::Fp8E5M2FNUZ],
        "the reserved set must be exactly the two fnuz variants — it grew or shrank"
    );
}

/// Obligations 2 and 3 together: a key using a reserved dtype is refused, **and**
/// the refusal is distinguishable from the unknown-token refusal.
#[test]
fn a_reserved_dtype_declines_distinctly_from_an_unknown_token() {
    // A well-formed token whose dtype field is reserved.
    let d = OperandDesc::new(1, &[7], &[1], ElementKind::F32, 256);
    let good = structure_key(OpCategory::BinaryElementwise, &[d, d, d], ArchSku::Sm89).to_token();
    let reserved_tok = good.replacen("|f32|", "|f8e4m3fnuz|", 1);
    assert_ne!(
        reserved_tok, good,
        "harness precondition: the dtype field must actually have been replaced"
    );

    match StructureKey::parse_token(&reserved_tok) {
        Err(TokenDecline::ReservedDtype { spelling }) => assert_eq!(spelling, "f8e4m3fnuz"),
        other => panic!(
            "a reserved dtype must decline AS reserved — §6.1-0001 requires this be \
             distinct from the unknown-token decline; got {other:?}"
        ),
    }

    // The contrast that gives the assertion above its meaning.
    let unknown_tok = good.replacen("|f32|", "|f13|", 1);
    assert_eq!(
        StructureKey::parse_token(&unknown_tok),
        Err(TokenDecline::Unrecognized),
        "a genuinely unknown spelling must NOT report as reserved"
    );

    // Obligation 3 via the legacy surface: it still refuses, it just cannot say why.
    assert_eq!(StructureKey::from_token(&reserved_tok), None);
    assert_eq!(StructureKey::from_token(&unknown_tok), None);
}

/// The positive control for `parse_token`.
///
/// Without it, every assertion above is satisfied by a `parse_token` that
/// declines unconditionally.
#[test]
fn parse_token_accepts_an_ordinary_key() {
    let d = OperandDesc::new(1, &[7], &[1], ElementKind::F32, 256);
    let key = structure_key(OpCategory::BinaryElementwise, &[d, d, d], ArchSku::Sm89);
    let tok = key.to_token();

    let back = StructureKey::parse_token(&tok).expect(
        "an ordinary f32 elementwise key must parse — otherwise the decline tests \
                 above are measuring a parser that rejects everything",
    );
    assert_eq!(back.to_token(), tok, "parse_token must round-trip");
}

/// A reserved dtype is refused in the **contraction precision group** too, not
/// only in the top-level dtype field.
///
/// §6.1-0001 says "any dtype position". The gem group carries three more
/// (`wdt`/`acc`/`out`), and guarding only the obvious one would leave the clause
/// half-implemented in the place most likely to matter for FP8 — a contraction
/// is exactly where an FP8 operand dtype would appear.
#[test]
fn a_reserved_dtype_is_refused_inside_the_contraction_precision_group() {
    let a = OperandDesc::new(2, &[64, 64], &[64, 1], ElementKind::F32, 256);
    let tok = structure_key(OpCategory::Gemm, &[a, a, a], ArchSku::Sm89).to_token();
    assert!(
        tok.contains("|c"),
        "harness precondition: a gem key must carry the contraction field, got {tok}"
    );

    // The precision group is the trailing `/<wdt>/<acc>/<out>/<mp>`; swap the
    // first of those dtype slots for a reserved spelling.
    let idx = tok.rfind("/f32/").expect("precision group present");
    let poisoned = format!("{}/f8e4m3fnuz/{}", &tok[..idx], &tok[idx + 5..]);

    assert!(
        matches!(
            StructureKey::parse_token(&poisoned),
            Err(TokenDecline::ReservedDtype { .. })
        ),
        "a reserved dtype inside the gem precision group must decline as reserved; \
         got {:?}",
        StructureKey::parse_token(&poisoned)
    );
}
