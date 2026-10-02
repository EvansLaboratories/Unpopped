//! KISS-CLASSIFY §6.6-0021 (KISS#517): a scale-type dtype (`f8e8m0`, `f8e6m2`)
//! at operand 0 has no defined `structure_key.dtype`, so derivation MUST decline,
//! with a typed decline (§7.1-0002), *"rather than emit a token"*, and MUST NOT
//! substitute another operand's dtype or the scale's own spelling.
//!
//! `structure_key` / `structure_key_token` are infallible, so they cannot
//! decline, and §6.8-0004 forbids panicking. `try_structure_key` /
//! `try_structure_key_token` are the conformant path. Measured before this, on
//! 0.13.0: a scale-first list `[f8e8m0, f32]` keyed with dtype `F8E8M0`.
//!
//! A scale stays valid as a *sibling* operand (§6.1-0013, the sk4 model in
//! `scale_sibling_model.rs`). Only the primary slot is constrained.

use unpopped_vocab::{
    ArchSku, DeriveDecline, ElementKind, OpCategory, OperandDesc, structure_key, try_structure_key,
    try_structure_key_token,
};

fn od(n: i64, dtype: ElementKind) -> OperandDesc {
    OperandDesc::new(1, &[n], &[1], dtype, 256)
}

#[test]
fn a_scale_first_list_declines() {
    for scale in [ElementKind::F8E8M0, ElementKind::F8E6M2] {
        let ops = [od(16, scale), od(512, ElementKind::F32)];
        assert_eq!(
            try_structure_key(OpCategory::Gemm, &ops, ArchSku::Sm89),
            Err(DeriveDecline::ScaleDtypeAtOperand0 { dtype: scale }),
            "{scale:?} at operand 0 must decline"
        );
        assert!(
            try_structure_key_token(OpCategory::Gemm, &ops, ArchSku::Sm89).is_err(),
            "{scale:?}: the token path must decline too, rather than emit"
        );
    }
}

/// Control: the sibling placement the sk4 model uses (data first, scale second)
/// still keys, so a fix that rejected every list containing a scale would fail.
#[test]
fn a_scale_as_a_sibling_still_keys() {
    let ops = [od(512, ElementKind::I4), od(16, ElementKind::F8E8M0)];
    let key = try_structure_key(OpCategory::Gemm, &ops, ArchSku::Sm89)
        .expect("a scale in a non-primary slot is valid");
    assert_eq!(key.dtype, ElementKind::I4);
}

/// The fallible path agrees with the infallible one on every input that does
/// not decline, so migrating to it changes no token.
#[test]
fn the_try_path_matches_the_infallible_one_when_it_succeeds() {
    let a = OperandDesc::new(2, &[128, 256], &[256, 1], ElementKind::F32, 256);
    let ops = [a, a, a];
    #[allow(deprecated)]
    let old = structure_key(OpCategory::BinaryElementwise, &ops, ArchSku::Sm89);
    let new = try_structure_key(OpCategory::BinaryElementwise, &ops, ArchSku::Sm89)
        .expect("plain f32 elementwise keys");
    assert_eq!(new, old);
    assert_eq!(
        try_structure_key_token(OpCategory::BinaryElementwise, &ops, ArchSku::Sm89).unwrap(),
        old.to_token()
    );
}
