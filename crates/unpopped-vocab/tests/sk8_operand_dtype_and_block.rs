// SPDX-License-Identifier: MIT OR Apache-2.0
//! `sk8` (KISS board 444): every operand sub-key ends in its own dtype, and a quantized contraction's weight block length rides
//! last, beside the scale, as `g16` or `g32`. The KISS vectors pin the verdicts of `parse_token`; these tests pin that the strict
//! decode `from_token` refuses the same tokens (two decoders that disagree are two different wire formats), and that a mixed-dtype
//! key round-trips byte for byte.

use unpopped_vocab::{StructureKey, TokenDecline};

/// A quantized contraction: weight i4 (primary), an f16 scale sibling and a bf16 output.
const OPS: &str = "co/00/v4/d16/f/i4/dl;co/00/v4/d16/f/f16;co/00/v4/d16/f/bf16";

fn ctr(ops: &str, field: &str) -> String {
    format!("sk8|ctr|i4|cuda:sm90|ix32|grid|r2|{ops}|-|{field}")
}

#[test]
fn a_mixed_dtype_key_round_trips_byte_for_byte() {
    let token = ctr(OPS, "ctll/d16/i4/f32/bf16/st/qf16/g32");
    let key = StructureKey::parse_token(&token).expect("the control parses");
    assert_eq!(key.to_token(), token);
    assert_eq!(StructureKey::from_token(&token), Some(key));
    let c = key.contraction.expect("a contraction");
    assert_eq!(c.block.map(|b| b.elements()), Some(32));
}

/// `token` is declined by `parse_token` with `verdict` and refused by the strict decode.
fn both_refuse(token: &str, verdict: TokenDecline) {
    assert_eq!(
        StructureKey::parse_token(token),
        Err(verdict),
        "typed verdict: {token}"
    );
    assert_eq!(
        StructureKey::from_token(token),
        None,
        "the strict decode accepted what the typed one declines: {token}"
    );
}

#[test]
fn a_dtype_the_field_names_but_no_sub_key_carries_is_refused_by_both_decoders() {
    // Each is the control above with exactly one dtype changed to one no sub-key carries.
    for field in [
        "ctll/d16/u8/f32/bf16/st/qf16/g32",    // the weight
        "ctll/d16/i4/f32/f64/st/qf16/g32",     // the output
        "ctll/d16/i4/f32/bf16/st/qf8e8m0/g32", // the scale
    ] {
        both_refuse(&ctr(OPS, field), TokenDecline::OperandDtypeDisagrees);
    }
}

#[test]
fn a_missing_stray_or_out_of_set_block_length_is_refused_by_both_decoders() {
    for field in [
        "ctll/d16/i4/f32/bf16/st/qf16",     // a scale with no block length
        "ctll/d16/i4/f32/bf16/st/g32",      // a block length with no scale
        "ctll/d16/i4/f32/bf16/st/qf16/g64", // outside {16, 32}
    ] {
        both_refuse(&ctr(OPS, field), TokenDecline::BadBlockLength);
    }
}

#[test]
fn a_sub_key_that_disagrees_with_the_cell_on_its_dtype_is_refused_by_both_decoders() {
    let field = "ctll/d16/i4/f32/bf16/st/qf16/g32";
    // the first sub-key's dtype is not the cell's primary dtype
    both_refuse(
        &ctr(
            "co/00/v4/d16/f/f32/dl;co/00/v4/d16/f/f16;co/00/v4/d16/f/bf16",
            field,
        ),
        TokenDecline::PrimaryDtypeMismatch,
    );
    // a sub-key without its dtype
    both_refuse(
        &ctr(
            "co/00/v4/d16/f;co/00/v4/d16/f/f16;co/00/v4/d16/f/bf16",
            field,
        ),
        TokenDecline::BadOperandSubKey,
    );
}

#[test]
fn a_block_length_is_spelled_canonically() {
    // `g032` and `g+32` parse as the number 32 under a bare integer parse; the closed set is spelled, not computed.
    for g in ["g032", "g+32", "g0x20"] {
        let token = ctr(OPS, &format!("ctll/d16/i4/f32/bf16/st/qf16/{g}"));
        assert_eq!(StructureKey::from_token(&token), None, "{g}");
        assert_eq!(
            StructureKey::parse_token(&token),
            Err(TokenDecline::BadBlockLength),
            "{g}"
        );
    }
}

#[test]
fn the_first_sub_key_dtype_must_be_the_primary_dtype_in_both_decoders() {
    // A non-contraction cell, so nothing but the primary-dtype rule can refuse it (the control parses).
    let ok = "sk8|ew2|f32|cuda:sm89|ix32|grid|r2|co/00/v4/d16/f/f32;co/00/v4/d16/f/f16;co/00/v4/d16/f/f32|-";
    assert!(StructureKey::parse_token(ok).is_ok());
    assert!(StructureKey::from_token(ok).is_some());
    let bad = ok.replacen("f/f32;co/00/v4/d16/f/f16", "f/f16;co/00/v4/d16/f/f16", 1);
    assert_eq!(
        StructureKey::parse_token(&bad),
        Err(TokenDecline::PrimaryDtypeMismatch)
    );
    assert_eq!(StructureKey::from_token(&bad), None);
}
