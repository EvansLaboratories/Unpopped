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

#[test]
fn the_strict_decode_refuses_what_parse_token_attributes() {
    // (token, the typed verdict parse_token must give). Each is the control above with exactly one thing broken.
    let cases: Vec<(String, TokenDecline)> = vec![
        // the weight dtype named in the field is carried by no sub-key
        (
            ctr(OPS, "ctll/d16/u8/f32/bf16/st/qf16/g32"),
            TokenDecline::OperandDtypeDisagrees,
        ),
        // the output dtype is carried by no sub-key
        (
            ctr(OPS, "ctll/d16/i4/f32/f64/st/qf16/g32"),
            TokenDecline::OperandDtypeDisagrees,
        ),
        // the scale dtype is carried by no sub-key
        (
            ctr(OPS, "ctll/d16/i4/f32/bf16/st/qf8e8m0/g32"),
            TokenDecline::OperandDtypeDisagrees,
        ),
        // a scale with no block length, a block length with no scale, and a length outside {16, 32}
        (
            ctr(OPS, "ctll/d16/i4/f32/bf16/st/qf16"),
            TokenDecline::BadBlockLength,
        ),
        (
            ctr(OPS, "ctll/d16/i4/f32/bf16/st/g32"),
            TokenDecline::BadBlockLength,
        ),
        (
            ctr(OPS, "ctll/d16/i4/f32/bf16/st/qf16/g64"),
            TokenDecline::BadBlockLength,
        ),
        // the first sub-key's dtype is not the cell's primary dtype
        (
            ctr(
                "co/00/v4/d16/f/f32/dl;co/00/v4/d16/f/f16;co/00/v4/d16/f/bf16",
                "ctll/d16/i4/f32/bf16/st/qf16/g32",
            ),
            TokenDecline::PrimaryDtypeMismatch,
        ),
        // a sub-key without its dtype
        (
            ctr(
                "co/00/v4/d16/f;co/00/v4/d16/f/f16;co/00/v4/d16/f/bf16",
                "ctll/d16/i4/f32/bf16/st/qf16/g32",
            ),
            TokenDecline::BadOperandSubKey,
        ),
    ];
    for (token, verdict) in cases {
        assert_eq!(
            StructureKey::parse_token(&token),
            Err(verdict),
            "typed verdict: {token}"
        );
        assert_eq!(
            StructureKey::from_token(&token),
            None,
            "the strict decode accepted what the typed one declines: {token}"
        );
    }
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
