//! The oracle's public constructors that no test had ever called.
//!
//! # Why these three
//!
//! Found by a conversion-direction audit: for each representation-mapping pair,
//! count the test call sites in EACH direction. `TypedBuffer`'s readers were heavily
//! driven — `to_f64_vec` 18 sites, `to_i128_vec` 17, `to_complex_vec` 9 — while
//! three of its seventeen public constructors had **zero**, in tests and in the
//! oracle itself:
//!
//! ```text
//! from_f64        0    from_u16        0    from_f16_bits   0
//! ```
//!
//! A large N on the read side reads as thoroughness and says nothing about the
//! write side. The precedent is `Fp8E5M2::from_f32`, which had zero callers and
//! was wrong — it disagreed with this same oracle on every overflow, and the
//! 256-pattern decode table could not see it.
//!
//! These three turn out to be correct. That is worth pinning rather than
//! discarding: an unexercised public constructor is one refactor away from
//! being wrong with nothing to say so, and this crate's oracle is the reference
//! every backend is checked against.

use unpopped::oracle::TypedBuffer;

#[test]
fn from_f64_round_trips_including_specials() {
    let vals = [0.0f64, -0.0, 1.0, -1.0, f64::MAX, f64::MIN_POSITIVE, 1e-300];
    let t = TypedBuffer::from_f64(&[vals.len() as i64], &vals);
    let back = t.to_f64_vec();
    assert_eq!(back.len(), vals.len(), "element count must survive");
    for (i, (&want, &got)) in vals.iter().zip(back.iter()).enumerate() {
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "index {i}: f64 is the widest float this oracle carries, so a store \
             and load must be BIT-exact, not merely numerically equal"
        );
    }
}

#[test]
fn from_f64_preserves_a_nan_payload_rather_than_canonicalising_it() {
    // Distinct from the value round-trip above: a constructor that rebuilt each
    // element through an arithmetic path instead of copying bits would pass the
    // finite cases and quietly canonicalise this one.
    let sig = f64::from_bits(0x7FF4_0000_0000_0001);
    let t = TypedBuffer::from_f64(&[1], &[sig]);
    assert_eq!(
        t.to_f64_vec()[0].to_bits(),
        0x7FF4_0000_0000_0001,
        "the exact NaN encoding must survive a store/load unchanged"
    );
}

#[test]
fn from_u16_reads_back_unsigned_rather_than_sign_extended() {
    // The whole point of a U16 buffer: 0xFFFF is 65535, not -1. A reader that
    // sign-extended would return -1 here and be self-consistent about it.
    let vals = [0u16, 1, 32767, 32768, 65534, u16::MAX];
    let t = TypedBuffer::from_u16(&[vals.len() as i64], &vals);
    let back = t.to_i128_vec();
    for (i, (&want, &got)) in vals.iter().zip(back.iter()).enumerate() {
        assert_eq!(
            got,
            i128::from(want),
            "index {i}: u16 {want} must read back as {want}, not as a negative"
        );
    }
    assert!(
        back.iter().all(|&v| v >= 0),
        "no unsigned value can read back negative: {back:?}"
    );
}

#[test]
fn from_f16_bits_decodes_the_half_lattice() {
    // Raw IEEE binary16 patterns, chosen so a byte-order or a
    // wrong-width decode changes the answer visibly.
    let cases: [(u16, f64); 6] = [
        (0x0000, 0.0),
        (0x3C00, 1.0),
        (0xBC00, -1.0),
        (0x4000, 2.0),
        (0x3555, 0.333_251_953_125), // nearest half to 1/3
        (0x7BFF, 65504.0),           // max finite half
    ];
    let bits: Vec<u16> = cases.iter().map(|&(b, _)| b).collect();
    let t = TypedBuffer::from_f16_bits(&[bits.len() as i64], &bits);
    let back = t.to_f64_vec();
    for (i, (&(pat, want), &got)) in cases.iter().zip(back.iter()).enumerate() {
        assert_eq!(
            got, want,
            "index {i}: half pattern 0x{pat:04X} decodes to {want}, got {got}"
        );
    }
}

#[test]
fn from_f16_bits_carries_infinity_and_nan_through() {
    // 0x7C00 is +inf in binary16 and 0x7E00 is a quiet NaN. A decode that
    // treated the payload as a finite integer would give a large number here
    // and still look plausible.
    let t = TypedBuffer::from_f16_bits(&[3], &[0x7C00, 0xFC00, 0x7E00]);
    let back = t.to_f64_vec();
    assert!(
        back[0].is_infinite() && back[0] > 0.0,
        "0x7C00 is +inf, got {}",
        back[0]
    );
    assert!(
        back[1].is_infinite() && back[1] < 0.0,
        "0xFC00 is -inf, got {}",
        back[1]
    );
    assert!(back[2].is_nan(), "0x7E00 is a NaN, got {}", back[2]);
}
