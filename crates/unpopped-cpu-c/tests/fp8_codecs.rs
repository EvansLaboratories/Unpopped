//! FP8 `f8e4m3fn` and `f8e5m2` — **every** one of the 256 bit patterns.
//!
//! # Why this file can prove what the others can only sample
//!
//! An 8-bit float's entire domain is 256 patterns. "Supports every possible
//! value" is normally a claim you argue for; here it is a claim you can
//! *enumerate*. Every test below walks all 256.
//!
//! # Two dtypes, not four
//!
//! `f8e8m0` and `f8e6m2` are **not** here, and their absence is the point.
//! KISS-CLASSIFY §6.1-0013 makes them the per-block shared **scale** of an
//! MX-encoded operand, "carried as a sibling operand, never an element value
//! dtype" — and §6.2-0002's float special-value list excludes them for the same
//! reason. Implementing them as compute dtypes would be a category error, not
//! progress: a kernel does not compute *in* a scale, it uses one to dequantize
//! the block it scales.
//!
//! # The independence requirement
//!
//! These decoders are written from the format definitions, not shared with any
//! emitter's. A differential oracle that reuses the implementation it is checking
//! is not a differential — it agrees with itself by construction.

use unpopped::oracle::TypedBuffer;
use unpopped_vocab::ElementKind;

const E4M3: ElementKind = ElementKind::Fp8E4M3FN;
const E5M2: ElementKind = ElementKind::Fp8E5M2;

fn decode_all(dt: ElementKind) -> Vec<f64> {
    let all: Vec<u8> = (0u8..=255).collect();
    TypedBuffer::from_fp8_bits(dt, &[256], &all).to_f64_vec()
}

/// **`f8e4m3fn` has no infinities and exactly one NaN encoding per sign.**
///
/// This is the trap for anyone assuming IEEE shape: `0x7E` is `448`, the maximum
/// finite value — not infinity. Only `S.1111.111` is NaN. A decoder that treats
/// all-ones-exponent as special returns the wrong *number*, not merely the wrong
/// classification, and 448 vs inf is the kind of difference that survives a
/// tolerance check.
#[test]
fn e4m3fn_has_no_infinities_and_one_nan_per_sign() {
    let v = decode_all(E4M3);
    assert_eq!(v.len(), 256);

    let infs = v.iter().filter(|x| x.is_infinite()).count();
    assert_eq!(
        infs, 0,
        "f8e4m3fn defines no infinity encodings (§6.1-0010)"
    );

    let nans: Vec<usize> = v
        .iter()
        .enumerate()
        .filter(|(_, x)| x.is_nan())
        .map(|(i, _)| i)
        .collect();
    assert_eq!(
        nans,
        vec![0x7f, 0xff],
        "exactly one NaN encoding per sign — S.1111.111"
    );

    assert_eq!(
        v[0x7e], 448.0,
        "0x7E is the maximum finite value, not infinity"
    );
    assert_eq!(v[0xfe], -448.0);
}

/// **`f8e5m2` is IEEE-shaped**: infinities and multiple NaN encodings.
///
/// The contrast with E4M3 is the reason both need their own decoder rather than
/// one parameterised by widths.
#[test]
fn e5m2_is_ieee_shaped_with_infinities() {
    let v = decode_all(E5M2);
    assert_eq!(v.len(), 256);

    assert!(v[0x7c].is_infinite() && v[0x7c] > 0.0, "0x7C is +inf");
    assert!(v[0xfc].is_infinite() && v[0xfc] < 0.0, "0xFC is -inf");
    let nans = v.iter().filter(|x| x.is_nan()).count();
    assert_eq!(nans, 6, "three NaN mantissas per sign");

    assert_eq!(v[0x7b], 57344.0, "maximum finite magnitude (§6.1-0011)");
    assert_eq!(v[0xfb], -57344.0);
}

/// Both formats distinguish `+0` from `-0` by bit pattern, per §6.2-0002.
#[test]
fn signed_zero_is_distinguishable_in_both() {
    for dt in [E4M3, E5M2] {
        let v = decode_all(dt);
        assert_eq!(v[0x00], 0.0, "{dt:?}: 0x00 is +0");
        assert_eq!(v[0x80], 0.0, "{dt:?}: 0x80 is numerically zero");
        assert!(
            v[0x80].is_sign_negative(),
            "{dt:?}: 0x80 must be -0, distinguishable by sign bit"
        );
        assert!(!v[0x00].is_sign_negative());
    }
}

/// **Pinned absolute values, computed from the format definition rather than
/// from this codec.**
///
/// Everything else in this file is self-referential in a way that took a
/// surviving mutation to expose: `every_bit_pattern_round_trips` cannot catch a
/// wrong *decoder*, because the encoder finds the nearest pattern **by searching
/// with that same decoder**. Decode-then-encode agrees with itself by
/// construction no matter what the decoder says a pattern means. Seeding E5M2's
/// subnormal scale as `2^-15` instead of `2^-16` left every other test green.
///
/// These numbers are derived from the format parameters — bias, mantissa width —
/// and can be checked against the OCP OFP8 specification without running
/// anything. They are the only assertions here that would notice if the whole
/// codec were consistently wrong.
#[test]
fn the_pinned_magnitudes_match_the_format_definitions() {
    let e4 = decode_all(E4M3);
    // E4M3: bias 7, 3 mantissa bits.
    //   smallest subnormal = 1/8 * 2^(1-7) = 2^-9
    //   smallest normal    = 2^(1-7)       = 2^-6
    assert_eq!(e4[0x01], 2f64.powi(-9), "E4M3 smallest subnormal");
    assert_eq!(e4[0x08], 2f64.powi(-6), "E4M3 smallest normal");
    assert_eq!(e4[0x7e], 448.0, "E4M3 max finite (§6.1-0010)");
    assert_eq!(
        e4[0x38], 1.0,
        "E4M3 exponent bias check: 1.0 is exp=7, mant=0"
    );

    let e5 = decode_all(E5M2);
    // E5M2: bias 15, 2 mantissa bits.
    //   smallest subnormal = 1/4 * 2^(1-15) = 2^-16
    //   smallest normal    = 2^(1-15)       = 2^-14
    assert_eq!(e5[0x01], 2f64.powi(-16), "E5M2 smallest subnormal");
    assert_eq!(e5[0x04], 2f64.powi(-14), "E5M2 smallest normal");
    assert_eq!(e5[0x7b], 57344.0, "E5M2 max finite (§6.1-0011)");
    assert_eq!(
        e5[0x3c], 1.0,
        "E5M2 exponent bias check: 1.0 is exp=15, mant=0"
    );
}

/// Subnormals decode to distinct, ordered, non-zero values.
///
/// A decoder that mishandles the `exp == 0` branch typically collapses the whole
/// subnormal range to zero — which looks harmless until it silently deletes the
/// smallest representable magnitudes.
#[test]
fn subnormals_are_distinct_and_ordered() {
    for (dt, count) in [(E4M3, 7usize), (E5M2, 3usize)] {
        let v = decode_all(dt);
        let subs: Vec<f64> = (1..=count).map(|i| v[i]).collect();
        assert!(
            subs.iter().all(|x| *x > 0.0),
            "{dt:?}: subnormals must be non-zero"
        );
        assert!(
            subs.windows(2).all(|w| w[0] < w[1]),
            "{dt:?}: subnormals must increase with the mantissa: {subs:?}"
        );
    }
}

/// **Round-trip: every one of the 256 patterns decodes and re-encodes to itself.**
///
/// The exhaustive statement of "supports every possible value". NaN is compared
/// by classification rather than equality (NaN != NaN), and the two formats' NaN
/// patterns differ, so each is checked against its own decode.
#[test]
fn every_bit_pattern_round_trips() {
    for dt in [E4M3, E5M2] {
        let all: Vec<u8> = (0u8..=255).collect();
        let decoded = TypedBuffer::from_fp8_bits(dt, &[256], &all).to_f64_vec();

        for (pat, &value) in all.iter().zip(decoded.iter()) {
            let bits = TypedBuffer::from_f64_as(dt, &[1], &[value]).bits_at(0) as u8;
            let back = TypedBuffer::from_fp8_bits(dt, &[1], &[bits]).to_f64_vec()[0];
            if value.is_nan() {
                assert!(
                    back.is_nan(),
                    "{dt:?}: pattern {pat:#04x} is NaN and must re-encode to some NaN, got {back}"
                );
            } else {
                assert_eq!(
                    back, value,
                    "{dt:?}: pattern {pat:#04x} decoded to {value} but re-encoded to {back}"
                );
            }
        }
    }
}
