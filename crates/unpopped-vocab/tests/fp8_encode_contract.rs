//! What the FP8 **encoders** do, which nothing called until 2026-09-02.
//!
//! # The hole this closes
//!
//! Every FP8 assertion in this workspace ran over the DECODE table — 256 bit
//! patterns through `to_f32`, checked against the OCP lattice. That is a good
//! test and it cannot see a thing about the other direction. `from_f32` had zero
//! callers in the entire test suite, so its contract was whatever the upstream
//! `float8` crate happened to implement, and its doc comment was a description of
//! that behaviour rather than a requirement placed on it.
//!
//! It had drifted. `float8` 0.7.0 saturates EVERY E5M2 overflow to max-finite
//! `0x7B` — including a literal `f32::INFINITY` — while `unpopped`'s oracle
//! emitted `0x7C` on overflow, as IEEE requires for a format that has infinities.
//! **Two contradictory contracts for one conversion, in one workspace, live.**
//! A differential could not catch it: the oracle and the codec were never
//! compared to each other on this path, and each was self-consistent.
//!
//! # The two formats differ, and that is the point
//!
//! `e4m3fn` has NO infinities, so saturating to `448` on overflow is correct and
//! is what NVIDIA's `SATFINITE` does. `e5m2` is IEEE-shaped and DOES have
//! infinities (§6.1-0011), so the same input must produce `0x7C`. A test that
//! checked only one format would read as coverage of "FP8 overflow" while
//! asserting the opposite rule for half of it.

use unpopped_vocab::{Fp8E4M3FN, Fp8E5M2};

/// E5M2 max-finite. The next power up is 65536; the midpoint is 61440.
const E5M2_MAX_FINITE: f32 = 57344.0;
const E5M2_OVERFLOW_MIDPOINT: f32 = 61440.0;

#[test]
fn an_e5m2_infinity_encodes_to_infinity_rather_than_saturating() {
    assert_eq!(
        Fp8E5M2::from_f32(f32::INFINITY).0,
        0x7c,
        "e5m2 HAS infinities (§6.1-0011), so an infinity must encode as one. \
         Saturating to 0x7B turns an overflow into the finite value 57344, which \
         reads downstream as a real measurement rather than an overflow."
    );
    assert_eq!(
        Fp8E5M2::from_f32(f32::NEG_INFINITY).0,
        0xfc,
        "the sign must survive the overflow"
    );
}

#[test]
fn the_e5m2_overflow_boundary_rounds_ties_to_infinity() {
    // Below the midpoint: nearest is max-finite.
    assert_eq!(
        Fp8E5M2::from_f32(E5M2_MAX_FINITE).0,
        0x7b,
        "max-finite itself is exactly representable"
    );
    assert_eq!(
        Fp8E5M2::from_f32(E5M2_OVERFLOW_MIDPOINT - 1.0).0,
        0x7b,
        "just below the midpoint, max-finite is nearer"
    );

    // AT the midpoint, ties-to-even picks the candidate whose trailing
    // significand bits are even: the overflow (`00`), not 0x7B (`11`).
    assert_eq!(
        Fp8E5M2::from_f32(E5M2_OVERFLOW_MIDPOINT).0,
        0x7c,
        "the halfway case rounds to infinity under ties-to-even. This is the \
         single input the oracle's `>` got wrong while its own comment said \
         `>=`, so it is pinned rather than left to the operator."
    );
    assert_eq!(
        Fp8E5M2::from_f32(E5M2_OVERFLOW_MIDPOINT + 1.0).0,
        0x7c,
        "above the midpoint is unambiguously an overflow"
    );
    assert_eq!(
        Fp8E5M2::from_f32(1e30).0,
        0x7c,
        "far overflow is still infinity"
    );
}

#[test]
fn an_e4m3fn_overflow_saturates_because_that_format_has_no_infinity() {
    // The OPPOSITE rule from e5m2 above, for a defensible reason: there is no
    // encoding to overflow TO. 0x7E is max-finite 448; 0x7F is the sole NaN.
    assert_eq!(
        Fp8E4M3FN::from_f32(f32::INFINITY).0,
        0x7e,
        "e4m3fn defines no infinity, so SATFINITE clamping to 448 is correct"
    );
    assert_eq!(Fp8E4M3FN::from_f32(-f32::INFINITY).0, 0xfe);
    assert_eq!(
        Fp8E4M3FN::from_f32(1e30).0,
        0x7e,
        "a large finite overflow saturates for the same reason"
    );
    assert_ne!(
        Fp8E4M3FN::from_f32(f32::INFINITY).0,
        0x7f,
        "saturation must not land on the NaN encoding — an overflow is not a NaN"
    );
}

#[test]
fn a_nan_stays_a_nan_through_both_encoders() {
    assert!(
        Fp8E5M2::from_f32(f32::NAN).to_f32().is_nan(),
        "e5m2 must not turn a NaN into an infinity or a finite value"
    );
    assert!(
        Fp8E4M3FN::from_f32(f32::NAN).to_f32().is_nan(),
        "e4m3fn has exactly one NaN per sign and a NaN input must reach it"
    );
    // Distinct from the overflow case above, which is the confusion this guards:
    // an infinity is not a NaN, and 448/57344 are not NaN either.
    assert!(!Fp8E4M3FN::from_f32(f32::INFINITY).to_f32().is_nan());
    assert!(!Fp8E5M2::from_f32(f32::INFINITY).to_f32().is_nan());
}

#[test]
fn ordinary_values_still_round_trip_through_the_upstream_encoder() {
    // A positive control for the three tests above: the overflow interception
    // must not have disturbed the in-range path, which still delegates.
    for v in [0.0f32, 1.0, -1.0, 0.5, 2.0, 256.0, -57344.0] {
        let back = Fp8E5M2::from_f32(v).to_f32();
        assert_eq!(
            back, v,
            "{v} is exactly representable in e5m2 and must survive unchanged"
        );
    }
    for v in [0.0f32, 1.0, -1.0, 0.5, 2.0, 256.0, 448.0] {
        let back = Fp8E4M3FN::from_f32(v).to_f32();
        assert_eq!(back, v, "{v} is exactly representable in e4m3fn");
    }
}
