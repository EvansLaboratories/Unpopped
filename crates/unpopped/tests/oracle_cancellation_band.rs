// SPDX-License-Identifier: MIT OR Apache-2.0
//! **What the derived fidelity band does with a cancellation that per-operation rounding changes** (ROADMAP U45; found 2026-11-23 lifting
//! `cuCmulf`). The oracle evaluates a statement's expression at f64 and rounds once at the store, so for `a*b - c*d` with
//! `a = b = x = 1 + 2^-12`, `c = 1 + 2^-11`, `d = 1` it holds `2^-24`, and so does a fused f32 kernel; an UNFUSED f32 kernel rounds
//! `x*x` to `1 + 2^-11` first and gets `0`. `required_fidelity` derives `abs: 0` (its doc says why: it cannot know the intermediate
//! magnitudes), so `Tolerant { rel, abs: 0 }` puts the unfused `0` OUTSIDE the band against the oracle's `2^-24` and a conforming
//! device is refused; the caller must widen `abs` and say why. The control: the same band accepts an ordinary one-ulp difference.

use unpopped::oracle::{Fidelity, TypedBuffer, compare};

fn band() -> Fidelity {
    // the band `required_fidelity` derives for a 3-node body (Mul, Mul, Sub) at f32: `(steps) * u`, `abs: 0`
    Fidelity::Tolerant {
        rel: 3.0 * f64::from(f32::EPSILON) / 2.0,
        abs: 0.0,
    }
}

fn one(v: f32) -> TypedBuffer {
    TypedBuffer::from_f32(&[1], &[v])
}

#[test]
fn the_unfused_zero_is_outside_a_band_with_no_absolute_floor_and_the_floor_admits_it() {
    let oracle = one(2f32.powi(-24));
    let unfused = one(0.0);
    let fused = one(2f32.powi(-24));
    // control: the fused result is the oracle's own value
    compare(&oracle, &fused, band()).expect("equal values are inside the band");
    // the cancellation: 2^-24 against 0 is a relative error of 1, far outside 3u
    assert!(
        compare(&oracle, &unfused, band()).is_err(),
        "an unfused f32 result of 0 is refused by a relative-only band"
    );
    // a caller whose cell cancels widens `abs` (here to the size of the cancelled term) and says why
    let widened = Fidelity::Tolerant {
        rel: 3.0 * f64::from(f32::EPSILON) / 2.0,
        abs: 2f64.powi(-24),
    };
    compare(&oracle, &unfused, widened).expect("an absolute floor of the cancelled term admits it");
    // control: an ordinary one-ulp difference of a non-cancelling value is inside the same relative band
    let ulp = f32::EPSILON;
    compare(
        &one(1.0),
        &one(1.0 + ulp),
        Fidelity::Tolerant {
            rel: 3.0 * f64::from(ulp),
            abs: 0.0,
        },
    )
    .expect("one ulp of 1.0 is inside 3u relative");
}
