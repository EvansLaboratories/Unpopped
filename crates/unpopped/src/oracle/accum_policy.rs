// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The oracle's models of reduced-precision accumulation** (ROADMAP U7 S1a; `docs/accum-policies.md`). A tensor core does
//! not add products in `f64`: it converts the inputs, adds a short run of exact products, and rounds into a narrow
//! accumulator. These models state that structure so an emitter can be held to it, not to a particular chip:
//!
//! * **Input conversion** is exact and stated per policy (TF32: 10 mantissa bits, ties away from zero; the `x3` forms add
//!   a remainder term; bf16: nearest even).
//! * **Products** of converted inputs are exact in `f64`.
//! * **The accumulator** is a narrow type (`f32`, `f16`) that is rounded after every `k_step` contracted elements:
//!   `acc = round(acc + sum of the step's products)`, steps ascending, the sum inside a step in `f64` in ascending order.
//!
//! Unpopped extension, not KISS (KISS pins no accumulator width: KISS-OPS-6.0-0004).

use super::{bf16_to_f64, f32_to_bf16_bits};
use crate::ir::AccumSpec;

/// Round `x` (an `f32`) to TF32: keep 10 explicit mantissa bits, ties away from zero (`cvt.rna.tf32.f32`). Infinities and
/// NaN are unchanged; a value that rounds up past the largest finite `f32` becomes infinite, as the instruction does.
#[must_use]
pub(super) fn tf32(x: f64) -> f64 {
    let f = x as f32;
    if !f.is_finite() {
        return f64::from(f);
    }
    let bits = f.to_bits();
    f64::from(f32::from_bits(bits.wrapping_add(0x1000) & 0xFFFF_E000))
}

/// Round to bf16 (nearest even). Every caller passes a value an `f32` holds exactly (an input or an input's remainder), so the
/// `f32` hop is lossless.
fn to_bf16(x: f64) -> f64 {
    bf16_to_f64(f32_to_bf16_bits(x as f32))
}

/// Round an `f64` to the nearest binary16 value (ties to even), directly, with no intermediate `f32` (a sum of products
/// need not be an `f32`). Beyond 65520 the result is infinite; below the smallest subnormal's half it is zero.
#[must_use]
pub(super) fn to_f16(x: f64) -> f64 {
    if !x.is_finite() || x == 0.0 {
        return x;
    }
    let mag = x.abs();
    if mag >= 65520.0 {
        return f64::INFINITY.copysign(x);
    }
    // the quantum of a binary16 of this magnitude: 2^(e - 10) for a normal value, 2^-24 for a subnormal one
    let e = ((mag.to_bits() >> 52) & 0x7ff) as i32 - 1023; // the unbiased exponent, exact (no `log2` rounding)
    let quantum = f64::powi(2.0, e.max(-14) - 10);
    (mag / quantum).round_ties_even() * quantum * x.signum()
}

fn to_f32(x: f64) -> f64 {
    f64::from(x as f32)
}

/// The product of one `(a, b)` pair under the policy's input conversion.
fn product(policy: &AccumSpec, a: f64, b: f64) -> f64 {
    let split = |round: fn(f64) -> f64, a: f64, b: f64| {
        let (ah, bh) = (round(a), round(b));
        let (al, bl) = (round(a - ah), round(b - bh));
        ah * bh + ah * bl + al * bh
    };
    match policy {
        AccumSpec::Tf32 { .. } => tf32(a) * tf32(b),
        AccumSpec::Tf32x3 { .. } => split(tf32, a, b),
        AccumSpec::Bf16x3 { .. } => split(to_bf16, a, b),
        _ => a * b,
    }
}

/// The K-sum of `k` pairs under a reduced-precision `policy`, as the accumulator's value widened to `f64`. `pair(i)` is
/// the `i`-th `(a, b)` pair, ascending. `None` for [`AccumSpec::WideFloat`] (the caller's own path).
pub(super) fn contract(
    policy: &AccumSpec,
    k: i64,
    pair: impl Fn(i64) -> (f64, f64),
) -> Option<f64> {
    let (step, round): (u32, fn(f64) -> f64) = match *policy {
        AccumSpec::Tf32 { k_step }
        | AccumSpec::Tf32x3 { k_step }
        | AccumSpec::Bf16x3 { k_step } => (k_step, to_f32),
        AccumSpec::F16Accum { k_step } => (k_step, to_f16),
        _ => return None,
    };
    let step = i64::from(step.max(1));
    let mut acc = 0.0f64;
    let mut at = 0;
    while at < k {
        let end = (at + step).min(k);
        let mut sum = 0.0f64;
        for i in at..end {
            let (a, b) = pair(i);
            sum += product(policy, a, b);
        }
        acc = round(acc + sum);
        at = end;
    }
    Some(acc)
}

#[cfg(test)]
mod tests {
    use super::{tf32, to_f16};

    /// A tiny deterministic generator (no `rand` dependency): xorshift64.
    fn next(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    /// The nearest binary16 to `x` by integer arithmetic on the exact bits: a different mechanism from `to_f16`'s scaling.
    fn exact_f16(x: f64) -> f64 {
        let bits = x.to_bits();
        let sign = if bits >> 63 == 1 { -1.0 } else { 1.0 };
        let biased = ((bits >> 52) & 0x7ff) as i32;
        if biased == 0 {
            return 0.0 * sign; // an f64 subnormal is far below half the smallest binary16
        }
        let mant = (bits & ((1 << 52) - 1)) | (1 << 52); // 53-bit integer significand
        let e = biased - 1075; // x = mant * 2^e
        let qe = (biased - 1023).max(-14) - 10; // the binary16 quantum is 2^qe
        let shift = qe - e; // mant * 2^e / 2^qe = mant / 2^shift
        let n: u128 = if shift <= 0 {
            u128::from(mant) << (-shift)
        } else if shift > 80 {
            0
        } else {
            let (q, r) = (
                u128::from(mant) >> shift,
                u128::from(mant) & ((1u128 << shift) - 1),
            );
            let half = 1u128 << (shift - 1);
            if r > half || (r == half && q & 1 == 1) {
                q + 1
            } else {
                q
            }
        };
        let v = n as f64 * f64::powi(2.0, qe);
        sign * if v >= 65520.0 { f64::INFINITY } else { v }
    }

    #[test]
    fn to_f16_agrees_with_exact_integer_rounding_over_the_whole_range() {
        let mut s = 0x9E37_79B9_7F4A_7C15u64;
        for _ in 0..200_000 {
            // random magnitudes from the f16 subnormals to well past overflow, both signs, full 53-bit mantissas
            let exp = (next(&mut s) % 60) as i32 - 30;
            let mant = 1.0 + (next(&mut s) >> 11) as f64 / (1u64 << 53) as f64;
            let x = mant * f64::powi(2.0, exp) * if next(&mut s) & 1 == 0 { 1.0 } else { -1.0 };
            assert_eq!(to_f16(x).to_bits(), exact_f16(x).to_bits(), "x = {x:e}");
        }
        // the boundaries: ties around the largest finite, the subnormal limit, exact halves
        for x in [
            65504.0,
            65519.999_999,
            65520.0,
            6.1e-5,
            5.96e-8,
            2.98e-8,
            2.980_232_238_769_531_3e-8,
            1.0,
            2049.0,
            2051.0,
        ] {
            assert_eq!(to_f16(x).to_bits(), exact_f16(x).to_bits(), "x = {x:e}");
            assert_eq!(to_f16(-x).to_bits(), exact_f16(-x).to_bits(), "x = -{x:e}");
        }
    }

    /// For a value an `f32` holds exactly the independent `half` codec is a fair judge. (For a general `f64` it is not:
    /// `half::f16::from_f64(0.04518128032270489)` returns 0.045166015625 although the value is 1480.5002 quanta from
    /// zero and so rounds up to 1481, the exact integer rounding above and this module's answer.)
    #[test]
    fn to_f16_agrees_with_the_half_codec_on_f32_values() {
        let mut s = 0xD1B5_4A32_D192_ED03u64;
        for _ in 0..200_000 {
            let bits = next(&mut s) as u32;
            let x = f32::from_bits(bits);
            if !x.is_finite() {
                continue;
            }
            let want = f64::from(half::f16::from_f32(x));
            assert_eq!(to_f16(f64::from(x)).to_bits(), want.to_bits(), "x = {x:e}");
        }
    }

    #[test]
    fn tf32_keeps_ten_mantissa_bits_ties_away_from_zero() {
        assert_eq!(tf32(1.0 + f64::powi(2.0, -11)), 1.0 + f64::powi(2.0, -10));
        assert_eq!(
            tf32(-(1.0 + f64::powi(2.0, -11))),
            -(1.0 + f64::powi(2.0, -10))
        );
        assert_eq!(tf32(1.0 + f64::powi(2.0, -12)), 1.0);
        assert!(tf32(f64::INFINITY).is_infinite());
        assert!(tf32(f64::NAN).is_nan());
        // rounding the largest finite f32 up overflows to infinity, as the instruction does
        assert!(tf32(f64::from(f32::MAX)).is_infinite());
    }
}
