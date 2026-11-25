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
    if let AccumSpec::Fp8FastAccum {
        k_step,
        promote_every,
        mantissa_bits,
    } = *policy
    {
        return Some(contract_fast(
            k,
            k_step,
            promote_every,
            mantissa_bits,
            &pair,
        ));
    }
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

/// Round an `f64` to `bits` explicit mantissa bits, nearest even, exponent range of `f32` (overflow is infinite).
fn to_mantissa(x: f64, bits: u8) -> f64 {
    if !x.is_finite() || x == 0.0 {
        return x;
    }
    let e = ((x.abs().to_bits() >> 52) & 0x7ff) as i32 - 1023;
    let quantum = f64::powi(2.0, e.max(-126) - i32::from(bits));
    let r = (x.abs() / quantum).round_ties_even() * quantum * x.signum();
    if r.abs() > f64::from(f32::MAX) {
        f64::INFINITY.copysign(x)
    } else {
        r
    }
}

/// [`AccumSpec::Fp8FastAccum`]: an imprecise accumulator of `bits` mantissa bits rounded after every `k_step` elements,
/// added into an `f32` accumulator every `promote_every` steps and at the end.
fn contract_fast(
    k: i64,
    k_step: u32,
    promote_every: u32,
    bits: u8,
    pair: &impl Fn(i64) -> (f64, f64),
) -> f64 {
    let step = i64::from(k_step.max(1));
    let every = i64::from(promote_every.max(1));
    let mut main = 0.0f64;
    let mut fast = 0.0f64;
    let (mut at, mut steps) = (0, 0);
    while at < k {
        let end = (at + step).min(k);
        let sum: f64 = (at..end)
            .map(|i| {
                let (a, b) = pair(i);
                a * b
            })
            .sum();
        fast = to_mantissa(fast + sum, bits);
        steps += 1;
        at = end;
        if steps == every || at >= k {
            main = to_f32(main + fast);
            fast = 0.0;
            steps = 0;
        }
    }
    main
}

/// The K-sum of `k` integer pairs under an integer policy ([`AccumSpec::Int32`]: the exact sum wrapped to `i32`;
/// [`AccumSpec::Int32Sat`]: clamped into the `i32` range after every `k_step` elements), as the accumulator's value.
/// `None` for every other policy. The operands are integers an `f64` holds exactly.
pub(super) fn contract_int(
    policy: &AccumSpec,
    k: i64,
    pair: impl Fn(i64) -> (f64, f64),
) -> Option<i128> {
    let term = |i: i64| {
        let (a, b) = pair(i);
        (a as i128) * (b as i128)
    };
    match *policy {
        AccumSpec::Int32 => {
            let sum: i128 = (0..k).map(term).sum();
            Some(i128::from(sum as i32))
        }
        AccumSpec::Int32Sat { k_step } => {
            let step = i64::from(k_step.max(1));
            let (lo, hi) = (i128::from(i32::MIN), i128::from(i32::MAX));
            let mut acc: i128 = 0;
            let mut at = 0;
            while at < k {
                let end = (at + step).min(k);
                let sum: i128 = (at..end).map(term).sum();
                acc = (acc + sum).clamp(lo, hi);
                at = end;
            }
            Some(acc)
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::{AccumSpec, contract, tf32, to_f16};

    /// Round an integer to `bits` explicit mantissa bits (`bits + 1` significant bits), nearest even, by its bit
    /// length: a different mechanism from `to_mantissa`'s scaling of an `f64`.
    fn round_bits(x: i64, bits: u32) -> i64 {
        let n = x.unsigned_abs();
        let len = 64 - n.leading_zeros();
        if len <= bits + 1 {
            return x;
        }
        let shift = len - (bits + 1);
        let (q, rem, half) = (n >> shift, n & ((1 << shift) - 1), 1u64 << (shift - 1));
        let q = if rem > half || (rem == half && q & 1 == 1) {
            q + 1
        } else {
            q
        };
        ((q << shift) as i64) * x.signum()
    }

    /// The `Fp8FastAccum` model in quarters (every product here is a multiple of 1/4), written from the policy text.
    fn fast_reference(k_step: usize, every: usize, bits: u32, a: &[f64], b: &[f64]) -> f64 {
        let quarters = |x: f64| (x * 4.0) as i64;
        let (mut main, mut fast, mut steps) = (0i64, 0i64, 0usize);
        let n_steps = a.len().div_ceil(k_step);
        for (s, (ca, cb)) in a.chunks(k_step).zip(b.chunks(k_step)).enumerate() {
            let sum: i64 = ca.iter().zip(cb).map(|(&x, &y)| quarters(x * y)).sum();
            fast = round_bits(fast + sum, bits);
            steps += 1;
            if steps == every || s + 1 == n_steps {
                main += fast;
                fast = 0;
                steps = 0;
            }
        }
        main as f64 / 4.0
    }

    /// 24 pairs: a product of 256 absorbs the 0.25 products behind it in a narrow accumulator; a negative 256 follows.
    fn fast_corpus() -> (Vec<f64>, Vec<f64>) {
        let a: Vec<f64> = (0..24)
            .map(|i| match i {
                0 => 16.0,
                1..=3 | 8..=10 => 0.0, // the first step holds the 256 alone, the third the -256
                11 => -16.0,
                20..=23 => 1.5,
                _ => 0.5,
            })
            .collect();
        let b: Vec<f64> = (0..24)
            .map(|i| if i == 0 || i == 11 { 16.0 } else { 0.5 })
            .collect();
        (a, b)
    }

    fn fast(k_step: u32, every: u32, bits: u8, a: &[f64], b: &[f64]) -> f64 {
        let p = AccumSpec::Fp8FastAccum {
            k_step,
            promote_every: every,
            mantissa_bits: bits,
        };
        contract(&p, a.len() as i64, |i| (a[i as usize], b[i as usize])).expect("a policy")
    }

    #[test]
    fn fp8_fast_accum_matches_an_integer_reference() {
        let (a, b) = fast_corpus();
        for (k_step, every, bits) in [(4u32, 2u32, 3u8), (2, 3, 2), (8, 1, 4), (3, 4, 5), (24, 1, 3), (1, 24, 1)] {
            let want = fast_reference(k_step as usize, every as usize, u32::from(bits), &a, &b);
            assert_eq!(fast(k_step, every, bits, &a, &b), want, "k_step {k_step} every {every} bits {bits}");
        }
    }

    #[test]
    fn fp8_fast_accum_corpus_separates_every_parameter_from_an_exact_sum() {
        let (a, b) = fast_corpus();
        let exact: f64 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        assert_ne!(fast(4, 8, 3, &a, &b), exact, "the imprecise accumulator must lose something");
        assert_ne!(fast(4, 8, 3, &a, &b), fast(4, 8, 10, &a, &b), "bits matter");
        assert_ne!(fast(4, 1, 3, &a, &b), fast(4, 6, 3, &a, &b), "promote_every matters");
        assert_ne!(fast(1, 24, 3, &a, &b), fast(24, 1, 3, &a, &b), "k_step matters");
        assert_eq!(fast(1, 1, 23, &a, &b), exact, "23 bits, promoted every step: the exact f32 sum");
    }

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
            65_519.999_999,
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
