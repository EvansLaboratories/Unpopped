// SPDX-License-Identifier: MIT OR Apache-2.0
//! The **independent reference** for [`ScalarExpr::Cast`](crate::ir::ScalarExpr::Cast)
//! and [`ScalarExpr::Bitcast`](crate::ir::ScalarExpr::Bitcast): what the conversion IS,
//! computed from exact integers and IEEE bit patterns rather than from any emitter's
//! spelling.
//!
//! Used by the oracle. Operates on `(Num, ElementKind)` so it knows nothing of the
//! oracle's private `Val`. The semantics are `docs/ir-typed-expressions.md` section 3:
//!
//! - **float to float**: round per `RoundMode`; `Saturation::ClampFinite` clamps any
//!   overflow, infinities included, to the largest finite value, NaN stays NaN.
//! - **float to integer**: round to an integer per `RoundMode`, NaN to 0, clamp to the
//!   destination range (always saturating; the type check demands `Saturation::Clamp`).
//! - **integer to float**: round per `RoundMode` (exact when the integer fits).
//! - **integer to integer**: `Wrap` truncates two's-complement, `Clamp` saturates.
//!
//! Only `{f16, bf16, f32, f32s, f64, i8, i16, i32, i64, u8, u16, u32}` are defined
//! (`types::check_cast`), plus the MX elements `f4e2m1`/`f6e2m3`/`f6e3m2` ([`encode_mx`]). The 16-bit floats go through [`encode_ieee`], one rounding from
//! the exact value; f32 and f64 keep their hardware-checked paths.

use crate::ir::{CastSpec, RoundMode, Saturation};
use crate::types::{float_format, int_range, is_cast_float_endpoint, is_mx_element, typed_bits};
use unpopped_vocab::ElementKind;

/// Whether `t` is one of the 16-bit floats, which the generic IEEE helpers below encode.
fn is_half(t: ElementKind) -> bool {
    matches!(t, ElementKind::F16 | ElementKind::Bf16)
}

/// Decode the IEEE-shaped bit pattern `bits` of a `(ebits, mbits)` float to its exact `f64`.
///
/// Exact for every format up to binary32 (and binary64 itself): it computes
/// `significand * 2^exponent` with a power-of-two scale, never a rounding step.
#[cfg(test)]
pub(crate) fn decode_ieee(bits: u128, ebits: u32, mbits: u32) -> f64 {
    let sign = if (bits >> (ebits + mbits)) & 1 == 1 {
        -1.0
    } else {
        1.0
    };
    let exp = ((bits >> mbits) & ((1u128 << ebits) - 1)) as i32;
    let mant = (bits & ((1u128 << mbits) - 1)) as f64;
    let bias = (1i32 << (ebits - 1)) - 1;
    let top = (1i32 << ebits) - 1;
    let scale = |e: i32| 2f64.powi(e);
    let mag = if exp == 0 {
        mant * scale(1 - bias - mbits as i32)
    } else if exp == top {
        if mant == 0.0 { f64::INFINITY } else { f64::NAN }
    } else {
        (1.0 + mant * scale(-(mbits as i32))) * scale(exp - bias)
    };
    sign * mag
}

/// `ax` (a positive finite value) in units of the last place of a format with `mbits`
/// mantissa bits whose exponent is `eff`, rounded to an integer under `mode`.
///
/// The scaling by a power of two is exact in `f64`, so the only rounding is the one asked for.
fn round_scaled(ax: f64, eff: i32, mbits: u32, mode: RoundMode, neg: bool) -> u128 {
    let q = ax * 2f64.powi(mbits as i32 - eff);
    let fl = q.floor();
    let frac = q - fl;
    let n = fl as u128;
    let up = match mode {
        RoundMode::NearestEven => frac > 0.5 || (frac == 0.5 && n % 2 == 1),
        RoundMode::TowardZero => false,
        RoundMode::TowardNegInf => frac > 0.0 && neg,
        RoundMode::TowardPosInf => frac > 0.0 && !neg,
    };
    n + u128::from(up)
}

/// Decode an MX element code (`1 + ebits + mbits` bits) to its exact value.
pub(crate) fn decode_mx(bits: u128, ebits: u32, mbits: u32) -> f64 {
    let sign = if (bits >> (ebits + mbits)) & 1 == 1 {
        -1.0
    } else {
        1.0
    };
    let exp = ((bits >> mbits) & ((1u128 << ebits) - 1)) as i32;
    let mant = (bits & ((1u128 << mbits) - 1)) as f64;
    let bias = (1i32 << (ebits - 1)) - 1;
    let mag = if exp == 0 {
        mant * 2f64.powi(1 - bias - mbits as i32)
    } else {
        (1.0 + mant * 2f64.powi(-(mbits as i32))) * 2f64.powi(exp - bias)
    };
    sign * mag
}

/// `x` converted to an MX element under `mode`, SATURATING: every overflow (an infinity
/// included) lands on the largest finite value of its sign, and a NaN becomes `+0`. This is
/// the IR's rule (mirroring float to integer), NOT a hardware fact: the NaN behaviour of
/// the vendors' conversion instructions is unverified here, so an emitter that cannot match
/// it must decline rather than approximate. The sign of a zero (and of a value that rounds to
/// zero) is kept.
pub(crate) fn encode_mx(x: f64, ebits: u32, mbits: u32, mode: RoundMode) -> u128 {
    if x.is_nan() {
        return 0;
    }
    let neg = x.is_sign_negative();
    let sign_bit = u128::from(neg) << (ebits + mbits);
    let max_code = (1u128 << (ebits + mbits)) - 1;
    let ax = x.abs();
    if ax == 0.0 {
        return sign_bit;
    }
    if ax.is_infinite() {
        return sign_bit | max_code;
    }
    let bias = (1i32 << (ebits - 1)) - 1;
    let emin = 1 - bias;
    let emax = ((1i32 << ebits) - 1) - bias;
    let e = (((ax.to_bits() >> 52) & 0x7ff) as i32 - 1023).max(-1022);
    if e > emax {
        return sign_bit | max_code;
    }
    let n = round_scaled(ax, e.max(emin), mbits, mode, neg);
    let bits = if e < emin {
        n
    } else {
        (((e + bias) as u128) << mbits) + (n - (1u128 << mbits))
    };
    sign_bit | bits.min(max_code)
}

/// Whether a FINITE value too large for the format rounds to infinity (rather than stopping at
/// the largest finite value) under `mode`: nearest does, and a directed mode does only when it
/// points toward the overflowing side.
fn overflows_to_infinity(mode: RoundMode, neg: bool) -> bool {
    match mode {
        RoundMode::NearestEven => true,
        RoundMode::TowardZero => false,
        RoundMode::TowardNegInf => neg,
        RoundMode::TowardPosInf => !neg,
    }
}

/// `x` rounded to the `(ebits, mbits)` IEEE-shaped float under `mode`, as that format's bits.
///
/// Single rounding from the exact value, so no double-rounding through `f32`. `clamp_finite`
/// sends every overflow (infinite input included) to the largest finite value; otherwise an
/// overflow follows the rounding direction (nearest-even goes to infinity, a directed mode
/// that points away from the overflow side stops at the largest finite value). A NaN keeps
/// its sign and becomes the format's quiet NaN.
pub(crate) fn encode_ieee(
    x: f64,
    ebits: u32,
    mbits: u32,
    mode: RoundMode,
    clamp_finite: bool,
) -> u128 {
    let sign_bit = u128::from(x.is_sign_negative()) << (ebits + mbits);
    let exp_all = (1u128 << ebits) - 1;
    let max_finite = (exp_all << mbits) - 1;
    let neg = x.is_sign_negative();
    if x.is_nan() {
        return sign_bit | (exp_all << mbits) | (1u128 << (mbits - 1));
    }
    let overflow = || {
        let to_inf = !clamp_finite && overflows_to_infinity(mode, neg);
        sign_bit | if to_inf { exp_all << mbits } else { max_finite }
    };
    let ax = x.abs();
    if ax == 0.0 {
        return sign_bit;
    }
    if ax.is_infinite() {
        // Exactly representable, so no rounding direction applies: only ClampFinite moves it.
        return sign_bit
            | if clamp_finite {
                max_finite
            } else {
                exp_all << mbits
            };
    }
    let bias = (1i32 << (ebits - 1)) - 1;
    let emin = 1 - bias;
    // Unbiased binary exponent of the exact value (f64 subnormals sit far below every
    // target format's range, so the clamp to `emin` below handles them).
    let e = (((ax.to_bits() >> 52) & 0x7ff) as i32 - 1023).max(-1022);
    if e > bias {
        return overflow();
    }
    let n = round_scaled(ax, e.max(emin), mbits, mode, neg);
    let bits = if e < emin {
        // Subnormal range: `n` is the mantissa field, and `n == 2^mbits` carries into the
        // exponent field as the smallest normal.
        n
    } else {
        // Normal: `n` in `[2^mbits, 2^(mbits+1)]`; `n == 2^(mbits+1)` carries into the exponent.
        (((e + bias) as u128) << mbits) + (n - (1u128 << mbits))
    };
    if bits > max_finite {
        return overflow();
    }
    sign_bit | bits
}

/// `i` as an `f64` that preserves every rounding decision a narrower target can make: exact
/// below 2^53, and above it truncated to 53 bits with the last bit forced odd when anything
/// was dropped (round-to-odd), so rounding it again to at most 24 bits matches rounding `i`.
fn int_to_f64_sticky(i: i128) -> f64 {
    let m = i.unsigned_abs();
    let bits = 128 - m.leading_zeros();
    let v = if bits <= 53 {
        m as f64
    } else {
        let s = bits - 53;
        let kept = m >> s;
        let lost = m & ((1u128 << s) - 1) != 0;
        ((kept | u128::from(lost)) as f64) * 2f64.powi(s as i32)
    };
    if i < 0 { -v } else { v }
}

/// A scalar value in the reference's two exact domains.
#[derive(Copy, Clone, Debug)]
pub(crate) enum Num {
    /// A float, held as an `f64` (exact for f32 and f64 values).
    F(f64),
    /// An integer, held as an `i128` (exact for every type in the table).
    I(i128),
}

/// The raw storage bits of `x` as a value of type `t`.
///
/// # Panics
///
/// If `x` is the wrong domain for `t`, or `t` is outside the typed table.
pub(crate) fn bits_of(x: Num, t: ElementKind) -> u128 {
    match (x, t) {
        (Num::F(f), ElementKind::F64) => u128::from(f.to_bits()),
        (Num::F(f), ElementKind::F32 | ElementKind::F32Strict) => u128::from((f as f32).to_bits()),
        (Num::F(f), t) if is_mx_element(t) => {
            let (e, m) = float_format(t).expect("an MX element has a format");
            encode_mx(f, e, m, RoundMode::NearestEven)
        }
        // A value HELD as a 16-bit float is already on its grid, so any mode agrees.
        (Num::F(f), t) if is_half(t) => {
            let (e, m) = float_format(t).expect("a half has a format");
            encode_ieee(f, e, m, RoundMode::NearestEven, false)
        }
        (Num::I(i), t) if int_range(t).is_some() => {
            let width = typed_bits(t).expect("integer types in the table have a width");
            (i as u128) & ((1u128 << width) - 1)
        }
        (x, t) => panic!("cast_ref: {x:?} is not a value of {t:?}"),
    }
}

/// The value stored in `bits` as type `t`.
#[cfg(test)]
pub(crate) fn num_of(bits: u128, t: ElementKind) -> Num {
    match t {
        ElementKind::F64 => Num::F(f64::from_bits(bits as u64)),
        ElementKind::F32 | ElementKind::F32Strict => Num::F(f64::from(f32::from_bits(bits as u32))),
        t if is_half(t) => {
            let (e, m) = float_format(t).expect("a half has a format");
            Num::F(decode_ieee(bits, e, m))
        }
        t if is_mx_element(t) => {
            let (e, m) = float_format(t).expect("an MX element has a format");
            Num::F(decode_mx(bits, e, m))
        }
        ElementKind::I8 => Num::I(i128::from(bits as u8 as i8)),
        ElementKind::I16 => Num::I(i128::from(bits as u16 as i16)),
        ElementKind::I32 => Num::I(i128::from(bits as u32 as i32)),
        ElementKind::I64 => Num::I(i128::from(bits as u64 as i64)),
        ElementKind::U8 => Num::I(i128::from(bits as u8)),
        ElementKind::U16 => Num::I(i128::from(bits as u16)),
        ElementKind::U32 => Num::I(i128::from(bits as u32)),
        other => panic!("cast_ref: {other:?} is outside the typed table"),
    }
}

/// Step an `f32` one unit in the last place toward `+inf` (`up`) or `-inf`.
/// (`f32::next_up` is stable only from 1.86; this crate's MSRV is 1.85.)
fn step32(x: f32, up: bool) -> f32 {
    if x.is_nan() || (x.is_infinite() && (x > 0.0) == up) {
        return x;
    }
    if x == 0.0 {
        let tiny = f32::from_bits(1);
        return if up { tiny } else { -tiny };
    }
    let b = x.to_bits();
    let away_from_zero = (x > 0.0) == up;
    f32::from_bits(if away_from_zero { b + 1 } else { b - 1 })
}

/// [`step32`] for `f64`.
fn step64(x: f64, up: bool) -> f64 {
    if x.is_nan() || (x.is_infinite() && (x > 0.0) == up) {
        return x;
    }
    if x == 0.0 {
        let tiny = f64::from_bits(1);
        return if up { tiny } else { -tiny };
    }
    let b = x.to_bits();
    let away_from_zero = (x > 0.0) == up;
    f64::from_bits(if away_from_zero { b + 1 } else { b - 1 })
}

/// `x` rounded to `f32` under `mode`, from the exact `f64` `x`.
///
/// Takes the round-to-nearest-even result (`as f32`) and, for a directed mode,
/// steps one unit toward the requested side when that result lies on the wrong side
/// of `x`. Comparing in `f64` is exact because every `f32` is an exact `f64`.
fn f64_to_f32(x: f64, mode: RoundMode) -> f32 {
    let r = x as f32;
    if x.is_nan() || f64::from(r) == x {
        return r;
    }
    let rd = f64::from(r);
    match mode {
        RoundMode::NearestEven => r,
        RoundMode::TowardZero => {
            if rd.abs() > x.abs() {
                step32(r, rd < 0.0)
            } else {
                r
            }
        }
        RoundMode::TowardNegInf => {
            if rd > x {
                step32(r, false)
            } else {
                r
            }
        }
        RoundMode::TowardPosInf => {
            if rd < x {
                step32(r, true)
            } else {
                r
            }
        }
    }
}

/// `i` rounded to `f32` under `mode`.
fn int_to_f32(i: i128, mode: RoundMode) -> f32 {
    let r = i as f32;
    // Every f32 that can appear here (|i| <= 2^64) converts to i128 exactly.
    let ri = r as i128;
    if ri == i {
        return r;
    }
    match mode {
        RoundMode::NearestEven => r,
        RoundMode::TowardZero => {
            if ri.abs() > i.abs() {
                step32(r, ri < 0)
            } else {
                r
            }
        }
        RoundMode::TowardNegInf => {
            if ri > i {
                step32(r, false)
            } else {
                r
            }
        }
        RoundMode::TowardPosInf => {
            if ri < i {
                step32(r, true)
            } else {
                r
            }
        }
    }
}

/// `i` rounded to `f64` under `mode`.
fn int_to_f64(i: i128, mode: RoundMode) -> f64 {
    let r = i as f64;
    let ri = r as i128;
    if ri == i {
        return r;
    }
    match mode {
        RoundMode::NearestEven => r,
        RoundMode::TowardZero => {
            if ri.abs() > i.abs() {
                step64(r, ri < 0)
            } else {
                r
            }
        }
        RoundMode::TowardNegInf => {
            if ri > i {
                step64(r, false)
            } else {
                r
            }
        }
        RoundMode::TowardPosInf => {
            if ri < i {
                step64(r, true)
            } else {
                r
            }
        }
    }
}

/// The integer-valued `f64` that `x` rounds to under `mode` (`NaN` and infinities
/// pass through for the caller's saturation).
fn round_to_integral(x: f64, mode: RoundMode) -> f64 {
    match mode {
        RoundMode::TowardZero => x.trunc(),
        RoundMode::TowardNegInf => x.floor(),
        RoundMode::TowardPosInf => x.ceil(),
        // ties to even: `round_ties_even` is stable from 1.77.
        RoundMode::NearestEven => x.round_ties_even(),
    }
}

/// Convert `x` (a value of type `from`) per `spec`; the result is the destination's
/// raw bits.
///
/// # Panics
///
/// If `(from, spec)` is not in the typed table (`types::check_cast` is the gate; this
/// is the reference, so it refuses rather than guess).
pub(crate) fn cast_bits(x: Num, from: ElementKind, spec: CastSpec) -> u128 {
    let to = spec.to;
    assert!(
        crate::types::check_cast(from, spec).is_ok(),
        "cast_ref: {from:?} -> {spec:?} is not a defined conversion"
    );
    match (is_cast_float_endpoint(from), is_cast_float_endpoint(to)) {
        (true, true) => {
            let Num::F(f) = x else {
                panic!("cast_ref: float source with integer value")
            };
            match to {
                ElementKind::F64 => u128::from(f.to_bits()),
                t if is_half(t) => {
                    let (e, m) = float_format(t).expect("a half has a format");
                    encode_ieee(f, e, m, spec.round, spec.sat == Saturation::ClampFinite)
                }
                t if is_mx_element(t) => {
                    let (e, m) = float_format(t).expect("an MX element has a format");
                    encode_mx(f, e, m, spec.round)
                }
                _ => {
                    let max = f64::from(f32::MAX);
                    let r = if spec.sat == Saturation::ClampFinite && !f.is_nan() {
                        // Clamp BEFORE rounding so a value that would round up to
                        // infinity (or is infinite) lands on the largest finite f32.
                        f64_to_f32(f.clamp(-max, max), spec.round)
                    } else {
                        f64_to_f32(f, spec.round)
                    };
                    u128::from(r.to_bits())
                }
            }
        }
        (true, false) => {
            let Num::F(f) = x else {
                panic!("cast_ref: float source with integer value")
            };
            let (lo, hi) = int_range(to).expect("table checked");
            let y = round_to_integral(f, spec.round);
            let i = if y.is_nan() {
                0
            } else {
                // `as i128` saturates at the i128 range, far outside every
                // destination, so the clamp below does the real work.
                (y as i128).clamp(lo, hi)
            };
            bits_of(Num::I(i), to)
        }
        (false, true) => {
            let Num::I(i) = x else {
                panic!("cast_ref: integer source with float value")
            };
            match to {
                ElementKind::F64 => u128::from(int_to_f64(i, spec.round).to_bits()),
                t if is_half(t) => {
                    let (e, m) = float_format(t).expect("a half has a format");
                    encode_ieee(int_to_f64_sticky(i), e, m, spec.round, false)
                }
                t if is_mx_element(t) => {
                    let (e, m) = float_format(t).expect("an MX element has a format");
                    encode_mx(int_to_f64_sticky(i), e, m, spec.round)
                }
                _ => u128::from(int_to_f32(i, spec.round).to_bits()),
            }
        }
        (false, false) => {
            let Num::I(i) = x else {
                panic!("cast_ref: integer source with float value")
            };
            let (lo, hi) = int_range(to).expect("table checked");
            let v = match spec.sat {
                Saturation::Clamp => i.clamp(lo, hi),
                _ => i,
            };
            bits_of(Num::I(v), to)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(to: ElementKind, round: RoundMode, sat: Saturation) -> CastSpec {
        CastSpec { to, round, sat }
    }

    const MODES: [RoundMode; 4] = [
        RoundMode::NearestEven,
        RoundMode::TowardZero,
        RoundMode::TowardNegInf,
        RoundMode::TowardPosInf,
    ];

    /// Probe values: every f32 boundary region the 16-bit formats care about, both signs.
    fn probes() -> Vec<f64> {
        let mut v = vec![
            0.0,
            1.0,
            1.5,
            65504.0,
            65519.9,
            65520.0,
            65536.0,
            1.0e-8,
            5.960_464_477_539_063e-8,
            2.980_232_238_769_531_3e-8,
            3.4e38,
            3.39e38,
            1.0e300,
            1.0 + 2f64.powi(-11),
            1.0 + 2f64.powi(-8),
            1.0 + 2f64.powi(-24),
            f64::INFINITY,
            f64::MAX,
            f64::MIN_POSITIVE,
            5e-324,
        ];
        let mut s = 0x9e37_79b9_7f4a_7c15u64;
        for _ in 0..4000 {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            // spread across the exponent range of every format
            let e = (s % 160) as i32 - 100;
            v.push((1.0 + (s >> 12) as f64 / (1u64 << 52) as f64) * 2f64.powi(e));
        }
        v.extend(v.clone().into_iter().map(|x| -x));
        v
    }

    const MX: [(ElementKind, u32, u32, f64); 3] = [
        (ElementKind::F4E2M1, 2, 1, 6.0),
        (ElementKind::F6E2M3, 2, 3, 7.5),
        (ElementKind::F6E3M2, 3, 2, 28.0),
    ];

    /// Every non-negative code of an MX format with its value, ascending (the grid).
    fn mx_grid(e: u32, m: u32) -> Vec<(u128, f64)> {
        (0..(1u128 << (e + m)))
            .map(|c| (c, decode_mx(c, e, m)))
            .collect()
    }

    /// The nearer of two adjacent grid points (`(code, value)`), a tie going to the even code.
    fn nearest_tie_even(lo: (u128, f64), hi: (u128, f64), a: f64) -> f64 {
        let (dl, dh) = (a - lo.1, hi.1 - a);
        if dl < dh || (dl == dh && lo.0 % 2 == 0) {
            lo.1
        } else {
            hi.1
        }
    }

    /// The independent reference: pick from the explicit grid, by definition.
    fn mx_reference(x: f64, e: u32, m: u32, mode: RoundMode) -> f64 {
        if x.is_nan() {
            return 0.0;
        }
        let grid = mx_grid(e, m);
        let max = grid.last().expect("non-empty").1;
        let a = x.abs().min(max);
        let neg = x < 0.0;
        let lo = grid
            .iter()
            .rfind(|(_, v)| *v <= a)
            .expect("0 is on the grid");
        let hi = grid
            .iter()
            .find(|(_, v)| *v >= a)
            .expect("clamped into the grid");
        let mag = if lo.1 == hi.1 {
            lo.1
        } else {
            match mode {
                RoundMode::TowardZero => lo.1,
                // away from zero is the upper neighbour for a positive value, the lower for a
                // negative one's magnitude, and the reverse for TowardNegInf
                RoundMode::TowardPosInf => [hi.1, lo.1][usize::from(neg)],
                RoundMode::TowardNegInf => [lo.1, hi.1][usize::from(neg)],
                RoundMode::NearestEven => nearest_tie_even(*lo, *hi, a),
            }
        };
        if neg { -mag } else { mag }
    }

    #[test]
    fn mx_grids_match_the_ocp_tables() {
        // OCP MX v1.0 element value sets (spot values read from the spec's tables).
        let e2m1: Vec<f64> = mx_grid(2, 1).iter().take(8).map(|g| g.1).collect();
        assert_eq!(e2m1, [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]);
        for (_, e, m, max) in MX {
            assert_eq!(mx_grid(e, m).last().expect("non-empty").1, max, "e{e}m{m}");
        }
        assert_eq!(
            decode_mx(0b111_111, 3, 2),
            -28.0,
            "e3m2 has no NaN or Inf: all-ones is -28"
        );
        assert_eq!(decode_mx(0b011_111, 2, 3), 7.5);
    }

    #[test]
    fn mx_encoder_equals_the_brute_force_grid_search_for_every_mode() {
        let extra = [0.25, 0.75, 2.5, 5.0, 0.25 + 1e-12, 6.0 + 1e-9, 7.75, 29.0];
        for (_, e, m, _) in MX {
            for x in probes().into_iter().chain(extra) {
                for mode in MODES {
                    let got = decode_mx(encode_mx(x, e, m, mode), e, m);
                    let want = mx_reference(x, e, m, mode);
                    assert_eq!(got, want, "e{e}m{m} {x:e} under {mode:?}");
                }
            }
        }
    }

    #[test]
    fn mx_saturates_nan_goes_to_plus_zero_and_zero_keeps_its_sign() {
        for (_, e, m, max) in MX {
            let enc = |x: f64| encode_mx(x, e, m, RoundMode::NearestEven);
            let sign = 1u128 << (e + m);
            let max_code = sign - 1;
            assert_eq!(
                enc(f64::NAN),
                0,
                "NaN -> +0 (the IR's rule; hardware unverified)"
            );
            assert_eq!(enc(-f64::NAN), 0, "even a negative NaN");
            assert_eq!(enc(f64::INFINITY), max_code);
            assert_eq!(enc(f64::NEG_INFINITY), sign | max_code);
            // the clamp edge: max, just past max
            assert_eq!(decode_mx(enc(max), e, m), max);
            assert_eq!(enc(max + 1e-9), max_code, "max + eps");
            assert_eq!(enc(-max - 1e-9), sign | max_code, "-max - eps");
            assert_eq!(enc(1.0e300), max_code);
            for mode in MODES {
                assert_eq!(
                    encode_mx(1.0e300, e, m, mode),
                    max_code,
                    "no mode escapes to a non-finite code"
                );
                assert_eq!(encode_mx(-1.0e300, e, m, mode), sign | max_code);
            }
            assert_eq!(enc(0.0), 0);
            assert_eq!(enc(-0.0), sign, "-0 keeps its sign bit");
            assert_eq!(
                enc(-1e-30),
                sign,
                "a negative value that rounds to zero is -0"
            );
        }
    }

    #[test]
    fn integers_into_an_mx_element_round_once_and_saturate() {
        let to = |i: i128| {
            decode_mx(
                encode_mx(int_to_f64_sticky(i), 2, 1, RoundMode::NearestEven),
                2,
                1,
            )
        };
        assert_eq!(to(5), 4.0, "5 is a tie between 4 and 6: even code (4) wins");
        assert_eq!(to(7), 6.0);
        assert_eq!(to(-1_000_000), -6.0);
        assert_eq!(to(i128::from(i64::MAX)), 6.0);
    }

    #[test]
    fn every_16_bit_pattern_survives_decode_then_encode() {
        for (e, m) in [(5, 10), (8, 7)] {
            for bits in 0u128..=0xffff {
                let x = decode_ieee(bits, e, m);
                let back = encode_ieee(x, e, m, RoundMode::NearestEven, false);
                if x.is_nan() {
                    assert!(decode_ieee(back, e, m).is_nan(), "{bits:#x}");
                } else {
                    assert_eq!(back, bits, "e{e}m{m} pattern {bits:#x} -> {x}");
                }
            }
        }
    }

    #[test]
    fn the_generic_encoder_agrees_with_the_hardware_f32_conversion() {
        // binary32 is the one format with a hardware reference: all four modes against
        // the independent `f64_to_f32` (nearest-even is `as f32`).
        for x in probes() {
            for mode in MODES {
                let want = f64_to_f32(x, mode).to_bits();
                let got = encode_ieee(x, 8, 23, mode, false) as u32;
                if x.is_nan() {
                    continue;
                }
                assert_eq!(got, want, "{x:e} under {mode:?}");
            }
        }
    }

    #[test]
    fn directed_rounding_brackets_the_value_and_nearest_is_the_closer_neighbour() {
        for (e, m) in [(5, 10), (8, 7)] {
            for x in probes() {
                let d = |mode| decode_ieee(encode_ieee(x, e, m, mode, false), e, m);
                let (dn, up, tz, ne) = (
                    d(RoundMode::TowardNegInf),
                    d(RoundMode::TowardPosInf),
                    d(RoundMode::TowardZero),
                    d(RoundMode::NearestEven),
                );
                assert!(dn <= x && x <= up, "e{e}m{m} {x:e}: [{dn}, {up}]");
                assert_eq!(tz, if x < 0.0 { up } else { dn }, "toward zero {x:e}");
                if dn != up {
                    // inexact: the neighbours are adjacent and nearest picks the closer one
                    assert!(ne == dn || ne == up);
                    if dn.is_finite() && up.is_finite() {
                        assert!((x - ne).abs() <= (x - dn).abs().min((x - up).abs()) + 0.0);
                    }
                } else {
                    assert_eq!(ne, dn);
                }
            }
        }
    }

    #[test]
    fn half_overflow_follows_the_direction_and_clamp_finite_never_reaches_infinity() {
        let f16 = |x: f64, mode, clamp| decode_ieee(encode_ieee(x, 5, 10, mode, clamp), 5, 10);
        // 65520 is the midpoint between f16's max finite (65504) and 2^16: ties-to-even goes up.
        assert_eq!(f16(65519.0, RoundMode::NearestEven, false), 65504.0);
        assert_eq!(f16(65520.0, RoundMode::NearestEven, false), f64::INFINITY);
        assert_eq!(f16(1.0e9, RoundMode::TowardZero, false), 65504.0);
        assert_eq!(f16(1.0e9, RoundMode::TowardPosInf, false), f64::INFINITY);
        assert_eq!(f16(-1.0e9, RoundMode::TowardPosInf, false), -65504.0);
        assert_eq!(
            f16(-1.0e9, RoundMode::TowardNegInf, false),
            f64::NEG_INFINITY
        );
        for mode in MODES {
            assert_eq!(f16(f64::INFINITY, mode, true), 65504.0, "{mode:?}");
            assert_eq!(f16(-1.0e30, mode, true), -65504.0, "{mode:?}");
        }
        assert!(
            f16(f64::NAN, RoundMode::NearestEven, true).is_nan(),
            "NaN stays NaN"
        );
    }

    #[test]
    fn subnormal_and_zero_edges_of_the_16_bit_formats() {
        let f16 = |x: f64, mode| encode_ieee(x, 5, 10, mode, false);
        let tiny = 2f64.powi(-24); // f16's smallest subnormal
        assert_eq!(f16(tiny, RoundMode::NearestEven), 1);
        assert_eq!(
            f16(tiny / 2.0, RoundMode::NearestEven),
            0,
            "tie to even goes to zero"
        );
        assert_eq!(f16(tiny * 0.75, RoundMode::NearestEven), 1);
        assert_eq!(
            f16(tiny / 4.0, RoundMode::TowardPosInf),
            1,
            "directed up from below the grid"
        );
        assert_eq!(f16(-tiny / 4.0, RoundMode::TowardNegInf), 0x8001);
        assert_eq!(
            f16(-0.0, RoundMode::NearestEven),
            0x8000,
            "the sign of zero survives"
        );
        // the largest subnormal rounds up into the smallest normal
        assert_eq!(
            f16(2f64.powi(-14) - tiny / 4.0, RoundMode::NearestEven),
            0x0400
        );
    }

    #[test]
    fn integers_round_once_into_a_16_bit_float_even_above_2_pow_53() {
        let to_bf16 = |i: i128, mode| encode_ieee(int_to_f64_sticky(i), 8, 7, mode, false);
        // 2^62 + 1 sits just above a bf16 value; a naive `as f64` would round it away.
        let i = (1i128 << 62) + 1;
        let below = encode_ieee(2f64.powi(62), 8, 7, RoundMode::NearestEven, false);
        assert_eq!(to_bf16(i, RoundMode::TowardZero), below);
        assert_eq!(to_bf16(i, RoundMode::NearestEven), below);
        assert_eq!(
            to_bf16(i, RoundMode::TowardPosInf),
            below + 1,
            "must see the dropped bit"
        );
        assert_eq!(to_bf16(-i, RoundMode::TowardNegInf), (below + 1) | 0x8000);
    }

    #[test]
    fn step_moves_one_ulp_and_crosses_zero() {
        assert_eq!(step32(1.0, true).to_bits(), 1.0f32.to_bits() + 1);
        assert_eq!(step32(1.0, false).to_bits(), 1.0f32.to_bits() - 1);
        assert_eq!(step32(-1.0, true).to_bits(), (-1.0f32).to_bits() - 1);
        assert_eq!(step32(0.0, true), f32::from_bits(1));
        assert_eq!(step32(0.0, false), -f32::from_bits(1));
        assert_eq!(step32(f32::INFINITY, false), f32::MAX);
        assert_eq!(step32(f32::INFINITY, true), f32::INFINITY);
        assert_eq!(step64(f64::INFINITY, false), f64::MAX);
    }

    #[test]
    fn directed_rounding_of_a_value_between_two_f32s() {
        // 1 + 2^-24 is exactly half way between 1 and 1 + 2^-23 (ties to even = 1).
        let x = 1.0 + 2.0f64.powi(-24);
        let lo = 1.0f32;
        let hi = f32::from_bits(1.0f32.to_bits() + 1);
        assert_eq!(f64_to_f32(x, RoundMode::NearestEven), lo);
        assert_eq!(f64_to_f32(x, RoundMode::TowardZero), lo);
        assert_eq!(f64_to_f32(x, RoundMode::TowardNegInf), lo);
        assert_eq!(f64_to_f32(x, RoundMode::TowardPosInf), hi);
        // Mirror image: the negative side flips which mode goes where.
        assert_eq!(f64_to_f32(-x, RoundMode::TowardZero), -lo);
        assert_eq!(f64_to_f32(-x, RoundMode::TowardNegInf), -hi);
        assert_eq!(f64_to_f32(-x, RoundMode::TowardPosInf), -lo);
    }

    #[test]
    fn overflow_follows_the_rounding_direction() {
        let big = 1.0e300;
        assert_eq!(f64_to_f32(big, RoundMode::NearestEven), f32::INFINITY);
        assert_eq!(f64_to_f32(big, RoundMode::TowardZero), f32::MAX);
        assert_eq!(f64_to_f32(big, RoundMode::TowardNegInf), f32::MAX);
        assert_eq!(f64_to_f32(big, RoundMode::TowardPosInf), f32::INFINITY);
        assert_eq!(f64_to_f32(-big, RoundMode::TowardPosInf), -f32::MAX);
        assert_eq!(f64_to_f32(-big, RoundMode::TowardNegInf), f32::NEG_INFINITY);
    }

    #[test]
    fn clamp_finite_catches_overflow_and_infinity_but_not_nan() {
        let s = spec(
            ElementKind::F32,
            RoundMode::NearestEven,
            Saturation::ClampFinite,
        );
        let f = |x: f64| f32::from_bits(cast_bits(Num::F(x), ElementKind::F64, s) as u32);
        assert_eq!(f(1.0e300), f32::MAX);
        assert_eq!(f(f64::INFINITY), f32::MAX);
        assert_eq!(f(f64::NEG_INFINITY), -f32::MAX);
        assert_eq!(f(0.5), 0.5);
        assert!(f(f64::NAN).is_nan());
    }

    #[test]
    fn float_to_int_saturates_and_maps_nan_to_zero() {
        let s = spec(ElementKind::I8, RoundMode::TowardZero, Saturation::Clamp);
        let f = |x: f64| num_of(cast_bits(Num::F(x), ElementKind::F32, s), ElementKind::I8);
        let i = |x: f64| match f(x) {
            Num::I(v) => v,
            Num::F(_) => unreachable!(),
        };
        assert_eq!(i(3.99), 3);
        assert_eq!(i(-3.99), -3);
        assert_eq!(i(1000.0), 127);
        assert_eq!(i(-1000.0), -128);
        assert_eq!(i(f64::INFINITY), 127);
        assert_eq!(i(f64::NAN), 0);
    }

    #[test]
    fn integer_narrowing_wraps_or_clamps() {
        let w = spec(ElementKind::I8, RoundMode::NearestEven, Saturation::Wrap);
        let c = spec(ElementKind::I8, RoundMode::NearestEven, Saturation::Clamp);
        let to8 = |s, v: i128| num_of(cast_bits(Num::I(v), ElementKind::I32, s), ElementKind::I8);
        let n = |x| match x {
            Num::I(v) => v,
            Num::F(_) => unreachable!(),
        };
        assert_eq!(n(to8(w, 300)), 44);
        assert_eq!(n(to8(c, 300)), 127);
        assert_eq!(n(to8(w, -129)), 127);
        assert_eq!(n(to8(c, -129)), -128);
    }

    #[test]
    fn int_to_float_directed_when_inexact() {
        // 2^24 + 1 is not an f32.
        let i = (1i128 << 24) + 1;
        let up = f32::from_bits(((1u32 << 24) as f32).to_bits() + 1);
        assert_eq!(int_to_f32(i, RoundMode::NearestEven), (1u32 << 24) as f32);
        assert_eq!(int_to_f32(i, RoundMode::TowardZero), (1u32 << 24) as f32);
        assert_eq!(int_to_f32(i, RoundMode::TowardPosInf), up);
        assert_eq!(int_to_f32(-i, RoundMode::TowardNegInf), -up);
    }
}
