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
//! Only `{f32, f32s, f64, i8, i16, i32, i64, u8, u16, u32}` are defined (`types::check_cast`).

use crate::ir::{CastSpec, RoundMode, Saturation};
use crate::types::{int_range, is_cast_float, typed_bits};
use unpopped_vocab::ElementKind;

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
    match (is_cast_float(from), is_cast_float(to)) {
        (true, true) => {
            let Num::F(f) = x else {
                panic!("cast_ref: float source with integer value")
            };
            match to {
                ElementKind::F64 => u128::from(f.to_bits()),
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
