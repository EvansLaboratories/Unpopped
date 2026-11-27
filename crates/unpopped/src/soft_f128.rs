// SPDX-License-Identifier: MIT OR Apache-2.0
//! `F128`: a software IEEE 754 binary128 for the oracle (ROADMAP E3).
//!
//! The design is the **integer-word** one: a value is its 128-bit pattern, and every operation is exact integer arithmetic on a
//! 256-bit word ([`W256`]) followed by ONE rounding (round to nearest, ties to even) in [`pack`]. There is no double-double
//! pair and no hardware floating point in the arithmetic, so the result cannot depend on the host.
//!
//! Rules (IEEE 754-2019 section 6): `x + (-x)` is `+0`; `inf - inf`, `0 * inf`, `0 / 0` and `inf / inf` are NaN; a finite
//! overflow rounds to infinity; a result below the smallest subnormal rounds to zero or to it by the nearest rule. **A NaN
//! result is the canonical quiet NaN** (`0x7fff8000...0`): a NaN operand's payload and sign are not preserved (the kernel
//! oracle does not specify a payload either, `docs/design-float-collective-nan.md`).
//!
//! Only the operations a kernel needs first are here (add, subtract, multiply, divide, negate, absolute value, compare, and
//! the conversions to and from `f64` and the integers). Fused multiply-add and square root are later rows.

use crate::wide_int::W256;
use core::cmp::Ordering;

const SIGN: u128 = 1 << 127;
const FRAC_BITS: u32 = 112;
const HIDDEN: u128 = 1 << FRAC_BITS;
const EXP_FIELD_MAX: u128 = 0x7fff;
const BIAS: i32 = 16383;
/// The canonical quiet NaN.
const QNAN: u128 = (EXP_FIELD_MAX << FRAC_BITS) | (1 << (FRAC_BITS - 1));
const INF: u128 = EXP_FIELD_MAX << FRAC_BITS;

/// A binary128 value, as its bit pattern.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct F128(pub u128);

/// A classified operand: a finite nonzero value is `sig * 2^(exp - 112)` with `sig` normalised to bit 112 set (a subnormal
/// is shifted up and its exponent lowered).
#[derive(Copy, Clone)]
enum Class {
    Nan,
    Inf(bool),
    Zero(bool),
    Finite { neg: bool, exp: i32, sig: u128 },
}

fn classify(bits: u128) -> Class {
    let neg = bits & SIGN != 0;
    let field = (bits >> FRAC_BITS) & EXP_FIELD_MAX;
    let frac = bits & (HIDDEN - 1);
    if field == EXP_FIELD_MAX {
        return if frac == 0 {
            Class::Inf(neg)
        } else {
            Class::Nan
        };
    }
    if field == 0 {
        if frac == 0 {
            return Class::Zero(neg);
        }
        // a subnormal: normalise so bit 112 is set
        let shift = frac.leading_zeros() - (127 - FRAC_BITS);
        return Class::Finite {
            neg,
            exp: 1 - BIAS - shift as i32,
            sig: frac << shift,
        };
    }
    Class::Finite {
        neg,
        exp: field as i32 - BIAS,
        sig: HIDDEN | frac,
    }
}

/// `m >> k`, rounded to nearest with ties to even (`k >= 1`).
fn shr_rne(m: W256, k: u32) -> W256 {
    if k == 0 {
        return m;
    }
    if k > 256 {
        return W256::ZERO; // m < 2^256 <= 2^(k-1): below half
    }
    let (q, rem, half) = if k == 256 {
        (W256::ZERO, m, W256::ONE.shift_left(255))
    } else {
        let q = m.shr_logical(k);
        (
            q,
            m.wrapping_sub(q.shift_left(k)),
            W256::ONE.shift_left(k - 1),
        )
    };
    match rem.cmp_unsigned(half) {
        Ordering::Greater => q.wrapping_add(W256::ONE),
        Ordering::Less => q,
        Ordering::Equal if q.low_u128() & 1 == 1 => q.wrapping_add(W256::ONE),
        Ordering::Equal => q,
    }
}

/// The IEEE format with `frac` fraction bits and `exp_bits` exponent bits holding `m * 2^e`, rounded once to nearest-even, as
/// its bit pattern (sign at bit `frac + exp_bits`). `m == 0` gives a signed zero.
///
/// A normal result is `((E + bias - 1) << frac) + m`: the leading bit of the 113-bit (or 53-bit) significand adds one to the
/// exponent field, so a rounding carry out of the significand increments the exponent with no special case. A subnormal result
/// is its significand alone (rounding up to the smallest normal falls out the same way).
fn pack(neg: bool, m: W256, e: i32, frac: u32, exp_bits: u32) -> u128 {
    let sign = if neg { 1u128 << (frac + exp_bits) } else { 0 };
    if m.is_zero() {
        return sign;
    }
    let bias = (1i32 << (exp_bits - 1)) - 1;
    let min_exp = 1 - bias; // exponent of the smallest normal
    let lsb_min = min_exp - frac as i32; // exponent of the smallest subnormal's lowest bit
    let len = m.bit_length() as i32;
    let lead = e + len - 1; // exponent of the leading bit
    let inf = ((1u128 << exp_bits) - 1) << frac;
    if lead >= min_exp {
        let drop = len - (frac as i32 + 1);
        let sig = if drop > 0 {
            shr_rne(m, drop as u32)
        } else {
            m.shift_left((-drop) as u32)
        };
        let bits = (((lead + bias - 1) as u128) << frac) + sig.low_u128();
        return sign | if bits >= inf { inf } else { bits };
    }
    let drop = lsb_min - e;
    let sig = if drop > 0 {
        shr_rne(m, drop as u32)
    } else {
        m.shift_left((-drop) as u32)
    };
    sign | sig.low_u128()
}

fn pack128(neg: bool, m: W256, e: i32) -> F128 {
    F128(pack(neg, m, e, FRAC_BITS, 15))
}

/// A finite operand: the value itself, its sign, its unbiased exponent and its 113-bit significand.
type Finite = (F128, bool, i32, u128);

/// The sum of two finite nonzero operands, rounded once to nearest even.
fn add_finite(a: Finite, b: Finite) -> F128 {
    // the operand with the larger exponent is `big`
    let ((big, nbg, ebg, sbg), (_, nsm, esm, ssm)) = if a.2 >= b.2 { (a, b) } else { (b, a) };
    let d = ebg - esm;
    // a small operand more than 130 bits below cannot move a 113-bit significand off its nearest value
    if d > 130 {
        return big;
    }
    let mb = W256::from_u128(sbg).shift_left(d as u32);
    let ms = W256::from_u128(ssm);
    let e = esm - FRAC_BITS as i32;
    if nbg == nsm {
        return pack128(nbg, mb.wrapping_add(ms), e);
    }
    match mb.cmp_unsigned(ms) {
        Ordering::Greater => pack128(nbg, mb.wrapping_sub(ms), e),
        Ordering::Less => pack128(nsm, ms.wrapping_sub(mb), e),
        Ordering::Equal => F128::ZERO,
    }
}

impl F128 {
    /// Positive zero.
    pub const ZERO: Self = Self(0);
    /// One.
    pub const ONE: Self = Self(0x3fff << FRAC_BITS);
    /// Positive infinity.
    pub const INFINITY: Self = Self(INF);
    /// The canonical quiet NaN.
    pub const NAN: Self = Self(QNAN);

    /// Whether this is a NaN.
    #[must_use]
    pub const fn is_nan(self) -> bool {
        self.0 & !SIGN > INF
    }

    /// `-self` (a sign flip, also of a NaN or a zero).
    #[must_use]
    pub const fn neg(self) -> Self {
        Self(self.0 ^ SIGN)
    }

    /// `|self|`.
    #[must_use]
    pub const fn abs(self) -> Self {
        Self(self.0 & !SIGN)
    }

    /// `self + o`, rounded to nearest even.
    #[must_use]
    pub fn add_rne(self, o: Self) -> Self {
        let (a, b) = (classify(self.0), classify(o.0));
        match (a, b) {
            (Class::Nan, _) | (_, Class::Nan) => Self::NAN,
            (Class::Inf(x), Class::Inf(y)) => {
                if x == y {
                    self
                } else {
                    Self::NAN
                }
            }
            (Class::Inf(_), _) => self,
            (_, Class::Inf(_)) => o,
            (Class::Zero(x), Class::Zero(y)) => {
                if x == y {
                    self
                } else {
                    Self::ZERO
                }
            }
            (Class::Zero(_), _) => o,
            (_, Class::Zero(_)) => self,
            (
                Class::Finite {
                    neg: na,
                    exp: ea,
                    sig: sa,
                },
                Class::Finite {
                    neg: nb,
                    exp: eb,
                    sig: sb,
                },
            ) => add_finite((self, na, ea, sa), (o, nb, eb, sb)),
        }
    }

    /// `self - o`.
    #[must_use]
    pub fn sub_rne(self, o: Self) -> Self {
        self.add_rne(o.neg())
    }

    /// `self * o`, rounded to nearest even.
    #[must_use]
    pub fn mul_rne(self, o: Self) -> Self {
        let (a, b) = (classify(self.0), classify(o.0));
        let sign = |x: bool, y: bool| x != y;
        match (a, b) {
            (Class::Nan, _) | (_, Class::Nan) => Self::NAN,
            (Class::Inf(_), Class::Zero(_)) | (Class::Zero(_), Class::Inf(_)) => Self::NAN,
            (Class::Inf(x), Class::Inf(y) | Class::Finite { neg: y, .. })
            | (Class::Finite { neg: x, .. }, Class::Inf(y)) => {
                Self(INF | if sign(x, y) { SIGN } else { 0 })
            }
            (Class::Zero(x), Class::Zero(y) | Class::Finite { neg: y, .. })
            | (Class::Finite { neg: x, .. }, Class::Zero(y)) => {
                Self(if sign(x, y) { SIGN } else { 0 })
            }
            (
                Class::Finite {
                    neg: na,
                    exp: ea,
                    sig: sa,
                },
                Class::Finite {
                    neg: nb,
                    exp: eb,
                    sig: sb,
                },
            ) => pack128(
                sign(na, nb),
                W256::from_u128(sa).wrapping_mul(W256::from_u128(sb)),
                (ea - FRAC_BITS as i32) + (eb - FRAC_BITS as i32),
            ),
        }
    }

    /// `self / o`, rounded to nearest even.
    #[must_use]
    pub fn div_rne(self, o: Self) -> Self {
        let (a, b) = (classify(self.0), classify(o.0));
        let sign = |x: bool, y: bool| x != y;
        let signed = |neg: bool, base: u128| Self(base | if neg { SIGN } else { 0 });
        match (a, b) {
            (Class::Nan, _) | (_, Class::Nan) => Self::NAN,
            (Class::Inf(_), Class::Inf(_)) | (Class::Zero(_), Class::Zero(_)) => Self::NAN,
            (Class::Inf(x), Class::Zero(y) | Class::Finite { neg: y, .. })
            | (Class::Finite { neg: x, .. }, Class::Zero(y)) => signed(sign(x, y), INF),
            (Class::Zero(x), Class::Inf(y) | Class::Finite { neg: y, .. })
            | (Class::Finite { neg: x, .. }, Class::Inf(y)) => signed(sign(x, y), 0),
            (
                Class::Finite {
                    neg: na,
                    exp: ea,
                    sig: sa,
                },
                Class::Finite {
                    neg: nb,
                    exp: eb,
                    sig: sb,
                },
            ) => {
                // 115 extra quotient bits: the quotient has 115 or 116 bits, so at least two lie below the 113 kept
                let num = W256::from_u128(sa).shift_left(115);
                let (q, r) = num
                    .divrem_unsigned(W256::from_u128(sb))
                    .expect("a finite nonzero divisor");
                // the remainder is folded into the lowest bit as a sticky bit, below the rounding position
                let m = q
                    .shift_left(1)
                    .or(if r.is_zero() { W256::ZERO } else { W256::ONE });
                pack128(sign(na, nb), m, ea - eb - 115 - 1)
            }
        }
    }

    /// Order two values: `None` when either is a NaN, and `-0 == +0`.
    #[must_use]
    pub fn partial_cmp(self, o: Self) -> Option<Ordering> {
        if self.is_nan() || o.is_nan() {
            return None;
        }
        let (ma, mb) = (self.0 & !SIGN, o.0 & !SIGN);
        if ma == 0 && mb == 0 {
            return Some(Ordering::Equal);
        }
        Some(match (self.0 & SIGN != 0, o.0 & SIGN != 0) {
            (false, true) => Ordering::Greater,
            (true, false) => Ordering::Less,
            (false, false) => ma.cmp(&mb),
            (true, true) => mb.cmp(&ma),
        })
    }

    /// The `f64` value `x` exactly (every finite `f64` is a binary128). A NaN becomes the canonical quiet NaN.
    #[must_use]
    pub fn from_f64(x: f64) -> Self {
        let bits = x.to_bits();
        let neg = bits >> 63 == 1;
        let field = ((bits >> 52) & 0x7ff) as i32;
        let frac = bits & ((1 << 52) - 1);
        if field == 0x7ff {
            return if frac == 0 {
                Self(INF | if neg { SIGN } else { 0 })
            } else {
                Self::NAN
            };
        }
        if field == 0 {
            return pack128(neg, W256::from_u128(u128::from(frac)), -1074);
        }
        pack128(
            neg,
            W256::from_u128(u128::from((1 << 52) | frac)),
            field - 1023 - 52,
        )
    }

    /// The nearest `f64` (ties to even; overflow gives infinity). A NaN gives the canonical `f64` quiet NaN.
    #[must_use]
    pub fn to_f64(self) -> f64 {
        match classify(self.0) {
            Class::Nan => f64::NAN,
            Class::Inf(neg) => {
                if neg {
                    f64::NEG_INFINITY
                } else {
                    f64::INFINITY
                }
            }
            Class::Zero(neg) => {
                if neg {
                    -0.0
                } else {
                    0.0
                }
            }
            Class::Finite { neg, exp, sig } => {
                f64::from_bits(
                    pack(neg, W256::from_u128(sig), exp - FRAC_BITS as i32, 52, 11) as u64,
                )
            }
        }
    }

    /// The integer `w`, read as signed or unsigned, rounded to nearest even (an unsigned `w` up to `2^256 - 1`).
    #[must_use]
    pub fn from_w256(w: W256, signed: bool) -> Self {
        if signed && w.is_negative() {
            pack128(true, w.wrapping_neg(), 0)
        } else {
            pack128(false, w, 0)
        }
    }
}

#[cfg(test)]
#[path = "soft_f128_tests.rs"]
mod tests;
