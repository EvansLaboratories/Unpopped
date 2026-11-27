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
            ) => {
                // the operand with the larger exponent is `big`
                let (big, small, nbg, nsm, ebg, esm, sbg, ssm) = if ea >= eb {
                    (self, o, na, nb, ea, eb, sa, sb)
                } else {
                    (o, self, nb, na, eb, ea, sb, sa)
                };
                let d = ebg - esm;
                // a small operand more than 130 bits below cannot move a 113-bit significand off its nearest value
                if d > 130 {
                    return big;
                }
                let _ = small;
                let mb = W256::from_u128(sbg).shift_left(d as u32);
                let ms = W256::from_u128(ssm);
                let e = esm - FRAC_BITS as i32;
                if nbg == nsm {
                    return pack128(nbg, mb.wrapping_add(ms), e);
                }
                match mb.cmp_unsigned(ms) {
                    Ordering::Greater => pack128(nbg, mb.wrapping_sub(ms), e),
                    Ordering::Less => pack128(nsm, ms.wrapping_sub(mb), e),
                    Ordering::Equal => Self::ZERO,
                }
            }
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
mod tests {
    use super::*;

    const TWO: F128 = F128(0x4000 << FRAC_BITS);
    const MAX: F128 = F128((0x7ffe << FRAC_BITS) | (HIDDEN - 1));
    const MIN_SUB: F128 = F128(1);

    fn int(v: i64) -> F128 {
        F128::from_w256(W256::from_i128(i128::from(v)), true)
    }

    /// A small deterministic generator (xorshift64*), so a failure names its inputs.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
        }
        /// A finite `f64` with exponent in `-40..=40` (no overflow or subnormal result from one operation).
        fn f64(&mut self) -> f64 {
            let (r, s) = (self.next(), self.next());
            let e = s % 81 + 1023 - 40;
            f64::from_bits((r & 1) << 63 | e << 52 | (r >> 12) & ((1 << 52) - 1))
        }
    }

    #[test]
    fn known_patterns() {
        assert_eq!(F128::ONE.add_rne(F128::ONE), TWO);
        assert_eq!(
            int(1).div_rne(int(3)).0,
            0x3ffd_5555_5555_5555_5555_5555_5555_5555
        );
        assert_eq!(
            int(1).div_rne(int(10)).0,
            0x3ffb_9999_9999_9999_9999_9999_9999_999a
        );
        assert_eq!(int(-2).0, 0xc000 << FRAC_BITS);
        // pi to binary128, and its nearest f64
        let pi = F128(0x4000_921f_b544_42d1_8469_898c_c517_01b8);
        assert_eq!(pi.to_f64().to_bits(), std::f64::consts::PI.to_bits());
        assert_eq!(
            F128::from_f64(std::f64::consts::PI).to_f64(),
            std::f64::consts::PI
        );
    }

    #[test]
    fn the_four_operations_match_f64_on_random_f64_operands() {
        // an f64 sum, product or quotient of operands with exponents in -40..=40 is a value the binary128 arithmetic
        // rounds once to 113 bits and the conversion rounds once more to 53: the two agree unless the 113-bit result is
        // exactly a 53-bit tie, which these seeds do not hit (and which would be a double rounding, not a defect)
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        for _ in 0..20_000 {
            let (a, b) = (rng.f64(), rng.f64());
            let (x, y) = (F128::from_f64(a), F128::from_f64(b));
            for (name, got, want) in [
                ("add", x.add_rne(y).to_f64(), a + b),
                ("sub", x.sub_rne(y).to_f64(), a - b),
                ("mul", x.mul_rne(y).to_f64(), a * b),
                ("div", x.div_rne(y).to_f64(), a / b),
            ] {
                assert_eq!(got.to_bits(), want.to_bits(), "{name} {a:e} {b:e}");
            }
        }
    }

    #[test]
    fn f64_conversion_round_trips_every_pattern_class() {
        let mut rng = Rng(12345);
        let check = |x: f64| {
            let r = F128::from_f64(x).to_f64();
            assert_eq!(r.to_bits(), x.to_bits(), "{:#x}", x.to_bits());
        };
        for x in [
            0.0,
            -0.0,
            1.0,
            -1.5,
            f64::MAX,
            f64::MIN_POSITIVE,
            f64::from_bits(1),
            f64::INFINITY,
            f64::NEG_INFINITY,
        ] {
            check(x);
        }
        for _ in 0..50_000 {
            let x = f64::from_bits(rng.next());
            if !x.is_nan() {
                check(x);
            }
        }
        assert!(F128::from_f64(f64::NAN).is_nan());
        assert!(F128::NAN.to_f64().is_nan());
    }

    #[test]
    fn rounding_at_the_subnormal_and_overflow_edges() {
        let half = F128(0x3ffe << FRAC_BITS);
        let three_quarters = F128((0x3ffe << FRAC_BITS) | (1 << (FRAC_BITS - 1)));
        // 1 * 0.5 ulp is a tie: to even, which is 0; 3 * 0.5 = 1.5 ties to 2; 5 * 0.5 = 2.5 ties to 2; 1 * 0.75 rounds to 1
        assert_eq!(MIN_SUB.mul_rne(half), F128::ZERO);
        assert_eq!(F128(3).mul_rne(half), F128(2));
        assert_eq!(F128(5).mul_rne(half), F128(2));
        assert_eq!(MIN_SUB.mul_rne(three_quarters), MIN_SUB);
        // the smallest normal is reached by a rounding carry out of the subnormals
        let min_normal = F128(HIDDEN);
        assert_eq!(F128(HIDDEN - 1).add_rne(MIN_SUB), min_normal);
        // overflow: the largest finite doubled is infinity, and MAX + MAX too; MAX itself is finite
        assert_eq!(MAX.mul_rne(TWO), F128::INFINITY);
        assert_eq!(MAX.add_rne(MAX), F128::INFINITY);
        assert_eq!(MAX.add_rne(F128::ZERO), MAX);
        // underflow: the smallest subnormal divided by two ties to zero; by 3 rounds to zero
        assert_eq!(MIN_SUB.div_rne(TWO), F128::ZERO);
        assert_eq!(MIN_SUB.div_rne(int(3)), F128::ZERO);
        assert_eq!(F128(3).div_rne(TWO), F128(2));
        // a subnormal operand is read exactly: min_sub + min_sub is two
        assert_eq!(MIN_SUB.add_rne(MIN_SUB), F128(2));
        assert_eq!(F128(HIDDEN | 5).sub_rne(F128(HIDDEN)), F128(5));
    }

    #[test]
    fn specials_and_signed_zero() {
        let inf = F128::INFINITY;
        let one = F128::ONE;
        assert!(inf.sub_rne(inf).is_nan());
        assert!(F128::ZERO.mul_rne(inf).is_nan());
        assert!(F128::ZERO.div_rne(F128::ZERO).is_nan());
        assert!(inf.div_rne(inf).is_nan());
        assert!(F128::NAN.add_rne(one).is_nan());
        assert_eq!(one.div_rne(F128::ZERO), inf);
        assert_eq!(one.neg().div_rne(F128::ZERO), inf.neg());
        assert_eq!(one.div_rne(inf), F128::ZERO);
        assert_eq!(one.neg().div_rne(inf), F128::ZERO.neg());
        assert_eq!(inf.add_rne(one), inf);
        assert_eq!(one.sub_rne(one), F128::ZERO, "x - x is +0");
        assert_eq!(one.neg().add_rne(one), F128::ZERO, "-x + x is +0");
        assert_eq!(
            F128::ZERO.neg().add_rne(F128::ZERO.neg()),
            F128::ZERO.neg(),
            "-0 + -0 is -0"
        );
        assert_eq!(F128::ZERO.neg().add_rne(F128::ZERO), F128::ZERO);
        assert_eq!(one.neg().mul_rne(F128::ZERO), F128::ZERO.neg());
    }

    #[test]
    fn compare_is_a_partial_order_with_equal_zeros() {
        let (one, two) = (F128::ONE, TWO);
        assert_eq!(one.partial_cmp(two), Some(Ordering::Less));
        assert_eq!(two.neg().partial_cmp(one.neg()), Some(Ordering::Less));
        assert_eq!(two.neg().partial_cmp(one), Some(Ordering::Less));
        assert_eq!(
            F128::ZERO.neg().partial_cmp(F128::ZERO),
            Some(Ordering::Equal)
        );
        assert_eq!(F128::NAN.partial_cmp(one), None);
        assert_eq!(F128::INFINITY.partial_cmp(MAX), Some(Ordering::Greater));
    }

    #[test]
    fn integers_convert_with_one_rounding() {
        assert_eq!(int(0), F128::ZERO);
        assert_eq!(int(1), F128::ONE);
        // 2^113 + 1 is not representable (113 bits kept): ties to even, down to 2^113
        let big = W256::ONE.shift_left(113).wrapping_add(W256::ONE);
        assert_eq!(
            F128::from_w256(big, false),
            F128::from_w256(W256::ONE.shift_left(113), false)
        );
        // 2^113 + 3 ties up to 2^113 + 4
        let up = W256::ONE.shift_left(113).wrapping_add(W256::from_u128(3));
        assert_eq!(
            F128::from_w256(up, false),
            F128::from_w256(
                W256::ONE.shift_left(113).wrapping_add(W256::from_u128(4)),
                false
            )
        );
        // the all-ones 256-bit word: unsigned is about 2^256, signed is -1
        assert_eq!(F128::from_w256(W256::ONES, true), int(-1));
        assert_eq!(
            F128::from_w256(W256::ONES, false),
            F128(0x40ff << FRAC_BITS)
        );
    }
}
