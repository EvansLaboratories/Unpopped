// SPDX-License-Identifier: MIT OR Apache-2.0
//! Tests for `soft_f128` (kept in their own file so the module stays under Codacy's file-length limit).

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
fn a_nonzero_division_remainder_breaks_a_rounding_tie_upward() {
    // random significands a < b: find operands whose 115-bit quotient ends in the pair 10 (a tie at
    // 113 bits) with a nonzero remainder. The true value is above the tie, so it must round UP; dropping the sticky bit
    // would call it a tie and round to even.
    let mut rng = Rng(77);
    let mut found = 0;
    for _ in 0..4000 {
        let frac = |r: &mut Rng| (u128::from(r.next()) << 64 | u128::from(r.next())) & (HIDDEN - 1);
        let (fa, fb) = (frac(&mut rng), frac(&mut rng));
        if fa >= fb {
            continue; // keep the quotient below one: 115 bits
        }
        let num = W256::from_u128(HIDDEN + fa).shift_left(115);
        let (q, r) = num.divrem_unsigned(W256::from_u128(HIDDEN + fb)).unwrap();
        if q.low_u128() & 3 == 2 && !r.is_zero() {
            let (x, y) = (
                F128((0x3fff << FRAC_BITS) | fa),
                F128((0x3fff << FRAC_BITS) | fb),
            );
            let want = (0x3ffdu128 << FRAC_BITS) + (q.shr_logical(2).low_u128() + 1);
            assert_eq!(x.div_rne(y).0, want, "{fa:#x} / {fb:#x}");
            found += 1;
        }
    }
    assert!(
        found >= 100,
        "the search must find tie cases, found {found}"
    );
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
