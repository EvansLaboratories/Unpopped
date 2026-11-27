// SPDX-License-Identifier: MIT OR Apache-2.0
//! `W256`: the oracle's wide integer (ROADMAP E2), a 256-bit two's-complement word.
//!
//! One bit pattern serves all four wide dtypes (`I128`, `U128`, `I256`, `U256`); the DTYPE decides how a
//! pattern is read (signed or unsigned) and where it wraps. Every operation here is the existing integer
//! contract (`ir.rs` on `BinaryOp`: exact WRAPPING semantics, never infinite precision), so none panics except
//! on a caller error that the operation names.
//!
//! Limbs are little-endian `u64`s, so `to_le_bytes` is the storage layout of an `I256`/`U256` buffer element.

use core::cmp::Ordering;
use unpopped_vocab::ElementKind;

/// A 256-bit two's-complement word. See the module docs.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct W256([u64; 4]);

/// The width in bits of a wide integer dtype (`128` or `256`), or `None` for any other dtype.
#[must_use]
pub const fn wide_int_bits(t: ElementKind) -> Option<u32> {
    match t {
        ElementKind::I128 | ElementKind::U128 => Some(128),
        ElementKind::I256 | ElementKind::U256 => Some(256),
        _ => None,
    }
}

/// Whether a wide integer dtype is signed, or `None` for any other dtype.
#[must_use]
pub const fn wide_int_signed(t: ElementKind) -> Option<bool> {
    match t {
        ElementKind::I128 | ElementKind::I256 => Some(true),
        ElementKind::U128 | ElementKind::U256 => Some(false),
        _ => None,
    }
}

/// `(min, max)` of a wide integer dtype, each as the dtype's own bit pattern (a signed minimum is the sign-extended
/// pattern; order them with [`W256::cmp_as`]). `None` for any other dtype.
#[must_use]
pub fn wide_int_range(t: ElementKind) -> Option<(W256, W256)> {
    let bits = wide_int_bits(t)?;
    let signed = wide_int_signed(t)?;
    Some(if signed {
        let max = W256::ONES.shr_logical(256 - bits + 1);
        (max.not(), max)
    } else {
        (W256::ZERO, W256::ONES.shr_logical(256 - bits))
    })
}

impl W256 {
    /// Zero.
    pub const ZERO: Self = Self([0; 4]);
    /// One.
    pub const ONE: Self = Self([1, 0, 0, 0]);
    /// Every bit set (`U256::MAX`, `-1` as a signed word).
    pub const ONES: Self = Self([u64::MAX; 4]);

    /// The unsigned value `v`.
    #[must_use]
    pub const fn from_u128(v: u128) -> Self {
        Self([v as u64, (v >> 64) as u64, 0, 0])
    }

    /// The signed value `v`, sign-extended.
    #[must_use]
    pub const fn from_i128(v: i128) -> Self {
        let ext = if v < 0 { u64::MAX } else { 0 };
        Self([v as u64, ((v as u128) >> 64) as u64, ext, ext])
    }

    /// The low 128 bits, as an unsigned value.
    #[must_use]
    pub const fn low_u128(self) -> u128 {
        (self.0[0] as u128) | ((self.0[1] as u128) << 64)
    }

    /// Read the 32 little-endian storage bytes of an `I256`/`U256` element.
    #[must_use]
    pub fn from_le_bytes(b: [u8; 32]) -> Self {
        let mut l = [0u64; 4];
        for (i, limb) in l.iter_mut().enumerate() {
            let mut w = [0u8; 8];
            w.copy_from_slice(&b[i * 8..i * 8 + 8]);
            *limb = u64::from_le_bytes(w);
        }
        Self(l)
    }

    /// The 32 little-endian storage bytes.
    #[must_use]
    pub fn to_le_bytes(self) -> [u8; 32] {
        let mut out = [0u8; 32];
        for (i, limb) in self.0.iter().enumerate() {
            out[i * 8..i * 8 + 8].copy_from_slice(&limb.to_le_bytes());
        }
        out
    }

    /// Whether bit 255 is set (the sign of a 256-bit signed word).
    #[must_use]
    pub const fn is_negative(self) -> bool {
        self.0[3] >> 63 == 1
    }

    /// Whether every bit is zero.
    #[must_use]
    pub const fn is_zero(self) -> bool {
        self.0[0] | self.0[1] | self.0[2] | self.0[3] == 0
    }

    /// Bitwise NOT.
    #[must_use]
    pub const fn not(self) -> Self {
        Self([!self.0[0], !self.0[1], !self.0[2], !self.0[3]])
    }

    /// Bitwise AND.
    #[must_use]
    pub const fn and(self, o: Self) -> Self {
        Self([
            self.0[0] & o.0[0],
            self.0[1] & o.0[1],
            self.0[2] & o.0[2],
            self.0[3] & o.0[3],
        ])
    }

    /// Bitwise OR.
    #[must_use]
    pub const fn or(self, o: Self) -> Self {
        Self([
            self.0[0] | o.0[0],
            self.0[1] | o.0[1],
            self.0[2] | o.0[2],
            self.0[3] | o.0[3],
        ])
    }

    /// Bitwise XOR.
    #[must_use]
    pub const fn xor(self, o: Self) -> Self {
        Self([
            self.0[0] ^ o.0[0],
            self.0[1] ^ o.0[1],
            self.0[2] ^ o.0[2],
            self.0[3] ^ o.0[3],
        ])
    }

    /// Addition modulo 2^256.
    #[must_use]
    pub fn wrapping_add(self, o: Self) -> Self {
        let mut out = [0u64; 4];
        let mut carry = false;
        for i in 0..4 {
            let (s1, c1) = self.0[i].overflowing_add(o.0[i]);
            let (s2, c2) = s1.overflowing_add(u64::from(carry));
            out[i] = s2;
            carry = c1 || c2;
        }
        Self(out)
    }

    /// Negation modulo 2^256.
    #[must_use]
    pub fn wrapping_neg(self) -> Self {
        self.not().wrapping_add(Self::ONE)
    }

    /// Subtraction modulo 2^256.
    #[must_use]
    pub fn wrapping_sub(self, o: Self) -> Self {
        self.wrapping_add(o.wrapping_neg())
    }

    /// Multiplication modulo 2^256 (the low 256 bits of the product, identical for signed and unsigned).
    #[must_use]
    pub fn wrapping_mul(self, o: Self) -> Self {
        let mut out = [0u64; 4];
        for i in 0..4 {
            let mut carry: u128 = 0;
            for j in 0..(4 - i) {
                let cur =
                    u128::from(out[i + j]) + u128::from(self.0[i]) * u128::from(o.0[j]) + carry;
                out[i + j] = cur as u64;
                carry = cur >> 64;
            }
        }
        Self(out)
    }

    /// Shift left by `n` bits. `n` must be below 256; the caller masks the count to the dtype width first.
    ///
    /// # Panics
    ///
    /// On `n >= 256`, which is a caller error (an unmasked count).
    #[must_use]
    pub fn shl(self, n: u32) -> Self {
        assert!(n < 256, "W256::shl count {n} is not below 256");
        let (limbs, bits) = ((n / 64) as usize, n % 64);
        let mut out = [0u64; 4];
        for i in limbs..4 {
            out[i] = self.0[i - limbs] << bits;
            if bits > 0 && i > limbs {
                out[i] |= self.0[i - limbs - 1] >> (64 - bits);
            }
        }
        Self(out)
    }

    /// Logical shift right by `n` bits (`n` below 256, as [`Self::shl`]).
    ///
    /// # Panics
    ///
    /// On `n >= 256`.
    #[must_use]
    pub fn shr_logical(self, n: u32) -> Self {
        assert!(n < 256, "W256::shr_logical count {n} is not below 256");
        let (limbs, bits) = ((n / 64) as usize, n % 64);
        let mut out = [0u64; 4];
        for i in 0..(4 - limbs) {
            out[i] = self.0[i + limbs] >> bits;
            if bits > 0 && i + limbs + 1 < 4 {
                out[i] |= self.0[i + limbs + 1] << (64 - bits);
            }
        }
        Self(out)
    }

    /// Arithmetic shift right by `n` bits (sign-filling, `n` below 256).
    ///
    /// # Panics
    ///
    /// On `n >= 256`.
    #[must_use]
    pub fn shr_arith(self, n: u32) -> Self {
        let logical = self.shr_logical(n);
        if self.is_negative() && n > 0 {
            logical.or(Self::ONES.shl(256 - n))
        } else {
            logical
        }
    }

    /// Order two words as unsigned.
    #[must_use]
    pub fn cmp_unsigned(self, o: Self) -> Ordering {
        for i in (0..4).rev() {
            match self.0[i].cmp(&o.0[i]) {
                Ordering::Equal => {}
                other => return other,
            }
        }
        Ordering::Equal
    }

    /// Order two words as signed 256-bit two's complement.
    #[must_use]
    pub fn cmp_signed(self, o: Self) -> Ordering {
        match (self.is_negative(), o.is_negative()) {
            (true, false) => Ordering::Less,
            (false, true) => Ordering::Greater,
            _ => self.cmp_unsigned(o),
        }
    }

    /// Order two patterns as values of dtype `t` (each already normalized to `t`, see [`Self::wrap_to`]).
    ///
    /// # Panics
    ///
    /// If `t` is not a wide integer dtype.
    #[must_use]
    pub fn cmp_as(self, o: Self, t: ElementKind) -> Ordering {
        if wide_int_signed(t).expect("cmp_as: not a wide integer dtype") {
            self.cmp_signed(o)
        } else {
            self.cmp_unsigned(o)
        }
    }

    /// Reduce to dtype `t`'s width and read it back in `t`'s signedness: the result is the canonical pattern of `t`
    /// (sign-extended for `I128`/`I256`, zero-extended for `U128`/`U256`). This is the wrap of every arithmetic op.
    ///
    /// # Panics
    ///
    /// If `t` is not a wide integer dtype.
    #[must_use]
    pub fn wrap_to(self, t: ElementKind) -> Self {
        let bits = wide_int_bits(t).expect("wrap_to: not a wide integer dtype");
        let signed = wide_int_signed(t).expect("wrap_to: not a wide integer dtype");
        if bits == 256 {
            return self;
        }
        if signed {
            Self::from_i128(self.low_u128() as i128)
        } else {
            Self::from_u128(self.low_u128())
        }
    }

    /// Clamp (`Saturation::Clamp`) a source value to the range of dtype `t`. `src_signed` says how to read `self`
    /// (a negative signed source into an unsigned target gives zero; an oversize source gives the target maximum).
    ///
    /// # Panics
    ///
    /// If `t` is not a wide integer dtype.
    #[must_use]
    pub fn clamp_to(self, src_signed: bool, t: ElementKind) -> Self {
        let (lo, hi) = wide_int_range(t).expect("clamp_to: not a wide integer dtype");
        let t_signed = wide_int_signed(t).expect("clamp_to: not a wide integer dtype");
        // The source value, read in its own signedness, against the target range read in the target's.
        match (src_signed && self.is_negative(), t_signed) {
            // A negative source into an unsigned target clamps to zero.
            (true, false) => lo,
            // A negative source into a signed target: in range unless below the target minimum (a narrower
            // source cannot be below a 256-bit minimum; a 256-bit source into I128 can).
            (true, true) => {
                if self.cmp_signed(lo) == Ordering::Less {
                    lo
                } else {
                    self
                }
            }
            // A non-negative source: compare unsigned against the target maximum (the maximum is non-negative).
            (false, _) => {
                if self.cmp_unsigned(hi) == Ordering::Greater {
                    hi
                } else {
                    self
                }
            }
        }
    }

    /// Unsigned quotient and remainder, or `None` when the divisor is zero (the existing trap).
    #[must_use]
    pub fn divrem_unsigned(self, d: Self) -> Option<(Self, Self)> {
        if d.is_zero() {
            return None;
        }
        let mut q = Self::ZERO;
        let mut r = Self::ZERO;
        for i in (0..256u32).rev() {
            r = r.shl(1);
            if self.shr_logical(i).0[0] & 1 == 1 {
                r.0[0] |= 1;
            }
            if r.cmp_unsigned(d) != Ordering::Less {
                r = r.wrapping_sub(d);
                q = q.or(Self::ONE.shl(i));
            }
        }
        Some((q, r))
    }

    /// Signed quotient (truncating toward zero) and remainder (sign of the dividend), or `None` on a zero divisor.
    /// `MIN / -1` wraps to `MIN` with remainder zero, as the narrower integers do.
    #[must_use]
    pub fn divrem_signed(self, d: Self) -> Option<(Self, Self)> {
        if d.is_zero() {
            return None;
        }
        let (n_neg, d_neg) = (self.is_negative(), d.is_negative());
        let abs = |x: Self, neg: bool| if neg { x.wrapping_neg() } else { x };
        let (q, r) = abs(self, n_neg).divrem_unsigned(abs(d, d_neg))?;
        let q = if n_neg != d_neg { q.wrapping_neg() } else { q };
        let r = if n_neg { r.wrapping_neg() } else { r };
        Some((q, r))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn u(v: u128) -> W256 {
        W256::from_u128(v)
    }
    fn i(v: i128) -> W256 {
        W256::from_i128(v)
    }

    #[test]
    fn add_sub_wrap_modulo_2_256() {
        assert_eq!(W256::ONES.wrapping_add(W256::ONE), W256::ZERO);
        assert_eq!(W256::ZERO.wrapping_sub(W256::ONE), W256::ONES);
        // a carry crossing every limb
        let a = W256([u64::MAX, u64::MAX, u64::MAX, 0]);
        assert_eq!(a.wrapping_add(W256::ONE), W256([0, 0, 0, 1]));
        assert_eq!(i(-5).wrapping_add(i(3)), i(-2));
    }

    #[test]
    fn mul_matches_i128_and_wraps_at_256() {
        assert_eq!(i(-7).wrapping_mul(i(6)), i(-42));
        assert_eq!(
            u(u128::MAX).wrapping_mul(u(u128::MAX)),
            // (2^128 - 1)^2 = 2^256 - 2^129 + 1
            W256([1, 0, u64::MAX - 1, u64::MAX])
        );
        // 2^255 * 2 wraps to zero
        assert_eq!(W256::ONE.shl(255).wrapping_mul(u(2)), W256::ZERO);
    }

    #[test]
    fn shifts_cross_limbs_and_fill_sign() {
        assert_eq!(W256::ONE.shl(64), W256([0, 1, 0, 0]));
        assert_eq!(W256::ONE.shl(255).shr_logical(255), W256::ONE);
        assert_eq!(i(-8).shr_arith(1), i(-4));
        assert_eq!(i(-1).shr_arith(255), W256::ONES);
        assert_eq!(u(0xF0).shr_arith(4), u(0xF));
        // 70 = 64 + 6: limb and bit part together
        assert_eq!(W256::ONE.shl(70).shr_logical(70), W256::ONE);
    }

    #[test]
    fn ranges_of_the_four_dtypes() {
        let (lo, hi) = wide_int_range(ElementKind::I128).unwrap();
        assert_eq!((lo, hi), (i(i128::MIN), i(i128::MAX)));
        let (lo, hi) = wide_int_range(ElementKind::U128).unwrap();
        assert_eq!((lo, hi), (W256::ZERO, u(u128::MAX)));
        let (lo, hi) = wide_int_range(ElementKind::I256).unwrap();
        assert_eq!(lo, W256::ONE.shl(255));
        assert_eq!(hi, W256::ONE.shl(255).wrapping_sub(W256::ONE));
        let (lo, hi) = wide_int_range(ElementKind::U256).unwrap();
        assert_eq!((lo, hi), (W256::ZERO, W256::ONES));
        assert!(wide_int_range(ElementKind::I64).is_none());
    }

    #[test]
    fn wrap_to_reads_back_in_the_dtypes_signedness() {
        // i128::MAX + 1 wraps to i128::MIN
        let past = i(i128::MAX).wrapping_add(W256::ONE);
        assert_eq!(past.wrap_to(ElementKind::I128), i(i128::MIN));
        // the same pattern read as U128 is 2^127, never negative
        assert_eq!(past.wrap_to(ElementKind::U128), u(1u128 << 127));
        // -1 as U128 is 2^128 - 1
        assert_eq!(i(-1).wrap_to(ElementKind::U128), u(u128::MAX));
        // 256-bit dtypes are the word itself
        assert_eq!(W256::ONES.wrap_to(ElementKind::U256), W256::ONES);
    }

    #[test]
    fn ordering_depends_on_signedness() {
        assert_eq!(i(-1).cmp_as(i(1), ElementKind::I256), Ordering::Less);
        assert_eq!(
            W256::ONES.cmp_as(i(1), ElementKind::U256),
            Ordering::Greater
        );
    }

    #[test]
    fn division_truncates_and_traps_on_zero() {
        assert_eq!(i(-7).divrem_signed(i(2)), Some((i(-3), i(-1))));
        assert_eq!(i(7).divrem_signed(i(-2)), Some((i(-3), i(1))));
        assert_eq!(u(100).divrem_unsigned(u(7)), Some((u(14), u(2))));
        assert_eq!(u(1).divrem_unsigned(W256::ZERO), None);
        assert_eq!(i(1).divrem_signed(W256::ZERO), None);
        // MIN / -1 wraps to MIN, remainder zero
        let min = W256::ONE.shl(255);
        assert_eq!(min.divrem_signed(i(-1)), Some((min, W256::ZERO)));
        // a quotient wider than 128 bits
        let big = W256::ONE.shl(200);
        assert_eq!(
            big.divrem_unsigned(W256::ONE.shl(100)),
            Some((W256::ONE.shl(100), W256::ZERO))
        );
    }

    #[test]
    fn clamp_saturates_per_target_range() {
        // -1 (signed source) into U128 clamps to 0
        assert_eq!(i(-1).clamp_to(true, ElementKind::U128), W256::ZERO);
        // U256::MAX into I128 clamps to i128::MAX
        assert_eq!(W256::ONES.clamp_to(false, ElementKind::I128), i(i128::MAX));
        // a signed I256 below i128::MIN clamps to i128::MIN
        let below = i(i128::MIN).wrapping_sub(W256::ONE);
        assert_eq!(below.clamp_to(true, ElementKind::I128), i(i128::MIN));
        // in-range values pass through
        assert_eq!(i(-5).clamp_to(true, ElementKind::I256), i(-5));
    }

    #[test]
    fn le_bytes_round_trip() {
        let v = W256([0x0102_0304_0506_0708, 9, 10, 0xFFFF_FFFF_FFFF_FFFF]);
        assert_eq!(W256::from_le_bytes(v.to_le_bytes()), v);
        assert_eq!(u(0x0201).to_le_bytes()[..2], [1, 2]);
    }
}
