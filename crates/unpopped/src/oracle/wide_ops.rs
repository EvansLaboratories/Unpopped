// SPDX-License-Identifier: MIT OR Apache-2.0
//! Operations of the wide integer dtypes (`I128`, `U128`, `I256`, `U256`) on [`W256`] values (ROADMAP E2, slice 2b).
//!
//! Each rule is the narrow rule of `docs/int-div-shift-semantics.md` stated at the wide width: arithmetic wraps
//! (two's complement) at the dtype's width, a shift count is read as unsigned and taken modulo the width, a zero divisor
//! gives the dividend, and a signed `MIN / -1` gives `MIN`. A comparison or logical operator yields the compute type's 1/0
//! (carried as `Val::Float`, like the narrow integer path). Nothing here touches a narrower dtype's path.
//!
//! The carry of the 256-bit add is computed in [`W256::wrapping_add`] from native `u64` limb sums, not from a
//! carry-flag instruction chain (see the E2 chart note in HANDOFF).

use super::*;
use crate::cast_ref::{Num, bits_of};
use crate::ir::Saturation;
use crate::wide_int::{W256, wide_int_bits, wide_int_range, wide_int_signed};
use core::cmp::Ordering;

/// A wide value read as `t`: only a `Val::Wide` of that dtype is one.
///
/// # Panics
///
/// On any other value (the type checker admits a wide operation only over wide operands of one dtype).
pub(super) fn as_wide(v: Val, t: ElementKind) -> W256 {
    match v {
        Val::Wide(w, dt) if dt == t => w,
        other => panic!("oracle: expected a {t:?} wide integer, got {other:?}"),
    }
}

/// `a + b`, `a - b`, `a * b` wrapped to `t`.
pub(super) fn wide_add(a: W256, b: W256, t: ElementKind) -> W256 {
    a.wrapping_add(b).wrap_to(t)
}

/// `a - b` wrapped to `t`.
pub(super) fn wide_sub(a: W256, b: W256, t: ElementKind) -> W256 {
    a.wrapping_sub(b).wrap_to(t)
}

/// `a * b` wrapped to `t`.
pub(super) fn wide_mul(a: W256, b: W256, t: ElementKind) -> W256 {
    a.wrapping_mul(b).wrap_to(t)
}

/// Integer `a / b` by the total rule: a zero divisor gives `a`, signed `MIN / -1` gives `MIN`, anything else truncates
/// toward zero.
pub(super) fn wide_div(a: W256, b: W256, t: ElementKind) -> W256 {
    if b.is_zero() {
        return a;
    }
    let qr = if wide_int_signed(t).expect("wide_div: not a wide integer dtype") {
        a.divrem_signed(b)
    } else {
        a.divrem_unsigned(b)
    };
    qr.expect("nonzero divisor").0.wrap_to(t)
}

/// Order `a` against `b` as values of `t`.
pub(super) fn wide_cmp(a: W256, b: W256, t: ElementKind) -> Ordering {
    a.cmp_as(b, t)
}

/// A comparison operator's 1/0 over two wide values, or `None` when `op` is not a comparison.
pub(super) fn wide_compare(op: BinaryOp, a: W256, b: W256, t: ElementKind) -> Option<bool> {
    let o = wide_cmp(a, b, t);
    Some(match op {
        BinaryOp::CmpEq => o == Ordering::Equal,
        BinaryOp::CmpNe => o != Ordering::Equal,
        BinaryOp::CmpLt => o == Ordering::Less,
        BinaryOp::CmpLe => o != Ordering::Greater,
        BinaryOp::CmpGt => o == Ordering::Greater,
        BinaryOp::CmpGe => o != Ordering::Less,
        _ => return None,
    })
}

/// A bitwise, shift or ordering operator over two wide values of `t` (the integer-only family plus `Max`/`Min`).
///
/// # Panics
///
/// If `op` is not one of `BitAnd`, `BitOr`, `BitXor`, `Shl`, `Shr`, `Max`, `Min`.
pub(super) fn wide_binary(op: BinaryOp, a: W256, b: W256, t: ElementKind) -> W256 {
    let width = wide_int_bits(t).expect("wide_binary: not a wide integer dtype");
    // The count is the operand's bits read as unsigned, modulo the width (a power of two).
    let count = (b.low_u128() & u128::from(width - 1)) as u32;
    let signed = wide_int_signed(t).expect("wide_binary: not a wide integer dtype");
    match op {
        BinaryOp::BitAnd => a.and(b).wrap_to(t),
        BinaryOp::BitOr => a.or(b).wrap_to(t),
        BinaryOp::BitXor => a.xor(b).wrap_to(t),
        BinaryOp::Shl => a.shift_left(count).wrap_to(t),
        BinaryOp::Shr if signed => a.shr_arith(count),
        BinaryOp::Shr => a.shr_logical(count),
        BinaryOp::Max => {
            if wide_cmp(a, b, t) == Ordering::Less {
                b
            } else {
                a
            }
        }
        BinaryOp::Min => {
            if wide_cmp(a, b, t) == Ordering::Greater {
                b
            } else {
                a
            }
        }
        other => panic!("oracle: {other:?} has no wide integer form"),
    }
}

/// Convert integer `w` of dtype `from` to dtype `to`, at least one of them wide, by `sat` (`Wrap` or `Clamp`).
///
/// # Panics
///
/// If `sat` is neither `Wrap` nor `Clamp`, or a dtype is not an integer of this version.
pub(super) fn wide_cast(w: W256, from: ElementKind, to: ElementKind, sat: Saturation) -> Val {
    let src_signed = wide_int_signed(from)
        .or_else(|| crate::types::int_range(from).map(|(lo, _)| lo < 0))
        .expect("wide_cast: source is not an integer dtype");
    let wide_to = wide_int_range(to);
    let (lo, hi) = match wide_to {
        Some(r) => r,
        None => {
            let (l, h) =
                crate::types::int_range(to).expect("wide_cast: target is not an integer dtype");
            (W256::from_i128(l), W256::from_i128(h))
        }
    };
    let v = match sat {
        Saturation::Wrap => w,
        Saturation::Clamp => {
            if src_signed && w.is_negative() {
                if w.cmp_signed(lo) == Ordering::Less {
                    lo
                } else {
                    w
                }
            } else if w.cmp_unsigned(hi) == Ordering::Greater {
                hi
            } else {
                w
            }
        }
        other => panic!("oracle: {other:?} is not a defined integer-to-integer saturation"),
    };
    match wide_to {
        Some(_) => Val::Wide(v.wrap_to(to), to),
        // A narrow target: the low bits as its wrapped pattern.
        None => Val::Raw(bits_of(Num::I(v.low_u128() as i128), to), to),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ElementKind::{I8, I64, I128, I256, U8, U128, U256};

    fn w(v: i128) -> W256 {
        W256::from_i128(v)
    }

    #[test]
    fn add_wraps_at_each_width() {
        // U128 max + 1 wraps to 0; I128 max + 1 wraps to I128 min; the 256-bit types wrap at 256.
        let u128max = W256::from_u128(u128::MAX);
        assert_eq!(wide_add(u128max, W256::ONE, U128), W256::ZERO);
        let i128max = W256::from_i128(i128::MAX);
        assert_eq!(
            wide_add(i128max, W256::ONE, I128),
            W256::from_i128(i128::MIN)
        );
        assert_eq!(wide_add(W256::ONES, W256::ONE, U256), W256::ZERO);
        assert_eq!(wide_sub(W256::ZERO, W256::ONE, U256), W256::ONES);
        assert_eq!(
            wide_sub(W256::ZERO, W256::ONE, U128),
            W256::from_u128(u128::MAX)
        );
        assert_eq!(wide_mul(u128max, u128max, U128), W256::ONE);
    }

    #[test]
    fn div_total_rule() {
        assert_eq!(
            wide_div(w(7), W256::ZERO, I256),
            w(7),
            "zero divisor gives the dividend"
        );
        let min = W256::from_i128(i128::MIN);
        assert_eq!(wide_div(min, w(-1), I128), min, "MIN / -1 gives MIN");
        assert_eq!(wide_div(w(-7), w(2), I128), w(-3), "truncates toward zero");
        assert_eq!(wide_div(W256::ONES, w(2), U256), W256::ONES.shr_logical(1));
    }

    #[test]
    fn compare_reads_signedness() {
        // All-ones is -1 as I256 and the maximum as U256.
        assert_eq!(
            wide_compare(BinaryOp::CmpLt, W256::ONES, W256::ZERO, I256),
            Some(true)
        );
        assert_eq!(
            wide_compare(BinaryOp::CmpLt, W256::ONES, W256::ZERO, U256),
            Some(false)
        );
        assert_eq!(
            wide_compare(BinaryOp::CmpEq, W256::ONES, W256::ONES, U256),
            Some(true)
        );
        assert_eq!(
            wide_compare(BinaryOp::Max, W256::ONES, W256::ONES, U256),
            None
        );
        assert_eq!(
            wide_binary(BinaryOp::Max, W256::ONES, W256::ZERO, I256),
            W256::ZERO
        );
        assert_eq!(
            wide_binary(BinaryOp::Max, W256::ONES, W256::ZERO, U256),
            W256::ONES
        );
        assert_eq!(
            wide_binary(BinaryOp::Min, W256::ONES, W256::ZERO, I256),
            W256::ONES
        );
    }

    #[test]
    fn shifts_take_the_count_modulo_the_width() {
        // 130 mod 128 = 2; 256 mod 256 = 0.
        assert_eq!(wide_binary(BinaryOp::Shl, W256::ONE, w(130), U128), w(4));
        assert_eq!(
            wide_binary(BinaryOp::Shl, W256::ONE, w(256), U256),
            W256::ONE
        );
        // A shift left past the top of a 128-bit type drops the bit.
        assert_eq!(
            wide_binary(BinaryOp::Shl, W256::ONE, w(127), I128),
            W256::from_i128(i128::MIN)
        );
        // Arithmetic for signed, logical for unsigned.
        assert_eq!(wide_binary(BinaryOp::Shr, w(-8), w(1), I256), w(-4));
        assert_eq!(
            wide_binary(BinaryOp::Shr, W256::ONES, w(255), U256),
            W256::ONE
        );
        assert_eq!(
            wide_binary(BinaryOp::Shr, w(i128::MIN), w(127), I128),
            w(-1)
        );
        // A negative count reads its two's-complement bits: -1 is 127 at 128 bits.
        assert_eq!(
            wide_binary(BinaryOp::Shr, W256::from_u128(1 << 127), w(-1), U128),
            W256::ONE
        );
    }

    #[test]
    fn bitwise_stays_canonical() {
        let a = W256::from_i128(-2);
        assert_eq!(wide_binary(BinaryOp::BitAnd, a, W256::ONES, I128), a);
        assert_eq!(wide_binary(BinaryOp::BitXor, a, a, I256), W256::ZERO);
        assert_eq!(
            wide_binary(
                BinaryOp::BitOr,
                W256::from_u128(1),
                W256::from_u128(2),
                U128
            ),
            W256::from_u128(3)
        );
    }

    #[test]
    fn casts_wrap_and_clamp() {
        // I64 -1 -> U128 wraps to 2^128 - 1; clamps to 0.
        assert!(matches!(
            wide_cast(w(-1), I64, U128, Saturation::Wrap),
            Val::Wide(x, U128) if x == W256::from_u128(u128::MAX)
        ));
        assert!(matches!(
            wide_cast(w(-1), I64, U128, Saturation::Clamp),
            Val::Wide(x, U128) if x == W256::ZERO
        ));
        // U256 all-ones -> I128: clamp to the I128 maximum, wrap to -1.
        assert!(matches!(
            wide_cast(W256::ONES, U256, I128, Saturation::Clamp),
            Val::Wide(x, I128) if x == W256::from_i128(i128::MAX)
        ));
        assert!(matches!(
            wide_cast(W256::ONES, U256, I128, Saturation::Wrap),
            Val::Wide(x, I128) if x == W256::from_i128(-1)
        ));
        // I256 min -> I128: clamp to the I128 minimum.
        let i256_min = W256::ONE.shift_left(255);
        assert!(matches!(
            wide_cast(i256_min, I256, I128, Saturation::Clamp),
            Val::Wide(x, I128) if x == W256::from_i128(i128::MIN)
        ));
        // Wide to narrow: 300 -> U8 wraps to 44, clamps to 255; -5 -> U8 clamps to 0.
        assert!(matches!(
            wide_cast(w(300), I128, U8, Saturation::Wrap),
            Val::Raw(44, U8)
        ));
        assert!(matches!(
            wide_cast(w(300), I128, U8, Saturation::Clamp),
            Val::Raw(255, U8)
        ));
        assert!(matches!(
            wide_cast(w(-5), I128, U8, Saturation::Clamp),
            Val::Raw(0, U8)
        ));
        assert!(matches!(
            wide_cast(w(-200), I256, I8, Saturation::Clamp),
            Val::Raw(0x80, I8)
        ));
        // U256 above the I128 positive range into I8 clamps to 127, not to a negative.
        assert!(matches!(
            wide_cast(W256::ONES, U256, I8, Saturation::Clamp),
            Val::Raw(0x7f, I8)
        ));
        // Narrow to wide keeps the value: U8 255 -> I256 is 255.
        assert!(
            matches!(wide_cast(w(255), U8, I256, Saturation::Wrap), Val::Wide(x, I256) if x == w(255))
        );
    }
}
