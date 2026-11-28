// SPDX-License-Identifier: MIT OR Apache-2.0
//! Directed rounding at **f64** (ROADMAP U43): the reference for `Qualified` `Add`, `Sub`, `Mul`, `Div`, `Fma` and `Sqrt` under
//! `RoundMode::{TowardZero, TowardNegInf, TowardPosInf}` at f64 operands.
//!
//! The f32 reference ([`super::numerics`]) rests on an f64 error-free transform; f64 has no wider float, so this one is **integer
//! arithmetic**: every operand is split into `(sign, integer mantissa, power of two)`, the exact result is formed in a [`W256`]
//! (a sum aligned into a 255-bit frame, a product of two 53-bit mantissas, a quotient or an integer square root with a sticky
//! remainder), and ONE directed rounding to 53 bits (or fewer, in the subnormal range) is applied to that exact value. It never
//! reads the host's rounding mode, so it does not depend on an emitter. `tests/numerics_f64_directed.rs` checks it against an
//! independent Python exact-rational reference.

use crate::ir::{Numerics, QualOp, RoundMode};
use crate::wide_int::W256;
use core::cmp::Ordering;

/// `neg`, `m * 2^e`, plus `sticky`: a nonzero fraction strictly below one unit of `m` that the integer part dropped. A zero `m` is
/// an exact zero (never sticky).
#[derive(Clone, Copy)]
struct Exact {
    neg: bool,
    m: W256,
    e: i32,
    sticky: bool,
}

fn term(x: f64) -> Exact {
    let bits = x.to_bits();
    let biased = ((bits >> 52) & 0x7ff) as i32;
    let frac = bits & ((1u64 << 52) - 1);
    let (m, e) = if biased == 0 {
        (frac, -1074)
    } else {
        (frac | (1u64 << 52), biased - 1075)
    };
    Exact {
        neg: x.is_sign_negative(),
        m: W256::from_u128(u128::from(m)),
        e,
        sticky: false,
    }
}

/// `2^k` for `k` in `-1074..=1023`.
fn pow2(k: i32) -> f64 {
    if k >= -1022 {
        f64::from_bits(((k + 1023) as u64) << 52)
    } else {
        f64::from_bits(1u64 << (k + 1074))
    }
}

/// IEEE 754 7.4: the result of an overflowing operation in a directed mode.
fn overflow(neg: bool, round: RoundMode) -> f64 {
    let up = match round {
        RoundMode::TowardPosInf => true,
        RoundMode::TowardNegInf => false,
        _ => neg,
    };
    match (neg, up) {
        (false, true) => f64::INFINITY,
        (false, false) => f64::MAX,
        (true, true) => -f64::MAX,
        (true, false) => f64::NEG_INFINITY,
    }
}

/// One directed rounding of the exact nonzero value `x` to f64 (subnormals and overflow included).
fn round_exact(x: Exact, round: RoundMode) -> f64 {
    let len = x.m.bit_length() as i32;
    // the weight of the last kept bit: 53 bits for a normal result, down to 2^-1074 for a subnormal one
    let lsb = (len + x.e - 53).max(-1074);
    let drop = lsb - x.e;
    let (mut kept, inexact) = if drop <= 0 {
        debug_assert!(!x.sticky, "a sticky value always carries more than 53 bits");
        (x.m.shift_left((-drop) as u32).low_u128() as u64, false)
    } else if drop >= 256 {
        (0, true)
    } else {
        let d = drop as u32;
        let kept = x.m.shr_logical(d).low_u128() as u64;
        let back = if kept == 0 {
            W256::ZERO
        } else {
            W256::from_u128(u128::from(kept)).shift_left(d)
        };
        (kept, x.sticky || x.m != back)
    };
    let up = match round {
        RoundMode::TowardPosInf => !x.neg,
        RoundMode::TowardNegInf => x.neg,
        _ => false,
    };
    if up && inexact {
        kept += 1;
    }
    let kept_bits = 64 - kept.leading_zeros() as i32;
    if kept_bits + lsb > 1024 {
        return overflow(x.neg, round);
    }
    let mag = (kept as f64) * pow2(lsb);
    if x.neg { -mag } else { mag }
}

/// The exact sum `x + y` of two nonzero terms, as one [`Exact`], or `None` when they cancel exactly.
fn exact_sum(x: Exact, y: Exact) -> Option<Exact> {
    // each term at the top of a 255-bit frame, so the larger has the larger scale
    let frame = |t: Exact| -> (W256, i32) {
        let s = 255 - t.m.bit_length();
        (t.m.shift_left(s), t.e - s as i32)
    };
    let (mut xm, mut xe) = frame(x);
    let (mut ym, mut ye) = frame(y);
    let (mut xneg, mut yneg) = (x.neg, y.neg);
    if xe < ye || (xe == ye && xm.cmp_unsigned(ym) == Ordering::Less) {
        core::mem::swap(&mut xm, &mut ym);
        core::mem::swap(&mut xe, &mut ye);
        core::mem::swap(&mut xneg, &mut yneg);
    }
    // align the smaller (shifted right, the lost bits are the sticky remainder)
    let gap = (xe - ye) as u32;
    let (ym, sticky) = if gap == 0 {
        (ym, false)
    } else if gap >= 256 {
        (W256::ZERO, true)
    } else {
        let kept = ym.shr_logical(gap);
        (kept, kept.shift_left(gap) != ym)
    };
    if xneg == yneg {
        // the frame is 255 bits and each term is below 2^255: the sum fits 256 bits
        return Some(Exact {
            neg: xneg,
            m: xm.wrapping_add(ym),
            e: xe,
            sticky,
        });
    }
    let mut m = xm.wrapping_sub(ym);
    if sticky {
        // X - (Y_kept + f) = (X - Y_kept - 1) + (1 - f), and 0 < 1 - f < 1
        m = m.wrapping_sub(W256::ONE);
    }
    if m.is_zero() && !sticky {
        return None;
    }
    Some(Exact {
        neg: xneg,
        m,
        e: xe,
        sticky,
    })
}

/// The sign of an exact zero result: like-signed zero terms keep their sign, every other cancellation is `+0` except toward `-inf`
/// (IEEE 754 6.3).
fn zero_result(xneg: bool, yneg: bool, both_zero: bool, round: RoundMode) -> f64 {
    let neg = if both_zero && xneg == yneg {
        xneg
    } else {
        round == RoundMode::TowardNegInf
    };
    if neg { -0.0 } else { 0.0 }
}

/// `x + y` for two terms (either may be an exact zero) in `round`.
fn add_terms(x: Exact, y: Exact, round: RoundMode) -> f64 {
    match (x.m.is_zero(), y.m.is_zero()) {
        (true, true) => zero_result(x.neg, y.neg, true, round),
        (true, false) => round_exact(y, round),
        (false, true) => round_exact(x, round),
        (false, false) => match exact_sum(x, y) {
            Some(s) => round_exact(s, round),
            None => zero_result(x.neg, y.neg, false, round),
        },
    }
}

fn product(a: f64, b: f64) -> Exact {
    let (x, y) = (term(a), term(b));
    Exact {
        neg: x.neg != y.neg,
        m: x.m.wrapping_mul(y.m),
        e: x.e + y.e,
        sticky: false,
    }
}

fn quotient(a: f64, b: f64) -> Exact {
    let (x, y) = (term(a), term(b));
    // 200 extra bits: the quotient carries well over 53 bits whatever the operands' lengths
    let (q, r) =
        x.m.shift_left(200)
            .divrem_unsigned(y.m)
            .expect("a nonzero divisor");
    Exact {
        neg: x.neg != y.neg,
        m: q,
        e: x.e - y.e - 200,
        sticky: !r.is_zero(),
    }
}

/// `floor(sqrt(n))` for `n < 2^256`.
fn isqrt(n: W256) -> W256 {
    let mut root = W256::ZERO;
    for i in (0..128u32).rev() {
        let cand = root.or(W256::ONE.shift_left(i));
        if cand.wrapping_mul(cand).cmp_unsigned(n) != Ordering::Greater {
            root = cand;
        }
    }
    root
}

fn square_root(a: f64) -> Exact {
    let mut x = term(a);
    if x.e % 2 != 0 {
        x.m = x.m.shift_left(1);
        x.e -= 1;
    }
    // sqrt(m * 2^e) = sqrt(m * 2^200) * 2^(e/2 - 100); the 53-bit m keeps m * 2^200 inside 256 bits
    let n = x.m.shift_left(200);
    let root = isqrt(n);
    Exact {
        neg: false,
        m: root,
        e: x.e / 2 - 100,
        sticky: root.wrapping_mul(root) != n,
    }
}

/// `op` over f64 operands in a directed `round` (`TowardZero`, `TowardNegInf` or `TowardPosInf`), one rounding of the exact result.
/// A non-finite operand, an exact zero operand of a division or a root, and every NaN follow IEEE 754 as the host computes them
/// (none depends on the direction).
///
/// # Panics
///
/// On an operation that takes no directed rounding, on `NearestEven` or `NearestAway`, or on a wrong operand count (the type check
/// rejects each).
#[must_use]
pub fn directed_f64(op: QualOp, round: RoundMode, args: &[f64]) -> f64 {
    assert!(
        matches!(
            round,
            RoundMode::TowardZero | RoundMode::TowardNegInf | RoundMode::TowardPosInf
        ),
        "numerics: directed_f64 takes a directed mode, got {round:?}"
    );
    assert_eq!(args.len(), op.arity(), "qualified operand count");
    if args.iter().any(|v| !v.is_finite()) {
        return native(op, args);
    }
    match op {
        QualOp::Add => add_terms(term(args[0]), term(args[1]), round),
        QualOp::Sub => add_terms(term(args[0]), term(-args[1]), round),
        QualOp::Mul => {
            if args[0] == 0.0 || args[1] == 0.0 {
                return args[0] * args[1];
            }
            round_exact(product(args[0], args[1]), round)
        }
        QualOp::Div => {
            if args[0] == 0.0 || args[1] == 0.0 {
                return args[0] / args[1];
            }
            round_exact(quotient(args[0], args[1]), round)
        }
        QualOp::Fma => add_terms(product(args[0], args[1]), term(args[2]), round),
        QualOp::Sqrt => {
            if args[0] <= 0.0 {
                return args[0].sqrt();
            }
            round_exact(square_root(args[0]), round)
        }
        other => {
            panic!("numerics: {other:?} takes no directed rounding (the plan gate rejects this)")
        }
    }
}

fn native(op: QualOp, a: &[f64]) -> f64 {
    match op {
        QualOp::Add => a[0] + a[1],
        QualOp::Sub => a[0] - a[1],
        QualOp::Mul => a[0] * a[1],
        QualOp::Div => a[0] / a[1],
        QualOp::Fma => a[0].mul_add(a[1], a[2]),
        QualOp::Sqrt => a[0].sqrt(),
        other => {
            panic!("numerics: {other:?} takes no directed rounding (the plan gate rejects this)")
        }
    }
}

/// A `Qualified` operation at f64 per [`Numerics`]: `NearestEven` is the host operation (it is correctly rounded), a directed mode is
/// [`directed_f64`].
///
/// # Panics
///
/// On `ftz` or `approx` (f64 has no flush form, and no `approx` bound is defined at f64), on ties-away, and on an operation outside
/// `Add`, `Sub`, `Mul`, `Div`, `Fma`, `Sqrt`: `check_kernel` rejects each by name.
#[must_use]
pub fn eval_qualified_f64(op: QualOp, n: Numerics, args: &[f64]) -> f64 {
    assert!(
        !n.ftz && !n.approx,
        "numerics: an f64 qualifier is a rounding mode only (the type check rejects ftz and approx at f64)"
    );
    match n.round {
        RoundMode::NearestEven => {
            assert_eq!(args.len(), op.arity(), "qualified operand count");
            native(op, args)
        }
        RoundMode::NearestAway => {
            panic!(
                "numerics: an arithmetic operation takes no ties-away rounding (the type check rejects this)"
            )
        }
        round => directed_f64(op, round, args),
    }
}
