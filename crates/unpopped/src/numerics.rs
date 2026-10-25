// SPDX-License-Identifier: MIT OR Apache-2.0
//! The semantics of [`ScalarExpr::Qualified`](crate::ir::ScalarExpr::Qualified):
//! flush-to-zero, directed rounding and the `approx` bound
//! (`docs/ir-numerics-qualifiers.md`).
//!
//! Everything here is the **reference**: the oracle calls [`eval_qualified_f32`],
//! and backend tests call [`approx_within`]. Directed rounding is built from the
//! exact residual of the operation (an error-free transform, or the sign of an
//! exactly-computed remainder), never from the host's rounding mode, so it is
//! independent of any emitter. `tests/numerics_qualifiers.rs` checks it against
//! exact integer arithmetic.

use crate::ir::{Numerics, QualOp, RoundMode};

/// The `approx` bound for every operation but `Log2`, in ulps of the exact result.
pub const APPROX_ULP: f64 = 4.0;

/// `Log2`'s absolute floor: near `log2(1) = 0` no hardware approximation meets an
/// ulp bound, so the bound is `max(4 ulp, 2^-20)`.
pub const LOG2_ABS_FLOOR: f64 = 1.0 / 1_048_576.0;

/// Flush a subnormal to a zero of the same sign (PTX `.ftz`); everything else,
/// NaN and infinity included, is returned unchanged.
#[must_use]
pub fn ftz_f32(x: f32) -> f32 {
    if x.is_subnormal() {
        0.0f32.copysign(x)
    } else {
        x
    }
}

/// The exponent `k` with `2^k <= |x| < 2^(k+1)` for a normal `x`.
fn exponent(x: f32) -> i32 {
    (((x.to_bits() >> 23) & 0xff) as i32) - 127
}

/// Is `got` within the `approx` contract of the exact (correctly rounded) result?
///
/// The contract holds on the domain "finite normal operands, finite normal exact
/// result"; the caller restricts to it. Outside it (`exact` zero, subnormal,
/// infinite or NaN) nothing is promised and this returns `true`, which is why a
/// backend test must also run inputs that land IN the domain (its control).
#[must_use]
pub fn approx_within(op: QualOp, got: f32, exact: f32) -> bool {
    if !exact.is_normal() {
        return true;
    }
    if !got.is_finite() {
        return false;
    }
    let ulp = 2f64.powi(exponent(exact) - 23);
    let mut bound = APPROX_ULP * ulp;
    if op == QualOp::Log2 {
        bound = bound.max(LOG2_ABS_FLOOR);
    }
    (f64::from(got) - f64::from(exact)).abs() <= bound
}

fn next_up(f: f32) -> f32 {
    if f.is_nan() || f == f32::INFINITY {
        return f;
    }
    if f == 0.0 {
        return f32::from_bits(1);
    }
    let b = f.to_bits();
    f32::from_bits(if f > 0.0 { b + 1 } else { b - 1 })
}

fn next_down(f: f32) -> f32 {
    -next_up(-f)
}

fn sign(x: f64) -> i32 {
    if x > 0.0 {
        1
    } else if x < 0.0 {
        -1
    } else {
        0
    }
}

/// IEEE 754 7.4: the result of an overflowing operation in a directed mode.
fn overflow(negative: bool, round: RoundMode) -> f32 {
    let up = match round {
        RoundMode::TowardPosInf => true,
        RoundMode::TowardNegInf => false,
        _ => negative, // toward zero: a negative overflow rounds up, to -MAX
    };
    match (negative, up) {
        (false, true) => f32::INFINITY,
        (false, false) => f32::MAX,
        (true, true) => -f32::MAX,
        (true, false) => f32::NEG_INFINITY,
    }
}

/// `f` is the nearest f32 to the exact result `x`; `cmp` is the sign of `x - f`.
/// Return the directed rounding of `x`.
fn directed(f: f32, cmp: i32, round: RoundMode) -> f32 {
    if cmp == 0 {
        return f;
    }
    let positive = if f == 0.0 { cmp > 0 } else { f > 0.0 };
    let up = match round {
        RoundMode::TowardPosInf => true,
        RoundMode::TowardNegInf => false,
        _ => !positive,
    };
    if up {
        if cmp > 0 { next_up(f) } else { f }
    } else if cmp < 0 {
        next_down(f)
    } else {
        f
    }
}

/// `x = s + e` exactly (`s`, `e` doubles), rounded to f32 in `round`.
fn round_pair(s: f64, e: f64, round: RoundMode) -> f32 {
    let f = s as f32;
    if f.is_infinite() {
        return overflow(s < 0.0, round);
    }
    directed(f, sign((s - f64::from(f)) + e), round)
}

fn zero_sign(negative: bool) -> f32 {
    if negative { -0.0 } else { 0.0 }
}

/// Error-free sum: `s + e == a + b` exactly, `s = fl(a + b)`.
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let bb = s - a;
    (s, (a - (s - bb)) + (b - bb))
}

/// `a + b` (with `b` already negated for a subtraction) in a directed mode.
fn add_directed(a: f32, b: f32, round: RoundMode) -> f32 {
    if !a.is_finite() || !b.is_finite() {
        return a + b;
    }
    let (s, e) = two_sum(f64::from(a), f64::from(b));
    if s == 0.0 && e == 0.0 {
        // An exact zero: like-signed zeros keep their sign, every other exact
        // cancellation is +0 except toward -inf (IEEE 754 6.3).
        let negative = if a == 0.0 && b == 0.0 && a.is_sign_negative() == b.is_sign_negative() {
            a.is_sign_negative()
        } else {
            round == RoundMode::TowardNegInf
        };
        return zero_sign(negative);
    }
    round_pair(s, e, round)
}

fn fma_directed(a: f32, b: f32, c: f32, round: RoundMode) -> f32 {
    if !a.is_finite() || !b.is_finite() || !c.is_finite() {
        return a.mul_add(b, c);
    }
    // The product of two f32 is exact in an f64.
    let p = f64::from(a) * f64::from(b);
    let (s, e) = two_sum(p, f64::from(c));
    if s == 0.0 && e == 0.0 {
        let negative = if p == 0.0 && c == 0.0 && p.is_sign_negative() == c.is_sign_negative() {
            c.is_sign_negative()
        } else {
            round == RoundMode::TowardNegInf
        };
        return zero_sign(negative);
    }
    round_pair(s, e, round)
}

fn div_directed(a: f32, b: f32, round: RoundMode) -> f32 {
    if !a.is_finite() || !b.is_finite() || a == 0.0 || b == 0.0 {
        return a / b;
    }
    let (a64, b64) = (f64::from(a), f64::from(b));
    let f = (a64 / b64) as f32;
    if f.is_infinite() {
        return overflow((a < 0.0) != (b < 0.0), round);
    }
    // sign(x - f) = sign(a - f*b) * sign(b); the fused product-difference is exact.
    let r = (-f64::from(f)).mul_add(b64, a64);
    directed(f, sign(r) * sign(b64), round)
}

fn sqrt_directed(a: f32, round: RoundMode) -> f32 {
    if a.is_nan() || a <= 0.0 || a.is_infinite() {
        return a.sqrt();
    }
    let a64 = f64::from(a);
    let f = a64.sqrt() as f32;
    // sign(sqrt(a) - f) = sign(a - f*f) for f > 0; the fused difference is exact.
    let r = (-f64::from(f)).mul_add(f64::from(f), a64);
    directed(f, sign(r), round)
}

/// Evaluate a qualified operation on f32 operands per [`Numerics`]:
/// flush the operands (if `ftz`), perform the operation in `n.round`, flush the
/// result (if `ftz`). `approx` evaluates the exact function (inside its own bound).
///
/// `args.len()` must equal `op.arity()`; the plan gate guarantees it.
#[must_use]
pub fn eval_qualified_f32(op: QualOp, n: Numerics, args: &[f32]) -> f32 {
    assert_eq!(args.len(), op.arity(), "qualified operand count");
    let a: Vec<f32> = args
        .iter()
        .map(|&x| if n.ftz { ftz_f32(x) } else { x })
        .collect();
    let r = match n.round {
        RoundMode::NearestEven => nearest(op, &a),
        round => match op {
            QualOp::Add => add_directed(a[0], a[1], round),
            QualOp::Sub => add_directed(a[0], -a[1], round),
            QualOp::Mul => {
                if a[0].is_finite() && a[1].is_finite() {
                    round_pair(f64::from(a[0]) * f64::from(a[1]), 0.0, round)
                } else {
                    a[0] * a[1]
                }
            }
            QualOp::Div => div_directed(a[0], a[1], round),
            QualOp::Fma => fma_directed(a[0], a[1], a[2], round),
            QualOp::Sqrt => sqrt_directed(a[0], round),
            other => panic!(
                "numerics: {other:?} takes no directed rounding (the plan gate rejects this)"
            ),
        },
    };
    if n.ftz { ftz_f32(r) } else { r }
}

fn nearest(op: QualOp, a: &[f32]) -> f32 {
    match op {
        QualOp::Add => a[0] + a[1],
        QualOp::Sub => a[0] - a[1],
        QualOp::Mul => a[0] * a[1],
        QualOp::Div => a[0] / a[1],
        QualOp::Fma => a[0].mul_add(a[1], a[2]),
        QualOp::Sqrt => a[0].sqrt(),
        QualOp::Rsqrt => (1.0 / f64::from(a[0]).sqrt()) as f32,
        QualOp::Recip => 1.0 / a[0],
        QualOp::Exp2 => f64::from(a[0]).exp2() as f32,
        QualOp::Log2 => f64::from(a[0]).log2() as f32,
    }
}
