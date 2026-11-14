// SPDX-License-Identifier: MIT OR Apache-2.0
//! Independent scalar op semantics (match the DEFINITIONS, not any emitter's spelling).
//!
//! A section of the oracle (ROADMAP U30): `oracle.rs` glob-imports it, so every path it had is unchanged.

use super::*;

// ===========================================================================
// Independent scalar op semantics (match DEFINITIONS, not CUDA spellings).
// ===========================================================================

/// Independent `f64` unary-op evaluator (op DEFINITIONS, not the emitter's
/// spelling). Transcendentals via Rust std / the local erf/lgamma helpers.
pub(super) fn unary_op_f64(op: UnaryOp, x: f64) -> f64 {
    use std::f64::consts::FRAC_1_SQRT_2;
    match op {
        UnaryOp::Neg => -x,
        UnaryOp::Abs => x.abs(),
        UnaryOp::Sqr => x * x,
        UnaryOp::Sqrt => x.sqrt(),
        UnaryOp::Rsqrt => 1.0 / x.sqrt(),
        UnaryOp::Recip => 1.0 / x,
        UnaryOp::Exp => x.exp(),
        UnaryOp::Log => x.ln(),
        UnaryOp::Tanh => x.tanh(),
        UnaryOp::Sigmoid => 1.0 / (1.0 + (-x).exp()),
        // Relu = x<0?0:x — NaN-propagating (NaN<0 is false), -0.0-preserving.
        UnaryOp::Relu => {
            if x < 0.0 {
                0.0
            } else {
                x
            }
        }
        UnaryOp::Erf => erf(x),
        UnaryOp::Gelu => 0.5 * x * (1.0 + erf(x * FRAC_1_SQRT_2)),
        UnaryOp::Silu => x * (1.0 / (1.0 + (-x).exp())),
        UnaryOp::Sin => x.sin(),
        UnaryOp::Cos => x.cos(),
        UnaryOp::Floor => x.floor(),
        UnaryOp::Ceil => x.ceil(),
        UnaryOp::Round => x.round_ties_even(),
        // Sign: NaN -> 0 (both comparisons false).
        UnaryOp::Sign => {
            if x > 0.0 {
                1.0
            } else if x < 0.0 {
                -1.0
            } else {
                0.0
            }
        }
        // Heaviside step(0)=0.
        UnaryOp::Step => {
            if x > 0.0 {
                1.0
            } else {
                0.0
            }
        }
        UnaryOp::Erfc => erfc(x),
        UnaryOp::Trunc => x.trunc(),
        UnaryOp::Exp2 => x.exp2(),
        UnaryOp::Expm1 => x.exp_m1(),
        UnaryOp::Log2 => x.log2(),
        UnaryOp::Log10 => x.log10(),
        UnaryOp::Log1p => x.ln_1p(),
        UnaryOp::Sinh => x.sinh(),
        UnaryOp::Cosh => x.cosh(),
        UnaryOp::Tan => x.tan(),
        UnaryOp::Asin => x.asin(),
        UnaryOp::Acos => x.acos(),
        UnaryOp::Atan => x.atan(),
        UnaryOp::Asinh => x.asinh(),
        UnaryOp::Acosh => x.acosh(),
        UnaryOp::Atanh => x.atanh(),
        UnaryOp::Cbrt => x.cbrt(),
        UnaryOp::Lgamma => lgamma(x),
    }
}

/// Independent `f64` binary-op evaluator for the NON-int-only ops. Max/Min are
/// NaN-PROPAGATING (distinct from FmaxIeee/FminIeee, NaN-suppressing); Rem is
/// FLOORED (sign-of-divisor), RemTrunc is C `fmod` (sign-of-dividend); the Cmp*
/// family yields exactly `1.0`/`0.0` with C NaN semantics. `Nextafter` is handled
/// dtype-aware by the caller (it steps a storage-dtype lattice) and never routes
/// here.
pub(super) fn binary_op_f64(op: BinaryOp, a: f64, b: f64) -> f64 {
    match op {
        // NaN-propagating, and A ON TIES (`a >= b ? a : b` after the NaN
        // checks) — the KISS-Ops `max_prop` normative decomposition
        // (`select(…, select(cmp_ge(a,b), a, b))`) and the numpy/torch
        // `maximum` = `where(a >= b, a, b)` semantics. Bit-visible ONLY on
        // signed-zero ties: `max_prop(-0.0, +0.0) = -0.0` (keep a), never the
        // b-biased `+0.0`. Caught by the kiss-ref step-2 recipe differential.
        BinaryOp::Max => {
            if a.is_nan() {
                a
            } else if b.is_nan() {
                b
            } else if a >= b {
                a
            } else {
                b
            }
        }
        // Mirror: `min_prop` = `select(…, select(cmp_le(a,b), a, b))` — a on ties.
        BinaryOp::Min => {
            if a.is_nan() {
                a
            } else if b.is_nan() {
                b
            } else if a <= b {
                a
            } else {
                b
            }
        }
        BinaryOp::Pow => a.powf(b),
        // Floored remainder: a - floor(a/b)*b (sign of divisor).
        BinaryOp::Rem => a - (a / b).floor() * b,
        // Truncated remainder: C fmod (Rust `%` on f64 is fmod).
        BinaryOp::RemTrunc => a % b,
        BinaryOp::Atan2 => a.atan2(b),
        BinaryOp::Copysign => a.copysign(b),
        // NaN-SUPPRESSING (std max/min return the non-NaN operand).
        BinaryOp::FmaxIeee => a.max(b),
        BinaryOp::FminIeee => a.min(b),
        BinaryOp::CmpEq => f64::from(a == b),
        BinaryOp::CmpNe => f64::from(a != b),
        BinaryOp::CmpLt => f64::from(a < b),
        BinaryOp::CmpLe => f64::from(a <= b),
        BinaryOp::CmpGt => f64::from(a > b),
        BinaryOp::CmpGe => f64::from(a >= b),
        BinaryOp::Nextafter => panic!("oracle: Nextafter is dtype-aware; handled in eval"),
        BinaryOp::BitAnd
        | BinaryOp::BitOr
        | BinaryOp::BitXor
        | BinaryOp::Shl
        | BinaryOp::Shr
        | BinaryOp::LogicalAnd
        | BinaryOp::LogicalOr
        | BinaryOp::LogicalXor => {
            panic!("oracle: {op:?} is int-only; handled in the integer eval path")
        }
    }
}

/// Integer `x / y` by the total rule (`docs/int-div-shift-semantics.md`, written independently of the
/// kernel interpreter's `int_div`): a zero divisor gives `x`, a signed `MIN / -1` gives `MIN`, anything
/// else truncates toward zero. Reached only for a backend that declares `total_integer_ops`, since the
/// plan gate refuses integer `Div` otherwise.
pub(super) fn int_div_total(dt: ElementKind, x: i128, y: i128) -> i128 {
    let signed_min = crate::types::int_range(dt).is_some_and(|(lo, _)| lo < 0 && x == lo);
    if y == 0 || (signed_min && y == -1) {
        x
    } else {
        x / y
    }
}

/// Independent integer binary-op evaluator (bitwise/shift/logical), wrapping at
/// the emitter's C promotion width. Matches the bespoke functor definitions.
pub(super) fn binary_op_int(op: BinaryOp, a: i128, b: i128, dt: ElementKind) -> i128 {
    let w = op_width(dt);
    match op {
        BinaryOp::BitAnd => wrap_for(a & b, w, dt),
        BinaryOp::BitOr => wrap_for(a | b, w, dt),
        BinaryOp::BitXor => wrap_for(a ^ b, w, dt),
        BinaryOp::Shl => {
            // The count's bits are read as unsigned and taken modulo the promoted width `w`
            // (`docs/int-div-shift-semantics.md` rule 3), as the kernel interpreter does. `w` is a
            // power of two, so the mask is the modulo, and for a negative `b` it reads the
            // two's-complement bits.
            wrap_for(a << ((b & i128::from(w - 1)) as u32), w, dt)
        }
        BinaryOp::Shr => {
            // C's `>>` is ARITHMETIC on a signed operand and LOGICAL on an
            // unsigned one. Both fall out of one expression: `wrap_for` keeps an
            // unsigned dtype non-negative in the wide accumulator, and an `i128`
            // shift of a non-negative value is already logical. The `a < 0` tail
            // is therefore unreachable for an unsigned dtype, which is what makes
            // `u32 >> k` correct rather than sign-propagating.
            // The count is taken modulo the promoted width, as for `Shl`.
            wrap_for(a >> ((b & i128::from(w - 1)) as u32), w, dt)
        }
        BinaryOp::LogicalAnd => i128::from(a != 0 && b != 0),
        BinaryOp::LogicalOr => i128::from(a != 0 || b != 0),
        BinaryOp::LogicalXor => i128::from((a != 0) != (b != 0)),
        other => panic!("oracle: {other:?} is not an integer op"),
    }
}

/// Independent nextafter over the f64 lattice (Rust std has no `nextafter`).
pub(super) fn next_after_f64(x: f64, y: f64) -> f64 {
    if x.is_nan() || y.is_nan() {
        return f64::NAN;
    }
    if x == y {
        return y;
    }
    if x == 0.0 {
        return f64::from_bits(1).copysign(y);
    }
    let bits = x.to_bits();
    let step_up = (y > x) == (x > 0.0);
    f64::from_bits(if step_up { bits + 1 } else { bits - 1 })
}

/// Independent nextafter over the f32 lattice.
pub(super) fn next_after_f32(x: f32, y: f32) -> f32 {
    if x.is_nan() || y.is_nan() {
        return f32::NAN;
    }
    if x == y {
        return y;
    }
    if x == 0.0 {
        return f32::from_bits(1).copysign(y);
    }
    let bits = x.to_bits();
    let step_up = (y > x) == (x > 0.0);
    f32::from_bits(if step_up { bits + 1 } else { bits - 1 })
}

/// Accurate `f64` `erfc` (libm, ~1 ULP). An INDEPENDENT implementation (pure-Rust
/// MUSL/fdlibm heritage) from the emitter's device `erfc`/`erfcf`, so it stays a
/// valid differential reference. (v1 used a compact ~1e-7 Numerical-Recipes
/// rational — too loose to reference an f32/f64 `erfc`/`gelu` kernel, whose device
/// `erf` is itself ~1 ULP; adversarial review flagged it, so upgraded.)
pub(super) fn erfc(x: f64) -> f64 {
    libm::erfc(x)
}

/// Accurate `f64` `erf` (libm, ~1 ULP) — independent of the emitter's device `erf`.
pub(super) fn erf(x: f64) -> f64 {
    libm::erf(x)
}

/// Lanczos `ln|Γ(x)|` (g=7, ~1e-13).
pub(super) fn lgamma(x: f64) -> f64 {
    use std::f64::consts::PI;
    // Reference Lanczos coefficients — kept at full published precision.
    #[allow(
        clippy::excessive_precision,
        reason = "reference Lanczos coefficients kept at full published precision"
    )]
    const G: [f64; 9] = [
        0.999_999_999_999_809_93,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        // reflection: ln|Γ(x)| = ln(π / |sin(π x)|) - ln|Γ(1-x)|
        (PI / (PI * x).sin()).abs().ln() - lgamma(1.0 - x)
    } else {
        let x = x - 1.0;
        let mut a = G[0];
        let t = x + 7.5;
        for (i, &g) in G.iter().enumerate().skip(1) {
            a += g / (x + i as f64);
        }
        0.5 * (2.0 * PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
    }
}
