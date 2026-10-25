// SPDX-License-Identifier: MIT OR Apache-2.0
//! The cpu_c spellings of [`ScalarExpr::Cast`](unpopped::ir::ScalarExpr::Cast) and
//! [`ScalarExpr::Bitcast`](unpopped::ir::ScalarExpr::Bitcast).
//!
//! Spelled only where plain C99 can honour the requested semantics exactly. A C cast
//! has the host's default rounding (to nearest even) and, for float to integer, is
//! undefined outside the destination range; so:
//!
//! - **rounding**: a conversion that cannot be exact is spelled only for
//!   `RoundMode::NearestEven`. A directed mode declines (`fesetround` is global
//!   state a kernel must not touch), except float to integer, where `trunc`,
//!   `floor`, `ceil` and `nearbyint` ARE the four modes.
//! - **saturation**: float to integer and integer `Clamp` go through the helpers in
//!   [`TYPED_HELPERS`], which clamp explicitly (NaN to 0), never by C's undefined
//!   behaviour.
//! - **bitcast**: a C99 compound-literal union, which is a defined reinterpretation
//!   (a pointer cast would break strict aliasing).
//!
//! The semantics being matched are the oracle's `cast_ref`, which is computed from
//! exact integers and IEEE bit patterns; `tests/cpu_end_to_end.rs` compiles and runs
//! these spellings against it.

use unpopped::backend::{Decline, DeclinedOp, LowerError, Spelling};
use unpopped::ir::{CastSpec, RoundMode, Saturation};
use unpopped::types::{int_range, is_cast_float, rounding_matters};
use unpopped_vocab::ElementKind;

/// C helpers a typed kernel carries (emitted only when the body has a `Cast` or
/// `Bitcast`, so every untyped kernel is byte-identical to before).
pub(crate) const TYPED_HELPERS: &str = "\
static long long unp_sat_f2ll(double y, long long lo, long long hi) {
    if (y != y) return 0;
    if (y >= 9223372036854775808.0) return hi;
    if (y < -9223372036854775808.0) return lo;
    {
        long long v = (long long)y;
        return v > hi ? hi : (v < lo ? lo : v);
    }
}
static long long unp_clamp_ll(long long v, long long lo, long long hi) {
    return v > hi ? hi : (v < lo ? lo : v);
}
";

/// The C type of a value in the typed table.
pub(crate) fn c_type(t: ElementKind) -> Option<&'static str> {
    Some(match t {
        ElementKind::F32 | ElementKind::F32Strict => "float",
        ElementKind::F64 => "double",
        ElementKind::I8 => "signed char",
        ElementKind::I16 => "short",
        ElementKind::I32 => "int",
        ElementKind::I64 => "long long",
        ElementKind::U8 => "unsigned char",
        ElementKind::U16 => "unsigned short",
        ElementKind::U32 => "unsigned int",
        _ => return None,
    })
}

fn decline(op: DeclinedOp, why: String) -> Result<Spelling, LowerError> {
    Ok(Spelling::Declined(Decline::UnsupportedOp { op, why }))
}

/// A signed 64-bit literal that is also valid for `i64::MIN` in C.
fn ll(v: i128) -> String {
    if v == i128::from(i64::MIN) {
        "(-9223372036854775807LL - 1)".to_string()
    } else {
        format!("{v}LL")
    }
}

/// Spell `Cast(x)` from `from` per `spec`, or decline.
pub(crate) fn cpu_cast(
    x: String,
    from: ElementKind,
    spec: CastSpec,
) -> Result<Spelling, LowerError> {
    let to = spec.to;
    let (Some(_), Some(to_c)) = (c_type(from), c_type(to)) else {
        return decline(
            DeclinedOp::Cast,
            format!("cpu_c has no C type for the Cast {from:?} -> {to:?}"),
        );
    };
    if rounding_matters(from, to)
        && !(is_cast_float(from) && !is_cast_float(to))
        && spec.round != RoundMode::NearestEven
    {
        return decline(
            DeclinedOp::Cast,
            format!(
                "cpu_c spells an inexact {from:?} -> {to:?} conversion only for RoundMode::\
                 NearestEven (the host default; a directed mode would need fesetround, which is \
                 global state), got {:?}",
                spec.round
            ),
        );
    }
    let spelled = match (is_cast_float(from), is_cast_float(to)) {
        // float -> float
        (true, true) => {
            if spec.sat == Saturation::ClampFinite && from == ElementKind::F64 && to != from {
                // Clamp BEFORE narrowing so an overflowing or infinite input lands on the
                // largest finite f32; NaN passes through. f32::MAX as a double literal.
                let m = "3.4028234663852886e+38";
                format!(
                    "(({to_c})(({x}) != ({x}) ? ({x}) : (({x}) > {m} ? {m} : \
                     (({x}) < -{m} ? -{m} : ({x})))))"
                )
            } else {
                format!("(({to_c})({x}))")
            }
        }
        // float -> integer: round per mode, then saturate with NaN to 0
        (true, false) => {
            let (lo, hi) = int_range(to).expect("table checked");
            let rnd = match spec.round {
                RoundMode::TowardZero => "trunc",
                RoundMode::NearestEven => "nearbyint",
                RoundMode::TowardNegInf => "floor",
                RoundMode::TowardPosInf => "ceil",
                other => {
                    return decline(
                        DeclinedOp::Cast,
                        format!("cpu_c has no spelling for the rounding mode {other:?}"),
                    );
                }
            };
            format!(
                "(({to_c})unp_sat_f2ll({rnd}((double)({x})), {}, {}))",
                ll(lo),
                ll(hi)
            )
        }
        // integer -> float: exact or round-to-nearest-even (checked above)
        (false, true) => format!("(({to_c})({x}))"),
        // integer -> integer
        (false, false) => match spec.sat {
            Saturation::Clamp => {
                let (lo, hi) = int_range(to).expect("table checked");
                format!(
                    "(({to_c})unp_clamp_ll((long long)({x}), {}, {}))",
                    ll(lo),
                    ll(hi)
                )
            }
            // Two's-complement truncation: every C compiler targeted here wraps an
            // out-of-range signed conversion (C23 makes it the rule).
            _ => format!("(({to_c})({x}))"),
        },
    };
    Ok(Spelling::Spelled(spelled))
}

/// Spell `Bitcast(x)` from `from` to `to`, or decline.
pub(crate) fn cpu_bitcast(
    x: String,
    from: ElementKind,
    to: ElementKind,
) -> Result<Spelling, LowerError> {
    let (Some(from_c), Some(to_c)) = (c_type(from), c_type(to)) else {
        return decline(
            DeclinedOp::Bitcast,
            format!("cpu_c has no C type for the Bitcast {from:?} -> {to:?}"),
        );
    };
    Ok(Spelling::Spelled(format!(
        "((union {{ {from_c} a; {to_c} b; }}){{ ({x}) }}).b"
    )))
}
