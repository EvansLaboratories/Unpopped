// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Typed expressions** — the type of every [`ScalarExpr`] node, and the legality of
//! [`ScalarExpr::Cast`] / [`ScalarExpr::Bitcast`].
//!
//! Until increment I1 an op computed in ONE dtype: every node was implicitly the
//! compute dtype `C`. `Cast` and `Bitcast` are the first nodes whose type
//! differs, so the type is now a function: [`infer`]. An expression with neither of
//! those nodes infers to `C` at every node, which is what keeps every pre-existing op
//! byte-identical (pinned by `tests/typed_expressions.rs`).
//!
//! The rules and the reasoning are in `docs/ir-typed-expressions.md`.

use crate::ir::{BinaryOp, CastSpec, Saturation, ScalarExpr};
use unpopped_vocab::ElementKind;

/// Why an expression does not type-check.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum TypeError {
    /// An operator's operands have different types. There is no implicit
    /// promotion: write the [`ScalarExpr::Cast`] you mean.
    Mismatch {
        /// The node kind, e.g. `"Add"`.
        node: &'static str,
        /// The first operand's type.
        left: ElementKind,
        /// The other operand's type.
        right: ElementKind,
    },
    /// A `Cast` this version does not define (the pair or the qualifiers).
    IllegalCast {
        /// Source type.
        from: ElementKind,
        /// The requested conversion.
        spec: CastSpec,
        /// Why it is rejected.
        why: &'static str,
    },
    /// A `Bitcast` between types of different bit widths, or of a type with no
    /// defined bit pattern here.
    IllegalBitcast {
        /// Source type.
        from: ElementKind,
        /// Destination type.
        to: ElementKind,
        /// Why it is rejected.
        why: &'static str,
    },
    /// A `CoordInt` / `ConstInt` leaf of a type or value this version does not
    /// define.
    IllegalIntLeaf {
        /// The leaf's declared type.
        ty: ElementKind,
        /// Why it is rejected.
        why: &'static str,
    },
    /// An operator that has no integer form applied to an integer-typed value
    /// (everything but `Add`/`Sub`/`Mul`, `Select`, `Min`/`Max` and the comparisons).
    IntegerOperand {
        /// The node kind, e.g. `"Div"`.
        node: &'static str,
        /// The integer type of the operand.
        ty: ElementKind,
    },
}

/// Integer types a `CoordInt` / `ConstInt` leaf may have.
#[must_use]
pub const fn is_index_int(t: ElementKind) -> bool {
    matches!(t, ElementKind::I32 | ElementKind::I64 | ElementKind::U32)
}

/// Check a `CoordInt` / `ConstInt` leaf's type, and for a constant its value.
///
/// # Errors
///
/// [`TypeError::IllegalIntLeaf`] naming the reason.
pub fn check_int_leaf(ty: ElementKind, value: Option<i64>) -> Result<(), TypeError> {
    if !is_index_int(ty) {
        return Err(TypeError::IllegalIntLeaf {
            ty,
            why: "an integer leaf is I32, I64 or U32",
        });
    }
    if let (Some(v), Some((lo, hi))) = (value, int_range(ty)) {
        if i128::from(v) < lo || i128::from(v) > hi {
            return Err(TypeError::IllegalIntLeaf {
                ty,
                why: "the constant is outside the type's range",
            });
        }
    }
    Ok(())
}

/// `true` for the binary operations an integer-typed operand pair admits: ordering
/// (`Min`, `Max`, whose NaN rules are moot) and the comparisons.
#[must_use]
pub fn int_binary_ok(op: BinaryOp) -> bool {
    op.is_cmp() || matches!(op, BinaryOp::Max | BinaryOp::Min)
}

/// Floating-point scalar types a `Cast` accepts in this version.
#[must_use]
pub const fn is_cast_float(t: ElementKind) -> bool {
    matches!(
        t,
        ElementKind::F32 | ElementKind::F32Strict | ElementKind::F64
    )
}

/// Integer scalar types a `Cast` accepts in this version.
#[must_use]
pub const fn is_cast_int(t: ElementKind) -> bool {
    matches!(
        t,
        ElementKind::I8
            | ElementKind::I16
            | ElementKind::I32
            | ElementKind::I64
            | ElementKind::U8
            | ElementKind::U16
            | ElementKind::U32
    )
}

/// Bit width of a type the typed nodes accept, or `None`.
#[must_use]
pub const fn typed_bits(t: ElementKind) -> Option<u32> {
    match t {
        ElementKind::I8 | ElementKind::U8 => Some(8),
        ElementKind::I16 | ElementKind::U16 => Some(16),
        ElementKind::I32 | ElementKind::U32 | ElementKind::F32 | ElementKind::F32Strict => Some(32),
        ElementKind::I64 | ElementKind::F64 => Some(64),
        _ => None,
    }
}

/// `(min, max)` of an integer type as exact `i128`s.
#[must_use]
pub const fn int_range(t: ElementKind) -> Option<(i128, i128)> {
    Some(match t {
        ElementKind::I8 => (i8::MIN as i128, i8::MAX as i128),
        ElementKind::I16 => (i16::MIN as i128, i16::MAX as i128),
        ElementKind::I32 => (i32::MIN as i128, i32::MAX as i128),
        ElementKind::I64 => (i64::MIN as i128, i64::MAX as i128),
        ElementKind::U8 => (0, u8::MAX as i128),
        ElementKind::U16 => (0, u16::MAX as i128),
        ElementKind::U32 => (0, u32::MAX as i128),
        _ => return None,
    })
}

/// Check that `from -> spec.to` with `spec.round` / `spec.sat` is a conversion this
/// version defines.
///
/// The v1 table covers `{f32, f32s, f64}` and `{i8, i16, i32, i64, u8, u16, u32, u64}`.
/// Half, fp8 and sub-byte endpoints are not in it yet (their codecs exist in
/// `unpopped-vocab`; wiring them is the next slice) and are rejected here by name,
/// never lowered by guesswork.
///
/// # Errors
///
/// [`TypeError::IllegalCast`] naming the reason.
pub fn check_cast(from: ElementKind, spec: CastSpec) -> Result<(), TypeError> {
    let bad = |why: &'static str| Err(TypeError::IllegalCast { from, spec, why });
    let to = spec.to;
    let known = |t| is_cast_float(t) || is_cast_int(t);
    if !known(from) || !known(to) {
        return bad(
            "a Cast endpoint outside {f32, f32s, f64, i8, i16, i32, i64, u8, u16, u32} is not defined yet",
        );
    }
    match (is_cast_float(from), is_cast_float(to)) {
        // float -> float
        (true, true) => match spec.sat {
            Saturation::Wrap | Saturation::ClampFinite => Ok(()),
            _ => bad("float to float takes Wrap or ClampFinite"),
        },
        // float -> int: saturating, NaN to 0, so no emitter reaches C's undefined behaviour
        (true, false) => match spec.sat {
            Saturation::Clamp => Ok(()),
            _ => bad("float to integer is saturating: use Saturation::Clamp"),
        },
        // int -> float: rounding only; saturation has no meaning
        (false, true) => match spec.sat {
            Saturation::Wrap => Ok(()),
            _ => bad("integer to float takes Saturation::Wrap (no out-of-range case)"),
        },
        // int -> int
        (false, false) => match spec.sat {
            Saturation::Wrap | Saturation::Clamp => Ok(()),
            _ => bad("integer to integer takes Wrap or Clamp"),
        },
    }
}

/// Check that a `Bitcast` `from -> to` is defined: equal bit widths, both in the
/// typed set.
///
/// # Errors
///
/// [`TypeError::IllegalBitcast`] naming the reason.
pub fn check_bitcast(from: ElementKind, to: ElementKind) -> Result<(), TypeError> {
    match (typed_bits(from), typed_bits(to)) {
        (Some(a), Some(b)) if a == b => Ok(()),
        (Some(_), Some(_)) => Err(TypeError::IllegalBitcast {
            from,
            to,
            why: "Bitcast needs equal bit widths",
        }),
        _ => Err(TypeError::IllegalBitcast {
            from,
            to,
            why: "a Bitcast endpoint outside {f32, f32s, f64, i8, i16, i32, i64, u8, u16, u32} is not defined yet",
        }),
    }
}

/// The type of `e` in an op whose compute dtype is `compute`.
///
/// Rules (`docs/ir-typed-expressions.md` section 2): leaves are
/// `compute`; `Cast`/`Bitcast` are their declared type; every operator needs
/// identical operand types and returns that type; `Select` takes any condition type
/// and needs matching arms.
///
/// # Errors
///
/// The first [`TypeError`] found, walking children before parents.
pub fn infer(e: &ScalarExpr, compute: ElementKind) -> Result<ElementKind, TypeError> {
    let same = |node: &'static str, a: &ScalarExpr, b: &ScalarExpr| {
        let (l, r) = (infer(a, compute)?, infer(b, compute)?);
        if l == r {
            Ok(l)
        } else {
            Err(TypeError::Mismatch {
                node,
                left: l,
                right: r,
            })
        }
    };
    // The operators with no integer form refuse an integer-typed operand. Integer-typed
    // values can only differ from the compute dtype here (the typed-expression gate
    // admits them under float compute), so an integer COMPUTE dtype is untouched.
    let no_int = |node: &'static str, t: ElementKind| {
        if t != compute && is_cast_int(t) {
            Err(TypeError::IntegerOperand { node, ty: t })
        } else {
            Ok(t)
        }
    };
    match e {
        ScalarExpr::Input(_)
        | ScalarExpr::Const(_)
        | ScalarExpr::Param(_)
        | ScalarExpr::Reduced(_)
        | ScalarExpr::Coord(_) => Ok(compute),
        ScalarExpr::CoordInt(_, t) => {
            check_int_leaf(*t, None)?;
            Ok(*t)
        }
        ScalarExpr::ConstInt(v, t) => {
            check_int_leaf(*t, Some(*v))?;
            Ok(*t)
        }
        ScalarExpr::Add(a, b) => same("Add", a, b),
        ScalarExpr::Sub(a, b) => same("Sub", a, b),
        ScalarExpr::Mul(a, b) => same("Mul", a, b),
        ScalarExpr::Div(a, b) => no_int("Div", same("Div", a, b)?),
        ScalarExpr::Binary(op, a, b) => {
            let t = same("Binary", a, b)?;
            if t != compute && is_cast_int(t) {
                if !int_binary_ok(*op) {
                    return Err(TypeError::IntegerOperand { node: "Binary", ty: t });
                }
                // A comparison of integers is the compute type's 0/1, exactly as a
                // comparison of floats is.
                return Ok(if op.is_cmp() { compute } else { t });
            }
            Ok(t)
        }
        ScalarExpr::Unary(_, x) => no_int("Unary", infer(x, compute)?),
        ScalarExpr::Fma(a, b, c) => {
            let ab = no_int("Fma", same("Fma", a, b)?)?;
            let t = infer(c, compute)?;
            if ab == t {
                Ok(t)
            } else {
                Err(TypeError::Mismatch {
                    node: "Fma",
                    left: ab,
                    right: t,
                })
            }
        }
        ScalarExpr::Select(c, a, b) => {
            infer(c, compute)?;
            same("Select", a, b)
        }
        ScalarExpr::Cast(x, spec) => {
            let from = infer(x, compute)?;
            check_cast(from, *spec)?;
            Ok(spec.to)
        }
        ScalarExpr::Bitcast(x, to) => {
            let from = infer(x, compute)?;
            check_bitcast(from, *to)?;
            Ok(*to)
        }
        // A qualifier changes how the wrapped operation rounds, not what type it has.
        ScalarExpr::Qualified(x, _) => infer(x, compute),
    }
}

/// The integer type of `e` when it is an integer-typed value, found by following the
/// first operand down the expression (O(depth), no full inference): a `CoordInt` /
/// `ConstInt` leaf, a `Cast` / `Bitcast` to an integer, or an `Add` / `Sub` / `Mul` /
/// `Min` / `Max` / `Select` whose first value operand is one. `None` for anything else.
///
/// Meant for hot evaluation loops that must route the integer cases without paying
/// [`infer`] per node per element. It is exact on a well-typed expression (the plan
/// gate has checked that); on an ill-typed one the answer is unspecified, and
/// [`infer`] is the authority.
#[must_use]
pub fn int_spine_type(e: &ScalarExpr) -> Option<ElementKind> {
    match e {
        ScalarExpr::CoordInt(_, t) | ScalarExpr::ConstInt(_, t) => Some(*t),
        ScalarExpr::Cast(_, spec) => is_cast_int(spec.to).then_some(spec.to),
        ScalarExpr::Bitcast(_, to) => is_cast_int(*to).then_some(*to),
        ScalarExpr::Add(a, _) | ScalarExpr::Sub(a, _) | ScalarExpr::Mul(a, _) => int_spine_type(a),
        ScalarExpr::Binary(op, a, _) if matches!(op, BinaryOp::Max | BinaryOp::Min) => {
            int_spine_type(a)
        }
        ScalarExpr::Select(_, a, _) => int_spine_type(a),
        _ => None,
    }
}

/// `true` when `e` contains a [`ScalarExpr::Qualified`] node. The optimizer returns
/// such an expression unchanged: a rewrite across the node would change which
/// operations round, flush or approximate, and how.
#[must_use]
pub fn has_qualified(e: &ScalarExpr) -> bool {
    match e {
        ScalarExpr::Qualified(..) => true,
        ScalarExpr::Input(_)
        | ScalarExpr::Const(_)
        | ScalarExpr::Param(_)
        | ScalarExpr::Reduced(_)
        | ScalarExpr::Coord(_) 
        | ScalarExpr::CoordInt(..)
        | ScalarExpr::ConstInt(..) => false,
        ScalarExpr::Unary(_, x) | ScalarExpr::Cast(x, _) | ScalarExpr::Bitcast(x, _) => {
            has_qualified(x)
        }
        ScalarExpr::Add(a, b)
        | ScalarExpr::Sub(a, b)
        | ScalarExpr::Mul(a, b)
        | ScalarExpr::Div(a, b)
        | ScalarExpr::Binary(_, a, b) => has_qualified(a) || has_qualified(b),
        ScalarExpr::Select(a, b, c) | ScalarExpr::Fma(a, b, c) => {
            has_qualified(a) || has_qualified(b) || has_qualified(c)
        }
    }
}

/// `true` when `e` contains any node whose type can differ from the compute dtype
/// (`Cast`, `Bitcast`, `CoordInt`, `ConstInt`). The back-compat fast path: an expression for which
/// this is `false` types to `compute` everywhere and needs no further type work.
#[must_use]
pub fn is_typed(e: &ScalarExpr) -> bool {
    match e {
        ScalarExpr::Cast(..) | ScalarExpr::Bitcast(..) => true,
        ScalarExpr::Input(_)
        | ScalarExpr::Const(_)
        | ScalarExpr::Param(_)
        | ScalarExpr::Reduced(_)
        | ScalarExpr::Coord(_) => false,
        // Integer-typed leaves are the typed expressions' second kind: their type is not
        // the compute dtype.
        ScalarExpr::CoordInt(..) | ScalarExpr::ConstInt(..) => true,
        ScalarExpr::Unary(_, x) | ScalarExpr::Qualified(x, _) => is_typed(x),
        ScalarExpr::Add(a, b)
        | ScalarExpr::Sub(a, b)
        | ScalarExpr::Mul(a, b)
        | ScalarExpr::Div(a, b)
        | ScalarExpr::Binary(_, a, b) => is_typed(a) || is_typed(b),
        ScalarExpr::Select(a, b, c) | ScalarExpr::Fma(a, b, c) => {
            is_typed(a) || is_typed(b) || is_typed(c)
        }
    }
}

/// Whether `round` changes the result of converting `from -> to` at all. A widening
/// float conversion and an integer that fits the destination's significand are exact,
/// so every mode agrees; emitters that spell only round-to-nearest-even use this to
/// know when they must decline a directed mode.
#[must_use]
pub fn rounding_matters(from: ElementKind, to: ElementKind) -> bool {
    let significand = |t: ElementKind| match t {
        ElementKind::F64 => 53,
        _ => 24,
    };
    match (is_cast_float(from), is_cast_float(to)) {
        (true, true) => typed_bits(from) > typed_bits(to),
        (true, false) => true,
        (false, true) => typed_bits(from).is_some_and(|b| b > significand(to)),
        (false, false) => false,
    }
}
