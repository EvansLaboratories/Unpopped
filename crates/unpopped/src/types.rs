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
    /// A `Pack` with a lane count other than 2 or 4.
    BadLanes {
        /// The number of children given (saturated at 255).
        lanes: u8,
    },
    /// An `Extract` index outside the packed value.
    BadLane {
        /// The index asked for.
        index: u8,
        /// The packed value's lane count.
        lanes: u8,
    },
    /// An `Extract` of a value that is not packed.
    NotPacked {
        /// The node kind, `"Extract"`.
        node: &'static str,
    },
    /// An operator with no packed form applied to a packed value (or a `Pack` lane
    /// that is itself packed).
    PackedOperand {
        /// The node kind, e.g. `"Add"`.
        node: &'static str,
    },
    /// [`infer`] was asked for the type of a packed value, which a bare
    /// [`ElementKind`] cannot describe: use [`infer_ty`].
    PackedResult {
        /// The packed value's lane count.
        lanes: u8,
    },
}

/// The type of a [`ScalarExpr`] node: a scalar, or a packed value of 2 or 4 lanes
/// (`docs/ir-typed-expressions.md` section 9).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Ty {
    /// One value of this element type.
    Scalar(ElementKind),
    /// `lanes` values of this element type, `lanes` in `{2, 4}`.
    Packed(ElementKind, u8),
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

/// Floating-point types a `Cast` converts from or to: the compute floats of
/// [`is_cast_float`] plus the 16-bit `F16` and `Bf16`. A narrow float is a conversion
/// ENDPOINT only: it is never a compute dtype (the plan gate keys on [`is_cast_float`]).
#[must_use]
pub const fn is_cast_float_endpoint(t: ElementKind) -> bool {
    is_cast_float(t) || matches!(t, ElementKind::F16 | ElementKind::Bf16) || is_mx_element(t)
}

/// An OCP MX element format (`F4E2M1`, `F6E2M3`, `F6E3M2`): IEEE-shaped fields with NO
/// infinity and NO NaN, so the all-ones exponent is an ordinary exponent and the largest
/// code is the largest finite value. A `Cast` into one saturates (`docs/ir-typed-expressions.md`
/// section 10). Like the 16-bit floats it is an ENDPOINT only, never a compute dtype.
#[must_use]
pub const fn is_mx_element(t: ElementKind) -> bool {
    matches!(
        t,
        ElementKind::F4E2M1 | ElementKind::F6E2M3 | ElementKind::F6E3M2
    )
}

/// `(exponent bits, mantissa bits)` of an IEEE-shaped float a `Cast` accepts, or `None`.
#[must_use]
pub const fn float_format(t: ElementKind) -> Option<(u32, u32)> {
    match t {
        ElementKind::F16 => Some((5, 10)),
        ElementKind::Bf16 => Some((8, 7)),
        ElementKind::F4E2M1 => Some((2, 1)),
        ElementKind::F6E2M3 => Some((2, 3)),
        ElementKind::F6E3M2 => Some((3, 2)),
        ElementKind::F32 | ElementKind::F32Strict => Some((8, 23)),
        ElementKind::F64 => Some((11, 52)),
        _ => None,
    }
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
        ElementKind::I16 | ElementKind::U16 | ElementKind::F16 | ElementKind::Bf16 => Some(16),
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
/// The table covers `{f4e2m1, f6e2m3, f6e3m2, f16, bf16, f32, f32s, f64}` and `{i8, i16, i32, i64, u8, u16, u32}`.
/// Fp8 and sub-byte endpoints are not in it yet (their codecs exist in
/// `unpopped-vocab`; wiring them is the next slice) and are rejected here by name,
/// never lowered by guesswork.
///
/// # Errors
///
/// [`TypeError::IllegalCast`] naming the reason.
pub fn check_cast(from: ElementKind, spec: CastSpec) -> Result<(), TypeError> {
    let bad = |why: &'static str| Err(TypeError::IllegalCast { from, spec, why });
    let to = spec.to;
    let known = |t| is_cast_float_endpoint(t) || is_cast_int(t);
    if !known(from) || !known(to) {
        return bad(
            "a Cast endpoint outside {f4e2m1, f6e2m3, f6e3m2, f16, bf16, f32, f32s, f64, i8, i16, i32, i64, u8, u16, u32} is not defined yet",
        );
    }
    // An MX element has no infinity or NaN to overflow into: a conversion into one
    // saturates (largest finite value; NaN to +0), from a float or an integer alike, so
    // the only legal saturation is the explicit ClampFinite.
    if is_mx_element(to) {
        return match spec.sat {
            Saturation::ClampFinite => Ok(()),
            _ => bad("a Cast into an MX element type saturates: use Saturation::ClampFinite"),
        };
    }
    match (is_cast_float_endpoint(from), is_cast_float_endpoint(to)) {
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
    match infer_ty(e, compute)? {
        Ty::Scalar(t) => Ok(t),
        Ty::Packed(_, lanes) => Err(TypeError::PackedResult { lanes }),
    }
}

/// The [`Ty`] of `e`: [`infer`] extended to packed values (`docs/ir-typed-expressions.md`
/// section 9). Where `e` has no packed node it is `Ty::Scalar(infer(e))`.
///
/// `Pack` makes a packed value and `Extract` takes a lane. `Add`, `Sub`, `Mul`, `Fma`,
/// `Min`/`Max`, a whole-value `Select` (scalar condition, packed arms) and `Cast` apply
/// lane-wise: every packed operand has the same lane count, the lane types obey the scalar
/// rules, and there is no broadcast of a scalar beside a packed value. Every other
/// operator refuses a packed operand with [`TypeError::PackedOperand`].
///
/// # Errors
///
/// The first [`TypeError`] found, walking children before parents.
pub fn infer_ty(e: &ScalarExpr, compute: ElementKind) -> Result<Ty, TypeError> {
    match e {
        ScalarExpr::Pack(xs) => {
            let lanes = u8::try_from(xs.len()).unwrap_or(u8::MAX);
            if !matches!(lanes, 2 | 4) {
                return Err(TypeError::BadLanes { lanes });
            }
            let first = child("Pack", &xs[0], compute)?;
            for x in &xs[1..] {
                let t = child("Pack", x, compute)?;
                if t != first {
                    return Err(TypeError::Mismatch {
                        node: "Pack",
                        left: first,
                        right: t,
                    });
                }
            }
            Ok(Ty::Packed(first, lanes))
        }
        ScalarExpr::BitcastPacked(x, elem, lanes) => {
            if !matches!(*lanes, 2 | 4) {
                return Err(TypeError::BadLanes { lanes: *lanes });
            }
            let from = child("BitcastPacked", x, compute)?;
            check_packed_bitcast(*elem, *lanes, from)?;
            Ok(Ty::Packed(*elem, *lanes))
        }
        ScalarExpr::Extract(x, i) => match infer_ty(x, compute)? {
            Ty::Packed(t, lanes) if *i < lanes => Ok(Ty::Scalar(t)),
            Ty::Packed(_, lanes) => Err(TypeError::BadLane { index: *i, lanes }),
            Ty::Scalar(_) => Err(TypeError::NotPacked { node: "Extract" }),
        },
        _ => {
            let tys = operands(e)
                .into_iter()
                .map(|x| infer_ty(x, compute))
                .collect::<Result<Vec<_>, _>>()?;
            let elems: Vec<ElementKind> = tys.iter().map(|t| elem_of(*t)).collect();
            let lanes = tys.iter().find_map(|t| match t {
                Ty::Packed(_, n) => Some(*n),
                Ty::Scalar(_) => None,
            });
            let Some(lanes) = lanes else {
                return node_ty(e, compute, &elems).map(Ty::Scalar);
            };
            let node = node_name(e);
            // A `Bitcast` of a packed value to a scalar of the same total bits; the inverse
            // (scalar to packed) needs a packed target, which `Bitcast` cannot name yet.
            if let ScalarExpr::Bitcast(_, to) = e {
                return check_packed_bitcast(elems[0], lanes, *to).map(|()| Ty::Scalar(*to));
            }
            if !lane_wise(e) {
                return Err(TypeError::PackedOperand { node });
            }
            // A whole-value `Select` takes a scalar condition; every other operand is packed.
            let first_value = usize::from(matches!(e, ScalarExpr::Select(..)));
            if first_value == 1 && !matches!(tys[0], Ty::Scalar(_)) {
                return Err(TypeError::PackedOperand { node });
            }
            if tys[first_value..]
                .iter()
                .any(|t| !matches!(t, Ty::Packed(_, n) if *n == lanes))
            {
                return Err(TypeError::PackedOperand { node });
            }
            Ok(Ty::Packed(node_ty(e, compute, &elems)?, lanes))
        }
    }
}

/// Check that a `Bitcast` of `Packed(elem, lanes)` to the scalar `to` is defined: lane bits
/// times lane count equal `to`'s width. Lane 0 is the least significant lane.
///
/// # Errors
///
/// [`TypeError::IllegalBitcast`] naming the reason.
pub fn check_packed_bitcast(
    elem: ElementKind,
    lanes: u8,
    to: ElementKind,
) -> Result<(), TypeError> {
    match (typed_bits(elem), typed_bits(to)) {
        (Some(a), Some(b)) if a * u32::from(lanes) == b => Ok(()),
        (Some(_), Some(_)) => Err(TypeError::IllegalBitcast {
            from: elem,
            to,
            why: "a packed Bitcast needs lane bits times lane count equal to the target width",
        }),
        _ => Err(TypeError::IllegalBitcast {
            from: elem,
            to,
            why: "a Bitcast endpoint outside {f32, f32s, f64, i8, i16, i32, i64, u8, u16, u32} is not defined yet",
        }),
    }
}

/// Lane `i` of the packed value `e`, as a scalar expression: `Pack` picks its child and a
/// lane-wise operator applies to the same lane of its operands. This is the reference
/// meaning of `Extract` (the oracle evaluates it) and the fallback spelling an emitter
/// without a native packed form uses: lane-by-lane scalar code.
///
/// # Errors
///
/// [`TypeError::BadLane`] for an index outside a `Pack`, [`TypeError::NotPacked`] for an
/// operand that is not packed, [`TypeError::PackedOperand`] for an operator with no packed
/// form.
pub fn lane_of(e: &ScalarExpr, i: u8) -> Result<ScalarExpr, TypeError> {
    let one = |x: &ScalarExpr| lane_of(x, i).map(Box::new);
    Ok(match e {
        ScalarExpr::Pack(xs) => {
            let lanes = u8::try_from(xs.len()).unwrap_or(u8::MAX);
            return xs
                .get(usize::from(i))
                .cloned()
                .ok_or(TypeError::BadLane { index: i, lanes });
        }
        ScalarExpr::Add(a, b) => ScalarExpr::Add(one(a)?, one(b)?),
        ScalarExpr::Sub(a, b) => ScalarExpr::Sub(one(a)?, one(b)?),
        ScalarExpr::Mul(a, b) => ScalarExpr::Mul(one(a)?, one(b)?),
        ScalarExpr::Binary(op @ (BinaryOp::Min | BinaryOp::Max), a, b) => {
            ScalarExpr::Binary(*op, one(a)?, one(b)?)
        }
        ScalarExpr::Fma(a, b, c) => ScalarExpr::Fma(one(a)?, one(b)?, one(c)?),
        ScalarExpr::Select(c, a, b) => ScalarExpr::Select(c.clone(), one(a)?, one(b)?),
        ScalarExpr::Cast(x, spec) => ScalarExpr::Cast(one(x)?, *spec),
        ScalarExpr::Input(_)
        | ScalarExpr::Const(_)
        | ScalarExpr::Param(_)
        | ScalarExpr::Reduced(_)
        | ScalarExpr::Coord(_)
        | ScalarExpr::CoordInt(..)
        | ScalarExpr::ConstInt(..)
        | ScalarExpr::Extract(..)
        | ScalarExpr::BitcastPacked(..) => return Err(TypeError::NotPacked { node: node_name(e) }),
        _ => return Err(TypeError::PackedOperand { node: node_name(e) }),
    })
}

/// The scalar type of `x` as an operand of `node`; a packed operand is declined (a `Pack`
/// lane is never itself packed).
fn child(
    node: &'static str,
    x: &ScalarExpr,
    compute: ElementKind,
) -> Result<ElementKind, TypeError> {
    match infer_ty(x, compute)? {
        Ty::Scalar(t) => Ok(t),
        Ty::Packed(..) => Err(TypeError::PackedOperand { node }),
    }
}

const fn elem_of(t: Ty) -> ElementKind {
    match t {
        Ty::Scalar(e) | Ty::Packed(e, _) => e,
    }
}

/// The operators that apply lane-wise to packed operands.
const fn lane_wise(e: &ScalarExpr) -> bool {
    matches!(
        e,
        ScalarExpr::Add(..)
            | ScalarExpr::Sub(..)
            | ScalarExpr::Mul(..)
            | ScalarExpr::Fma(..)
            | ScalarExpr::Select(..)
            | ScalarExpr::Cast(..)
            | ScalarExpr::Binary(BinaryOp::Min | BinaryOp::Max, ..)
    )
}

/// The child expressions of `e`, in the order [`node_ty`] reads their types.
fn operands(e: &ScalarExpr) -> Vec<&ScalarExpr> {
    match e {
        ScalarExpr::Input(_)
        | ScalarExpr::Const(_)
        | ScalarExpr::Param(_)
        | ScalarExpr::Reduced(_)
        | ScalarExpr::Coord(_)
        | ScalarExpr::CoordInt(..)
        | ScalarExpr::ConstInt(..) => vec![],
        ScalarExpr::Add(a, b)
        | ScalarExpr::Sub(a, b)
        | ScalarExpr::Mul(a, b)
        | ScalarExpr::Div(a, b)
        | ScalarExpr::Binary(_, a, b) => vec![a, b],
        ScalarExpr::Unary(_, x)
        | ScalarExpr::Cast(x, _)
        | ScalarExpr::Bitcast(x, _)
        | ScalarExpr::Qualified(x, _)
        | ScalarExpr::Extract(x, _)
        | ScalarExpr::BitcastPacked(x, ..) => vec![x],
        ScalarExpr::Fma(a, b, c) | ScalarExpr::Select(a, b, c) => vec![a, b, c],
        ScalarExpr::Pack(xs) => xs.iter().collect(),
    }
}

/// The node kind's name, as `TypeError`s report it.
const fn node_name(e: &ScalarExpr) -> &'static str {
    match e {
        ScalarExpr::Input(_) => "Input",
        ScalarExpr::Const(_) => "Const",
        ScalarExpr::Param(_) => "Param",
        ScalarExpr::Reduced(_) => "Reduced",
        ScalarExpr::Coord(_) => "Coord",
        ScalarExpr::CoordInt(..) => "CoordInt",
        ScalarExpr::ConstInt(..) => "ConstInt",
        ScalarExpr::Add(..) => "Add",
        ScalarExpr::Sub(..) => "Sub",
        ScalarExpr::Mul(..) => "Mul",
        ScalarExpr::Div(..) => "Div",
        ScalarExpr::Unary(..) => "Unary",
        ScalarExpr::Binary(..) => "Binary",
        ScalarExpr::Fma(..) => "Fma",
        ScalarExpr::Select(..) => "Select",
        ScalarExpr::Cast(..) => "Cast",
        ScalarExpr::Bitcast(..) => "Bitcast",
        ScalarExpr::Qualified(..) => "Qualified",
        ScalarExpr::Pack(_) => "Pack",
        ScalarExpr::Extract(..) => "Extract",
        ScalarExpr::BitcastPacked(..) => "BitcastPacked",
    }
}

/// The scalar typing rules for one node given the element types of its operands (`k`, in
/// [`operands`] order). The single rule table: scalar expressions use it directly and
/// lane-wise packed operators use it on the lane types.
fn node_ty(
    e: &ScalarExpr,
    compute: ElementKind,
    k: &[ElementKind],
) -> Result<ElementKind, TypeError> {
    let same = |node: &'static str, l: ElementKind, r: ElementKind| {
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
        ScalarExpr::Add(..) => same("Add", k[0], k[1]),
        ScalarExpr::Sub(..) => same("Sub", k[0], k[1]),
        ScalarExpr::Mul(..) => same("Mul", k[0], k[1]),
        ScalarExpr::Div(..) => no_int("Div", same("Div", k[0], k[1])?),
        ScalarExpr::Binary(op, ..) => {
            let t = same("Binary", k[0], k[1])?;
            if t != compute && is_cast_int(t) {
                if !int_binary_ok(*op) {
                    return Err(TypeError::IntegerOperand {
                        node: "Binary",
                        ty: t,
                    });
                }
                // A comparison of integers is the compute type's 0/1, exactly as a
                // comparison of floats is.
                return Ok(if op.is_cmp() { compute } else { t });
            }
            Ok(t)
        }
        ScalarExpr::Unary(..) => no_int("Unary", k[0]),
        ScalarExpr::Fma(..) => {
            let ab = no_int("Fma", same("Fma", k[0], k[1])?)?;
            same("Fma", ab, k[2])
        }
        // The condition may have any type (tested `!= 0`); the arms must match.
        ScalarExpr::Select(..) => same("Select", k[1], k[2]),
        ScalarExpr::Cast(_, spec) => {
            check_cast(k[0], *spec)?;
            Ok(spec.to)
        }
        ScalarExpr::Bitcast(_, to) => {
            check_bitcast(k[0], *to)?;
            Ok(*to)
        }
        // A qualifier changes how the wrapped operation rounds, not what type it has.
        ScalarExpr::Qualified(..) => Ok(k[0]),
        // `infer_ty` answers these itself; reaching here is an internal misuse, reported
        // rather than panicked on.
        ScalarExpr::Pack(_) | ScalarExpr::Extract(..) | ScalarExpr::BitcastPacked(..) => {
            Err(TypeError::NotPacked { node: node_name(e) })
        }
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
        ScalarExpr::Binary(BinaryOp::Max | BinaryOp::Min, a, _) => int_spine_type(a),
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
        ScalarExpr::Unary(_, x)
        | ScalarExpr::Cast(x, _)
        | ScalarExpr::Bitcast(x, _)
        | ScalarExpr::Extract(x, _)
        | ScalarExpr::BitcastPacked(x, ..) => has_qualified(x),
        ScalarExpr::Pack(xs) => xs.iter().any(has_qualified),
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

/// `true` when `e` contains a [`ScalarExpr::Pack`] or [`ScalarExpr::Extract`] node.
#[must_use]
pub fn has_packed(e: &ScalarExpr) -> bool {
    match e {
        ScalarExpr::Pack(_) | ScalarExpr::Extract(..) | ScalarExpr::BitcastPacked(..) => true,
        ScalarExpr::Input(_)
        | ScalarExpr::Const(_)
        | ScalarExpr::Param(_)
        | ScalarExpr::Reduced(_)
        | ScalarExpr::Coord(_)
        | ScalarExpr::CoordInt(..)
        | ScalarExpr::ConstInt(..) => false,
        ScalarExpr::Unary(_, x) | ScalarExpr::Cast(x, _) | ScalarExpr::Bitcast(x, _) => {
            has_packed(x)
        }
        ScalarExpr::Qualified(x, _) => has_packed(x),
        ScalarExpr::Add(a, b)
        | ScalarExpr::Sub(a, b)
        | ScalarExpr::Mul(a, b)
        | ScalarExpr::Div(a, b)
        | ScalarExpr::Binary(_, a, b) => has_packed(a) || has_packed(b),
        ScalarExpr::Select(a, b, c) | ScalarExpr::Fma(a, b, c) => {
            has_packed(a) || has_packed(b) || has_packed(c)
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
        // A packed value is never the compute dtype; `Extract` is typed because its operand is.
        ScalarExpr::Pack(_) | ScalarExpr::Extract(..) | ScalarExpr::BitcastPacked(..) => true,
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
    let significand = |t: ElementKind| float_format(t).map_or(0, |(_, m)| m + 1);
    match (is_cast_float_endpoint(from), is_cast_float_endpoint(to)) {
        // Exact only when the destination has at least the source's exponent range AND
        // significand: equal widths do not say so (f16 -> bf16 loses precision, bf16 ->
        // f16 loses range).
        (true, true) => match (float_format(from), float_format(to)) {
            (Some((fe, fm)), Some((te, tm))) => !(te >= fe && tm >= fm),
            _ => true,
        },
        (true, false) => true,
        (false, true) => typed_bits(from).is_some_and(|b| b > significand(to)),
        (false, false) => false,
    }
}
