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

use crate::ir::{BinaryOp, CastSpec, RoundMode, Saturation, ScalarExpr};
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
    /// An integer bit operation (`Popc`, `Clz`, `Brev`) applied to a value that is not an integer of a kernel body (a float,
    /// or any operand in a plan body).
    IntegerOnlyOperand {
        /// The node kind, e.g. `"Unary"`.
        node: &'static str,
        /// The type of the operand.
        ty: ElementKind,
    },
    /// A `Pack` or `BitcastPacked` with a lane count other than 2, 4, 8 or 16, or an 8- or 16-lane one wider than 128 bits.
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
    /// A [`ScalarExpr::Local`] was typed with no kernel to say what the local is: a local has
    /// meaning only inside a `kernel::KernelFn` (use [`infer_ty_in`]).
    LocalOutsideKernel,
    /// A [`ScalarExpr::Qualified`] whose rounding mode its operation does not take, or whose operands are not f32/f32s.
    IllegalQualifier {
        /// Why it is rejected.
        why: &'static str,
    },
}

/// The type of a [`ScalarExpr`] node: a scalar, or a packed value of 2, 4, 8 or 16 lanes
/// (`docs/ir-typed-expressions.md` section 9).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Ty {
    /// One value of this element type.
    Scalar(ElementKind),
    /// `lanes` values of this element type, `lanes` in `{2, 4, 8, 16}` (8 and 16 within 128 bits).
    Packed(ElementKind, u8),
}

/// Integer types a `CoordInt` / `ConstInt` leaf may have.
///
/// `U64` is a leaf type, but a `ConstInt` literal's payload is an `i64`, so a `U64` literal
/// is limited to `0..=i64::MAX` (checked by [`check_int_leaf`]). Larger `U64` values are
/// reached with `Cast`, arithmetic or `CoordInt`; widening the payload is a separate change
/// together with the `Lowering::const_int` seam.
#[must_use]
pub const fn is_index_int(t: ElementKind) -> bool {
    matches!(
        t,
        ElementKind::I32 | ElementKind::I64 | ElementKind::U32 | ElementKind::U64
    )
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
            why: "an integer leaf is I32, I64, U32 or U64",
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

/// The integer-only operators the **kernel** layer admits at any integer type, not just the compute type: the shifts and the
/// bitwise operators, which the kernel oracle evaluates at the operand's own type (`docs/int-div-shift-semantics.md`). The
/// logical operators are not in the list: the oracle has no meaning for them.
#[must_use]
pub const fn kernel_int_op(op: BinaryOp) -> bool {
    matches!(
        op,
        BinaryOp::Shl | BinaryOp::Shr | BinaryOp::BitAnd | BinaryOp::BitOr | BinaryOp::BitXor
    )
}

/// The binary operators defined on floating-point values only: no integer type takes them, as an operand or as the compute
/// type.
#[must_use]
pub fn float_only_binary(op: BinaryOp) -> bool {
    matches!(
        op,
        BinaryOp::Pow
            | BinaryOp::Rem
            | BinaryOp::RemTrunc
            | BinaryOp::Atan2
            | BinaryOp::Copysign
            | BinaryOp::Nextafter
            | BinaryOp::FmaxIeee
            | BinaryOp::FminIeee
    )
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
    is_cast_float(t)
        || matches!(
            t,
            ElementKind::F16 | ElementKind::Bf16 | ElementKind::F8E3M4 | ElementKind::F8UE4M3
        )
        || is_mx_element(t)
}

/// A dtype of union batch 1 (KISS 0.24.0, §6.1-0017 to -0020; board 501 and 515): `F128`, `I128`, `U128`, `I256`, `U256`,
/// `F8E3M4`, `F8UE4M3`. **Storage and key identity only** until the casts (plan E1b), the wide oracle integer (E2) and a
/// software binary128 (E3) land: `check_kernel` refuses a kernel that declares one (`KernelError::DtypeNotComputed`), so the
/// oracle never meets a value it cannot hold.
#[must_use]
pub const fn is_union_batch1(t: ElementKind) -> bool {
    matches!(
        t,
        ElementKind::F128
            | ElementKind::I128
            | ElementKind::U128
            | ElementKind::I256
            | ElementKind::U256
            | ElementKind::F8E3M4
            | ElementKind::F8UE4M3
    )
}

/// A wide integer dtype (`I128`, `U128`, `I256`, `U256`, ROADMAP E2): a **kernel-layer** integer, computed by the oracle on a 256-bit
/// pattern (`wide_int::W256`) because [`int_range`] holds `i128` pairs and cannot describe them. A plan never types one.
#[must_use]
pub const fn is_wide_int(t: ElementKind) -> bool {
    crate::wide_int::wide_int_bits(t).is_some()
}

/// An integer a **kernel** may compute on: [`is_cast_int`] or [`is_wide_int`].
#[must_use]
pub const fn is_kernel_int(t: ElementKind) -> bool {
    is_cast_int(t) || is_wide_int(t)
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
        // KISS 0.24.0 §6.1-0019: IEEE-style FP8 E3M4 (sign 1, exponent 3, mantissa 4, bias 3, infinity and NaN).
        ElementKind::F8E3M4 => Some((3, 4)),
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
            | ElementKind::U64
    )
}

/// A sub-byte integer (`I4`, `U4`): a `Cast` endpoint (ROADMAP U9b S1, `docs/options-u9b-subbyte-dequant.md`),
/// against an integer endpoint and with `Saturation::Wrap` only. It is not an arithmetic type: `int_range` does not
/// cover it, and a kernel reads one from a `Read` buffer and widens it.
#[must_use]
pub const fn is_subbyte_int(t: ElementKind) -> bool {
    matches!(t, ElementKind::I4 | ElementKind::U4)
}

/// The value range of a sub-byte integer: `I4` is `-8..=7`, `U4` is `0..=15`.
#[must_use]
pub const fn subbyte_range(t: ElementKind) -> Option<(i128, i128)> {
    match t {
        ElementKind::I4 => Some((-8, 7)),
        ElementKind::U4 => Some((0, 15)),
        _ => None,
    }
}

/// Bit width of a type the typed nodes accept, or `None`.
#[must_use]
pub const fn typed_bits(t: ElementKind) -> Option<u32> {
    match t {
        ElementKind::I8 | ElementKind::U8 => Some(8),
        ElementKind::I16 | ElementKind::U16 | ElementKind::F16 | ElementKind::Bf16 => Some(16),
        ElementKind::I32 | ElementKind::U32 | ElementKind::F32 | ElementKind::F32Strict => Some(32),
        ElementKind::I64 | ElementKind::U64 | ElementKind::F64 => Some(64),
        _ => None,
    }
}

/// The size in bytes of a value a `Place::Raw` access may name (ROADMAP U9c): the types with a [`typed_bits`] width, all of a
/// whole number of bytes. Sub-byte, FP8 and other untyped kinds are `None` (a raw access to them would pick a layout).
#[must_use]
pub const fn raw_access_bytes(t: ElementKind) -> Option<u32> {
    match typed_bits(t) {
        Some(bits) => Some(bits / 8),
        None => None,
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
        ElementKind::U64 => (0, u64::MAX as i128),
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
    // A wide integer converts to and from the integers only, by wrap or clamp: a float or sub-byte endpoint has no
    // reference conversion yet (E2), so it is refused by name.
    if is_wide_int(from) || is_wide_int(to) {
        if !(is_kernel_int(from) && is_kernel_int(to)) {
            return bad(
                "a wide integer (i128, u128, i256, u256) casts to and from the integer types only: no float or sub-byte conversion is defined yet",
            );
        }
        return match spec.sat {
            Saturation::Wrap | Saturation::Clamp => Ok(()),
            _ => bad("integer to integer takes Wrap or Clamp"),
        };
    }
    if is_subbyte_int(from) || is_subbyte_int(to) {
        // the other endpoint must be an integer (widen to i32 before a float), the only conversion is a wrap
        let other = if is_subbyte_int(from) { to } else { from };
        if !is_cast_int(other) && !is_subbyte_int(other) {
            return bad("a sub-byte Cast takes an integer on the other side: widen to i32 first");
        }
        return match spec.sat {
            Saturation::Wrap => Ok(()),
            _ => bad(
                "a sub-byte Cast takes Saturation::Wrap: clamp with Min and Max before narrowing",
            ),
        };
    }
    if !known(from) || !known(to) {
        return bad(
            "a Cast endpoint outside {f4e2m1, f6e2m3, f6e3m2, f8e3m4, f8ue4m3, f16, bf16, f32, f32s, f64, i8, i16, i32, i64, u8, u16, u32, u64} is not defined yet",
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
    // `F8UE4M3` (KISS 0.24.0 §6.1-0020, the unsigned NVFP4 scale) has no infinity either: it saturates at 448, NaN is its one NaN
    // code, and a value below zero lands on 0 (the IR's rule, UNVERIFIED against a vendor conversion), so the only legal
    // saturation is the explicit ClampFinite.
    if to == ElementKind::F8UE4M3 {
        return match spec.sat {
            Saturation::ClampFinite => Ok(()),
            _ => bad("a Cast into f8ue4m3 saturates: use Saturation::ClampFinite"),
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
            why: "a Bitcast endpoint outside {f32, f32s, f64, i8, i16, i32, i64, u8, u16, u32, u64} is not defined yet",
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
    infer_ty_mode(e, compute, &[], TyMode::Plan)
}

/// Which layer is typing an expression. The **plan** layer keeps the rule that an integer operand other than the compute type may
/// only be compared, ordered (`Min`/`Max`) or added, subtracted and multiplied; the **kernel** layer (ROADMAP U19, pm-kernels
/// 2026-11-22, option A) also admits integer `Div` and the bitwise and shift operators (`Shl`, `Shr`, `BitAnd`, `BitOr`,
/// `BitXor`) at any integer type, with the semantics of `docs/int-div-shift-semantics.md` that the kernel oracle already
/// evaluates. The plan gate is untouched: it opts integer operators in per backend (`Backend::total_integer_ops`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TyMode {
    Plan,
    Kernel,
}

/// [`infer_ty`] inside a kernel: `locals[i]` is the declared type of `Local(i)`. A `Local`
/// whose id is outside `locals` (so every `Local` under [`infer_ty`]) is
/// [`TypeError::LocalOutsideKernel`]. Scope (declared before use) is the kernel checker's
/// business, not this function's.
///
/// # Errors
///
/// The first [`TypeError`] found, walking children before parents.
pub fn infer_ty_in(e: &ScalarExpr, compute: ElementKind, locals: &[Ty]) -> Result<Ty, TypeError> {
    infer_ty_mode(e, compute, locals, TyMode::Kernel)
}

fn infer_ty_mode(
    e: &ScalarExpr,
    compute: ElementKind,
    locals: &[Ty],
    mode: TyMode,
) -> Result<Ty, TypeError> {
    match e {
        ScalarExpr::Local(i) => locals
            .get(*i as usize)
            .copied()
            .ok_or(TypeError::LocalOutsideKernel),
        ScalarExpr::Pack(xs) => pack_ty(xs, compute, locals, mode),
        ScalarExpr::BitcastPacked(x, elem, lanes) => {
            bitcast_packed_ty(x, *elem, *lanes, compute, locals, mode)
        }
        ScalarExpr::Extract(x, i) => match infer_ty_mode(x, compute, locals, mode)? {
            Ty::Packed(t, lanes) if *i < lanes => Ok(Ty::Scalar(t)),
            Ty::Packed(_, lanes) => Err(TypeError::BadLane { index: *i, lanes }),
            Ty::Scalar(_) => Err(TypeError::NotPacked { node: "Extract" }),
        },
        _ => operator_ty(e, compute, locals, mode),
    }
}

/// ROADMAP U15: 8 and 16 lanes are packed values only while they fit one 128-bit register (`uint4`-style): 8 lanes of
/// at most 16 bits, 16 lanes of 8 bits. 2 and 4 lanes keep the earlier rule (any typed lane). An element with no bit
/// width (a lane type outside the typed set) is left to the existing operand rules.
fn check_packed_width(elem: ElementKind, lanes: u8) -> Result<(), TypeError> {
    match typed_bits(elem) {
        Some(bits) if lanes > 4 && bits * u32::from(lanes) > 128 => {
            Err(TypeError::BadLanes { lanes })
        }
        _ => Ok(()),
    }
}

/// The type of a `Pack`: 2, 4, 8 or 16 scalar children of one type (8 and 16 only within 128 bits).
fn pack_ty(
    xs: &[ScalarExpr],
    compute: ElementKind,
    locals: &[Ty],
    mode: TyMode,
) -> Result<Ty, TypeError> {
    let lanes = u8::try_from(xs.len()).unwrap_or(u8::MAX);
    if !matches!(lanes, 2 | 4 | 8 | 16) {
        return Err(TypeError::BadLanes { lanes });
    }
    let first = child("Pack", &xs[0], compute, locals, mode)?;
    for x in &xs[1..] {
        let t = child("Pack", x, compute, locals, mode)?;
        if t != first {
            return Err(TypeError::Mismatch {
                node: "Pack",
                left: first,
                right: t,
            });
        }
    }
    check_packed_width(first, lanes)?;
    Ok(Ty::Packed(first, lanes))
}

/// The type of a `BitcastPacked`: a scalar operand reinterpreted as `lanes` lanes of `elem`.
fn bitcast_packed_ty(
    x: &ScalarExpr,
    elem: ElementKind,
    lanes: u8,
    compute: ElementKind,
    locals: &[Ty],
    mode: TyMode,
) -> Result<Ty, TypeError> {
    if !matches!(lanes, 2 | 4 | 8 | 16) {
        return Err(TypeError::BadLanes { lanes });
    }
    check_packed_width(elem, lanes)?;
    let from = child("BitcastPacked", x, compute, locals, mode)?;
    check_packed_bitcast(elem, lanes, from)?;
    Ok(Ty::Packed(elem, lanes))
}

/// The type of every other node: scalar rules when no operand is packed, the lane-wise rules
/// otherwise.
fn operator_ty(
    e: &ScalarExpr,
    compute: ElementKind,
    locals: &[Ty],
    mode: TyMode,
) -> Result<Ty, TypeError> {
    let tys = operands(e)
        .into_iter()
        .map(|x| infer_ty_mode(x, compute, locals, mode))
        .collect::<Result<Vec<_>, _>>()?;
    let elems: Vec<ElementKind> = tys.iter().map(|t| elem_of(*t)).collect();
    let lanes = tys.iter().find_map(|t| match t {
        Ty::Packed(_, n) => Some(*n),
        Ty::Scalar(_) => None,
    });
    let Some(lanes) = lanes else {
        return node_ty(e, compute, &elems, mode).map(Ty::Scalar);
    };
    let node = node_name(e);
    // A `Bitcast` of a packed value to a scalar of the same total bits; the inverse
    // (scalar to packed) is `BitcastPacked`.
    if let ScalarExpr::Bitcast(_, to) = e {
        return check_packed_bitcast(elems[0], lanes, *to).map(|()| Ty::Scalar(*to));
    }
    if !lane_wise(e) {
        return Err(TypeError::PackedOperand { node });
    }
    // A `Select` takes a scalar condition (whole-value select) or a packed one with the
    // arms' lane count (lane-wise mask); every other operand is packed.
    let is_select = matches!(e, ScalarExpr::Select(..));
    let lane_mask = is_select && matches!(tys[0], Ty::Packed(_, n) if n == lanes);
    if is_select && !lane_mask && !matches!(tys[0], Ty::Scalar(_)) {
        return Err(TypeError::PackedOperand { node });
    }
    let first_value = usize::from(is_select && !lane_mask);
    if tys[first_value..]
        .iter()
        .any(|t| !matches!(t, Ty::Packed(_, n) if *n == lanes))
    {
        return Err(TypeError::PackedOperand { node });
    }
    Ok(Ty::Packed(node_ty(e, compute, &elems, mode)?, lanes))
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
            why: "a Bitcast endpoint outside {f32, f32s, f64, i8, i16, i32, i64, u8, u16, u32, u64} is not defined yet",
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
        // A scalar condition selects the whole value, so it is the same in every lane; a packed
        // one (lane-wise mask) contributes its own lane `i`.
        ScalarExpr::Select(c, a, b) => {
            let cond = match lane_of(c, i) {
                Ok(lane) => lane,
                // `PackedOperand`: a scalar condition built from a packed value (a `Bitcast` of one)
                // has no lane form and is not one; `infer_ty` has already vetted the whole node.
                Err(TypeError::NotPacked { .. } | TypeError::PackedOperand { .. }) => (**c).clone(),
                Err(err) => return Err(err),
            };
            ScalarExpr::Select(Box::new(cond), one(a)?, one(b)?)
        }
        ScalarExpr::Cast(x, spec) => ScalarExpr::Cast(one(x)?, *spec),
        ScalarExpr::Input(_)
        | ScalarExpr::Const(_)
        | ScalarExpr::Param(_)
        | ScalarExpr::Reduced(_)
        | ScalarExpr::Coord(_)
        | ScalarExpr::CoordInt(..)
        | ScalarExpr::ConstInt(..)
        | ScalarExpr::Local(_)
        | ScalarExpr::Exec(_)
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
    locals: &[Ty],
    mode: TyMode,
) -> Result<ElementKind, TypeError> {
    match infer_ty_mode(x, compute, locals, mode)? {
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
pub(crate) fn operands(e: &ScalarExpr) -> Vec<&ScalarExpr> {
    match e {
        ScalarExpr::Input(_)
        | ScalarExpr::Const(_)
        | ScalarExpr::Param(_)
        | ScalarExpr::Reduced(_)
        | ScalarExpr::Coord(_)
        | ScalarExpr::CoordInt(..)
        | ScalarExpr::ConstInt(..)
        | ScalarExpr::Local(_)
        | ScalarExpr::Exec(_) => vec![],
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
pub(crate) const fn node_name(e: &ScalarExpr) -> &'static str {
    match e {
        ScalarExpr::Input(_) => "Input",
        ScalarExpr::Const(_) => "Const",
        ScalarExpr::Param(_) => "Param",
        ScalarExpr::Reduced(_) => "Reduced",
        ScalarExpr::Coord(_) => "Coord",
        ScalarExpr::CoordInt(..) => "CoordInt",
        ScalarExpr::ConstInt(..) => "ConstInt",
        ScalarExpr::Local(_) => "Local",
        ScalarExpr::Exec(_) => "Exec",
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
    mode: TyMode,
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
        if t != compute && is_kernel_int(t) {
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
        // Every execution id is a `U32` (decision D-R4-2); whether the kernel has a launch for
        // it to read is `check_kernel`'s business, as scope is for `Local`.
        ScalarExpr::Exec(_) => Ok(ElementKind::U32),
        ScalarExpr::Add(..) => same("Add", k[0], k[1]),
        ScalarExpr::Sub(..) => same("Sub", k[0], k[1]),
        ScalarExpr::Mul(..) => same("Mul", k[0], k[1]),
        ScalarExpr::Div(..) => {
            let t = same("Div", k[0], k[1])?;
            // the kernel layer admits an integer `Div` at any integer type (total, `docs/int-div-shift-semantics.md`)
            if mode == TyMode::Kernel && is_kernel_int(t) {
                return Ok(t);
            }
            no_int("Div", t)
        }
        ScalarExpr::Binary(op, ..) => {
            let t = same("Binary", k[0], k[1])?;
            // The float-only operators have no integer meaning anywhere (the oracle defines none, and a C emitter would
            // spell `fmod` and `pow` on integers), whether or not the integer type is the compute type. An integer
            // remainder is `Select(b == 0, 0, a - (a / b) * b)` over the total `Div` (`docs/int-div-shift-semantics.md`).
            if is_kernel_int(t) && float_only_binary(*op) {
                return Err(TypeError::IntegerOperand {
                    node: "Binary",
                    ty: t,
                });
            }
            if t != compute && is_kernel_int(t) {
                if !(int_binary_ok(*op) || (mode == TyMode::Kernel && kernel_int_op(*op))) {
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
        ScalarExpr::Unary(op, _) if op.is_bit_op() => {
            // an integer-only operation: defined at the operand's own width, kernel layer only
            if mode == TyMode::Kernel && int_range(k[0]).is_some() {
                Ok(k[0])
            } else {
                Err(TypeError::IntegerOnlyOperand {
                    node: "Unary",
                    ty: k[0],
                })
            }
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
            if mode == TyMode::Plan && (is_wide_int(k[0]) || is_wide_int(spec.to)) {
                return Err(TypeError::IllegalCast {
                    from: k[0],
                    spec: *spec,
                    why: "a wide integer (i128, u128, i256, u256) is a kernel-layer type: a plan does not cast to or from one",
                });
            }
            Ok(spec.to)
        }
        ScalarExpr::Bitcast(_, to) => {
            check_bitcast(k[0], *to)?;
            Ok(*to)
        }
        // A qualifier changes how the wrapped operation rounds, not what type it has. Arithmetic
        // takes the four IEEE directions only: ties-away is a conversion's mode (PTX `.rna`).
        ScalarExpr::Qualified(_, n) => {
            if n.round == RoundMode::NearestAway {
                return Err(TypeError::IllegalQualifier {
                    why: "an arithmetic operation takes no ties-away rounding (RoundMode::NearestAway is a Cast's)",
                });
            }
            // The plan gate (`plan/admissibility.rs`) and cpu_c already say f32/f32s only; the oracle asserts it
            // (`oracle/expr_eval.rs`). A kernel never went through the gate, so without this a `KernelFn` with a qualified f64 or
            // half operation passed `check_kernel` and panicked in the oracle. F64 has no ftz form and no exact reference for a
            // directed rounding (ROADMAP U43).
            if !matches!(k[0], ElementKind::F32 | ElementKind::F32Strict) {
                return Err(TypeError::IllegalQualifier {
                    why: "a rounding/ftz/approx qualifier needs f32 or f32s operands in v1 (f64 has no ftz form and no reference for a directed rounding yet; half types compute through f32)",
                });
            }
            Ok(k[0])
        }
        // `infer_ty` answers these itself; reaching here is an internal misuse, reported
        // rather than panicked on.
        ScalarExpr::Pack(_)
        | ScalarExpr::Extract(..)
        | ScalarExpr::BitcastPacked(..)
        | ScalarExpr::Local(_) => Err(TypeError::NotPacked { node: node_name(e) }),
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
        | ScalarExpr::ConstInt(..)
        | ScalarExpr::Local(_)
        | ScalarExpr::Exec(_) => false,
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

/// `true` when `e` contains a [`ScalarExpr::Local`] or [`ScalarExpr::Exec`] leaf (both meaningful
/// only inside a kernel).
#[must_use]
pub fn has_local(e: &ScalarExpr) -> bool {
    matches!(e, ScalarExpr::Local(_) | ScalarExpr::Exec(_))
        || operands(e).into_iter().any(has_local)
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
        | ScalarExpr::ConstInt(..)
        | ScalarExpr::Local(_)
        | ScalarExpr::Exec(_) => false,
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
    is_typed_beyond(e, None)
}

/// [`is_typed`], except that a `ConstInt` of exactly the `compute` dtype is not typed: at an integer
/// compute dtype that leaf IS compute-typed. `None` is plain [`is_typed`]. Used only by the plan
/// gate for a backend that declares `Backend::total_integer_ops`.
#[must_use]
pub(crate) fn is_typed_beyond(e: &ScalarExpr, compute: Option<ElementKind>) -> bool {
    let is_typed = |x: &ScalarExpr| is_typed_beyond(x, compute);
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
        ScalarExpr::ConstInt(_, t) => compute != Some(*t),
        ScalarExpr::CoordInt(..) => true,
        // A local's type comes from its kernel's declaration, never the compute dtype.
        ScalarExpr::Local(_) => true,
        // An execution id is `U32`, never the compute dtype.
        ScalarExpr::Exec(_) => true,
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

/// Rebuild `e` with `f` applied to each child.
pub(crate) fn map_children<E>(
    e: &ScalarExpr,
    f: &mut dyn FnMut(&ScalarExpr) -> Result<ScalarExpr, E>,
) -> Result<ScalarExpr, E> {
    use ScalarExpr as S;
    let bx = |r: Result<ScalarExpr, E>| r.map(Box::new);
    Ok(match e {
        S::Input(_)
        | S::Const(_)
        | S::Param(_)
        | S::Reduced(_)
        | S::Coord(_)
        | S::CoordInt(..)
        | S::ConstInt(..)
        | S::Local(_)
        | S::Exec(_) => e.clone(),
        S::Add(a, b) => S::Add(bx(f(a))?, bx(f(b))?),
        S::Sub(a, b) => S::Sub(bx(f(a))?, bx(f(b))?),
        S::Mul(a, b) => S::Mul(bx(f(a))?, bx(f(b))?),
        S::Div(a, b) => S::Div(bx(f(a))?, bx(f(b))?),
        S::Binary(op, a, b) => S::Binary(*op, bx(f(a))?, bx(f(b))?),
        S::Unary(op, x) => S::Unary(*op, bx(f(x))?),
        S::Select(c, a, b) => S::Select(bx(f(c))?, bx(f(a))?, bx(f(b))?),
        S::Fma(a, b, c) => S::Fma(bx(f(a))?, bx(f(b))?, bx(f(c))?),
        S::Cast(x, s) => S::Cast(bx(f(x))?, *s),
        S::Bitcast(x, t) => S::Bitcast(bx(f(x))?, *t),
        S::Qualified(x, n) => S::Qualified(bx(f(x))?, *n),
        S::Pack(xs) => S::Pack(xs.iter().map(&mut *f).collect::<Result<_, E>>()?),
        S::Extract(x, i) => S::Extract(bx(f(x))?, *i),
        S::BitcastPacked(x, t, n) => S::BitcastPacked(bx(f(x))?, *t, *n),
    })
}

/// Storage width in **bits**, which is the only width that describes every §6.1
/// dtype.
///
/// `elem_size` is bytes and cannot express `i4`/`u4` (4 bits) or `b1` (1 bit).
/// Those are not stored one-per-byte: KISS-CLASSIFY §6.1 pins their packing
/// normatively, and packing is part of what a dtype *is* there (§6.0-0001 makes
/// the packing convention normative alongside the token and the width).
pub(crate) fn elem_bits(dt: ElementKind) -> usize {
    match dt {
        ElementKind::I4 | ElementKind::U4 | ElementKind::F4E2M1 => 4,
        ElementKind::B1 => 1,
        other => elem_size(other) * 8,
    }
}

/// Element byte size for the v1-supported dtypes (the CONTAINER width for a sub-byte dtype).
pub(crate) fn elem_size(dt: ElementKind) -> usize {
    match dt {
        ElementKind::F16 | ElementKind::Bf16 | ElementKind::I16 | ElementKind::U16 => 2,
        ElementKind::F32 | ElementKind::F32Strict | ElementKind::I32 | ElementKind::U32 => 4,
        ElementKind::F64 | ElementKind::I64 | ElementKind::U64 => 8,
        ElementKind::I8
        | ElementKind::U8
        | ElementKind::Bool
        | ElementKind::Fp8E4M3FN
        | ElementKind::Fp8E5M2 => 1,
        // A complex element is a PAIR: c64 is two f32, c128 is two f64. The
        // §6.1 token names the total width, so the name already says this.
        ElementKind::Complex64 => 8,
        ElementKind::Complex128 => 16,
        // Union batch 1 (KISS 0.24.0): storage widths only (§6.1-0017 to -0020); no operation computes them yet.
        ElementKind::F128 | ElementKind::I128 | ElementKind::U128 => 16,
        ElementKind::I256 | ElementKind::U256 => 32,
        ElementKind::F8E3M4 | ElementKind::F8UE4M3 | ElementKind::F8E8M0 => 1,
        // Sub-byte dtypes share a byte. `elem_size` is the CONTAINER width here,
        // not the element width — use `elem_bits` for the latter and
        // `packed_bytes` for allocation.
        ElementKind::I4 | ElementKind::U4 | ElementKind::F4E2M1 | ElementKind::B1 => 1,
        other => panic!("oracle: unsupported dtype {other:?} (v1)"),
    }
}
