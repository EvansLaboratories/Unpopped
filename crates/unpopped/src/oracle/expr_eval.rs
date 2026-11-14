// SPDX-License-Identifier: MIT OR Apache-2.0
//! The independent `ScalarExpr` evaluator and the store that encodes a value into the output dtype.
//!
//! A section of the oracle (ROADMAP U30): `oracle.rs` glob-imports it, so every path it had is unchanged.

use super::*;

// ===========================================================================
// The independent ScalarExpr evaluator.
// ===========================================================================

/// Leaf/reduced/coord accessors for one evaluation context — the oracle's
/// independent analog of the emitter's `Lowering`, producing VALUES not strings
/// and sharing no code with `lower_expr`.
pub(super) struct Eval<'a> {
    /// Compute dtype (`plan.dtype`) — selects float vs. integer arithmetic.
    pub(super) dtype: ElementKind,
    /// Runtime scalar params.
    pub(super) params: &'a [f64],
    /// `Input(i)` loader.
    pub(super) leaf: &'a dyn Fn(u8) -> Val,
    /// `Reduced(i)` scalar accessor.
    pub(super) reduced: &'a dyn Fn(u8) -> Val,
    /// `Coord(d)` accessor (Elementwise only).
    pub(super) coord: &'a dyn Fn(u8) -> Val,
}

/// `Reduced` leaf reached outside a reduction context.
pub(super) fn panic_reduced(s: u8) -> Val {
    panic!("oracle: Reduced({s}) leaf outside a reduction/scan/window context")
}

/// `Coord` leaf reached outside an Elementwise body.
pub(super) fn panic_coord(d: u8) -> Val {
    panic!("oracle: Coord({d}) is Elementwise-only")
}

/// Evaluate a `ScalarExpr` TREE (hash-consing / `ExprDag` is an emission concern,
/// ignored). Every node is computed from its DEFINITION.
pub(super) fn eval(e: &ScalarExpr, ev: &Eval<'_>) -> Val {
    // Integer-typed arithmetic (`docs/ir-typed-expressions.md` section 8): routed by the
    // first operand's integer type, so a float body pays one `match` per node.
    if !is_int(ev.dtype) && !is_complex(ev.dtype) {
        if let Some(t) = int_op_ty(e) {
            return eval_int_op(e, t, ev);
        }
    }
    match e {
        ScalarExpr::CoordInt(d, t) => int_val((ev.coord)(*d).i128(), *t),
        ScalarExpr::ConstInt(v, t) => int_val(i128::from(*v), *t),
        ScalarExpr::Local(_) => panic!(
            "oracle: a kernel-layer Local outside a kernel (the plan gate rejects an OpDef body containing one)"
        ),
        ScalarExpr::Exec(_) => panic!(
            "oracle: a kernel-layer Exec outside a kernel (the plan gate rejects an OpDef body containing one)"
        ),
        ScalarExpr::Input(i) => (ev.leaf)(*i),
        ScalarExpr::Const(v) => Val::Float(*v),
        ScalarExpr::Param(i) => Val::Float(ev.params[*i as usize]),
        ScalarExpr::Reduced(i) => (ev.reduced)(*i),
        ScalarExpr::Coord(d) => (ev.coord)(*d),
        ScalarExpr::Add(a, b) => arith(
            ev,
            a,
            b,
            |x, y| x + y,
            i128::wrapping_add,
            |(ar, ai), (br, bi)| (ar + br, ai + bi),
        ),
        ScalarExpr::Sub(a, b) => arith(
            ev,
            a,
            b,
            |x, y| x - y,
            i128::wrapping_sub,
            |(ar, ai), (br, bi)| (ar - br, ai - bi),
        ),
        // The one that is NOT component-wise.
        ScalarExpr::Mul(a, b) => arith(
            ev,
            a,
            b,
            |x, y| x * y,
            i128::wrapping_mul,
            |(ar, ai), (br, bi)| (ar * br - ai * bi, ar * bi + ai * br),
        ),
        // Div goes through `arith` like the other three. Reading `.f64()`
        // directly — as this did — PANICS on a complex operand, because
        // `Val::f64()` is deliberately fatal there. That was safe only for as
        // long as the plan gate refused complex `Div`, i.e. a correctness
        // property held at a distance by an unrelated file.
        ScalarExpr::Div(a, b) => arith(
            ev,
            a,
            b,
            |x, y| x / y,
            |x, y| int_div_total(ev.dtype, x, y),
            // Smith's algorithm — scale by the larger denominator component so
            // no intermediate squares. The emitted C runs the same recurrence;
            // the textbook form would agree to rounding on well-conditioned
            // inputs and disagree wildly where either overflows, so matching the
            // algorithm is what makes the differential test meaningful rather
            // than a tolerance negotiation.
            |(ar, ai), (br, bi)| {
                if br.abs() >= bi.abs() {
                    let (q, den) = (bi / br, br + bi * (bi / br));
                    ((ar + ai * q) / den, (ai - ar * q) / den)
                } else {
                    let (q, den) = (br / bi, br * (br / bi) + bi);
                    ((ar * q + ai) / den, (ai * q - ar) / den)
                }
            },
        ),
        ScalarExpr::Unary(op, x) => {
            // A sign edit on a narrow float is a MASK, not an f64 detour: it
            // computes nothing, so the operand's payload must survive it.
            if matches!(op, UnaryOp::Neg | UnaryOp::Abs) {
                if let Val::Raw(b, dt) = eval(x, ev) {
                    if let Some((sign, mag)) = raw_sign_masks(dt) {
                        return Val::Raw(
                            match op {
                                UnaryOp::Neg => b ^ sign,
                                _ => b & mag,
                            },
                            dt,
                        );
                    }
                }
            }
            let xv = eval(x, ev).f64();
            // Sign/Step are DISCONTINUOUS (a full ±1/0 flip): the emitter decides
            // on the compute-dtype float register (`x > 0.0f`), so round the
            // argument to the compute dtype first. Continuous unary ops
            // (exp/log/tanh/…) stay accurate-f64 + tolerance.
            let xv = if matches!(op, UnaryOp::Sign | UnaryOp::Step) {
                round_to_compute(xv, ev.dtype)
            } else {
                xv
            };
            Val::Float(unary_op_f64(*op, xv))
        }
        ScalarExpr::Binary(op, a, b) => eval_binary(*op, a, b, ev),
        // Fused multiply-add: ONE rounding of the exact `a*b + c`. `f64::mul_add`
        // is the correctly-rounded fused operation, and the operands here are at
        // most f32-wide (or f64, where it is the definition itself), so the f64
        // result is the exact value rounded once at 53 bits, finer than the
        // compute dtype's own single rounding the device performs. Narrower than a
        // two-step `x * y + z`, which would round the product on its own.
        ScalarExpr::Fma(a, b, c) => Val::Float(
            eval(a, ev)
                .f64()
                .mul_add(eval(b, ev).f64(), eval(c, ev).f64()),
        ),
        // Conversions are computed by `cast_ref`, from exact integers and IEEE bit
        // patterns, never from an emitter's spelling. The operand's type is inferred
        // (the plan gate has already rejected an ill-typed expression), and its value
        // is first rounded to that type: a conversion is a DISCONTINUOUS decision
        // (float to int, saturation), so deciding on the oracle's unrounded f64 would
        // flip where the device, holding a rounded f32, does not.
        ScalarExpr::Cast(x, spec) => {
            let from = crate::types::infer(x, ev.dtype)
                .expect("oracle: ill-typed Cast operand (the plan gate rejects this)");
            let n = val_to_num(eval(x, ev), from);
            Val::Raw(crate::cast_ref::cast_bits(n, from, *spec), spec.to)
        }
        // A qualified operation: operands rounded to the compute dtype (the device
        // holds f32 values), then `numerics::eval_qualified_f32` applies the
        // operand flush, the operation in its stated rounding mode, and the result
        // flush. f32/f32s compute only; the plan gate rejects the rest.
        ScalarExpr::Qualified(x, n) => {
            assert!(
                matches!(ev.dtype, ElementKind::F32 | ElementKind::F32Strict),
                "oracle: Qualified at {:?} (the plan gate rejects this)",
                ev.dtype
            );
            let (op, kids) = crate::ir::QualOp::of(x).expect(
                "oracle: Qualified over a non-qualifiable node (the plan gate rejects this)",
            );
            let args: Vec<f32> = kids
                .iter()
                .map(|k| round_to_compute(eval(k, ev).f64(), ev.dtype) as f32)
                .collect();
            Val::Float(f64::from(crate::numerics::eval_qualified_f32(
                op, *n, &args,
            )))
        }
        // A lane of a packed value is its scalar meaning (`types::lane_of`): `Pack` picks a
        // child and a lane-wise operator applies to the same lane of its operands. A bare
        // `Pack` or an ill-typed `Extract` is rejected by the plan gate before any evaluation.
        ScalarExpr::Extract(x, i) => packed_lane(x, *i, ev),
        ScalarExpr::Pack(_) | ScalarExpr::BitcastPacked(..) => {
            panic!("oracle: a packed value has no scalar result")
        }
        // A packed operand is its lanes' bits, lane 0 least significant.
        ScalarExpr::Bitcast(x, to)
            if matches!(
                crate::types::infer_ty(x, ev.dtype),
                Ok(crate::types::Ty::Packed(..))
            ) =>
        {
            let Ok(crate::types::Ty::Packed(elem, lanes)) = crate::types::infer_ty(x, ev.dtype)
            else {
                unreachable!("matched as packed above")
            };
            let width = crate::types::typed_bits(elem)
                .expect("oracle: untyped packed lane (the plan gate rejects this)");
            let mut bits = 0u128;
            for i in 0..lanes {
                let b = match packed_lane(x, i, ev) {
                    Val::Raw(b, dt) if dt == elem => b,
                    other => crate::cast_ref::bits_of(val_to_num(other, elem), elem),
                };
                bits |= b << (u32::from(i) * width);
            }
            Val::Raw(bits, *to)
        }
        ScalarExpr::Bitcast(x, to) => {
            let from = crate::types::infer(x, ev.dtype)
                .expect("oracle: ill-typed Bitcast operand (the plan gate rejects this)");
            let bits = match eval(x, ev) {
                // A value already carrying its own bits moves verbatim (NaN payloads,
                // the sign of zero).
                Val::Raw(b, dt) if dt == from => b,
                other => crate::cast_ref::bits_of(val_to_num(other, from), from),
            };
            Val::Raw(bits, *to)
        }
        ScalarExpr::Select(c, a, b) => {
            // cond tested in the compute dtype: -0.0 is false, NaN is true. Round
            // the condition to the compute dtype first — the emitter tests
            // `((float)(c)) != 0.0f`, so a cond sub-expression that underflows to
            // ±0 there must pick the same arm here. The chosen arm MOVES verbatim
            // (Raw stays Raw so -0.0 / NaN payloads / signaling survive; no
            // arithmetic ever touches an arm).
            if round_to_compute(eval(c, ev).f64(), ev.dtype) != 0.0 {
                eval(a, ev)
            } else {
                eval(b, ev)
            }
        }
    }
}

/// Lane `i` of the packed expression `x`. A `BitcastPacked` is sliced out of its operand's
/// bits (lane 0 least significant); every other packed form is its scalar meaning
/// ([`crate::types::lane_of`]).
pub(super) fn packed_lane(x: &ScalarExpr, i: u8, ev: &Eval<'_>) -> Val {
    if let ScalarExpr::BitcastPacked(y, elem, lanes) = x {
        assert!(
            i < *lanes,
            "oracle: Extract lane out of range (the plan gate rejects this)"
        );
        let from = crate::types::infer(y, ev.dtype)
            .expect("oracle: ill-typed BitcastPacked operand (the plan gate rejects this)");
        let bits = match eval(y, ev) {
            Val::Raw(b, dt) if dt == from => b,
            other => crate::cast_ref::bits_of(val_to_num(other, from), from),
        };
        let width = crate::types::typed_bits(*elem)
            .expect("oracle: untyped packed lane (the plan gate rejects this)");
        return Val::Raw(
            (bits >> (u32::from(i) * width)) & ((1u128 << width) - 1),
            *elem,
        );
    }
    match crate::types::lane_of(x, i) {
        Ok(lane) => eval(&lane, ev),
        Err(err) => panic!("oracle: ill-typed Extract: {err:?}"),
    }
}

/// An integer value of type `t`: `v` wrapped to the type's width, carried as its bits.
pub(super) fn int_val(v: i128, t: ElementKind) -> Val {
    Val::Raw(crate::cast_ref::bits_of(crate::cast_ref::Num::I(v), t), t)
}

/// The integer type `e` computes in, when `e` is an operation with an integer form
/// (`Add`, `Sub`, `Mul`, `Min`, `Max`, a comparison, `Select`) over integer operands.
pub(super) fn int_op_ty(e: &ScalarExpr) -> Option<ElementKind> {
    match e {
        ScalarExpr::Add(a, _) | ScalarExpr::Sub(a, _) | ScalarExpr::Mul(a, _) => {
            crate::types::int_spine_type(a)
        }
        ScalarExpr::Binary(op, a, _) if crate::types::int_binary_ok(*op) => {
            crate::types::int_spine_type(a)
        }
        ScalarExpr::Select(_, a, _) => crate::types::int_spine_type(a),
        _ => None,
    }
}

/// Evaluate an integer-typed operation from its definition: exact `i128` arithmetic
/// wrapped to the type's width (two's complement), so no overflow is undefined here
/// and the emitted C is compared against this.
pub(super) fn eval_int_op(e: &ScalarExpr, t: ElementKind, ev: &Eval<'_>) -> Val {
    let int = |x: &ScalarExpr| eval(x, ev).i128();
    match e {
        ScalarExpr::Add(a, b) => int_val(int(a).wrapping_add(int(b)), t),
        ScalarExpr::Sub(a, b) => int_val(int(a).wrapping_sub(int(b)), t),
        ScalarExpr::Mul(a, b) => int_val(int(a).wrapping_mul(int(b)), t),
        ScalarExpr::Binary(op, a, b) => {
            let (x, y) = (int(a), int(b));
            match op {
                BinaryOp::Max => int_val(x.max(y), t),
                BinaryOp::Min => int_val(x.min(y), t),
                // A comparison yields the compute type's 1/0, like a float comparison.
                BinaryOp::CmpEq => Val::Float(f64::from(u8::from(x == y))),
                BinaryOp::CmpNe => Val::Float(f64::from(u8::from(x != y))),
                BinaryOp::CmpLt => Val::Float(f64::from(u8::from(x < y))),
                BinaryOp::CmpLe => Val::Float(f64::from(u8::from(x <= y))),
                BinaryOp::CmpGt => Val::Float(f64::from(u8::from(x > y))),
                BinaryOp::CmpGe => Val::Float(f64::from(u8::from(x >= y))),
                other => {
                    panic!("oracle: {other:?} has no integer form (the plan gate rejects this)")
                }
            }
        }
        ScalarExpr::Select(c, a, b) => {
            // The condition is any scalar, tested `!= 0`: an integer condition by value,
            // a float one rounded to the compute dtype first, like the float `Select`.
            let cv = eval(c, ev);
            let taken = match cv {
                Val::Raw(bits, dt) if is_int(dt) => raw_to_i128(bits, dt) != 0,
                other => round_to_compute(other.f64(), ev.dtype) != 0.0,
            };
            if taken { eval(a, ev) } else { eval(b, ev) }
        }
        other => unreachable!("oracle: {other:?} is not an integer-typed operation"),
    }
}

/// A [`Val`] as the exact number a value of type `from` holds (floats rounded to
/// `from` first, see the `Cast` arm of [`eval`]).
pub(super) fn val_to_num(v: Val, from: ElementKind) -> crate::cast_ref::Num {
    use crate::cast_ref::Num;
    if crate::types::is_cast_float_endpoint(from) {
        Num::F(round_to_compute(v.f64(), from))
    } else {
        Num::I(v.i128())
    }
}

/// Infix arithmetic (`Add`/`Sub`/`Mul`) in the compute dtype's domain: exact-
/// wrapping integers (each op wraps at the C width) or `f64` floats.
pub(super) fn arith(
    ev: &Eval<'_>,
    a: &ScalarExpr,
    b: &ScalarExpr,
    ff: impl Fn(f64, f64) -> f64,
    fi: impl Fn(i128, i128) -> i128,
    // The complex rule is passed EXPLICITLY rather than derived from `ff`.
    // Component-wise application of the real rule is correct for `+` and `-`
    // and WRONG for `*`: complex multiplication mixes components
    // (`ac - bd, ad + bc`), and no inspection of a closure can recover which
    // operation it represents. Three closures is the price of not guessing.
    fc: impl Fn((f64, f64), (f64, f64)) -> (f64, f64),
) -> Val {
    if is_complex(ev.dtype) {
        let (ar, ai) = eval(a, ev).complex();
        let (br, bi) = eval(b, ev).complex();
        let (re, im) = fc((ar, ai), (br, bi));
        return Val::Complex(re, im);
    }
    if is_int(ev.dtype) {
        let r = fi(eval(a, ev).i128(), eval(b, ev).i128());
        // `+`/`-`/`*` produce identical BIT patterns signed or unsigned, but the
        // wrapped VALUE differs — `wrap_for` zero-extends an unsigned dtype so
        // `3_000_000_000u32` stays positive instead of reading as negative.
        Val::Int(wrap_for(r, op_width(ev.dtype), ev.dtype))
    } else {
        Val::Float(ff(eval(a, ev).f64(), eval(b, ev).f64()))
    }
}

/// A comparison (the compute type's 1/0) or `Max`/`Min` over two integers, in exact `i128`. Going
/// through `f64` (as the float path does) is exact only below 2^53: at a 64-bit compute dtype
/// `max(2^53 + 1, 2^53 + 1)` came out `2^53` and `i64::MAX == i64::MAX - 1` came out true.
pub(super) fn int_cmp_or_extreme(op: BinaryOp, x: i128, y: i128) -> i128 {
    match op {
        BinaryOp::Max => x.max(y),
        BinaryOp::Min => x.min(y),
        BinaryOp::CmpEq => i128::from(x == y),
        BinaryOp::CmpNe => i128::from(x != y),
        BinaryOp::CmpLt => i128::from(x < y),
        BinaryOp::CmpLe => i128::from(x <= y),
        BinaryOp::CmpGt => i128::from(x > y),
        _ => i128::from(x >= y),
    }
}

/// `Binary` node: int-only ops (bitwise/shift/logical) route to the integer
/// speller; `Nextafter` is dtype-aware; everything else is the float speller.
pub(super) fn eval_binary(op: BinaryOp, a: &ScalarExpr, b: &ScalarExpr, ev: &Eval<'_>) -> Val {
    if op.is_int_only() {
        return Val::Int(binary_op_int(
            op,
            eval(a, ev).i128(),
            eval(b, ev).i128(),
            ev.dtype,
        ));
    }
    if is_int(ev.dtype) && (op.is_cmp() || matches!(op, BinaryOp::Max | BinaryOp::Min)) {
        return Val::Int(int_cmp_or_extreme(
            op,
            eval(a, ev).i128(),
            eval(b, ev).i128(),
        ));
    }
    if matches!(op, BinaryOp::Nextafter) {
        let (x, y) = (eval(a, ev).f64(), eval(b, ev).f64());
        return Val::Float(match ev.dtype {
            // F32Strict stores/steps the f32 lattice too (the emitter lowers it
            // through binary_f32 → nextafterf); the f64 arm would demote-round
            // back to a silent no-op.
            ElementKind::F32 | ElementKind::F32Strict => {
                f64::from(next_after_f32(x as f32, y as f32))
            }
            _ => next_after_f64(x, y),
        });
    }
    // Bit-preserving binaries on a narrow float. `Copysign` is a pure mask;
    // `Max`/`Min` COMPARE through `f64` — which is exact, the comparison is not
    // the lossy step — and then move the winner's RAW bytes, exactly as the
    // emitter's bit-move path does. NaN-first ordering mirrors both the emitted
    // `(pa != pa ? a : ...)` and the f64 arm below, so the two cannot drift.
    if matches!(op, BinaryOp::Copysign | BinaryOp::Max | BinaryOp::Min) {
        if let (Val::Raw(ba, da), Val::Raw(bb, db)) = (eval(a, ev), eval(b, ev)) {
            if da == db {
                if let Some((sign, mag)) = raw_sign_masks(da) {
                    let (va, vb) = (raw_to_f64(ba, da), raw_to_f64(bb, db));
                    return Val::Raw(
                        match op {
                            BinaryOp::Copysign => (ba & mag) | (bb & sign),
                            BinaryOp::Max => {
                                if va.is_nan() || vb.is_nan() {
                                    if va.is_nan() { ba } else { bb }
                                } else if va >= vb {
                                    ba
                                } else {
                                    bb
                                }
                            }
                            _ => {
                                if va.is_nan() || vb.is_nan() {
                                    if va.is_nan() { ba } else { bb }
                                } else if va <= vb {
                                    ba
                                } else {
                                    bb
                                }
                            }
                        },
                        da,
                    );
                }
            }
        }
    }
    let (x, y) = (eval(a, ev).f64(), eval(b, ev).f64());
    // The Cmp* family is DISCONTINUOUS (yields exactly 1.0/0.0): the emitter rounds
    // BOTH operands to the compute dtype before comparing (`(float)a > (float)b`),
    // so an inexact Const or a compute-dtype-underflowing operand decides the same
    // way here. Continuous ops (Max/Min/Pow/Atan2/Rem/…) stay accurate-f64: a
    // flipped near-tie there returns a within-tolerance operand.
    let (x, y) = if op.is_cmp() {
        (round_to_compute(x, ev.dtype), round_to_compute(y, ev.dtype))
    } else {
        (x, y)
    };
    Val::Float(binary_op_f64(op, x, y))
}

// ===========================================================================
// Store — encode a Val into the output storage dtype.
// ===========================================================================

/// Encode `v` into `buf` at element index `off` for output dtype `out`. A `Raw`
/// value whose source dtype equals `out` moves VERBATIM (bit-preserving); a float
/// rounds to `out` (RNE for the halves), an integer wraps to `out`'s width.
pub(super) fn store_val(v: Val, out: ElementKind, buf: &mut [u8], off: usize) {
    let bits = match v {
        Val::Raw(b, dt) if dt == out => b,
        Val::Raw(b, dt) => {
            if is_int(dt) {
                encode_int(raw_to_i128(b, dt), out)
            } else {
                encode_float(raw_to_f64(b, dt), out)
            }
        }
        Val::Float(f) if is_complex(out) => encode_complex(f, 0.0, out),
        Val::Float(f) => encode_float(f, out),
        Val::Int(i) if is_complex(out) => encode_int_as_complex(i, out),
        Val::Int(i) => encode_int(i, out),
        Val::Complex(re, im) => encode_complex(re, im, out),
    };
    write_elem(buf, off, out, bits);
}

/// `f` as an `f32` rounded **to odd** (the magnitude truncated, its last bit set when anything was dropped). A 16-bit float
/// narrowed from this rounds exactly as it would from `f` itself (the `f32` keeps at least two bits more than either
/// format, so the sticky bit is all the second rounding needs), where the nearest-even `f as f32` first can land on a tie
/// and double-round (ROADMAP U26). NaN, infinities and zero pass through with their sign.
pub(super) fn f64_to_f32_odd(f: f64) -> f32 {
    let r = f as f32;
    if !f.is_finite() || f == 0.0 || f64::from(r) == f {
        return r;
    }
    // toward zero: step the nearest result back when it went up in magnitude (`r` may be infinite, whose step is MAX)
    let bits = r.to_bits();
    let t = if f64::from(r).abs() > f.abs() {
        f32::from_bits(bits - 1)
    } else {
        r
    };
    f32::from_bits(t.to_bits() | 1)
}

/// Encode an `f64` compute value to `out` (RNE halves; U8 narrows a 0/1
/// predicate; I64 widens a count).
pub(super) fn encode_float(f: f64, out: ElementKind) -> u128 {
    match out {
        ElementKind::F64 => u128::from(f.to_bits()),
        ElementKind::F32 | ElementKind::F32Strict => u128::from((f as f32).to_bits()),
        ElementKind::Fp8E4M3FN => u128::from(f64_to_fp8_e4m3fn_bits(f)),
        ElementKind::Fp8E5M2 => u128::from(f64_to_fp8_e5m2_bits(f)),
        ElementKind::F16 => u128::from(f32_to_f16_bits(f64_to_f32_odd(f))),
        ElementKind::Bf16 => u128::from(f32_to_bf16_bits(f64_to_f32_odd(f))),
        ElementKind::U8 | ElementKind::Bool => u128::from(f as u8),
        ElementKind::I64 => (f as i64) as u128,
        ElementKind::I32 => u128::from((f as i32) as u32),
        ElementKind::I8 => u128::from((f as i8) as u8),
        ElementKind::I16 => u128::from((f as i16) as u16),
        ElementKind::U16 => u128::from(f as u16),
        ElementKind::U32 => u128::from(f as u32),
        ElementKind::U64 => u128::from(f as u64),
        ElementKind::I4 | ElementKind::U4 => u128::from((f as i8 as u8) & 0x0f),
        ElementKind::B1 => u128::from(u8::from(f != 0.0)),
        other => panic!("oracle: encode_float to unsupported dtype {other:?}"),
    }
}

/// Encode a wide integer to `out`, truncating (two's-complement wrap) to width.
pub(super) fn encode_int(i: i128, out: ElementKind) -> u128 {
    match out {
        ElementKind::I32 => u128::from((i as i32) as u32),
        ElementKind::I64 => (i as i64) as u128,
        ElementKind::I8 => u128::from((i as i8) as u8),
        ElementKind::I16 => u128::from((i as i16) as u16),
        ElementKind::U8 | ElementKind::Bool => u128::from(i as u8),
        ElementKind::U16 => u128::from(i as u16),
        ElementKind::U32 => u128::from(i as u32),
        ElementKind::U64 => u128::from(i as u64),
        ElementKind::I4 | ElementKind::U4 => u128::from((i as u8) & 0x0f),
        ElementKind::B1 => u128::from((i as u8) & 1),
        other => panic!("oracle: encode_int to unsupported dtype {other:?}"),
    }
}
