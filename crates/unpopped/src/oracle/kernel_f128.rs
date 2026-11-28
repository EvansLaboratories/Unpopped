// SPDX-License-Identifier: MIT OR Apache-2.0
//! **`F128` in the kernel oracle** (ROADMAP E3): a binary128 value is its 128-bit pattern, carried as `Val::Raw(bits, F128)` (so a
//! `Load`, a `Store` and a local copy already move it byte for byte), and every operation is `soft_f128`'s: add, subtract,
//! multiply, divide (one rounding to nearest even each), negate, absolute value, the six comparisons, `Select`, and `Cast` to and
//! from `F32`, `F64` and the integers. `types::check_f128_node` refuses every other node over an `F128` operand, so nothing else
//! reaches here. **A comparison yields the type of its operands**, as every float comparison does here: `1` or `0` in `F128`.

use super::kernel_eval::{KernelTrap, Machine};
use super::wide_ops::as_wide;
use super::{Val, int_val, raw_to_f64};
use crate::ir::{BinaryOp, CastSpec, Saturation, ScalarExpr, UnaryOp};
use crate::soft_f128::F128;
use crate::types::{self, Ty};
use crate::wide_int::{W256, wide_int_signed};
use core::cmp::Ordering;
use unpopped_vocab::ElementKind;

const F: ElementKind = ElementKind::F128;

/// A scalar float argument bound to a parameter of type `ty`: an `F128` parameter holds the exact binary128 of the `f64` given, any
/// other float the `f64` itself.
pub(super) fn scalar_float(ty: ElementKind, v: f64) -> Val {
    if ty == F {
        val(F128::from_f64(v))
    } else {
        Val::Float(v)
    }
}

fn val(x: F128) -> Val {
    Val::Raw(x.0, F)
}

fn one_or_zero(b: bool) -> F128 {
    if b { F128::ONE } else { F128::ZERO }
}

/// The signed 256-bit word `w` as an `i128`, saturating at the `i128` range.
fn saturate_i128(w: W256) -> i128 {
    if w.cmp_signed(W256::from_i128(i128::MIN)) == Ordering::Less {
        i128::MIN
    } else if w.cmp_signed(W256::from_i128(i128::MAX)) == Ordering::Greater {
        i128::MAX
    } else {
        w.low_u128() as i128
    }
}

/// `a op b` for a comparison: a NaN operand makes every comparison false except `!=`; `+0` equals `-0`.
fn compare(op: BinaryOp, a: F128, b: F128) -> bool {
    let ord = a.partial_cmp(b);
    match op {
        BinaryOp::CmpEq => ord == Some(Ordering::Equal),
        BinaryOp::CmpNe => ord != Some(Ordering::Equal),
        BinaryOp::CmpLt => ord == Some(Ordering::Less),
        BinaryOp::CmpLe => matches!(ord, Some(Ordering::Less | Ordering::Equal)),
        BinaryOp::CmpGt => ord == Some(Ordering::Greater),
        _ => matches!(ord, Some(Ordering::Greater | Ordering::Equal)),
    }
}

impl Machine<'_> {
    /// Whether `e` has type `F128`.
    pub(super) fn is_f128(&self, e: &ScalarExpr) -> bool {
        matches!(
            types::infer_ty_in(e, self.k.compute, &self.local_tys),
            Ok(Ty::Scalar(F))
        )
    }

    /// Whether `e` is a `Cast` out of an `F128` operand.
    pub(super) fn is_f128_source_cast(&self, e: &ScalarExpr) -> bool {
        matches!(e, ScalarExpr::Cast(x, _) if self.is_f128(x))
    }

    fn f128_at(&self, e: &ScalarExpr) -> Result<F128, KernelTrap> {
        match self.eval_expr(e)? {
            Val::Raw(b, F) => Ok(F128(b)),
            _ => Err(KernelTrap::Unsupported(
                "an F128 operand that holds no F128 value",
            )),
        }
    }

    /// Whether the value `c` (a condition) is taken: an `F128` is taken when it is not zero (a NaN is taken, as in C), a
    /// comparison by its operands, any other integer by value.
    pub(super) fn f128_taken(&self, c: &ScalarExpr) -> Result<bool, KernelTrap> {
        if self.is_f128(c) {
            let v = self.f128_at(c)?;
            return Ok(v.partial_cmp(F128::ZERO) != Some(Ordering::Equal));
        }
        Ok(self.eval_operand_int(c)? != 0)
    }

    /// An `F128` expression (the value is `Val::Raw(bits, F128)`).
    pub(super) fn eval_f128(&self, e: &ScalarExpr) -> Result<Val, KernelTrap> {
        let two = |a: &ScalarExpr, b: &ScalarExpr| {
            Ok::<_, KernelTrap>((self.f128_at(a)?, self.f128_at(b)?))
        };
        let v = match e {
            ScalarExpr::Local(i) => return Ok(self.local_val(*i)),
            ScalarExpr::Const(c) => F128::from_f64(*c),
            ScalarExpr::Add(a, b) => {
                let (x, y) = two(a, b)?;
                x.add_rne(y)
            }
            ScalarExpr::Sub(a, b) => {
                let (x, y) = two(a, b)?;
                x.sub_rne(y)
            }
            ScalarExpr::Mul(a, b) => {
                let (x, y) = two(a, b)?;
                x.mul_rne(y)
            }
            ScalarExpr::Div(a, b) => {
                let (x, y) = two(a, b)?;
                x.div_rne(y)
            }
            ScalarExpr::Binary(op, a, b) if op.is_cmp() => {
                let (x, y) = two(a, b)?;
                one_or_zero(compare(*op, x, y))
            }
            ScalarExpr::Unary(UnaryOp::Neg, x) => self.f128_at(x)?.neg(),
            ScalarExpr::Unary(UnaryOp::Abs, x) => self.f128_at(x)?.abs(),
            ScalarExpr::Select(c, a, b) => self.f128_at(if self.f128_taken(c)? { a } else { b })?,
            ScalarExpr::Cast(x, spec) => self.f128_cast_into(x, spec)?,
            _ => {
                return Err(KernelTrap::Unsupported(
                    "an F128 expression outside the computed subset",
                ));
            }
        };
        Ok(val(v))
    }

    /// A comparison of two `F128` operands, as the boolean (the integer evaluator's `eval_cmp`).
    pub(super) fn f128_cmp(
        &self,
        op: BinaryOp,
        a: &ScalarExpr,
        b: &ScalarExpr,
    ) -> Result<bool, KernelTrap> {
        Ok(compare(op, self.f128_at(a)?, self.f128_at(b)?))
    }

    /// A `Cast` whose result is `F128`: from `F128` itself, a float (exact), or an integer (rounded to nearest even when it has
    /// more than 113 significant bits).
    fn f128_cast_into(&self, x: &ScalarExpr, _spec: &CastSpec) -> Result<F128, KernelTrap> {
        let Ok(Ty::Scalar(from)) = types::infer_ty_in(x, self.k.compute, &self.local_tys) else {
            return Err(KernelTrap::Unsupported(
                "an F128 cast of an ill-typed operand",
            ));
        };
        if from == F {
            return self.f128_at(x);
        }
        if crate::wide_int::wide_int_bits(from).is_some() {
            let w = as_wide(self.eval_expr(x)?, from);
            return Ok(F128::from_w256(w, wide_int_signed(from).unwrap_or(false)));
        }
        if types::int_range(from).is_some() {
            return Ok(F128::from_w256(
                W256::from_i128(self.eval_int(x, from)?),
                true,
            ));
        }
        match self.eval_expr(x)? {
            Val::Float(f) => Ok(F128::from_f64(f)),
            Val::Raw(b, dt) => Ok(F128::from_f64(raw_to_f64(b, dt))),
            _ => Err(KernelTrap::Unsupported(
                "an F128 cast of a value that is not a float",
            )),
        }
    }

    /// A `Cast` out of an `F128` operand to `F32`, `F64` or a narrow integer (the checker admits no other destination).
    pub(super) fn f128_cast_out(&self, e: &ScalarExpr) -> Result<Val, KernelTrap> {
        let ScalarExpr::Cast(x, spec) = e else {
            return Err(KernelTrap::Unsupported("not an F128 cast"));
        };
        let v = self.f128_at(x)?;
        let clamp = spec.sat == Saturation::ClampFinite;
        match spec.to {
            ElementKind::F32 | ElementKind::F32Strict => {
                let r = v.to_f32();
                // ClampFinite: an overflow (an infinity included) lands on the largest finite value of its sign
                let r = if clamp && r.is_infinite() {
                    f32::MAX.copysign(r)
                } else {
                    r
                };
                Ok(Val::Float(f64::from(r)))
            }
            ElementKind::F64 => {
                let r = v.to_f64();
                let r = if clamp && r.is_infinite() {
                    f64::MAX.copysign(r)
                } else {
                    r
                };
                Ok(Val::Float(r))
            }
            t if types::int_range(t).is_some() => Ok(int_val(self.f128_to_int(v, spec)?, t)),
            _ => Err(KernelTrap::Unsupported(
                "an F128 cast to an unsupported destination",
            )),
        }
    }

    /// The integer a `Cast` of the `F128` operand `x` to a narrow integer type gives (the integer evaluator's entry).
    pub(super) fn f128_cast_to_int(
        &self,
        x: &ScalarExpr,
        spec: &CastSpec,
    ) -> Result<i128, KernelTrap> {
        self.f128_to_int(self.f128_at(x)?, spec)
    }

    /// `v` rounded to an integer under `spec.round` and saturated into `spec.to`'s range; a NaN is `0`.
    fn f128_to_int(&self, v: F128, spec: &CastSpec) -> Result<i128, KernelTrap> {
        let (lo, hi) = types::int_range(spec.to).ok_or(KernelTrap::Unsupported(
            "an F128 cast to a non-integer range",
        ))?;
        if v.is_nan() {
            return Ok(0);
        }
        let neg = v.0 >> 127 == 1;
        let r = match v.to_i256(spec.round) {
            // |v| < 2^255: the word is a signed 256-bit integer; anything outside i128 saturates by its sign
            Some(w) => saturate_i128(w),
            None if neg => i128::MIN,
            None => i128::MAX,
        };
        Ok(r.clamp(lo, hi))
    }
}
