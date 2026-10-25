// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Numerics qualifiers** (`docs/ir-numerics-qualifiers.md`): the reference
//! semantics, the plan gate, the declines, and the `approx` checker.
//!
//! The directed-rounding reference in `unpopped::numerics` is built from an
//! error-free transform and the sign of an exactly computed remainder. These tests
//! do NOT check it against itself: they recompute the exact result with `i128`
//! integer arithmetic (`value = mantissa * 2^exponent`) and round it with an
//! independent integer procedure, so the two share no code.

use std::cmp::Ordering;

use unpopped::backend::{Decline, DeclinedOp, Lowering, Spelling, lower_expr};
use unpopped::ir::{
    Numerics, OpDef, QualOp, ReduceOp, RoundMode, Saturation, ScalarExpr, input, konst,
};
use unpopped::numerics::{APPROX_ULP, LOG2_ABS_FLOOR, approx_within, eval_qualified_f32, ftz_f32};
use unpopped::optimize::optimize;
use unpopped::plan::{PlanError, try_build_plan};
use unpopped_vocab::{ArchSku, ElementKind, OpCategory, OperandDesc, structure_key};

const F32: ElementKind = ElementKind::F32;
const MODES: [RoundMode; 3] = [
    RoundMode::TowardZero,
    RoundMode::TowardNegInf,
    RoundMode::TowardPosInf,
];

fn mode(round: RoundMode) -> Numerics {
    Numerics {
        round,
        ..Numerics::default()
    }
}

// ---------------------------------------------------------------- exact reference

/// `x = m * 2^e` exactly (a finite f32).
fn decompose(x: f32) -> (i128, i32) {
    let bits = x.to_bits();
    let neg = bits >> 31 == 1;
    let e = ((bits >> 23) & 0xff) as i32;
    let frac = i128::from(bits & 0x7f_ffff);
    let (m, exp) = if e == 0 {
        (frac, -149)
    } else {
        (frac | (1 << 23), e - 150)
    };
    (if neg { -m } else { m }, exp)
}

/// Round the exact value `m * 2^e` (non-zero) to an f32 in `round`, by integer
/// arithmetic only. `None` if the operands are so far apart in magnitude that the
/// caller's exact sum would overflow `i128` (those cases are skipped, not guessed).
fn round_exact(m: i128, e: i32, round: RoundMode) -> f32 {
    assert!(m != 0);
    let neg = m < 0;
    let a = m.unsigned_abs();
    let top = 127 - a.leading_zeros() as i32;
    let q = (top + e - 23).max(-149);
    let up_for_magnitude = |has_rem: bool| match round {
        RoundMode::TowardZero => false,
        RoundMode::TowardNegInf => neg && has_rem,
        RoundMode::TowardPosInf => !neg && has_rem,
        _ => unreachable!(),
    };
    let (sig, inexact) = if e >= q {
        (a << (e - q) as u32, false)
    } else {
        let shift = (q - e) as u32;
        if shift >= 127 {
            (0u128, true)
        } else {
            (a >> shift, a & ((1u128 << shift) - 1) != 0)
        }
    };
    let sig = sig + u128::from(up_for_magnitude(inexact));
    let v = sig as f64 * 2f64.powi(q);
    let max = f64::from(f32::MAX);
    let mag = if v > max {
        // Overflow, IEEE 754 7.4.
        let to_inf = match round {
            RoundMode::TowardZero => false,
            RoundMode::TowardNegInf => neg,
            RoundMode::TowardPosInf => !neg,
            _ => unreachable!(),
        };
        if to_inf { f32::INFINITY } else { f32::MAX }
    } else {
        v as f32
    };
    if neg { -mag } else { mag }
}

/// Align two `m * 2^e` values to a common exponent, or `None` past `i128`'s reach.
fn align(a: (i128, i32), b: (i128, i32)) -> Option<(i128, i128, i32)> {
    let e = a.1.min(b.1);
    let (sa, sb) = ((a.1 - e) as u32, (b.1 - e) as u32);
    let fits = |m: i128, s: u32| s < 100 && m.unsigned_abs().leading_zeros() > s + 2;
    if !fits(a.0, sa) || !fits(b.0, sb) {
        return None;
    }
    Some((a.0 << sa, b.0 << sb, e))
}

/// Exact `a + b` as `(m, e)`; `None` when the spread is too wide for `i128`.
fn exact_add(a: f32, b: f32) -> Option<(i128, i32)> {
    let (x, y, e) = align(decompose(a), decompose(b))?;
    Some((x + y, e))
}

fn exact_mul(a: f32, b: f32) -> (i128, i32) {
    let (ma, ea) = decompose(a);
    let (mb, eb) = decompose(b);
    (ma * mb, ea + eb)
}

fn cmp_exact(x: (i128, i32), y: (i128, i32)) -> Option<Ordering> {
    let (a, b, _) = align(x, y)?;
    Some(a.cmp(&b))
}

/// The edge-value grid: signs, powers of two, neighbours of 1, subnormals, the
/// extremes, and a few ordinary values.
fn grid() -> Vec<f32> {
    let base = [
        1.0f32,
        3.0,
        0.1,
        1.0 + f32::EPSILON,
        1.0 - f32::EPSILON / 2.0,
        2.5e-39, // subnormal
        f32::from_bits(1),
        f32::MIN_POSITIVE,
        f32::MAX,
        1.0e30,
        7.0e-20,
        16_777_217.0, // 2^24 + 1 (rounds)
        0.3,
    ];
    let mut v = Vec::new();
    for &x in &base {
        v.push(x);
        v.push(-x);
    }
    v
}

// ---------------------------------------------------------------- directed rounding

#[test]
fn add_sub_mul_directed_rounding_equals_exact_integer_rounding() {
    let g = grid();
    let (mut checked, mut skipped, mut inexact) = (0u32, 0u32, 0u32);
    for &a in &g {
        for &b in &g {
            for round in MODES {
                let n = mode(round);
                // Add
                match exact_add(a, b) {
                    Some((m, e)) if m != 0 => {
                        let want = round_exact(m, e, round);
                        let got = eval_qualified_f32(QualOp::Add, n, &[a, b]);
                        assert_eq!(got.to_bits(), want.to_bits(), "{a:e} + {b:e} {round:?}");
                        // Sub is Add of the negation.
                        let (ms, es) = exact_add(a, -b).unwrap();
                        if ms != 0 {
                            let want = round_exact(ms, es, round);
                            let got = eval_qualified_f32(QualOp::Sub, n, &[a, b]);
                            assert_eq!(got.to_bits(), want.to_bits(), "{a:e} - {b:e} {round:?}");
                        }
                        checked += 1;
                        if want != (a + b) {
                            inexact += 1;
                        }
                    }
                    Some(_) => {}
                    None => skipped += 1,
                }
                // Mul
                let (m, e) = exact_mul(a, b);
                let want = round_exact(m, e, round);
                let got = eval_qualified_f32(QualOp::Mul, n, &[a, b]);
                assert_eq!(got.to_bits(), want.to_bits(), "{a:e} * {b:e} {round:?}");
                checked += 1;
            }
        }
    }
    // Controls: the grid exercises the cases directed rounding exists for. Without
    // this the loop above could pass over operands where every mode agrees.
    assert!(checked > 1000, "only {checked} cases checked");
    assert!(
        inexact > 50,
        "only {inexact} cases where a directed mode differs from RN"
    );
    assert!(skipped < checked / 3, "{skipped} skipped of {checked}");
}

#[test]
fn directed_modes_differ_on_an_inexact_sum_and_agree_on_an_exact_one() {
    // 1 + 2^-30 is inexact in f32: toward +inf is the next float up, the others 1.
    let (one, tiny) = (1.0f32, 2f32.powi(-30));
    let r = |m| eval_qualified_f32(QualOp::Add, mode(m), &[one, tiny]);
    assert_eq!(r(RoundMode::TowardPosInf), 1.0 + f32::EPSILON);
    assert_eq!(r(RoundMode::TowardZero), 1.0);
    assert_eq!(r(RoundMode::TowardNegInf), 1.0);
    // Control: an exact sum is the same in every mode.
    for m in MODES {
        assert_eq!(eval_qualified_f32(QualOp::Add, mode(m), &[1.0, 2.0]), 3.0);
    }
}

/// `r` brackets the exact quotient `a / b`, with `step` the next float in the
/// direction of `round`.
fn quotient_ok(a: f32, b: f32, r: f32, round: RoundMode) -> bool {
    // r <= a/b  <=>  r*b <= a  (b > 0), reversed for b < 0.
    let (ma, ea) = decompose(a);
    let below = |x: f32| -> Option<bool> {
        let o = cmp_exact(exact_mul(x, b), (ma, ea))?;
        Some(if b > 0.0 {
            o != Ordering::Greater
        } else {
            o != Ordering::Less
        })
    };
    let strictly_below = |x: f32| -> Option<bool> {
        let o = cmp_exact(exact_mul(x, b), (ma, ea))?;
        Some(if b > 0.0 {
            o == Ordering::Less
        } else {
            o == Ordering::Greater
        })
    };
    let up = |x: f32| {
        let bits = x.to_bits();
        if x == 0.0 {
            f32::from_bits(1)
        } else if x > 0.0 {
            f32::from_bits(bits + 1)
        } else {
            f32::from_bits(bits - 1)
        }
    };
    let down = |x: f32| -up(-x);
    let positive = (a < 0.0) == (b < 0.0);
    let want_down = match round {
        RoundMode::TowardNegInf => true,
        RoundMode::TowardPosInf => false,
        _ => positive,
    };
    if want_down {
        // r is the greatest float <= a/b.
        below(r) == Some(true) && below(up(r)) == Some(false)
    } else {
        // r is the least float >= a/b.
        strictly_below(r) == Some(false) && strictly_below(down(r)) == Some(true)
    }
}

#[test]
fn div_directed_rounding_brackets_the_exact_quotient() {
    let g = grid();
    let mut checked = 0u32;
    for &a in &g {
        for &b in &g {
            let exact_ratio = f64::from(a) / f64::from(b);
            // Skip results near overflow/underflow, where this bracket check's
            // `i128` alignment gives up; the overflow table covers them.
            if !(1e-30..1e30).contains(&exact_ratio.abs()) {
                continue;
            }
            for round in MODES {
                let r = eval_qualified_f32(QualOp::Div, mode(round), &[a, b]);
                if quotient_ok(a, b, r, round) {
                    checked += 1;
                } else if cmp_exact(exact_mul(r, b), decompose(a)).is_none() {
                    // alignment out of reach: not a verdict
                } else {
                    panic!("{a:e} / {b:e} {round:?} = {r:e} does not bracket the quotient");
                }
            }
        }
    }
    assert!(checked > 200, "only {checked} quotients verified");
}

#[test]
fn sqrt_directed_rounding_brackets_the_exact_root() {
    let mut checked = 0u32;
    for &a in grid().iter().filter(|x| **x > 0.0) {
        let (ma, ea) = decompose(a);
        for round in MODES {
            let r = eval_qualified_f32(QualOp::Sqrt, mode(round), &[a]);
            let sq = exact_mul(r, r);
            let up = f32::from_bits(r.to_bits() + 1);
            let up_sq = exact_mul(up, up);
            let down = f32::from_bits(r.to_bits() - 1);
            let down_sq = exact_mul(down, down);
            let (Some(c), Some(cu), Some(cd)) = (
                cmp_exact(sq, (ma, ea)),
                cmp_exact(up_sq, (ma, ea)),
                cmp_exact(down_sq, (ma, ea)),
            ) else {
                continue;
            };
            let toward_down = !matches!(round, RoundMode::TowardPosInf);
            if toward_down {
                // greatest float whose square <= a
                assert!(
                    c != Ordering::Greater && cu == Ordering::Greater,
                    "sqrt({a:e}) RD"
                );
            } else {
                // least float whose square >= a
                assert!(
                    c != Ordering::Less && cd == Ordering::Less,
                    "sqrt({a:e}) RU"
                );
            }
            checked += 1;
        }
    }
    assert!(checked > 20, "only {checked} roots verified");
}

#[test]
fn fma_directed_rounding_equals_exact_integer_rounding() {
    let g = grid();
    let mut checked = 0u32;
    for &a in &g {
        for &b in &g {
            for &c in g.iter().step_by(3) {
                let (mp, ep) = exact_mul(a, b);
                let Some((x, y, e)) = align((mp, ep), decompose(c)) else {
                    continue;
                };
                if x + y == 0 {
                    continue;
                }
                for round in MODES {
                    let want = round_exact(x + y, e, round);
                    let got = eval_qualified_f32(QualOp::Fma, mode(round), &[a, b, c]);
                    assert_eq!(
                        got.to_bits(),
                        want.to_bits(),
                        "fma({a:e},{b:e},{c:e}) {round:?}"
                    );
                    checked += 1;
                }
            }
        }
    }
    assert!(checked > 500, "only {checked} fused results verified");
}

#[test]
fn directed_overflow_and_exact_zero_follow_ieee_754() {
    let big = f32::MAX;
    // 7.4: overflow in each directed mode, both signs.
    let mul = |m, a: f32, b: f32| eval_qualified_f32(QualOp::Mul, mode(m), &[a, b]);
    assert_eq!(mul(RoundMode::TowardZero, big, 2.0), f32::MAX);
    assert_eq!(mul(RoundMode::TowardNegInf, big, 2.0), f32::MAX);
    assert_eq!(mul(RoundMode::TowardPosInf, big, 2.0), f32::INFINITY);
    assert_eq!(mul(RoundMode::TowardZero, -big, 2.0), -f32::MAX);
    assert_eq!(mul(RoundMode::TowardNegInf, -big, 2.0), f32::NEG_INFINITY);
    assert_eq!(mul(RoundMode::TowardPosInf, -big, 2.0), -f32::MAX);
    // Control: round-to-nearest overflows to infinity.
    assert_eq!(
        eval_qualified_f32(QualOp::Mul, Numerics::default(), &[big, 2.0]),
        f32::INFINITY
    );
    // 6.3: an exact zero sum of opposite signs is +0, except toward -inf.
    let add = |m, a: f32, b: f32| eval_qualified_f32(QualOp::Add, mode(m), &[a, b]);
    for m in [RoundMode::TowardZero, RoundMode::TowardPosInf] {
        assert!(add(m, 1.5, -1.5).is_sign_positive() && add(m, 1.5, -1.5) == 0.0);
    }
    assert!(add(RoundMode::TowardNegInf, 1.5, -1.5).is_sign_negative());
    // Like-signed zeros keep their sign in every mode.
    for m in MODES {
        assert!(add(m, -0.0, -0.0).is_sign_negative(), "{m:?}");
        assert!(add(m, 0.0, 0.0).is_sign_positive(), "{m:?}");
    }
    // Underflow: a positive tiny product rounds up to the smallest subnormal toward
    // +inf and to +0 toward zero; the mirror image for a negative one.
    let tiny = 2f32.powi(-100);
    assert_eq!(mul(RoundMode::TowardPosInf, tiny, tiny), f32::from_bits(1));
    assert_eq!(mul(RoundMode::TowardZero, tiny, tiny), 0.0);
    assert_eq!(
        mul(RoundMode::TowardNegInf, -tiny, tiny),
        -f32::from_bits(1)
    );
    assert!(
        mul(RoundMode::TowardPosInf, -tiny, tiny) == 0.0
            && mul(RoundMode::TowardPosInf, -tiny, tiny).is_sign_negative()
    );
}

// ---------------------------------------------------------------- ftz

#[test]
fn ftz_flushes_operands_and_results_and_keeps_the_sign() {
    let ftz = Numerics {
        ftz: true,
        ..Numerics::default()
    };
    let sub = 2f32.powi(-130); // subnormal
    let mul = |n, a: f32, b: f32| eval_qualified_f32(QualOp::Mul, n, &[a, b]);
    // Step 1, operand flush: the subnormal input reads as zero.
    assert_eq!(mul(ftz, sub, 2f32.powi(100)), 0.0);
    assert_eq!(
        mul(Numerics::default(), sub, 2f32.powi(100)),
        2f32.powi(-30)
    );
    // Step 3, result flush: normal inputs, subnormal exact result.
    assert_eq!(mul(ftz, 2f32.powi(-100), 2f32.powi(-30)), 0.0);
    assert_eq!(
        mul(Numerics::default(), 2f32.powi(-100), 2f32.powi(-30)),
        sub
    );
    // The sign of a flushed value is kept.
    let neg = mul(ftz, -2f32.powi(-100), 2f32.powi(-30));
    assert!(neg == 0.0 && neg.is_sign_negative());
    assert!(ftz_f32(-sub).is_sign_negative() && ftz_f32(-sub) == 0.0);
    // NaN, infinity and normals are untouched.
    assert!(ftz_f32(f32::NAN).is_nan());
    assert_eq!(ftz_f32(f32::INFINITY), f32::INFINITY);
    assert_eq!(ftz_f32(f32::MIN_POSITIVE), f32::MIN_POSITIVE);
    assert_eq!(ftz_f32(-0.0).to_bits(), (-0.0f32).to_bits());
    // ftz composes with a directed mode: operands flushed, then rounded, then flushed.
    let both = Numerics {
        round: RoundMode::TowardPosInf,
        ftz: true,
        approx: false,
    };
    assert_eq!(eval_qualified_f32(QualOp::Add, both, &[sub, sub]), 0.0);
}

// ---------------------------------------------------------------- approx

#[test]
fn approx_within_accepts_inside_the_bound_and_rejects_outside_it() {
    let exact = 1.5f32;
    let ulp = f64::from(f32::EPSILON); // ulp of values in [1, 2)
    let off = |k: f64| (f64::from(exact) + k * ulp) as f32;
    assert!(approx_within(QualOp::Div, exact, exact));
    assert!(
        approx_within(QualOp::Div, off(3.0), exact),
        "3 ulp is inside"
    );
    assert!(
        approx_within(QualOp::Div, off(-4.0), exact),
        "4 ulp is the edge"
    );
    assert!(
        !approx_within(QualOp::Div, off(5.0), exact),
        "5 ulp is outside"
    );
    // The spec's failing control: 2^-18 relative error is far outside.
    assert!(!approx_within(
        QualOp::Exp2,
        exact * (1.0 + 2f32.powi(-18)),
        exact
    ));
    // A non-finite result for a finite normal exact value is never within the bound.
    assert!(!approx_within(QualOp::Div, f32::NAN, exact));
    assert!(!approx_within(QualOp::Div, f32::INFINITY, exact));
    // Outside the domain nothing is promised (and the checker says so).
    assert!(approx_within(QualOp::Div, 123.0, 0.0));
    assert!(approx_within(QualOp::Div, 123.0, f32::INFINITY));
    assert_eq!(APPROX_ULP, 4.0);
}

#[test]
fn log2_has_an_absolute_floor_near_one() {
    // exact = 2^-10 has a tiny ulp; 1e-7 away is far beyond 4 ulp but inside 2^-20.
    let exact = 2f32.powi(-10);
    assert!(approx_within(QualOp::Log2, exact + 5.0e-7, exact));
    assert!(
        !approx_within(QualOp::Div, exact + 5.0e-7, exact),
        "control: no floor for Div"
    );
    assert!(!approx_within(
        QualOp::Log2,
        exact + 2.0 * LOG2_ABS_FLOOR as f32,
        exact
    ));
}

// ---------------------------------------------------------------- plan gate

fn plan(op: &OpDef, dtype: ElementKind) -> Result<(), PlanError> {
    let a = OperandDesc::new(1, &[7], &[1], dtype, 256);
    let o = OperandDesc::new(1, &[7], &[1], dtype, 256);
    let key = structure_key(OpCategory::BinaryElementwise, &[a, o], ArchSku::Sm89);
    try_build_plan(op, &key).map(|_| ())
}

fn elementwise(body: unpopped::ir::Expr) -> OpDef {
    OpDef::elementwise("q", 2, &[F32], body)
}

fn declines(r: Result<(), PlanError>, needle: &str) {
    match r {
        Err(PlanError::InadmissibleOpAtDtype { detail, .. })
        | Err(PlanError::InadmissibleOpInCell { detail }) => {
            assert!(
                detail.contains(needle),
                "decline text lacks {needle:?}: {detail}"
            );
        }
        other => panic!("expected a typed decline mentioning {needle:?}, got {other:?}"),
    }
}

#[test]
fn the_legal_cells_of_the_qualifier_table_plan() {
    let ab = || (input(0), input(1));
    for round in [
        RoundMode::NearestEven,
        RoundMode::TowardZero,
        RoundMode::TowardNegInf,
        RoundMode::TowardPosInf,
    ] {
        let (a, b) = ab();
        plan(&elementwise((a + b).rounded(round)), F32).expect("Add rounded");
        let (a, b) = ab();
        plan(&elementwise((a * b).rounded(round)), F32).expect("Mul rounded");
        let (a, b) = ab();
        plan(&elementwise((a / b).rounded(round)), F32).expect("Div rounded");
        let (a, b) = ab();
        plan(&elementwise(a.fma(b, konst(1.0)).rounded(round)), F32).expect("Fma rounded");
        plan(
            &elementwise(input(0).unary(unpopped::ir::UnaryOp::Sqrt).rounded(round)),
            F32,
        )
        .expect("Sqrt rounded");
    }
    let (a, b) = ab();
    plan(&elementwise((a * b).ftz()), F32).expect("Mul ftz");
    let (a, b) = ab();
    plan(&elementwise((a / b).approx()), F32).expect("Div approx");
    for u in [
        unpopped::ir::UnaryOp::Rsqrt,
        unpopped::ir::UnaryOp::Recip,
        unpopped::ir::UnaryOp::Exp2,
        unpopped::ir::UnaryOp::Log2,
        unpopped::ir::UnaryOp::Sqrt,
    ] {
        plan(&elementwise(input(0).unary(u).approx()), F32).expect("approx unary");
        plan(&elementwise(input(0).unary(u).ftz()), F32).expect("ftz unary");
    }
    plan(
        &elementwise((input(0) + input(1)).ftz()),
        ElementKind::F32Strict,
    )
    .expect("f32s");
}

#[test]
fn each_illegal_cell_is_a_typed_decline_naming_it() {
    use unpopped::ir::UnaryOp;
    // A rounding mode on an op that takes none.
    declines(
        plan(
            &elementwise(input(0).unary(UnaryOp::Exp2).rounded(RoundMode::TowardZero)),
            F32,
        ),
        "takes no rounding mode",
    );
    // approx on an op with no approximate form.
    declines(
        plan(&elementwise((input(0) + input(1)).approx()), F32),
        "no approximate form",
    );
    // approx with a directed mode.
    let both = Numerics {
        round: RoundMode::TowardZero,
        ftz: false,
        approx: true,
    };
    declines(
        plan(&elementwise((input(0) / input(1)).qualified(both)), F32),
        "approx with RoundMode",
    );
    // A node that is not a qualifiable operation.
    declines(
        plan(&elementwise(input(0).unary(UnaryOp::Tanh).ftz()), F32),
        "not one of",
    );
    declines(plan(&elementwise(input(0).ftz()), F32), "not one of");
    // f64 and half compute.
    declines(
        plan(&elementwise((input(0) + input(1)).ftz()), ElementKind::F64),
        "f32/f32s",
    );
    declines(
        plan(&elementwise((input(0) + input(1)).ftz()), ElementKind::F16),
        "f32/f32s",
    );
    // Combined with a Cast.
    declines(
        plan(
            &elementwise((input(0) + input(1)).ftz().cast(
                F32,
                RoundMode::NearestEven,
                Saturation::Wrap,
            )),
            F32,
        ),
        "Cast/Bitcast",
    );
    // Non-elementwise.
    let red = OpDef::reduction("s", 1, &[F32], (input(0) + input(0)).ftz(), ReduceOp::Sum);
    let a = OperandDesc::new(2, &[4, 8], &[8, 1], F32, 256);
    let o = OperandDesc::new(1, &[4], &[1], F32, 256);
    let key = structure_key(OpCategory::Reduction, &[a, o], ArchSku::Sm89);
    assert!(
        matches!(
            try_build_plan(&red, &key),
            Err(PlanError::InadmissibleOpInCell { .. })
        ),
        "a Qualified inside a reduction must be a typed decline"
    );
}

// ---------------------------------------------------------------- optimizer, pattern

#[test]
fn the_optimizer_leaves_a_qualified_expression_exactly_as_authored() {
    let plain = input(0) * konst(1.0);
    assert_ne!(
        optimize(&plain.0, F32),
        plain.0,
        "control: x*1 is rewritten"
    );
    // The identical shape under a qualifier is not touched.
    let q = (input(0) * konst(1.0)).ftz();
    assert_eq!(optimize(&q.0, F32), q.0);
    // And a qualifier deep inside protects the whole expression.
    let outer = (q.clone() + konst(0.0)) * konst(1.0);
    assert_eq!(optimize(&outer.0, F32), outer.0);
}

#[test]
fn pattern_derivation_and_recipes_decline_a_qualified_body() {
    let op = elementwise((input(0) + input(1)).ftz());
    assert!(matches!(
        unpopped::derive_pattern(&op),
        Err(unpopped::PatternError::QualifiedUnsupported)
    ));
    // Control: the same op without the qualifier derives a pattern.
    let plain = elementwise(input(0) + input(1));
    assert!(unpopped::derive_pattern(&plain).is_ok());
}

// ---------------------------------------------------------------- the seam declines

fn plain_lowering<'a>(
    leaf: &'a dyn Fn(u8) -> Result<Spelling, unpopped::backend::LowerError>,
    unary: &'a dyn Fn(
        unpopped::ir::UnaryOp,
        String,
    ) -> Result<Spelling, unpopped::backend::LowerError>,
    binary: &'a dyn Fn(
        unpopped::ir::BinaryOp,
        String,
        String,
    ) -> Result<Spelling, unpopped::backend::LowerError>,
) -> Lowering<'a> {
    Lowering::builder(leaf, unary, binary).build()
}

#[test]
fn a_lowering_with_no_qualified_seam_declines_and_names_the_qualifier() {
    let leaf = |i: u8| Ok(Spelling::Spelled(format!("x{i}")));
    let un = |_o, x: String| Ok(Spelling::Spelled(x));
    let bin = |_o, a: String, b: String| Ok(Spelling::Spelled(format!("{a}{b}")));
    let lo = plain_lowering(&leaf, &un, &bin);
    let e = (input(0) + input(1)).rounded(RoundMode::TowardPosInf);
    let got = lower_expr(&e.0, &lo).expect("a decline is a value, not an error");
    match got {
        Spelling::Declined(Decline::UnsupportedOp { op, why }) => {
            assert_eq!(op, DeclinedOp::Qualified);
            assert!(why.contains("TowardPosInf") && why.contains("Add"), "{why}");
        }
        other => panic!("expected a decline, got {other:?}"),
    }
    // Control: the same lowering spells the UNqualified add.
    let plain = lower_expr(&(input(0) + input(1)).0, &lo).unwrap();
    assert!(matches!(plain, Spelling::Spelled(_)), "{plain:?}");
    // A qualifier over a non-qualifiable child also declines.
    let odd = ScalarExpr::Qualified(Box::new(input(0).0), Numerics::default());
    assert!(matches!(
        lower_expr(&odd, &lo).unwrap(),
        Spelling::Declined(Decline::UnsupportedOp {
            op: DeclinedOp::Qualified,
            ..
        })
    ));
}

#[test]
fn a_seam_that_spells_the_operation_receives_its_operands_in_order() {
    let leaf = |i: u8| Ok(Spelling::Spelled(format!("x{i}")));
    let un = |_o, x: String| Ok(Spelling::Spelled(x));
    let bin = |_o, a: String, b: String| Ok(Spelling::Spelled(format!("{a}{b}")));
    let seen = std::cell::RefCell::new(Vec::new());
    let q = |op: QualOp, operands: Vec<String>, n: Numerics| {
        seen.borrow_mut().push((op, operands.clone(), n));
        Ok(Spelling::Spelled(format!(
            "q{op:?}({})",
            operands.join(",")
        )))
    };
    let lo = Lowering::builder(&leaf, &un, &bin).qualified(&q).build();
    let e = input(0)
        .fma(input(1), konst(2.0))
        .rounded(RoundMode::TowardZero);
    let got = lower_expr(&e.0, &lo).unwrap();
    assert!(
        matches!(&got, Spelling::Spelled(s) if s.starts_with("qFma(x0,x1,")),
        "{got:?}"
    );
    let seen = seen.borrow();
    assert_eq!(seen.len(), 1);
    assert_eq!(seen[0].0, QualOp::Fma);
    assert_eq!(seen[0].2.round, RoundMode::TowardZero);
}
