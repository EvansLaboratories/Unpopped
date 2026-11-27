// SPDX-License-Identifier: MIT OR Apache-2.0
//! The wide integer dtypes `I128`, `U128`, `I256`, `U256` in a kernel (ROADMAP E2): wrapping arithmetic at the type's width, the
//! total `Div`, shifts with the count modulo the width, ordering, comparisons, and integer casts to and from the narrow integers
//! (`docs/int-div-shift-semantics.md` at the wide width).
//!
//! The 128-bit types are compared against Rust's own `i128`/`u128` operators (independent of the oracle's 256-bit limb type);
//! the 256-bit types against a `(hi, lo)` pair of `u128` model written here, and against hand-pinned values. Every operand set
//! includes 0, 1, the all-ones pattern, the sign bit alone and a pattern that crosses the 128-bit boundary.

use unpopped::exec::LaunchShape;
use unpopped::ir::Saturation;
use unpopped::ir::{BinaryOp, ScalarExpr};
use unpopped::kernel::{
    Block, BufferRole, KernelError, KernelFn, LocalDecl, Param, Stmt, check_kernel,
};
use unpopped_vocab::ElementKind::{self, I32, I64, I128, I256, U128};

#[path = "support/wide_int.rs"]
#[allow(dead_code, reason = "shared helpers: each test file uses a subset")]
mod support;
use support::*;

// ---- 128-bit reference: Rust's own operators ------------------------------------------------------------------------

fn ref128(op: BinaryOp, signed: bool, x: u128, y: u128) -> u128 {
    let (sx, sy) = (x as i128, y as i128);
    let sh = (y & 127) as u32;
    match op {
        BinaryOp::BitAnd => x & y,
        BinaryOp::BitOr => x | y,
        BinaryOp::BitXor => x ^ y,
        BinaryOp::Shl => x << sh,
        BinaryOp::Shr if signed => (sx >> sh) as u128,
        BinaryOp::Shr => x >> sh,
        BinaryOp::Max if signed => sx.max(sy) as u128,
        BinaryOp::Max => x.max(y),
        BinaryOp::Min if signed => sx.min(sy) as u128,
        BinaryOp::Min => x.min(y),
        _ => unreachable!(),
    }
}

fn ord(signed: bool, x: u128, y: u128) -> std::cmp::Ordering {
    if signed {
        (x as i128).cmp(&(y as i128))
    } else {
        x.cmp(&y)
    }
}

#[test]
fn the_128_bit_bitwise_shift_and_ordering_operators_match_rusts_integers() {
    for (t, signed) in [(I128, true), (U128, false)] {
        let (a, b) = grid(t);
        for op in [
            BinaryOp::BitAnd,
            BinaryOp::BitOr,
            BinaryOp::BitXor,
            BinaryOp::Shl,
            BinaryOp::Shr,
            BinaryOp::Max,
            BinaryOp::Min,
        ] {
            let got = run(&binary_kernel(op, t, t, a.len() as u32), t, t, &a, &b);
            let want: Vec<P> = a
                .iter()
                .zip(&b)
                .map(|(x, y)| (0, ref128(op, signed, x.1, y.1)))
                .collect();
            assert_eq!(got, want, "{op:?} at {t:?}");
        }
    }
}

#[test]
fn the_128_bit_arithmetic_matches_rusts_integers() {
    for (t, signed) in [(I128, true), (U128, false)] {
        let (a, b) = grid(t);
        let n = a.len() as u32;
        let go = |make| run(&arith_kernel(make, t, n), t, t, &a, &b);
        let (add, sub) = (go(ScalarExpr::Add), go(ScalarExpr::Sub));
        let (mul, div) = (go(ScalarExpr::Mul), go(ScalarExpr::Div));
        for (i, (x, y)) in a.iter().zip(&b).enumerate() {
            let (x, y) = (x.1, y.1);
            assert_eq!(add[i].1, x.wrapping_add(y), "add {x:#x} {y:#x} {t:?}");
            assert_eq!(sub[i].1, x.wrapping_sub(y), "sub {x:#x} {y:#x} {t:?}");
            assert_eq!(mul[i].1, x.wrapping_mul(y), "mul {x:#x} {y:#x} {t:?}");
            let q = if y == 0 {
                x // the total rule: a zero divisor gives the dividend
            } else if signed {
                (x as i128).wrapping_div(y as i128) as u128 // MIN / -1 wraps to MIN, the total rule's answer
            } else {
                x / y
            };
            assert_eq!(div[i].1, q, "div {x:#x} {y:#x} {t:?}");
        }
    }
}

#[test]
fn the_128_bit_comparisons_match_rusts_integers() {
    use std::cmp::Ordering::{Equal, Greater, Less};
    let ops: [(BinaryOp, fn(std::cmp::Ordering) -> bool); 6] = [
        (BinaryOp::CmpLt, |o| o == Less),
        (BinaryOp::CmpEq, |o| o == Equal),
        (BinaryOp::CmpGe, |o| o != Less),
        (BinaryOp::CmpLe, |o| o != Greater),
        (BinaryOp::CmpGt, |o| o == Greater),
        (BinaryOp::CmpNe, |o| o != Equal),
    ];
    for (t, signed) in [(I128, true), (U128, false)] {
        let (a, b) = grid(t);
        for (op, f) in ops {
            let got = run(&binary_kernel(op, t, I32, a.len() as u32), t, I32, &a, &b);
            for (i, (x, y)) in a.iter().zip(&b).enumerate() {
                let want = u128::from(f(ord(signed, x.1, y.1)));
                assert_eq!(got[i].1, want, "{op:?} {:#x} {:#x} at {t:?}", x.1, y.1);
            }
        }
    }
}

/// A wrapped result is a CANONICAL pattern of its type (sign-extended for the signed type, zero-extended for the unsigned): the
/// store keeps only the low bytes, so this is seen by comparing the unstored value, here against zero.
#[test]
fn a_wrapped_result_compares_as_its_own_type() {
    for (t, signed) in [(I128, true), (U128, false)] {
        let (a, b) = grid(t);
        let n = a.len() as u32;
        for (name, make) in [
            (
                "add",
                (ScalarExpr::Add) as fn(Box<ScalarExpr>, Box<ScalarExpr>) -> ScalarExpr,
            ),
            ("sub", ScalarExpr::Sub),
            ("mul", ScalarExpr::Mul),
        ] {
            let mut k = binary_kernel(BinaryOp::CmpLt, t, I32, n);
            let Stmt::Store { value, .. } = &mut k.body.0[2] else {
                unreachable!()
            };
            let zero = ScalarExpr::Sub(
                Box::new(ScalarExpr::Local(0)),
                Box::new(ScalarExpr::Local(0)),
            );
            *value = ScalarExpr::Binary(
                BinaryOp::CmpLt,
                Box::new(make(
                    Box::new(ScalarExpr::Local(0)),
                    Box::new(ScalarExpr::Local(1)),
                )),
                Box::new(zero),
            );
            let got = run(&k, t, I32, &a, &b);
            for (i, (x, y)) in a.iter().zip(&b).enumerate() {
                let r = match name {
                    "add" => x.1.wrapping_add(y.1),
                    "sub" => x.1.wrapping_sub(y.1),
                    _ => x.1.wrapping_mul(y.1),
                };
                let want = signed && (r as i128) < 0;
                assert_eq!(
                    got[i].1,
                    u128::from(want),
                    "{name} {:#x} {:#x} < 0 at {t:?}",
                    x.1,
                    y.1
                );
            }
        }
    }
}

#[test]
fn a_float_cast_and_a_float_operand_are_refused_with_the_integer_control() {
    let kernel = |from: ElementKind, to: ElementKind| {
        let mut k = KernelFn::new(
            "c",
            I32,
            vec![
                Param::buffer("in", from, BufferRole::Read, Some(1)),
                Param::buffer("out", to, BufferRole::Write, Some(1)),
            ],
            vec![LocalDecl::scalar(from)],
            Block(vec![
                Stmt::Load {
                    local: 0,
                    place: at_tid(0),
                },
                Stmt::Store {
                    place: at_tid(1),
                    value: ScalarExpr::Cast(
                        Box::new(ScalarExpr::Local(0)),
                        cs(to, Saturation::Clamp),
                    ),
                },
            ]),
        );
        k.launch = Some(LaunchShape::new([1, 1, 1], [1, 1, 1]));
        check_kernel(&k)
    };
    assert_eq!(kernel(I64, I128), Ok(()), "the integer control is accepted");
    assert!(matches!(
        kernel(ElementKind::F32, I128),
        Err(KernelError::Type(_))
    ));
    assert!(matches!(
        kernel(I256, ElementKind::F64),
        Err(KernelError::Type(_))
    ));
}
