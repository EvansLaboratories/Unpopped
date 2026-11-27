// SPDX-License-Identifier: MIT OR Apache-2.0
//! The wide integer dtypes `I128`, `U128`, `I256`, `U256` in a kernel (ROADMAP E2): wrapping arithmetic at the type's width, the
//! total `Div`, shifts with the count modulo the width, ordering, comparisons, and integer casts to and from the narrow integers
//! (`docs/int-div-shift-semantics.md` at the wide width).
//!
//! The 128-bit types are compared against Rust's own `i128`/`u128` operators (independent of the oracle's 256-bit limb type);
//! the 256-bit types against a `(hi, lo)` pair of `u128` model written here, and against hand-pinned values. Every operand set
//! includes 0, 1, the all-ones pattern, the sign bit alone and a pattern that crosses the 128-bit boundary.

use unpopped::ir::{BinaryOp, ScalarExpr};
use unpopped_vocab::ElementKind::{self, I32, I256, U256};

#[path = "support/wide_int.rs"]
#[allow(dead_code, reason = "shared helpers: each test file uses a subset")]
mod support;
use support::*;

// ---- 256-bit reference: a (hi, lo) pair of u128 -----------------------------------------------------------------------

fn add256(a: P, b: P) -> P {
    let (lo, c) = a.1.overflowing_add(b.1);
    (a.0.wrapping_add(b.0).wrapping_add(u128::from(c)), lo)
}

fn not256(a: P) -> P {
    (!a.0, !a.1)
}

fn neg256(a: P) -> P {
    add256(not256(a), (0, 1))
}

fn shl256(a: P, n: u32) -> P {
    match n {
        0 => a,
        1..=127 => ((a.0 << n) | (a.1 >> (128 - n)), a.1 << n),
        128 => (a.1, 0),
        _ => (a.1 << (n - 128), 0),
    }
}

fn shr256(a: P, n: u32, arith: bool) -> P {
    let fill = if arith && (a.0 >> 127) == 1 {
        u128::MAX
    } else {
        0
    };
    match n {
        0 => a,
        1..=127 => (
            (a.0 >> n) | if fill != 0 { fill << (128 - n) } else { 0 },
            (a.1 >> n) | (a.0 << (128 - n)),
        ),
        128 => (fill, a.0),
        _ => (
            fill,
            (a.0 >> (n - 128))
                | if fill != 0 && n > 128 {
                    fill << (256 - n)
                } else {
                    0
                },
        ),
    }
}

fn cmp256(a: P, b: P, signed: bool) -> std::cmp::Ordering {
    if signed {
        ((a.0 as i128), a.1).cmp(&((b.0 as i128), b.1))
    } else {
        a.cmp(&b)
    }
}

fn grid_model(t: ElementKind) -> (Vec<P>, Vec<P>, u32, bool) {
    let (a, b) = grid(t);
    let n = a.len() as u32;
    (a, b, n, t == I256)
}

fn check(t: ElementKind, op: BinaryOp, a: &[P], b: &[P], f: &dyn Fn(P, P) -> P) {
    let n = a.len() as u32;
    let got = run(&binary_kernel(op, t, t, n), t, t, a, b);
    let want: Vec<P> = a.iter().zip(b).map(|(x, y)| f(*x, *y)).collect();
    assert_eq!(got, want, "{op:?} at {t:?}");
}

#[test]
fn the_256_bit_add_and_sub_match_the_limb_pair_model() {
    for t in [I256, U256] {
        let (a, b, n, _) = grid_model(t);
        let go = |make| run(&arith_kernel(make, t, n), t, t, &a, &b);
        let want = |f: &dyn Fn(P, P) -> P| -> Vec<P> {
            a.iter().zip(&b).map(|(x, y)| f(*x, *y)).collect()
        };
        assert_eq!(go(ScalarExpr::Add), want(&add256), "add at {t:?}");
        assert_eq!(
            go(ScalarExpr::Sub),
            want(&|x, y| add256(x, neg256(y))),
            "sub at {t:?}"
        );
    }
}

#[test]
fn the_256_bit_bitwise_shift_and_max_match_the_limb_pair_model() {
    for t in [I256, U256] {
        let signed = t == I256;
        let (a, b, _, _) = grid_model(t);
        check(t, BinaryOp::BitAnd, &a, &b, &|x, y| (x.0 & y.0, x.1 & y.1));
        check(t, BinaryOp::BitXor, &a, &b, &|x, y| (x.0 ^ y.0, x.1 ^ y.1));
        check(t, BinaryOp::Shl, &a, &b, &|x, y| {
            shl256(x, (y.1 & 255) as u32)
        });
        check(t, BinaryOp::Shr, &a, &b, &|x, y| {
            shr256(x, (y.1 & 255) as u32, signed)
        });
        check(t, BinaryOp::Max, &a, &b, &|x, y| {
            if cmp256(x, y, signed) == std::cmp::Ordering::Less {
                y
            } else {
                x
            }
        });
    }
}

#[test]
fn the_256_bit_comparisons_match_the_limb_pair_model() {
    use std::cmp::Ordering::{Equal, Greater, Less};
    let ops: [(BinaryOp, fn(std::cmp::Ordering) -> bool); 3] = [
        (BinaryOp::CmpLt, |o| o == Less),
        (BinaryOp::CmpLe, |o| o != Greater),
        (BinaryOp::CmpNe, |o| o != Equal),
    ];
    for t in [I256, U256] {
        let signed = t == I256;
        let (a, b, n, _) = grid_model(t);
        for (op, f) in ops {
            let got = run(&binary_kernel(op, t, I32, n), t, I32, &a, &b);
            for (i, (x, y)) in a.iter().zip(&b).enumerate() {
                let want = u128::from(f(cmp256(*x, *y, signed)));
                assert_eq!(got[i].1, want, "{op:?} {x:x?} {y:x?} at {t:?}");
            }
        }
    }
}

#[test]
fn the_256_bit_product_and_quotient_are_pinned_by_hand() {
    let (x, one) = ((1u128, 0u128), (0u128, 1u128)); // 2^128 and 1
    let mul = |t, a: P, b: P| run(&arith_kernel(ScalarExpr::Mul, t, 1), t, t, &[a], &[b])[0];
    let div = |t, a: P, b: P| run(&arith_kernel(ScalarExpr::Div, t, 1), t, t, &[a], &[b])[0];
    for t in [I256, U256] {
        assert_eq!(mul(t, x, x), (0, 0), "2^128 * 2^128 wraps to 0 at 256 bits");
        assert_eq!(mul(t, x, one), x);
        assert_eq!(
            mul(t, (0, u128::MAX), (0, u128::MAX)),
            (u128::MAX - 1, 1),
            "(2^128-1)^2 = 2^256 - 2^129 + 1"
        );
        assert_eq!(div(t, x, one), x);
        assert_eq!(div(t, x, (0, 2)), (0, 1 << 127), "2^128 / 2");
        assert_eq!(
            div(t, (5, 7), (0, 0)),
            (5, 7),
            "a zero divisor gives the dividend"
        );
    }
    let all_ones = (u128::MAX, u128::MAX);
    assert_eq!(
        div(U256, all_ones, (0, 2)),
        (u128::MAX >> 1, u128::MAX),
        "2^256-1 / 2 as unsigned"
    );
    assert_eq!(
        div(I256, all_ones, (0, 2)),
        (0, 0),
        "-1 / 2 truncates toward zero"
    );
    let min = (1u128 << 127, 0u128);
    assert_eq!(div(I256, min, all_ones), min, "MIN / -1 gives MIN");
}
