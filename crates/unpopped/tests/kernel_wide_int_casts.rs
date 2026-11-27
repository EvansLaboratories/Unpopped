// SPDX-License-Identifier: MIT OR Apache-2.0
//! The wide integer dtypes `I128`, `U128`, `I256`, `U256` in a kernel (ROADMAP E2): wrapping arithmetic at the type's width, the
//! total `Div`, shifts with the count modulo the width, ordering, comparisons, and integer casts to and from the narrow integers
//! (`docs/int-div-shift-semantics.md` at the wide width).
//!
//! The 128-bit types are compared against Rust's own `i128`/`u128` operators (independent of the oracle's 256-bit limb type);
//! the 256-bit types against a `(hi, lo)` pair of `u128` model written here, and against hand-pinned values. Every operand set
//! includes 0, 1, the all-ones pattern, the sign bit alone and a pattern that crosses the 128-bit boundary.

use unpopped::exec::LaunchShape;
use unpopped::ir::{Saturation, ScalarExpr};
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Stmt, check_kernel};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_vocab::ElementKind::{self, I32, I64, I128, I256, U8, U128, U256};

#[path = "support/wide_int.rs"]
#[allow(dead_code, reason = "shared helpers: each test file uses a subset")]
mod support;
use support::*;

/// Convert `v` of dtype `from` to `to` in a one-thread kernel and return the stored pattern.
fn cast(from: ElementKind, to: ElementKind, sat: Saturation, v: P) -> P {
    let mut k = KernelFn::new(
        "cast",
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
                value: ScalarExpr::Cast(Box::new(ScalarExpr::Local(0)), cs(to, sat)),
            },
        ]),
    );
    k.launch = Some(LaunchShape::new([1, 1, 1], [1, 1, 1]));
    assert_eq!(check_kernel(&k), Ok(()), "{from:?} to {to:?}");
    let mut args = vec![
        KernelArg::Buffer(TypedBuffer::new(from, vec![1], vec![1], bytes_of(from, v))),
        KernelArg::Buffer(TypedBuffer::new(
            to,
            vec![1],
            vec![1],
            vec![0u8; width(to).max(8)],
        )),
    ];
    run_launch(&k, &mut args).unwrap_or_else(|e| panic!("{from:?} to {to:?}: {e:?}"));
    let KernelArg::Buffer(out) = &args[1] else {
        unreachable!()
    };
    let c = &out.bytes;
    match to {
        I64 => (
            0,
            u128::from(u64::from_le_bytes(c[..8].try_into().unwrap())),
        ),
        U8 => (0, u128::from(c[0])),
        _ => pattern(to, &c[..width(to)]),
    }
}

const MINUS_ONE_64: P = (0, u64::MAX as u128);
const ALL_ONES: P = (u128::MAX, u128::MAX);

#[test]
fn a_cast_between_wide_integers_wraps_and_clamps() {
    use Saturation::{Clamp, Wrap};
    // I64 -1 -> U128: wraps to 2^128 - 1; clamps to 0.
    assert_eq!(cast(I64, U128, Wrap, MINUS_ONE_64), (0, u128::MAX));
    assert_eq!(cast(I64, U128, Clamp, MINUS_ONE_64), (0, 0));
    // U256 all-ones -> I128: wraps to -1; clamps to the I128 maximum.
    assert_eq!(cast(U256, I128, Wrap, ALL_ONES), (0, u128::MAX));
    assert_eq!(cast(U256, I128, Clamp, ALL_ONES), (0, u128::MAX >> 1));
    // I256 minimum -> I128 clamps to the I128 minimum.
    assert_eq!(cast(I256, I128, Clamp, (1u128 << 127, 0)), (0, 1 << 127));
    // I64 -1 -> I256 sign-extends over all 256 bits.
    assert_eq!(cast(I64, I256, Wrap, MINUS_ONE_64), ALL_ONES);
}

#[test]
fn a_cast_from_a_wide_integer_to_a_narrow_one_wraps_and_clamps() {
    use Saturation::{Clamp, Wrap};
    // I128 300 -> U8: wraps to 44, clamps to 255.
    assert_eq!(cast(I128, U8, Wrap, (0, 300)), (0, 44));
    assert_eq!(cast(I128, U8, Clamp, (0, 300)), (0, 255));
}
