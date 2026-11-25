// SPDX-License-Identifier: MIT OR Apache-2.0
//! Shared by the `Wgmma` and `Tcgen05Mma` tests: the PTX canonical shared-memory layouts for a 16-bit element, written from
//! PTX ISA 9.5 9.7.17.5.1.2.1.3 independently of `OperandLayout::element_offset` (the thing under test), as a reference
//! offset in Rust and as the IR expression a kernel uses to store a tile in that layout.

use unpopped::ir::{BinaryOp, ScalarExpr};
use unpopped::kernel::{Major, OperandLayout, Swizzle};
use unpopped_vocab::ElementKind;

const K: usize = 16;

fn u(v: i64) -> ScalarExpr {
    ScalarExpr::ConstInt(v, ElementKind::U32)
}
fn add(a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Add(Box::new(a), Box::new(b))
}
fn mul(a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Mul(Box::new(a), Box::new(b))
}

// ---- operand layouts (U8c slice 1): the PTX canonical layouts, written here from PTX ISA 9.5 9.7.17.5.1.2.1.3 ----------
// K-major, no swizzle:  ((8,m),(T,2)):((1T,SBO),(1,LBO))      T = 8 elements (16 bytes of a 16-bit type)
// MN-major, no swizzle: ((T,1,m),(8,k)):((1,T,SBO),(1T,LBO))   (LBO, SBO in bytes in the descriptor: /2 here)

pub(crate) fn k_major(lbo: u32, sbo: u32) -> OperandLayout {
    OperandLayout::Canonical {
        major: Major::K,
        swizzle: Swizzle::None,
        lbo: Some(lbo),
        sbo,
    }
}
pub(crate) fn mn_major(lbo: u32, sbo: u32) -> OperandLayout {
    OperandLayout::Canonical {
        major: Major::MN,
        swizzle: Swizzle::None,
        lbo: Some(lbo),
        sbo,
    }
}

/// The element offset of `(mn, k)` under `layout`, by the formulas above.
pub(crate) fn ref_offset(layout: OperandLayout, mn: usize, k: usize) -> usize {
    match layout {
        OperandLayout::RowMajor => mn * 16 + k,
        OperandLayout::Canonical {
            major, lbo, sbo, ..
        } => {
            let (l, s) = (lbo.unwrap() as usize / 2, sbo as usize / 2);
            match major {
                Major::K => (mn % 8) * 8 + (mn / 8) * s + k % 8 + (k / 8) * l,
                Major::MN => mn % 8 + (mn / 8) * s + (k % 8) * 8 + (k / 8) * l,
            }
        }
    }
}

/// Elements an allocation must hold for `rows` rows under `layout`.
pub(crate) fn span(layout: OperandLayout, rows: usize) -> u32 {
    let last = (0..rows)
        .flat_map(|mn| (0..K).map(move |k| ref_offset(layout, mn, k)))
        .max()
        .unwrap_or(0);
    (last + 1) as u32
}

/// The IR expression for the element offset of flat tile index `i` (row `i / 16`, `k = i % 16`) under `layout`.
pub(crate) fn store_index(layout: OperandLayout, i: ScalarExpr) -> ScalarExpr {
    // powers of two only: `% b` is `& (b - 1)` and `/ b` is `>> log2(b)` (an integer `Rem` has no kernel-layer meaning)
    let rem = |a: ScalarExpr, b: i64| {
        ScalarExpr::Binary(BinaryOp::BitAnd, Box::new(a), Box::new(u(b - 1)))
    };
    let div = |a: ScalarExpr, b: i64| {
        ScalarExpr::Binary(
            BinaryOp::Shr,
            Box::new(a),
            Box::new(u(i64::from(b.trailing_zeros()))),
        )
    };
    let OperandLayout::Canonical {
        major, lbo, sbo, ..
    } = layout
    else {
        return i;
    };
    let (l, s) = (i64::from(lbo.unwrap() / 2), i64::from(sbo / 2));
    let (mn, k) = (div(i.clone(), 16), rem(i, 16));
    let (mn0, mn1) = (rem(mn.clone(), 8), div(mn, 8));
    let (k0, k1) = (rem(k.clone(), 8), div(k, 8));
    match major {
        Major::K => add(add(mul(mn0, u(8)), mul(mn1, u(s))), add(k0, mul(k1, u(l)))),
        Major::MN => add(add(mn0, mul(mn1, u(s))), add(mul(k0, u(8)), mul(k1, u(l)))),
    }
}
