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

/// A swizzled layout. K-major uses no leading offset ("not used, assumed to be 1"); MN-major takes both.
pub(crate) fn k_major_sw(swizzle: Swizzle, sbo: u32) -> OperandLayout {
    OperandLayout::Canonical {
        major: Major::K,
        swizzle,
        lbo: None,
        sbo,
    }
}
pub(crate) fn mn_major_sw(swizzle: Swizzle, lbo: u32, sbo: u32) -> OperandLayout {
    OperandLayout::Canonical {
        major: Major::MN,
        swizzle,
        lbo: Some(lbo),
        sbo,
    }
}

/// `(B, s = 2^B)` of a swizzle: the chunk position of a swizzled tile is, per PTX ISA 9.5 Figures 159-164 (units of
/// 16-byte chunks = 8 elements), `p = s * row + (chunk XOR (row >> (3 - B)))` where `row` is the 8-deep row index within
/// an atom (K-major: `mn % 8`; MN-major: `k % 8`) and `chunk` the chunk index along the contiguous dimension.
fn bits(sw: Swizzle) -> u32 {
    match sw {
        Swizzle::None => 0,
        Swizzle::B32 => 1,
        Swizzle::B64 => 2,
        Swizzle::B128 => 3,
    }
}

/// The element offset of `(mn, k)` under `layout`, by the formulas above and the figure rule for the swizzled ones.
pub(crate) fn ref_offset(layout: OperandLayout, mn: usize, k: usize) -> usize {
    match layout {
        OperandLayout::RowMajor => mn * 16 + k,
        OperandLayout::Canonical {
            major,
            swizzle: Swizzle::None,
            lbo,
            sbo,
        } => {
            let (l, s) = (lbo.unwrap() as usize / 2, sbo as usize / 2);
            match major {
                Major::K => (mn % 8) * 8 + (mn / 8) * s + k % 8 + (k / 8) * l,
                Major::MN => mn % 8 + (mn / 8) * s + (k % 8) * 8 + (k / 8) * l,
            }
        }
        OperandLayout::Canonical {
            major,
            swizzle,
            lbo,
            sbo,
        } => {
            let b = bits(swizzle) as usize;
            let s = 1usize << b;
            let sbo = sbo as usize / 2;
            match major {
                Major::K => {
                    let (row, atom, chunk, e) = (mn % 8, mn / 8, k / 8, k % 8);
                    let p = s * row + (chunk ^ (row >> (3 - b)));
                    atom * sbo + p * 8 + e
                }
                Major::MN => {
                    let lbo = lbo.unwrap() as usize / 2;
                    let (row, kg, chunk, atom, e) =
                        (k % 8, k / 8, (mn / 8) % s, mn / (8 * s), mn % 8);
                    let p = s * row + (chunk ^ (row >> (3 - b)));
                    atom * lbo + kg * sbo + p * 8 + e
                }
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

/// The IR expression for the element offset of flat tile index `i` (row `i / 16`, `k = i % 16`) under `layout`, built
/// from the same figure-rule formulas as [`ref_offset`] (powers of two only: `% b` is `& (b - 1)`, `/ b` is `>> log2 b`).
pub(crate) fn store_index(layout: OperandLayout, i: ScalarExpr) -> ScalarExpr {
    let and =
        |a: ScalarExpr, m: i64| ScalarExpr::Binary(BinaryOp::BitAnd, Box::new(a), Box::new(u(m)));
    let shr = |a: ScalarExpr, n: u32| {
        ScalarExpr::Binary(BinaryOp::Shr, Box::new(a), Box::new(u(i64::from(n))))
    };
    let xor = |a: ScalarExpr, b: ScalarExpr| {
        ScalarExpr::Binary(BinaryOp::BitXor, Box::new(a), Box::new(b))
    };
    let OperandLayout::Canonical {
        major,
        swizzle,
        lbo,
        sbo,
    } = layout
    else {
        return i;
    };
    let sbo = i64::from(sbo / 2);
    let (mn, k) = (shr(i.clone(), 4), and(i, 15));
    if swizzle == Swizzle::None {
        let l = i64::from(lbo.unwrap() / 2);
        let (mn0, mn1) = (and(mn.clone(), 7), shr(mn, 3));
        let (k0, k1) = (and(k.clone(), 7), shr(k, 3));
        return match major {
            Major::K => add(
                add(mul(mn0, u(8)), mul(mn1, u(sbo))),
                add(k0, mul(k1, u(l))),
            ),
            Major::MN => add(
                add(mn0, mul(mn1, u(sbo))),
                add(mul(k0, u(8)), mul(k1, u(l))),
            ),
        };
    }
    let b = bits(swizzle);
    let s = 1i64 << b;
    match major {
        Major::K => {
            let (row, atom, chunk, e) =
                (and(mn.clone(), 7), shr(mn, 3), shr(k.clone(), 3), and(k, 7));
            let p = add(mul(row.clone(), u(s)), xor(chunk, shr(row, 3 - b)));
            add(add(mul(atom, u(sbo)), mul(p, u(8))), e)
        }
        Major::MN => {
            let l = i64::from(lbo.unwrap() / 2);
            let (row, kg) = (and(k.clone(), 7), shr(k, 3));
            let (chunk, atom, e) = (
                and(shr(mn.clone(), 3), s - 1),
                shr(mn.clone(), 3 + b),
                and(mn, 7),
            );
            let p = add(mul(row.clone(), u(s)), xor(chunk, shr(row, 3 - b)));
            add(add(add(mul(atom, u(l)), mul(kg, u(sbo))), mul(p, u(8))), e)
        }
    }
}
