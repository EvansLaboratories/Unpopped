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

/// The element offset of `(mn, k)` under `layout` for a 16-bit element (see [`ref_offset_eb`]).
#[allow(
    dead_code,
    reason = "used by some test binaries that include this module and not others"
)]
pub(crate) fn ref_offset(layout: OperandLayout, mn: usize, k: usize) -> usize {
    ref_offset_eb(layout, mn, k, 2)
}

/// The element offset of `(mn, k)` under `layout` for an element of `eb` bytes, by the formulas above and the figure rule
/// for the swizzled ones. A 16-byte core-matrix row holds `t = 16 / eb` elements (PTX: `T = 128 / sizeof_bits`), and `k`
/// runs to `2t`.
pub(crate) fn ref_offset_eb(layout: OperandLayout, mn: usize, k: usize, eb: usize) -> usize {
    let t = 16 / eb;
    match layout {
        OperandLayout::RowMajor => mn * 2 * t + k,
        OperandLayout::Canonical {
            major,
            swizzle: Swizzle::None,
            lbo,
            sbo,
        } => {
            let (l, s) = (lbo.unwrap() as usize / eb, sbo as usize / eb);
            match major {
                Major::K => (mn % 8) * t + (mn / 8) * s + k % t + (k / t) * l,
                Major::MN => mn % t + (mn / t) * s + (k % 8) * t + (k / 8) * l,
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
            let sbo = sbo as usize / eb;
            match major {
                Major::K => {
                    let (row, atom, chunk, e) = (mn % 8, mn / 8, k / t, k % t);
                    let p = s * row + (chunk ^ (row >> (3 - b)));
                    atom * sbo + p * t + e
                }
                Major::MN => {
                    let lbo = lbo.unwrap() as usize / eb;
                    let (row, kg, chunk, atom, e) =
                        (k % 8, k / 8, (mn / t) % s, mn / (t * s), mn % t);
                    let p = s * row + (chunk ^ (row >> (3 - b)));
                    atom * lbo + kg * sbo + p * t + e
                }
            }
        }
    }
}

/// Elements an allocation must hold for `rows` rows under `layout` (16-bit elements).
#[allow(
    dead_code,
    reason = "used by some test binaries that include this module and not others"
)]
pub(crate) fn span(layout: OperandLayout, rows: usize) -> u32 {
    span_eb(layout, rows, 2)
}

/// Elements an allocation must hold for `rows` rows under `layout`, for an element of `eb` bytes.
pub(crate) fn span_eb(layout: OperandLayout, rows: usize, eb: usize) -> u32 {
    let last = (0..rows)
        .flat_map(|mn| (0..K * 2 / eb).map(move |k| ref_offset_eb(layout, mn, k, eb)))
        .max()
        .unwrap_or(0);
    (last + 1) as u32
}

/// [`store_index_eb`] for a 16-bit element.
#[allow(
    dead_code,
    reason = "used by some test binaries that include this module and not others"
)]
pub(crate) fn store_index(layout: OperandLayout, i: ScalarExpr) -> ScalarExpr {
    store_index_eb(layout, i, 2)
}

/// The IR expression for the element offset of flat tile index `i` (row `i / 2t`, `k = i % 2t`, `t = 16 / eb`) under
/// `layout`, built from the same figure-rule formulas as [`ref_offset_eb`] (powers of two only: `% b` is `& (b - 1)`,
/// `/ b` is `>> log2 b`).
pub(crate) fn store_index_eb(layout: OperandLayout, i: ScalarExpr, eb: usize) -> ScalarExpr {
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
    let t = (16 / eb) as i64;
    let lt = t.trailing_zeros();
    let sbo = i64::from(sbo) / eb as i64;
    let (mn, k) = (shr(i.clone(), lt + 1), and(i, 2 * t - 1));
    if swizzle == Swizzle::None {
        let l = i64::from(lbo.unwrap()) / eb as i64;
        return match major {
            Major::K => {
                let (mn0, mn1) = (and(mn.clone(), 7), shr(mn, 3));
                let (k0, k1) = (and(k.clone(), t - 1), shr(k, lt));
                add(
                    add(mul(mn0, u(t)), mul(mn1, u(sbo))),
                    add(k0, mul(k1, u(l))),
                )
            }
            Major::MN => {
                let (mn0, mn1) = (and(mn.clone(), t - 1), shr(mn, lt));
                let (k0, k1) = (and(k.clone(), 7), shr(k, 3));
                add(
                    add(mn0, mul(mn1, u(sbo))),
                    add(mul(k0, u(t)), mul(k1, u(l))),
                )
            }
        };
    }
    let b = bits(swizzle);
    let s = 1i64 << b;
    match major {
        Major::K => {
            let (row, atom, chunk, e) = (
                and(mn.clone(), 7),
                shr(mn, 3),
                shr(k.clone(), lt),
                and(k, t - 1),
            );
            let p = add(mul(row.clone(), u(s)), xor(chunk, shr(row, 3 - b)));
            add(add(mul(atom, u(sbo)), mul(p, u(t))), e)
        }
        Major::MN => {
            let l = i64::from(lbo.unwrap()) / eb as i64;
            let (row, kg) = (and(k.clone(), 7), shr(k, 3));
            let (chunk, atom, e) = (
                and(shr(mn.clone(), lt), s - 1),
                shr(mn.clone(), lt + b),
                and(mn, t - 1),
            );
            let p = add(mul(row.clone(), u(s)), xor(chunk, shr(row, 3 - b)));
            add(add(add(mul(atom, u(l)), mul(kg, u(sbo))), mul(p, u(t))), e)
        }
    }
}
