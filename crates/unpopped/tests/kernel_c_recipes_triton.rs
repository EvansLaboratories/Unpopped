// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Triton rows 2, 13 and 26 as C-recipes** (`docs/ir-union-evidence/rescore-2026-11-25/triton.md` section 2; ruled by the
//! unpopped lane 2026-11-28): block tensors and their shape ops (row 2), `tl.gather` over a register tile (row 13) and
//! `tl.swizzle2d` (row 26). Each is index arithmetic over a `Shared` or `Private` allocation: no statement, no `Tile`
//! allocation (US12405770B2 hold), no tile-granular operation. Every test has a hand-written reference and a mutant that
//! changes the VALUE, and the verdict of each row is PASS.

#[path = "c_recipes/harness.rs"]
mod harness;
use harness::*;
use unpopped::exec::{AllocDecl, Extent, MemorySpace};
use unpopped::ir::{BinaryOp, ScalarExpr};
use unpopped::kernel::{KernelFn, Place, Stmt};
use unpopped_vocab::ElementKind::U32;

fn tdiv(a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Div(Box::new(a), Box::new(b))
}

/// `a % b` with a runtime divisor, as `a - (a / b) * b`.
fn trem(a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    sub(a.clone(), mul(tdiv(a, b.clone()), b))
}

fn lt(a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Binary(BinaryOp::CmpLt, Box::new(a), Box::new(b))
}

fn select(c: ScalarExpr, a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Select(Box::new(c), Box::new(a), Box::new(b))
}

// ---- row 2: block tensors and their shape ops ------------------------------------------------------------------------

/// `out[t] = sh[f(t)]` over 64 elements: a shape op on a block tensor is an index map.
fn index_map(f: ScalarExpr) -> KernelFn {
    recipe(64, 1, 1, &|_| f.clone())
}

fn mapped(k: &KernelFn) -> Vec<u32> {
    run64(k, 64)
}

fn reference(f: impl Fn(i64) -> i64) -> Vec<u32> {
    (0..64).map(|t| val(f(t))).collect()
}

/// `tl.arange(0, 64)` and `tl.full`: the value is the thread index, or a constant, with no load.
fn arange_or_full(value: ScalarExpr) -> KernelFn {
    let s = Shape {
        threads: 64,
        blocks: 1,
        total: 64,
        locals: vec![U32; 2],
        allocs: vec![shared(64)],
    };
    let store = Stmt::Store {
        place: at(1, tid()),
        value,
    };
    kernel(s, vec![], vec![store])
}

#[test]
fn triton_2_arange_and_full() {
    assert_eq!(
        mapped(&arange_or_full(tid())),
        (0..64).collect::<Vec<u32>>(),
        "arange"
    );
    assert_eq!(mapped(&arange_or_full(u(7))), vec![7; 64], "full");
    assert_ne!(
        mapped(&arange_or_full(add(tid(), u(1)))),
        (0..64).collect::<Vec<u32>>(),
        "mutant: arange from 1"
    );
}

#[test]
fn triton_2_flip() {
    let want = reference(|t| 63 - t);
    assert_eq!(mapped(&index_map(sub(u(63), tid()))), want, "flip");
    assert_ne!(mapped(&index_map(tid())), want, "mutant: no flip");
}

/// `tl.cat(a, flip(b))` with `a = in[0..32]`, `b = in[32..64]`: the first half as it is, the second reversed.
#[test]
fn triton_2_cat() {
    let want = reference(|t| if t < 32 { t } else { 95 - t });
    let f = select(lt(tid(), u(32)), tid(), sub(u(95), tid()));
    assert_eq!(mapped(&index_map(f)), want, "cat(a, flip b)");
    let wrong = select(lt(tid(), u(32)), tid(), tid());
    assert_ne!(mapped(&index_map(wrong)), want, "mutant: b not flipped");
}

/// `tl.interleave(a, b)` and `tl.join(a, b)` (a new trailing axis of 2, flattened row-major) are one index map: even slots
/// from `a = in[0..32]`, odd slots from `b = in[32..64]`.
#[test]
fn triton_2_interleave_and_join() {
    let want = reference(|t| (t % 2) * 32 + t / 2);
    let f = add(mul(rem(tid(), 2), u(32)), div(tid(), 2));
    assert_eq!(mapped(&index_map(f)), want, "interleave / join");
    let swapped = add(mul(sub(u(1), rem(tid(), 2)), u(32)), div(tid(), 2));
    assert_ne!(
        mapped(&index_map(swapped)),
        want,
        "mutant: operands swapped"
    );
}

/// `tl.split` of a trailing axis of 2: the even elements first, then the odd ones (the inverse of `join`).
#[test]
fn triton_2_split() {
    let want = reference(|t| t / 32 + 2 * (t % 32));
    let f = add(div(tid(), 32), mul(rem(tid(), 32), u(2)));
    assert_eq!(mapped(&index_map(f)), want, "split");
    let odd_first = add(sub(u(1), div(tid(), 32)), mul(rem(tid(), 32), u(2)));
    assert_ne!(mapped(&index_map(odd_first)), want, "mutant: odd first");
}

// ---- row 13: tl.gather over a register tile --------------------------------------------------------------------------

const ROWS: u32 = 4;
const COLS: u32 = 8;

fn private(n: u32) -> AllocDecl {
    AllocDecl::new("rt", MemorySpace::Private, U32, Extent::Static(n))
}

/// Thread `r` holds row `r` of a 4x8 source tile (`in[0..32]`) in a `Private` allocation and the row's 8 indices
/// (`in[32..64] % 8`); `out[r][k] = src[r][idx[r][k]]`, a gather along the column axis with a computed index. `skew` is the
/// mutant (the index read one column late).
fn gather_kernel(skew: i64) -> KernelFn {
    let row = mul(tid(), u(i64::from(COLS)));
    let mut mid = Vec::new();
    // The register tile first: every element of the row is stored before any is read at a computed index.
    for k in 0..COLS {
        mid.push(Stmt::Load {
            local: 2 + k,
            place: sh(0, add(row.clone(), u(i64::from(k)))),
        });
        mid.push(Stmt::Store {
            place: sh(1, u(i64::from(k))),
            value: l(2 + k),
        });
    }
    for k in 0..COLS {
        let (idx_in, ix, got) = (10 + k, 18 + k, 26 + k);
        mid.push(Stmt::Load {
            local: idx_in,
            place: sh(0, add(u(32), add(row.clone(), u(i64::from(k))))),
        });
        mid.push(Stmt::Let {
            local: ix,
            init: trem(add(l(idx_in), u(skew)), u(i64::from(COLS))),
        });
        mid.push(Stmt::Load {
            local: got,
            place: Place::Alloc {
                alloc: 1,
                index: l(ix),
            },
        });
    }
    mid.extend(store_all(row, COLS, &|k| l(26 + k)));
    let s = Shape {
        threads: ROWS,
        blocks: 1,
        total: 64,
        locals: vec![U32; 34],
        allocs: vec![shared(64), private(COLS)],
    };
    kernel(s, vec![], mid)
}

#[test]
fn triton_13_gather_over_a_register_tile() {
    let (r, c) = (i64::from(ROWS), i64::from(COLS));
    let want: Vec<u32> = (0..r * c)
        .map(|g| {
            let (row, k) = (g / c, g % c);
            let idx = i64::from(val(32 + row * c + k) % c as u32);
            val(row * c + idx)
        })
        .collect();
    assert_eq!(run64(&gather_kernel(0), 32), want, "gather along columns");
    assert_ne!(run64(&gather_kernel(1), 32), want, "mutant: index one late");
}

// ---- row 26: tl.swizzle2d --------------------------------------------------------------------------------------------

const SIZE_I: i64 = 6;
const SIZE_J: i64 = 4;
const SIZE_G: i64 = 4;

/// The reference, as Triton defines it: element `ij` of a `size_i x size_j` row-major grid moves to `(new_i, new_j)` so that
/// groups of `size_g` rows are walked column by column (the last group shorter when `size_i` is not a multiple).
fn swizzle2d_ref(ij: i64) -> i64 {
    let size_gj = SIZE_G * SIZE_J;
    let group = ij / size_gj;
    let off_i = group * SIZE_G;
    let size_g = (SIZE_I - off_i).min(SIZE_G);
    let r = ij % size_gj;
    (off_i + r % size_g) * SIZE_J + r / size_g
}

/// `(off_i + rm % size_g) * size_j + rm / size_g` over the locals `off_i`, `size_g` and `rm`.
fn swizzled_index(off_i: u32, size_g: u32, rm: u32) -> ScalarExpr {
    let new_i = add(l(off_i), trem(l(rm), l(size_g)));
    add(mul(new_i, u(SIZE_J)), tdiv(l(rm), l(size_g)))
}

/// Thread `ij` stores `in[ij]` at `swizzle2d(ij)`. `clamp` false is the mutant: the last group is not shortened, so its
/// elements land on a different, still collision-free, set of offsets (a mutant that collides traps instead of changing a value).
fn swizzle_kernel(clamp: bool) -> KernelFn {
    let ij = tid();
    let (group, off_i, size_g, rm) = (2, 3, 4, 5);
    let stmts = vec![
        Stmt::Load {
            local: 6,
            place: sh(0, ij.clone()),
        },
        Stmt::Let {
            local: group,
            init: div(ij.clone(), SIZE_G * SIZE_J),
        },
        Stmt::Let {
            local: off_i,
            init: mul(l(group), u(SIZE_G)),
        },
        Stmt::Let {
            local: size_g,
            init: if clamp {
                select(
                    lt(sub(u(SIZE_I), l(off_i)), u(SIZE_G)),
                    sub(u(SIZE_I), l(off_i)),
                    u(SIZE_G),
                )
            } else {
                u(SIZE_G)
            },
        },
        Stmt::Let {
            local: rm,
            init: sub(ij, mul(l(group), u(SIZE_G * SIZE_J))),
        },
        Stmt::Store {
            place: at(1, swizzled_index(off_i, size_g, rm)),
            value: l(6),
        },
    ];
    let s = Shape {
        threads: (SIZE_I * SIZE_J) as u32,
        blocks: 1,
        total: SIZE_I * SIZE_J,
        locals: vec![U32; 7],
        allocs: vec![shared(64)],
    };
    kernel(s, vec![], stmts)
}

#[test]
fn triton_26_swizzle2d() {
    let n = SIZE_I * SIZE_J;
    let mut want = vec![0u32; 64];
    for ij in 0..n {
        want[swizzle2d_ref(ij) as usize] = val(ij);
    }
    let got = run(&swizzle_kernel(true), n, 64, vec![]);
    assert_eq!(got, want, "swizzle2d, ragged last group");
    let mut seen: Vec<i64> = (0..n).map(swizzle2d_ref).collect();
    seen.sort_unstable();
    assert_eq!(seen, (0..n).collect::<Vec<_>>(), "a permutation");
    let wrong = run(&swizzle_kernel(false), n, 64, vec![]);
    assert_ne!(wrong, want, "mutant: last group not shortened");
}
