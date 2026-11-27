// SPDX-License-Identifier: MIT OR Apache-2.0
//! **ThunderKittens register and shared tiles and vectors as C-recipes** (pm-kernels [TASK] `c-recipes-accepted`, batch 2):
//! TK 1 and 2 (a register tile `rt` with a layout, and a swap between row and column layout), TK 3 (register vectors `rv`,
//! `col_vec`/`row_vec` broadcasts), TK 5 (a shared vector `sv` with alignment), TK 39 (a tile-granular cooperative move in a
//! fragment layout), and CuTe 17 (a tensor as engine plus hierarchical layout). A tile is a `Private` allocation or a run of
//! locals plus a `Layout`; a vector is a short run of locals. No view statement and no `Tile` allocation (US12405770B2 hold).
//! Each test has a hand-written reference and a mutant that changes the VALUE.

#[path = "c_recipes/harness.rs"]
mod harness;
use harness::*;
use unpopped::exec::{AllocDecl, Extent, MemorySpace, Scope};
use unpopped::ir::{BinaryOp, ScalarExpr};
use unpopped::kernel::{BarrierKind, Stmt};
use unpopped_vocab::ElementKind::U32;

fn private(n: u32) -> AllocDecl {
    AllocDecl::new("rt", MemorySpace::Private, U32, Extent::Static(n))
}

fn tiles_shape(threads: u32, locals: u32, allocs: Vec<AllocDecl>) -> Shape {
    Shape {
        threads,
        blocks: 1,
        total: 64,
        locals: vec![U32; locals as usize],
        allocs,
    }
}

// ---- TK 1 and 2: a register tile and a layout swap -------------------------------------------------------------------

/// Thread `t` of 4 loads the 4x4 tile at `sh[16 t..]` (row-major) into a register tile `rt1`, swaps it into a column-major
/// register tile `rt2` (storage `r + 4 c`) through the row-major layout `(4,4):(4,1)`, and stores `rt2`. `swap = false` keeps the
/// order (the mutant).
fn swap_kernel(swap: bool) -> unpopped::kernel::KernelFn {
    let rm = lay("(4,4):(4,1)");
    let base = mul(tid(), u(16));
    let mut mid = load_all(2, 16, &|i| sh(0, add(base.clone(), u(i64::from(i)))));
    for i in 0..16u32 {
        mid.push(Stmt::Store {
            place: sh(1, u(i64::from(i))),
            value: l(2 + i),
        });
    }
    // rt2 (column-major storage): rt2[r + 4 c] = rt1[4 r + c]; the mutant keeps rt1's order
    let from = |i: i64| if swap { lay_expr(&rm, u(i)) } else { u(i) };
    mid.extend(load_all(18, 16, &|i| sh(1, from(i64::from(i)))));
    for i in 0..16u32 {
        mid.push(Stmt::Store {
            place: sh(2, u(i64::from(i))),
            value: l(18 + i),
        });
    }
    mid.extend(load_all(34, 16, &|i| sh(2, u(i64::from(i)))));
    mid.extend(store_all(base, 16, &|i| l(34 + i)));
    let allocs = vec![shared(64), private(16), private(16)];
    kernel(tiles_shape(4, 50, allocs), vec![], mid)
}

#[test]
fn tk_1_2_register_tile_with_a_layout_and_a_swap() {
    let want: Vec<u32> = (0..64i64)
        .map(|g| {
            let (t, i) = (g / 16, g % 16);
            let (r, c) = (i % 4, i / 4);
            val(16 * t + 4 * r + c)
        })
        .collect();
    assert_eq!(
        run64(&swap_kernel(true), 64),
        want,
        "column-major storage of the row-major tile"
    );
    assert_ne!(run64(&swap_kernel(false), 64), want, "mutant: no swap");
}

// ---- TK 3: register vectors ------------------------------------------------------------------------------------------

/// A `col_vec` of per-row maxima of a 4x4 register tile, broadcast back: `out(r, c) = rowmax(r) - tile(r, c)`. The vector is
/// four locals. `by_col` is the mutant (maxima over columns instead of rows).
fn rowmax_kernel(by_col: bool) -> unpopped::kernel::KernelFn {
    let base = mul(tid(), u(16));
    let mut mid = load_all(2, 16, &|i| sh(0, add(base.clone(), u(i64::from(i)))));
    let tile = |r: u32, c: u32| l(2 + 4 * r + c);
    let max_of = |xs: Vec<ScalarExpr>| {
        xs.into_iter()
            .reduce(|a, b| binop(BinaryOp::Max, a, b))
            .expect("non-empty")
    };
    for v in 0..4u32 {
        let lane: Vec<ScalarExpr> = (0..4)
            .map(|k| if by_col { tile(k, v) } else { tile(v, k) })
            .collect();
        mid.push(Stmt::Let {
            local: 18 + v,
            init: max_of(lane),
        });
    }
    mid.extend(store_all(base, 16, &|i| {
        sub(l(18 + i / 4), tile(i / 4, i % 4))
    }));
    kernel(tiles_shape(4, 22, vec![shared(64)]), vec![], mid)
}

#[test]
fn tk_3_register_vector_broadcast_down_a_tile() {
    let want: Vec<u32> = (0..64i64)
        .map(|g| {
            let (t, r, c) = (g / 16, (g % 16) / 4, g % 4);
            let row = |k: i64| val(16 * t + 4 * r + k);
            let mx = (0..4).map(row).max().expect("4");
            mx - val(16 * t + 4 * r + c)
        })
        .collect();
    assert_eq!(run64(&rowmax_kernel(false), 64), want);
    assert_ne!(
        run64(&rowmax_kernel(true), 64),
        want,
        "mutant: maxima over the other axis"
    );
}

// ---- TK 5: a shared vector with alignment ------------------------------------------------------------------------------

/// `sv`: 8 elements in a 16-byte-aligned `Shared` array (allocation 1), filled by threads 0..8 from `sh[56..]`, then added to
/// every row of a tile by `out[t] = sh[t] + sv[t % 8]`. `sv[t / 8]` is the mutant.
fn sv_kernel(fault: bool) -> unpopped::kernel::KernelFn {
    let fill_sv = Stmt::If {
        cond: ScalarExpr::Binary(BinaryOp::CmpLt, Box::new(tid()), Box::new(u(8))),
        then: unpopped::kernel::Block(vec![
            Stmt::Load {
                local: 2,
                place: sh(0, add(u(56), tid())),
            },
            Stmt::Store {
                place: sh(1, tid()),
                value: l(2),
            },
        ]),
        els: unpopped::kernel::Block(vec![]),
    };
    let bar = Stmt::Barrier {
        scope: Scope::Block,
        kind: BarrierKind::Full,
    };
    let k = if fault { div(tid(), 8) } else { rem(tid(), 8) };
    let mut mid = vec![fill_sv, bar];
    mid.extend(load_all(3, 1, &|_| sh(0, tid())));
    mid.extend(load_all(4, 1, &|_| sh(1, k.clone())));
    mid.extend(store_all(tid(), 1, &|_| add(l(3), l(4))));
    let sv = AllocDecl::new("sv", MemorySpace::Shared, U32, Extent::Static(8)).with_align_bytes(16);
    kernel(tiles_shape(64, 5, vec![shared(64), sv]), vec![], mid)
}

#[test]
fn tk_5_shared_vector_with_alignment_broadcast_over_rows() {
    let want: Vec<u32> = (0..64i64)
        .map(|t| val(t).wrapping_add(val(56 + t % 8)))
        .collect();
    assert_eq!(run64(&sv_kernel(false), 64), want);
    assert_ne!(
        run64(&sv_kernel(true), 64),
        want,
        "mutant: sv indexed by row"
    );
}

// ---- TK 39: a tile-granular move in the mma fragment layout ------------------------------------------------------------

/// A warp moves a 16x8 row-major tile into the `m16n8` C-fragment layout (lane `l`: rows `l/4` and `l/4 + 8`, columns `2 (l%4)`
/// and `+1`) as a thread-value layout, and stores it transposed (column-major `out[row + 16 col]`). `v1` is the offset of a
/// lane's second row pair: 64 is right, 32 is the mutant.
fn fragment_kernel(v1: i64) -> unpopped::kernel::KernelFn {
    let tv = lay(&format!("((4,8),(2,2)):((2,8),(1,{v1}))"));
    let col = lay("(16,8):(1,16)");
    let coord = |j: u32| add(tid(), u(32 * i64::from(j)));
    let mut mid = load_all(2, 4, &|j| sh(0, lay_expr(&tv, coord(j))));
    let dst = |j: u32| {
        // the element's (row, col) from the TRUE layout, written to out(row, col) of the column-major view
        let (row, c) = fragment_rc(j);
        lay_expr(&col, add(mul(c, u(16)), row)) // idx = row + 16 col over a 128-element col-major frame
    };
    for j in 0..4u32 {
        mid.push(Stmt::Store {
            place: at(1, dst(j)),
            value: l(2 + j),
        });
    }
    let s = Shape {
        threads: 32,
        blocks: 1,
        total: 128,
        locals: vec![U32; 6],
        allocs: vec![shared(128)],
    };
    kernel(s, vec![], mid)
}

/// The (row, col) the C fragment names for this lane's value `j` (the PTX `m16n8` layout, written out by hand).
fn fragment_rc(j: u32) -> (ScalarExpr, ScalarExpr) {
    let (g, tq) = (div(tid(), 4), rem(tid(), 4));
    let row = add(g, u(8 * i64::from(j / 2)));
    (row, add(mul(tq, u(2)), u(i64::from(j % 2))))
}

#[test]
fn tk_39_tile_move_in_the_fragment_layout_stored_transposed() {
    let want: Vec<u32> = (0..128i64).map(|i| val(8 * (i % 16) + i / 16)).collect();
    assert_eq!(run(&fragment_kernel(64), 128, 128, vec![]), want);
    assert_ne!(
        run(&fragment_kernel(32), 128, 128, vec![]),
        want,
        "mutant: second row pair at +4 rows"
    );
}

// ---- CuTe 17: a tensor = engine + hierarchical layout ----------------------------------------------------------------

/// A tensor is (allocation, layout); `copy` walks the tensor's coordinate space in colex order. Here a 16-element tensor with
/// the hierarchical layout `((2,2),(2,2)):((1,4),(2,8))` is copied to a flat one. `flat` is the mutant (the layout dropped).
fn tensor_kernel(flat: bool) -> unpopped::kernel::KernelFn {
    let h = if flat {
        lay("16:1")
    } else {
        lay("((2,2),(2,2)):((1,4),(2,8))")
    };
    let mut mid = load_all(2, 16, &|k| sh(0, lay_expr(&h, u(i64::from(k)))));
    mid.extend(store_all(u(0), 16, &|k| l(2 + k)));
    kernel(tiles_shape(1, 18, vec![shared(64)]), vec![], mid)
}

#[test]
fn cute_17_tensor_with_a_hierarchical_layout() {
    let off = |k: i64| (k & 1) + 4 * ((k >> 1) & 1) + 2 * ((k >> 2) & 1) + 8 * (k >> 3);
    let want: Vec<u32> = (0..16).map(|k| val(off(k))).collect();
    assert_eq!(run64(&tensor_kernel(false), 16), want);
    assert_ne!(run64(&tensor_kernel(true), 16), want, "mutant: flat layout");
}
