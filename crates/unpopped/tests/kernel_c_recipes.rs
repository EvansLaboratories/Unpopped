// SPDX-License-Identifier: MIT OR Apache-2.0
//! **CuTe and ThunderKittens tile views as C-recipes** (`docs/design-tile-and-schedule-plan.md` T3 question, pm-kernels [TASK]
//! `t3-c-recipes`): slice, `local_tile`, `local_partition`, TV-layout composition, `partition_A/B/C`, `partition_S/D` with
//! `retile`, a layout-driven GEMM, predication by identity tensor, subtile views and `swap_layout`, each written WITHOUT any
//! view statement. A view is a `Layout` from `unpopped-vocab` evaluated as ordinary index arithmetic (`lay_expr`: the layout's
//! `(shape, stride)` pairs become `/`, `%`, `*`, `+` over an index expression, so a thread or block index is a runtime
//! coordinate) over a `Shared` allocation. Each result is compared with a hand-written element formula (never with another
//! layout call), and each test has a mutant that must change the VALUE. The two `declines_` tests record what the recipe
//! route cannot say, by name. Survey rows here: CuTe 22 to 25 and ThunderKittens 6; the rest are in `kernel_c_recipes_mma`.

#[path = "c_recipes/harness.rs"]
mod harness;
use harness::*;
use unpopped::ir::ScalarExpr;
use unpopped::kernel::{BufferRole, Param, Place, Stmt};
use unpopped_vocab::ElementKind::{self, U32};
use unpopped_vocab::layout_algebra::SliceCrd;

// ---- CuTe 22: slice with `_`, domain offset, recast ----------------------------------------------------------------------

/// Slice a row-major 8x8 view at a STATIC coordinate (the layout and offset come from `Layout::slice`) and at a RUNTIME one
/// (the block index pins mode 0, so the offset is the pinned mode's layout at `blk`).
#[test]
fn cute_22_slice_at_a_static_and_a_runtime_coordinate() {
    let v = lay("(8,8):(8,1)");
    let (row, off) = v
        .slice(&SliceCrd::Tuple(vec![SliceCrd::At(3), SliceCrd::All]))
        .unwrap();
    let k = recipe(8, 1, 1, &|_| add(u(off), lay_expr(&row, tid())));
    let want: Vec<u32> = (0..8).map(|t| val(8 * 3 + t)).collect();
    assert_eq!(run64(&k, 8), want, "row 3, static");
    let (col, off) = v
        .slice(&SliceCrd::Tuple(vec![SliceCrd::All, SliceCrd::At(5)]))
        .unwrap();
    let k = recipe(8, 1, 1, &|_| add(u(off), lay_expr(&col, tid())));
    let want: Vec<u32> = (0..8).map(|t| val(8 * t + 5)).collect();
    assert_eq!(run64(&k, 8), want, "column 5, static");
    let runtime = |fault: bool| {
        let (pin, keep) = if fault { (1, 0) } else { (0, 1) };
        let (pin, keep) = (v.modes()[pin].clone(), v.modes()[keep].clone());
        recipe(8, 8, 1, &move |_| {
            add(lay_expr(&pin, blk()), lay_expr(&keep, tid()))
        })
    };
    let want: Vec<u32> = (0..64).map(|i| val(8 * (i / 8) + i % 8)).collect();
    assert_eq!(run64(&runtime(false), 64), want, "row = block index");
    assert_ne!(
        run64(&runtime(true), 64),
        want,
        "mutant: pins the other mode"
    );
}

/// `recast_ptr`: a `u32` array read as `u16` halves through `AllocRaw` at byte offset `2 * i` (the recast is the byte offset).
#[test]
fn cute_22_recast_reads_u32_storage_as_u16() {
    let kern = |scale: i64| {
        let s = Shape {
            threads: 8,
            blocks: 1,
            total: 64,
            locals: vec![U32, U32, ElementKind::U16],
            allocs: vec![shared(64)],
        };
        let raw = Place::AllocRaw {
            alloc: 0,
            byte_offset: mul(tid(), u(scale)),
            elem: ElementKind::U16,
        };
        let widen = ScalarExpr::Cast(
            Box::new(l(2)),
            unpopped::ir::CastSpec {
                to: U32,
                round: unpopped::ir::RoundMode::NearestEven,
                sat: unpopped::ir::Saturation::Wrap,
            },
        );
        let mid = vec![
            Stmt::Load {
                local: 2,
                place: raw,
            },
            Stmt::Store {
                place: at(1, tid()),
                value: widen,
            },
        ];
        kernel(s, vec![], mid)
    };
    let half = |t: i64| (val(t / 2) >> (16 * (t % 2))) & 0xffff;
    let want: Vec<u32> = (0..8).map(half).collect();
    assert_eq!(run64(&kern(2), 8), want, "u16 element t at byte 2t");
    assert_ne!(
        run64(&kern(4), 8),
        want,
        "mutant: byte offset 4t (a u32 stride)"
    );
}

// ---- CuTe 23 and ThunderKittens 6: local_tile and subtile views ------------------------------------------------------------

/// `local_tile(4x4)` of a row-major 8x8 view, the tile coordinate the RUNTIME block index (colex over the 2x2 rest grid).
#[test]
fn cute_23_local_tile_at_the_block_coordinate() {
    let v = lay("(8,8):(8,1)");
    let t = [lay("4:1"), lay("4:1")];
    let build = |fault: bool| {
        let (tile, rest) = (zmode(&v, &t, 0), zmode(&v, &t, 1));
        let b = if fault {
            add(mul(rem(blk(), 2), u(2)), div(blk(), 2))
        } else {
            blk()
        };
        recipe(16, 4, 1, &move |_| {
            add(lay_expr(&rest, b.clone()), lay_expr(&tile, tid()))
        })
    };
    let want: Vec<u32> = (0..64)
        .map(|g| {
            let (b, t) = (g / 16, g % 16);
            let (bi, bj, p, q) = (b % 2, b / 2, t % 4, t / 4);
            val(8 * (4 * bi + p) + 4 * bj + q)
        })
        .collect();
    assert_eq!(run64(&build(false), 64), want);
    assert_ne!(
        run64(&build(true), 64),
        want,
        "mutant: tile coordinate transposed"
    );
}

/// ThunderKittens `subtile<4,4>(r, c)` at an UNALIGNED origin read from a buffer: not a `local_tile` (tiles are aligned), so the
/// view is the layout `(4,4):(8,1)` built directly plus a runtime offset `8 r0 + c0`.
#[test]
fn tk_6_subtile_at_a_data_dependent_unaligned_origin() {
    let sub = lay("(4,4):(8,1)");
    let build = |fault: bool| {
        let s = Shape {
            threads: 16,
            blocks: 1,
            total: 64,
            locals: vec![U32; 5],
            allocs: vec![shared(64)],
        };
        let (r0, c0) = if fault { (3, 2) } else { (2, 3) };
        let origin = add(mul(l(r0), u(8)), l(c0));
        let mid = vec![
            Stmt::Load {
                local: 2,
                place: at(2, u(0)),
            },
            Stmt::Load {
                local: 3,
                place: at(2, u(1)),
            },
            Stmt::Load {
                local: 4,
                place: sh(0, add(origin, lay_expr(&sub, tid()))),
            },
            Stmt::Store {
                place: at(1, tid()),
                value: l(4),
            },
        ];
        kernel(
            s,
            vec![Param::buffer("org", U32, BufferRole::Read, Some(1))],
            mid,
        )
    };
    let org = || vec![buf(vec![2, 3])];
    let want: Vec<u32> = (0..16).map(|t| val(8 * (2 + t % 4) + 3 + t / 4)).collect();
    assert_eq!(run(&build(false), 64, 16, org()), want);
    assert_ne!(
        run(&build(true), 64, 16, org()),
        want,
        "mutant: origin (3,2)"
    );
}

// ---- CuTe 24: local_partition ------------------------------------------------------------------------------------------

/// `local_partition` over a 4x4 thread grid of a row-major 8x8 view: thread `t` holds the 2x2 strided share
/// `(t%4 + 4r, t/4 + 4s)`. The thread offset is the tile mode at `tid`, the share the rest mode.
#[test]
fn cute_24_local_partition_gives_each_thread_its_strided_share() {
    let v = lay("(8,8):(8,1)");
    let g = [lay("4:1"), lay("4:1")];
    let build = |fault: bool| {
        let (thr, share) = (zmode(&v, &g, 0), zmode(&v, &g, 1));
        let thr_idx = if fault { div(tid(), 2) } else { tid() };
        recipe(16, 1, 4, &move |j| {
            add(
                lay_expr(&thr, thr_idx.clone()),
                lay_expr(&share, u(i64::from(j))),
            )
        })
    };
    let want: Vec<u32> = (0..16)
        .flat_map(|t| (0..4).map(move |j| val(8 * (t % 4 + 4 * (j % 2)) + t / 4 + 4 * (j / 2))))
        .collect();
    assert_eq!(run64(&build(false), 64), want);
    assert_ne!(run64(&build(true), 64), want, "mutant: thread index halved");
}

// ---- CuTe 25: TV layout and composition --------------------------------------------------------------------------------

/// `composition(tensor, tv)` then slice by thread: `tv` maps `(thread, value)` to a coordinate of the 8x8 frame (column-major
/// index `row + 8 col`); composing the row-major view with it gives the offset of every `(thread, value)`.
#[test]
fn cute_25_tv_layout_composed_with_a_tensor() {
    let v = lay("(8,8):(8,1)");
    let tv = lay("((2,4),(2,2)):((1,8),(2,32))");
    let composed = v.composition(&tv).expect("composes");
    let build = |fault: bool| {
        let used = if fault { tv.clone() } else { composed.clone() };
        recipe(8, 1, 4, &move |j| {
            lay_expr(&used, add(tid(), u(8 * i64::from(j))))
        })
    };
    let want: Vec<u32> = (0..8)
        .flat_map(|t| (0..4).map(move |j| val(8 * (t % 2 + 2 * (j % 2)) + t / 2 + 4 * (j / 2))))
        .collect();
    assert_eq!(run64(&build(false), 32), want);
    assert_ne!(
        run64(&build(true), 32),
        want,
        "mutant: tv applied without the tensor"
    );
}
