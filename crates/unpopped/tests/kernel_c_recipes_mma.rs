// SPDX-License-Identifier: MIT OR Apache-2.0
//! **CuTe and ThunderKittens tile views as C-recipes** (`docs/design-tile-and-schedule-plan.md` T3 question, pm-kernels [TASK]
//! `t3-c-recipes`): slice, `local_tile`, `local_partition`, TV-layout composition, `partition_A/B/C`, `partition_S/D` with
//! `retile`, a layout-driven GEMM, predication by identity tensor, subtile views and `swap_layout`, each written WITHOUT any
//! view statement. A view is a `Layout` from `unpopped-vocab` evaluated as ordinary index arithmetic (`lay_expr`: the layout's
//! `(shape, stride)` pairs become `/`, `%`, `*`, `+` over an index expression, so a thread or block index is a runtime
//! coordinate) over a `Shared` allocation. Each result is compared with a hand-written element formula (never with another
//! layout call), and each test has a mutant that must change the VALUE. The two `declines_` tests record what the recipe
//! route cannot say, by name. Survey rows here: CuTe 26, 27, 31, 47 and ThunderKittens 38 (harness in `c_recipes/harness.rs`).

#[path = "c_recipes/harness.rs"]
mod harness;
use harness::*;
use unpopped::exec::{AllocDecl, Extent, LaunchShape, MemorySpace, Scope};
use unpopped::ir::ScalarExpr;
use unpopped::kernel::{BarrierKind, Block, KernelError, KernelFn, LocalDecl, Stmt, check_kernel};
use unpopped_vocab::ElementKind::{self, U32};
use unpopped_vocab::layout_algebra::{IntTuple, Layout};

// ---- CuTe 26 and 31: partition_A/B/C and a GEMM over the partitions ----------------------------------------------------

/// A 2x2 thread grid over `C(4,4) += A(4,4) B(4,4)^T`; `A(m,k)` at `sh[m + 4k]`, `B(n,k)` at `sh[16 + n + 4k]`. `partition_A`
/// keeps M, takes all of K (the K tiler is `1:1`, the "projection": the thread coordinate is `tm` alone); `partition_B` likewise
/// with `tn`; `partition_C` is the 2x2 share.
struct Mma {
    pa: (Layout, Layout),
    pb: (Layout, Layout),
    pc: (Layout, Layout),
}

fn mma_parts() -> Mma {
    let a = lay("(4,4):(1,4)");
    let side = [lay("2:1"), lay("1:1")];
    let both = [lay("2:1"), lay("2:1")];
    let part = |x: &Layout, t: &[Layout]| (zmode(x, t, 0), zmode(x, t, 1));
    Mma {
        pa: part(&a, &side),
        pb: part(&a, &side),
        pc: part(&a, &both),
    }
}

fn mma_shape(locals: u32) -> Shape {
    Shape {
        threads: 4,
        blocks: 1,
        total: 32,
        locals: vec![U32; locals as usize],
        allocs: vec![shared(32)],
    }
}

/// `(tm, tn)` of this thread, with `fault` swapping which one drives the B partition.
fn thread_mn(fault: bool) -> (ScalarExpr, ScalarExpr) {
    let (tm, tn) = (rem(tid(), 2), div(tid(), 2));
    if fault { (tm.clone(), tm) } else { (tm, tn) }
}

/// Element `(i, k)` of `A`'s partition for thread coordinate `tm`: `sh[off(tm) + rest(i + 2k)]`.
fn part_idx(p: &(Layout, Layout), t: ScalarExpr, v: i64, base: i64) -> ScalarExpr {
    add(u(base), add(lay_expr(&p.0, t), lay_expr(&p.1, u(v))))
}

fn partition_dump(fault: bool) -> KernelFn {
    let m = mma_parts();
    let (tm, tn) = thread_mn(fault);
    let mut mid = Vec::new();
    for q in 0..8i64 {
        mid.push(Stmt::Load {
            local: 2 + q as u32,
            place: sh(0, part_idx(&m.pa, tm.clone(), q, 0)),
        });
        mid.push(Stmt::Load {
            local: 10 + q as u32,
            place: sh(0, part_idx(&m.pb, tn.clone(), q, 16)),
        });
    }
    for q in 0..8u32 {
        for (which, local) in [(0i64, 2 + q), (8, 10 + q)] {
            mid.push(Stmt::Store {
                place: at(1, add(mul(tid(), u(16)), u(which + i64::from(q)))),
                value: l(local),
            });
        }
    }
    kernel(mma_shape(18), vec![], mid)
}

#[test]
fn cute_26_partition_a_b_c_per_thread_shares() {
    let want: Vec<u32> = (0..4)
        .flat_map(|t: i64| {
            let (tm, tn) = (t % 2, t / 2);
            let a = (0..8).map(move |q| val((tm + 2 * (q % 2)) + 4 * (q / 2)));
            let b = (0..8).map(move |q| val(16 + (tn + 2 * (q % 2)) + 4 * (q / 2)));
            a.chain(b)
        })
        .collect();
    assert_eq!(run(&partition_dump(false), 32, 64, vec![]), want);
    assert_ne!(
        run(&partition_dump(true), 32, 64, vec![]),
        want,
        "mutant: B uses tm"
    );
}

fn gemm(fault: bool) -> KernelFn {
    let m = mma_parts();
    let (tm, tn) = thread_mn(fault);
    let mut mid = Vec::new();
    for i in 0..2i64 {
        for k in 0..4i64 {
            mid.push(Stmt::Load {
                local: (2 + i + 2 * k) as u32,
                place: sh(0, part_idx(&m.pa, tm.clone(), i + 2 * k, 0)),
            });
            mid.push(Stmt::Load {
                local: (10 + i + 2 * k) as u32,
                place: sh(0, part_idx(&m.pb, tn.clone(), i + 2 * k, 16)),
            });
        }
    }
    for j in 0..2i64 {
        for i in 0..2i64 {
            let dot = (0..4i64).fold(u(0), |acc, k| {
                add(
                    acc,
                    mul(l((2 + i + 2 * k) as u32), l((10 + j + 2 * k) as u32)),
                )
            });
            let at_c = part_idx(&m.pc, rem(tid(), 4), i + 2 * j, 0);
            mid.push(Stmt::Store {
                place: at(1, at_c),
                value: dot,
            });
        }
    }
    kernel(mma_shape(18), vec![], mid)
}

#[test]
fn cute_31_gemm_over_the_partitions() {
    let want: Vec<u32> = (0..16i64)
        .map(|c| {
            let (mm, nn) = (c % 4, c / 4);
            (0..4i64).fold(0u32, |s, k| {
                s.wrapping_add(val(mm + 4 * k).wrapping_mul(val(16 + nn + 4 * k)))
            })
        })
        .collect();
    assert_eq!(run(&gemm(false), 32, 16, vec![]), want);
    assert_ne!(
        run(&gemm(true), 32, 16, vec![]),
        want,
        "mutant: B partition by tm"
    );
}

// ---- CuTe 27: partition_S / partition_D and retile ---------------------------------------------------------------------

/// A copy atom moves a thread's 2x2 share from a row-major 8x8 source to a column-major destination through a register
/// fragment stored row-major (`(2,2):(2,1)`, a `Private` allocation); the destination reads it through `retile`.
fn copy_kernel(fault: bool) -> KernelFn {
    let (src, dst) = (lay("(8,8):(8,1)"), lay("(8,8):(1,8)"));
    let g = [lay("4:1"), lay("4:1")];
    let frag = lay("(2,2):(2,1)");
    let retiled = frag.retile(&IntTuple::ints([2, 2])).expect("same size");
    let (s_thr, s_rest) = (zmode(&src, &g, 0), zmode(&src, &g, 1));
    let (d_thr, d_rest) = (zmode(&dst, &g, 0), zmode(&dst, &g, 1));
    let mut mid = Vec::new();
    for v in 0..4i64 {
        let from = add(lay_expr(&s_thr, tid()), lay_expr(&s_rest, u(v)));
        mid.push(Stmt::Load {
            local: 2 + v as u32,
            place: sh(0, from),
        });
        mid.push(Stmt::Store {
            place: sh(1, lay_expr(&frag, u(v))),
            value: l(2 + v as u32),
        });
    }
    for d in 0..4i64 {
        let read = if fault {
            u(d)
        } else {
            lay_expr(&retiled, u(d))
        };
        mid.push(Stmt::Load {
            local: 6 + d as u32,
            place: sh(1, read),
        });
        let to = add(lay_expr(&d_thr, tid()), lay_expr(&d_rest, u(d)));
        mid.push(Stmt::Store {
            place: at(1, to),
            value: l(6 + d as u32),
        });
    }
    let frag_alloc = AllocDecl::new("frag", MemorySpace::Private, U32, Extent::Static(4));
    let s = Shape {
        threads: 16,
        blocks: 1,
        total: 64,
        locals: vec![U32; 10],
        allocs: vec![shared(64), frag_alloc],
    };
    kernel(s, vec![], mid)
}

#[test]
fn cute_27_partition_s_d_with_retile_through_a_register_fragment() {
    let want: Vec<u32> = (0..64i64).map(|i| val(8 * (i % 8) + i / 8)).collect();
    assert_eq!(run64(&copy_kernel(false), 64), want, "out = in transposed");
    assert_ne!(run64(&copy_kernel(true), 64), want, "mutant: no retile");
}

// ---- CuTe 47: predication by an identity tensor over a residue ---------------------------------------------------------

/// A 6x6 row-major matrix covered by 4x4 tiles: 6 is not a multiple of 4, so the tiling is over the PADDED 8x8 coordinate frame
/// and an identity-tensor predicate `row < 6 && col < `bound`` masks the residue. `bound = 5` is the mutant.
fn residue_kernel(bound: i64) -> KernelFn {
    let frame = lay("(8,8):(1,8)");
    let t = [lay("4:1"), lay("4:1")];
    let e = add(
        lay_expr(&zmode(&frame, &t, 1), blk()),
        lay_expr(&zmode(&frame, &t, 0), tid()),
    );
    let (r, c) = (rem(e.clone(), 8), div(e, 8));
    let flat = add(mul(r.clone(), u(6)), c.clone());
    let cmp = |x: ScalarExpr, n: i64| {
        ScalarExpr::Binary(unpopped::ir::BinaryOp::CmpLt, Box::new(x), Box::new(u(n)))
    };
    let copy = Block(vec![
        Stmt::Load {
            local: 2,
            place: sh(0, flat.clone()),
        },
        Stmt::Store {
            place: at(1, flat),
            value: l(2),
        },
    ]);
    let inner = Stmt::If {
        cond: cmp(c, bound),
        then: copy,
        els: Block(vec![]),
    };
    let outer = Stmt::If {
        cond: cmp(r, 6),
        then: Block(vec![inner]),
        els: Block(vec![]),
    };
    let s = Shape {
        threads: 16,
        blocks: 4,
        total: 36,
        locals: vec![U32; 3],
        allocs: vec![shared(36)],
    };
    kernel(s, vec![], vec![outer])
}

#[test]
fn cute_47_identity_tensor_predicate_masks_the_residue() {
    let want: Vec<u32> = (0..36).map(val).collect();
    assert_eq!(
        run(&residue_kernel(6), 36, 36, vec![]),
        want,
        "every element once"
    );
    assert_ne!(
        run(&residue_kernel(5), 36, 36, vec![]),
        want,
        "mutant: column 5 dropped"
    );
}

/// CuTe 47, flipped by the NotDivisible fix: `local_tile` REFUSES a tile that does not divide the extent (4 into 6) instead of
/// rounding the rest mode up (it used to end at offset 49 in a 36-element view). The vocabulary carries no residue and no
/// predicate, which is why the recipe above builds the identity-tensor mask itself.
#[test]
fn a_local_tile_over_a_residue_is_refused_and_a_dividing_tile_stays_inside() {
    let six = lay("(6,6):(6,1)");
    assert_eq!(
        six.local_tile(&[lay("4:1"), lay("4:1")], &[1, 1]),
        Err(unpopped_vocab::layout_algebra::LayoutError::NotDivisible)
    );
    let (piece, off) = six.local_tile(&[lay("3:1"), lay("3:1")], &[1, 1]).unwrap();
    let reach = off
        + piece
            .flat_modes()
            .iter()
            .map(|(s, d)| (s - 1) * d)
            .sum::<i64>();
    assert!(reach < 36, "control: 3 divides 6 and stays inside");
}

// ---- ThunderKittens 38: swap_layout between addressable layouts; declined on an opaque tile -----------------------------

fn swap_kernel(swap: bool) -> KernelFn {
    let (row, col) = (lay("(8,8):(8,1)"), lay("(8,8):(1,8)"));
    let from = if swap {
        lay_expr(&row, tid())
    } else {
        lay_expr(&col, tid())
    };
    let dst = shared(64).with_layout("(8,8):(8,1)".parse().unwrap());
    let mid = vec![
        Stmt::Load {
            local: 2,
            place: sh(0, from),
        },
        Stmt::Store {
            place: sh(1, lay_expr(&col, tid())),
            value: l(2),
        },
        Stmt::Barrier {
            scope: Scope::Block,
            kind: BarrierKind::Full,
        },
        Stmt::Load {
            local: 3,
            place: sh(1, tid()),
        },
        Stmt::Store {
            place: at(1, tid()),
            value: l(3),
        },
    ];
    let s = Shape {
        threads: 64,
        blocks: 1,
        total: 64,
        locals: vec![U32; 4],
        allocs: vec![shared(64), dst],
    };
    kernel(s, vec![], mid)
}

#[test]
fn tk_38_swap_layout_between_two_shared_arrays() {
    let want: Vec<u32> = (0..64i64).map(|t| val(8 * (t % 8) + t / 8)).collect();
    assert_eq!(
        run64(&swap_kernel(true), 64),
        want,
        "row-major in, column-major out"
    );
    assert_ne!(run64(&swap_kernel(false), 64), want, "mutant: no swap");
}

/// The recipe route has no element to index on an opaque tile (`design-opaque-tile-mma.md`): an in-register layout swap of a
/// `Tile` allocation is declined by the checker, by name.
#[test]
fn declines_an_element_swap_of_an_opaque_tile_by_name() {
    let tile = AllocDecl::tile("t", U32, 4, 4);
    let load = Stmt::Load {
        local: 0,
        place: sh(0, u(0)),
    };
    let k = KernelFn::new(
        "tile_elem",
        ElementKind::I32,
        vec![],
        vec![LocalDecl::scalar(U32)],
        Block(vec![load]),
    )
    .with_launch(LaunchShape::new([1, 1, 1], [1, 1, 1]))
    .with_allocs(vec![tile]);
    assert!(
        matches!(check_kernel(&k), Err(KernelError::Tile { .. })),
        "{:?}",
        check_kernel(&k)
    );
}
