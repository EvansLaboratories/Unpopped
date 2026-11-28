// SPDX-License-Identifier: MIT OR Apache-2.0
//! **C-recipes, batch 2b** (pm-kernels [TASK] `c-recipes-accepted`): CuTe 3 and 66 (mixed static and dynamic modes), CuTe 15
//! (swizzle and a composed layout over a shared tile), CuTe 30 (a `TiledMma` as thread-value layouts, with a permutation
//! tiler), CuTe 40 (`autovec_copy`, `fill`, `clear`, `axpby`), Slang 25 and 26 (cooperative-matrix arithmetic and clamp modes).
//! Layouts from `unpopped-vocab` evaluated as index arithmetic; no view statement; no `Tile` allocation. Each result is
//! compared with a hand-written reference and has a value-changing mutant. One `declines_` test records what a recipe cannot say.

#[path = "c_recipes/harness.rs"]
mod harness;
use harness::*;
use unpopped::ir::{BinaryOp, ScalarExpr};
use unpopped::kernel::{Block, KernelFn, Param, Stmt};
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped_vocab::ElementKind::U32;
use unpopped_vocab::layout_algebra::{ComposedLayout, Layout, Swizzle};

fn shape(threads: u32, total: i64, locals: usize) -> Shape {
    Shape {
        threads,
        blocks: 1,
        total,
        locals: vec![U32; locals],
        allocs: vec![shared(total as u32)],
    }
}

// ---- CuTe 3 and 66: mixed static and dynamic modes ---------------------------------------------------------------------

/// A view `(m, n):(lda, 1)` whose extents and leading dimension are RUNTIME scalar parameters (locals 2, 3, 4): element
/// `(r, c)` is `sh[r * lda + c]`, thread `t < m * n` reads `(t / n, t % n)`. `stride_n` replaces `lda` (the mutant).
fn dynamic_kernel(stride_n: bool) -> KernelFn {
    let (lda, m, n) = (l(2), l(3), l(4));
    let ld = if stride_n { n.clone() } else { lda };
    let at_rc = add(mul(div_by(tid(), n.clone()), ld), rem_by(tid(), n.clone()));
    let guard = ScalarExpr::Binary(BinaryOp::CmpLt, Box::new(tid()), Box::new(mul(m, n)));
    let body = Block(vec![
        Stmt::Load {
            local: 5,
            place: sh(0, at_rc),
        },
        Stmt::Store {
            place: at(1, tid()),
            value: l(5),
        },
    ]);
    let mid = vec![Stmt::If {
        cond: guard,
        then: body,
        els: Block(vec![]),
    }];
    let extra = ["lda", "m", "n"]
        .iter()
        .zip(2u32..)
        .map(|(name, local)| Param::scalar(*name, U32, local))
        .collect();
    kernel(shape(64, 64, 6), extra, mid)
}

fn div_by(a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Div(Box::new(a), Box::new(b))
}

fn rem_by(a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    sub(a.clone(), mul(div_by(a, b.clone()), b))
}

fn dyn_run(k: &KernelFn, (lda, m, n): (i128, i128, i128)) -> Vec<u32> {
    let args = vec![KernelArg::Int(lda), KernelArg::Int(m), KernelArg::Int(n)];
    run(k, 64, 64, args)
}

#[test]
fn cute_3_66_dynamic_extents_and_leading_dimension() {
    for (lda, m, n) in [(8, 5, 6), (8, 3, 4), (7, 6, 7)] {
        let want: Vec<u32> = (0..64i128)
            .map(|t| {
                if t < m * n {
                    val(((t / n) * lda + t % n) as i64)
                } else {
                    0
                }
            })
            .collect();
        assert_eq!(
            dyn_run(&dynamic_kernel(false), (lda, m, n)),
            want,
            "{lda} {m} {n}"
        );
    }
    let want: Vec<u32> = (0..64i64)
        .map(|t| if t < 20 { val((t / 5) * 8 + t % 5) } else { 0 })
        .collect();
    assert_eq!(dyn_run(&dynamic_kernel(false), (8, 4, 5)), want);
    assert_ne!(
        dyn_run(&dynamic_kernel(true), (8, 4, 5)),
        want,
        "mutant: lda taken from n"
    );
}

/// CuTe 3 and 66, after #536: a mode that names a runtime value is a `DynLayout`, bound at launch to the static `Layout` the algebra
/// takes. (Before #536 the same text was refused; `Layout` itself still refuses a symbolic mode.)
#[test]
fn dyn_layout_binds_runtime_modes_to_a_static_layout() {
    use unpopped_vocab::layout_algebra::DynLayout;
    let names = [("lda", 2), ("m", 3), ("n", 4)];
    let d = DynLayout::parse_with("(m,n):(lda,1)", &names)
        .expect("symbolic modes parse as a DynLayout");
    assert!(
        "(m,n):(lda,1)".parse::<Layout>().is_err(),
        "Layout itself stays static"
    );
    let bound = d.bind(&[(2, 8), (3, 5), (4, 6)]).expect("binds");
    assert_eq!(bound, "(5,6):(8,1)".parse::<Layout>().unwrap());
    let other = d.bind(&[(2, 7), (3, 6), (4, 7)]).expect("binds");
    assert_ne!(bound, other, "a different launch binds a different layout");
    assert!(
        d.bind(&[(2, 8), (3, 5)]).is_err(),
        "an unbound parameter is refused"
    );
}

// ---- CuTe 15: swizzle, composed layout, shared-memory atom ----------------------------------------------------------------

/// `swizzle(x) = x ^ (((x & msk<<(base+shift)) >> shift)` for the `Swizzle<2,1,2>` used here, as bit arithmetic on a runtime
/// index (the recipe for `Swizzle::apply`).
fn swz_expr(x: ScalarExpr) -> ScalarExpr {
    let moved = div(binop(BinaryOp::BitAnd, x.clone(), u(3 << 3)), 4);
    binop(BinaryOp::BitXor, x, moved)
}

fn swizzle_kernel(apply: bool) -> KernelFn {
    let atom = lay("(8,8):(8,1)");
    let idx = |_j: u32| {
        let base = lay_expr(&atom, tid());
        if apply { swz_expr(base) } else { base }
    };
    kernel(shape(64, 64, 3), vec![], gather(64, 1, &idx))
}

#[test]
fn cute_15_swizzled_composed_layout_over_a_shared_atom() {
    let hand = |x: i64| x ^ (((x >> 3) & 3) << 1);
    let want: Vec<u32> = (0..64i64).map(|t| val(hand(8 * (t % 8) + t / 8))).collect();
    assert_eq!(
        run64(&swizzle_kernel(true), 64),
        want,
        "swizzle(row-major(t))"
    );
    assert_ne!(
        run64(&swizzle_kernel(false), 64),
        want,
        "mutant: no swizzle"
    );
    // the vocabulary agrees with the recipe: the composed layout is the swizzle after the layout
    let sw = Swizzle::new(2, 1, 2).unwrap();
    let c = ComposedLayout::new(sw, lay("(8,8):(8,1)"));
    assert!(
        (0..64).all(|t| c.call(t).unwrap() == hand(8 * (t % 8) + t / 8)),
        "ComposedLayout::call"
    );
}

/// A shared-memory atom tiled to a larger shape: the `(4,4):(4,1)` atom repeated over an 8x8 view, atoms ordered column-major.
#[test]
fn cute_15_atom_tiled_to_shape() {
    let tiled = lay("(4,4):(4,1)").tile_to_shape(&[8, 8]).expect("tiles");
    let k = recipe(64, 1, 1, &move |_| lay_expr(&tiled, tid()));
    let want: Vec<u32> = (0..64i64)
        .map(|t| {
            let (r, c) = (t % 8, t / 8);
            val(16 * (r / 4 + 2 * (c / 4)) + 4 * (r % 4) + c % 4)
        })
        .collect();
    assert_eq!(run64(&k, 64), want);
    let plain = recipe(64, 1, 1, &|_| tid());
    assert_ne!(run64(&plain, 64), want, "mutant: untiled");
}

// ---- CuTe 30: a TiledMma as thread-value layouts, with a permutation tiler ---------------------------------------------

/// A 2x2-thread `TiledMma` over a 1x1x1 FMA atom, as three thread-value layouts (thread colex `tm + 2 tn`, value colex):
/// `tv_A` and `tv_B` have a stride-0 mode where a thread coordinate is projected away, `tv_C` the 2x2 share. `A(m,k)` is at
/// `sh[m + 4k]`, `B(n,k)` at `sh[16 + n + 4k]`. `Mode::Permuted` writes `C` through a permuted M tiler; `Mode::Fault` swaps
/// the thread strides of `tv_B` (the mutant).
#[derive(Clone, Copy, PartialEq)]
enum Mode {
    Plain,
    Permuted,
    Fault,
}

fn tiled_mma(mode: Mode) -> KernelFn {
    let tv_a = lay("((2,2),(2,4)):((1,0),(2,4))");
    let tv_b = lay(if mode == Mode::Fault {
        "((2,2),(2,4)):((1,0),(2,4))"
    } else {
        "((2,2),(2,4)):((0,1),(2,4))"
    });
    let mut tv_c = lay("((2,2),(2,2)):((1,4),(2,8))");
    if mode == Mode::Permuted {
        let tiler = lay("((2,2),4):((2,1),4)");
        tv_c = tiler
            .composition(&tv_c)
            .expect("a permutation tiler composes");
    }
    let at_v = |tv: &Layout, v: i64, base: i64| add(u(base), lay_expr(tv, add(tid(), u(4 * v))));
    let mut mid = Vec::new();
    for v in 0..8i64 {
        mid.push(Stmt::Load {
            local: 2 + v as u32,
            place: sh(0, at_v(&tv_a, v, 0)),
        });
        mid.push(Stmt::Load {
            local: 10 + v as u32,
            place: sh(0, at_v(&tv_b, v, 16)),
        });
    }
    for v in 0..4i64 {
        let (vm, vn) = (v % 2, v / 2);
        let dot = (0..4i64).fold(u(0), |acc, k| {
            add(
                acc,
                mul(l((2 + vm + 2 * k) as u32), l((10 + vn + 2 * k) as u32)),
            )
        });
        mid.push(Stmt::Store {
            place: at(1, lay_expr(&tv_c, add(tid(), u(4 * v)))),
            value: dot,
        });
    }
    kernel(shape(4, 32, 18), vec![], mid)
}

#[test]
fn cute_30_tiled_mma_thread_value_layouts_and_a_permutation_tiler() {
    let c = |m: i64, n: i64| {
        (0..4i64).fold(0u32, |s, k| {
            s.wrapping_add(val(m + 4 * k).wrapping_mul(val(16 + n + 4 * k)))
        })
    };
    let plain: Vec<u32> = (0..16i64).map(|i| c(i % 4, i / 4)).collect();
    assert_eq!(run(&tiled_mma(Mode::Plain), 32, 16, vec![]), plain);
    let perm = |m: i64| 2 * (m % 2) + m / 2;
    let permuted: Vec<u32> = (0..16i64)
        .map(|i| {
            let (mp, n) = (i % 4, i / 4);
            let m = (0..4).find(|m| perm(*m) == mp).expect("a permutation");
            c(m, n)
        })
        .collect();
    assert_eq!(run(&tiled_mma(Mode::Permuted), 32, 16, vec![]), permuted);
    assert_ne!(
        run(&tiled_mma(Mode::Fault), 32, 16, vec![]),
        plain,
        "mutant: B's thread projection swapped"
    );
}

// ---- CuTe 40: autovec_copy, fill, clear, axpby --------------------------------------------------------------------------

/// `autovec_copy` picks the widest vector both layouts agree is contiguous (`max_common_vector`); the copy then moves `w`
/// contiguous elements per thread. Source `(8,8):(1,8)`, destination `(8,8):(1,12)` (padded columns): `w` is 8, so each thread
/// copies one column as one vector (`w` contiguous elements from the first destination offset). `w = 16` is the mutant.
fn autovec_kernel(w: i64) -> KernelFn {
    let dst = lay("(8,8):(1,12)");
    let first = mul(tid(), u(w));
    let to = add(lay_expr(&dst, first.clone()), u(0));
    let n = w as u32;
    let mut mid = load_all(2, n, &|j| sh(0, add(first.clone(), u(i64::from(j)))));
    for j in 0..n {
        mid.push(Stmt::Store {
            place: at(1, add(to.clone(), u(i64::from(j)))),
            value: l(2 + j),
        });
    }
    kernel(shape(64 / w as u32, 64, 2 + n as usize), vec![], mid)
}

#[test]
fn cute_40_autovec_copy_width_comes_from_the_layouts() {
    let (src, dst) = (lay("(8,8):(1,8)"), lay("(8,8):(1,12)"));
    let w = src.max_common_vector(&dst).expect("composes");
    assert_eq!(
        w, 8,
        "the padded destination is contiguous for one 8-column"
    );
    let want: Vec<u32> = (0..96i64)
        .map(|o| {
            if o % 12 < 8 {
                val(8 * (o / 12) + o % 12)
            } else {
                0
            }
        })
        .collect();
    assert_eq!(run(&autovec_kernel(w), 64, 96, vec![]), want);
    assert_ne!(
        run(&autovec_kernel(16), 64, 96, vec![]),
        want,
        "mutant: vector of 16 crosses the padding"
    );
}

/// `clear`, `fill` and `axpby` over a shared tensor: `y = 0`, `y = 5`, then `y = 3 x + 2 y`. The final tensor is stored.
/// `clear_only` skips the fill (the mutant: `y` stays 0).
fn axpby_kernel(skip_fill: bool) -> KernelFn {
    let mut mid = vec![Stmt::Store {
        place: sh(1, tid()),
        value: u(0),
    }];
    if !skip_fill {
        mid.push(Stmt::Store {
            place: sh(1, tid()),
            value: u(5),
        });
    }
    mid.extend(load_all(2, 1, &|_| sh(0, tid())));
    mid.extend(load_all(3, 1, &|_| sh(1, tid())));
    mid.extend(store_all(tid(), 1, &|_| {
        add(mul(u(3), l(2)), mul(u(2), l(3)))
    }));
    let mut s = shape(64, 64, 4);
    s.allocs.push(shared(64));
    kernel(s, vec![], mid)
}

#[test]
fn cute_40_clear_fill_axpby() {
    let want: Vec<u32> = (0..64i64).map(|t| 3 * val(t) + 10).collect();
    assert_eq!(run64(&axpby_kernel(false), 64), want);
    assert_ne!(run64(&axpby_kernel(true), 64), want, "mutant: fill skipped");
}

// ---- Slang 25 and 26: cooperative-matrix arithmetic and clamp modes ---------------------------------------------------

/// A 4x4 cooperative matrix per thread (16 locals): `MapElement` `x -> 2x + 1`, a transpose, and a row reduction. Thread `t`
/// stores the transposed map (16 values) then the row sums (4). `keep` skips the transpose (the mutant).
fn coopmat_kernel(keep: bool) -> KernelFn {
    let base = mul(tid(), u(16));
    let mut mid = load_all(2, 16, &|i| sh(0, add(base.clone(), u(i64::from(i)))));
    let m = |r: u32, c: u32| add(mul(u(2), l(2 + 4 * r + c)), u(1));
    let out = mul(tid(), u(20));
    mid.extend(store_all(out.clone(), 16, &|i| {
        let (r, c) = (i % 4, i / 4);
        if keep { m(c, r) } else { m(r, c) }
    }));
    let sums = |i: u32| (0..4).fold(u(0), |a, c| add(a, m(i, c)));
    mid.extend(store_all(add(out, u(16)), 4, &sums));
    kernel(shape(4, 64, 18), vec![], mid)
}

#[test]
fn slang_25_cooperative_matrix_map_transpose_reduce() {
    let want: Vec<u32> = (0..80i64)
        .map(|g| {
            let (t, i) = (g / 20, g % 20);
            let e = |r: i64, c: i64| 2 * val(16 * t + 4 * r + c) + 1;
            if i < 16 {
                e(i / 4, i % 4)
            } else {
                (0..4).map(|c| e(i - 16, c)).fold(0, u32::wrapping_add)
            }
        })
        .collect();
    let got = run64(&coopmat_kernel(true), 80);
    assert_eq!(got, want, "transpose(map(tile)) then row sums");
    assert_ne!(
        run64(&coopmat_kernel(false), 80),
        want,
        "mutant: transpose skipped"
    );
}

/// Clamp modes of a tensor-layout load: an 8x8 tile read from a 6x6 view with `ClampToEdge`, `Zero` and `Wrap`. `mode` 0, 1, 2.
fn clamp_kernel(mode: u32) -> KernelFn {
    let (r, c) = (div(tid(), 8), rem(tid(), 8));
    let at6 = |rr: ScalarExpr, cc: ScalarExpr| add(mul(rr, u(6)), cc);
    let min5 = |x: ScalarExpr| binop(BinaryOp::Min, x, u(5));
    let cmp = |x: ScalarExpr| ScalarExpr::Binary(BinaryOp::CmpLt, Box::new(x), Box::new(u(6)));
    let inside = (cmp(r.clone()), cmp(c.clone()));
    let flat = match mode {
        0 => at6(min5(r), min5(c)),
        1 => at6(min5(r), min5(c)),
        _ => at6(rem(r, 6), rem(c, 6)),
    };
    let mut copy = load_all(2, 1, &|_| sh(0, flat.clone()));
    copy.extend(store_all(tid(), 1, &|_| l(2)));
    let guarded = |cond, then: Vec<Stmt>| Stmt::If {
        cond,
        then: Block(then),
        els: Block(vec![]),
    };
    let mid = if mode == 1 {
        vec![guarded(inside.0, vec![guarded(inside.1, copy)])]
    } else {
        copy
    };
    kernel(shape(64, 36, 3), vec![], mid)
}

#[test]
fn slang_26_clamp_modes_of_a_tensor_layout_load() {
    let cell = |r: i64, c: i64| (r.min(5) * 6 + c.min(5), r < 6 && c < 6, (r % 6) * 6 + c % 6);
    let want = |mode: u32| -> Vec<u32> {
        (0..64i64)
            .map(|t| {
                let (edge, inside, wrap) = cell(t / 8, t % 8);
                match mode {
                    0 => val(edge),
                    1 => {
                        if inside {
                            val(edge)
                        } else {
                            0
                        }
                    }
                    _ => val(wrap),
                }
            })
            .collect()
    };
    for mode in 0..3 {
        assert_eq!(
            run(&clamp_kernel(mode), 36, 64, vec![]),
            want(mode),
            "mode {mode}"
        );
    }
    assert_ne!(
        want(0),
        want(2),
        "control: the clamp and wrap references differ"
    );
    assert_ne!(
        run(&clamp_kernel(2), 36, 64, vec![]),
        want(0),
        "mutant: wrap where clamp was meant"
    );
}
