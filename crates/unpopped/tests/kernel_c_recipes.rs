// SPDX-License-Identifier: MIT OR Apache-2.0
//! **CuTe and ThunderKittens tile views as C-recipes** (`docs/design-tile-and-schedule-plan.md` T3 question, pm-kernels [TASK]
//! `t3-c-recipes`): slice, `local_tile`, `local_partition`, TV-layout composition, `partition_A/B/C`, `partition_S/D` with
//! `retile`, a layout-driven GEMM, predication by identity tensor, subtile views and `swap_layout`, each written WITHOUT any
//! view statement. A view is a `Layout` from `unpopped-vocab` evaluated as ordinary index arithmetic (`lay_expr`: the layout's
//! `(shape, stride)` pairs become `/`, `%`, `*`, `+` over an index expression, so a thread or block index is a runtime
//! coordinate) over a `Shared` allocation. Each result is compared with a hand-written element formula (never with another
//! layout call), and each test has a mutant that must change the VALUE. The two `declines_` tests record what the recipe
//! route cannot say, by name. Survey rows: CuTe 22 to 27, 31, 47; ThunderKittens 6, 38.

use unpopped::exec::{AllocDecl, ExecId, Extent, LaunchShape, MemorySpace, Scope};
use unpopped::ir::ScalarExpr;
use unpopped::kernel::{
    BarrierKind, Block, BufferRole, KernelError, KernelFn, LocalDecl, Param, Place, Stmt, Unroll,
    check_kernel,
};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_vocab::ElementKind::{self, U32};
use unpopped_vocab::layout_algebra::{IntTuple, Layout, SliceCrd};

// ---- index arithmetic ------------------------------------------------------------------------------------------------

fn u(v: i64) -> ScalarExpr {
    ScalarExpr::ConstInt(v, U32)
}
fn l(i: u32) -> ScalarExpr {
    ScalarExpr::Local(i)
}
fn tid() -> ScalarExpr {
    ScalarExpr::Exec(ExecId::ThreadIdx(0))
}
fn blk() -> ScalarExpr {
    ScalarExpr::Exec(ExecId::BlockIdx(0))
}
fn add(a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Add(Box::new(a), Box::new(b))
}
fn mul(a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Mul(Box::new(a), Box::new(b))
}
fn div(a: ScalarExpr, b: i64) -> ScalarExpr {
    ScalarExpr::Div(Box::new(a), Box::new(u(b)))
}
/// `a % b` for integers, as `a - (a / b) * b` (the integer `Div` is the only integer quotient the IR has).
fn rem(a: ScalarExpr, b: i64) -> ScalarExpr {
    ScalarExpr::Sub(Box::new(a.clone()), Box::new(mul(div(a, b), u(b))))
}

fn lay(t: &str) -> Layout {
    t.parse().unwrap_or_else(|e| panic!("{t}: {e}"))
}

/// The layout `l` evaluated at the (runtime) linear index `idx`, colexicographic over the flattened shape: the C-recipe for
/// "apply this layout" (`Layout::call` as arithmetic). The last mode takes the remaining quotient, as `Layout::call` does.
fn lay_expr(layout: &Layout, idx: ScalarExpr) -> ScalarExpr {
    let modes = layout.flat_modes();
    let (mut prod, mut acc) = (1i64, u(0));
    for (k, (s, d)) in modes.iter().enumerate() {
        let mut c = if prod == 1 {
            idx.clone()
        } else {
            div(idx.clone(), prod)
        };
        if k + 1 != modes.len() {
            c = rem(c, *s);
        }
        acc = add(acc, mul(c, u(*d)));
        prod *= s;
    }
    acc
}

/// Mode `k` of `zipped_divide(tilers)`: 0 is the tile (thread) mode, 1 the rest mode.
fn zmode(layout: &Layout, tilers: &[Layout], k: usize) -> Layout {
    layout.zipped_divide(tilers).expect("divisible").modes()[k].clone()
}

// ---- the harness ---------------------------------------------------------------------------------------------------

/// The value the run puts at logical element `i` of the shared array.
fn val(i: i64) -> u32 {
    (i * 7 + 3) as u32
}

fn at(buffer: u32, i: ScalarExpr) -> Place {
    Place::Logical {
        buffer,
        coords: vec![i],
    }
}

fn sh(alloc: u32, index: ScalarExpr) -> Place {
    Place::Alloc { alloc, index }
}

/// `total` elements of `in` copied to allocation 0 by the block's threads (locals 0 and 1 are the loop counter and the value),
/// then a block barrier.
fn fill(total: i64, threads: u32) -> Vec<Stmt> {
    let copy = Block(vec![
        Stmt::Load {
            local: 1,
            place: at(0, l(0)),
        },
        Stmt::Store {
            place: sh(0, l(0)),
            value: l(1),
        },
    ]);
    vec![
        Stmt::For {
            local: 0,
            start: tid(),
            end: u(total),
            step: u(i64::from(threads)),
            unroll: Unroll::None,
            body: copy,
        },
        Stmt::Barrier {
            scope: Scope::Block,
            kind: BarrierKind::Full,
        },
    ]
}

struct Shape {
    threads: u32,
    blocks: u32,
    total: i64,
    locals: Vec<ElementKind>,
    allocs: Vec<AllocDecl>,
}

fn shared(n: u32) -> AllocDecl {
    AllocDecl::new("sh", MemorySpace::Shared, U32, Extent::Static(n))
}

/// A kernel `(in, out, ..extra)`: fill allocation 0 from `in`, then `mid`.
fn kernel(s: Shape, extra: Vec<Param>, mid: Vec<Stmt>) -> KernelFn {
    let mut params = vec![
        Param::buffer("in", U32, BufferRole::Read, Some(1)),
        Param::buffer("out", U32, BufferRole::Write, Some(1)),
    ];
    params.extend(extra);
    let mut body = fill(s.total, s.threads);
    body.extend(mid);
    let locals = s.locals.iter().map(|t| LocalDecl::scalar(*t)).collect();
    KernelFn::new("c_recipe", ElementKind::I32, params, locals, Block(body))
        .with_launch(LaunchShape::new([s.threads, 1, 1], [s.blocks, 1, 1]))
        .with_allocs(s.allocs)
}

fn buf(data: Vec<u32>) -> KernelArg {
    let n = data.len() as i64;
    let bytes = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    KernelArg::Buffer(TypedBuffer::new(U32, vec![n], vec![1], bytes))
}

/// Run on `in[i] = val(i)` (`in_len` of them), `extra` arguments after `out` (`out_len` zeros).
fn run(k: &KernelFn, in_len: i64, out_len: usize, extra: Vec<KernelArg>) -> Vec<u32> {
    assert_eq!(check_kernel(k), Ok(()), "well formed");
    let mut args = vec![buf((0..in_len).map(val).collect()), buf(vec![0; out_len])];
    args.extend(extra);
    run_launch(k, &mut args).unwrap_or_else(|e| panic!("{e:?}"));
    let KernelArg::Buffer(out) = &args[1] else {
        unreachable!()
    };
    out.bytes
        .chunks(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

/// `m` loads of allocation 0 at `idx(j)` into locals `2..2+m`, stored at `out[(block * threads + tid) * m + j]`.
fn gather(threads: u32, m: u32, idx: &dyn Fn(u32) -> ScalarExpr) -> Vec<Stmt> {
    let mut v = Vec::new();
    for j in 0..m {
        v.push(Stmt::Load {
            local: 2 + j,
            place: sh(0, idx(j)),
        });
    }
    let base = mul(
        add(mul(blk(), u(i64::from(threads))), tid()),
        u(i64::from(m)),
    );
    for j in 0..m {
        v.push(Stmt::Store {
            place: at(1, add(base.clone(), u(i64::from(j)))),
            value: l(2 + j),
        });
    }
    v
}

/// The common one-array recipe: `threads` x `blocks`, `m` values per thread at `idx(j)`.
fn recipe(threads: u32, blocks: u32, m: u32, idx: &dyn Fn(u32) -> ScalarExpr) -> KernelFn {
    let s = Shape {
        threads,
        blocks,
        total: 64,
        locals: vec![U32; 2 + m as usize],
        allocs: vec![shared(64).with_layout("(4,16):(16,1)".parse().unwrap())],
    };
    kernel(s, vec![], gather(threads, m, idx))
}

fn run64(k: &KernelFn, out_len: usize) -> Vec<u32> {
    run(k, 64, out_len, vec![])
}

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

/// FINDING for CuTe 47: `local_tile` does NOT refuse a tile that does not divide the extent (4 into 6): it rounds the rest mode
/// up, so the last tile's image leaves the matrix (largest offset 49 in a 36-element view). The vocabulary carries no residue
/// and no predicate, which is why the recipe above builds the identity-tensor mask itself.
#[test]
fn a_local_tile_over_a_residue_leaves_the_extent_and_needs_a_predicate() {
    let six = lay("(6,6):(6,1)");
    let reach = |t: &str, c: [i64; 2]| {
        let (piece, off) = six.local_tile(&[lay(t), lay(t)], &c).unwrap();
        off + piece
            .flat_modes()
            .iter()
            .map(|(s, d)| (s - 1) * d)
            .sum::<i64>()
    };
    assert_eq!(
        reach("4:1", [1, 1]),
        28 + 3 * 6 + 3,
        "last 4-tile ends at 49"
    );
    assert!(reach("4:1", [1, 1]) >= 36, "past the 36 elements");
    assert!(
        reach("3:1", [1, 1]) < 36,
        "control: 3 divides 6 and stays inside"
    );
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
