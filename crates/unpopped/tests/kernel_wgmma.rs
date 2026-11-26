// SPDX-License-Identifier: MIT OR Apache-2.0
//! **`Stmt::Wgmma` in the launch interpreter** (ROADMAP U7 slice S4, option A: `docs/options-u7-mma-atoms.md`). A
//! warpgroup (4 warps, 128 threads) multiplies a 64 x 16 `A` tile by an `N x 16` `B` tile (both K-major, row-major in
//! `Shared`, no swizzle, no descriptor bits) into a `64 x N` accumulator spread over the 128 threads' `Private` fragments.
//! The reference product and the `(thread, value) -> (row, col)` fragment layout are written in this test from the PTX
//! `wgmma.m64nNk16` description, independent of the oracle. The operation is asynchronous: its result lands at the
//! `WgmmaWait` that covers its committed group. Each behaviour has a program that shows it and a control that does not.

use unpopped::exec::{AllocDecl, ExecId, Extent, LaunchShape, MemorySpace, Scope};
use unpopped::ir::{BinaryOp, ScalarExpr};
use unpopped::kernel::{
    Block, BufferRole, KernelError, KernelFn, LocalDecl, Major, OperandLayout, Param, Place, Stmt,
    Swizzle, check_kernel,
};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::{KernelArg, KernelTrap};
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_vocab::ElementKind;

#[path = "support/operand_layout.rs"]
mod operand_layout;
use operand_layout::{k_major, k_major_sw, mn_major, mn_major_sw, span, store_index};

const U32: ElementKind = ElementKind::U32;
const BF16: ElementKind = ElementKind::Bf16;
const F32: ElementKind = ElementKind::F32;
const K: usize = 16;
const INIT: f32 = 100.0;

fn u(v: i64) -> ScalarExpr {
    ScalarExpr::ConstInt(v, U32)
}
fn local(i: u32) -> ScalarExpr {
    ScalarExpr::Local(i)
}
fn add(a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Add(Box::new(a), Box::new(b))
}
fn mul(a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Mul(Box::new(a), Box::new(b))
}
fn tid() -> ScalarExpr {
    ScalarExpr::Exec(ExecId::ThreadIdx(0))
}
fn buf(buffer: u32, i: ScalarExpr) -> Place {
    Place::Logical {
        buffer,
        coords: vec![i],
    }
}
fn alloc(a: u32, i: ScalarExpr) -> Place {
    Place::Alloc { alloc: a, index: i }
}
fn counted(var: u32, end: i64, body: Vec<Stmt>) -> Stmt {
    Stmt::For {
        unroll: unpopped::kernel::Unroll::None,
        local: var,
        start: u(0),
        end: u(end),
        step: u(1),
        body: Block(body),
    }
}

/// How a scenario differs from the plain one.
#[derive(Clone, Copy)]
struct Opt {
    n: u32,
    scale_d: i64,
    commit: bool,
    wait: u32,
    barrier: bool,
    /// Each thread issues `fence.proxy.async` after storing its share of a tile.
    fence: bool,
    /// How the tiles sit in their allocations (the stores below use the layout formulas written in this file).
    a_layout: OperandLayout,
    b_layout: OperandLayout,
    /// `A` comes from each thread's register fragment (alloc 3), filled by the formulas of PTX Figure 151.
    a_regs: bool,
}
impl Opt {
    fn plain(n: u32) -> Self {
        Opt {
            n,
            scale_d: 0,
            commit: true,
            wait: 0,
            barrier: true,
            fence: true,
            a_layout: OperandLayout::RowMajor,
            b_layout: OperandLayout::RowMajor,
            a_regs: false,
        }
    }
}

/// Each thread copies `per` elements of input buffer `src` into `Shared` allocation `dst`.
fn copy_tile(src: u32, dst: u32, per: i64, fence: bool, layout: OperandLayout) -> Stmt {
    counted(
        0,
        per,
        vec![
            Stmt::Let {
                local: 1,
                init: add(mul(tid(), u(per)), local(0)),
            },
            Stmt::Load {
                local: 2,
                place: buf(src, local(1)),
            },
            Stmt::Store {
                place: alloc(dst, store_index(layout, local(1))),
                value: local(2),
            },
        ]
        .into_iter()
        .chain(fence.then_some(Stmt::ProxyFence {
            space: MemorySpace::Shared,
        }))
        .collect(),
    )
}

/// **Second-source check of the layout below (2026-11-25).** The layout was read from PTX ISA 9.5 Figure 151 (a page image).
/// Two other sources, by other methods, agree with it: (1) the PTX text for `mma.m16n8k16` `.f16`/`.bf16` matrix A
/// (9.7.16.5.8): `groupID = %laneid >> 2`, `row = groupID` for `a0 a1 a4 a5` and `groupID + 8` otherwise,
/// `col = threadID_in_group * 2 + (i & 1)` (`+ 8` for `i >= 4`): the same per-warp 16 x 16 tile, which `wgmma`'s warp `w`
/// holds at rows `16 w ..`; (2) CUTLASS `include/cute/atom/mma_traits_sm90_gmma.hpp` (commit 5c179edeb3e2, BSD-3-Clause,
/// facts only, no code copied), `GMMA::ALayout_64x16 = Layout<Shape<Shape<_4,_8,_4>,Shape<_2,_2,_2>>,
/// Stride<Stride<_128,_1,_16>,Stride<_64,_8,_512>>>`, the register-source layout of the `SM90_64x*x16_*_RS` atoms, read as
/// (thread, value) -> `m + 64 k`: thread `l = t0 + 4 t1 + 32 t2` gives `k += 2 t0` (stride 128), `m += t1` (stride 1),
/// `m += 16 t2` (stride 16); value `i = v0 + 2 v1 + 4 v2` gives `k += v0` (64), `m += 8 v1` (8), `k += 8 v2` (512): the
/// formulas below, term by term. All three agree.
/// Each thread loads its 8 elements of `A` into the register fragment (alloc 3), at the (row, column) PTX ISA 9.5
/// Figure 151 gives them: warp `w` holds rows `16 w ..`; lane `l` holds rows `l / 4` and `l / 4 + 8`, columns
/// `2 (l % 4)`, `+ 1`, and the same 8 columns on (`a0 a1`, `a2 a3`, `a4 a5`, `a6 a7`). One statement (a one-trip loop), so
/// the statement positions the other tests rely on do not move.
fn fill_a_fragment() -> Stmt {
    let shr =
        |a: ScalarExpr, n: i64| ScalarExpr::Binary(BinaryOp::Shr, Box::new(a), Box::new(u(n)));
    let and =
        |a: ScalarExpr, m: i64| ScalarExpr::Binary(BinaryOp::BitAnd, Box::new(a), Box::new(u(m)));
    let mut body = Vec::new();
    for i in 0..8i64 {
        let row = add(
            add(mul(shr(tid(), 5), u(16)), shr(and(tid(), 31), 2)),
            u(8 * ((i / 2) % 2)),
        );
        let col = add(mul(and(tid(), 3), u(2)), u(i % 2 + 8 * (i / 4)));
        // each load in its own scope: a `Load` declares its local
        body.push(counted(
            1,
            1,
            vec![
                Stmt::Load {
                    local: 2,
                    place: buf(0, add(mul(row, u(16)), col)),
                },
                Stmt::Store {
                    place: alloc(3, u(i)),
                    value: local(2),
                },
            ],
        ));
    }
    counted(0, 1, body)
}

/// The statements of a scenario: fill the tiles, (barrier), initialise `d`, wgmma, (commit), wait, copy `d` out.
fn body(o: Opt) -> Vec<Stmt> {
    let n = o.n as i64;
    let copy = |src, dst, per, layout| copy_tile(src, dst, per, o.fence, layout);
    let first = if o.a_regs {
        fill_a_fragment()
    } else {
        copy(0, 0, 8, o.a_layout)
    };
    let mut body = vec![first];
    // `b` has n * 16 elements: n / 8 per thread
    if n >= 8 {
        body.push(copy(1, 1, n / 8, o.b_layout));
    }
    if o.barrier {
        body.push(Stmt::barrier(Scope::Block));
    }
    body.push(counted(
        0,
        n / 2,
        vec![Stmt::Store {
            place: alloc(2, local(0)),
            value: ScalarExpr::Const(f64::from(INIT)),
        }],
    ));
    body.push(Stmt::Wgmma {
        n: o.n,
        ab: BF16,
        acc: F32,
        a: alloc(0, u(0)),
        b: alloc(1, u(0)),
        d: 2,
        scale_d: u(o.scale_d),
        a_layout: o.a_layout,
        b_layout: o.b_layout,
        a_regs: o.a_regs.then_some(3),
    });
    if o.commit {
        body.push(Stmt::WgmmaCommit);
    }
    body.push(Stmt::WgmmaWait { pending: o.wait });
    body.push(counted(
        0,
        n / 2,
        vec![
            Stmt::Load {
                local: 3,
                place: alloc(2, local(0)),
            },
            Stmt::Store {
                place: buf(2, add(mul(tid(), u(n / 2)), local(0))),
                value: local(3),
            },
        ],
    ));
    body
}

/// Params: 0 `a_in` `[64 * 16]`, 1 `b_in` `[n * 16]` (bf16, read), 2 `out` `[128 * n / 2]` (f32, read-write). Allocs: 0 `a`
/// Shared, 1 `b` Shared, 2 `d` Private `[n / 2]` f32. Locals: 0 `i`, 1 `idx`, 2 bf16 temp, 3 f32 temp.
fn kernel(o: Opt) -> KernelFn {
    let n = o.n as i64;
    let mut k = KernelFn::new(
        "wgmma",
        F32,
        vec![
            Param::buffer("a_in", BF16, BufferRole::Read, Some(1)),
            Param::buffer("b_in", BF16, BufferRole::Read, Some(1)),
            Param::buffer("out", F32, BufferRole::ReadWrite, Some(1)),
        ],
        vec![
            LocalDecl::scalar(U32),
            LocalDecl::scalar(U32),
            LocalDecl::scalar(BF16),
            LocalDecl::scalar(F32),
        ],
        Block(body(o)),
    )
    .with_launch(LaunchShape::new([128, 1, 1], [1, 1, 1]).with_subgroup(32, 32));
    k.allocs = vec![
        AllocDecl::new(
            "a",
            MemorySpace::Shared,
            BF16,
            Extent::Static(span(o.a_layout, 64)),
        ),
        AllocDecl::new(
            "b",
            MemorySpace::Shared,
            BF16,
            Extent::Static(span(o.b_layout, o.n as usize)),
        ),
        AllocDecl::new(
            "d",
            MemorySpace::Private,
            F32,
            Extent::Static((n / 2) as u32),
        ),
    ];
    if o.a_regs {
        // alloc 3: this thread's register fragment of A (8 bf16 elements)
        k.allocs.push(AllocDecl::new(
            "a_frag",
            MemorySpace::Private,
            BF16,
            Extent::Static(8),
        ));
    }
    k
}

fn a_val(m: usize, k: usize) -> f32 {
    ((m * 5 + k * 3) % 11) as f32
}
fn b_val(n: usize, k: usize) -> f32 {
    ((n * 7 + k * 5) % 13) as f32
}
fn bf16(v: f32) -> u16 {
    (v.to_bits() >> 16) as u16
}
fn bf16_buf(rows: usize, f: impl Fn(usize, usize) -> f32) -> KernelArg {
    let bytes: Vec<u8> = (0..rows * K)
        .flat_map(|i| bf16(f(i / K, i % K)).to_le_bytes())
        .collect();
    KernelArg::Buffer(TypedBuffer::new(
        BF16,
        vec![(rows * K) as i64],
        vec![1],
        bytes,
    ))
}

fn run(k: &KernelFn, n: u32) -> Result<Vec<f32>, KernelTrap> {
    assert_eq!(check_kernel(k), Ok(()));
    let mut args = [
        bf16_buf(64, a_val),
        bf16_buf(n as usize, b_val),
        KernelArg::Buffer(TypedBuffer::new(
            F32,
            vec![(128 * n / 2) as i64],
            vec![1],
            vec![0; 128 * n as usize * 2],
        )),
    ];
    run_launch(k, &mut args)?;
    let KernelArg::Buffer(b) = &args[2] else {
        unreachable!("a buffer")
    };
    Ok(b.bytes
        .chunks(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect())
}

/// What thread `t` should hold in value `v` after `D = A.B + scale * INIT`: the PTX fragment layout (warp `w` holds rows
/// `16 w ..`; value `v` is n8 block `v / 4`, then row `lane / 4 (+ 8)`, column `2 * (lane % 4) (+ 1)`).
fn expected(n: u32, scale: bool) -> Vec<f32> {
    let mut out = Vec::new();
    for t in 0..128usize {
        for v in 0..(n as usize / 2) {
            let row = 16 * (t / 32) + (t % 32) / 4 + 8 * ((v % 4) / 2);
            let col = 8 * (v / 4) + 2 * (t % 4) + v % 2;
            let dot: f32 = (0..K).map(|k| a_val(row, k) * b_val(col, k)).sum();
            out.push(dot + if scale { INIT } else { 0.0 });
        }
    }
    out
}

#[test]
fn every_fragment_value_is_the_reference_product() {
    for n in [8u32, 16, 64] {
        let got = run(&kernel(Opt::plain(n)), n).expect("the oracle runs it");
        assert_eq!(got, expected(n, false), "n = {n}");
    }
}

#[test]
fn the_reference_is_not_degenerate() {
    // distinct values, so a layout that scrambles rows or columns cannot pass by accident
    let e = expected(16, false);
    let distinct: std::collections::BTreeSet<u32> = e.iter().map(|v| v.to_bits()).collect();
    assert!(distinct.len() > 40, "{} distinct values", distinct.len());
}

#[test]
fn scale_d_picks_overwrite_or_accumulate() {
    let mut o = Opt::plain(16);
    assert_eq!(run(&kernel(o), 16), Ok(expected(16, false)));
    o.scale_d = 1;
    assert_eq!(run(&kernel(o), 16), Ok(expected(16, true)));
}

#[test]
fn the_result_lands_only_at_the_wait_that_covers_its_group() {
    let landed = |o: Opt| run(&kernel(o), 8).unwrap() == expected(8, false);
    let untouched = |o: Opt| run(&kernel(o), 8).unwrap().iter().all(|v| *v == INIT);
    let mut o = Opt::plain(8);
    assert!(landed(o), "wait 0 after commit lands it");
    o.wait = 1; // the newest group may stay in flight
    assert!(untouched(o), "wait 1 leaves the group in flight");
    o.wait = 0;
    o.commit = false; // an uncommitted operation is not in any group
    assert!(untouched(o), "an uncommitted wgmma is not waited for");
}

#[test]
fn reading_the_tiles_before_the_barrier_is_a_data_race() {
    let mut o = Opt::plain(8);
    o.barrier = false;
    assert!(matches!(
        run_unchecked(&kernel(o)),
        Err(KernelTrap::DataRace | KernelTrap::UninitializedShared { .. })
    ));
}

fn run_unchecked(k: &KernelFn) -> Result<Vec<f32>, KernelTrap> {
    run(k, 8)
}

#[test]
fn the_checker_refuses_what_a_warpgroup_mma_cannot_be() {
    let refused =
        |k: &KernelFn| matches!(check_kernel(k), Err(KernelError::WgmmaUnsupported { .. }));
    let mut half = kernel(Opt::plain(8));
    half.launch = Some(LaunchShape::new([96, 1, 1], [1, 1, 1]).with_subgroup(32, 32));
    let mut sub16 = kernel(Opt::plain(8));
    sub16.launch = Some(LaunchShape::new([128, 1, 1], [1, 1, 1]).with_subgroup(16, 16));
    let mut bad_n = kernel(Opt::plain(8));
    if let Stmt::Wgmma { n, .. } = &mut bad_n.body.0[4] {
        *n = 12;
    } else {
        panic!("the wgmma is statement 4");
    }
    let mut short_d = kernel(Opt::plain(16));
    short_d.allocs[2] = AllocDecl::new("d", MemorySpace::Private, F32, Extent::Static(7));
    let mut shared_d = kernel(Opt::plain(8));
    shared_d.allocs[2] = AllocDecl::new("d", MemorySpace::Shared, F32, Extent::Static(4));
    for (what, k) in [
        ("96 threads", &half),
        ("subgroup 16", &sub16),
        ("n = 12", &bad_n),
        ("short fragment", &short_d),
        ("Shared fragment", &shared_d),
    ] {
        assert!(refused(k), "{what}: {:?}", check_kernel(k));
    }
    // the control
    assert_eq!(check_kernel(&kernel(Opt::plain(8))), Ok(()));
}

// ---- `ProxyFence` (U8c slice 0): PTX ISA 9.5 9.7.17.7.1 "An async proxy fence must be used to establish an ordering
// between prior writes to shared memory matrices and subsequent reads of the same matrices in a wgmma.mma_async".

fn on(t: i64, body: Vec<Stmt>) -> Stmt {
    Stmt::If {
        cond: ScalarExpr::Binary(BinaryOp::CmpEq, Box::new(tid()), Box::new(u(t))),
        then: Block(body),
        els: Block(vec![]),
    }
}

fn proxy_fence() -> Stmt {
    Stmt::ProxyFence {
        space: MemorySpace::Shared,
    }
}

#[test]
fn a_tile_written_without_the_proxy_fence_traps_when_the_wgmma_reads_it() {
    let mut o = Opt::plain(8);
    o.fence = false;
    assert!(matches!(
        run_unchecked(&kernel(o)),
        Err(KernelTrap::MissingProxyFence { .. })
    ));
}

#[test]
fn another_threads_fence_does_not_cover_a_write() {
    // thread 0 fences after the barrier, but threads 1..127 wrote their share of the tiles
    let mut o = Opt::plain(8);
    o.fence = false;
    let mut k = kernel(o);
    k.body.0.insert(3, on(0, vec![proxy_fence()]));
    assert!(matches!(
        run_unchecked(&k),
        Err(KernelTrap::MissingProxyFence { .. })
    ));
}

#[test]
fn a_fence_by_every_writer_before_the_barrier_is_enough() {
    let mut o = Opt::plain(8);
    o.fence = false;
    let mut k = kernel(o);
    // the fence goes before the barrier that publishes the tile (CUTLASS's `fence_view_async_shared()` then a sync)
    k.body.0.insert(2, proxy_fence());
    let got = run_unchecked(&k).expect("fenced by each writer, then ordered by the barrier");
    assert_eq!(got, expected(8, false));
}

#[test]
fn bytes_a_bulk_copy_wrote_need_no_fence_but_a_generic_store_does() {
    // A arrives by `CopyBulk` (the async proxy: completion carries the implicit fence); B is stored by threads
    let bulk = |fence_b: bool| {
        let mut o = Opt::plain(8);
        o.fence = fence_b;
        let mut k = kernel(o);
        let mut body = k.body.0.clone();
        let copy_b = body.remove(1);
        body.remove(0); // the generic copy of A
        let bar = Place::Alloc {
            alloc: 3,
            index: u(0),
        };
        let mut new = vec![
            on(
                0,
                vec![Stmt::MbarInit {
                    bar: bar.clone(),
                    count: u(1),
                }],
            ),
            Stmt::barrier(Scope::Block),
            on(
                0,
                vec![
                    Stmt::MbarArrive {
                        bar: bar.clone(),
                        tx: Some(u(64 * 16 * 2)),
                    },
                    Stmt::CopyBulk {
                        dst: alloc(0, u(0)),
                        src: Place::Logical {
                            buffer: 0,
                            coords: vec![u(0)],
                        },
                        elems: 64 * 16,
                        bar: bar.clone(),
                    },
                ],
            ),
            copy_b,
            Stmt::MbarWait { bar, parity: u(0) },
        ];
        new.extend(body);
        k.body = Block(new);
        k.allocs.push(AllocDecl::new(
            "bar",
            MemorySpace::Shared,
            ElementKind::U64,
            Extent::Static(1),
        ));
        k
    };
    let got = run_unchecked(&bulk(true)).expect("A by CopyBulk, B fenced");
    assert_eq!(got, expected(8, false));
    assert!(
        matches!(
            run_unchecked(&bulk(false)),
            Err(KernelTrap::MissingProxyFence { alloc: 1 })
        ),
        "B was stored by threads, so its read needs their fences (A, from the bulk copy, does not)"
    );
}

#[test]
fn the_checker_admits_a_shared_proxy_fence_and_refuses_other_spaces() {
    let with = |space| {
        let mut k = kernel(Opt::plain(8));
        k.body.0.insert(0, Stmt::ProxyFence { space });
        check_kernel(&k)
    };
    assert_eq!(with(MemorySpace::Shared), Ok(()));
    assert!(matches!(
        with(MemorySpace::Global),
        Err(KernelError::SyncUnsupported { .. })
    ));
}

#[test]
fn a_proxy_fence_survives_the_text_round_trip() {
    let json = serde_json::to_string(&proxy_fence()).expect("serializes");
    assert!(json.contains("ProxyFence"), "{json}");
    let back: Stmt = serde_json::from_str(&json).expect("parses");
    assert_eq!(back, proxy_fence());
}

// ---- operand layouts, end to end -------------------------------------------------------------------------------------

/// Dense canonical layouts for `rows` rows: K-major packs the two 8-element K halves of a core matrix `LBO` apart and
/// 8-row groups `SBO` apart; MN-major packs the 8-row groups `SBO` apart and the K halves `LBO` apart.
fn dense_k(rows: usize) -> OperandLayout {
    let _ = rows;
    k_major(128, 256) // core matrix 64 elements (128 B); the two K halves 64 elements apart, row groups 128 apart
}
fn dense_mn(rows: usize) -> OperandLayout {
    mn_major(2 * 64 * (rows as u32 / 8), 128) // row groups 64 elements (128 B) apart; the K halves a whole column of groups
}

#[test]
fn every_fragment_value_is_the_reference_product_under_each_canonical_layout() {
    for n in [8u32, 16, 64] {
        for (a, b) in [
            (dense_k(64), dense_k(n as usize)),
            (dense_mn(64), dense_mn(n as usize)),
            (dense_k(64), dense_mn(n as usize)),
            (dense_mn(64), dense_k(n as usize)),
            // padded: row groups a core matrix apart with a gap, the K halves beyond them
            (k_major(128, 512), k_major(256, 384)),
        ] {
            let mut o = Opt::plain(n);
            (o.a_layout, o.b_layout) = (a, b);
            let got = run(&kernel(o), n).expect("the oracle runs it");
            assert_eq!(got, expected(n, false), "n = {n}, a = {a:?}, b = {b:?}");
        }
    }
}

#[test]
fn declaring_a_different_layout_than_the_one_stored_gives_a_different_product() {
    // the tiles are stored K-major; the statement is told MN-major (valid, but another arrangement): the result moves
    let mut o = Opt::plain(16);
    o.a_layout = dense_k(64);
    o.b_layout = dense_k(16);
    let mut k = kernel(o);
    if let Some(Stmt::Wgmma {
        a_layout, b_layout, ..
    }) = k
        .body
        .0
        .iter_mut()
        .find(|s| matches!(s, Stmt::Wgmma { .. }))
    {
        *a_layout = dense_mn(64);
        *b_layout = dense_mn(16);
    }
    assert_ne!(
        run(&k, 16).expect("it still runs"),
        expected(16, false),
        "the oracle read through the declared layout"
    );
}

#[test]
fn the_checker_refuses_layouts_a_descriptor_cannot_carry() {
    let refused = |a: OperandLayout, mutate: &dyn Fn(&mut KernelFn)| {
        let mut o = Opt::plain(8);
        o.a_layout = dense_k(64);
        let mut k = kernel(o);
        if let Some(Stmt::Wgmma { a_layout, .. }) = k
            .body
            .0
            .iter_mut()
            .find(|s| matches!(s, Stmt::Wgmma { .. }))
        {
            *a_layout = a;
        }
        mutate(&mut k);
        matches!(check_kernel(&k), Err(KernelError::WgmmaUnsupported { .. }))
    };
    let keep: &dyn Fn(&mut KernelFn) = &|_| {};
    assert!(!refused(dense_k(64), keep), "the control is admitted");
    let swz = |sw| OperandLayout::Canonical {
        major: Major::K,
        swizzle: sw,
        lbo: Some(128),
        sbo: 256,
    };
    assert!(refused(swz(Swizzle::B128), keep), "swizzle not built");
    assert!(
        refused(
            OperandLayout::Canonical {
                major: Major::K,
                swizzle: Swizzle::None,
                lbo: None,
                sbo: 256
            },
            keep
        ),
        "no leading offset"
    );
    assert!(refused(k_major(0, 256), keep), "a zero offset");
    assert!(refused(k_major(136, 256), keep), "not a multiple of 16");
    assert!(
        refused(k_major(128, 1 << 18), keep),
        "beyond 14 bits of offset >> 4"
    );
    assert!(
        refused(k_major(128, 128), keep),
        "two core matrices at one place"
    );
    assert!(
        refused(k_major(128, 128 * 1024), keep),
        "leaves the allocation"
    );
    // a tile start that is not a constant, or not 16-byte aligned
    let start = |e: ScalarExpr| {
        move |k: &mut KernelFn| {
            if let Some(Stmt::Wgmma {
                a: Place::Alloc { index, .. },
                ..
            }) = k
                .body
                .0
                .iter_mut()
                .find(|s| matches!(s, Stmt::Wgmma { .. }))
            {
                *index = e.clone();
            }
        }
    };
    assert!(
        refused(dense_k(64), &start(u(4))),
        "start not 16-byte aligned"
    );
    assert!(refused(dense_k(64), &start(tid())), "start not a constant");
}

#[test]
fn an_operand_layout_survives_the_text_round_trip_and_defaults_when_absent() {
    let layout = dense_mn(64);
    let json = serde_json::to_string(&layout).expect("serializes");
    assert_eq!(
        serde_json::from_str::<OperandLayout>(&json).unwrap(),
        layout
    );
    // a Wgmma written before the fields existed parses with RowMajor
    let mut o = Opt::plain(8);
    o.a_layout = layout;
    let k = kernel(o);
    let mut v = serde_json::to_value(&k).expect("a value");
    fn strip(v: &mut serde_json::Value) {
        match v {
            serde_json::Value::Object(m) => {
                m.remove("a_layout");
                m.remove("b_layout");
                m.values_mut().for_each(strip);
            }
            serde_json::Value::Array(a) => a.iter_mut().for_each(strip),
            _ => {}
        }
    }
    strip(&mut v);
    let old: KernelFn = serde_json::from_value(v).expect("an old kernel parses");
    let Some(Stmt::Wgmma {
        a_layout, b_layout, ..
    }) = old.body.0.iter().find(|s| matches!(s, Stmt::Wgmma { .. }))
    else {
        panic!("a wgmma");
    };
    assert_eq!(
        (*a_layout, *b_layout),
        (OperandLayout::RowMajor, OperandLayout::RowMajor)
    );
}

/// Dense swizzled layouts: an atom is 8 rows of `s` 16-byte chunks (`128 s` bytes). K-major packs the 8-row groups one
/// atom apart; MN-major packs the two 8-row K groups one atom apart (`SBO`) and the MN atoms after both (`LBO`).
fn dense_sw_k(sw: Swizzle) -> OperandLayout {
    k_major_sw(sw, 128 * swizzle_chunks(sw))
}
fn dense_sw_mn(sw: Swizzle) -> OperandLayout {
    let s = swizzle_chunks(sw);
    mn_major_sw(sw, 256 * s, 128 * s)
}
fn swizzle_chunks(sw: Swizzle) -> u32 {
    match sw {
        Swizzle::None => 1,
        Swizzle::B32 => 2,
        Swizzle::B64 => 4,
        Swizzle::B128 => 8,
    }
}

#[test]
fn every_fragment_value_is_the_reference_product_under_each_swizzled_layout() {
    for sw in [Swizzle::B32, Swizzle::B64, Swizzle::B128] {
        for n in [8u32, 16, 64] {
            for (a, b) in [
                (dense_sw_k(sw), dense_sw_k(sw)),
                (dense_sw_mn(sw), dense_sw_mn(sw)),
                (dense_sw_k(sw), dense_sw_mn(sw)),
                (dense_sw_mn(sw), dense_sw_k(sw)),
            ] {
                let mut o = Opt::plain(n);
                (o.a_layout, o.b_layout) = (a, b);
                let got = run(&kernel(o), n).expect("the oracle runs it");
                assert_eq!(
                    got,
                    expected(n, false),
                    "{sw:?} n = {n}, a = {a:?}, b = {b:?}"
                );
            }
        }
    }
}

#[test]
fn a_swizzled_layout_stored_unswizzled_gives_another_product() {
    // the tiles are stored in the 128B-swizzled arrangement; the statement is told the unswizzled K-major layout of the
    // same size (a valid arrangement, with the chunks in their un-XORed places): the product moves
    let mut o = Opt::plain(16);
    o.a_layout = dense_sw_k(Swizzle::B128);
    o.b_layout = dense_sw_k(Swizzle::B128);
    let mut k = kernel(o);
    if let Some(Stmt::Wgmma {
        a_layout, b_layout, ..
    }) = k
        .body
        .0
        .iter_mut()
        .find(|s| matches!(s, Stmt::Wgmma { .. }))
    {
        *a_layout = k_major(128, 1024);
        *b_layout = k_major(128, 1024);
    }
    // either the oracle reads a cell the swizzled stores never wrote (the rows use 2 of the 8 chunks, and the XOR moves
    // them), or it reads other values: it is never the reference product
    match run(&k, 16) {
        Err(KernelTrap::UninitializedShared { .. }) => {}
        other => assert_ne!(other.expect("it runs or traps"), expected(16, false)),
    }
}

#[test]
fn the_checker_applies_the_swizzle_rules() {
    let set = |a: OperandLayout, start: Option<i64>| {
        let mut o = Opt::plain(8);
        o.a_layout = dense_sw_k(Swizzle::B128);
        let mut k = kernel(o);
        if let Some(Stmt::Wgmma {
            a_layout,
            a: Place::Alloc { index, .. },
            ..
        }) = k
            .body
            .0
            .iter_mut()
            .find(|s| matches!(s, Stmt::Wgmma { .. }))
        {
            *a_layout = a;
            if let Some(st) = start {
                *index = u(st);
            }
        }
        check_kernel(&k)
    };
    let refused =
        |r: Result<(), KernelError>| matches!(r, Err(KernelError::WgmmaUnsupported { .. }));
    assert_eq!(set(dense_sw_k(Swizzle::B128), None), Ok(()), "the control");
    // a swizzled K-major operand does not use the leading offset
    assert!(refused(set(
        OperandLayout::Canonical {
            major: Major::K,
            swizzle: Swizzle::B128,
            lbo: Some(128),
            sbo: 1024
        },
        None
    )));
    // an MN-major one needs it
    assert!(refused(set(
        OperandLayout::Canonical {
            major: Major::MN,
            swizzle: Swizzle::B64,
            lbo: None,
            sbo: 512
        },
        None
    )));
    // a start that is a multiple of 16 bytes but not of the 1024-byte pattern (512 elements)
    assert!(refused(set(dense_sw_k(Swizzle::B128), Some(64))));
}

// ---- `A` in registers (U8c): PTX ISA 9.5 9.7.17.5.1.2, Figure 151 ---------------------------------------------------------

#[test]
fn every_fragment_value_is_the_reference_product_with_a_in_registers() {
    for n in [8u32, 16, 64] {
        let mut o = Opt::plain(n);
        o.a_regs = true;
        let got = run(&kernel(o), n).expect("the oracle runs it");
        assert_eq!(got, expected(n, false), "n = {n}");
    }
}

#[test]
fn a_in_registers_agrees_with_a_in_shared_memory() {
    let mut regs = Opt::plain(16);
    regs.a_regs = true;
    assert_eq!(
        run(&kernel(regs), 16).expect("registers"),
        run(&kernel(Opt::plain(16)), 16).expect("shared"),
    );
}

#[test]
fn a_fragment_filled_by_the_wrong_formula_gives_another_product() {
    // swap the two row halves of the fragment fill (a2 a3 and a0 a1 trade rows): the Figure 151 layout is what the
    // oracle reads, so the product must move
    let mut o = Opt::plain(16);
    o.a_regs = true;
    let mut k = kernel(o);
    if let Some(Stmt::For { body, .. }) = k.body.0.first_mut() {
        for st in &mut body.0 {
            let Stmt::For { body: inner, .. } = st else {
                continue;
            };
            for st in &mut inner.0 {
                if let Stmt::Store {
                    place: Place::Alloc { index, .. },
                    ..
                } = st
                {
                    if let ScalarExpr::ConstInt(i, t) = index.clone() {
                        *index = ScalarExpr::ConstInt(i ^ 2, t); // a_i stored in a_{i ^ 2}
                    }
                }
            }
        }
    }
    assert_ne!(run(&k, 16).expect("it still runs"), expected(16, false));
}

#[test]
fn the_checker_applies_the_register_a_rules() {
    let with = |f: &dyn Fn(&mut KernelFn)| {
        let mut o = Opt::plain(8);
        o.a_regs = true;
        let mut k = kernel(o);
        f(&mut k);
        check_kernel(&k)
    };
    assert_eq!(with(&|_| {}), Ok(()), "the control is admitted");
    let refused =
        |r: Result<(), KernelError>| matches!(r, Err(KernelError::WgmmaUnsupported { .. }));
    // a fragment of the wrong length
    assert!(refused(with(&|k| k.allocs[3] = AllocDecl::new(
        "a_frag",
        MemorySpace::Private,
        BF16,
        Extent::Static(4)
    ))));
    // a fragment in Shared memory
    assert!(refused(with(&|k| k.allocs[3] = AllocDecl::new(
        "a_frag",
        MemorySpace::Shared,
        BF16,
        Extent::Static(8)
    ))));
    // a layout on an operand that is in registers
    assert!(refused(with(&|k| {
        if let Some(Stmt::Wgmma { a_layout, .. }) = k
            .body
            .0
            .iter_mut()
            .find(|s| matches!(s, Stmt::Wgmma { .. }))
        {
            *a_layout = dense_k(64);
        }
    })));
}
