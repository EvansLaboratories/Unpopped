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
use operand_layout::{k_major, mn_major, span, store_index};

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

/// The statements of a scenario: fill the tiles, (barrier), initialise `d`, wgmma, (commit), wait, copy `d` out.
fn body(o: Opt) -> Vec<Stmt> {
    let n = o.n as i64;
    let copy = |src, dst, per, layout| copy_tile(src, dst, per, o.fence, layout);
    let mut body = vec![copy(0, 0, 8, o.a_layout)];
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
