// SPDX-License-Identifier: MIT OR Apache-2.0
//! **`tcgen05` in the launch interpreter** (ROADMAP U7 slice S5, `docs/options-u7-mma-atoms.md`): `Stmt::Tcgen05Mma`
//! (one thread issues a 128 x 16 by N x 16 product into **tensor memory**, `MemorySpace::Tensor`), `Tcgen05Commit` (the
//! mbarrier arrival that says the issued products are done) and `Tcgen05Ld` (a warp reads its 32-lane quadrant of tensor
//! memory into registers). The reference product and the lane mapping (row `m` of `D` is tensor-memory lane `m`, column `j`
//! is column `d_col + j`; warp `w` may read lanes `32 (w % 4) ..`) are written in this test from the PTX `tcgen05`
//! description, independent of the oracle. Each behaviour has a program that shows it and a control that does not.

use unpopped::exec::{AllocDecl, ExecId, Extent, LaunchShape, MemorySpace, Scope};
use unpopped::ir::{BinaryOp, ScalarExpr};
use unpopped::kernel::{
    Block, BufferRole, KernelError, KernelFn, LocalDecl, OperandLayout, Param, Place, Stmt,
    Swizzle, check_kernel,
};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::{KernelArg, KernelTrap};
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_vocab::{ElementKind, Fp8E4M3FN, Fp8E5M2};

#[path = "support/operand_layout.rs"]
mod operand_layout;
use operand_layout::{k_major, k_major_sw, mn_major, mn_major_sw, span, store_index};

const U32: ElementKind = ElementKind::U32;
const BF16: ElementKind = ElementKind::Bf16;
const F32: ElementKind = ElementKind::F32;
/// Columns of tensor memory before the product's first column.
const D_COL: i64 = 8;

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
fn on_thread0(body: Vec<Stmt>) -> Stmt {
    Stmt::If {
        cond: ScalarExpr::Binary(BinaryOp::CmpEq, Box::new(tid()), Box::new(u(0))),
        then: Block(body),
        els: Block(vec![]),
    }
}

/// How a scenario differs from the plain one.
#[derive(Clone, Copy)]
struct Opt {
    n: u32,
    threads: u32,
    /// How many products thread 0 issues (each accumulating into the last when more than one).
    products: u32,
    commit: bool,
    wait: bool,
    /// Each thread issues `fence.proxy.async` after storing its share of a tile.
    fence: bool,
    /// How the tiles sit in their allocations.
    a_layout: OperandLayout,
    b_layout: OperandLayout,
    /// Rows of `A` and `D`: 128, or 64 (Layout F) with `d_lane` 0 or 16.
    m: u32,
    d_lane: u32,
    /// The element types of A and B: bf16 (K = 16) or the 8-bit floats (K = 32).
    a_ty: ElementKind,
    b_ty: ElementKind,
}
impl Opt {
    fn k(self) -> usize {
        if self.a_ty == BF16 { 16 } else { 32 }
    }
    fn plain(n: u32) -> Self {
        Opt {
            n,
            threads: 128,
            products: 1,
            commit: true,
            wait: true,
            fence: true,
            a_layout: OperandLayout::RowMajor,
            b_layout: OperandLayout::RowMajor,
            m: 128,
            d_lane: 0,
            a_ty: BF16,
            b_ty: BF16,
        }
    }
    fn f8(n: u32, a_ty: ElementKind, b_ty: ElementKind) -> Self {
        Opt {
            a_ty,
            b_ty,
            ..Opt::plain(n)
        }
    }
}

/// Each thread copies `total / threads` elements of input buffer `src` into `Shared` allocation `dst`.
fn copy_tile(
    src: u32,
    dst: u32,
    total: i64,
    threads: i64,
    fence: bool,
    layout: OperandLayout,
    tmp: u32,
) -> Stmt {
    let per = total / threads;
    counted(
        0,
        per,
        vec![
            Stmt::Let {
                local: 1,
                init: add(mul(tid(), u(per)), local(0)),
            },
            Stmt::Load {
                local: tmp,
                place: buf(src, local(1)),
            },
            Stmt::Store {
                place: alloc(dst, store_index(layout, local(1))),
                value: local(tmp),
            },
        ]
        .into_iter()
        .chain(fence.then_some(Stmt::ProxyFence {
            space: MemorySpace::Shared,
        }))
        .collect(),
    )
}

/// The columns the `tmem` allocation holds: a power of two from 32 (`tcgen05.alloc`'s `nCols`) that fits `D_COL + n`.
fn tmem_cols(n: u32) -> u32 {
    (n + D_COL as u32).next_power_of_two().max(32)
}

fn mma(o: Opt, scale_d: i64) -> Stmt {
    Stmt::Tcgen05Mma {
        n: o.n,
        ab: o.a_ty,
        acc: F32,
        a: alloc(0, u(0)),
        b: alloc(1, u(0)),
        d: 2,
        d_col: u(D_COL),
        scale_d: u(scale_d),
        a_layout: o.a_layout,
        b_layout: o.b_layout,
        m: o.m,
        d_lane: o.d_lane,
        b_ty: (o.b_ty != o.a_ty).then_some(o.b_ty),
    }
}

/// init the barrier, fill the tiles, barrier, thread 0 issues (and commits), everyone waits, every thread reads its lane.
fn body(o: Opt) -> Vec<Stmt> {
    let (n, t) = (i64::from(o.n), i64::from(o.threads));
    let mut issue: Vec<Stmt> = (0..o.products).map(|i| mma(o, i64::from(i > 0))).collect();
    if o.commit {
        issue.push(Stmt::Tcgen05Commit {
            bar: alloc(3, u(0)),
        });
    }
    let mut body = vec![
        on_thread0(vec![
            Stmt::Tcgen05Alloc {
                d: 2,
                ncols: tmem_cols(o.n),
            },
            Stmt::MbarInit {
                bar: alloc(3, u(0)),
                count: u(1),
            },
        ]),
        copy_tile(
            0,
            0,
            i64::from(o.m) * o.k() as i64,
            t,
            o.fence,
            o.a_layout,
            2,
        ),
        copy_tile(1, 1, n * o.k() as i64, t, o.fence, o.b_layout, 4),
        Stmt::barrier(Scope::Block),
        on_thread0(issue),
    ];
    if o.wait {
        body.push(Stmt::MbarWait {
            bar: alloc(3, u(0)),
            parity: u(0),
        });
    }
    let ld = Stmt::Tcgen05Ld {
        src: 2,
        col: u(D_COL),
        count: o.n,
        dst: 4,
    };
    let store_out = counted(
        0,
        n,
        vec![
            Stmt::Load {
                local: 3,
                place: alloc(4, local(0)),
            },
            Stmt::Store {
                place: buf(2, add(mul(tid(), u(n)), local(0))),
                value: local(3),
            },
        ],
    );
    if o.m == 128 {
        body.push(ld);
        body.push(store_out);
    } else {
        // Layout F: only 16 of a warp's 32 lanes hold rows of D (the half `d_lane` names); the others read nothing
        let lane = ScalarExpr::Binary(BinaryOp::BitAnd, Box::new(tid()), Box::new(u(31)));
        let cmp = |op, v: i64| ScalarExpr::Binary(op, Box::new(lane.clone()), Box::new(u(v)));
        body.push(Stmt::If {
            cond: cmp(BinaryOp::CmpGe, i64::from(o.d_lane)),
            then: Block(vec![Stmt::If {
                cond: cmp(BinaryOp::CmpLt, i64::from(o.d_lane) + 16),
                then: Block(vec![ld, store_out]),
                els: Block(vec![]),
            }]),
            els: Block(vec![]),
        });
    }
    // every warp has read its lanes: free the tensor memory (PTX: all of it must be deallocated before the kernel exits)
    body.push(Stmt::barrier(Scope::Block));
    body.push(on_thread0(vec![Stmt::Tcgen05Dealloc {
        d: 2,
        ncols: tmem_cols(o.n),
    }]));
    body
}

/// Params: 0 `a_in` `[128 * 16]`, 1 `b_in` `[n * 16]` (bf16), 2 `out` `[threads * n]` (f32). Allocs: 0 `a` Shared, 1 `b`
/// Shared, 2 `tmem` Tensor `[128 * tmem_cols(n)]`, 3 `bar` Shared u64, 4 `frag` Private `[n]`.
fn kernel(o: Opt) -> KernelFn {
    let n = i64::from(o.n);
    let mut k = KernelFn::new(
        "tcgen05",
        F32,
        vec![
            Param::buffer("a_in", o.a_ty, BufferRole::Read, Some(1)),
            Param::buffer("b_in", o.b_ty, BufferRole::Read, Some(1)),
            Param::buffer("out", F32, BufferRole::ReadWrite, Some(1)),
        ],
        vec![
            LocalDecl::scalar(U32),
            LocalDecl::scalar(U32),
            LocalDecl::scalar(o.a_ty),
            LocalDecl::scalar(F32),
            LocalDecl::scalar(o.b_ty),
        ],
        Block(body(o)),
    )
    .with_launch(LaunchShape::new([o.threads, 1, 1], [1, 1, 1]).with_subgroup(32, 32));
    k.allocs = vec![
        AllocDecl::new(
            "a",
            MemorySpace::Shared,
            o.a_ty,
            Extent::Static(span(o.a_layout, o.m as usize) * o.k() as u32 / 16),
        ),
        AllocDecl::new(
            "b",
            MemorySpace::Shared,
            o.b_ty,
            Extent::Static(span(o.b_layout, n as usize) * o.k() as u32 / 16),
        ),
        AllocDecl::new(
            "tmem",
            MemorySpace::Tensor,
            F32,
            Extent::Static(128 * tmem_cols(o.n)),
        ),
        AllocDecl::new(
            "bar",
            MemorySpace::Shared,
            ElementKind::U64,
            Extent::Static(1),
        ),
        AllocDecl::new("frag", MemorySpace::Private, F32, Extent::Static(n as u32)),
    ];
    k
}

fn a_val(m: usize, k: usize) -> f32 {
    ((m * 5 + k * 3) % 11) as f32
}
fn b_val(n: usize, k: usize) -> f32 {
    ((n * 7 + k * 5) % 13) as f32
}
/// Small non-negative integers (0..=4) are exact in bf16, e4m3 and e5m2 alike, so the f64 reference needs no rounding.
fn a8_val(m: usize, k: usize) -> f32 {
    ((m * 3 + k * 2) % 5) as f32
}
fn b8_val(n: usize, k: usize) -> f32 {
    ((n * 2 + k * 3 + 1) % 5) as f32
}
fn typed_buf(ty: ElementKind, rows: usize, k: usize, f: impl Fn(usize, usize) -> f32) -> KernelArg {
    let vals = (0..rows * k).map(|i| f(i / k, i % k));
    let bytes: Vec<u8> = match ty {
        BF16 => vals
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect(),
        ElementKind::Fp8E4M3FN => vals.map(|v| Fp8E4M3FN::from_f32(v).0).collect(),
        ElementKind::Fp8E5M2 => vals.map(|v| Fp8E5M2::from_f32(v).0).collect(),
        _ => unreachable!("a type this test feeds"),
    };
    KernelArg::Buffer(TypedBuffer::new(
        ty,
        vec![(rows * k) as i64],
        vec![1],
        bytes,
    ))
}

fn val_a(o: Opt) -> fn(usize, usize) -> f32 {
    if o.a_ty == BF16 { a_val } else { a8_val }
}
fn val_b(o: Opt) -> fn(usize, usize) -> f32 {
    if o.b_ty == BF16 { b_val } else { b8_val }
}

fn run(k: &KernelFn, o: Opt) -> Result<Vec<f32>, KernelTrap> {
    assert_eq!(check_kernel(k), Ok(()));
    let len = (o.threads * o.n) as usize;
    let mut args = [
        typed_buf(o.a_ty, 128, o.k(), val_a(o)),
        typed_buf(o.b_ty, o.n as usize, o.k(), val_b(o)),
        KernelArg::Buffer(TypedBuffer::new(
            F32,
            vec![len as i64],
            vec![1],
            vec![0; len * 4],
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

/// What thread `t` should read: tensor-memory lane `32 ((t / 32) % 4) + t % 32` holds row `lane` of `D`, column `j` of it
/// being `sum_k A[lane][k] B[j][k]`, times the number of accumulated products.
fn expected(o: Opt) -> Vec<f32> {
    let mut out = Vec::new();
    for t in 0..o.threads as usize {
        let lane = 32 * ((t / 32) % 4) + t % 32;
        // the row of D this lane holds: lane r for m = 128; for m = 64 (Layout F) lane 32 w + d_lane + r' holds row
        // 16 w + r' for r' < 16, and the other 16 lanes of the quadrant hold nothing (their thread reads none)
        let row = if o.m == 128 {
            Some(lane)
        } else {
            let (w, l) = (lane / 32, lane % 32);
            (l >= o.d_lane as usize && l < o.d_lane as usize + 16)
                .then(|| 16 * w + (l - o.d_lane as usize))
        };
        for j in 0..o.n as usize {
            out.push(row.map_or(0.0, |r| {
                let (fa, fb) = (val_a(o), val_b(o));
                let dot: f32 = (0..o.k()).map(|k| fa(r, k) * fb(j, k)).sum();
                dot * o.products as f32
            }));
        }
    }
    out
}

#[test]
fn fp8_dense_k32_matches_the_reference_for_each_pairing_of_e4m3_and_e5m2() {
    const E4: ElementKind = ElementKind::Fp8E4M3FN;
    const E5: ElementKind = ElementKind::Fp8E5M2;
    for (a, b) in [(E4, E4), (E5, E5), (E4, E5), (E5, E4)] {
        for n in [8u32, 64] {
            let o = Opt::f8(n, a, b);
            assert_eq!(
                run(&kernel(o), o),
                Ok(expected(o)),
                "{a:?} x {b:?}, n = {n}"
            );
        }
    }
}

#[test]
fn the_fp8_reference_needs_all_32_k_not_the_first_16() {
    let o = Opt::f8(8, ElementKind::Fp8E4M3FN, ElementKind::Fp8E4M3FN);
    let full = expected(o);
    let (fa, fb) = (val_a(o), val_b(o));
    let half: f32 = (0..16).map(|k| fa(1, k) * fb(0, k)).sum();
    // thread 1 reads row 1, column 0: the 32-term sum is not the 16-term one
    assert_ne!(full[8], half);
}

#[test]
fn the_checker_applies_the_fp8_rules() {
    const E4: ElementKind = ElementKind::Fp8E4M3FN;
    let refuses = |o: Opt| check_kernel(&kernel(o)).is_err();
    // mixed f16-family types, and an fp8 operand paired with a 16-bit one, are refused
    assert!(refuses(Opt {
        b_ty: E4,
        ..Opt::plain(8)
    }));
    assert!(refuses(Opt {
        a_ty: E4,
        b_ty: BF16,
        ..Opt::plain(8)
    }));
    // a canonical layout on an 8-bit tile is not built
    assert!(refuses(Opt {
        a_layout: dense_k(),
        ..Opt::f8(8, E4, E4)
    }));
    // the accepted shape passes
    assert_eq!(check_kernel(&kernel(Opt::f8(8, E4, E4))), Ok(()));
}

#[test]
fn every_lane_reads_its_row_of_the_reference_product() {
    for n in [8u32, 16, 64] {
        let o = Opt::plain(n);
        assert_eq!(run(&kernel(o), o), Ok(expected(o)), "n = {n}");
    }
}

/// Dense canonical layouts for a 16-bit tile: core matrix 64 elements (128 B), K halves 128 B apart and 8-row groups
/// 256 B apart (K-major); or 8-row groups 128 B apart and the K halves a column of groups apart (MN-major).
fn dense_k() -> OperandLayout {
    k_major(128, 256)
}
fn dense_mn(rows: u32) -> OperandLayout {
    mn_major(2 * 64 * (rows / 8), 128)
}

#[test]
fn the_reference_product_holds_under_each_canonical_layout() {
    for n in [8u32, 16, 64] {
        for (a, b) in [
            (dense_k(), dense_k()),
            (dense_mn(128), dense_mn(n)),
            (dense_k(), dense_mn(n)),
            (dense_mn(128), dense_k()),
        ] {
            let mut o = Opt::plain(n);
            (o.a_layout, o.b_layout) = (a, b);
            assert_eq!(
                run(&kernel(o), o),
                Ok(expected(o)),
                "n = {n}, a = {a:?}, b = {b:?}"
            );
        }
    }
}

#[test]
fn a_layout_other_than_the_one_stored_gives_another_product() {
    let mut o = Opt::plain(16);
    o.a_layout = dense_k();
    o.b_layout = dense_k();
    let mut k = kernel(o);
    fn find(k: &mut KernelFn) -> Option<&mut Stmt> {
        k.body.0.iter_mut().find_map(|s| match s {
            Stmt::If { then, .. } => then
                .0
                .iter_mut()
                .find(|t| matches!(t, Stmt::Tcgen05Mma { .. })),
            _ => None,
        })
    }
    if let Some(Stmt::Tcgen05Mma { a_layout, .. }) = find(&mut k) {
        *a_layout = dense_mn(128);
    } else {
        panic!("the Tcgen05Mma is inside thread 0's `if`");
    }
    let got = run(&k, o).expect("it still runs");
    assert_ne!(got, expected(o));
}

#[test]
fn the_checker_refuses_layouts_a_descriptor_cannot_carry() {
    let with = |a: OperandLayout| {
        let mut o = Opt::plain(8);
        o.a_layout = dense_k();
        let mut k = kernel(o);
        for st in &mut k.body.0 {
            if let Stmt::If { then, .. } = st {
                for t in &mut then.0 {
                    if let Stmt::Tcgen05Mma { a_layout, .. } = t {
                        *a_layout = a;
                    }
                }
            }
        }
        check_kernel(&k)
    };
    assert_eq!(with(dense_k()), Ok(()), "the control is admitted");
    let refused = |a: OperandLayout, why: &str| {
        let r = with(a);
        assert!(
            matches!(r, Err(KernelError::Tcgen05Unsupported { .. })),
            "{why}: {r:?}"
        );
    };
    refused(k_major(128, 128), "two core matrices at one place");
    refused(k_major(0, 256), "a zero offset");
}

#[test]
fn the_reference_product_holds_under_swizzled_layouts() {
    for (sw, s) in [(Swizzle::B32, 2u32), (Swizzle::B64, 4), (Swizzle::B128, 8)] {
        for n in [8u32, 16, 64] {
            let mut o = Opt::plain(n);
            o.a_layout = k_major_sw(sw, 128 * s);
            o.b_layout = mn_major_sw(sw, 256 * s, 128 * s);
            assert_eq!(run(&kernel(o), o), Ok(expected(o)), "{sw:?} n = {n}");
        }
    }
}

// ---- tensor-memory allocation (U8c slice 4), PTX ISA 9.5 9.7.18.1.2 and 9.7.18.7.1 ------------------------------------

/// The statements of thread 0's opening `if` (the barrier init, preceded by the `tcgen05.alloc`).
fn opening(k: &mut KernelFn) -> &mut Vec<Stmt> {
    match &mut k.body.0[0] {
        Stmt::If { then, .. } => &mut then.0,
        _ => panic!("the first statement is thread 0's `if`"),
    }
}

fn alloc_of(o: Opt) -> Stmt {
    Stmt::Tcgen05Alloc {
        d: 2,
        ncols: tmem_cols(o.n),
    }
}

fn misuse(r: Result<Vec<f32>, KernelTrap>, what: &str) {
    match r {
        Err(KernelTrap::TmemMisuse { what: got }) => {
            assert!(got.contains(what), "wanted `{what}`, got `{got}`");
        }
        other => panic!("wanted a TmemMisuse about `{what}`, got {other:?}"),
    }
}

#[test]
fn tensor_memory_used_without_an_allocation_traps() {
    let o = Opt::plain(8);
    let mut k = kernel(o);
    opening(&mut k).remove(0); // no tcgen05.alloc
    misuse(run_unchecked(&k, o), "before tcgen05.alloc");
}

#[test]
fn tensor_memory_never_deallocated_traps_when_the_block_ends() {
    let o = Opt::plain(8);
    let mut k = kernel(o);
    k.body.0.pop(); // no tcgen05.dealloc
    misuse(run_unchecked(&k, o), "still allocated");
}

#[test]
fn allocating_twice_or_after_relinquishing_traps() {
    let o = Opt::plain(8);
    let mut twice = kernel(o);
    opening(&mut twice).insert(1, alloc_of(o));
    misuse(run_unchecked(&twice, o), "already allocated");
    let mut late = kernel(o);
    opening(&mut late).insert(0, Stmt::Tcgen05RelinquishPermit);
    misuse(run_unchecked(&late, o), "relinquish_alloc_permit");
    // the control: relinquishing AFTER the allocation is the ordinary sequence
    let mut fine = kernel(o);
    opening(&mut fine).insert(1, Stmt::Tcgen05RelinquishPermit);
    assert_eq!(run_unchecked(&fine, o), Ok(expected(o)));
}

#[test]
fn tensor_memory_read_after_its_deallocation_traps() {
    let o = Opt::plain(8);
    let mut k = kernel(o);
    k.body.0.pop(); // the closing dealloc
    let at = k
        .body
        .0
        .iter()
        .position(|s| matches!(s, Stmt::Tcgen05Ld { .. }))
        .expect("the load");
    k.body.0.insert(
        at,
        on_thread0(vec![Stmt::Tcgen05Dealloc {
            d: 2,
            ncols: tmem_cols(o.n),
        }]),
    );
    k.body.0.insert(at, Stmt::barrier(Scope::Block));
    misuse(run_unchecked(&k, o), "after tcgen05.dealloc");
}

#[test]
fn a_column_count_that_grows_between_allocations_traps() {
    let o = Opt::plain(8);
    let big = |k: &mut KernelFn| {
        k.allocs.push(AllocDecl::new(
            "tmem2",
            MemorySpace::Tensor,
            F32,
            Extent::Static(128 * 64),
        ));
    };
    let alloc64 = Stmt::Tcgen05Alloc { d: 5, ncols: 64 };
    let dealloc64 = Stmt::Tcgen05Dealloc { d: 5, ncols: 64 };
    // 32 columns, then 64: the count increased
    let mut grows = kernel(o);
    big(&mut grows);
    opening(&mut grows).insert(1, alloc64.clone());
    misuse(run_unchecked(&grows, o), "increased");
    // 64 columns, then 32: allowed (and deallocated at the end)
    let mut shrinks = kernel(o);
    big(&mut shrinks);
    opening(&mut shrinks).insert(0, alloc64);
    shrinks.body.0.push(on_thread0(vec![dealloc64]));
    assert_eq!(run_unchecked(&shrinks, o), Ok(expected(o)));
}

#[test]
fn the_checker_refuses_an_alloc_that_tensor_memory_cannot_be() {
    let refused = |s: Stmt| {
        let o = Opt::plain(8);
        let mut k = kernel(o);
        opening(&mut k)[0] = s;
        matches!(
            check_kernel(&k),
            Err(KernelError::Tcgen05Unsupported { .. })
        )
    };
    let o = Opt::plain(8);
    assert!(!refused(alloc_of(o)), "the control is admitted");
    assert!(
        refused(Stmt::Tcgen05Alloc { d: 2, ncols: 48 }),
        "not a power of two"
    );
    assert!(refused(Stmt::Tcgen05Alloc { d: 2, ncols: 16 }), "below 32");
    assert!(
        refused(Stmt::Tcgen05Alloc { d: 2, ncols: 1024 }),
        "above 512"
    );
    assert!(
        refused(Stmt::Tcgen05Alloc { d: 2, ncols: 64 }),
        "not the declared columns (32)"
    );
    assert!(
        refused(Stmt::Tcgen05Alloc { d: 0, ncols: 32 }),
        "a Shared allocation"
    );
}

// ---- M = 64, Layout F (U8c): PTX ISA 9.5 9.7.18.10.5.6, Figures 221 and 222 -----------------------------------------------

#[test]
fn m64_puts_16_rows_in_each_warps_lower_or_upper_half_quadrant() {
    for d_lane in [0u32, 16] {
        for n in [8u32, 16, 64] {
            let mut o = Opt::plain(n);
            (o.m, o.d_lane) = (64, d_lane);
            assert_eq!(
                run(&kernel(o), o),
                Ok(expected(o)),
                "n = {n}, d_lane = {d_lane}"
            );
        }
    }
}

#[test]
fn the_other_half_of_a_quadrant_is_not_written_by_an_m64_mma() {
    // d_lane 0 writes lanes 0..16 of each quadrant: reading lane 16 of warp 0 (never written) traps
    let mut o = Opt::plain(8);
    (o.m, o.d_lane) = (64, 0);
    let mut k = kernel(o);
    // make every thread read, whatever its lane: drop the lane guards (the Ld sits inside two nested Ifs)
    let at = k
        .body
        .0
        .iter()
        .position(|s| {
            matches!(
                s,
                Stmt::If {
                    cond: ScalarExpr::Binary(BinaryOp::CmpGe, ..),
                    ..
                }
            )
        })
        .expect("the guarded readout");
    if let Stmt::If { then, .. } = k.body.0[at].clone() {
        if let Some(Stmt::If { then: inner, .. }) = then.0.first() {
            k.body.0.splice(at..=at, inner.0.clone());
        }
    }
    assert!(
        matches!(
            run_unchecked(&k, o),
            Err(KernelTrap::UninitializedShared { .. }
                | KernelTrap::UninitializedPrivate { .. }
                | KernelTrap::Unsupported(_))
        ),
        "reading a lane no row was written to must not look like data: {:?}",
        run_unchecked(&k, o)
    );
}

#[test]
fn the_checker_applies_the_m_and_lane_alignment_rules() {
    let with = |m: u32, d_lane: u32| {
        let mut o = Opt::plain(8);
        (o.m, o.d_lane) = (m, d_lane);
        let mut k = kernel(o);
        for st in &mut k.body.0 {
            if let Stmt::If { then, .. } = st {
                for t in &mut then.0 {
                    if let Stmt::Tcgen05Mma {
                        m: mm, d_lane: dl, ..
                    } = t
                    {
                        (*mm, *dl) = (m, d_lane);
                    }
                }
            }
        }
        check_kernel(&k)
    };
    assert_eq!(with(128, 0), Ok(()));
    assert_eq!(with(64, 0), Ok(()));
    assert_eq!(with(64, 16), Ok(()));
    let bad = |r: Result<(), KernelError>| matches!(r, Err(KernelError::Tcgen05Unsupported { .. }));
    assert!(bad(with(64, 8)), "an alignment other than 0 or 16");
    assert!(bad(with(128, 16)), "an alignment on m = 128");
    assert!(bad(with(32, 0)), "m = 32 (Layout G) is not built");
    assert!(bad(with(256, 0)), "m = 256 needs cta_group::2");
}

#[test]
fn a_warp_reads_the_quadrant_its_index_modulo_four_names() {
    for n in [16u32, 64] {
        let mut o = Opt::plain(n);
        o.threads = 256; // warps 4..8 read lanes 0..128 again
        assert_eq!(run(&kernel(o), o), Ok(expected(o)), "n = {n}");
    }
}

#[test]
fn the_reference_is_not_degenerate() {
    let e = expected(Opt::plain(16));
    let distinct: std::collections::BTreeSet<u32> = e.iter().map(|v| v.to_bits()).collect();
    assert!(distinct.len() > 40, "{} distinct values", distinct.len());
}

#[test]
fn a_later_product_accumulates_when_scale_d_is_set() {
    let mut o = Opt::plain(16);
    o.products = 2; // overwrite, then accumulate: twice the product
    assert_eq!(run(&kernel(o), o), Ok(expected(o)));
}

#[test]
fn the_result_needs_the_commit_and_an_acquire_on_its_barrier() {
    let mut o = Opt::plain(8);
    o.wait = false; // reading without the acquire is a race with the issuing thread
    assert!(matches!(
        run_unchecked(&kernel(o), o),
        Err(KernelTrap::DataRace | KernelTrap::UninitializedShared { .. })
    ));
    let mut o = Opt::plain(8);
    o.commit = false; // nothing ever arrives on the barrier
    assert_eq!(
        run_unchecked(&kernel(o), o),
        Err(KernelTrap::MbarrierDeadlock)
    );
}

#[test]
fn tiles_written_without_the_proxy_fence_trap_when_the_tcgen05_reads_them() {
    // PTX ISA 9.5 9.7.18.6.5: "The shared memory accesses by tcgen05.mma and tcgen05.cp operations are performed asynchronously
    // and via the asynchronous proxy ... fence.proxy.async should be used to synchronize memory between generic proxy and the async proxy"
    let mut o = Opt::plain(8);
    o.fence = false;
    assert!(matches!(
        run_unchecked(&kernel(o), o),
        Err(KernelTrap::MissingProxyFence { .. })
    ));
}

fn run_unchecked(k: &KernelFn, o: Opt) -> Result<Vec<f32>, KernelTrap> {
    run(k, o)
}

#[test]
fn the_checker_refuses_what_a_tensor_core_op_cannot_be() {
    let refused =
        |k: &KernelFn| matches!(check_kernel(k), Err(KernelError::Tcgen05Unsupported { .. }));
    let tmem = |k: &mut KernelFn, space, elem, len| {
        k.allocs[2] = AllocDecl::new("tmem", space, elem, Extent::Static(len));
    };
    let mut shared_d = kernel(Opt::plain(8));
    tmem(&mut shared_d, MemorySpace::Shared, F32, 128 * 16);
    let mut ragged = kernel(Opt::plain(8));
    tmem(&mut ragged, MemorySpace::Tensor, F32, 128 * 16 + 1);
    let mut wide_elem = kernel(Opt::plain(8));
    tmem(
        &mut wide_elem,
        MemorySpace::Tensor,
        ElementKind::F64,
        128 * 16,
    );
    let mut bad_n = kernel(Opt::plain(8));
    if let Stmt::If { then, .. } = &mut bad_n.body.0[4] {
        if let Stmt::Tcgen05Mma { n, .. } = &mut then.0[0] {
            *n = 12;
        }
    }
    let mut sub16 = kernel(Opt::plain(8));
    sub16.launch = Some(LaunchShape::new([128, 1, 1], [1, 1, 1]).with_subgroup(16, 16));
    let mut ordinary = kernel(Opt::plain(8));
    ordinary.body.0.insert(
        0,
        Stmt::Load {
            local: 3,
            place: alloc(2, u(0)),
        },
    );
    for (what, k) in [
        ("tmem in Shared", &shared_d),
        ("tmem not a multiple of 128 elements", &ragged),
        ("64-bit tmem element", &wide_elem),
        ("n = 12", &bad_n),
        ("subgroup 16", &sub16),
        ("an ordinary load of tmem", &ordinary),
    ] {
        assert!(refused(k), "{what}: {:?}", check_kernel(k));
    }
    assert_eq!(check_kernel(&kernel(Opt::plain(8))), Ok(()));
}
