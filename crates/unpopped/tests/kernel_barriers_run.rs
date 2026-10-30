// SPDX-License-Identifier: MIT OR Apache-2.0
//! U3 row 4 PR 3: `Barrier` in `run_launch`, the resumable evaluator, and barrier-aware data races
//! (`docs/u3-row4-scopes-and-memory-spaces.md` section 5, steps 2 and 3).
//!
//! The shape of every race test: a program that is a race without its barrier, the same program
//! with the barrier running clean, so a test goes red if the ordering rule is deleted and also if
//! it is made to forbid everything.

use unpopped::exec::{AllocDecl, ExecId, Extent, LaunchShape, MemorySpace, Scope};
use unpopped::ir::ScalarExpr;
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::{KernelArg, KernelTrap};
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_vocab::ElementKind;

const F32: ElementKind = ElementKind::F32;
const I32: ElementKind = ElementKind::I32;
const U32: ElementKind = ElementKind::U32;

const N: u32 = 0; // scalar param n: i32
const G: u32 = 1; // global thread index: u32
const X: u32 = 2; // f32
const Y: u32 = 3; // f32
const R: u32 = 4; // loop counter: i32

fn local(i: u32) -> ScalarExpr {
    ScalarExpr::Local(i)
}

fn exec(id: ExecId) -> ScalarExpr {
    ScalarExpr::Exec(id)
}

fn tid() -> ScalarExpr {
    exec(ExecId::ThreadIdx(0))
}

fn u(v: i64) -> ScalarExpr {
    ScalarExpr::ConstInt(v, U32)
}

fn ci(v: i64) -> ScalarExpr {
    ScalarExpr::ConstInt(v, I32)
}

fn add(a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Add(Box::new(a), Box::new(b))
}

fn sub(a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Sub(Box::new(a), Box::new(b))
}

fn mul(a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Mul(Box::new(a), Box::new(b))
}

fn buf(buffer: u32, c: ScalarExpr) -> Place {
    Place::Logical {
        buffer,
        coords: vec![c],
    }
}

fn tile(index: ScalarExpr) -> Place {
    Place::Alloc { alloc: 0, index }
}

fn global_x() -> ScalarExpr {
    add(
        mul(exec(ExecId::BlockIdx(0)), exec(ExecId::BlockDim(0))),
        tid(),
    )
}

fn let_g() -> Stmt {
    Stmt::Let {
        local: G,
        init: global_x(),
    }
}

fn barrier(scope: Scope) -> Stmt {
    Stmt::Barrier { scope }
}

fn kernel(body: Vec<Stmt>, launch: LaunchShape, tile_len: u32) -> KernelFn {
    KernelFn::new(
        "barriers",
        F32,
        vec![
            Param::buffer("in", F32, BufferRole::Read, Some(1)),
            Param::buffer("out", F32, BufferRole::ReadWrite, Some(1)),
            Param::buffer("aux", F32, BufferRole::Write, Some(1)),
            Param::scalar("n", I32, N),
        ],
        vec![
            LocalDecl::scalar(I32),
            LocalDecl::scalar(U32),
            LocalDecl::scalar(F32),
            LocalDecl::scalar(F32),
            LocalDecl::scalar(I32),
        ],
        Block(body),
    )
    .with_launch(launch)
    .with_allocs(vec![AllocDecl::new(
        "tile",
        MemorySpace::Shared,
        F32,
        Extent::Static(tile_len),
    )])
}

fn args(n: usize) -> Vec<KernelArg> {
    let input: Vec<f32> = (0..n).map(|i| i as f32 + 1.0).collect();
    let zeros = vec![0.0; n];
    vec![
        KernelArg::Buffer(TypedBuffer::from_f32(&[n as i64], &input)),
        KernelArg::Buffer(TypedBuffer::from_f32(&[n as i64], &zeros)),
        KernelArg::Buffer(TypedBuffer::from_f32(&[n as i64], &zeros)),
        KernelArg::Int(n as i128),
    ]
}

fn read(a: &[KernelArg], which: usize) -> Vec<f32> {
    let KernelArg::Buffer(b) = &a[which] else {
        panic!("a buffer")
    };
    b.bytes
        .chunks(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn launch(block: u32, grid: u32) -> LaunchShape {
    LaunchShape::new([block, 1, 1], [grid, 1, 1]).with_subgroup(4, 4)
}

fn load_in(into: u32) -> Stmt {
    Stmt::Load {
        local: into,
        place: buf(0, local(G)),
    }
}

fn store_tile(at: ScalarExpr, v: ScalarExpr) -> Stmt {
    Stmt::Store {
        place: tile(at),
        value: v,
    }
}

fn load_tile(into: u32, at: ScalarExpr) -> Stmt {
    Stmt::Load {
        local: into,
        place: tile(at),
    }
}

fn store_aux(v: ScalarExpr) -> Stmt {
    Stmt::Store {
        place: buf(2, local(G)),
        value: v,
    }
}

/// `blockDim - 1 - threadIdx`: the mirror thread in the block.
fn mirror() -> ScalarExpr {
    sub(sub(exec(ExecId::BlockDim(0)), u(1)), tid())
}

// ---- the tile reverse -------------------------------------------------------------------------

/// `tile[t] = in[g]; barrier; aux[g] = tile[mirror]`, with the barrier optional.
fn reverse(with_barrier: Option<Scope>, block: u32, grid: u32) -> KernelFn {
    let mut body = vec![let_g(), load_in(X), store_tile(tid(), local(X))];
    body.extend(with_barrier.map(barrier));
    body.push(load_tile(Y, mirror()));
    body.push(store_aux(local(Y)));
    kernel(body, launch(block, grid), block)
}

#[test]
fn a_barrier_lets_a_thread_read_a_neighbours_shared_store() {
    let mut a = args(8);
    run_launch(&reverse(Some(Scope::Block), 4, 2), &mut a).expect("runs");
    assert_eq!(read(&a, 2), vec![4.0, 3.0, 2.0, 1.0, 8.0, 7.0, 6.0, 5.0]);
}

#[test]
fn without_the_barrier_the_same_read_is_a_race() {
    // Thread 0 reads tile[3] before thread 3 has stored it: the missing-barrier bug.
    let r = run_launch(&reverse(None, 4, 2), &mut args(8));
    assert!(
        matches!(
            r,
            Err(KernelTrap::UninitializedShared { .. } | KernelTrap::DataRace)
        ),
        "{r:?}"
    );
}

// ---- a barrier inside a loop ------------------------------------------------------------------

/// Three rounds of `tile[t] = x; barrier; x = tile[mirror]; barrier`: an odd number of reversals.
fn rounds(second_barrier: bool) -> KernelFn {
    let mut inner = vec![
        store_tile(tid(), local(X)),
        barrier(Scope::Block),
        load_tile(Y, mirror()),
        Stmt::Assign {
            local: X,
            value: local(Y),
        },
    ];
    if second_barrier {
        inner.push(barrier(Scope::Block));
    }
    let body = vec![
        let_g(),
        load_in(X),
        Stmt::For {
            local: R,
            start: ci(0),
            end: ci(3),
            step: ci(1),
            body: Block(inner),
        },
        store_aux(local(X)),
    ];
    kernel(body, launch(4, 2), 4)
}

#[test]
fn a_barrier_in_a_loop_orders_every_round() {
    let mut a = args(8);
    run_launch(&rounds(true), &mut a).expect("runs");
    assert_eq!(read(&a, 2), vec![4.0, 3.0, 2.0, 1.0, 8.0, 7.0, 6.0, 5.0]);
}

#[test]
fn dropping_the_second_barrier_makes_the_next_rounds_store_race_the_last_read() {
    // The write-after-read hazard: round r+1 stores tile[t] while a slower thread still reads
    // round r's value. Only a reader log that survives the barrier-free stretch can see it.
    assert_eq!(
        run_launch(&rounds(false), &mut args(8)),
        Err(KernelTrap::DataRace)
    );
}

// ---- scopes -----------------------------------------------------------------------------------

/// Eight threads in two subgroups of four. After `scope` each thread reads `at`.
fn after(scope: Scope, at: ScalarExpr) -> KernelFn {
    let body = vec![
        let_g(),
        load_in(X),
        store_tile(tid(), local(X)),
        barrier(scope),
        load_tile(Y, at),
        store_aux(local(Y)),
    ];
    kernel(body, launch(8, 1), 8)
}

/// The lane mirrored inside its own subgroup.
fn lane_mirror() -> ScalarExpr {
    add(
        mul(exec(ExecId::SubgroupId), exec(ExecId::SubgroupSize)),
        sub(sub(exec(ExecId::SubgroupSize), u(1)), exec(ExecId::LaneId)),
    )
}

#[test]
fn a_subgroup_barrier_orders_the_lanes_of_its_subgroup() {
    let mut a = args(8);
    run_launch(&after(Scope::Subgroup, lane_mirror()), &mut a).expect("runs");
    assert_eq!(read(&a, 2), vec![4.0, 3.0, 2.0, 1.0, 8.0, 7.0, 6.0, 5.0]);
}

#[test]
fn a_subgroup_barrier_does_not_order_another_subgroup() {
    // Thread 0 reads tile[7], stored by a lane of the other subgroup.
    let across = sub(u(7), tid());
    assert_eq!(
        run_launch(&after(Scope::Subgroup, across.clone()), &mut args(8)),
        Err(KernelTrap::DataRace)
    );
    // Control: a block barrier orders the same read.
    let mut a = args(8);
    run_launch(&after(Scope::Block, across), &mut a).expect("a block barrier orders it");
    assert_eq!(read(&a, 2), vec![8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);
}

// ---- barriers order global memory too, within a block only ------------------------------------

#[test]
fn a_barrier_orders_a_global_store_against_later_reads_in_its_block() {
    // Thread 0 stores out[0]; after the barrier every thread reads it.
    let build = |with_barrier: bool| {
        let mut body = vec![
            let_g(),
            Stmt::If {
                cond: tid(),
                then: Block(vec![]),
                els: Block(vec![Stmt::Store {
                    place: buf(1, u(0)),
                    value: ScalarExpr::Const(7.0),
                }]),
            },
        ];
        if with_barrier {
            body.push(barrier(Scope::Block));
        }
        body.push(Stmt::Load {
            local: Y,
            place: buf(1, u(0)),
        });
        body.push(store_aux(local(Y)));
        kernel(body, launch(4, 1), 1)
    };
    let mut a = args(4);
    run_launch(&build(true), &mut a).expect("runs");
    assert_eq!(read(&a, 2), vec![7.0; 4]);
    assert_eq!(
        run_launch(&build(false), &mut args(4)),
        Err(KernelTrap::DataRace)
    );
}

#[test]
fn a_barrier_never_orders_two_blocks() {
    // Both blocks store out[0] after a block barrier of their own.
    let build = |at: ScalarExpr| {
        kernel(
            vec![
                barrier(Scope::Block),
                Stmt::Store {
                    place: buf(1, at),
                    value: ScalarExpr::Const(1.0),
                },
            ],
            launch(1, 2),
            1,
        )
    };
    run_launch(&build(exec(ExecId::BlockIdx(0))), &mut args(2)).expect("control: distinct");
    assert_eq!(
        run_launch(&build(u(0)), &mut args(2)),
        Err(KernelTrap::DataRace)
    );
}

// ---- barrier mismatch -------------------------------------------------------------------------

#[test]
fn a_thread_that_returns_before_a_barrier_strands_its_peers() {
    // The checker allows a Return before a barrier; the runtime does not.
    let build = |returning: bool| {
        let mut body = vec![let_g()];
        if returning {
            body.push(Stmt::If {
                cond: sub(tid(), u(3)),
                then: Block(vec![]),
                els: Block(vec![Stmt::Return]),
            });
        }
        body.push(barrier(Scope::Block));
        body.push(store_aux(ScalarExpr::Const(1.0)));
        kernel(body, launch(4, 1), 1)
    };
    run_launch(&build(false), &mut args(4)).expect("control: every thread arrives");
    assert_eq!(
        run_launch(&build(true), &mut args(4)),
        Err(KernelTrap::BarrierMismatch)
    );
}

#[test]
fn a_lane_that_returns_before_a_subgroup_barrier_strands_its_subgroup() {
    let build = |returning: bool| {
        let mut body = vec![let_g()];
        if returning {
            body.push(Stmt::If {
                cond: sub(tid(), u(1)),
                then: Block(vec![]),
                els: Block(vec![Stmt::Return]),
            });
        }
        body.push(barrier(Scope::Subgroup));
        body.push(store_aux(ScalarExpr::Const(1.0)));
        kernel(body, launch(4, 1), 1)
    };
    run_launch(&build(false), &mut args(4)).expect("control");
    assert_eq!(
        run_launch(&build(true), &mut args(4)),
        Err(KernelTrap::BarrierMismatch)
    );
}

#[test]
fn threads_that_all_return_without_reaching_a_barrier_are_fine() {
    // Mismatch is about stranded peers, not about barriers existing in the program.
    let k = kernel(vec![Stmt::Return], launch(4, 1), 1);
    run_launch(&k, &mut args(4)).expect("nobody waits");
}

#[test]
fn a_barrier_with_a_one_thread_scope_releases_at_once() {
    let k = kernel(
        vec![
            let_g(),
            barrier(Scope::Block),
            store_aux(ScalarExpr::Const(2.0)),
        ],
        launch(1, 3),
        1,
    );
    let mut a = args(3);
    run_launch(&k, &mut a).expect("runs");
    assert_eq!(read(&a, 2), vec![2.0; 3]);
}

#[test]
fn a_store_races_a_read_by_another_block_at_the_same_thread_index() {
    // Both blocks read out[0] as thread 0; only the block named `writer` then stores it. Reader
    // bookkeeping that tells readers apart by thread index alone would forget block 0's read.
    let build = |writer: i64| {
        kernel(
            vec![
                Stmt::Load {
                    local: Y,
                    place: buf(1, u(0)),
                },
                Stmt::If {
                    cond: sub(u(writer), exec(ExecId::BlockIdx(0))),
                    then: Block(vec![]),
                    els: Block(vec![Stmt::Store {
                        place: buf(1, u(0)),
                        value: ScalarExpr::Const(1.0),
                    }]),
                },
            ],
            launch(1, 2),
            1,
        )
    };
    run_launch(&build(9), &mut args(2)).expect("control: nobody stores");
    assert_eq!(
        run_launch(&build(1), &mut args(2)),
        Err(KernelTrap::DataRace)
    );
}
