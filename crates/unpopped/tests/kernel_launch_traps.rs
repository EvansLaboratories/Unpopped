// SPDX-License-Identifier: MIT OR Apache-2.0
//! U3 row 4 PR 2: the memory traps of `run_launch` (uninitialized reads, data races), split from
//! `kernel_launch.rs` to keep each file reviewable (`docs/u3-row4-scopes-and-memory-spaces.md`
//! section 5).
//!
//! Each trap has a program that raises it and its nearest legal variant that does not, so a test
//! goes red if the check is deleted, not merely if the interpreter breaks.

use unpopped::exec::{AllocDecl, ExecId, Extent, LaunchShape, MemorySpace};
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

fn local(i: u32) -> ScalarExpr {
    ScalarExpr::Local(i)
}

fn exec(id: ExecId) -> ScalarExpr {
    ScalarExpr::Exec(id)
}

fn u(v: i64) -> ScalarExpr {
    ScalarExpr::ConstInt(v, U32)
}

fn add(a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Add(Box::new(a), Box::new(b))
}

fn mul(a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Mul(Box::new(a), Box::new(b))
}

fn sub(a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Sub(Box::new(a), Box::new(b))
}

fn buf(buffer: u32, c: ScalarExpr) -> Place {
    Place::Logical {
        buffer,
        coords: vec![c],
    }
}

fn alloc(alloc: u32, index: ScalarExpr) -> Place {
    Place::Alloc { alloc, index }
}

/// `blockIdx.x * blockDim.x + threadIdx.x`
fn global_x() -> ScalarExpr {
    add(
        mul(exec(ExecId::BlockIdx(0)), exec(ExecId::BlockDim(0))),
        exec(ExecId::ThreadIdx(0)),
    )
}

fn let_g() -> Stmt {
    Stmt::Let {
        local: G,
        init: global_x(),
    }
}

fn kernel(body: Vec<Stmt>, allocs: Vec<AllocDecl>, launch: LaunchShape) -> KernelFn {
    KernelFn::new(
        "launch",
        F32,
        vec![
            Param::buffer("in", F32, BufferRole::Read, Some(1)),
            Param::buffer("out", F32, BufferRole::ReadWrite, Some(1)),
            Param::scalar("n", I32, N),
        ],
        vec![
            LocalDecl::scalar(I32),
            LocalDecl::scalar(U32),
            LocalDecl::scalar(F32),
            LocalDecl::scalar(F32),
            LocalDecl::scalar(U32),
        ],
        Block(body),
    )
    .with_launch(launch)
    .with_allocs(allocs)
}

fn args(n: usize) -> Vec<KernelArg> {
    let input: Vec<f32> = (0..n).map(|i| i as f32 + 1.0).collect();
    vec![
        KernelArg::Buffer(TypedBuffer::from_f32(&[n as i64], &input)),
        KernelArg::Buffer(TypedBuffer::from_f32(&[n as i64], &vec![0.0; n])),
        KernelArg::Int(n as i128),
    ]
}

fn out_f32(a: &[KernelArg]) -> Vec<f32> {
    let KernelArg::Buffer(b) = &a[1] else {
        panic!("out is a buffer")
    };
    b.bytes
        .chunks(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn shared(len: u32) -> Vec<AllocDecl> {
    vec![AllocDecl::new(
        "tile",
        MemorySpace::Shared,
        F32,
        Extent::Static(len),
    )]
}

fn launch(block: u32, grid: u32) -> LaunchShape {
    LaunchShape::new([block, 1, 1], [grid, 1, 1]).with_subgroup(4, 8)
}

fn load_in(into: u32, at: ScalarExpr) -> Stmt {
    Stmt::Load {
        local: into,
        place: buf(0, at),
    }
}

fn store_out(at: ScalarExpr, value: ScalarExpr) -> Stmt {
    Stmt::Store {
        place: buf(1, at),
        value,
    }
}

// ---- uninitialized memory ---------------------------------------------------------------------

/// `y = tile[t]` with no store first.
fn read_before_store(space: MemorySpace, store_first: bool) -> KernelFn {
    let t = exec(ExecId::ThreadIdx(0));
    let mut body = vec![let_g()];
    if store_first {
        body.push(Stmt::Store {
            place: alloc(0, t.clone()),
            value: ScalarExpr::Const(3.0),
        });
    }
    body.push(Stmt::Load {
        local: Y,
        place: alloc(0, t),
    });
    body.push(store_out(local(G), local(Y)));
    kernel(
        body,
        vec![AllocDecl::new("a", space, F32, Extent::Static(4))],
        launch(4, 1),
    )
}

#[test]
fn reading_an_unstored_shared_element_traps() {
    let mut a = args(4);
    run_launch(&read_before_store(MemorySpace::Shared, true), &mut a).expect("control");
    assert_eq!(out_f32(&a), vec![3.0; 4]);
    assert_eq!(
        run_launch(&read_before_store(MemorySpace::Shared, false), &mut args(4)),
        Err(KernelTrap::UninitializedShared { alloc: 0 })
    );
}

#[test]
fn reading_an_unstored_private_element_traps() {
    let mut a = args(4);
    run_launch(&read_before_store(MemorySpace::Private, true), &mut a).expect("control");
    assert_eq!(
        run_launch(
            &read_before_store(MemorySpace::Private, false),
            &mut args(4)
        ),
        Err(KernelTrap::UninitializedPrivate { alloc: 0 })
    );
}

#[test]
fn private_memory_does_not_leak_between_threads() {
    // Thread 0 stores scratch[0]; every thread then reads it. Thread 1 has its own instance.
    let t = exec(ExecId::ThreadIdx(0));
    let build = |store_in_all: bool| {
        let store = Stmt::Store {
            place: alloc(0, u(0)),
            value: ScalarExpr::Const(1.0),
        };
        let guarded = if store_in_all {
            store
        } else {
            Stmt::If {
                cond: t.clone(),
                then: Block(vec![]),
                els: Block(vec![store]),
            }
        };
        kernel(
            vec![
                let_g(),
                guarded,
                Stmt::Load {
                    local: Y,
                    place: alloc(0, u(0)),
                },
                store_out(local(G), local(Y)),
            ],
            vec![AllocDecl::new(
                "s",
                MemorySpace::Private,
                F32,
                Extent::Static(1),
            )],
            launch(2, 1),
        )
    };
    run_launch(&build(true), &mut args(2)).expect("control: every thread stores");
    assert_eq!(
        run_launch(&build(false), &mut args(2)),
        Err(KernelTrap::UninitializedPrivate { alloc: 0 })
    );
}

#[test]
fn shared_memory_does_not_leak_between_blocks() {
    // Only block 0 stores tile[0]; block 1 then reads it. Each block has its own instance.
    let build = |store_in_all: bool| {
        let store = Stmt::Store {
            place: alloc(0, u(0)),
            value: ScalarExpr::Const(1.0),
        };
        let guarded = if store_in_all {
            store
        } else {
            Stmt::If {
                cond: exec(ExecId::BlockIdx(0)),
                then: Block(vec![]),
                els: Block(vec![store]),
            }
        };
        kernel(
            vec![
                let_g(),
                guarded,
                Stmt::Load {
                    local: Y,
                    place: alloc(0, u(0)),
                },
                store_out(local(G), local(Y)),
            ],
            shared(1),
            // One thread per block so the store/read pair is never cross-thread.
            launch(1, 2),
        )
    };
    run_launch(&build(true), &mut args(2)).expect("control: every block stores");
    assert_eq!(
        run_launch(&build(false), &mut args(2)),
        Err(KernelTrap::UninitializedShared { alloc: 0 })
    );
}

// ---- data races -------------------------------------------------------------------------------

#[test]
fn two_threads_storing_one_global_element_race() {
    let build = |at: ScalarExpr| {
        kernel(
            vec![let_g(), store_out(at, ScalarExpr::Const(1.0))],
            vec![],
            launch(4, 1),
        )
    };
    run_launch(&build(global_x()), &mut args(4)).expect("control: distinct elements");
    assert_eq!(
        run_launch(&build(u(0)), &mut args(4)),
        Err(KernelTrap::DataRace)
    );
}

#[test]
fn two_blocks_storing_one_global_element_race() {
    let build = |at: ScalarExpr| {
        kernel(
            vec![store_out(at, ScalarExpr::Const(1.0))],
            vec![],
            launch(1, 2),
        )
    };
    run_launch(&build(exec(ExecId::BlockIdx(0))), &mut args(2)).expect("control");
    assert_eq!(
        run_launch(&build(u(0)), &mut args(2)),
        Err(KernelTrap::DataRace)
    );
}

#[test]
fn reading_another_threads_global_store_races() {
    // out[g] = 1; y = out[(g + 1) % ...]: reads the neighbour's store. Control reads its own.
    let build = |read_at: ScalarExpr| {
        kernel(
            vec![
                let_g(),
                store_out(local(G), ScalarExpr::Const(1.0)),
                Stmt::Load {
                    local: Y,
                    place: buf(1, read_at),
                },
            ],
            vec![],
            launch(4, 1),
        )
    };
    run_launch(&build(local(G)), &mut args(4)).expect("control: its own element");
    assert_eq!(
        run_launch(&build(sub(u(3), local(G))), &mut args(4)),
        Err(KernelTrap::DataRace)
    );
}

#[test]
fn a_read_of_an_element_a_lower_thread_stored_races() {
    // Every thread reads out[0] then stores out[t]. Thread 0 reads and stores its own element;
    // thread 1 then reads out[0], which thread 0 wrote.
    let k = kernel(
        vec![
            let_g(),
            Stmt::Load {
                local: Y,
                place: buf(1, u(0)),
            },
            store_out(local(G), local(Y)),
        ],
        vec![],
        launch(4, 1),
    );
    assert_eq!(run_launch(&k, &mut args(4)), Err(KernelTrap::DataRace));
    // Control: the same loop reading the read-only input is not a race.
    let ok = kernel(
        vec![let_g(), load_in(Y, u(0)), store_out(local(G), local(Y))],
        vec![],
        launch(4, 1),
    );
    run_launch(&ok, &mut args(4)).expect("a broadcast read of an input");
}

#[test]
fn a_shared_read_of_another_threads_store_is_the_missing_barrier_bug() {
    // tile[t] = x; y = tile[0]. With a barrier between them this is a broadcast from thread 0;
    // without one, thread 1 reads what thread 0 stored, and the sequential driver must say so
    // rather than let the order the threads happen to run in satisfy the read.
    let t = exec(ExecId::ThreadIdx(0));
    let build = |read_at: ScalarExpr| {
        kernel(
            vec![
                let_g(),
                load_in(X, local(G)),
                Stmt::Store {
                    place: alloc(0, t.clone()),
                    value: local(X),
                },
                Stmt::Load {
                    local: Y,
                    place: alloc(0, read_at),
                },
                store_out(local(G), local(Y)),
            ],
            shared(4),
            launch(4, 1),
        )
    };
    run_launch(&build(t.clone()), &mut args(4)).expect("control: its own element");
    assert_eq!(
        run_launch(&build(u(0)), &mut args(4)),
        Err(KernelTrap::DataRace)
    );
}

#[test]
fn readers_then_a_later_writer_race() {
    // Every thread reads in-out element 0 (a broadcast), then thread 3 stores it.
    let t = exec(ExecId::ThreadIdx(0));
    let build = |writer: ScalarExpr| {
        kernel(
            vec![
                Stmt::Load {
                    local: Y,
                    place: buf(1, u(0)),
                },
                Stmt::If {
                    cond: sub(t.clone(), writer),
                    then: Block(vec![]),
                    els: Block(vec![store_out(u(0), ScalarExpr::Const(1.0))]),
                },
            ],
            vec![],
            launch(4, 1),
        )
    };
    // Writer is a thread index no thread has (9): nobody stores. Control.
    run_launch(&build(u(9)), &mut args(4)).expect("control: no writer");
    assert_eq!(
        run_launch(&build(u(3)), &mut args(4)),
        Err(KernelTrap::DataRace)
    );
}
