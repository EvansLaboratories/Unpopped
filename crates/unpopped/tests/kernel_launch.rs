// SPDX-License-Identifier: MIT OR Apache-2.0
//! U3 row 4 PR 2: `run_launch`, the per-thread driver for barrier-free kernels
//! (`docs/u3-row4-scopes-and-memory-spaces.md` section 5, steps 1, 3 and 4).
//!
//! Each trap has a program that raises it and its nearest legal variant that does not, so a test
//! goes red if the check is deleted, not merely if the interpreter breaks.

use unpopped::exec::{AllocDecl, ExecId, Extent, LaunchShape, MemorySpace, Scope};
use unpopped::ir::ScalarExpr;
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::{KernelArg, KernelTrap, run_kernel};
use unpopped::oracle::kernel_launch::{run_launch, run_launch_with_subgroup};
use unpopped_vocab::ElementKind;

const F32: ElementKind = ElementKind::F32;
const I32: ElementKind = ElementKind::I32;
const U32: ElementKind = ElementKind::U32;

const N: u32 = 0; // scalar param n: i32
const G: u32 = 1; // global thread index: u32
const X: u32 = 2; // f32
const Y: u32 = 3; // f32
const P: u32 = 4; // u32 scratch

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

// ---- it computes ------------------------------------------------------------------------------

/// `g = blockIdx*blockDim + threadIdx; out[g] = in[g] * 2`
fn doubling() -> Vec<Stmt> {
    vec![
        let_g(),
        load_in(X, local(G)),
        store_out(
            local(G),
            ScalarExpr::Mul(Box::new(local(X)), Box::new(ScalarExpr::Const(2.0))),
        ),
    ]
}

#[test]
fn every_thread_of_every_block_runs_once() {
    let k = kernel(doubling(), vec![], launch(4, 3));
    let mut a = args(12);
    run_launch(&k, &mut a).expect("runs");
    let want: Vec<f32> = (1..=12).map(|v| v as f32 * 2.0).collect();
    assert_eq!(out_f32(&a), want);
}

#[test]
fn a_thread_may_use_shared_memory_as_its_own_scratch() {
    // tile[t] = in[g]; out[g] = tile[t] + 1: each thread reads only what it stored.
    let t = exec(ExecId::ThreadIdx(0));
    let k = kernel(
        vec![
            let_g(),
            load_in(X, local(G)),
            Stmt::Store {
                place: alloc(0, t.clone()),
                value: local(X),
            },
            Stmt::Load {
                local: Y,
                place: alloc(0, t),
            },
            store_out(
                local(G),
                ScalarExpr::Add(Box::new(local(Y)), Box::new(ScalarExpr::Const(1.0))),
            ),
        ],
        shared(4),
        launch(4, 2),
    );
    let mut a = args(8);
    run_launch(&k, &mut a).expect("runs");
    let want: Vec<f32> = (1..=8).map(|v| v as f32 + 1.0).collect();
    assert_eq!(out_f32(&a), want);
}

#[test]
fn a_private_array_is_per_thread() {
    let k = kernel(
        vec![
            let_g(),
            load_in(X, local(G)),
            Stmt::Store {
                place: alloc(0, u(1)),
                value: local(X),
            },
            Stmt::Load {
                local: Y,
                place: alloc(0, u(1)),
            },
            store_out(local(G), local(Y)),
        ],
        vec![AllocDecl::new(
            "scratch",
            MemorySpace::Private,
            F32,
            Extent::Static(2),
        )],
        launch(4, 2),
    );
    let mut a = args(8);
    run_launch(&k, &mut a).expect("runs");
    let want: Vec<f32> = (1..=8).map(|v| v as f32).collect();
    assert_eq!(out_f32(&a), want);
}

// ---- execution ids ----------------------------------------------------------------------------

fn id_kernel(id: ExecId, shape: LaunchShape) -> KernelFn {
    // out[linear block * threads per block + linear thread] = id, as f32.
    let [bx, by, _] = shape.block;
    let [gx, gy, _] = shape.grid;
    let id_ = |f: fn(u8) -> ExecId, a: u8| exec(f(a));
    let lin_t = add(
        id_(ExecId::ThreadIdx, 0),
        mul(
            u(i64::from(bx)),
            add(
                id_(ExecId::ThreadIdx, 1),
                mul(u(i64::from(by)), id_(ExecId::ThreadIdx, 2)),
            ),
        ),
    );
    let lin_b = add(
        id_(ExecId::BlockIdx, 0),
        mul(
            u(i64::from(gx)),
            add(
                id_(ExecId::BlockIdx, 1),
                mul(u(i64::from(gy)), id_(ExecId::BlockIdx, 2)),
            ),
        ),
    );
    let per_block = i64::from(shape.block.iter().product::<u32>());
    let slot = add(mul(lin_b, u(per_block)), lin_t);
    let mut k = kernel(
        vec![
            Stmt::Let {
                local: G,
                init: slot,
            },
            Stmt::Let {
                local: P,
                init: exec(id),
            },
            store_out(
                local(G),
                ScalarExpr::Cast(
                    Box::new(local(P)),
                    unpopped::ir::CastSpec {
                        to: F32,
                        round: unpopped::ir::RoundMode::NearestEven,
                        sat: unpopped::ir::Saturation::Wrap,
                    },
                ),
            ),
        ],
        vec![],
        shape,
    );
    k.name = "ids".into();
    k
}

#[test]
fn every_exec_id_reads_its_value() {
    let shape = LaunchShape::new([4, 3, 2], [2, 2, 3]).with_subgroup(2, 8);
    let threads = 24usize;
    let blocks = 12usize;
    let sg = 2u32; // run_launch picks the range's minimum
    let cases: Vec<(ExecId, Box<dyn Fn(usize, usize) -> u32>)> = vec![
        (ExecId::ThreadIdx(0), Box::new(|_, t| (t % 4) as u32)),
        (ExecId::ThreadIdx(1), Box::new(|_, t| ((t / 4) % 3) as u32)),
        (ExecId::ThreadIdx(2), Box::new(|_, t| (t / 12) as u32)),
        (ExecId::BlockIdx(0), Box::new(|b, _| (b % 2) as u32)),
        (ExecId::BlockIdx(1), Box::new(|b, _| ((b / 2) % 2) as u32)),
        (ExecId::BlockIdx(2), Box::new(|b, _| (b / 4) as u32)),
        (ExecId::BlockDim(1), Box::new(|_, _| 3)),
        (ExecId::GridDim(2), Box::new(|_, _| 3)),
        (ExecId::LaneId, Box::new(move |_, t| t as u32 % sg)),
        (ExecId::SubgroupId, Box::new(move |_, t| t as u32 / sg)),
        (ExecId::SubgroupSize, Box::new(move |_, _| sg)),
    ];
    for (id, want) in cases {
        let k = id_kernel(id, shape);
        let mut a = vec![
            KernelArg::Buffer(TypedBuffer::from_f32(&[1], &[0.0])),
            KernelArg::Buffer(TypedBuffer::from_f32(
                &[(blocks * threads) as i64],
                &vec![0.0; blocks * threads],
            )),
            KernelArg::Int(0),
        ];
        run_launch(&k, &mut a).expect("runs");
        let got = out_f32(&a);
        for b in 0..blocks {
            for t in 0..threads {
                assert_eq!(got[b * threads + t], want(b, t) as f32, "{id:?} b{b} t{t}");
            }
        }
    }
}

#[test]
fn the_subgroup_size_can_be_chosen_inside_the_range() {
    let shape = LaunchShape::new([8, 1, 1], [1, 1, 1]).with_subgroup(2, 8);
    let k = id_kernel(ExecId::LaneId, shape);
    let mk = || {
        vec![
            KernelArg::Buffer(TypedBuffer::from_f32(&[1], &[0.0])),
            KernelArg::Buffer(TypedBuffer::from_f32(&[8], &[0.0; 8])),
            KernelArg::Int(0),
        ]
    };
    let mut a = mk();
    run_launch_with_subgroup(&k, &mut a, Some(4)).expect("runs");
    assert_eq!(out_f32(&a), vec![0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0]);
    for bad in [1, 9] {
        assert_eq!(
            run_launch_with_subgroup(&k, &mut mk(), Some(bad)),
            Err(KernelTrap::ArgMismatch(
                "subgroup size outside the launch's range"
            )),
            "{bad}"
        );
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
    // tile[t] = x; y = tile[(t + 1) % 4]. With a barrier between them this is the standard
    // neighbour read; without one it is a race, and the sequential driver must say so rather
    // than let thread 0's store satisfy thread 3's read.
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
    // Thread 0 reads tile[3] before thread 3 has run: uninitialized, which is also the bug.
    let r = run_launch(&build(sub(u(3), t.clone())), &mut args(4));
    assert!(
        matches!(
            r,
            Err(KernelTrap::UninitializedShared { .. } | KernelTrap::DataRace)
        ),
        "{r:?}"
    );
}

#[test]
fn many_readers_then_a_writer_race() {
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

// ---- bounds and arguments ---------------------------------------------------------------------

#[test]
fn an_index_outside_an_allocation_traps() {
    let build = |len: u32| {
        kernel(
            vec![
                let_g(),
                Stmt::Store {
                    place: alloc(0, u(3)),
                    value: ScalarExpr::Const(1.0),
                },
                store_out(local(G), ScalarExpr::Const(1.0)),
            ],
            shared(len),
            launch(1, 1),
        )
    };
    run_launch(&build(4), &mut args(1)).expect("control: index 3 of 4");
    assert_eq!(
        run_launch(&build(3), &mut args(1)),
        Err(KernelTrap::AllocOutOfBounds { alloc: 0 })
    );
}

#[test]
fn a_dynamic_extent_comes_from_its_argument() {
    let k = kernel(
        vec![
            let_g(),
            Stmt::Store {
                place: alloc(0, u(5)),
                value: ScalarExpr::Const(1.0),
            },
            store_out(local(G), ScalarExpr::Const(1.0)),
        ],
        vec![AllocDecl::new(
            "dyn",
            MemorySpace::Shared,
            F32,
            Extent::Param(N),
        )],
        launch(1, 1),
    );
    let with_n = |n: i128| {
        let mut a = args(1);
        a[2] = KernelArg::Int(n);
        a
    };
    run_launch(&k, &mut with_n(6)).expect("control: n = 6 holds index 5");
    assert_eq!(
        run_launch(&k, &mut with_n(5)),
        Err(KernelTrap::AllocOutOfBounds { alloc: 0 })
    );
    assert!(matches!(
        run_launch(&k, &mut with_n(0)),
        Err(KernelTrap::ArgMismatch(_))
    ));
}

#[test]
fn run_launch_needs_a_launch_and_run_kernel_refuses_one() {
    let mut plain = kernel(doubling(), vec![], launch(1, 1));
    plain.launch = None;
    // Exec leaves need a launch, so check a body that has none.
    plain.body = Block(vec![store_out(u(0), ScalarExpr::Const(1.0))]);
    assert!(matches!(
        run_launch(&plain, &mut args(1)),
        Err(KernelTrap::ArgMismatch(_))
    ));
    assert_eq!(run_kernel(&plain, &mut args(1)), Ok(()), "control");
    let launched = kernel(doubling(), vec![], launch(1, 1));
    assert!(matches!(
        run_kernel(&launched, &mut args(1)),
        Err(KernelTrap::Unsupported(_))
    ));
}

#[test]
fn a_barrier_and_a_constant_read_are_declined_by_name() {
    let barrier = kernel(
        vec![Stmt::Barrier {
            scope: Scope::Block,
        }],
        vec![],
        launch(1, 1),
    );
    assert!(matches!(
        run_launch(&barrier, &mut args(1)),
        Err(KernelTrap::Unsupported(m)) if m.contains("Barrier")
    ));
    let constant = kernel(
        vec![Stmt::Load {
            local: Y,
            place: alloc(0, u(0)),
        }],
        vec![AllocDecl::new(
            "c",
            MemorySpace::Constant,
            F32,
            Extent::Static(2),
        )],
        launch(1, 1),
    );
    assert!(matches!(
        run_launch(&constant, &mut args(1)),
        Err(KernelTrap::Unsupported(m)) if m.contains("Constant")
    ));
}
