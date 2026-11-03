// SPDX-License-Identifier: MIT OR Apache-2.0
//! ROADMAP U4, slice S5 (`docs/question-u4-memory-model.md`, option A): `Stmt::Fence { scope, order }` turns the
//! relaxed atomics next to it into the release/acquire of the C++ fence rules. A release fence (`Release`, `AcqRel`,
//! `SeqCst`) *before* a relaxed atomic write publishes what the thread knew at the fence; an acquire fence
//! (`Acquire`, `AcqRel`, `SeqCst`) *after* a relaxed atomic read takes in what that read saw. A fence orders no memory
//! of its own. `check_kernel` refuses a `Relaxed` fence (it would order nothing) and the scopes `Thread` and `Cluster`.
//!
//! Every kernel is message passing as in `kernel_atomic_order.rs`: the producer writes `data[0] = 42` and sets
//! `flag[0]` with a relaxed atomic; the consumer reads the flag with a relaxed atomic and, if set, reads the data
//! into `out[0]`. Only the fences differ, and with them the oracle's verdict.

use unpopped::exec::{ExecId, LaunchShape, Scope};
use unpopped::ir::{BinaryOp, ScalarExpr};
use unpopped::kernel::{
    AtomicOp, Block, BufferRole, KernelError, KernelFn, LocalDecl, MemoryOrder, Param, Place, Stmt,
    check_kernel,
};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::{KernelArg, KernelTrap, run_kernel};
use unpopped::oracle::kernel_launch::run_launch_with_limit;
use unpopped_vocab::ElementKind;

const I32: ElementKind = ElementKind::I32;
const U32: ElementKind = ElementKind::U32;
const DEV: Scope = Scope::Device;

use MemoryOrder::{AcqRel, Acquire, Relaxed, Release, SeqCst};

fn c(v: i64) -> ScalarExpr {
    ScalarExpr::ConstInt(v, I32)
}

fn at(buffer: u32, i: u32) -> Place {
    Place::Logical {
        buffer,
        coords: vec![ScalarExpr::ConstInt(i64::from(i), U32)],
    }
}

fn local(i: u32) -> ScalarExpr {
    ScalarExpr::Local(i)
}

fn is(by_block: bool, t: i64) -> ScalarExpr {
    let id = if by_block {
        ExecId::BlockIdx(0)
    } else {
        ExecId::ThreadIdx(0)
    };
    ScalarExpr::Binary(
        BinaryOp::CmpEq,
        Box::new(ScalarExpr::Exec(id)),
        Box::new(ScalarExpr::ConstInt(t, U32)),
    )
}

fn when(cond: ScalarExpr, then: Vec<Stmt>) -> Stmt {
    Stmt::If {
        cond,
        then: Block(then),
        els: Block(vec![]),
    }
}

fn fence(order: MemoryOrder, scope: Scope) -> Stmt {
    Stmt::Fence { scope, order }
}

fn set_flag() -> Stmt {
    Stmt::AtomicRmw {
        op: AtomicOp::Exchange,
        place: at(1, 0),
        value: c(1),
        old: None,
        order: Relaxed,
        scope: DEV,
    }
}

fn read_flag() -> Stmt {
    Stmt::AtomicRmw {
        op: AtomicOp::Or,
        place: at(1, 0),
        value: c(0),
        old: Some(0),
        order: Relaxed,
        scope: DEV,
    }
}

fn seen() -> ScalarExpr {
    ScalarExpr::Binary(BinaryOp::CmpEq, Box::new(local(0)), Box::new(c(1)))
}

fn copy_data() -> Vec<Stmt> {
    vec![
        Stmt::Load {
            local: 1,
            place: at(0, 0),
        },
        Stmt::Store {
            place: at(2, 0),
            value: local(1),
        },
    ]
}

fn kernel(block: u32, grid: u32, body: Vec<Stmt>) -> KernelFn {
    let params = vec![
        Param::buffer("data", I32, BufferRole::ReadWrite, Some(1)),
        Param::buffer("flag", I32, BufferRole::ReadWrite, Some(1)),
        Param::buffer("out", I32, BufferRole::ReadWrite, Some(1)),
    ];
    let locals = vec![LocalDecl::scalar(I32), LocalDecl::scalar(I32)];
    let launch = LaunchShape::new([block, 1, 1], [grid, 1, 1]).with_subgroup(4, 4);
    KernelFn::new("mp", I32, params, locals, Block(body)).with_launch(launch)
}

fn buf(n: i64) -> KernelArg {
    KernelArg::Buffer(TypedBuffer::new(
        I32,
        vec![n],
        vec![1],
        vec![0; 4 * n as usize],
    ))
}

fn run(k: &KernelFn) -> Result<i32, KernelTrap> {
    assert_eq!(check_kernel(k), Ok(()), "well formed");
    let mut args = vec![buf(4), buf(4), buf(4)];
    run_launch_with_limit(k, &mut args, 20_000)?;
    let KernelArg::Buffer(out) = &args[2] else {
        unreachable!()
    };
    Ok(i32::from_le_bytes(out.bytes[..4].try_into().unwrap()))
}

/// Thread 0 produces, thread 1 consumes; each side's fence is optional and placed where the rules need it.
fn fenced(rel: Option<(MemoryOrder, Scope)>, acq: Option<(MemoryOrder, Scope)>) -> KernelFn {
    let mut produce = vec![Stmt::Store {
        place: at(0, 0),
        value: c(42),
    }];
    produce.extend(rel.map(|(o, s)| fence(o, s)));
    produce.push(set_flag());
    let mut consume = vec![read_flag()];
    consume.extend(acq.map(|(o, s)| fence(o, s)));
    consume.push(when(seen(), copy_data()));
    kernel(
        2,
        1,
        vec![when(is(false, 0), produce), when(is(false, 1), consume)],
    )
}

#[test]
fn a_release_fence_and_an_acquire_fence_order_relaxed_message_passing() {
    for (rel, acq, race) in [
        (Release, Acquire, false),
        (AcqRel, AcqRel, false),
        (SeqCst, SeqCst, false),
        (Release, SeqCst, false),
        // either fence missing, or the wrong way round
        (Release, Relaxed, true),
        (Relaxed, Acquire, true),
        (Acquire, Acquire, true),
        (Release, Release, true),
    ] {
        let side = |o| (o != Relaxed).then_some((o, DEV));
        let r = run(&fenced(side(rel), side(acq)));
        if race {
            assert_eq!(r, Err(KernelTrap::DataRace), "{rel:?} fence, {acq:?} fence");
        } else {
            assert_eq!(r, Ok(42), "{rel:?} fence, {acq:?} fence");
        }
    }
}

#[test]
fn no_fences_at_all_is_a_race() {
    assert_eq!(run(&fenced(None, None)), Err(KernelTrap::DataRace));
}

#[test]
fn the_release_fence_must_precede_the_atomic_and_the_acquire_fence_follow_the_read() {
    // producer: data; flag; fence (the fence is after the write that publishes)
    let late_release = kernel(
        2,
        1,
        vec![
            when(
                is(false, 0),
                vec![
                    Stmt::Store {
                        place: at(0, 0),
                        value: c(42),
                    },
                    set_flag(),
                    fence(Release, DEV),
                ],
            ),
            when(
                is(false, 1),
                vec![read_flag(), fence(Acquire, DEV), when(seen(), copy_data())],
            ),
        ],
    );
    assert_eq!(run(&late_release), Err(KernelTrap::DataRace));
    // consumer: fence; read (the fence is before the read it should follow)
    let early_acquire = kernel(
        2,
        1,
        vec![
            when(
                is(false, 0),
                vec![
                    Stmt::Store {
                        place: at(0, 0),
                        value: c(42),
                    },
                    fence(Release, DEV),
                    set_flag(),
                ],
            ),
            when(
                is(false, 1),
                vec![fence(Acquire, DEV), read_flag(), when(seen(), copy_data())],
            ),
        ],
    );
    assert_eq!(run(&early_acquire), Err(KernelTrap::DataRace));
}

#[test]
fn a_fence_reaches_only_as_far_as_its_scope() {
    let two_blocks = |rel_scope| {
        kernel(
            1,
            2,
            vec![
                when(
                    is(true, 0),
                    vec![
                        Stmt::Store {
                            place: at(0, 0),
                            value: c(42),
                        },
                        fence(Release, rel_scope),
                        set_flag(),
                    ],
                ),
                when(
                    is(true, 1),
                    vec![read_flag(), fence(Acquire, DEV), when(seen(), copy_data())],
                ),
            ],
        )
    };
    assert_eq!(
        run(&two_blocks(DEV)),
        Ok(42),
        "control: a Device-scope fence"
    );
    assert_eq!(run(&two_blocks(Scope::Block)), Err(KernelTrap::DataRace));
}

#[test]
fn an_acquire_fence_takes_in_every_read_before_it_not_just_the_last() {
    // the consumer reads the published flag, then an unrelated one, and only then fences
    let read_other = Stmt::AtomicRmw {
        op: AtomicOp::Or,
        place: at(1, 1),
        value: c(0),
        old: None,
        order: Relaxed,
        scope: DEV,
    };
    let k = kernel(
        2,
        1,
        vec![
            when(
                is(false, 0),
                vec![
                    Stmt::Store {
                        place: at(0, 0),
                        value: c(42),
                    },
                    fence(Release, DEV),
                    set_flag(),
                ],
            ),
            when(
                is(false, 1),
                vec![
                    read_flag(),
                    read_other,
                    fence(Acquire, DEV),
                    when(seen(), copy_data()),
                ],
            ),
        ],
    );
    assert_eq!(run(&k), Ok(42));
}

#[test]
fn a_wait_made_of_a_relaxed_read_and_an_acquire_fence_is_a_named_cross_block_wait() {
    let spin = Stmt::While {
        cond: c(1),
        body: Block(vec![
            read_flag(),
            fence(Acquire, DEV),
            when(seen(), vec![Stmt::Break]),
        ]),
    };
    let k = kernel(
        1,
        2,
        vec![
            when(is(true, 0), vec![spin]),
            when(is(true, 1), vec![set_flag()]),
        ],
    );
    assert_eq!(run(&k), Err(KernelTrap::CrossBlockWait));
}

#[test]
fn the_checker_refuses_a_relaxed_fence_and_the_unmodelled_scopes() {
    for (order, scope) in [
        (Relaxed, DEV),
        (Release, Scope::Thread),
        (SeqCst, Scope::Cluster),
    ] {
        let k = kernel(1, 1, vec![fence(order, scope)]);
        assert!(
            matches!(check_kernel(&k), Err(KernelError::SyncUnsupported { .. })),
            "{order:?} {scope:?}"
        );
    }
    for scope in [Scope::Subgroup, Scope::Block, DEV] {
        assert_eq!(
            check_kernel(&kernel(1, 1, vec![fence(AcqRel, scope)])),
            Ok(())
        );
    }
}

#[test]
fn a_fence_outside_a_launch_runs_and_does_nothing() {
    let params = vec![Param::buffer("o", I32, BufferRole::ReadWrite, Some(1))];
    let body = vec![
        fence(SeqCst, DEV),
        Stmt::Store {
            place: at(0, 0),
            value: c(7),
        },
    ];
    let k = KernelFn::new("f", I32, params, vec![], Block(body));
    let mut args = vec![buf(1)];
    run_kernel(&k, &mut args).expect("runs");
    let KernelArg::Buffer(o) = &args[0] else {
        unreachable!()
    };
    assert_eq!(i32::from_le_bytes(o.bytes[..4].try_into().unwrap()), 7);
}
