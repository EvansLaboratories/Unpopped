// SPDX-License-Identifier: MIT OR Apache-2.0
//! ROADMAP U4, slice S4 (`docs/question-u4-memory-model.md`, option A, ruled by CireSnave 2026-11-03): an atomic
//! carries a memory order and a scope, and the oracle's race rule honours them. A release-class atomic followed, in
//! another thread, by an acquire-class atomic on the same element orders the plain accesses before the first against
//! the plain accesses after the second; with `Relaxed` on either side the same kernel is a `DataRace`. Atomics at a
//! scope that does not contain both threads are a `DataRace`. `SeqCst` is treated as `AcqRel`. A block that spins on
//! a later block's write is `CrossBlockWait`, not a hang.
//!
//! Every kernel is message passing: a producer writes `data[0] = 42` (a plain store) and publishes with an atomic on
//! `flag[..]`; a consumer reads the flag atomically and, if it saw the publication, reads `data[0]` (a plain load)
//! into `out[0]`. The oracle runs threads in index order and blocks in order, so the producer is always the lower id
//! and the consumer sees the flag set: the only question is whether the oracle calls the program race-free.

use unpopped::exec::{ExecId, LaunchShape, Scope};
use unpopped::ir::{BinaryOp, ScalarExpr};
use unpopped::kernel::{
    AtomicOp, Block, BufferRole, KernelError, KernelFn, LocalDecl, MemoryOrder, Param, Place, Stmt,
    check_kernel,
};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::{KernelArg, KernelTrap};
use unpopped::oracle::kernel_launch::run_launch_with_limit;
use unpopped_vocab::ElementKind;

const I32: ElementKind = ElementKind::I32;
const U32: ElementKind = ElementKind::U32;

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

/// `thread == t` (axis 0 of the thread id), or `block == t` when `by_block`.
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

fn store(buffer: u32, i: u32, v: i64) -> Stmt {
    Stmt::Store {
        place: at(buffer, i),
        value: c(v),
    }
}

/// A release-style publication: `flag[i] = v` as an exchange.
fn publish(i: u32, v: i64, (order, scope): (MemoryOrder, Scope)) -> Stmt {
    Stmt::AtomicRmw {
        op: AtomicOp::Exchange,
        place: at(1, i),
        value: c(v),
        old: None,
        order,
        scope,
    }
}

/// An atomic read of `flag[i]` into `local 0` (an `Or` of zero).
fn observe(i: u32, (order, scope): (MemoryOrder, Scope)) -> Stmt {
    Stmt::AtomicRmw {
        op: AtomicOp::Or,
        place: at(1, i),
        value: c(0),
        old: Some(0),
        order,
        scope,
    }
}

/// `out[0] = data[0]`, through `local 1`.
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

fn seen(v: i64) -> ScalarExpr {
    ScalarExpr::Binary(BinaryOp::CmpEq, Box::new(local(0)), Box::new(c(v)))
}

const DEV: Scope = Scope::Device;

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

/// Run on zeroed buffers and return `out[0]`.
fn run(k: &KernelFn) -> Result<i32, KernelTrap> {
    assert_eq!(check_kernel(k), Ok(()), "well formed");
    let mut args = vec![buf(4), buf(4), buf(4)];
    // every kernel here finishes in a few hundred statements; a runaway spin stops at 20 000, not ten million
    run_launch_with_limit(k, &mut args, 20_000)?;
    let KernelArg::Buffer(out) = &args[2] else {
        unreachable!()
    };
    Ok(i32::from_le_bytes(out.bytes[..4].try_into().unwrap()))
}

/// Producer is thread 0, consumer thread 1 (one block of 2 threads).
fn two_threads(rel: (MemoryOrder, Scope), acq: (MemoryOrder, Scope)) -> KernelFn {
    let mut consume = vec![observe(0, acq)];
    consume.push(when(seen(1), copy_data()));
    kernel(
        2,
        1,
        vec![
            when(is(false, 0), vec![store(0, 0, 42), publish(0, 1, rel)]),
            when(is(false, 1), consume),
        ],
    )
}

#[test]
fn release_then_acquire_orders_the_data_and_relaxed_does_not() {
    for (rel, acq, race) in [
        (Release, Acquire, false),
        (AcqRel, AcqRel, false),
        (SeqCst, SeqCst, false),
        (Release, SeqCst, false),
        (Relaxed, Acquire, true),
        (Release, Relaxed, true),
        (Relaxed, Relaxed, true),
        (Acquire, Release, true),
    ] {
        let r = run(&two_threads((rel, DEV), (acq, DEV)));
        if race {
            assert_eq!(r, Err(KernelTrap::DataRace), "{rel:?} then {acq:?}");
        } else {
            assert_eq!(r, Ok(42), "{rel:?} then {acq:?}");
        }
    }
}

#[test]
fn a_compare_and_swap_acquires_too() {
    let cas = |order| Stmt::AtomicCas {
        place: at(1, 0),
        expected: c(1),
        desired: c(1),
        old: Some(0),
        order,
        scope: DEV,
    };
    for (order, race) in [(Acquire, false), (Relaxed, true)] {
        let k = kernel(
            2,
            1,
            vec![
                when(
                    is(false, 0),
                    vec![store(0, 0, 42), publish(0, 1, (Release, DEV))],
                ),
                when(is(false, 1), vec![cas(order), when(seen(1), copy_data())]),
            ],
        );
        let want = if race {
            Err(KernelTrap::DataRace)
        } else {
            Ok(42)
        };
        assert_eq!(run(&k), want, "{order:?}");
    }
}

#[test]
fn the_edge_is_transitive_through_a_third_thread() {
    // t0 publishes flag0; t1 acquires it and publishes flag1; t2 acquires flag1 and reads the data
    let chain = |middle: MemoryOrder| {
        kernel(
            3,
            1,
            vec![
                when(
                    is(false, 0),
                    vec![store(0, 0, 42), publish(0, 1, (Release, DEV))],
                ),
                when(
                    is(false, 1),
                    vec![
                        observe(0, (Acquire, DEV)),
                        when(seen(1), vec![publish(1, 1, (middle, DEV))]),
                    ],
                ),
                when(
                    is(false, 2),
                    vec![observe(1, (Acquire, DEV)), when(seen(1), copy_data())],
                ),
            ],
        )
    };
    assert_eq!(run(&chain(Release)), Ok(42));
    assert_eq!(run(&chain(Relaxed)), Err(KernelTrap::DataRace));
}

#[test]
fn a_relaxed_read_modify_write_continues_the_release_sequence() {
    // t0: data = 42; flag += 1 (Release). t1: flag += 1 (Relaxed). t2: acquires the flag, sees 2, reads the data.
    let add = |order| Stmt::AtomicRmw {
        op: AtomicOp::Add,
        place: at(1, 0),
        value: c(1),
        old: None,
        order,
        scope: DEV,
    };
    let k = kernel(
        3,
        1,
        vec![
            when(is(false, 0), vec![store(0, 0, 42), add(Release)]),
            when(is(false, 1), vec![add(Relaxed)]),
            when(
                is(false, 2),
                vec![observe(0, (Acquire, DEV)), when(seen(2), copy_data())],
            ),
        ],
    );
    assert_eq!(run(&k), Ok(42));
}

#[test]
fn it_orders_blocks_of_a_grid_at_device_scope() {
    let two_blocks = |scope| {
        kernel(
            1,
            2,
            vec![
                when(
                    is(true, 0),
                    vec![store(0, 0, 42), publish(0, 1, (Release, scope))],
                ),
                when(
                    is(true, 1),
                    vec![observe(0, (Acquire, scope)), when(seen(1), copy_data())],
                ),
            ],
        )
    };
    assert_eq!(run(&two_blocks(DEV)), Ok(42));
    // Block-scope atomics do not reach across blocks: the atomicity itself does not hold
    assert_eq!(run(&two_blocks(Scope::Block)), Err(KernelTrap::DataRace));
}

#[test]
fn a_subgroup_scope_atomic_does_not_reach_another_subgroup() {
    // every thread bumps one counter; 8 threads are two subgroups of 4
    let bump = |scope, block| {
        kernel(
            block,
            1,
            vec![Stmt::AtomicRmw {
                op: AtomicOp::Add,
                place: at(1, 0),
                value: c(1),
                old: None,
                order: Relaxed,
                scope,
            }],
        )
    };
    assert!(run(&bump(Scope::Subgroup, 4)).is_ok());
    assert_eq!(
        run(&bump(Scope::Subgroup, 8)),
        Err(KernelTrap::DataRace),
        "two subgroups"
    );
    assert!(run(&bump(Scope::Block, 8)).is_ok());
}

#[test]
fn a_release_at_a_narrow_scope_gives_an_edge_only_inside_that_scope() {
    // block 0 releases at Block scope, then touches the flag again at Device scope (so the flag's atomic log no
    // longer shows the narrow one); block 1 acquires at Device scope. The Block-scope release does not reach block 1.
    let k = |first: Scope| {
        kernel(
            1,
            2,
            vec![
                when(
                    is(true, 0),
                    vec![
                        store(0, 0, 42),
                        publish(0, 1, (Release, first)),
                        publish(0, 1, (Relaxed, DEV)),
                    ],
                ),
                when(
                    is(true, 1),
                    vec![observe(0, (Acquire, DEV)), when(seen(1), copy_data())],
                ),
            ],
        )
    };
    assert_eq!(run(&k(DEV)), Ok(42), "control: the release at Device scope");
    assert_eq!(run(&k(Scope::Block)), Err(KernelTrap::DataRace));
}

/// `loop { old = flag[0]; if old == 1 { break } }`, then `out[0] = data[0]`.
fn spin_and_read(order: MemoryOrder) -> Vec<Stmt> {
    let mut v = vec![Stmt::While {
        cond: c(1),
        body: Block(vec![
            observe(0, (order, DEV)),
            when(seen(1), vec![Stmt::Break]),
        ]),
    }];
    v.extend(copy_data());
    v
}

/// Block 0 spins on the flag, block 1 sets it (the consumer is the earlier block).
fn spin_then_publish(order: MemoryOrder) -> KernelFn {
    kernel(
        1,
        2,
        vec![
            when(is(true, 0), spin_and_read(order)),
            when(
                is(true, 1),
                vec![store(0, 0, 42), publish(0, 1, (Release, DEV))],
            ),
        ],
    )
}

#[test]
fn waiting_on_a_later_block_is_named() {
    assert_eq!(
        run(&spin_then_publish(Acquire)),
        Err(KernelTrap::CrossBlockWait)
    );
    // without an acquire there is nothing to call a wait: it is the plain runaway
    assert_eq!(run(&spin_then_publish(Relaxed)), Err(KernelTrap::StepLimit));
}

#[test]
fn the_producer_in_the_earlier_block_runs_to_completion() {
    let k = kernel(
        1,
        2,
        vec![
            when(
                is(true, 0),
                vec![store(0, 0, 42), publish(0, 1, (Release, DEV))],
            ),
            when(is(true, 1), spin_and_read(Acquire)),
        ],
    );
    assert_eq!(run(&k), Ok(42));
}

#[test]
fn thread_and_cluster_scopes_are_still_refused() {
    for scope in [Scope::Thread, Scope::Cluster] {
        let k = kernel(1, 1, vec![publish(0, 1, (Relaxed, scope))]);
        assert!(
            matches!(check_kernel(&k), Err(KernelError::AtomicUnsupported { .. })),
            "{scope:?}"
        );
    }
}

#[test]
fn the_release_itself_is_ordered_for_the_acquirer() {
    // after acquiring, the consumer plain-reads the flag the producer's atomic wrote
    let k = |acq| {
        kernel(
            2,
            1,
            vec![
                when(is(false, 0), vec![publish(0, 1, (Release, DEV))]),
                when(
                    is(false, 1),
                    vec![
                        observe(0, (acq, DEV)),
                        Stmt::Load {
                            local: 1,
                            place: at(1, 0),
                        },
                        Stmt::Store {
                            place: at(2, 0),
                            value: local(1),
                        },
                    ],
                ),
            ],
        )
    };
    assert_eq!(run(&k(Acquire)), Ok(1));
    assert_eq!(run(&k(Relaxed)), Err(KernelTrap::DataRace));
}

#[test]
fn two_releasers_on_one_flag_are_both_acquired() {
    // t0 and t1 each publish their own data with a release add; t2 acquires the flag after both
    let add = Stmt::AtomicRmw {
        op: AtomicOp::Add,
        place: at(1, 0),
        value: c(1),
        old: None,
        order: Release,
        scope: DEV,
    };
    let k = kernel(
        3,
        1,
        vec![
            when(is(false, 0), vec![store(0, 0, 10), add.clone()]),
            when(is(false, 1), vec![store(0, 1, 20), add]),
            when(
                is(false, 2),
                vec![observe(0, (Acquire, DEV)), when(seen(2), copy_data())],
            ),
        ],
    );
    assert_eq!(run(&k), Ok(10));
}

#[test]
fn a_runaway_in_the_last_block_is_not_a_cross_block_wait() {
    // nothing is left to run that could satisfy it
    let k = kernel(1, 1, spin_and_read(Acquire));
    assert_eq!(run(&k), Err(KernelTrap::StepLimit));
}
