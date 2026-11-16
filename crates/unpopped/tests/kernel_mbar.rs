// SPDX-License-Identifier: MIT OR Apache-2.0
//! **`mbarrier` in the launch interpreter** (ROADMAP U8c, `docs/options-u8-copy-atoms.md` S3): `MbarInit`, `MbarArrive`
//! (with `expect_tx`), `MbarCompleteTx` and `MbarWait`. Each behaviour has a program that shows it and a control that
//! does not: a consumer thread blocks until the producers' arrivals complete the phase and then sees their data without a
//! `DataRace`; a store made after the arrival is a `DataRace`; a phase that nobody completes is `MbarrierDeadlock`; the
//! parity rule (`parity` 1 passes on a fresh barrier, `parity` 0 does not) and the transaction count are each pinned.
//! The consumer is thread 0 and runs first, so every successful wait here really blocked and was woken.

use unpopped::exec::{AllocDecl, ExecId, Extent, LaunchShape, MemorySpace, Scope};
use unpopped::ir::{BinaryOp, ScalarExpr};
use unpopped::kernel::{
    Block, BufferRole, KernelError, KernelFn, LocalDecl, Param, Place, Stmt, check_kernel,
};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::{KernelArg, KernelTrap};
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_vocab::ElementKind;

const U32: ElementKind = ElementKind::U32;

fn u(v: i64) -> ScalarExpr {
    ScalarExpr::ConstInt(v, U32)
}
fn bar() -> Place {
    Place::Alloc {
        alloc: 0,
        index: u(0),
    }
}
fn data(i: i64) -> Place {
    Place::Alloc {
        alloc: 1,
        index: u(i),
    }
}
/// Run `body` in thread `t` only.
fn on(t: i64, body: Vec<Stmt>) -> Stmt {
    Stmt::If {
        cond: ScalarExpr::Binary(
            BinaryOp::CmpEq,
            Box::new(ScalarExpr::Exec(ExecId::ThreadIdx(0))),
            Box::new(u(t)),
        ),
        then: Block(body),
        els: Block(vec![]),
    }
}
fn init(count: i64) -> Stmt {
    Stmt::MbarInit {
        bar: bar(),
        count: u(count),
    }
}
fn arrive(tx: Option<i64>) -> Stmt {
    Stmt::MbarArrive {
        bar: bar(),
        tx: tx.map(u),
    }
}
fn wait(parity: i64) -> Stmt {
    Stmt::MbarWait {
        bar: bar(),
        parity: u(parity),
    }
}
fn put(i: i64, v: i64) -> Stmt {
    Stmt::Store {
        place: data(i),
        value: u(v),
    }
}
/// `out[0] = data[0] + data[1]` (both elements are written before it by the scenarios that read two).
fn publish() -> Vec<Stmt> {
    vec![
        Stmt::Load {
            local: 0,
            place: data(0),
        },
        Stmt::Load {
            local: 1,
            place: data(1),
        },
        Stmt::Store {
            place: Place::Logical {
                buffer: 0,
                coords: vec![u(0)],
            },
            value: ScalarExpr::Add(
                Box::new(ScalarExpr::Local(0)),
                Box::new(ScalarExpr::Local(1)),
            ),
        },
    ]
}

/// A block of `threads` threads with the barrier (Shared `u64`) as allocation 0 and two `u32` cells as allocation 1.
fn kernel(threads: u32, body: Vec<Stmt>) -> KernelFn {
    let mut k = KernelFn::new(
        "mbar",
        U32,
        vec![Param::buffer("out", U32, BufferRole::ReadWrite, Some(1))],
        vec![LocalDecl::scalar(U32), LocalDecl::scalar(U32)],
        Block(body),
    )
    .with_launch(LaunchShape::new([threads, 1, 1], [1, 1, 1]));
    k.allocs = vec![
        AllocDecl::new(
            "bar",
            MemorySpace::Shared,
            ElementKind::U64,
            Extent::Static(1),
        ),
        AllocDecl::new("cells", MemorySpace::Shared, U32, Extent::Static(2)),
    ];
    k
}

fn run(k: &KernelFn) -> Result<u32, KernelTrap> {
    assert_eq!(check_kernel(k), Ok(()));
    let mut args = [KernelArg::Buffer(TypedBuffer::new(
        U32,
        vec![1],
        vec![1],
        vec![0; 4],
    ))];
    run_launch(k, &mut args)?;
    let KernelArg::Buffer(b) = &args[0] else {
        unreachable!("a buffer")
    };
    Ok(u32::from_le_bytes(b.bytes[..4].try_into().unwrap()))
}

/// Thread 0 initialises, everyone meets at a block barrier, then `rest` runs.
fn after_init(count: i64, mut rest: Vec<Stmt>) -> Vec<Stmt> {
    let mut body = vec![
        on(0, vec![init(count), put(0, 0), put(1, 0)]),
        Stmt::barrier(Scope::Block),
    ];
    body.append(&mut rest);
    body
}

/// Thread 0 (the consumer, first in the schedule) waits for `parity` then publishes the two cells.
fn consumer(parity: i64) -> Stmt {
    let mut then = vec![wait(parity)];
    then.extend(publish());
    on(0, then)
}

#[test]
fn a_consumer_blocks_until_the_producer_arrives_and_then_sees_its_data() {
    let k = kernel(
        2,
        after_init(
            1,
            vec![
                consumer(0),
                on(1, vec![put(0, 21), put(1, 21), arrive(None)]),
            ],
        ),
    );
    assert_eq!(run(&k), Ok(42));
}

#[test]
fn a_store_after_the_arrival_is_not_ordered_before_the_consumers_read() {
    let k = kernel(
        2,
        after_init(
            1,
            vec![
                consumer(0),
                on(1, vec![put(0, 21), arrive(None), put(1, 21)]),
            ],
        ),
    );
    assert_eq!(run(&k), Err(KernelTrap::DataRace));
}

#[test]
fn a_phase_nobody_completes_is_a_deadlock() {
    let k = kernel(
        2,
        after_init(1, vec![consumer(0), on(1, vec![put(0, 1), put(1, 1)])]),
    );
    assert_eq!(run(&k), Err(KernelTrap::MbarrierDeadlock));
}

#[test]
fn the_phase_completes_when_every_expected_arrival_has_come() {
    let two =
        |producers: Vec<Stmt>| kernel(3, after_init(2, [vec![consumer(0)], producers].concat()));
    let both = two(vec![
        on(1, vec![put(0, 10), arrive(None)]),
        on(2, vec![put(1, 20), arrive(None)]),
    ]);
    assert_eq!(run(&both), Ok(30));
    // the control: with one arrival missing the phase never completes
    let one = two(vec![
        on(1, vec![put(0, 10), arrive(None)]),
        on(2, vec![put(1, 20)]),
    ]);
    assert_eq!(run(&one), Err(KernelTrap::MbarrierDeadlock));
}

#[test]
fn the_phase_parity_alternates() {
    // count 1, so each arrival completes a phase; one thread, so a wait that cannot pass is a deadlock
    let after = |arrivals: usize, parity| {
        let mut body = vec![init(1)];
        body.extend((0..arrivals).map(|_| arrive(None)));
        body.push(wait(parity));
        kernel(1, body)
    };
    // phase 0 is complete (the current phase is 1): parity 0 passes, parity 1 blocks
    assert_eq!(run(&after(1, 0)), Ok(0));
    assert_eq!(run(&after(1, 1)), Err(KernelTrap::MbarrierDeadlock));
    // phases 0 and 1 are complete (the current phase is 2): parity 1 passes, parity 0 blocks
    assert_eq!(run(&after(2, 1)), Ok(0));
    assert_eq!(run(&after(2, 0)), Err(KernelTrap::MbarrierDeadlock));
}

#[test]
fn parity_one_passes_on_a_fresh_barrier_and_parity_zero_does_not() {
    let alone = |parity| {
        kernel(
            1,
            vec![
                init(1),
                put(0, 3),
                put(1, 4),
                wait(parity),
                publish().remove(0),
            ],
        )
    };
    let pass = alone(1);
    assert_eq!(run(&pass), Ok(0));
    assert_eq!(run(&alone(0)), Err(KernelTrap::MbarrierDeadlock));
}

#[test]
fn the_transaction_count_holds_the_phase_open_until_the_bytes_land() {
    let with = |landed: i64| {
        kernel(
            3,
            after_init(
                1,
                vec![
                    consumer(0),
                    on(1, vec![arrive(Some(4))]),
                    on(
                        2,
                        vec![
                            put(0, 5),
                            put(1, 6),
                            Stmt::MbarCompleteTx {
                                bar: bar(),
                                bytes: u(landed),
                            },
                        ],
                    ),
                ],
            ),
        )
    };
    assert_eq!(run(&with(4)), Ok(11));
    // the control: 3 of the 4 announced bytes leave the phase open
    assert_eq!(run(&with(3)), Err(KernelTrap::MbarrierDeadlock));
}

#[test]
fn using_a_barrier_outside_its_contract_is_a_misuse_trap() {
    let misuse = |body: Vec<Stmt>| {
        matches!(
            run(&kernel(1, body)),
            Err(KernelTrap::MbarrierMisuse { .. })
        )
    };
    assert!(misuse(vec![arrive(None)]), "arrive before init");
    assert!(misuse(vec![init(0)]), "arrival count 0");
    assert!(misuse(vec![init(1), wait(2)]), "parity 2");
}

#[test]
fn the_checker_refuses_what_an_mbarrier_cannot_be() {
    let refused = |k: KernelFn| {
        matches!(
            check_kernel(&k),
            Err(KernelError::MbarrierUnsupported { .. })
        )
    };
    let mut narrow = kernel(1, vec![init(1)]);
    narrow.allocs[0] = AllocDecl::new("bar", MemorySpace::Shared, U32, Extent::Static(1));
    let mut private = kernel(1, vec![init(1)]);
    private.allocs[0] = AllocDecl::new(
        "bar",
        MemorySpace::Private,
        ElementKind::U64,
        Extent::Static(1),
    );
    // a bare `Const` takes the compute type, so a float compute type makes the parity a float
    let mut float_parity = kernel(
        1,
        vec![Stmt::MbarWait {
            bar: bar(),
            parity: ScalarExpr::Const(1.0),
        }],
    );
    float_parity.compute = ElementKind::F32;
    for (what, k) in [
        ("32-bit", narrow),
        ("private", private),
        ("float parity", float_parity),
    ] {
        assert!(refused(k), "{what}");
    }
    assert_eq!(check_kernel(&kernel(1, vec![init(1)])), Ok(()));
}
