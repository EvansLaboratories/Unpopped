// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Named barriers in the launch interpreter** (ROADMAP U4 S6b-2, `docs/options-u4-sync-and-atomics.md`):
//! `BarrierKind::Named { id, count }` (`bar.sync id, count`: arrive and wait) and `BarrierKind::NamedArrive { id, count }`
//! (`bar.arrive`: arrive, do not wait). A named barrier is a rendezvous of `count` *threads* of a block, any subset, not
//! the whole scope: it opens when `count` arrivals have been made, and every arrival is release-ordered before every
//! later `Named` wait. Each behaviour has a program that shows it and a control that does not: a producer warp group and a
//! consumer warp group hand a tile over twice (the second write waits for the consumers' release).

use unpopped::exec::{AllocDecl, ExecId, Extent, LaunchShape, MemorySpace, Scope};
use unpopped::ir::{BinaryOp, ScalarExpr};
use unpopped::kernel::{
    BarrierKind, Block, BufferRole, KernelError, KernelFn, LocalDecl, Param, Place, Stmt,
    check_kernel,
};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::{KernelArg, KernelTrap};
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_vocab::ElementKind;

const U32: ElementKind = ElementKind::U32;

fn u(v: i64) -> ScalarExpr {
    ScalarExpr::ConstInt(v, U32)
}
fn tid() -> ScalarExpr {
    ScalarExpr::Exec(ExecId::ThreadIdx(0))
}
fn sub(a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Sub(Box::new(a), Box::new(b))
}
fn add(a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Add(Box::new(a), Box::new(b))
}
fn data(i: ScalarExpr) -> Place {
    Place::Alloc { alloc: 0, index: i }
}
fn out(i: ScalarExpr) -> Place {
    Place::Logical {
        buffer: 0,
        coords: vec![i],
    }
}
fn named(id: u32, count: u32) -> Stmt {
    Stmt::Barrier {
        scope: Scope::Block,
        kind: BarrierKind::Named { id, count },
    }
}
fn arrive(id: u32, count: u32) -> Stmt {
    Stmt::Barrier {
        scope: Scope::Block,
        kind: BarrierKind::NamedArrive { id, count },
    }
}
fn store(place: Place, value: ScalarExpr) -> Stmt {
    Stmt::Store { place, value }
}
fn cond_lt(n: i64) -> ScalarExpr {
    ScalarExpr::Binary(BinaryOp::CmpLt, Box::new(tid()), Box::new(u(n)))
}

/// How a scenario differs from the plain one.
#[derive(Clone, Copy)]
struct Opt {
    /// The producers skip their first arrival.
    no_first_arrive: bool,
    /// The producers do not wait for the consumers' release before the second write.
    no_release_wait: bool,
    /// The id the producers arrive on first (the consumers wait on 1).
    first_id: u32,
    /// The thread count the consumers give the first barrier (the producers give 128).
    consumer_count: u32,
}
impl Opt {
    fn plain() -> Self {
        Opt {
            no_first_arrive: false,
            no_release_wait: false,
            first_id: 1,
            consumer_count: 128,
        }
    }
}

/// 128 threads: 0..64 produce into `data`, 64..128 consume. Round 1: produce `tid + 100`, arrive on 1; consumers wait on
/// 1, read into `out[tid]`, arrive on 2 (they are done with the tile); producers wait on 2, write `tid + 200`, arrive on 1;
/// consumers wait on 1 and read into `out[tid + 64]`.
fn kernel(o: Opt) -> KernelFn {
    let mut producer = vec![store(data(tid()), add(tid(), u(100)))];
    if !o.no_first_arrive {
        producer.push(arrive(o.first_id, 128));
    }
    if !o.no_release_wait {
        producer.push(named(2, 128));
    }
    producer.push(store(data(tid()), add(tid(), u(200))));
    producer.push(arrive(3, 128));
    let at = || data(sub(tid(), u(64)));
    let consumer = vec![
        named(1, o.consumer_count),
        Stmt::Load {
            local: 0,
            place: at(),
        },
        store(out(tid()), ScalarExpr::Local(0)),
        arrive(2, 128),
        named(3, 128),
        Stmt::Load {
            local: 1,
            place: at(),
        },
        store(out(add(tid(), u(64))), ScalarExpr::Local(1)),
    ];
    let body = vec![Stmt::If {
        cond: cond_lt(64),
        then: Block(producer),
        els: Block(consumer),
    }];
    let mut k = KernelFn::new(
        "named",
        U32,
        vec![Param::buffer("out", U32, BufferRole::ReadWrite, Some(1))],
        vec![LocalDecl::scalar(U32), LocalDecl::scalar(U32)],
        Block(body),
    )
    .with_launch(LaunchShape::new([128, 1, 1], [1, 1, 1]).with_subgroup(32, 32));
    k.allocs = vec![AllocDecl::new(
        "data",
        MemorySpace::Shared,
        U32,
        Extent::Static(64),
    )];
    k
}

fn run(k: &KernelFn) -> Result<Vec<u32>, KernelTrap> {
    assert_eq!(check_kernel(k), Ok(()));
    let mut args = [KernelArg::Buffer(TypedBuffer::new(
        U32,
        vec![192],
        vec![1],
        vec![0; 192 * 4],
    ))];
    run_launch(k, &mut args)?;
    let KernelArg::Buffer(b) = &args[0] else {
        unreachable!("a buffer")
    };
    Ok(b.bytes
        .chunks(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect())
}

#[test]
fn a_producer_group_hands_a_tile_to_a_consumer_group_twice() {
    let got = run(&kernel(Opt::plain())).expect("the oracle runs it");
    let mut want = vec![0u32; 192];
    for t in 64..128u32 {
        want[t as usize] = t - 64 + 100;
        want[t as usize + 64] = t - 64 + 200;
    }
    assert_eq!(got, want);
}

#[test]
fn a_barrier_nobody_completes_is_a_deadlock() {
    let mut o = Opt::plain();
    o.no_first_arrive = true; // the consumers wait for arrivals that never come
    assert_eq!(run(&kernel(o)), Err(KernelTrap::MbarrierDeadlock));
    let mut o = Opt::plain();
    o.first_id = 5; // another barrier: id 1 never opens
    assert_eq!(run(&kernel(o)), Err(KernelTrap::MbarrierDeadlock));
}

#[test]
fn the_participants_must_agree_on_the_count() {
    let mut o = Opt::plain();
    o.consumer_count = 64;
    assert_eq!(run(&kernel(o)), Err(KernelTrap::BarrierMismatch));
}

#[test]
fn a_write_not_ordered_after_the_readers_release_is_a_race() {
    let mut o = Opt::plain();
    o.no_release_wait = true; // the producers overwrite the tile the consumers may still be reading
    assert_eq!(run(&kernel(o)), Err(KernelTrap::DataRace));
}

#[test]
fn a_named_barrier_may_sit_under_thread_dependent_control_but_a_full_one_may_not() {
    // the producer/consumer kernel above has named barriers inside an `If` on the thread index and checks clean
    assert_eq!(check_kernel(&kernel(Opt::plain())), Ok(()));
    let mut k = kernel(Opt::plain());
    k.body = Block(vec![Stmt::If {
        cond: cond_lt(64),
        then: Block(vec![Stmt::barrier(Scope::Block)]),
        els: Block(vec![]),
    }]);
    assert_eq!(check_kernel(&k), Err(KernelError::BarrierDivergent));
}

#[test]
fn the_checker_refuses_what_a_named_barrier_cannot_be() {
    let with = |s: Stmt| {
        let mut k = kernel(Opt::plain());
        k.body = Block(vec![s]);
        check_kernel(&k)
    };
    let bad =
        |r: Result<(), KernelError>| matches!(r, Err(KernelError::NamedBarrierUnsupported { .. }));
    assert!(bad(with(named(16, 128))), "id 16");
    assert!(bad(with(named(1, 48))), "48 threads is not whole warps");
    assert!(bad(with(named(1, 0))), "no threads");
    assert!(bad(with(named(1, 256))), "more threads than the block");
    assert!(
        bad(with(Stmt::Barrier {
            scope: Scope::Subgroup,
            kind: BarrierKind::Named { id: 1, count: 32 }
        })),
        "a subgroup-scope named barrier"
    );
    assert_eq!(with(named(15, 128)), Ok(()));
    assert_eq!(with(arrive(0, 32)), Ok(()));
}
