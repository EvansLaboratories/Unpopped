// SPDX-License-Identifier: MIT OR Apache-2.0
//! Idiom-gap row 9: a device-scope semaphore. A producer block releases a flag, a consumer block spins on
//! `AtomicCas(Acquire, Device)` until it sees it, then reads the producer's plain data. This probes what the oracle
//! does with the three orderings of the blocks: producer first (completes), consumer first (the consumer waits on a
//! LATER block: `CrossBlockWait`), and a spin nobody releases (`CrossBlockWait` while a block is left to run,
//! `StepLimit` when the spinner is the last block). `Acquire` weakened to `Relaxed` is a `DataRace` (producer first)
//! or a plain `StepLimit` (consumer first, where nothing was acquired to call a wait). The oracle runs blocks one after
//! another in index order, so the first case is the only one that can complete.

use unpopped::exec::{ExecId, LaunchShape, Scope};
use unpopped::ir::{BinaryOp, ScalarExpr};
use unpopped::kernel::{
    AtomicOp, Block, BufferRole, KernelFn, LocalDecl, MemoryOrder, Param, Place, Stmt, check_kernel,
};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::{KernelArg, KernelTrap};
use unpopped::oracle::kernel_launch::run_launch_with_limit;
use unpopped_vocab::ElementKind;

use MemoryOrder::{Acquire, Relaxed, Release};

const I32: ElementKind = ElementKind::I32;
const U32: ElementKind = ElementKind::U32;
const DEV: Scope = Scope::Device;

fn c(v: i64) -> ScalarExpr {
    ScalarExpr::ConstInt(v, I32)
}

fn at(buffer: u32) -> Place {
    Place::Logical {
        buffer,
        coords: vec![ScalarExpr::ConstInt(0, U32)],
    }
}

fn block_is(t: i64) -> ScalarExpr {
    ScalarExpr::Binary(
        BinaryOp::CmpEq,
        Box::new(ScalarExpr::Exec(ExecId::BlockIdx(0))),
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

/// Buffers: 0 `data`, 1 `flag`, 2 `out`.
fn produce(order: MemoryOrder) -> Vec<Stmt> {
    vec![
        Stmt::Store {
            place: at(0),
            value: c(42),
        },
        Stmt::AtomicRmw {
            op: AtomicOp::Exchange,
            place: at(1),
            value: c(1),
            old: None,
            order,
            scope: DEV,
        },
    ]
}

/// `loop { old = cas(flag[0], 1 -> 1); if old == 1 { break } }`, then `out[0] = data[0]`.
fn consume(order: MemoryOrder) -> Vec<Stmt> {
    let seen = ScalarExpr::Binary(
        BinaryOp::CmpEq,
        Box::new(ScalarExpr::Local(0)),
        Box::new(c(1)),
    );
    vec![
        Stmt::While {
            cond: c(1),
            body: Block(vec![
                Stmt::AtomicCas {
                    place: at(1),
                    expected: c(1),
                    desired: c(1),
                    old: Some(0),
                    order,
                    scope: DEV,
                },
                when(seen, vec![Stmt::Break]),
            ]),
        },
        Stmt::Load {
            local: 1,
            place: at(0),
        },
        Stmt::Store {
            place: at(2),
            value: ScalarExpr::Local(1),
        },
    ]
}

fn kernel(grid: u32, body: Vec<Stmt>) -> KernelFn {
    let params = vec![
        Param::buffer("data", I32, BufferRole::ReadWrite, Some(1)),
        Param::buffer("flag", I32, BufferRole::ReadWrite, Some(1)),
        Param::buffer("out", I32, BufferRole::ReadWrite, Some(1)),
    ];
    let locals = vec![LocalDecl::scalar(I32), LocalDecl::scalar(I32)];
    let launch = LaunchShape::new([1, 1, 1], [grid, 1, 1]).with_subgroup(4, 4);
    KernelFn::new("semaphore", I32, params, locals, Block(body)).with_launch(launch)
}

fn buf() -> KernelArg {
    KernelArg::Buffer(TypedBuffer::new(I32, vec![4], vec![1], vec![0; 16]))
}

fn run(k: &KernelFn) -> Result<i32, KernelTrap> {
    assert_eq!(check_kernel(k), Ok(()), "well formed");
    let mut args = vec![buf(), buf(), buf()];
    run_launch_with_limit(k, &mut args, 20_000)?;
    let KernelArg::Buffer(out) = &args[2] else {
        unreachable!()
    };
    Ok(i32::from_le_bytes(out.bytes[..4].try_into().unwrap()))
}

fn producer_first(rel: MemoryOrder, acq: MemoryOrder) -> KernelFn {
    kernel(
        2,
        vec![
            when(block_is(0), produce(rel)),
            when(block_is(1), consume(acq)),
        ],
    )
}

fn consumer_first(rel: MemoryOrder, acq: MemoryOrder) -> KernelFn {
    kernel(
        2,
        vec![
            when(block_is(0), consume(acq)),
            when(block_is(1), produce(rel)),
        ],
    )
}

#[test]
fn a_producer_in_the_earlier_block_completes() {
    assert_eq!(run(&producer_first(Release, Acquire)), Ok(42));
}

#[test]
fn a_consumer_in_the_earlier_block_is_a_cross_block_wait() {
    assert_eq!(
        run(&consumer_first(Release, Acquire)),
        Err(KernelTrap::CrossBlockWait)
    );
}

#[test]
fn a_spin_nobody_releases_is_named_while_a_block_is_left_and_a_runaway_when_none_is() {
    // block 1 never releases, and block 0 spins with block 1 still to run
    let unreleased = kernel(2, vec![when(block_is(0), consume(Acquire))]);
    assert_eq!(run(&unreleased), Err(KernelTrap::CrossBlockWait));
    // the spinner is the last block: nothing is left to wait on, so it is the plain runaway
    let last = kernel(2, vec![when(block_is(1), consume(Acquire))]);
    assert_eq!(run(&last), Err(KernelTrap::StepLimit));
}

#[test]
fn weakening_acquire_to_relaxed_changes_both_orders() {
    // producer first: the spin ends but nothing orders the producer's plain store before the consumer's plain load
    assert_eq!(
        run(&producer_first(Release, Relaxed)),
        Err(KernelTrap::DataRace)
    );
    // consumer first: with nothing acquired there is nothing to call a wait
    assert_eq!(
        run(&consumer_first(Release, Relaxed)),
        Err(KernelTrap::StepLimit)
    );
}
