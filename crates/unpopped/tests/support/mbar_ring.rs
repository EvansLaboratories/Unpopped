// SPDX-License-Identifier: MIT OR Apache-2.0
//! A two-stage full/empty `mbarrier` ring inside one block (CUTLASS survey row 88: pipelines, producer/consumer). Thread 1 is the
//! producer and thread 0 the consumer (thread 0 runs first in the oracle's schedule, so every wait really blocks). `ITEMS` items
//! pass through `STAGES` slots of a shared ring. Per slot there is a `full` barrier and an `empty` barrier, each arrival count 1.
//! The producer waits the slot's `empty` barrier (but for the first fill), writes the slot, and arrives at `full`; the consumer
//! waits `full`, reads the slot, arrives at `empty`, and stores `out[i] = 2 * slot + 1`. Waits use the phase parity of the round.
//! The barriers are initialised, arrived at and waited on as separate statements; no arrival carries a transaction count, and
//! the scope is one block (`docs/PATENT-SWEEP.md`, barrier chart #442: the re-read triggers for an arrival fused with a
//! transaction count, a barrier with a reduction or scan, and divergence compaction are not exercised here).

use unpopped::exec::{AllocDecl, ExecId, Extent, LaunchShape, MemorySpace, Scope};
use unpopped::ir::{BinaryOp, ScalarExpr};
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt};
use unpopped_vocab::ElementKind;

pub(crate) const F32: ElementKind = ElementKind::F32;
pub(crate) const U32: ElementKind = ElementKind::U32;
pub(crate) const STAGES: usize = 2;
pub(crate) const ITEMS: usize = 6;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Mutant {
    None,
    /// The consumer reads the other slot than the one it waited on.
    ReadOtherSlot,
    /// The consumer's item counter is off by one: item `i` is stored at `out[(i + 1) % ITEMS]`.
    ShiftedItem,
    /// The consumer never arrives at `empty` (the producer's refill wait can never complete).
    NoEmptyArrive,
    /// The producer refills without waiting for `empty`.
    NoEmptyWait,
}

fn u(v: usize) -> ScalarExpr {
    ScalarExpr::ConstInt(v as i64, U32)
}
fn f(v: f32) -> ScalarExpr {
    ScalarExpr::Const(f64::from(v))
}
/// The barrier `full[s]` (allocation 0, element `s`) or `empty[s]` (element `STAGES + s`).
fn full(s: usize) -> Place {
    Place::Alloc {
        alloc: 0,
        index: u(s),
    }
}
fn empty(s: usize) -> Place {
    Place::Alloc {
        alloc: 0,
        index: u(STAGES + s),
    }
}
fn slot(s: usize) -> Place {
    Place::Alloc {
        alloc: 1,
        index: u(s),
    }
}
fn buffer(b: u32, i: usize) -> Place {
    Place::Logical {
        buffer: b,
        coords: vec![u(i)],
    }
}
/// Run `body` in thread `t` only.
fn on(t: usize, body: Vec<Stmt>) -> Stmt {
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

/// The producer's statements for item `i`: wait the slot's `empty` barrier for its previous round, fill, arrive at `full`.
/// The wait of round 0 uses parity 1, which passes at once on a fresh barrier.
fn produce(i: usize, mutant: Mutant) -> Vec<Stmt> {
    let (s, round) = (i % STAGES, i / STAGES);
    let mut out = Vec::new();
    if mutant != Mutant::NoEmptyWait {
        out.push(Stmt::MbarWait {
            bar: empty(s),
            parity: u((round + 1) % 2),
        });
    }
    out.push(Stmt::Load {
        local: i as u32,
        place: buffer(0, i),
    });
    out.push(Stmt::Store {
        place: slot(s),
        value: ScalarExpr::Local(i as u32),
    });
    out.push(Stmt::MbarArrive {
        bar: full(s),
        tx: None,
    });
    out
}

/// The consumer's statements for item `i`: wait `full` for the round, read the slot, release it, store the result.
fn consume(i: usize, mutant: Mutant) -> Vec<Stmt> {
    let (s, round) = (i % STAGES, i / STAGES);
    let read = if mutant == Mutant::ReadOtherSlot {
        (s + 1) % STAGES
    } else {
        s
    };
    let local = (ITEMS + i) as u32;
    let mut out = vec![
        Stmt::MbarWait {
            bar: full(s),
            parity: u(round % 2),
        },
        Stmt::Load {
            local,
            place: slot(read),
        },
    ];
    if mutant != Mutant::NoEmptyArrive {
        out.push(Stmt::MbarArrive {
            bar: empty(s),
            tx: None,
        });
    }
    let dest = if mutant == Mutant::ShiftedItem {
        (i + 1) % ITEMS
    } else {
        i
    };
    out.push(Stmt::Store {
        place: buffer(1, dest),
        value: ScalarExpr::Add(
            Box::new(ScalarExpr::Mul(
                Box::new(ScalarExpr::Local(local)),
                Box::new(f(2.0)),
            )),
            Box::new(f(1.0)),
        ),
    });
    out
}

/// Buffers: 0 `input` `[ITEMS]` (read), 1 `out` `[ITEMS]` (write). Allocations: 0 the `2 * STAGES` barriers (`u64`, shared),
/// 1 the `STAGES` ring slots (`f32`, shared). Two threads.
pub(crate) fn ring_kernel(mutant: Mutant) -> KernelFn {
    let inits = (0..STAGES)
        .flat_map(|s| [full(s), empty(s)])
        .map(|bar| Stmt::MbarInit { bar, count: u(1) });
    let mut body = vec![on(0, inits.collect()), Stmt::barrier(Scope::Block)];
    body.push(on(0, (0..ITEMS).flat_map(|i| consume(i, mutant)).collect()));
    body.push(on(1, (0..ITEMS).flat_map(|i| produce(i, mutant)).collect()));
    let params = vec![
        Param::buffer("input", F32, BufferRole::Read, Some(1)),
        Param::buffer("out", F32, BufferRole::Write, Some(1)),
    ];
    let locals = vec![LocalDecl::scalar(F32); 2 * ITEMS];
    KernelFn::new("mbar_ring", F32, params, locals, Block(body))
        .with_launch(LaunchShape::new([2, 1, 1], [1, 1, 1]))
        .with_allocs(vec![
            AllocDecl::new(
                "bars",
                MemorySpace::Shared,
                ElementKind::U64,
                Extent::Static((2 * STAGES) as u32),
            ),
            AllocDecl::new(
                "ring",
                MemorySpace::Shared,
                F32,
                Extent::Static(STAGES as u32),
            ),
        ])
}
