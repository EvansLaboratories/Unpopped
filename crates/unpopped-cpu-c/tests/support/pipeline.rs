// COPY of `crates/unpopped/tests/support/pipeline.rs`: a test may not `#[path]` into another crate's directory (sweep
// SUP-004). Change both files together.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! The double-buffered copy-and-sum kernel of ROADMAP U8 S4 (`docs/options-u8-copy-atoms.md`): four threads sum a
//! 4-tile column pipeline through a two-stage `Shared` ring filled by `cp.async`. Each thread prefetches tile `k + 1`
//! while it consumes tile `k` (the element of its **mirror** thread, so that a barrier is needed), commits a group
//! every iteration (an empty one on the last, so `AsyncWait { pending: 1 }` always means "tile `k` has landed"), and
//! has two barriers per iteration: one so the landed tile is visible, one so no thread issues the copy that refills a
//! stage another thread is still reading. Shared by the oracle test (`kernel_pipeline_double_buffer.rs`) and the C test
//! (`unpopped-cpu-c/tests/kernel_pipeline_emit_agrees.rs`, which keeps a copy: change both).

use unpopped::exec::{AllocDecl, ExecId, Extent, LaunchShape, MemorySpace, Scope};
use unpopped::ir::ScalarExpr;
use unpopped::kernel::{BarrierKind, Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt};
use unpopped_vocab::ElementKind;

pub(crate) const I32: ElementKind = ElementKind::I32;
pub(crate) const U32: ElementKind = ElementKind::U32;
pub(crate) const THREADS: u32 = 4;
pub(crate) const TILES: i64 = 4;

/// What the pipeline gets wrong, if anything.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Fault {
    /// Nothing: the correct pipeline.
    None,
    /// `AsyncWait { pending: 2 }`: the tile being consumed has not landed.
    WaitsTooLittle,
    /// No barrier between the landing wait and the read of another thread's element.
    NoReadBarrier,
    /// No barrier after the read: a thread refills a stage that another may still be reading.
    NoReuseBarrier,
}

fn cu(v: i64) -> ScalarExpr {
    ScalarExpr::ConstInt(v, U32)
}

fn tid() -> ScalarExpr {
    ScalarExpr::Exec(ExecId::ThreadIdx(0))
}

fn add(a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Add(Box::new(a), Box::new(b))
}

fn shared(i: ScalarExpr) -> Place {
    Place::Alloc { alloc: 0, index: i }
}

fn global(buffer: u32, i: ScalarExpr) -> Place {
    Place::Logical {
        buffer,
        coords: vec![i],
    }
}

fn barrier() -> Stmt {
    Stmt::Barrier {
        scope: Scope::Block,
        kind: BarrierKind::Full,
    }
}

fn stage(tile: i64) -> i64 {
    (tile % 2) * i64::from(THREADS)
}

/// The prologue: `acc = 0`, and tile 0 copied into stage 0 as the first group.
fn prologue() -> Vec<Stmt> {
    vec![
        Stmt::Let {
            local: 0,
            init: ScalarExpr::ConstInt(0, I32),
        },
        Stmt::CopyAsync {
            dst: shared(add(cu(stage(0)), tid())),
            src: global(0, tid()),
            elems: 1,
        },
        Stmt::AsyncCommit,
    ]
}

/// Iteration `k`: prefetch tile `k + 1` (if any), commit, wait, barrier, consume the mirror thread's element of tile
/// `k`, barrier.
fn iteration(k: i64, fault: Fault) -> Vec<Stmt> {
    let mut body = Vec::new();
    if k + 1 < TILES {
        body.push(Stmt::CopyAsync {
            dst: shared(add(cu(stage(k + 1)), tid())),
            src: global(0, add(cu((k + 1) * i64::from(THREADS)), tid())),
            elems: 1,
        });
    }
    body.push(Stmt::AsyncCommit);
    body.push(Stmt::AsyncWait {
        pending: if fault == Fault::WaitsTooLittle { 2 } else { 1 },
    });
    if fault != Fault::NoReadBarrier {
        body.push(barrier());
    }
    // the mirror thread's element of tile k: stage + (3 - tid)
    body.push(Stmt::Load {
        local: 1 + k as u32,
        place: shared(add(
            cu(stage(k)),
            ScalarExpr::Sub(Box::new(cu(i64::from(THREADS) - 1)), Box::new(tid())),
        )),
    });
    body.push(Stmt::Assign {
        local: 0,
        value: add(ScalarExpr::Local(0), ScalarExpr::Local(1 + k as u32)),
    });
    if fault != Fault::NoReuseBarrier {
        body.push(barrier());
    }
    body
}

/// `src` is 16 `I32` (four tiles of four), `out` four `I32`.
pub(crate) fn double_buffered(fault: Fault) -> KernelFn {
    let mut body = prologue();
    for k in 0..TILES {
        body.extend(iteration(k, fault));
    }
    body.push(Stmt::AsyncWait { pending: 0 });
    body.push(Stmt::Store {
        place: global(1, tid()),
        value: ScalarExpr::Local(0),
    });
    let locals = (0..=TILES).map(|_| LocalDecl::scalar(I32)).collect();
    KernelFn::new(
        "pipeline",
        I32,
        vec![
            Param::buffer("src", I32, BufferRole::Read, Some(1)),
            Param::buffer("out", I32, BufferRole::ReadWrite, Some(1)),
        ],
        locals,
        Block(body),
    )
    .with_allocs(vec![AllocDecl::new(
        "ring",
        MemorySpace::Shared,
        I32,
        Extent::Static(2 * THREADS),
    )])
    .with_launch(LaunchShape::new([THREADS, 1, 1], [1, 1, 1]))
}

/// `out[t] = sum over tiles k of src[k * 4 + (3 - t)]` for `src[i] = 100 + i`.
pub(crate) const EXPECTED: [i32; 4] = [436, 432, 428, 424];
