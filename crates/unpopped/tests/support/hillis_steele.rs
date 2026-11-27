// SPDX-License-Identifier: MIT OR Apache-2.0
//! A Hillis-Steele inclusive prefix sum in shared memory, one 32-lane group per named barrier (the ThunderKittens mamba2 gate
//! cumsum pattern: an in-place scan fenced per step by a subset barrier; survey row 34 of
//! `docs/ir-union-evidence/conformance-idiom-gaps-2026-11-27.md`). A block of 64 threads is two groups of 32; group 0 syncs on
//! named barrier 1 and group 1 on barrier 2, each with count 32, so neither group waits for the other. Each step `d` in 1, 2, 4,
//! 8, 16: read `s[i - d]` (zero for the first `d` lanes of the group), sync, add into `s[i]`, sync. Block-scope named barriers
//! only (`bar.sync`, arrive and wait together); no `bar.arrive`, no cluster scope, no fused transaction count. The block
//! scope and the barrier statements are `docs/PATENT-SWEEP.md` barrier chart (#442) territory: outside as read, not a
//! clearance; its re-read triggers (a barrier with a reduction or scan, divergence compaction) are not exercised, since the
//! scan here is ordinary shared-memory loads, adds and stores between barriers.

use unpopped::exec::{AllocDecl, ExecId, Extent, LaunchShape, MemorySpace, Scope};
use unpopped::ir::{BinaryOp, ScalarExpr};
use unpopped::kernel::{BarrierKind, Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt};
use unpopped_vocab::ElementKind;

pub(crate) const F32: ElementKind = ElementKind::F32;
pub(crate) const U32: ElementKind = ElementKind::U32;
pub(crate) const GROUP: usize = 32;
pub(crate) const THREADS: usize = 2 * GROUP;
const STEPS: [usize; 5] = [1, 2, 4, 8, 16];

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Mutant {
    None,
    /// The last step (`d = 16`) is skipped.
    DropLastStep,
    /// The guard is `lane > d`: the lane exactly `d` in reads zero instead of `s[i - d]`.
    OffByOneGuard,
    /// The sync between the read and the write of a step is missing.
    NoSecondSync,
}

fn u(v: usize) -> ScalarExpr {
    ScalarExpr::ConstInt(v as i64, U32)
}
fn l(i: u32) -> ScalarExpr {
    ScalarExpr::Local(i)
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
fn cmp(op: BinaryOp, a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Binary(op, Box::new(a), Box::new(b))
}
fn tile(i: ScalarExpr) -> Place {
    Place::Alloc { alloc: 0, index: i }
}
fn named(id: u32) -> Stmt {
    Stmt::Barrier {
        scope: Scope::Block,
        kind: BarrierKind::Named {
            id,
            count: GROUP as u32,
        },
    }
}

/// Step number `k` (distance `d`) for the group on barrier `id`: locals 1 `lane` (`U32`), 2 `v`, 3 the loaded neighbour, `6 + k`
/// the step's own `s[i]` (a local is loaded once per scope).
fn step(d: usize, k: u32, id: u32, mutant: Mutant) -> Vec<Stmt> {
    let guard = if mutant == Mutant::OffByOneGuard {
        cmp(BinaryOp::CmpGt, l(1), u(d))
    } else {
        cmp(BinaryOp::CmpGe, l(1), u(d))
    };
    let mut out = vec![
        Stmt::Assign {
            local: 2,
            value: ScalarExpr::Const(0.0),
        },
        Stmt::If {
            cond: guard,
            then: Block(vec![
                Stmt::Load {
                    local: 3,
                    place: tile(sub(tid(), u(d))),
                },
                Stmt::Assign {
                    local: 2,
                    value: l(3),
                },
            ]),
            els: Block(vec![]),
        },
        named(id),
    ];
    if mutant == Mutant::NoSecondSync {
        out.pop();
    }
    out.push(Stmt::Load {
        local: 6 + k,
        place: tile(tid()),
    });
    out.push(Stmt::Store {
        place: tile(tid()),
        value: add(l(6 + k), l(2)),
    });
    out.push(named(id));
    out
}

/// The whole scan for one group: load, sync, the steps, then the result to `out`.
fn group_body(id: u32, mutant: Mutant) -> Vec<Stmt> {
    let mut body = vec![
        Stmt::Load {
            local: 0,
            place: Place::Logical {
                buffer: 0,
                coords: vec![tid()],
            },
        },
        Stmt::Store {
            place: tile(tid()),
            value: l(0),
        },
        named(id),
    ];
    let steps: &[usize] = if mutant == Mutant::DropLastStep {
        &STEPS[..4]
    } else {
        &STEPS
    };
    for (k, d) in steps.iter().enumerate() {
        body.extend(step(*d, k as u32, id, mutant));
    }
    body.push(Stmt::Load {
        local: 5,
        place: tile(tid()),
    });
    body.push(Stmt::Store {
        place: Place::Logical {
            buffer: 1,
            coords: vec![tid()],
        },
        value: l(5),
    });
    body
}

/// The statements of group `g` (lanes `g * GROUP ..`), guarded by the thread index, on named barrier `g + 1`.
fn in_group(g: usize, mutant: Mutant) -> Stmt {
    let (lo, hi) = (g * GROUP, (g + 1) * GROUP);
    let cond = if g == 0 {
        cmp(BinaryOp::CmpLt, tid(), u(hi))
    } else {
        cmp(BinaryOp::CmpGe, tid(), u(lo))
    };
    Stmt::If {
        cond,
        then: Block(group_body(g as u32 + 1, mutant)),
        els: Block(vec![]),
    }
}

/// Buffers: 0 `input` `[THREADS]` (read), 1 `out` `[THREADS]` (write), both `F32`; allocation 0 the shared tile `[THREADS]`.
/// Locals: 0 the input, 1 the lane within the group (`U32`), 2 the neighbour term, 3 the loaded neighbour, 4 unused, 5 the result,
/// 6 onward each step's `s[i]`.
pub(crate) fn hillis_steele(mutant: Mutant) -> KernelFn {
    let body = vec![
        Stmt::Let {
            local: 1,
            init: sub(
                tid(),
                ScalarExpr::Mul(
                    Box::new(ScalarExpr::Div(Box::new(tid()), Box::new(u(GROUP)))),
                    Box::new(u(GROUP)),
                ),
            ),
        },
        Stmt::Let {
            local: 2,
            init: ScalarExpr::Const(0.0),
        },
        in_group(0, mutant),
        in_group(1, mutant),
    ];
    let params = vec![
        Param::buffer("input", F32, BufferRole::Read, Some(1)),
        Param::buffer("out", F32, BufferRole::Write, Some(1)),
    ];
    let mut locals = [F32, U32, F32, F32, F32, F32]
        .map(LocalDecl::scalar)
        .to_vec();
    locals.extend(vec![LocalDecl::scalar(F32); STEPS.len()]);
    KernelFn::new("hillis_steele", F32, params, locals, Block(body))
        .with_launch(LaunchShape::new([THREADS as u32, 1, 1], [1, 1, 1]))
        .with_allocs(vec![AllocDecl::new(
            "tile",
            MemorySpace::Shared,
            F32,
            Extent::Static(THREADS as u32),
        )])
}
