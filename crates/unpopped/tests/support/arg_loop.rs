// SPDX-License-Identifier: MIT OR Apache-2.0
//! The sequential form a kernel states for an arg-reduction (ROADMAP U5 P3): PyTorch's rule written as a loop over
//! locals, since the kernel layer has no lift of an arg-reduction. Shared by the oracle test
//! (`reduce_argmax.rs`) and the C test (`unpopped-cpu-c/tests/kernel_emit_argmax_agrees.rs`).
//!
//! ```text
//! for r in 0..rows { best = in[r][0]; bi = 0;
//!   for j in 1..cols { x = in[r][j]; if best == best { if x != x { take } else if x BETTER best { take } } }
//!   out[r] = bi }                     where take is `best = x; bi = j`
//! ```
//!
//! Locals: 0 `rows`, 1 `cols` (scalar parameters), 2 `r`, 3 `j` (loop counters), 4 `best`, 5 `bi`, 6 `x`.

use unpopped::ir::{BinaryOp, CastSpec, RoundMode, Saturation, ScalarExpr};
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt};
use unpopped_vocab::ElementKind;

const I32: ElementKind = ElementKind::I32;
const I64: ElementKind = ElementKind::I64;
const F32: ElementKind = ElementKind::F32;

fn l(i: u32) -> ScalarExpr {
    ScalarExpr::Local(i)
}

fn cmp(op: BinaryOp, a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Binary(op, Box::new(a), Box::new(b))
}

fn input_at(row_local: u32, col: ScalarExpr) -> Place {
    Place::Logical {
        buffer: 0,
        coords: vec![l(row_local), col],
    }
}

/// `best = x; bi = j`.
fn take() -> Block {
    let to_i64 = CastSpec {
        to: I64,
        round: RoundMode::NearestEven,
        sat: Saturation::Wrap,
    };
    Block(vec![
        Stmt::Assign {
            local: 4,
            value: l(6),
        },
        Stmt::Assign {
            local: 5,
            value: ScalarExpr::Cast(Box::new(l(3)), to_i64),
        },
    ])
}

/// The NaN-first rule: a NaN element takes over, otherwise only a strictly better one does.
fn consider(better: BinaryOp) -> Stmt {
    Stmt::If {
        cond: cmp(BinaryOp::CmpNe, l(6), l(6)),
        then: take(),
        els: Block(vec![Stmt::If {
            cond: cmp(better, l(6), l(4)),
            then: take(),
            els: Block(vec![]),
        }]),
    }
}

/// `for j in 1..cols { x = in[r][j]; if best == best { consider } }`: a NaN `best` is final.
fn scan_row(better: BinaryOp) -> Stmt {
    Stmt::For {
        local: 3,
        start: ScalarExpr::ConstInt(1, I32),
        end: l(1),
        step: ScalarExpr::ConstInt(1, I32),
        body: Block(vec![
            Stmt::Load {
                local: 6,
                place: input_at(2, l(3)),
            },
            Stmt::If {
                cond: cmp(BinaryOp::CmpEq, l(4), l(4)),
                then: Block(vec![consider(better)]),
                els: Block(vec![]),
            },
        ]),
    }
}

/// `for r in 0..rows { best = in[r][0]; bi = 0; <scan>; out[r] = bi }`.
fn rows(scan: Stmt) -> Stmt {
    Stmt::For {
        local: 2,
        start: ScalarExpr::ConstInt(0, I32),
        end: l(0),
        step: ScalarExpr::ConstInt(1, I32),
        body: Block(vec![
            Stmt::Load {
                local: 4,
                place: input_at(2, ScalarExpr::ConstInt(0, I32)),
            },
            Stmt::Let {
                local: 5,
                init: ScalarExpr::ConstInt(0, I64),
            },
            scan,
            Stmt::Store {
                place: Place::Logical {
                    buffer: 1,
                    coords: vec![l(2)],
                },
                value: l(5),
            },
        ]),
    }
}

/// The kernel: `x` (`F32`, rank 2, read), `out` (`I64`, rank 1, write), then `rows` and `cols`.
pub(crate) fn loop_kernel(max: bool) -> KernelFn {
    let better = if max {
        BinaryOp::CmpGt
    } else {
        BinaryOp::CmpLt
    };
    let locals = [I32, I32, I32, I32, F32, I64, F32]
        .into_iter()
        .map(LocalDecl::scalar)
        .collect();
    let params = vec![
        Param::buffer("x", F32, BufferRole::Read, Some(2)),
        Param::buffer("out", I64, BufferRole::Write, Some(1)),
        Param::scalar("rows", I32, 0),
        Param::scalar("cols", I32, 1),
    ];
    let body = Block(vec![rows(scan_row(better))]);
    KernelFn::new("argloop", F32, params, locals, body)
}
