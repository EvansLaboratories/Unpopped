// SPDX-License-Identifier: MIT OR Apache-2.0
//! A gather/scatter GEMM at the kernel layer (CUTLASS survey row 13; ROADMAP U9 scope A): `D[idx_d[m]][n] = sum_k A[idx_a[m]][k] *
//! B[k][n]` for `m in 0..M`, where `idx_a` gathers rows of `A` (duplicates allowed) and `idx_d` scatters rows of `D` (a negative
//! index skips the row, as a padded problem does). The indices are `I32` buffers read in the kernel, not pointers. Integer-valued
//! `F32` data, so the comparison is exact.

use unpopped::exec::{ExecId, LaunchShape};
use unpopped::ir::{BinaryOp, CastSpec, RoundMode, Saturation, ScalarExpr};
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt, Unroll};
use unpopped_vocab::ElementKind;

pub(crate) const F32: ElementKind = ElementKind::F32;
pub(crate) const I32: ElementKind = ElementKind::I32;
/// Problem rows, rows of `A`, rows of `D`, contraction, columns.
pub(crate) const M: usize = 5;
pub(crate) const RA: usize = 6;
pub(crate) const RD: usize = 7;
pub(crate) const K: usize = 4;
pub(crate) const N: usize = 3;
/// The gather indices (a repeated row) and the scatter indices (a skipped problem row, rows 0, 3 and 5 of `D` never written).
pub(crate) const IDX_A: [i32; M] = [3, 0, 5, 5, 2];
pub(crate) const IDX_D: [i32; M] = [6, 1, -1, 4, 2];
/// What an unwritten `D` element holds.
pub(crate) const POISON: f32 = 777.0;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Mutant {
    None,
    /// The gather index is applied to the scatter side too (rows 2 and 3 then both write `D` row 5: the oracle traps a data race).
    GatherIndexForScatter,
    /// The scatter index is read from the mirrored problem row `M - 1 - m`: unique targets, wrong rows.
    MirroredScatter,
    /// The scatter index is ignored: row `m` of `D` is written.
    NoScatter,
    /// A negative scatter index is not skipped (it is clamped to row 0).
    NegativeNotSkipped,
}

fn ci(v: i64) -> ScalarExpr {
    ScalarExpr::ConstInt(v, I32)
}
fn l(i: u32) -> ScalarExpr {
    ScalarExpr::Local(i)
}
fn add(a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Add(Box::new(a), Box::new(b))
}
fn mul(a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Mul(Box::new(a), Box::new(b))
}
fn exec_i32(id: ExecId) -> ScalarExpr {
    ScalarExpr::Cast(
        Box::new(ScalarExpr::Exec(id)),
        CastSpec {
            to: I32,
            round: RoundMode::NearestEven,
            sat: Saturation::Wrap,
        },
    )
}
fn at(buffer: u32, coords: Vec<ScalarExpr>) -> Place {
    Place::Logical { buffer, coords }
}
fn load(local: u32, place: Place) -> Stmt {
    Stmt::Load { local, place }
}

/// Loads the gather index into local 2 and the scatter index into local 3, as the mutant reads them.
fn index_prologue(mutant: Mutant, m: &ScalarExpr) -> Vec<Stmt> {
    let scatter_row = match mutant {
        Mutant::MirroredScatter => ScalarExpr::Sub(Box::new(ci(M as i64 - 1)), Box::new(m.clone())),
        _ => m.clone(),
    };
    let mut prologue = vec![
        load(2, at(2, vec![m.clone()])),
        load(3, at(3, vec![scatter_row])),
    ];
    match mutant {
        Mutant::GatherIndexForScatter => prologue.push(Stmt::Assign {
            local: 3,
            value: l(2),
        }),
        Mutant::NoScatter => prologue.push(Stmt::Assign {
            local: 3,
            value: m.clone(),
        }),
        Mutant::NegativeNotSkipped => prologue.push(Stmt::Assign {
            local: 3,
            value: ScalarExpr::Binary(BinaryOp::Max, Box::new(l(3)), Box::new(ci(0))),
        }),
        Mutant::None | Mutant::MirroredScatter => {}
    }
    prologue
}

/// The buffer parameters.
fn gather_params() -> Vec<Param> {
    vec![
        Param::buffer("a", F32, BufferRole::Read, Some(2)),
        Param::buffer("b", F32, BufferRole::Read, Some(2)),
        Param::buffer("idx_a", I32, BufferRole::Read, Some(1)),
        Param::buffer("idx_d", I32, BufferRole::Read, Some(1)),
        Param::buffer("d", F32, BufferRole::ReadWrite, Some(2)),
    ]
}

/// Buffers: 0 `a` `[RA, K]`, 1 `b` `[K, N]`, 2 `idx_a` `[M]` `I32`, 3 `idx_d` `[M]` `I32` (read), 4 `d` `[RD, N]` (write). Thread
/// `(x, y)` is problem element `(n, m)`. Locals: 0 acc, 1 `k`, 2 `ia`, 3 `id`, 4 `a`, 5 `b`.
pub(crate) fn gather_scatter_gemm(mutant: Mutant) -> KernelFn {
    let (n, m) = (
        exec_i32(ExecId::ThreadIdx(0)),
        exec_i32(ExecId::ThreadIdx(1)),
    );
    let mut prologue = index_prologue(mutant, &m);
    let reduce = Stmt::For {
        unroll: Unroll::None,
        local: 1,
        start: ci(0),
        end: ci(K as i64),
        step: ci(1),
        body: Block(vec![
            load(4, at(0, vec![l(2), l(1)])),
            load(5, at(1, vec![l(1), n.clone()])),
            Stmt::Assign {
                local: 0,
                value: add(l(0), mul(l(4), l(5))),
            },
        ]),
    };
    let written = Stmt::If {
        cond: ScalarExpr::Binary(BinaryOp::CmpLt, Box::new(ci(-1)), Box::new(l(3))),
        then: Block(vec![
            Stmt::Let {
                local: 0,
                init: ScalarExpr::Const(0.0),
            },
            reduce,
            Stmt::Store {
                place: at(4, vec![l(3), n]),
                value: l(0),
            },
        ]),
        els: Block(vec![]),
    };
    prologue.push(written);
    let mut locals = vec![LocalDecl::scalar(F32); 6];
    for i in [1, 2, 3] {
        locals[i] = LocalDecl::scalar(I32);
    }
    KernelFn::new(
        "gather_scatter_gemm",
        F32,
        gather_params(),
        locals,
        Block(prologue),
    )
    .with_launch(LaunchShape::new([N as u32, M as u32, 1], [1, 1, 1]))
}
