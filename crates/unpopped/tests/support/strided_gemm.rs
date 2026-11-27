// SPDX-License-Identifier: MIT OR Apache-2.0
//! GEMM over arbitrary strides at the kernel layer (CUTLASS survey row 7: `lda`, non-contiguous and transposed operands). The
//! kernel addresses `A[m][k]`, `B[k][n]` and `C[m][n]` by logical coordinates; the strides live on the argument views, so the same
//! kernel serves a padded row-major `A`, a column-major `B` and a padded `C`. One thread per output element; `ThreadIdx(0)` is
//! the column, `ThreadIdx(1)` the row.

use unpopped::exec::{ExecId, LaunchShape};
use unpopped::ir::ScalarExpr;
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt, Unroll};
use unpopped_vocab::ElementKind;

pub(crate) const F32: ElementKind = ElementKind::F32;
pub(crate) const U32: ElementKind = ElementKind::U32;
pub(crate) const M: usize = 3;
pub(crate) const N: usize = 4;
pub(crate) const K: usize = 4;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Mutant {
    None,
    /// `B` is read as `B[n][k]` (in range, since `N == K`).
    SwappedB,
    /// The output row is shifted by one, wrapping: row `m` is written to row `(m + 1) % M`.
    ShiftedRow,
}

fn cu(v: usize) -> ScalarExpr {
    ScalarExpr::ConstInt(v as i64, U32)
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
fn at2(buffer: u32, i: ScalarExpr, j: ScalarExpr) -> Place {
    Place::Logical {
        buffer,
        coords: vec![i, j],
    }
}

/// The output row of thread row `r`: `r`, or for the `ShiftedRow` mutant `(r + 1) % M` written as `r + 1 - M * (r / (M - 1))`.
fn out_row(r: ScalarExpr, mutant: Mutant) -> ScalarExpr {
    if mutant != Mutant::ShiftedRow {
        return r;
    }
    let wrap = mul(
        cu(M),
        ScalarExpr::Div(Box::new(r.clone()), Box::new(cu(M - 1))),
    );
    ScalarExpr::Sub(Box::new(add(r, cu(1))), Box::new(wrap))
}

/// The body of one `kk` step: the two loads and the accumulation.
fn k_step(mutant: Mutant, r: &ScalarExpr, c: &ScalarExpr) -> Vec<Stmt> {
    let b_place = if mutant == Mutant::SwappedB {
        at2(1, c.clone(), l(1))
    } else {
        at2(1, l(1), c.clone())
    };
    vec![
        Stmt::Load {
            local: 2,
            place: at2(0, r.clone(), l(1)),
        },
        Stmt::Load {
            local: 3,
            place: b_place,
        },
        Stmt::Assign {
            local: 0,
            value: add(l(0), mul(l(2), l(3))),
        },
    ]
}

/// Buffers: 0 `a` `[M, K]`, 1 `b` `[K, N]` (read), 2 `c` `[M, N]` (write). Locals: 0 `acc`, 1 `kk`, 2 `a`, 3 `b`.
pub(crate) fn strided_gemm(mutant: Mutant) -> KernelFn {
    let r = ScalarExpr::Exec(ExecId::ThreadIdx(1));
    let c = ScalarExpr::Exec(ExecId::ThreadIdx(0));
    let body = vec![
        Stmt::Let {
            local: 0,
            init: ScalarExpr::Const(0.0),
        },
        Stmt::For {
            unroll: Unroll::None,
            local: 1,
            start: cu(0),
            end: cu(K),
            step: cu(1),
            body: Block(k_step(mutant, &r, &c)),
        },
        Stmt::Store {
            place: at2(2, out_row(r, mutant), c),
            value: l(0),
        },
    ];
    let params = vec![
        Param::buffer("a", F32, BufferRole::Read, Some(2)),
        Param::buffer("b", F32, BufferRole::Read, Some(2)),
        Param::buffer("c", F32, BufferRole::Write, Some(2)),
    ];
    let locals = [F32, U32, F32, F32].map(LocalDecl::scalar).to_vec();
    KernelFn::new("strided_gemm", F32, params, locals, Block(body))
        .with_launch(LaunchShape::new([N as u32, M as u32, 1], [1, 1, 1]))
}
