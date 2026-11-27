// SPDX-License-Identifier: MIT OR Apache-2.0
//! Interleaved-layout contractions at the kernel layer (CUTLASS survey row 6). Two kernels read operands stored interleaved
//! by addressing the interleave as extra coordinates of a higher-rank logical buffer, which is how a union-language kernel
//! spells it:
//!
//! * `gemm_interleaved`: `C[M,N] = A[M,K] B[K,N]` with `A` in `ColumnMajorInterleaved<I>` and `B` in `RowMajorInterleaved<I>`
//!   (interleaved along `K`, the int8 "TN" shape). Physical `A` is `[K/I, M, I]` read at `(k / I, m, k % I)`; physical `B` is
//!   `[K/I, N, I]` read at `(k / I, n, k % I)`.
//! * `conv1x1_nc4hw4`: a 1x1 convolution `out[n,o,h,w] = sum_c in[n,c,h,w] * wgt[o,c]` with the input in `NCxHWx` (`x = I`),
//!   physical `[N, C/I, H, W, I]` read at `(n, c / I, h, w, c % I)`.
//!
//! The layout formulas are CUTLASS's `ColumnMajorInterleaved` / `RowMajorInterleaved` (`cutlass/layout/matrix.h`) and the
//! `TensorNCxHWx` layout (`cutlass/layout/tensor.h`), recalled and UNVERIFIED against a checkout; the test packs with flat
//! offset formulas written independently of the kernel's coordinates.

use unpopped::exec::{ExecId, LaunchShape};
use unpopped::ir::ScalarExpr;
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt, Unroll};
use unpopped_vocab::ElementKind;

pub(crate) const F32: ElementKind = ElementKind::F32;
pub(crate) const U32: ElementKind = ElementKind::U32;
/// The interleave.
pub(crate) const I: usize = 4;
/// GEMM: rows of `A` and `C`, columns of `B` and `C`, the contraction.
pub(crate) const M: usize = 4;
pub(crate) const N: usize = 6;
pub(crate) const K: usize = 8;
/// Conv: batch, input channels, output channels, height, width.
pub(crate) const NB: usize = 2;
pub(crate) const C: usize = 8;
pub(crate) const O: usize = 3;
pub(crate) const HH: usize = 2;
pub(crate) const WW: usize = 3;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Mutant {
    None,
    /// `A` is read at `(k % (K/I), m, k / (K/I))`: major and minor of the interleave swapped, in range for these extents. `B`
    /// keeps the right reading: the same swap on both operands is a permutation of `k` and the sum does not change, so it
    /// would be an equivalent mutant.
    SwappedMinor,
    /// The conv input is read at `(n, c / I, h, w, c / 2 % I)`: a wrong minor coordinate, in range.
    WrongMinor,
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
fn sub(a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Sub(Box::new(a), Box::new(b))
}
fn mul(a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Mul(Box::new(a), Box::new(b))
}
fn div(a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Div(Box::new(a), Box::new(b))
}
/// `x % m` for an unsigned integer.
fn rem(x: ScalarExpr, m: usize) -> ScalarExpr {
    sub(x.clone(), mul(div(x, cu(m)), cu(m)))
}
fn tid(d: u8) -> ScalarExpr {
    ScalarExpr::Exec(ExecId::ThreadIdx(d))
}
fn at(buffer: u32, coords: Vec<ScalarExpr>) -> Place {
    Place::Logical { buffer, coords }
}
fn counted(local: u32, end: usize, body: Vec<Stmt>) -> Stmt {
    Stmt::For {
        unroll: Unroll::None,
        local,
        start: cu(0),
        end: cu(end),
        step: cu(1),
        body: Block(body),
    }
}

/// Buffers: 0 `a` `[K/I, M, I]`, 1 `b` `[K/I, N, I]` (read), 2 `c` `[M, N]` (write). Thread `m` computes row `m`.
/// Locals: 0 acc, 1 `n`, 2 `k`, 3 `a`, 4 `b`.
pub(crate) fn gemm_interleaved(mutant: Mutant) -> KernelFn {
    let (major, minor) = (div(l(2), cu(I)), rem(l(2), I));
    let (a_major, a_minor) = if mutant == Mutant::SwappedMinor {
        (rem(l(2), K / I), div(l(2), cu(K / I)))
    } else {
        (major.clone(), minor.clone())
    };
    let inner = vec![
        Stmt::Load {
            local: 3,
            place: at(0, vec![a_major, tid(0), a_minor]),
        },
        Stmt::Load {
            local: 4,
            place: at(1, vec![major, l(1), minor]),
        },
        Stmt::Assign {
            local: 0,
            value: add(l(0), mul(l(3), l(4))),
        },
    ];
    let column = vec![
        Stmt::Let {
            local: 0,
            init: ScalarExpr::Const(0.0),
        },
        counted(2, K, inner),
        Stmt::Store {
            place: at(2, vec![tid(0), l(1)]),
            value: l(0),
        },
    ];
    let mut locals = vec![LocalDecl::scalar(F32); 5];
    locals[1] = LocalDecl::scalar(U32);
    locals[2] = LocalDecl::scalar(U32);
    KernelFn::new(
        "gemm_interleaved",
        F32,
        vec![
            Param::buffer("a", F32, BufferRole::Read, Some(3)),
            Param::buffer("b", F32, BufferRole::Read, Some(3)),
            Param::buffer("c", F32, BufferRole::ReadWrite, Some(2)),
        ],
        locals,
        Block(vec![counted(1, N, column)]),
    )
    .with_launch(LaunchShape::new([M as u32, 1, 1], [1, 1, 1]))
}

/// Buffers: 0 `inp` `[N, C/I, H, W, I]`, 1 `wgt` `[O, C]` (read), 2 `out` `[N, O, H, W]` (write). Thread `(w, h, n)` is block
/// index `(0, 1, 2)`. Locals: 0 acc, 1 `o`, 2 `c`, 3 `x`, 4 `w`.
pub(crate) fn conv1x1_nc4hw4(mutant: Mutant) -> KernelFn {
    let major = div(l(2), cu(I));
    let minor = if mutant == Mutant::WrongMinor {
        rem(div(l(2), cu(2)), I)
    } else {
        rem(l(2), I)
    };
    let inner = vec![
        Stmt::Load {
            local: 3,
            place: at(0, vec![tid(2), major, tid(1), tid(0), minor]),
        },
        Stmt::Load {
            local: 4,
            place: at(1, vec![l(1), l(2)]),
        },
        Stmt::Assign {
            local: 0,
            value: add(l(0), mul(l(3), l(4))),
        },
    ];
    let per_out = vec![
        Stmt::Let {
            local: 0,
            init: ScalarExpr::Const(0.0),
        },
        counted(2, C, inner),
        Stmt::Store {
            place: at(2, vec![tid(2), l(1), tid(1), tid(0)]),
            value: l(0),
        },
    ];
    let mut locals = vec![LocalDecl::scalar(F32); 5];
    locals[1] = LocalDecl::scalar(U32);
    locals[2] = LocalDecl::scalar(U32);
    KernelFn::new(
        "conv1x1_nc4hw4",
        F32,
        vec![
            Param::buffer("inp", F32, BufferRole::Read, Some(5)),
            Param::buffer("wgt", F32, BufferRole::Read, Some(2)),
            Param::buffer("out", F32, BufferRole::ReadWrite, Some(4)),
        ],
        locals,
        Block(vec![counted(1, O, per_out)]),
    )
    .with_launch(LaunchShape::new(
        [WW as u32, HH as u32, NB as u32],
        [1, 1, 1],
    ))
}
