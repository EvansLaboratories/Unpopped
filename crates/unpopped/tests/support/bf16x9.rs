// SPDX-License-Identifier: MIT OR Apache-2.0
//! Emulated `f32` GEMM by nine-term `bf16` splitting at the kernel layer (CUTLASS survey row 27). Each `f32` operand is split
//! into three `bf16` pieces (`x0 = bf16(x)`, `x1 = bf16(x - x0)`, `x2 = bf16(x - x0 - x1)`), and the nine products `a_i * b_j`
//! (each exact in `f32`: two 8-bit mantissas) are accumulated in `f32`, in `i`-major order. Arithmetic only; no hardware atom.
//! One thread per output element; `ThreadIdx(0)` is the column, `ThreadIdx(1)` the row.

use unpopped::exec::{ExecId, LaunchShape};
use unpopped::ir::{CastSpec, RoundMode, Saturation, ScalarExpr};
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt, Unroll};
use unpopped_vocab::ElementKind;

pub(crate) const F32: ElementKind = ElementKind::F32;
pub(crate) const U32: ElementKind = ElementKind::U32;
pub(crate) const M: usize = 2;
pub(crate) const N: usize = 3;
pub(crate) const K: usize = 4;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Mutant {
    None,
    /// Only the leading piece: a plain `bf16` product.
    HiOnly,
    /// The last piece forgets the first one: `x2 = bf16(x - x1)`.
    UnchainedResidual,
    /// The pieces of `B` are paired in reverse: `a_i * b_(2-i)` style crossing (`a_i * b_(2-j)`).
    ReversedPieces,
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
fn at(buffer: u32, i: ScalarExpr) -> Place {
    Place::Logical {
        buffer,
        coords: vec![i],
    }
}
fn cast(x: ScalarExpr, to: ElementKind) -> ScalarExpr {
    ScalarExpr::Cast(
        Box::new(x),
        CastSpec {
            to,
            round: RoundMode::NearestEven,
            sat: Saturation::Wrap,
        },
    )
}
/// `x` rounded to `bf16` and widened back to `f32`.
fn round_bf16(x: ScalarExpr) -> ScalarExpr {
    cast(cast(x, ElementKind::Bf16), F32)
}

/// The three `Let`s of the pieces `base`, `base+1`, `base+2` of local `src`.
fn split(src: u32, base: u32, mutant: Mutant) -> Vec<Stmt> {
    let tail = if mutant == Mutant::UnchainedResidual {
        sub(l(src), l(base + 1))
    } else {
        sub(sub(l(src), l(base)), l(base + 1))
    };
    let lets = [
        round_bf16(l(src)),
        round_bf16(sub(l(src), l(base))),
        round_bf16(tail),
    ];
    (0..3u32)
        .zip(lets)
        .map(|(i, init)| Stmt::Let {
            local: base + i,
            init,
        })
        .collect()
}

/// The nine accumulations in `i`-major order (one for the `HiOnly` mutant), each its own `Assign` so that every add rounds to `f32`
/// (the oracle rounds a whole expression once, at the `Assign`; a step per term keeps the rounding independent of that).
fn terms(mutant: Mutant) -> Vec<Stmt> {
    let mut out = Vec::new();
    for i in 0..3u32 {
        for j in 0..3u32 {
            if mutant == Mutant::HiOnly && (i, j) != (0, 0) {
                continue;
            }
            let bj = if mutant == Mutant::ReversedPieces {
                2 - j
            } else {
                j
            };
            out.push(Stmt::Assign {
                local: 0,
                value: add(l(0), mul(l(4 + i), l(7 + bj))),
            });
        }
    }
    out
}

/// Buffers: 0 `a` `[M*K]`, 1 `b` `[K*N]`, 2 `out` `[M*N]`, all `F32`. Locals: 0 `acc`, 1 `kk`, 2 `a`, 3 `b`, 4..=6 the pieces
/// of `a`, 7..=9 the pieces of `b`.
pub(crate) fn bf16x9_gemm(mutant: Mutant) -> KernelFn {
    let r = ScalarExpr::Exec(ExecId::ThreadIdx(1));
    let c = ScalarExpr::Exec(ExecId::ThreadIdx(0));
    let mut step = vec![
        Stmt::Load {
            local: 2,
            place: at(0, add(mul(r.clone(), cu(K)), l(1))),
        },
        Stmt::Load {
            local: 3,
            place: at(1, add(mul(l(1), cu(N)), c.clone())),
        },
    ];
    step.extend(split(2, 4, mutant));
    step.extend(split(3, 7, mutant));
    step.extend(terms(mutant));
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
            body: Block(step),
        },
        Stmt::Store {
            place: at(2, add(mul(r, cu(N)), c)),
            value: l(0),
        },
    ];
    let params = vec![
        Param::buffer("a", F32, BufferRole::Read, Some(1)),
        Param::buffer("b", F32, BufferRole::Read, Some(1)),
        Param::buffer("out", F32, BufferRole::ReadWrite, Some(1)),
    ];
    let mut locals = vec![LocalDecl::scalar(F32), LocalDecl::scalar(U32)];
    locals.extend(vec![LocalDecl::scalar(F32); 8]);
    KernelFn::new("bf16x9_gemm", F32, params, locals, Block(body))
        .with_launch(LaunchShape::new([N as u32, M as u32, 1], [1, 1, 1]))
}
