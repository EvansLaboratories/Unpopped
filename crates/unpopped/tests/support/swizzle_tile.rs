// SPDX-License-Identifier: MIT OR Apache-2.0
//! A shared-memory tile addressed through an XOR swizzle at the kernel layer (CUTLASS survey row 66: swizzled shared tiles,
//! `Crosswise`/`Congruous`, the CuTe `Swizzle`). The swizzle is index arithmetic only (`col ^ row`), the pure-data form of
//! `docs/PATENT-SWEEP.md` idea 15 (no copy hardware, no lookup tables). An 8 x 8 transpose goes through the tile: thread
//! `(x, y)` stores `in[y][x]` at `tile[y][x ^ y]`, the block synchronises, and thread `(x, y)` loads `tile[x][y ^ x]`, which
//! holds `in[x][y]`, and stores it as `out[y][x]`.

use unpopped::exec::{AllocDecl, ExecId, Extent, LaunchShape, MemorySpace, Scope};
use unpopped::ir::{BinaryOp, CastSpec, RoundMode, Saturation, ScalarExpr};
use unpopped::kernel::{BarrierKind, Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt};
use unpopped_vocab::ElementKind;

pub(crate) const F32: ElementKind = ElementKind::F32;
pub(crate) const I32: ElementKind = ElementKind::I32;
pub(crate) const T: usize = 8;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Mutant {
    None,
    /// The load forgets the swizzle: `tile[x][y]`.
    ReadUnswizzled,
    /// The store narrows the swizzle mask: `x ^ (y & 3)`, while the load still uses `y ^ x`.
    NarrowMask,
}

fn ci(v: usize) -> ScalarExpr {
    ScalarExpr::ConstInt(v as i64, I32)
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
fn bin(op: BinaryOp, a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Binary(op, Box::new(a), Box::new(b))
}
fn i32_of(x: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Cast(
        Box::new(x),
        CastSpec {
            to: I32,
            round: RoundMode::NearestEven,
            sat: Saturation::Wrap,
        },
    )
}
fn flat(buffer: u32, i: ScalarExpr) -> Place {
    Place::Logical {
        buffer,
        coords: vec![i],
    }
}
/// The tile slot of row `row`, column `col`.
fn slot(row: ScalarExpr, col: ScalarExpr) -> Place {
    Place::Alloc {
        alloc: 0,
        index: add(mul(row, ci(T)), col),
    }
}

/// Buffers: 0 `in` `[64]` (read), 1 `out` `[64]` (write). Locals: 0 `x`, 1 `y`, 2 the loaded element, 3 the tile element.
/// Tile: allocation 0, 64 `F32` in shared memory.
pub(crate) fn swizzle_transpose(mutant: Mutant) -> KernelFn {
    let store_col = if mutant == Mutant::NarrowMask {
        bin(BinaryOp::BitXor, l(0), bin(BinaryOp::BitAnd, l(1), ci(3)))
    } else {
        bin(BinaryOp::BitXor, l(0), l(1))
    };
    let load_col = if mutant == Mutant::ReadUnswizzled {
        l(1)
    } else {
        bin(BinaryOp::BitXor, l(1), l(0))
    };
    let body = vec![
        Stmt::Let {
            local: 0,
            init: i32_of(ScalarExpr::Exec(ExecId::ThreadIdx(0))),
        },
        Stmt::Let {
            local: 1,
            init: i32_of(ScalarExpr::Exec(ExecId::ThreadIdx(1))),
        },
        Stmt::Load {
            local: 2,
            place: flat(0, add(mul(l(1), ci(T)), l(0))),
        },
        Stmt::Store {
            place: slot(l(1), store_col),
            value: l(2),
        },
        Stmt::Barrier {
            scope: Scope::Block,
            kind: BarrierKind::Full,
        },
        Stmt::Load {
            local: 3,
            place: slot(l(0), load_col),
        },
        Stmt::Store {
            place: flat(1, add(mul(l(1), ci(T)), l(0))),
            value: l(3),
        },
    ];
    let params = vec![
        Param::buffer("input", F32, BufferRole::Read, Some(1)),
        Param::buffer("out", F32, BufferRole::Write, Some(1)),
    ];
    let locals = [I32, I32, F32, F32].map(LocalDecl::scalar).to_vec();
    KernelFn::new("swizzle_transpose", F32, params, locals, Block(body))
        .with_launch(LaunchShape::new([T as u32, T as u32, 1], [1, 1, 1]))
        .with_allocs(vec![AllocDecl::new(
            "tile",
            MemorySpace::Shared,
            F32,
            Extent::Static((T * T) as u32),
        )])
}
