// SPDX-License-Identifier: MIT OR Apache-2.0
//! Grouped block rasterization (the "grouped ordering" of Triton's public matmul tutorial; TileLang `T.use_swizzle`; survey row
//! 31 of `docs/ir-union-evidence/conformance-idiom-gaps-2026-11-27.md`): a linear tile id is mapped to `(tile_m, tile_n)` so that
//! consecutive ids walk `GROUP_M` rows of tiles down a column before moving to the next column, for L2 reuse. Integer index
//! arithmetic only: one thread per tile id writes the coordinates. It models the mapping, not scheduling: no persistent loop, no
//! work stealing, no division of a tile between blocks. `ThreadIdx(0)` is the tile id.

use unpopped::exec::{ExecId, LaunchShape};
use unpopped::ir::{BinaryOp, ScalarExpr};
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt};
use unpopped_vocab::ElementKind;

pub(crate) const F32: ElementKind = ElementKind::F32;
pub(crate) const U32: ElementKind = ElementKind::U32;
pub(crate) const NUM_M: usize = 5;
pub(crate) const NUM_N: usize = 4;
pub(crate) const GROUP_M: usize = 2;
pub(crate) const TILES: usize = NUM_M * NUM_N;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Mutant {
    None,
    /// The last group is not clamped: `group_size_m` is always `GROUP_M`.
    NoClamp,
    /// `tile_n` is taken modulo the group instead of divided: `(pid % group_in) % group_size_m`.
    ModuloN,
}

fn u(v: usize) -> ScalarExpr {
    ScalarExpr::ConstInt(v as i64, U32)
}
fn l(i: u32) -> ScalarExpr {
    ScalarExpr::Local(i)
}
fn bin(op: BinaryOp, a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Binary(op, Box::new(a), Box::new(b))
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
/// `a % b` for unsigned `a`, `b`.
fn rem(a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    sub(a.clone(), mul(div(a, b.clone()), b))
}
fn at(buffer: u32) -> Place {
    Place::Logical {
        buffer,
        coords: vec![ScalarExpr::Exec(ExecId::ThreadIdx(0))],
    }
}

/// Buffers: 0 `tile_m` `[TILES]`, 1 `tile_n` `[TILES]` (write, `U32`). Locals: 0 the tile id, 1 `group_in` (tiles per group), 2 the
/// group number, 3 the group's first row, 4 `group_size_m`, 5 `tile_m`, 6 `tile_n`.
pub(crate) fn raster_group(mutant: Mutant) -> KernelFn {
    let size = if mutant == Mutant::NoClamp {
        u(GROUP_M)
    } else {
        bin(BinaryOp::Min, sub(u(NUM_M), l(3)), u(GROUP_M))
    };
    let within = rem(l(0), l(1));
    let tile_n = if mutant == Mutant::ModuloN {
        rem(within.clone(), l(4))
    } else {
        div(within, l(4))
    };
    let lets = [
        (0, ScalarExpr::Exec(ExecId::ThreadIdx(0))),
        (1, u(GROUP_M * NUM_N)),
        (2, div(l(0), l(1))),
        (3, mul(l(2), u(GROUP_M))),
        (4, size),
        (5, add(l(3), rem(l(0), l(4)))),
        (6, tile_n),
    ];
    let mut body: Vec<Stmt> = lets
        .into_iter()
        .map(|(local, init)| Stmt::Let { local, init })
        .collect();
    body.push(Stmt::Store {
        place: at(0),
        value: l(5),
    });
    body.push(Stmt::Store {
        place: at(1),
        value: l(6),
    });
    let params = vec![
        Param::buffer("tile_m", U32, BufferRole::Write, Some(1)),
        Param::buffer("tile_n", U32, BufferRole::Write, Some(1)),
    ];
    let locals = vec![LocalDecl::scalar(U32); 7];
    KernelFn::new("raster_group", F32, params, locals, Block(body))
        .with_launch(LaunchShape::new([TILES as u32, 1, 1], [1, 1, 1]))
}
