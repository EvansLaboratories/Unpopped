// SPDX-License-Identifier: MIT OR Apache-2.0
//! The dynamic-grid kernels of ROADMAP U22, shared by the oracle test (`kernel_grid_dynamic.rs`) and the C test
//! (`unpopped-cpu-c/tests/kernel_grid_dynamic_emit_agrees.rs`, which includes this file).

use unpopped::exec::{ExecId, GridDim, LaunchShape};
use unpopped::ir::{BinaryOp, CastSpec, RoundMode, Saturation, ScalarExpr};
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt};
use unpopped_vocab::ElementKind;

pub(crate) const F32: ElementKind = ElementKind::F32;
pub(crate) const I32: ElementKind = ElementKind::I32;
pub(crate) const U32: ElementKind = ElementKind::U32;
pub(crate) const BLOCK: u32 = 4;

pub(crate) fn local(i: u32) -> ScalarExpr {
    ScalarExpr::Local(i)
}

pub(crate) fn at(buffer: u32, i: ScalarExpr) -> Place {
    Place::Logical {
        buffer,
        coords: vec![i],
    }
}

pub(crate) fn to_u32(x: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Cast(
        Box::new(x),
        CastSpec {
            to: U32,
            round: RoundMode::NearestEven,
            sat: Saturation::Wrap,
        },
    )
}

/// `out[g] = 2 * in[g]` for `g < n`, `g = blockIdx.x * blockDim.x + threadIdx.x`; the grid is `ceil(n / BLOCK)`.
/// Params: 0 `in`, 1 `out`, 2 `n` (local 0). Locals: 0 `n`, 1 `g`, 2 `x`.
pub(crate) fn doubler(grid: GridDim) -> KernelFn {
    let g = ScalarExpr::Add(
        Box::new(ScalarExpr::Mul(
            Box::new(ScalarExpr::Exec(ExecId::BlockIdx(0))),
            Box::new(ScalarExpr::Exec(ExecId::BlockDim(0))),
        )),
        Box::new(ScalarExpr::Exec(ExecId::ThreadIdx(0))),
    );
    let body = vec![
        Stmt::Let { local: 1, init: g },
        Stmt::If {
            cond: ScalarExpr::Binary(
                BinaryOp::CmpLt,
                Box::new(local(1)),
                Box::new(to_u32(local(0))),
            ),
            then: Block(vec![
                Stmt::Load {
                    local: 2,
                    place: at(0, local(1)),
                },
                Stmt::Store {
                    place: at(1, local(1)),
                    value: ScalarExpr::Mul(Box::new(local(2)), Box::new(ScalarExpr::Const(2.0))),
                },
            ]),
            els: Block(vec![]),
        },
    ];
    KernelFn::new(
        "doubler",
        F32,
        vec![
            Param::buffer("in", F32, BufferRole::Read, Some(1)),
            Param::buffer("out", F32, BufferRole::ReadWrite, Some(1)),
            Param::scalar("n", I32, 0),
        ],
        vec![
            LocalDecl::scalar(I32),
            LocalDecl::scalar(U32),
            LocalDecl::scalar(F32),
        ],
        Block(body),
    )
    .with_launch(LaunchShape::new([BLOCK, 1, 1], [1, 1, 1]).with_grid_dims([
        grid,
        GridDim::Static(1),
        GridDim::Static(1),
    ]))
}

/// `gridDim.x` as a float, stored into `out[0]` by the one thread whose block and thread index are both 0.
pub(crate) fn grid_probe(grid: GridDim) -> KernelFn {
    let mut k = doubler(grid);
    let first = ScalarExpr::Binary(
        BinaryOp::CmpLt,
        Box::new(ScalarExpr::Add(
            Box::new(ScalarExpr::Exec(ExecId::BlockIdx(0))),
            Box::new(ScalarExpr::Exec(ExecId::ThreadIdx(0))),
        )),
        Box::new(ScalarExpr::ConstInt(1, U32)),
    );
    k.body = Block(vec![Stmt::If {
        cond: first,
        then: Block(vec![Stmt::Store {
            place: at(1, ScalarExpr::ConstInt(0, U32)),
            value: ScalarExpr::Cast(
                Box::new(ScalarExpr::Exec(ExecId::GridDim(0))),
                CastSpec {
                    to: F32,
                    round: RoundMode::NearestEven,
                    sat: Saturation::Wrap,
                },
            ),
        }]),
        els: Block(vec![]),
    }]);
    k
}
