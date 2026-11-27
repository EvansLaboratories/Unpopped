// SPDX-License-Identifier: MIT OR Apache-2.0
//! Split-K convolution, serial-reduction form, at the kernel layer (CUTLASS survey row 50). A 1-D convolution whose reduction runs
//! over the input channels is cut into `S` slices of channels. Launch 1 has one block per slice and writes that slice's partial
//! sums to a workspace `[S][O][WO]`; launch 2 sums the `S` partials of each output in slice order. The work is split by a fixed
//! partition of the reduction axis and combined by an ordered sum, with no work-stealing and no division of one tile between
//! blocks (the stream-K form is held and is not modelled here).

use unpopped::exec::{ExecId, LaunchShape};
use unpopped::ir::ScalarExpr;
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt, Unroll};
use unpopped_vocab::ElementKind;

pub(crate) const F32: ElementKind = ElementKind::F32;
pub(crate) const U32: ElementKind = ElementKind::U32;
pub(crate) const C: usize = 8;
pub(crate) const W: usize = 6;
pub(crate) const R: usize = 3;
pub(crate) const WO: usize = W - R + 1;
pub(crate) const O: usize = 3;
pub(crate) const S: usize = 4;
pub(crate) const CS: usize = C / S;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Mutant {
    None,
    /// Every slice reads the channels of slice 0.
    SliceZero,
    /// The reduction launch drops the last slice.
    DropLastSlice,
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
fn at(buffer: u32, i: ScalarExpr) -> Place {
    Place::Logical {
        buffer,
        coords: vec![i],
    }
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
fn load(local: u32, buffer: u32, i: ScalarExpr) -> Stmt {
    Stmt::Load {
        local,
        place: at(buffer, i),
    }
}

/// Launch 1. Buffers: 0 `x` `[C*W]`, 1 `w` `[O*C*R]`, 2 `ws` `[S*O*WO]`. One block per slice; `ThreadIdx(0)` is the output
/// position, `ThreadIdx(1)` the output channel. Locals: 0 `acc`, 1 `cc`, 2 `r`, 3 `x`, 4 `w`.
pub(crate) fn partial_conv(mutant: Mutant) -> KernelFn {
    let wo = ScalarExpr::Exec(ExecId::ThreadIdx(0));
    let o = ScalarExpr::Exec(ExecId::ThreadIdx(1));
    let slice = if mutant == Mutant::SliceZero {
        cu(0)
    } else {
        ScalarExpr::Exec(ExecId::BlockIdx(0))
    };
    let chan = add(mul(slice, cu(CS)), l(1));
    let step = vec![
        load(3, 0, add(mul(chan.clone(), cu(W)), add(wo.clone(), l(2)))),
        load(
            4,
            1,
            add(mul(add(mul(o.clone(), cu(C)), chan), cu(R)), l(2)),
        ),
        Stmt::Assign {
            local: 0,
            value: add(l(0), mul(l(3), l(4))),
        },
    ];
    let ws = add(
        mul(
            add(mul(ScalarExpr::Exec(ExecId::BlockIdx(0)), cu(O)), o),
            cu(WO),
        ),
        wo,
    );
    let body = vec![
        Stmt::Let {
            local: 0,
            init: ScalarExpr::Const(0.0),
        },
        counted(1, CS, vec![counted(2, R, step)]),
        Stmt::Store {
            place: at(2, ws),
            value: l(0),
        },
    ];
    let params = vec![
        Param::buffer("x", F32, BufferRole::Read, Some(1)),
        Param::buffer("w", F32, BufferRole::Read, Some(1)),
        Param::buffer("ws", F32, BufferRole::Write, Some(1)),
    ];
    let locals = [F32, U32, U32, F32, F32].map(LocalDecl::scalar).to_vec();
    KernelFn::new("splitk_partial", F32, params, locals, Block(body))
        .with_launch(LaunchShape::new([WO as u32, O as u32, 1], [S as u32, 1, 1]))
}

/// Launch 2. Buffers: 0 `ws` `[S*O*WO]`, 1 `out` `[O*WO]`. One thread per output. Locals: 0 `acc`, 1 `s`, 2 `p`.
pub(crate) fn reduce_slices(mutant: Mutant) -> KernelFn {
    let wo = ScalarExpr::Exec(ExecId::ThreadIdx(0));
    let o = ScalarExpr::Exec(ExecId::ThreadIdx(1));
    let cell = add(mul(o, cu(WO)), wo);
    let slices = if mutant == Mutant::DropLastSlice {
        S - 1
    } else {
        S
    };
    let step = vec![
        load(2, 0, add(mul(l(1), cu(O * WO)), cell.clone())),
        Stmt::Assign {
            local: 0,
            value: add(l(0), l(2)),
        },
    ];
    let body = vec![
        Stmt::Let {
            local: 0,
            init: ScalarExpr::Const(0.0),
        },
        counted(1, slices, step),
        Stmt::Store {
            place: at(1, cell),
            value: l(0),
        },
    ];
    let params = vec![
        Param::buffer("ws", F32, BufferRole::Read, Some(1)),
        Param::buffer("out", F32, BufferRole::Write, Some(1)),
    ];
    let locals = [F32, U32, F32].map(LocalDecl::scalar).to_vec();
    KernelFn::new("splitk_reduce", F32, params, locals, Block(body))
        .with_launch(LaunchShape::new([WO as u32, O as u32, 1], [1, 1, 1]))
}
