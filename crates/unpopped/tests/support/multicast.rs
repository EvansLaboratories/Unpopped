// SPDX-License-Identifier: MIT OR Apache-2.0
//! The kernels of `kernel_multicast.rs` (ROADMAP U12 slice 3, `docs/options-u12-peer-multicast.md`): `Place::Multicast`
//! names one element on every device at once. A `Store` writes all copies (`multimem.st`); an `AtomicRmw` combines into
//! all copies (`multimem.red`). It builds on `device_peer.rs` (`super::fixture`), whose launch it shares: 3 devices of 2
//! blocks of 2 threads.

use super::fixture::{at, barrier, c, gather_launch, gid, load, local, rank, store};
use unpopped::exec::Scope;
use unpopped::kernel::{
    AtomicOp, Block, BufferRole, KernelFn, LocalDecl, MemoryOrder, Param, Place, Stmt,
};
use unpopped_vocab::ElementKind;

/// Element `i` of buffer `buffer` on every device.
pub(crate) fn multicast(buffer: u32, i: unpopped::ir::ScalarExpr) -> Place {
    Place::Multicast {
        buffer,
        coords: vec![i],
    }
}

/// A kernel over buffers `inp` (read), `acc` and `res` (read and written) of one element type, the launch of
/// `device_peer.rs`.
fn kernel_of(elem: ElementKind, name: &str, (locals, body): (usize, Vec<Stmt>)) -> KernelFn {
    let params = [
        ("inp", BufferRole::Read),
        ("acc", BufferRole::ReadWrite),
        ("res", BufferRole::ReadWrite),
    ]
    .into_iter()
    .map(|(n, role)| Param::buffer(n, elem, role, Some(1)))
    .collect();
    KernelFn::new(
        name,
        elem,
        params,
        (0..locals).map(|_| LocalDecl::scalar(elem)).collect(),
        Block(body),
    )
    .with_launch(gather_launch())
}

/// `multimem.red` of `value` into `acc[gid]` of every device.
pub(crate) fn red(op: AtomicOp, value: unpopped::ir::ScalarExpr) -> Stmt {
    Stmt::AtomicRmw {
        op,
        place: multicast(1, gid()),
        value,
        old: None,
        order: MemoryOrder::Release,
        scope: Scope::System,
    }
}

/// An all-reduce by `op` over the devices' `inp[gid]` into the `acc[gid]` of every device by one `multimem.red` per
/// thread, then (after `sync`, a barrier of that scope, if any) every device reads its own copy into `res[gid]`.
/// Locals: 0 the input, 1 the reduced value.
pub(crate) fn all_reduce(elem: ElementKind, op: AtomicOp, sync: Option<Scope>) -> KernelFn {
    let mut body = vec![load(0, at(0, gid())), red(op, local(0))];
    body.extend(sync.map(barrier));
    body.push(load(1, at(1, gid())));
    body.push(store(at(2, gid()), local(1)));
    kernel_of(elem, "all_reduce_red", (2, body))
}

/// Device 0 writes its `inp[gid]` into `acc[gid]` of every device by one `multimem.st` per thread; after a `System`
/// barrier every device reads its own copy into `res[gid]`. Locals: 0 the input, 1 the broadcast value.
pub(crate) fn broadcast(elem: ElementKind, sync: Option<Scope>) -> KernelFn {
    let mut body = vec![
        load(0, at(0, gid())),
        Stmt::If {
            cond: unpopped::ir::ScalarExpr::Binary(
                unpopped::ir::BinaryOp::CmpEq,
                Box::new(rank()),
                Box::new(c(0)),
            ),
            then: Block(vec![store(multicast(1, gid()), local(0))]),
            els: Block(vec![]),
        },
    ];
    body.extend(sync.map(barrier));
    body.push(load(1, at(1, gid())));
    body.push(store(at(2, gid()), local(1)));
    kernel_of(elem, "broadcast_st", (2, body))
}

/// A kernel whose one statement is `st`, over the buffers of `all_reduce` (for the checker's refusals).
pub(crate) fn one(elem: ElementKind, st: Stmt) -> KernelFn {
    kernel_of(elem, "multicast_one", (1, vec![st]))
}
