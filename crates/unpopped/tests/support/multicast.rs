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

/// `multimem.ld_reduce` of element `buffer[gid]` over the devices into local `local`.
pub(crate) fn ld_reduce(local: u32, buffer: u32, (op, order): (AtomicOp, MemoryOrder)) -> Stmt {
    Stmt::MulticastLoadReduce {
        local,
        place: multicast(buffer, gid()),
        op,
        order,
    }
}

/// `if rank == 0 { then }`.
fn on_device_zero(then: Vec<Stmt>) -> Stmt {
    Stmt::If {
        cond: unpopped::ir::ScalarExpr::Binary(
            unpopped::ir::BinaryOp::CmpEq,
            Box::new(rank()),
            Box::new(c(0)),
        ),
        then: Block(then),
        els: Block(vec![]),
    }
}

/// Every device reduces the devices' `inp[gid]` by `ld_reduce` into its own `res[gid]`. Locals: 0 the reduced value.
pub(crate) fn reduce_inputs(elem: ElementKind, form: (AtomicOp, MemoryOrder)) -> KernelFn {
    let body = vec![ld_reduce(0, 0, form), store(at(2, gid()), local(0))];
    kernel_of(elem, "ld_reduce_inputs", (1, body))
}

/// Device 0 writes its `inp[gid]` into `acc[gid]` of every device by `multimem.st`; after `sync` every device reduces
/// the devices' `acc[gid]` by `ld_reduce` into its `res[gid]`. Locals: 0 the input, 1 the reduced value.
pub(crate) fn st_then_ld_reduce(
    elem: ElementKind,
    form: (AtomicOp, MemoryOrder),
    sync: Option<Scope>,
) -> KernelFn {
    let mut body = vec![
        load(0, at(0, gid())),
        on_device_zero(vec![store(multicast(1, gid()), local(0))]),
    ];
    body.extend(sync.map(barrier));
    body.push(ld_reduce(1, 1, form));
    body.push(store(at(2, gid()), local(1)));
    kernel_of(elem, "st_then_ld_reduce", (2, body))
}

/// The second all-reduce form: device 0 reduces the devices' `inp[gid]` by `ld_reduce` and writes the result into
/// `acc[gid]` of every device by `multimem.st`; after a `System` barrier every device reads its own copy into `res[gid]`.
/// Locals: 0 the reduced value, 1 the copy read back.
pub(crate) fn ld_reduce_then_st(elem: ElementKind, form: (AtomicOp, MemoryOrder)) -> KernelFn {
    let body = vec![
        on_device_zero(vec![
            ld_reduce(0, 0, form),
            store(multicast(1, gid()), local(0)),
        ]),
        barrier(Scope::System),
        load(1, at(1, gid())),
        store(at(2, gid()), local(1)),
    ];
    kernel_of(elem, "ld_reduce_then_st", (2, body))
}

/// A hand-off across devices ordered by the `ld_reduce`'s `order`. Thread 0 of device 0 writes 7 into its own `res[0]`,
/// then publishes with `multimem.red` (`Release`, `System`) on `acc[0]`; thread 0 of every other device `ld_reduce`s
/// `acc[0]` with `order` and, having seen the flag, reads device 0's `res[0]` and keeps it in its own `res[0]`. With an
/// `Acquire` the read is ordered after the write; with a `Relaxed` it is a data race. Locals: 0 the flag, 1 the value.
pub(crate) fn acquire_handoff(order: MemoryOrder) -> KernelFn {
    use unpopped::ir::{BinaryOp, ScalarExpr};
    let cmp = |op, a, b| ScalarExpr::Binary(op, Box::new(a), Box::new(b));
    let when = |cond, then| Stmt::If {
        cond,
        then: Block(then),
        els: Block(vec![]),
    };
    let first_thread = || cmp(BinaryOp::CmpEq, gid(), c(0));
    let publish = vec![
        store(at(2, c(0)), c(7)),
        Stmt::AtomicRmw {
            op: AtomicOp::Add,
            place: multicast(1, c(0)),
            value: c(1),
            old: None,
            order: MemoryOrder::Release,
            scope: Scope::System,
        },
    ];
    let consume = vec![
        ld_reduce(0, 1, (AtomicOp::Add, order)),
        load(1, super::fixture::peer(2, c(0), c(0))),
        store(at(2, c(0)), local(1)),
    ];
    let body = vec![
        when(
            cmp(BinaryOp::CmpEq, rank(), c(0)),
            vec![when(first_thread(), publish)],
        ),
        when(
            cmp(BinaryOp::CmpGt, rank(), c(0)),
            vec![when(first_thread(), consume)],
        ),
    ];
    kernel_of(ElementKind::I32, "acquire_handoff", (2, body))
}

/// Every device adds its `inp[gid]` into `acc[gid]` of every device by `multimem.red`, then, with nothing between,
/// reduces the devices' `acc[gid]` by `ld_reduce` into its `res[gid]`: atomics do not race with each other, so the
/// launch runs (the value read depends on the schedule). Locals: 0 the input, 1 the reduced value.
pub(crate) fn red_then_ld_reduce(elem: ElementKind) -> KernelFn {
    let body = vec![
        load(0, at(0, gid())),
        red(AtomicOp::Add, local(0)),
        ld_reduce(1, 1, (AtomicOp::Add, MemoryOrder::Acquire)),
        store(at(2, gid()), local(1)),
    ];
    kernel_of(elem, "red_then_ld_reduce", (2, body))
}

/// A kernel whose one statement is `st`, over the buffers of `all_reduce` (for the checker's refusals).
pub(crate) fn one(elem: ElementKind, st: Stmt) -> KernelFn {
    kernel_of(elem, "multicast_one", (1, vec![st]))
}
