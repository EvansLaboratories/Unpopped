// SPDX-License-Identifier: MIT OR Apache-2.0
//! The kernels of `kernel_device_peer.rs` (ROADMAP U12 slice 2, `docs/options-u12-peer-multicast.md`): a launch spans
//! several devices, each with its own copy of every buffer, and `Place::Peer` reaches another device's copy. The
//! devices run one after the other (device 0 first), the blocks of each in order.

use unpopped::exec::{ExecId, LaunchShape, Scope};
use unpopped::ir::{BinaryOp, CastSpec, RoundMode, Saturation, ScalarExpr};
use unpopped::kernel::{
    AtomicOp, BarrierKind, Block, BufferRole, KernelFn, LocalDecl, MemoryOrder, Param, Place, Stmt,
};
use unpopped_vocab::ElementKind;

pub(crate) const I32: ElementKind = ElementKind::I32;
/// Devices of the launches below.
pub(crate) const D: i64 = 3;

pub(crate) fn c(v: i64) -> ScalarExpr {
    ScalarExpr::ConstInt(v, I32)
}

pub(crate) fn local(i: u32) -> ScalarExpr {
    ScalarExpr::Local(i)
}

fn to_i32(e: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Cast(
        Box::new(e),
        CastSpec {
            to: I32,
            round: RoundMode::NearestEven,
            sat: Saturation::Wrap,
        },
    )
}

pub(crate) fn add(a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Add(Box::new(a), Box::new(b))
}

fn mul(a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Mul(Box::new(a), Box::new(b))
}

fn cmp(op: BinaryOp, a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Binary(op, Box::new(a), Box::new(b))
}

/// `DeviceRank` as an `I32`.
pub(crate) fn rank() -> ScalarExpr {
    to_i32(ScalarExpr::Exec(ExecId::DeviceRank))
}

/// `DeviceCount` as an `I32`.
pub(crate) fn count() -> ScalarExpr {
    to_i32(ScalarExpr::Exec(ExecId::DeviceCount))
}

/// `blockIdx.x` as an `I32`.
pub(crate) fn block() -> ScalarExpr {
    to_i32(ScalarExpr::Exec(ExecId::BlockIdx(0)))
}

/// `threadIdx.x` as an `I32`.
pub(crate) fn thread() -> ScalarExpr {
    to_i32(ScalarExpr::Exec(ExecId::ThreadIdx(0)))
}

/// `blockIdx * 2 + threadIdx`: the element of this thread in a launch of blocks of 2 threads.
pub(crate) fn gid() -> ScalarExpr {
    add(mul(block(), c(2)), thread())
}

pub(crate) fn at(buffer: u32, i: ScalarExpr) -> Place {
    Place::Logical {
        buffer,
        coords: vec![i],
    }
}

/// Element `i` of buffer `buffer` on device `device`.
pub(crate) fn peer(buffer: u32, device: ScalarExpr, i: ScalarExpr) -> Place {
    Place::Peer {
        buffer,
        device,
        coords: vec![i],
    }
}

pub(crate) fn store(place: Place, value: ScalarExpr) -> Stmt {
    Stmt::Store { place, value }
}

pub(crate) fn load(local: u32, place: Place) -> Stmt {
    Stmt::Load { local, place }
}

pub(crate) fn barrier(scope: Scope) -> Stmt {
    Stmt::Barrier {
        scope,
        kind: BarrierKind::Full,
    }
}

fn when(cond: ScalarExpr, then: Vec<Stmt>, els: Vec<Stmt>) -> Stmt {
    Stmt::If {
        cond,
        then: Block(then),
        els: Block(els),
    }
}

/// A kernel over the given buffers (all `I32`; the first is read, the others read and written).
pub(crate) fn kernel(
    name: &str,
    shape: LaunchShape,
    buffers: &[&str],
    (locals, body): (usize, Vec<Stmt>),
) -> KernelFn {
    let params = buffers
        .iter()
        .enumerate()
        .map(|(i, n)| {
            let role = if i == 0 {
                BufferRole::Read
            } else {
                BufferRole::ReadWrite
            };
            Param::buffer(*n, I32, role, Some(1))
        })
        .collect();
    KernelFn::new(
        name,
        I32,
        params,
        (0..locals).map(|_| LocalDecl::scalar(I32)).collect(),
        Block(body),
    )
    .with_launch(shape)
}

/// 3 devices of 2 blocks of 2 threads, the 2 blocks of a device one cluster.
pub(crate) fn gather_launch() -> LaunchShape {
    LaunchShape::new([2, 1, 1], [2, 1, 1])
        .with_cluster([2, 1, 1])
        .with_devices(D as u32)
}

/// Every thread writes its `inp[gid]` into slot `rank * 4 + gid` of the `out` of every device (its own included); after
/// `barrier` (a barrier of that scope, if any) it adds the three slots of its own `out` for its element into
/// `sum[gid]`. With a `System` barrier every device's writes are ordered before the reads; with none, or with a `Device`
/// barrier (which orders one device only), a peer's write and the owner's read are unordered. Locals: 0 the input, 1 the
/// rank, 2 to 4 the slots.
pub(crate) fn all_gather(sync: Option<Scope>) -> KernelFn {
    let mut body = vec![
        load(0, at(0, gid())),
        Stmt::Let {
            local: 1,
            init: rank(),
        },
    ];
    for d in 0..D {
        body.push(store(
            peer(1, c(d), add(mul(local(1), c(4)), gid())),
            local(0),
        ));
    }
    body.extend(sync.map(barrier));
    for d in 0..D {
        body.push(load(2 + d as u32, at(1, add(c(d * 4), gid()))));
    }
    body.push(store(at(2, gid()), add(add(local(2), local(3)), local(4))));
    kernel(
        "all_gather",
        gather_launch(),
        &["inp", "out", "sum"],
        (5, body),
    )
}

/// 3 devices of one thread: a chain reduction. Device 0 sends its `inp[0]` to device 1's `out[0]` and releases device
/// 1's flag; device `r` acquires its own flag, adds its `inp[0]` to its `out[0]` and sends the partial to device `r + 1`;
/// the last device stores the total in its own `out[0]`. The orders and scopes are those of the release and acquire.
/// Locals: 0 the rank, 1 the running sum, 2 the flag's old value, 3 the partial received.
pub(crate) fn chain(
    (release, rel_scope): (MemoryOrder, Scope),
    (acquire, acq_scope): (MemoryOrder, Scope),
) -> KernelFn {
    let flag = |place: Place, order, scope| Stmt::AtomicRmw {
        op: AtomicOp::Add,
        place,
        value: c(0),
        old: Some(2),
        order,
        scope,
    };
    let receive = vec![
        flag(at(2, c(0)), acquire, acq_scope),
        load(3, at(1, c(0))),
        Stmt::Assign {
            local: 1,
            value: add(local(1), local(3)),
        },
    ];
    let next = add(local(0), c(1));
    let send = vec![
        store(peer(1, next.clone(), c(0)), local(1)),
        flag(peer(2, next, c(0)), release, rel_scope),
    ];
    let body = vec![
        Stmt::Let {
            local: 0,
            init: rank(),
        },
        load(1, at(0, c(0))),
        when(cmp(BinaryOp::CmpGt, local(0), c(0)), receive, vec![]),
        when(
            cmp(BinaryOp::CmpLt, local(0), c(D - 1)),
            send,
            vec![store(at(1, c(0)), local(1))],
        ),
    ];
    kernel(
        "chain",
        LaunchShape::new([1, 1, 1], [1, 1, 1]).with_devices(D as u32),
        &["inp", "out", "flag"],
        (4, body),
    )
}

/// Each device stores `rank * 10 + count` in its own `out[0]`: the execution ids tell the devices apart.
pub(crate) fn rank_and_count() -> KernelFn {
    kernel(
        "rank_and_count",
        LaunchShape::new([1, 1, 1], [1, 1, 1]).with_devices(D as u32),
        &["inp", "out"],
        (
            0,
            vec![store(at(1, c(0)), add(mul(rank(), c(10)), count()))],
        ),
    )
}

/// 2 devices of 2 blocks of one thread: each block stores `rank * 100 + blockIdx + 1` in its device's `out[blockIdx]`,
/// the device meets at a `Device` barrier, and each block reads the other block's slot into `res[blockIdx]`. A `Device`
/// barrier is per device: device 0 does not wait for device 1.
pub(crate) fn device_barrier() -> KernelFn {
    kernel(
        "device_barrier",
        LaunchShape::new([1, 1, 1], [2, 1, 1]).with_devices(2),
        &["inp", "out", "res"],
        (
            1,
            vec![
                store(at(1, block()), add(add(mul(rank(), c(100)), block()), c(1))),
                barrier(Scope::Device),
                load(0, at(1, add(c(1), mul(block(), c(-1))))),
                store(at(2, block()), local(0)),
            ],
        ),
    )
}

/// Every device reads `inp[0]` of the device `which` (an `I32` expression) into its own `out[0]`.
pub(crate) fn read_device(which: ScalarExpr) -> KernelFn {
    kernel(
        "read_device",
        LaunchShape::new([1, 1, 1], [1, 1, 1]).with_devices(2),
        &["inp", "out"],
        (
            1,
            vec![load(0, peer(0, which, c(0))), store(at(1, c(0)), local(0))],
        ),
    )
}
