// SPDX-License-Identifier: MIT OR Apache-2.0
//! The kernels of `kernel_cluster_peer.rs` (ROADMAP U12 slice 1, `docs/options-u12-peer-multicast.md`): a block reaches
//! the `Shared` memory of another block of its cluster through `Place::ClusterPeer`. All of them run the same launch: 4
//! blocks of 4 threads in clusters of 2 blocks (clusters `{0, 1}` and `{2, 3}`), each block with a `Shared` array `sh`
//! of 4 `I32` elements, over an input buffer `inp` and an output buffer `out` of 16 elements (`blockIdx * 4 + threadIdx`).

use unpopped::exec::{AllocDecl, ExecId, Extent, LaunchShape, MemorySpace, Scope};
use unpopped::ir::{BinaryOp, CastSpec, RoundMode, Saturation, ScalarExpr};
use unpopped::kernel::{
    AtomicOp, BarrierKind, Block, BufferRole, KernelFn, LocalDecl, MemoryOrder, Param, Place, Stmt,
};
use unpopped_vocab::ElementKind;

pub(crate) const I32: ElementKind = ElementKind::I32;
pub(crate) const U32: ElementKind = ElementKind::U32;
/// Threads per block, and elements of `sh`.
pub(crate) const T: i64 = 4;
/// Elements of the buffers: 4 blocks of `T` threads.
pub(crate) const N: usize = 16;

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

fn add(a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Add(Box::new(a), Box::new(b))
}

fn mul(a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Mul(Box::new(a), Box::new(b))
}

fn sub(a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Sub(Box::new(a), Box::new(b))
}

/// `blockIdx.x` as an `I32`.
pub(crate) fn block() -> ScalarExpr {
    to_i32(ScalarExpr::Exec(ExecId::BlockIdx(0)))
}

/// `threadIdx.x` as an `I32`.
pub(crate) fn thread() -> ScalarExpr {
    to_i32(ScalarExpr::Exec(ExecId::ThreadIdx(0)))
}

/// This block's rank in its cluster of 2: `blockIdx % 2`.
pub(crate) fn rank() -> ScalarExpr {
    sub(
        block(),
        mul(ScalarExpr::Div(Box::new(block()), Box::new(c(2))), c(2)),
    )
}

/// The other block of the cluster: `1 - rank`.
pub(crate) fn other() -> ScalarExpr {
    sub(c(1), rank())
}

/// The global element of this thread: `blockIdx * 4 + threadIdx`.
pub(crate) fn gid() -> ScalarExpr {
    add(mul(block(), c(T)), thread())
}

pub(crate) fn at(buffer: u32, i: ScalarExpr) -> Place {
    Place::Logical {
        buffer,
        coords: vec![i],
    }
}

pub(crate) fn sh(i: ScalarExpr) -> Place {
    Place::Alloc { alloc: 0, index: i }
}

/// `sh` of the block with cluster rank `rank`.
pub(crate) fn peer(rank: ScalarExpr, i: ScalarExpr) -> Place {
    Place::ClusterPeer {
        alloc: 0,
        block: rank,
        index: i,
    }
}

pub(crate) fn cluster_barrier() -> Stmt {
    Stmt::Barrier {
        scope: Scope::Cluster,
        kind: BarrierKind::Full,
    }
}

pub(crate) fn store(place: Place, value: ScalarExpr) -> Stmt {
    Stmt::Store { place, value }
}

pub(crate) fn load(local: u32, place: Place) -> Stmt {
    Stmt::Load { local, place }
}

pub(crate) fn launch() -> LaunchShape {
    LaunchShape::new([T as u32, 1, 1], [4, 1, 1]).with_cluster([2, 1, 1])
}

/// The kernel every fixture builds: buffers `inp` (read) and `out` (written), the `Shared` array `sh`, `locals` locals
/// (all `I32`) and `body`.
pub(crate) fn kernel(name: &str, shape: LaunchShape, locals: usize, body: Vec<Stmt>) -> KernelFn {
    KernelFn::new(
        name,
        I32,
        vec![
            Param::buffer("inp", I32, BufferRole::Read, Some(1)),
            Param::buffer("out", I32, BufferRole::ReadWrite, Some(1)),
        ],
        (0..locals).map(|_| LocalDecl::scalar(I32)).collect(),
        Block(body),
    )
    .with_allocs(vec![AllocDecl::new(
        "sh",
        MemorySpace::Shared,
        I32,
        Extent::Static(T as u32),
    )])
    .with_launch(shape)
}

/// Each thread publishes `inp[gid]` in its own `sh[threadIdx]`, the cluster meets at a barrier, and each thread reads
/// `sh[threadIdx]` of the other block of the cluster: `out[gid] = inp[gid] + inp[gid of the same thread in the peer]`.
/// Locals: 0 the input, 1 the peer's value.
pub(crate) fn exchange() -> KernelFn {
    kernel(
        "exchange",
        launch(),
        2,
        vec![
            load(0, at(0, gid())),
            store(sh(thread()), local(0)),
            cluster_barrier(),
            load(1, peer(other(), thread())),
            store(at(1, gid()), add(local(0), local(1))),
        ],
    )
}

/// `exchange`, but each thread reads `sh` of the rank `rank`.
pub(crate) fn read_rank(rank: ScalarExpr) -> KernelFn {
    kernel(
        "read_rank",
        launch(),
        2,
        vec![
            load(0, at(0, gid())),
            store(sh(thread()), local(0)),
            cluster_barrier(),
            load(1, peer(rank, thread())),
            store(at(1, gid()), local(1)),
        ],
    )
}

/// Each thread clears its own `sh[threadIdx]`, the cluster meets, each thread writes `inp[gid]` into the peer's
/// `sh[threadIdx]`, and (with `ordered`) the cluster meets again before each thread reads its own `sh[threadIdx]`: the
/// value the peer's thread of the same index wrote. Without the second barrier the peer's write and the owner's read
/// are unordered. Locals: 0 the input, 1 the value read.
pub(crate) fn push(ordered: bool) -> KernelFn {
    let mut body = vec![
        store(sh(thread()), c(0)),
        cluster_barrier(),
        load(0, at(0, gid())),
        store(peer(other(), thread()), local(0)),
    ];
    if ordered {
        body.push(cluster_barrier());
    }
    body.push(load(1, sh(thread())));
    body.push(store(at(1, gid()), local(1)));
    kernel("push", launch(), 2, body)
}

/// Every thread adds `rank + 1` to element 0 of the peer's `sh` (an atomic at `Cluster` scope), once the owner has
/// cleared it and the cluster has met at a barrier; after a second barrier thread 0 stores its own `sh[0]`, which is
/// the sum over the peer block's 4 threads, in `out[gid]`. Locals: 0 the value read.
pub(crate) fn count(order: MemoryOrder) -> KernelFn {
    let zero = Stmt::If {
        cond: ScalarExpr::Binary(BinaryOp::CmpEq, Box::new(thread()), Box::new(c(0))),
        then: Block(vec![store(sh(c(0)), c(0))]),
        els: Block(vec![]),
    };
    let publish = Stmt::If {
        cond: ScalarExpr::Binary(BinaryOp::CmpEq, Box::new(thread()), Box::new(c(0))),
        then: Block(vec![load(0, sh(c(0))), store(at(1, gid()), local(0))]),
        els: Block(vec![]),
    };
    kernel(
        "count",
        launch(),
        1,
        vec![
            zero,
            cluster_barrier(),
            Stmt::AtomicRmw {
                op: AtomicOp::Add,
                place: peer(other(), c(0)),
                value: add(rank(), c(1)),
                old: None,
                order,
                scope: Scope::Cluster,
            },
            cluster_barrier(),
            publish,
        ],
    )
}

/// Rank 0 writes `inp[gid]` into the peer's `sh[threadIdx]` and then releases through the global flag (an atomic
/// add of 0 at `Cluster` scope with order `release`); rank 1 acquires the flag (the same add with order `acquire`) and
/// then reads its own `sh[threadIdx]` into `out[gid]`. No barrier orders the two blocks. The flag of a cluster is
/// `out[8 * cluster]` (a rank-0 block stores nothing else to `out`). Locals: 0 the input, 1 the flag's old value, 2 the value read.
pub(crate) fn handoff(release: MemoryOrder, acquire: MemoryOrder) -> KernelFn {
    // one flag per cluster: a cluster-scope atomic does not reach another cluster's
    let flag = at(
        1,
        mul(ScalarExpr::Div(Box::new(block()), Box::new(c(2))), c(8)),
    );
    let producer = Stmt::If {
        cond: ScalarExpr::Binary(BinaryOp::CmpEq, Box::new(rank()), Box::new(c(0))),
        then: Block(vec![
            load(0, at(0, gid())),
            store(peer(c(1), thread()), local(0)),
            Stmt::AtomicRmw {
                op: AtomicOp::Add,
                place: flag.clone(),
                value: c(0),
                old: Some(1),
                order: release,
                scope: Scope::Cluster,
            },
        ]),
        els: Block(vec![
            Stmt::AtomicRmw {
                op: AtomicOp::Add,
                place: flag,
                value: c(0),
                old: Some(1),
                order: acquire,
                scope: Scope::Cluster,
            },
            load(2, sh(thread())),
            store(at(1, gid()), local(2)),
        ]),
    };
    kernel("handoff", launch(), 3, vec![producer])
}
