// SPDX-License-Identifier: MIT OR Apache-2.0
//! `ExecId::ClusterCtaRank`: the index of the block in its cluster, `x` fastest (PTX `%cluster_ctarank`, PTX ISA
//! "Special Registers"), `0` where the launch has no clusters. The expected arrays are worked by hand.

use unpopped::exec::{ExecId, LaunchShape};
use unpopped::ir::{BinaryOp, CastSpec, RoundMode, Saturation, ScalarExpr};
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt, check_kernel};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_vocab::ElementKind;

const I32: ElementKind = ElementKind::I32;
const U32: ElementKind = ElementKind::U32;

fn exec(id: ExecId) -> ScalarExpr {
    ScalarExpr::Exec(id)
}

fn bin(op: BinaryOp, a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Binary(op, Box::new(a), Box::new(b))
}

/// Every block's first thread stores its cluster rank at `blockIdx.x + gridDim.x * blockIdx.y`.
fn rank_kernel(shape: LaunchShape) -> KernelFn {
    let slot = ScalarExpr::Add(
        Box::new(exec(ExecId::BlockIdx(0))),
        Box::new(ScalarExpr::Mul(
            Box::new(exec(ExecId::GridDim(0))),
            Box::new(exec(ExecId::BlockIdx(1))),
        )),
    );
    let spec = CastSpec {
        to: I32,
        round: RoundMode::NearestEven,
        sat: Saturation::Wrap,
    };
    let store = Stmt::Store {
        place: Place::Logical {
            buffer: 0,
            coords: vec![slot],
        },
        value: ScalarExpr::Cast(Box::new(exec(ExecId::ClusterCtaRank)), spec),
    };
    let first = Stmt::If {
        cond: bin(
            BinaryOp::CmpEq,
            exec(ExecId::ThreadIdx(0)),
            ScalarExpr::ConstInt(0, U32),
        ),
        then: Block(vec![store]),
        els: Block(vec![]),
    };
    KernelFn::new(
        "rank",
        I32,
        vec![Param::buffer("out", I32, BufferRole::ReadWrite, Some(1))],
        vec![LocalDecl::scalar(I32)],
        Block(vec![first]),
    )
    .with_launch(shape)
}

fn ranks(shape: LaunchShape) -> Vec<i32> {
    let k = rank_kernel(shape);
    assert_eq!(check_kernel(&k), Ok(()), "well formed");
    let mut args = vec![KernelArg::Buffer(TypedBuffer::new(
        I32,
        vec![8],
        vec![1],
        vec![0xFF; 32],
    ))];
    run_launch(&k, &mut args).unwrap();
    let KernelArg::Buffer(b) = &args[0] else {
        unreachable!()
    };
    b.bytes
        .chunks(4)
        .map(|w| i32::from_le_bytes(w.try_into().unwrap()))
        .collect()
}

#[test]
fn the_rank_counts_x_fastest_inside_each_cluster() {
    let g = |grid: [u32; 3]| LaunchShape::new([1, 1, 1], grid);
    assert_eq!(
        ranks(g([4, 1, 1]).with_cluster([2, 1, 1]))[..4],
        [0, 1, 0, 1]
    );
    assert_eq!(
        ranks(g([4, 1, 1]).with_cluster([4, 1, 1]))[..4],
        [0, 1, 2, 3]
    );
    // 2D: rank = x % 2 + 2 * (y % 2), x fastest
    assert_eq!(
        ranks(g([4, 2, 1]).with_cluster([2, 2, 1])),
        [0, 1, 0, 1, 2, 3, 2, 3]
    );
}

#[test]
fn without_clusters_every_block_is_rank_zero() {
    let g = LaunchShape::new([1, 1, 1], [4, 1, 1]);
    assert_eq!(ranks(g)[..4], [0, 0, 0, 0]);
    // control: the same grid with clusters is not all zero
    assert_ne!(ranks(g.with_cluster([2, 1, 1]))[..4], [0, 0, 0, 0]);
}
