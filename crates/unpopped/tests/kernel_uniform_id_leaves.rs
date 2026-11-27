// SPDX-License-Identifier: MIT OR Apache-2.0
//! The barrier-uniformity rule (`kernel_uniform.rs`) for the ids of #435: `ClusterDim` is a launch constant (uniform, like
//! `GridDim`); `ClusterCtaId` is the block's position in its cluster (uniform within a block, like `ClusterCtaRank`);
//! `WarpgroupId` is the subgroup index divided by four, so it differs between the subgroups of a block (a `Block` barrier under it
//! is divergent) and not between the lanes of one subgroup (a `Subgroup` barrier under it is uniform). Each acceptance has its
//! refusal beside it, differing in the one id.

use unpopped::exec::{ExecId, LaunchShape, Scope};
use unpopped::ir::{BinaryOp, ReduceOp, ScalarExpr};
use unpopped::kernel::{
    BarrierKind, Block, CollectiveKind, KernelError, KernelFn, LocalDecl, Stmt, check_kernel,
};
use unpopped_vocab::ElementKind;

fn under(id: ExecId, inner: Stmt) -> Stmt {
    Stmt::If {
        cond: ScalarExpr::Binary(
            BinaryOp::CmpEq,
            Box::new(ScalarExpr::Exec(id)),
            Box::new(ScalarExpr::ConstInt(0, ElementKind::U32)),
        ),
        then: Block(vec![inner]),
        els: Block(vec![]),
    }
}

fn barrier(scope: Scope) -> Stmt {
    Stmt::Barrier {
        scope,
        kind: BarrierKind::Full,
    }
}

fn sum() -> Stmt {
    Stmt::Collective {
        kind: CollectiveKind::Reduce(ReduceOp::Sum),
        value: ScalarExpr::ConstInt(1, ElementKind::I32),
        result: 0,
        scope: Scope::Subgroup,
    }
}

fn kernel(body: Stmt) -> KernelFn {
    KernelFn::new(
        "ids",
        ElementKind::F32,
        vec![],
        vec![LocalDecl::scalar(ElementKind::I32)],
        Block(vec![body]),
    )
    .with_launch(
        LaunchShape::new([32, 1, 1], [256, 1, 1])
            .with_cluster([2, 1, 1])
            .with_subgroup(32, 32),
    )
}

/// `if (id == 0) { barrier(scope) }`.
fn check(id: ExecId, scope: Scope) -> Result<(), KernelError> {
    check_kernel(&kernel(under(id, barrier(scope))))
}

/// `if (id == 0) { subgroup sum }`: judged by what may differ between the lanes of one subgroup.
fn check_collective(id: ExecId) -> Result<(), KernelError> {
    check_kernel(&kernel(under(id, sum())))
}

#[test]
fn cluster_dim_is_uniform_and_the_thread_index_is_not() {
    assert_eq!(check(ExecId::ClusterDim(0), Scope::Block), Ok(()));
    assert_eq!(
        check(ExecId::ThreadIdx(0), Scope::Block),
        Err(KernelError::BarrierDivergent)
    );
}

#[test]
fn cluster_cta_id_is_block_uniform_for_every_axis() {
    for axis in 0..3 {
        assert_eq!(check(ExecId::ClusterCtaId(axis), Scope::Block), Ok(()));
    }
}

#[test]
fn warpgroup_id_differs_between_subgroups_but_not_between_lanes() {
    // a Block barrier under it: the warpgroups of the block differ, so it is divergent
    assert_eq!(
        check(ExecId::WarpgroupId, Scope::Block),
        Err(KernelError::BarrierDivergent)
    );
    // a Subgroup collective under it: every lane of a subgroup has the same warpgroup (control: the thread index is divergent)
    assert_eq!(check_collective(ExecId::WarpgroupId), Ok(()));
    assert_eq!(
        check_collective(ExecId::ThreadIdx(0)),
        Err(KernelError::BarrierDivergent)
    );
}
