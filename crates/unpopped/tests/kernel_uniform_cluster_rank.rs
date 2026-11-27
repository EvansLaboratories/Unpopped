// SPDX-License-Identifier: MIT OR Apache-2.0
//! `ExecId::ClusterCtaRank` in the barrier-uniformity rule (`kernel_uniform.rs`): the block's index in its cluster is the same for
//! every thread of a block, like `BlockIdx`, so a `Block` barrier under `if (cluster_ctarank == 0)` is reached by the whole block
//! and is accepted; it is divergent only where the kernel also holds a `Cluster` or `Device` barrier (whose threads span blocks of
//! the cluster). Before this rule `ClusterCtaRank` was read as divergent (conservative, the leaves lane's note on #428).

use unpopped::exec::{ExecId, LaunchShape, Scope};
use unpopped::ir::{BinaryOp, ScalarExpr};
use unpopped::kernel::{BarrierKind, Block, KernelError, KernelFn, Stmt, check_kernel};
use unpopped_vocab::ElementKind;

fn cond(id: ExecId) -> ScalarExpr {
    ScalarExpr::Binary(
        BinaryOp::CmpEq,
        Box::new(ScalarExpr::Exec(id)),
        Box::new(ScalarExpr::ConstInt(0, ElementKind::U32)),
    )
}

fn barrier(scope: Scope) -> Stmt {
    Stmt::Barrier {
        scope,
        kind: BarrierKind::Full,
    }
}

fn under(id: ExecId, inner: Stmt) -> Stmt {
    Stmt::If {
        cond: cond(id),
        then: Block(vec![inner]),
        els: Block(vec![]),
    }
}

fn kernel(body: Vec<Stmt>) -> KernelFn {
    KernelFn::new(
        "uniform_rank",
        ElementKind::F32,
        vec![],
        vec![],
        Block(body),
    )
    .with_launch(LaunchShape::new([32, 1, 1], [4, 1, 1]).with_cluster([2, 1, 1]))
}

#[test]
fn a_block_barrier_under_the_cluster_rank_is_accepted_and_the_controls_are_not() {
    // accepted: every thread of a block has the same cluster rank
    assert_eq!(
        check_kernel(&kernel(vec![under(
            ExecId::ClusterCtaRank,
            barrier(Scope::Block)
        )])),
        Ok(())
    );
    // the control: the same barrier under a thread-dependent condition is divergent
    assert_eq!(
        check_kernel(&kernel(vec![under(
            ExecId::ThreadIdx(0),
            barrier(Scope::Block)
        )])),
        Err(KernelError::BarrierDivergent)
    );
}

#[test]
fn with_a_cluster_barrier_in_the_kernel_the_cluster_rank_is_divergent() {
    // the blocks of a cluster differ in rank and all take part in a Cluster barrier: a barrier under the rank is then divergent
    let k = kernel(vec![
        barrier(Scope::Cluster),
        under(ExecId::ClusterCtaRank, barrier(Scope::Cluster)),
    ]);
    assert_eq!(check_kernel(&k), Err(KernelError::BarrierDivergent));
    // the rule is conservative, as for `BlockIdx`: a Cluster barrier anywhere in the kernel makes the rank divergent for EVERY
    // rank-guarded barrier, a Block barrier included (the accepted control is the kernel of the first test, which has none)
    let ok = kernel(vec![
        barrier(Scope::Cluster),
        under(ExecId::ClusterCtaRank, barrier(Scope::Block)),
    ]);
    assert_eq!(check_kernel(&ok), Err(KernelError::BarrierDivergent));
}
