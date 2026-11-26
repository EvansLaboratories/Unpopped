// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The collective guard after a `Return`** (ROADMAP U63, board 506): a `Subgroup`-scope collective is refused when a `Return`
//! under control that may differ between the lanes of its subgroup precedes it (`KernelError::CollectiveAfterReturn`), and
//! `threadIdx.y` and `.z` are lane-uniform when the launch states one subgroup size that divides the block's x extent (so
//! `if (row >= n) return;` with `row` built from `threadIdx.y` is a whole-subgroup exit, as in the gguf mmvq kernels).
//! Each refusal has its control beside it.

use unpopped::exec::{ExecId, LaunchShape, Scope};
use unpopped::ir::{BinaryOp, ReduceOp, ScalarExpr};
use unpopped::kernel::{
    Block, CollectiveKind, KernelError, KernelFn, LocalDecl, Stmt, check_kernel,
};
use unpopped_vocab::ElementKind;

const I32: ElementKind = ElementKind::I32;
const U32: ElementKind = ElementKind::U32;

fn ci(v: i64) -> ScalarExpr {
    ScalarExpr::ConstInt(v, I32)
}

fn thread(axis: u8) -> ScalarExpr {
    ScalarExpr::Exec(ExecId::ThreadIdx(axis))
}

/// `threadIdx.x & 1`: differs between the lanes of one subgroup.
fn odd_lane() -> ScalarExpr {
    ScalarExpr::Binary(
        BinaryOp::BitAnd,
        Box::new(thread(0)),
        Box::new(ScalarExpr::ConstInt(1, U32)),
    )
}

fn if_(cond: ScalarExpr, then: Vec<Stmt>) -> Stmt {
    Stmt::If {
        cond,
        then: Block(then),
        els: Block(vec![]),
    }
}

fn sum(scope: Scope) -> Stmt {
    Stmt::Collective {
        kind: CollectiveKind::Reduce(ReduceOp::Sum),
        value: ci(1),
        result: 0,
        scope,
    }
}

/// A kernel of `body` in a launch of `block` threads whose subgroup range is `(min, max)`.
fn kernel(block: [u32; 3], subgroup: (u32, u32), body: Vec<Stmt>) -> KernelFn {
    KernelFn::new(
        "return_guard",
        ElementKind::F32,
        vec![],
        vec![LocalDecl::scalar(I32)],
        Block(body),
    )
    .with_launch(LaunchShape::new(block, [1, 1, 1]).with_subgroup(subgroup.0, subgroup.1))
}

fn standard(body: Vec<Stmt>) -> KernelFn {
    kernel([32, 1, 1], (32, 32), body)
}

fn after_return(k: &KernelFn) {
    assert_eq!(check_kernel(k), Err(KernelError::CollectiveAfterReturn));
}

#[test]
fn a_lane_divergent_return_before_a_subgroup_collective_is_refused_and_the_control_is_accepted() {
    after_return(&standard(vec![
        if_(odd_lane(), vec![Stmt::Return]),
        sum(Scope::Subgroup),
    ]));
    // the control: no `Return`
    assert_eq!(check_kernel(&standard(vec![sum(Scope::Subgroup)])), Ok(()));
    // the control: the `Return` after the collective
    assert_eq!(
        check_kernel(&standard(vec![
            sum(Scope::Subgroup),
            if_(odd_lane(), vec![Stmt::Return]),
        ])),
        Ok(())
    );
    // the control: a `Return` under control no thread-dependent value reaches
    assert_eq!(
        check_kernel(&standard(vec![
            if_(ci(1), vec![Stmt::Return]),
            sum(Scope::Subgroup),
        ])),
        Ok(())
    );
}

#[test]
fn the_error_is_its_own_variant_and_not_the_divergent_control_one() {
    let k = standard(vec![
        if_(odd_lane(), vec![Stmt::Return]),
        sum(Scope::Subgroup),
    ]);
    assert_ne!(check_kernel(&k), Err(KernelError::BarrierDivergent));
    // a collective UNDER the divergent condition is still `BarrierDivergent`
    assert_eq!(
        check_kernel(&standard(vec![if_(odd_lane(), vec![sum(Scope::Subgroup)])])),
        Err(KernelError::BarrierDivergent)
    );
}

#[test]
fn a_return_on_threadidx_y_is_a_whole_subgroup_exit_when_the_subgroup_size_is_stated() {
    // `if (threadIdx.y != 0) return;` then a subgroup reduction, in a 32 x 4 block with the stated subgroup size 32: the shape of
    // the mmvq kernels' `if (row >= nrows) return;` with `row` built from `threadIdx.y`
    let body = vec![if_(thread(1), vec![Stmt::Return]), sum(Scope::Subgroup)];
    assert_eq!(check_kernel(&kernel([32, 4, 1], (32, 32), body.clone())), Ok(()));
    // the controls: a launch that states a RANGE of subgroup sizes, and a block whose x extent is not a multiple of the size
    // (a subgroup may then straddle rows), do not prove it
    after_return(&kernel([32, 4, 1], (16, 32), body.clone()));
    after_return(&kernel([48, 4, 1], (32, 32), body));
    // the same refinement for a collective under `if (threadIdx.y == 0)`
    let under = vec![if_(thread(1), vec![sum(Scope::Subgroup)])];
    assert_eq!(check_kernel(&kernel([32, 4, 1], (32, 32), under.clone())), Ok(()));
    assert_eq!(
        check_kernel(&kernel([32, 4, 1], (16, 32), under)),
        Err(KernelError::BarrierDivergent)
    );
}

#[test]
fn a_block_collective_after_a_lane_divergent_return_is_unchanged() {
    // out of this rule's scope (the oracle traps a stranded peer at run time): pinned so a later change is on purpose
    assert_eq!(
        check_kernel(&standard(vec![
            if_(odd_lane(), vec![Stmt::Return]),
            sum(Scope::Block),
        ])),
        Ok(())
    );
}
