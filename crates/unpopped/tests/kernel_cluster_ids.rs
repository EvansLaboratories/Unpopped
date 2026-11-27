// SPDX-License-Identifier: MIT OR Apache-2.0
//! `ExecId::{ClusterCtaId, ClusterDim, WarpgroupId}`: PTX `%cluster_ctaid`, `%cluster_nctaid`, and the index of a thread's
//! warpgroup (four consecutive subgroups, `Scope::Warpgroup`). The expected arrays are worked by hand.

use unpopped::exec::{ExecId, LaunchShape};
use unpopped::ir::{BinaryOp, CastSpec, RoundMode, Saturation, ScalarExpr};
use unpopped::kernel::{
    Block, BufferRole, KernelError, KernelFn, LocalDecl, Param, Place, Stmt, check_kernel,
};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_vocab::ElementKind;

const I32: ElementKind = ElementKind::I32;
const U32: ElementKind = ElementKind::U32;

fn exec(id: ExecId) -> ScalarExpr {
    ScalarExpr::Exec(id)
}

fn spec() -> CastSpec {
    CastSpec {
        to: I32,
        round: RoundMode::NearestEven,
        sat: Saturation::Wrap,
    }
}

/// Thread 0 of every block stores `id` at `slot` (as an I32), or every thread does when `all_threads`.
fn kernel(id: ExecId, slot: ScalarExpr, shape: LaunchShape, all_threads: bool) -> KernelFn {
    let store = Stmt::Store {
        place: Place::Logical {
            buffer: 0,
            coords: vec![slot],
        },
        value: ScalarExpr::Cast(Box::new(exec(id)), spec()),
    };
    let body = if all_threads {
        store
    } else {
        Stmt::If {
            cond: ScalarExpr::Binary(
                BinaryOp::CmpEq,
                Box::new(exec(ExecId::ThreadIdx(0))),
                Box::new(ScalarExpr::ConstInt(0, U32)),
            ),
            then: Block(vec![store]),
            els: Block(vec![]),
        }
    };
    KernelFn::new(
        "id",
        I32,
        vec![Param::buffer("out", I32, BufferRole::ReadWrite, Some(1))],
        vec![LocalDecl::scalar(I32)],
        Block(vec![body]),
    )
    .with_launch(shape)
}

fn run(k: &KernelFn, len: i64) -> Vec<i32> {
    assert_eq!(check_kernel(k), Ok(()), "well formed");
    let mut args = vec![KernelArg::Buffer(TypedBuffer::new(
        I32,
        vec![len],
        vec![1],
        vec![0xFF; 4 * len as usize],
    ))];
    run_launch(k, &mut args).unwrap();
    let KernelArg::Buffer(b) = &args[0] else {
        unreachable!()
    };
    b.bytes
        .chunks(4)
        .map(|w| i32::from_le_bytes(w.try_into().unwrap()))
        .collect()
}

/// `blockIdx.x + gridDim.x * blockIdx.y`
fn block_slot() -> ScalarExpr {
    ScalarExpr::Add(
        Box::new(exec(ExecId::BlockIdx(0))),
        Box::new(ScalarExpr::Mul(
            Box::new(exec(ExecId::GridDim(0))),
            Box::new(exec(ExecId::BlockIdx(1))),
        )),
    )
}

fn grid() -> LaunchShape {
    LaunchShape::new([1, 1, 1], [4, 2, 1]).with_cluster([2, 2, 1])
}

#[test]
fn the_cluster_ctaid_is_the_block_index_modulo_the_cluster_on_each_axis() {
    let id = |a| {
        run(
            &kernel(ExecId::ClusterCtaId(a), block_slot(), grid(), false),
            8,
        )
    };
    assert_eq!(id(0), [0, 1, 0, 1, 0, 1, 0, 1]);
    assert_eq!(id(1), [0, 0, 0, 0, 1, 1, 1, 1]);
    assert_eq!(id(2), [0; 8]);
}

#[test]
fn the_cluster_dim_is_the_cluster_extent_and_one_without_clusters() {
    let dim = |a, shape| {
        run(
            &kernel(ExecId::ClusterDim(a), block_slot(), shape, false),
            8,
        )
    };
    assert_eq!(dim(0, grid())[0], 2);
    assert_eq!(dim(1, grid())[0], 2);
    assert_eq!(dim(2, grid())[0], 1);
    let none = LaunchShape::new([1, 1, 1], [4, 2, 1]);
    assert_eq!(dim(0, none)[0], 1);
    // control: the same read with clusters is not 1
    assert_ne!(dim(0, none.with_cluster([4, 1, 1]))[0], 1);
}

#[test]
fn the_warpgroup_is_four_subgroups_of_threads() {
    let shape = LaunchShape::new([256, 1, 1], [1, 1, 1]).with_subgroup(32, 32);
    let k = kernel(ExecId::WarpgroupId, exec(ExecId::ThreadIdx(0)), shape, true);
    let out = run(&k, 256);
    assert!(out[..128].iter().all(|&g| g == 0) && out[128..].iter().all(|&g| g == 1));
    // a launch that names no subgroup size cannot say what a warpgroup is: refused, beside the accepted run above
    let bare = LaunchShape::new([256, 1, 1], [1, 1, 1]);
    let k = kernel(ExecId::WarpgroupId, exec(ExecId::ThreadIdx(0)), bare, true);
    assert_eq!(
        check_kernel(&k),
        Err(KernelError::SubgroupIdWithoutSubgroup)
    );
}

#[test]
fn an_axis_above_two_is_refused() {
    let ok = kernel(ExecId::ClusterCtaId(2), block_slot(), grid(), false);
    assert_eq!(check_kernel(&ok), Ok(()));
    for id in [ExecId::ClusterCtaId(3), ExecId::ClusterDim(3)] {
        let bad = kernel(id, block_slot(), grid(), false);
        assert_eq!(
            check_kernel(&bad),
            Err(KernelError::ExecAxisOutOfRange { axis: 3 })
        );
    }
}
