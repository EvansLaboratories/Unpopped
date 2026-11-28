// SPDX-License-Identifier: MIT OR Apache-2.0
//! `ExecId::SubgroupCount` is read by the oracle; cpu_c has no subgroup model and declines it by name (the same as
//! `SubgroupSize`); a launch id it does spell (`BlockDim`, the control) lowers.

use unpopped::backend::Backend;
use unpopped::exec::{ExecId, LaunchShape};
use unpopped::ir::ScalarExpr;
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt};
use unpopped_cpu_c::CpuC;
use unpopped_vocab::ElementKind;

fn kernel(id: ExecId) -> KernelFn {
    let u = ElementKind::U32;
    KernelFn::new(
        "ids",
        u,
        vec![Param::buffer("out", u, BufferRole::Write, Some(1))],
        vec![LocalDecl::scalar(u)],
        Block(vec![
            Stmt::Let {
                local: 0,
                init: ScalarExpr::Exec(id),
            },
            Stmt::Store {
                place: Place::Logical {
                    buffer: 0,
                    coords: vec![ScalarExpr::Exec(ExecId::ThreadIdx(0))],
                },
                value: ScalarExpr::Local(0),
            },
        ]),
    )
    .with_launch(LaunchShape::new([4, 1, 1], [1, 1, 1]).with_subgroup(2, 2))
}

#[test]
fn subgroup_count_declines_by_name_and_block_dim_lowers() {
    assert!(CpuC.lower_kernel(&kernel(ExecId::BlockDim(0))).is_ok());
    let err = CpuC
        .lower_kernel(&kernel(ExecId::SubgroupCount))
        .expect_err("not spelled");
    assert!(format!("{err:?}").contains("SubgroupCount"), "{err:?}");
}
