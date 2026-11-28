// SPDX-License-Identifier: MIT OR Apache-2.0
//! A `dyn_layout` on an allocation (a layout with scalar-parameter leaves, `docs/design-dynamic-layout.md`) is declined by name by
//! `unpopped-cpu-c`, which spells the identity layout only: never silently ignored. The same kernel without the field lowers.

use unpopped::backend::Backend;
use unpopped::exec::{AllocDecl, ExecId, Extent, LaunchShape, MemorySpace, Scope};
use unpopped::ir::ScalarExpr;
use unpopped::kernel::{
    BarrierKind, Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt, check_kernel,
};
use unpopped_cpu_c::CpuC;
use unpopped_vocab::ElementKind::{self, I32, U32};
use unpopped_vocab::layout_algebra::DynLayout;

fn body() -> Block {
    let tid = || ScalarExpr::Exec(ExecId::ThreadIdx(0));
    let at = |buffer| Place::Logical {
        buffer,
        coords: vec![tid()],
    };
    Block(vec![
        Stmt::Load {
            local: 0,
            place: at(0),
        },
        Stmt::Store {
            place: Place::Alloc {
                alloc: 0,
                index: tid(),
            },
            value: ScalarExpr::Local(0),
        },
        Stmt::Barrier {
            scope: Scope::Block,
            kind: BarrierKind::Full,
        },
        Stmt::Load {
            local: 1,
            place: Place::Alloc {
                alloc: 0,
                index: tid(),
            },
        },
        Stmt::Store {
            place: at(1),
            value: ScalarExpr::Local(1),
        },
    ])
}

fn kernel(alloc: AllocDecl) -> KernelFn {
    KernelFn::new(
        "layout",
        ElementKind::I32,
        vec![
            Param::buffer("in", U32, BufferRole::Read, Some(1)),
            Param::buffer("out", U32, BufferRole::Write, Some(1)),
            Param::scalar("m", I32, 2),
            Param::scalar("n", I32, 3),
        ],
        vec![
            LocalDecl::scalar(U32),
            LocalDecl::scalar(U32),
            LocalDecl::scalar(I32),
            LocalDecl::scalar(I32),
        ],
        body(),
    )
    .with_launch(LaunchShape::new([64, 1, 1], [1, 1, 1]))
    .with_allocs(vec![alloc])
}

#[test]
fn a_dyn_layout_declines_by_name_and_the_plain_allocation_lowers() {
    let plain = AllocDecl::new("sh", MemorySpace::Shared, U32, Extent::Static(64));
    assert!(CpuC.lower_kernel(&kernel(plain.clone())).is_ok());
    let dynamic = plain.with_dyn_layout("(?2,?3):(?3,1)".parse::<DynLayout>().unwrap());
    let k = kernel(dynamic);
    assert_eq!(check_kernel(&k), Ok(()), "well formed");
    let err = CpuC
        .lower_kernel(&k)
        .expect_err("cpu_c spells the identity layout only");
    assert!(format!("{err:?}").contains("dyn_layout"), "{err:?}");
}
