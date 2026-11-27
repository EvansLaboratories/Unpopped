// SPDX-License-Identifier: MIT OR Apache-2.0
//! A layout or an alignment on an allocation (ROADMAP T1) is declined by name by `unpopped-cpu-c`, which spells the identity layout
//! only (`docs/design-swizzled-allocation.md` section 6): never silently ignored. The same kernel without the field lowers.

use unpopped::backend::Backend;
use unpopped::exec::{AllocDecl, AllocLayout, ExecId, Extent, LaunchShape, MemorySpace, Scope};
use unpopped::ir::ScalarExpr;
use unpopped::kernel::{
    BarrierKind, Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt, check_kernel,
};
use unpopped_cpu_c::CpuC;
use unpopped_vocab::ElementKind::{self, U32};

fn kernel(alloc: AllocDecl) -> KernelFn {
    let tid = || ScalarExpr::Exec(ExecId::ThreadIdx(0));
    let at = |buffer| Place::Logical {
        buffer,
        coords: vec![tid()],
    };
    KernelFn::new(
        "layout",
        ElementKind::I32,
        vec![
            Param::buffer("in", U32, BufferRole::Read, Some(1)),
            Param::buffer("out", U32, BufferRole::Write, Some(1)),
        ],
        vec![LocalDecl::scalar(U32), LocalDecl::scalar(U32)],
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
        ]),
    )
    .with_launch(LaunchShape::new([64, 1, 1], [1, 1, 1]))
    .with_allocs(vec![alloc])
}

#[test]
fn a_layout_or_an_alignment_declines_by_name_and_the_plain_allocation_lowers() {
    let plain = AllocDecl::new("sh", MemorySpace::Shared, U32, Extent::Static(64));
    assert!(CpuC.lower_kernel(&kernel(plain.clone())).is_ok());
    let laid_out = plain.clone().with_layout(AllocLayout::new(
        "(8,8):(1,8)".parse::<AllocLayout>().unwrap().layout,
    ));
    for (what, decl) in [
        ("layout", laid_out),
        ("alignment", plain.with_align_bytes(128)),
    ] {
        let k = kernel(decl);
        assert_eq!(check_kernel(&k), Ok(()), "{what} is well formed");
        let err = CpuC
            .lower_kernel(&k)
            .expect_err("cpu_c spells the identity layout only");
        assert!(
            format!("{err:?}").contains("layout or an alignment"),
            "{what}: {err:?}"
        );
    }
}
