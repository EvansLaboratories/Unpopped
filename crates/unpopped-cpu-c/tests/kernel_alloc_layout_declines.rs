// SPDX-License-Identifier: MIT OR Apache-2.0
//! A static layout on an allocation (ROADMAP T1) is spelled by `unpopped-cpu-c` (`kernel_alloc_layout_emit_agrees.rs`); an alignment on
//! its own, and a layout in a kernel whose body would bypass it (a Device barrier), are declined by name, never silently ignored.

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
fn a_layout_lowers_an_alignment_alone_declines_by_name_and_a_bypassed_layout_declines() {
    let plain = AllocDecl::new("sh", MemorySpace::Shared, U32, Extent::Static(64));
    assert!(CpuC.lower_kernel(&kernel(plain.clone())).is_ok());
    // since the static layout is spelled (kernel_alloc_layout_emit_agrees.rs) a layout lowers; an alignment on its own is still declined
    let laid_out = plain.clone().with_layout(AllocLayout::new(
        "(8,8):(1,8)".parse::<AllocLayout>().unwrap().layout,
    ));
    CpuC.lower_kernel(&kernel(laid_out.clone()))
        .expect("a static layout is spelled");
    let k = kernel(plain.with_align_bytes(128));
    assert_eq!(check_kernel(&k), Ok(()), "well formed");
    let err = CpuC.lower_kernel(&k).expect_err("an alignment alone");
    assert!(format!("{err:?}").contains("alignment"), "{err:?}");
    // a Device barrier makes the body multi-phase, which this emitter does not give the layout: declined by name
    let mut phased = kernel(laid_out);
    phased.body.0.insert(1, Stmt::barrier(Scope::Device));
    assert_eq!(check_kernel(&phased), Ok(()), "well formed");
    let err = CpuC.lower_kernel(&phased).expect_err("declined");
    assert!(format!("{err:?}").contains("layout"), "{err:?}");
}
