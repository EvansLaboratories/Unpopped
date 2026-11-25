// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The serial C refuses a `Wgmma`** (ROADMAP U7 S4): a warpgroup-wide asynchronous operation with commit and wait groups,
//! which the serial C does not spell. The refusal names the reason; the control is the same kernel without it, which lowers.

use unpopped::backend::Backend;
use unpopped::exec::{AllocDecl, Extent, LaunchShape, MemorySpace};
use unpopped::ir::ScalarExpr;
use unpopped::kernel::{Block, BufferRole, KernelFn, OperandLayout, Param, Place, Stmt};
use unpopped_cpu_c::CpuC;
use unpopped_vocab::ElementKind;

fn kernel(body: Vec<Stmt>) -> KernelFn {
    let mut k = KernelFn::new(
        "wg",
        ElementKind::F32,
        vec![Param::buffer(
            "out",
            ElementKind::F32,
            BufferRole::ReadWrite,
            Some(1),
        )],
        vec![],
        Block(body),
    )
    .with_launch(LaunchShape::new([128, 1, 1], [1, 1, 1]).with_subgroup(32, 32));
    k.allocs = vec![
        AllocDecl::new(
            "a",
            MemorySpace::Shared,
            ElementKind::Bf16,
            Extent::Static(1024),
        ),
        AllocDecl::new(
            "b",
            MemorySpace::Shared,
            ElementKind::Bf16,
            Extent::Static(128),
        ),
        AllocDecl::new(
            "d",
            MemorySpace::Private,
            ElementKind::F32,
            Extent::Static(4),
        ),
    ];
    k
}

#[test]
fn a_wgmma_is_refused_with_its_reason_and_the_control_lowers() {
    let tile = |alloc| Place::Alloc {
        alloc,
        index: ScalarExpr::ConstInt(0, ElementKind::U32),
    };
    let with = kernel(vec![Stmt::Wgmma {
        n: 8,
        ab: ElementKind::Bf16,
        acc: ElementKind::F32,
        a: tile(0),
        b: tile(1),
        d: 2,
        scale_d: ScalarExpr::ConstInt(0, ElementKind::U32),
        a_layout: OperandLayout::RowMajor,
        b_layout: OperandLayout::RowMajor,
    }]);
    let err = CpuC.lower_kernel(&with).expect_err("a wgmma is refused");
    assert!(
        format!("{err:?}").contains("wgmma"),
        "the refusal names it: {err:?}"
    );
    assert!(
        CpuC.lower_kernel(&kernel(vec![])).is_ok(),
        "the control lowers"
    );
}

#[test]
fn a_proxy_fence_lowers_as_nothing_because_serial_c_has_one_proxy() {
    // the serial C has no async proxy to order against, so `fence.proxy.async` has nothing to do (the sibling test shows
    // a Wgmma is still refused, so a lowerable kernel is not a symptom of the fence)
    let fenced = kernel(vec![Stmt::ProxyFence {
        space: MemorySpace::Shared,
    }]);
    assert!(
        CpuC.lower_kernel(&fenced).is_ok(),
        "a lone ProxyFence lowers"
    );
}
