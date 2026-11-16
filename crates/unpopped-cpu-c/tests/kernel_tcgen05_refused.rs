// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The serial C refuses `tcgen05` statements** (ROADMAP U7 S5): they work on tensor memory, which the serial C does not
//! spell. The refusal names the reason; the control is the same kernel without it, which lowers.

use unpopped::backend::Backend;
use unpopped::exec::{AllocDecl, Extent, LaunchShape, MemorySpace};
use unpopped::ir::ScalarExpr;
use unpopped::kernel::{Block, BufferRole, KernelFn, Param, Place, Stmt};
use unpopped_cpu_c::CpuC;
use unpopped_vocab::ElementKind;

fn kernel(body: Vec<Stmt>) -> KernelFn {
    let mut k = KernelFn::new(
        "tc",
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
    k.allocs = vec![AllocDecl::new(
        "bar",
        MemorySpace::Shared,
        ElementKind::U64,
        Extent::Static(1),
    )];
    k
}

#[test]
fn a_tcgen05_commit_is_refused_with_its_reason_and_the_control_lowers() {
    let with = kernel(vec![Stmt::Tcgen05Commit {
        bar: Place::Alloc {
            alloc: 0,
            index: ScalarExpr::ConstInt(0, ElementKind::U32),
        },
    }]);
    let err = CpuC.lower_kernel(&with).expect_err("a tcgen05 is refused");
    assert!(
        format!("{err:?}").contains("tcgen05"),
        "the refusal names it: {err:?}"
    );
    assert!(
        CpuC.lower_kernel(&kernel(vec![])).is_ok(),
        "the control lowers"
    );
}
