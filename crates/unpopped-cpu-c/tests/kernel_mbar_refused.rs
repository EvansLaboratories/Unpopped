// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The serial C refuses an `mbarrier`** (ROADMAP U8c): a wait blocks a thread until another thread arrives, and the serial C
//! runs the threads of a block one after another, so it cannot spell one. The refusal names the reason; the control is the
//! same kernel without the barrier statement, which lowers.

use unpopped::backend::Backend;
use unpopped::exec::{AllocDecl, Extent, LaunchShape, MemorySpace};
use unpopped::ir::ScalarExpr;
use unpopped::kernel::{Block, BufferRole, KernelFn, Param, Place, Stmt};
use unpopped_cpu_c::CpuC;
use unpopped_vocab::ElementKind;

fn kernel(body: Vec<Stmt>) -> KernelFn {
    let mut k = KernelFn::new(
        "mbar",
        ElementKind::U32,
        vec![Param::buffer(
            "out",
            ElementKind::U32,
            BufferRole::ReadWrite,
            Some(1),
        )],
        vec![],
        Block(body),
    )
    .with_launch(LaunchShape::new([2, 1, 1], [1, 1, 1]));
    k.allocs = vec![AllocDecl::new(
        "bar",
        MemorySpace::Shared,
        ElementKind::U64,
        Extent::Static(1),
    )];
    k
}

#[test]
fn an_mbarrier_is_refused_with_its_reason_and_the_control_lowers() {
    let bar = Place::Alloc {
        alloc: 0,
        index: ScalarExpr::ConstInt(0, ElementKind::U32),
    };
    let with = kernel(vec![Stmt::MbarInit {
        bar,
        count: ScalarExpr::ConstInt(1, ElementKind::U32),
    }]);
    let err = CpuC
        .lower_kernel(&with)
        .expect_err("an mbarrier is refused");
    assert!(
        format!("{err:?}").contains("mbarrier"),
        "the refusal names it: {err:?}"
    );
    assert!(
        CpuC.lower_kernel(&kernel(vec![])).is_ok(),
        "the control lowers"
    );
}
