// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The serial C refuses a named barrier** (ROADMAP U4 S6b-2): it waits for a subset of the threads of a block, and the
//! serialized thread loop runs the threads one after another. The refusal names the reason; the control is the same kernel
//! with a full block barrier, which lowers.

use unpopped::backend::Backend;
use unpopped::exec::{LaunchShape, Scope};
use unpopped::kernel::{BarrierKind, Block, BufferRole, KernelFn, Param, Stmt};
use unpopped_cpu_c::CpuC;
use unpopped_vocab::ElementKind;

fn kernel(barrier: Stmt) -> KernelFn {
    KernelFn::new(
        "nb",
        ElementKind::F32,
        vec![Param::buffer(
            "out",
            ElementKind::F32,
            BufferRole::ReadWrite,
            Some(1),
        )],
        vec![],
        Block(vec![barrier]),
    )
    .with_launch(LaunchShape::new([64, 1, 1], [1, 1, 1]).with_subgroup(32, 32))
}

#[test]
fn a_named_barrier_is_refused_with_its_reason_and_the_full_barrier_lowers() {
    let named = Stmt::Barrier {
        scope: Scope::Block,
        kind: BarrierKind::Named { id: 1, count: 64 },
    };
    let err = CpuC
        .lower_kernel(&kernel(named))
        .expect_err("a named barrier is refused");
    assert!(
        format!("{err:?}").contains("named barrier"),
        "the refusal names it: {err:?}"
    );
    assert!(
        CpuC.lower_kernel(&kernel(Stmt::barrier(Scope::Block)))
            .is_ok(),
        "the control lowers"
    );
}
