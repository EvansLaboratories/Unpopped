// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The serial C refuses a `CopyBulk`** (ROADMAP U8c slice 2): it completes on an `mbarrier`, which the serial C cannot
//! block a thread on. The control is the same kernel without the copy, which lowers.

use unpopped::backend::Backend;
use unpopped::exec::{AllocDecl, Extent, LaunchShape, MemorySpace};
use unpopped::ir::ScalarExpr;
use unpopped::kernel::{Block, BufferRole, KernelFn, Param, Place, Stmt};
use unpopped_cpu_c::CpuC;
use unpopped_vocab::ElementKind;

const U32: ElementKind = ElementKind::U32;

fn u(v: i64) -> ScalarExpr {
    ScalarExpr::ConstInt(v, U32)
}

fn kernel(body: Vec<Stmt>) -> KernelFn {
    let mut k = KernelFn::new(
        "bulk",
        U32,
        vec![Param::buffer("in", U32, BufferRole::Read, Some(1))],
        vec![],
        Block(body),
    )
    .with_launch(LaunchShape::new([2, 1, 1], [1, 1, 1]));
    k.allocs = vec![
        AllocDecl::new(
            "bar",
            MemorySpace::Shared,
            ElementKind::U64,
            Extent::Static(1),
        ),
        AllocDecl::new("tile", MemorySpace::Shared, U32, Extent::Static(4)),
    ];
    k
}

#[test]
fn a_bulk_copy_is_refused_with_its_reason_and_the_control_lowers() {
    let with = kernel(vec![Stmt::CopyBulk {
        dst: Place::Alloc {
            alloc: 1,
            index: u(0),
        },
        src: Place::Logical {
            buffer: 0,
            coords: vec![u(0)],
        },
        elems: 4,
        bar: Place::Alloc {
            alloc: 0,
            index: u(0),
        },
    }]);
    let err = CpuC
        .lower_kernel(&with)
        .expect_err("a bulk copy is refused");
    assert!(
        format!("{err:?}").contains("bulk copy"),
        "the refusal names it: {err:?}"
    );
    assert!(
        CpuC.lower_kernel(&kernel(vec![])).is_ok(),
        "the control lowers"
    );
}
