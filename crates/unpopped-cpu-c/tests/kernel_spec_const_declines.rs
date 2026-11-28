// SPDX-License-Identifier: MIT OR Apache-2.0
//! A kernel with a specialization constant declines by name in cpu_c (it does not bake one into the C signature yet); the same
//! kernel without the constant lowers (the control).

use unpopped::backend::Backend;
use unpopped::ir::ScalarExpr;
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, SpecValue, Stmt};
use unpopped_cpu_c::CpuC;
use unpopped_vocab::ElementKind;

fn kernel() -> KernelFn {
    let i32t = ElementKind::I32;
    KernelFn::new(
        "doubles",
        i32t,
        vec![
            Param::buffer("out", i32t, BufferRole::Write, Some(1)),
            Param::scalar("n", i32t, 0),
        ],
        vec![LocalDecl::scalar(i32t)],
        Block(vec![Stmt::Store {
            place: Place::Logical {
                buffer: 0,
                coords: vec![ScalarExpr::ConstInt(0, i32t)],
            },
            value: ScalarExpr::Local(0),
        }]),
    )
}

#[test]
fn a_spec_const_declines_by_name_and_the_plain_kernel_lowers() {
    assert!(CpuC.lower_kernel(&kernel()).is_ok());
    let err = CpuC
        .lower_kernel(&kernel().with_spec_const(1, SpecValue::Int(3)))
        .expect_err("not baked yet");
    assert!(
        format!("{err:?}").contains("specialization constant"),
        "{err:?}"
    );
}
