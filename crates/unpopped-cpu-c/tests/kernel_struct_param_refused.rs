// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The serial C refuses a struct parameter** (ROADMAP U23): it flattens nothing and spells no by-value struct, so a
//! `ParamKind::Struct` is declined with its name in the reason. The control is the same kernel with scalar parameters,
//! which lowers.

use unpopped::backend::Backend;
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, StructField};
use unpopped_cpu_c::CpuC;
use unpopped_vocab::ElementKind;

const F32: ElementKind = ElementKind::F32;

fn kernel(second: Param) -> KernelFn {
    KernelFn::new(
        "s",
        F32,
        vec![
            Param::buffer("out", F32, BufferRole::Write, Some(1)),
            second,
        ],
        vec![LocalDecl::scalar(F32), LocalDecl::scalar(F32)],
        Block(vec![]),
    )
}

#[test]
fn a_struct_parameter_is_refused_by_name_and_the_scalar_control_lowers() {
    let with = kernel(Param::structure(
        "cfg",
        vec![StructField::new("a", F32, 0), StructField::new("b", F32, 1)],
    ));
    let err = CpuC
        .lower_kernel(&with)
        .expect_err("a struct parameter is refused");
    assert!(
        format!("{err:?}").contains("struct parameter"),
        "the refusal names it: {err:?}"
    );
    let control = kernel(Param::scalar("a", F32, 0));
    assert!(CpuC.lower_kernel(&control).is_ok(), "the control lowers");
}
