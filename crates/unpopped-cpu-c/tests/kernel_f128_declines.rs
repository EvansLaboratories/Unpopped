// SPDX-License-Identifier: MIT OR Apache-2.0
//! `unpopped-cpu-c` has no spelling for binary128 (ROADMAP E3 made `F128` a kernel dtype the oracle computes): it declines such a
//! kernel by name, never lowers it to `double` or `long double`. The same kernel over `F64` lowers.

use unpopped::backend::Backend;
use unpopped::exec::{ExecId, LaunchShape};
use unpopped::ir::ScalarExpr;
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt, check_kernel};
use unpopped_cpu_c::CpuC;
use unpopped_vocab::ElementKind::{self, F64, F128};

fn add(dt: ElementKind) -> KernelFn {
    let at = |buffer| Place::Logical {
        buffer,
        coords: vec![ScalarExpr::Exec(ExecId::ThreadIdx(0))],
    };
    KernelFn::new(
        "add",
        F64,
        vec![
            Param::buffer("a", dt, BufferRole::Read, Some(1)),
            Param::buffer("out", dt, BufferRole::Write, Some(1)),
        ],
        vec![LocalDecl::scalar(dt)],
        Block(vec![
            Stmt::Load {
                local: 0,
                place: at(0),
            },
            Stmt::Store {
                place: at(1),
                value: ScalarExpr::Add(
                    Box::new(ScalarExpr::Local(0)),
                    Box::new(ScalarExpr::Local(0)),
                ),
            },
        ]),
    )
    .with_launch(LaunchShape::new([4, 1, 1], [1, 1, 1]))
}

#[test]
fn an_f128_kernel_is_declined_by_name_and_the_f64_control_lowers() {
    assert!(CpuC.lower_kernel(&add(F64)).is_ok(), "the control lowers");
    let k = add(F128);
    assert_eq!(check_kernel(&k), Ok(()), "check_kernel computes F128 now");
    let err = CpuC
        .lower_kernel(&k)
        .expect_err("a binary128 buffer is not spelled");
    let text = format!("{err:?}").to_lowercase();
    assert!(text.contains("f128"), "{err:?}");
}
