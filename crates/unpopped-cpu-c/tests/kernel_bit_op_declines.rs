// SPDX-License-Identifier: MIT OR Apache-2.0
//! An integer bit operation (`UnaryOp::Popc`/`Clz`/`Brev`) is defined by the oracle, not yet spelled by cpu_c: it declines by
//! name, and the same kernel with an operation cpu_c does spell (the control) lowers.

use unpopped::backend::Backend;
use unpopped::exec::{ExecId, LaunchShape};
use unpopped::ir::{BinaryOp, ScalarExpr, UnaryOp};
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt, check_kernel};
use unpopped_cpu_c::CpuC;
use unpopped_vocab::ElementKind;

const U32: ElementKind = ElementKind::U32;

fn kernel(value: impl FnOnce(ScalarExpr) -> ScalarExpr) -> KernelFn {
    let at = |buffer| Place::Logical {
        buffer,
        coords: vec![ScalarExpr::Exec(ExecId::ThreadIdx(0))],
    };
    KernelFn::new(
        "bitop",
        ElementKind::I32,
        vec![
            Param::buffer("in", U32, BufferRole::Read, Some(1)),
            Param::buffer("out", U32, BufferRole::Write, Some(1)),
        ],
        vec![LocalDecl::scalar(U32)],
        Block(vec![
            Stmt::Load {
                local: 0,
                place: at(0),
            },
            Stmt::Store {
                place: at(1),
                value: value(ScalarExpr::Local(0)),
            },
        ]),
    )
    .with_launch(LaunchShape::new([4, 1, 1], [1, 1, 1]))
}

#[test]
fn bit_ops_decline_by_name_and_an_integer_op_it_spells_lowers() {
    for op in [UnaryOp::Popc, UnaryOp::Clz, UnaryOp::Brev] {
        let k = kernel(|x| ScalarExpr::Unary(op, Box::new(x)));
        assert_eq!(check_kernel(&k), Ok(()));
        let err = CpuC
            .lower_kernel(&k)
            .expect_err("a bit operation is not lowered yet");
        assert!(
            format!("{err:?}").contains("bit operation"),
            "{op:?}: {err:?}"
        );
    }
    // control: the same shape with an operation cpu_c spells (a bitwise `x & x`)
    let k = kernel(|x| ScalarExpr::Binary(BinaryOp::BitAnd, Box::new(x.clone()), Box::new(x)));
    assert!(CpuC.lower_kernel(&k).is_ok());
}
