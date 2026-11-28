// SPDX-License-Identifier: MIT OR Apache-2.0
//! ROADMAP U50: `Stmt::Assume` is a fact for an optimiser, undefined behaviour when false. The serial C ignores it (a correct kernel
//! cannot tell, and the oracle is what checks it), so a kernel with one lowers to the same source as without.

use unpopped::backend::Backend;
use unpopped::exec::{ExecId, LaunchShape};
use unpopped::ir::{BinaryOp, ScalarExpr};
use unpopped::kernel::{Block, BufferRole, KernelFn, Param, Place, Stmt, check_kernel};
use unpopped_cpu_c::CpuC;
use unpopped_vocab::ElementKind::{F32, U32};

fn kernel(with_assume: bool) -> KernelFn {
    let tid = || ScalarExpr::Exec(ExecId::ThreadIdx(0));
    let mut body = Vec::new();
    if with_assume {
        body.push(Stmt::Assume {
            cond: ScalarExpr::Binary(
                BinaryOp::CmpLt,
                Box::new(tid()),
                Box::new(ScalarExpr::ConstInt(4, U32)),
            ),
        });
    }
    body.push(Stmt::Store {
        place: Place::Logical {
            buffer: 0,
            coords: vec![tid()],
        },
        value: ScalarExpr::Const(1.0),
    });
    KernelFn::new(
        "assume",
        F32,
        vec![Param::buffer("out", F32, BufferRole::ReadWrite, Some(1))],
        vec![],
        Block(body),
    )
    .with_launch(LaunchShape::new([4, 1, 1], [1, 1, 1]))
}

#[test]
fn an_assume_kernel_lowers_to_the_same_c_as_the_kernel_without_it() {
    assert_eq!(check_kernel(&kernel(true)), Ok(()));
    let plain = CpuC
        .lower_kernel(&kernel(false))
        .expect("the control lowers");
    let with = CpuC.lower_kernel(&kernel(true)).expect("an Assume lowers");
    assert_eq!(format!("{with:?}"), format!("{plain:?}"));
}
