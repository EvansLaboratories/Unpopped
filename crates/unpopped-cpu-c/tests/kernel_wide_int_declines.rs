// SPDX-License-Identifier: MIT OR Apache-2.0
//! The wide integers `I128`, `U128`, `I256`, `U256` are computed by the oracle (ROADMAP E2) and not spelled by cpu_c: a kernel that
//! names one declines by name (a C `__int128` is not portable and has no 256-bit form), never panics, and the same kernel at an
//! integer type cpu_c does spell (the control) lowers. No carry-flag instruction chain is involved: nothing is lowered.

use unpopped::backend::Backend;
use unpopped::exec::{ExecId, LaunchShape};
use unpopped::ir::{BinaryOp, ScalarExpr};
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt, check_kernel};
use unpopped_cpu_c::CpuC;
use unpopped_vocab::ElementKind;

fn kernel(t: ElementKind) -> KernelFn {
    let at = |buffer| Place::Logical {
        buffer,
        coords: vec![ScalarExpr::Exec(ExecId::ThreadIdx(0))],
    };
    KernelFn::new(
        "wide",
        ElementKind::I32,
        vec![
            Param::buffer("in", t, BufferRole::Read, Some(1)),
            Param::buffer("out", t, BufferRole::Write, Some(1)),
        ],
        vec![LocalDecl::scalar(t)],
        Block(vec![
            Stmt::Load {
                local: 0,
                place: at(0),
            },
            Stmt::Store {
                place: at(1),
                value: ScalarExpr::Binary(
                    BinaryOp::BitAnd,
                    Box::new(ScalarExpr::Local(0)),
                    Box::new(ScalarExpr::Local(0)),
                ),
            },
        ]),
    )
    .with_launch(LaunchShape::new([4, 1, 1], [1, 1, 1]))
}

#[test]
fn a_wide_integer_kernel_declines_by_name_and_the_narrow_control_lowers() {
    for t in [
        ElementKind::I128,
        ElementKind::U128,
        ElementKind::I256,
        ElementKind::U256,
    ] {
        let k = kernel(t);
        assert_eq!(check_kernel(&k), Ok(()), "{t:?} is well formed");
        let err = CpuC
            .lower_kernel(&k)
            .expect_err("a wide integer is not spelled by cpu_c");
        assert!(
            format!("{err:?}").contains(&format!("{t:?}")),
            "{t:?}: {err:?}"
        );
    }
    assert!(CpuC.lower_kernel(&kernel(ElementKind::U32)).is_ok());
}
