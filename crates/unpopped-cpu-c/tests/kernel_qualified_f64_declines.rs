// SPDX-License-Identifier: MIT OR Apache-2.0
//! ROADMAP U43: `check_kernel` and the oracle take a directed rounding at f64, `unpopped-cpu-c` does not spell one (C99 reaches a
//! directed mode only through `fesetround`, global state a kernel must not touch): it declines the kernel by name, never lowers it to
//! the round-to-nearest operation. The f32 round-to-nearest qualifier is the control that still lowers.

use unpopped::backend::Backend;
use unpopped::exec::{ExecId, LaunchShape};
use unpopped::ir::{Numerics, RoundMode, ScalarExpr};
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt, check_kernel};
use unpopped_cpu_c::CpuC;
use unpopped_vocab::ElementKind::{self, F32, F64};

fn qualified_add(dt: ElementKind, round: RoundMode) -> KernelFn {
    let at = |buffer| Place::Logical {
        buffer,
        coords: vec![ScalarExpr::Exec(ExecId::ThreadIdx(0))],
    };
    let sum = ScalarExpr::Add(
        Box::new(ScalarExpr::Local(0)),
        Box::new(ScalarExpr::Local(0)),
    );
    KernelFn::new(
        "qadd",
        dt,
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
                value: ScalarExpr::Qualified(
                    Box::new(sum),
                    Numerics {
                        round,
                        ..Numerics::default()
                    },
                ),
            },
        ]),
    )
    .with_launch(LaunchShape::new([4, 1, 1], [1, 1, 1]))
}

#[test]
fn an_f64_directed_kernel_checks_and_cpu_c_declines_it_by_name_beside_the_f32_control() {
    let control = qualified_add(F32, RoundMode::NearestEven);
    assert_eq!(check_kernel(&control), Ok(()));
    assert!(CpuC.lower_kernel(&control).is_ok(), "the control lowers");
    for round in [
        RoundMode::TowardZero,
        RoundMode::TowardPosInf,
        RoundMode::NearestEven,
    ] {
        let k = qualified_add(F64, round);
        assert_eq!(
            check_kernel(&k),
            Ok(()),
            "{round:?}: check_kernel computes it now"
        );
        let err = CpuC
            .lower_kernel(&k)
            .expect_err("a qualified f64 operation is not spelled");
        let text = format!("{err:?}");
        assert!(
            text.contains("Qualified")
                && text.contains("only at f32/f32s compute")
                && text.contains("F64"),
            "{round:?}: {text}"
        );
    }
}
