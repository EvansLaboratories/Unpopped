// SPDX-License-Identifier: MIT OR Apache-2.0
//! **`check_kernel` and a qualified operation at a compute type the oracle cannot run** (found 2026-11-23 lifting `__dadd_rn`:
//! `Qualified` at F64 passed `check_kernel`, then `run_launch` panicked at `oracle/expr_eval.rs:173`, "oracle: Qualified at {:?}
//! (the plan gate rejects this)"; a kernel never goes through the plan gate). The type check now says it: f32/f32s operands only, a
//! named `TypeError::IllegalQualifier`. The f32 kernel is the control and still checks and runs.

use unpopped::exec::LaunchShape;
use unpopped::ir::{Numerics, RoundMode, ScalarExpr};
use unpopped::kernel::{
    Block, BufferRole, KernelError, KernelFn, Param, Place, Stmt, check_kernel,
};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch;
use unpopped::types::TypeError;
use unpopped_vocab::ElementKind;

/// `out[0] = Qualified(1.0 + 2.0)` with the rounding toward zero, computed at `compute`.
fn kernel(compute: ElementKind) -> KernelFn {
    let sum = ScalarExpr::Add(
        Box::new(ScalarExpr::Const(1.0)),
        Box::new(ScalarExpr::Const(2.0)),
    );
    let qualified = ScalarExpr::Qualified(
        Box::new(sum),
        Numerics {
            round: RoundMode::TowardZero,
            ..Numerics::default()
        },
    );
    KernelFn::new(
        "q",
        compute,
        vec![Param::buffer(
            "out",
            compute,
            BufferRole::ReadWrite,
            Some(1),
        )],
        vec![],
        Block(vec![Stmt::Store {
            place: Place::Logical {
                buffer: 0,
                coords: vec![ScalarExpr::ConstInt(0, ElementKind::I32)],
            },
            value: qualified,
        }]),
    )
    .with_launch(LaunchShape::new([1, 1, 1], [1, 1, 1]))
}

#[test]
fn the_f32_kernel_checks_and_runs() {
    let k = kernel(ElementKind::F32);
    check_kernel(&k).expect("an f32 qualified operation checks");
    let mut args = vec![KernelArg::Buffer(TypedBuffer::from_f32(&[1], &[0.0]))];
    run_launch(&k, &mut args).expect("runs");
    let KernelArg::Buffer(o) = &args[0] else {
        panic!()
    };
    assert_eq!(o.to_f64_vec(), vec![3.0]);
}

#[test]
fn f64_and_half_kernels_are_a_named_check_error_not_an_oracle_panic() {
    for compute in [ElementKind::F64, ElementKind::F16, ElementKind::Bf16] {
        let r = check_kernel(&kernel(compute));
        assert!(
            matches!(
                r,
                Err(KernelError::Type(TypeError::IllegalQualifier { .. }))
            ),
            "{compute:?}: {r:?}"
        );
    }
}
