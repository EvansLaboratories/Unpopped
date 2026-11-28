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
    kernel_with(
        compute,
        Numerics {
            round: RoundMode::TowardZero,
            ..Numerics::default()
        },
    )
}

/// The same kernel under any qualifier.
fn kernel_with(compute: ElementKind, n: Numerics) -> KernelFn {
    let sum = ScalarExpr::Add(
        Box::new(ScalarExpr::Const(1.0)),
        Box::new(ScalarExpr::Const(2.0)),
    );
    let qualified = ScalarExpr::Qualified(Box::new(sum), n);
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
fn half_kernels_are_a_named_check_error_not_an_oracle_panic() {
    for compute in [ElementKind::F16, ElementKind::Bf16] {
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

fn illegal(k: &KernelFn) -> Option<&'static str> {
    match check_kernel(k) {
        Err(KernelError::Type(TypeError::IllegalQualifier { why })) => Some(why),
        _ => None,
    }
}

/// ROADMAP U43: a directed rounding at f64 checks and runs (an exact integer reference, `numerics_f64`); `1.0 + 2.0` is exact, so the
/// value is the same in every mode, and a case that is NOT exact is in `tests/numerics_f64_directed.rs`.
#[test]
fn an_f64_directed_kernel_checks_and_runs_in_every_mode() {
    for round in [
        RoundMode::TowardZero,
        RoundMode::TowardNegInf,
        RoundMode::TowardPosInf,
        RoundMode::NearestEven,
    ] {
        let k = kernel_with(
            ElementKind::F64,
            Numerics {
                round,
                ..Numerics::default()
            },
        );
        check_kernel(&k).unwrap_or_else(|e| panic!("{round:?}: {e:?}"));
        let mut args = vec![KernelArg::Buffer(TypedBuffer::from_f64(&[1], &[0.0]))];
        run_launch(&k, &mut args).expect("runs");
        let KernelArg::Buffer(o) = &args[0] else {
            panic!()
        };
        assert_eq!(o.to_f64_vec(), vec![3.0], "{round:?}");
    }
}

/// `x + y` where the exact sum is `1 + 2^-60`: the three directions differ, the kernel carries the oracle's answer.
#[test]
fn an_f64_directed_kernel_rounds_in_the_stated_direction() {
    for (round, want) in [
        (RoundMode::TowardZero, 1.0),
        (RoundMode::TowardNegInf, 1.0),
        (
            RoundMode::TowardPosInf,
            f64::from_bits(1.0f64.to_bits() + 1),
        ),
    ] {
        let sum = ScalarExpr::Add(
            Box::new(ScalarExpr::Const(1.0)),
            Box::new(ScalarExpr::Const(2f64.powi(-60))),
        );
        let mut k = kernel_with(
            ElementKind::F64,
            Numerics {
                round,
                ..Numerics::default()
            },
        );
        let Stmt::Store { value, .. } = &mut k.body.0[0] else {
            panic!()
        };
        *value = ScalarExpr::Qualified(
            Box::new(sum),
            Numerics {
                round,
                ..Numerics::default()
            },
        );
        check_kernel(&k).expect("checks");
        let mut args = vec![KernelArg::Buffer(TypedBuffer::from_f64(&[1], &[0.0]))];
        run_launch(&k, &mut args).expect("runs");
        let KernelArg::Buffer(o) = &args[0] else {
            panic!()
        };
        assert_eq!(o.to_f64_vec(), vec![want], "{round:?}");
    }
}

#[test]
fn f64_ftz_and_approx_and_ties_away_are_named_refusals_beside_the_directed_control() {
    let ftz = Numerics {
        ftz: true,
        ..Numerics::default()
    };
    let approx = Numerics {
        approx: true,
        ..Numerics::default()
    };
    let away = Numerics {
        round: RoundMode::NearestAway,
        ..Numerics::default()
    };
    let why = illegal(&kernel_with(ElementKind::F64, ftz)).expect("ftz at f64 is refused");
    assert!(why.contains("ftz"), "{why}");
    let why = illegal(&kernel_with(ElementKind::F64, approx)).expect("approx at f64 is refused");
    assert!(why.contains("approx at f64"), "{why}");
    assert!(illegal(&kernel_with(ElementKind::F64, away)).is_some());
    // the control: the same kernel with a directed mode is not refused
    let ok = Numerics {
        round: RoundMode::TowardZero,
        ..Numerics::default()
    };
    assert!(illegal(&kernel_with(ElementKind::F64, ok)).is_none());
}
