// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The float-only binary operators have no integer form, in the checker as in the oracle** (U19 slice 1 finding, PM task
//! 2026-11-21). `check_kernel` used to accept `RemTrunc` (C `fmod`) on integers when the integer was the compute type, while
//! the oracle trapped on it, so a "valid" kernel nothing could run existed. The two now agree: the checker refuses. An
//! integer remainder is the total composition of `docs/int-div-shift-semantics.md`, which the oracle runs.

use unpopped::ir::{BinaryOp, ScalarExpr};
use unpopped::kernel::{
    Block, BufferRole, KernelError, KernelFn, LocalDecl, Param, Place, Stmt, check_kernel,
};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::{KernelArg, run_kernel};
use unpopped::types::TypeError;
use unpopped_vocab::ElementKind as E;

/// `out[0] = op(a, b)` over locals loaded from `x[0]`, `x[1]`, at element type `ty` (also the compute type).
fn kernel(ty: E, op: BinaryOp) -> KernelFn {
    let at = |i: i64| Place::Logical {
        buffer: 1,
        coords: vec![ScalarExpr::ConstInt(i, E::I32)],
    };
    KernelFn::new(
        "k",
        ty,
        vec![
            Param::buffer("out", ty, BufferRole::Write, Some(1)),
            Param::buffer("x", ty, BufferRole::Read, Some(1)),
        ],
        vec![LocalDecl::scalar(ty), LocalDecl::scalar(ty)],
        Block(vec![
            Stmt::Load {
                local: 0,
                place: at(0),
            },
            Stmt::Load {
                local: 1,
                place: at(1),
            },
            Stmt::Store {
                place: Place::Logical {
                    buffer: 0,
                    coords: vec![ScalarExpr::ConstInt(0, E::I32)],
                },
                value: ScalarExpr::Binary(
                    op,
                    Box::new(ScalarExpr::Local(0)),
                    Box::new(ScalarExpr::Local(1)),
                ),
            },
        ]),
    )
}

const FLOAT_ONLY: [BinaryOp; 8] = [
    BinaryOp::Pow,
    BinaryOp::Rem,
    BinaryOp::RemTrunc,
    BinaryOp::Atan2,
    BinaryOp::Copysign,
    BinaryOp::Nextafter,
    BinaryOp::FmaxIeee,
    BinaryOp::FminIeee,
];

#[test]
fn the_checker_refuses_every_float_only_operator_on_integers() {
    for ty in [E::I32, E::I64, E::U32, E::U64] {
        for op in FLOAT_ONLY {
            let r = check_kernel(&kernel(ty, op));
            assert!(
                matches!(r, Err(KernelError::Type(TypeError::IntegerOperand { .. }))),
                "{op:?} on {ty:?}: {r:?}"
            );
        }
    }
}

#[test]
fn the_same_operators_stay_legal_on_floats_and_the_integer_ops_on_integers() {
    // Controls: the refusal is the integer type, not the operator or the kernel's shape.
    for op in FLOAT_ONLY {
        assert_eq!(check_kernel(&kernel(E::F32, op)), Ok(()), "{op:?} on F32");
    }
    for op in [BinaryOp::BitAnd, BinaryOp::Min, BinaryOp::Shl] {
        assert_eq!(check_kernel(&kernel(E::I32, op)), Ok(()), "{op:?} on I32");
    }
}

#[test]
fn a_kernel_the_checker_accepts_the_oracle_runs() {
    let k = kernel(E::I32, BinaryOp::BitXor);
    let mut args = vec![
        KernelArg::Buffer(TypedBuffer::from_i32(&[1], &[0])),
        KernelArg::Buffer(TypedBuffer::from_i32(&[2], &[12, 10])),
    ];
    run_kernel(&k, &mut args).expect("runs");
    let KernelArg::Buffer(o) = &args[0] else {
        panic!()
    };
    assert_eq!(o.to_f64_vec(), vec![6.0]);
}
