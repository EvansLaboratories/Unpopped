// SPDX-License-Identifier: MIT OR Apache-2.0
//! **A struct passed by value to a call** (ROADMAP U23 slice 4): `CallArg::Struct` gives a callee's `ParamKind::Struct`
//! parameter one expression per field, in the callee's field order. Layout only: the callee reads the fields as scalars.
//!
//! Each behaviour has a program that shows it and a control that does not.

use unpopped::exec::{ExecId, LaunchShape, Scope};
use unpopped::ir::{BinaryOp, ScalarExpr};
use unpopped::kernel::{
    BarrierKind, Block, BufferRole, CallArg, KernelError, KernelFn, LocalDecl, Module, Param,
    Place, Stmt, StructField, check_module,
};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::run_module;
use unpopped_vocab::ElementKind;

const F32: ElementKind = ElementKind::F32;
const I32: ElementKind = ElementKind::I32;
const U32: ElementKind = ElementKind::U32;

fn f(v: f64) -> ScalarExpr {
    ScalarExpr::Const(v)
}

fn at0() -> Place {
    Place::Logical {
        buffer: 0,
        coords: vec![ScalarExpr::ConstInt(0, U32)],
    }
}

fn rw() -> Param {
    Param::buffer("x", F32, BufferRole::ReadWrite, Some(1))
}

fn cfg() -> Param {
    Param::structure(
        "cfg",
        vec![StructField::new("a", F32, 0), StructField::new("b", F32, 1)],
    )
}

/// `put(x, cfg { a, b })`: `x[0] = a * 10 + b`, so the field ORDER is visible in the answer.
fn put() -> KernelFn {
    let v = ScalarExpr::Add(
        Box::new(ScalarExpr::Mul(
            Box::new(ScalarExpr::Local(0)),
            Box::new(f(10.0)),
        )),
        Box::new(ScalarExpr::Local(1)),
    );
    KernelFn::new(
        "put",
        F32,
        vec![rw(), cfg()],
        vec![LocalDecl::scalar(F32), LocalDecl::scalar(F32)],
        Block(vec![Stmt::Store {
            place: at0(),
            value: v,
        }]),
    )
}

fn call(args: Vec<CallArg>) -> Stmt {
    Stmt::Call {
        callee: 1,
        args,
        result: None,
    }
}

fn main_with(args: Vec<CallArg>) -> Module {
    let main = KernelFn::new("main", F32, vec![rw()], vec![], Block(vec![call(args)]));
    Module::new(vec![main, put()])
}

fn pass(a: f64, b: f64) -> Vec<CallArg> {
    vec![CallArg::Buffer(0), CallArg::Struct(vec![f(a), f(b)])]
}

fn arg_mismatch(m: &Module) -> bool {
    matches!(
        check_module(m, 0),
        Err(KernelError::CallArgMismatch { arg: 1, .. })
    )
}

#[test]
fn a_struct_argument_binds_each_field_in_order() {
    let m = main_with(pass(2.0, 3.0));
    assert_eq!(check_module(&m, 0), Ok(()));
    let mut args = vec![KernelArg::Buffer(TypedBuffer::from_f32(&[1], &[0.0]))];
    run_module(&m, 0, &mut args).expect("runs");
    let KernelArg::Buffer(b) = &args[0] else {
        unreachable!("a buffer")
    };
    assert_eq!(f32::from_le_bytes(b.bytes[..4].try_into().unwrap()), 23.0);
}

#[test]
fn a_struct_argument_must_match_its_parameter() {
    // controls: the right shape passes
    assert_eq!(check_module(&main_with(pass(2.0, 3.0)), 0), Ok(()));
    // too few fields, a field of another type, a scalar for the struct
    assert!(arg_mismatch(&main_with(vec![
        CallArg::Buffer(0),
        CallArg::Struct(vec![f(2.0)]),
    ])));
    assert!(arg_mismatch(&main_with(vec![
        CallArg::Buffer(0),
        CallArg::Struct(vec![f(2.0), ScalarExpr::ConstInt(3, I32)]),
    ])));
    assert!(arg_mismatch(&main_with(vec![
        CallArg::Buffer(0),
        CallArg::Scalar(f(2.0)),
    ])));
}

#[test]
fn a_barrier_callee_needs_uniform_struct_fields() {
    // gate(x, cfg { k }): a barrier only when k > 0, so `k` must be uniform at the call
    let gate = KernelFn::new(
        "gate",
        F32,
        vec![
            rw(),
            Param::structure("cfg", vec![StructField::new("k", U32, 0)]),
        ],
        vec![LocalDecl::scalar(U32)],
        Block(vec![Stmt::If {
            cond: ScalarExpr::Binary(
                BinaryOp::CmpGt,
                Box::new(ScalarExpr::Local(0)),
                Box::new(ScalarExpr::ConstInt(0, U32)),
            ),
            then: Block(vec![Stmt::Barrier {
                scope: Scope::Block,
                kind: BarrierKind::Full,
            }]),
            els: Block(vec![]),
        }]),
    );
    let module = |k: ScalarExpr| {
        let main = KernelFn::new(
            "main",
            F32,
            vec![rw()],
            vec![],
            Block(vec![Stmt::Call {
                callee: 1,
                args: vec![CallArg::Buffer(0), CallArg::Struct(vec![k])],
                result: None,
            }]),
        )
        .with_launch(LaunchShape::new([4, 1, 1], [1, 1, 1]));
        Module::new(vec![main, gate.clone()])
    };
    assert_eq!(
        check_module(&module(ScalarExpr::ConstInt(1, U32)), 0),
        Ok(())
    );
    assert_eq!(
        check_module(&module(ScalarExpr::Exec(ExecId::ThreadIdx(0))), 0),
        Err(KernelError::BarrierDivergent)
    );
}
