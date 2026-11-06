// SPDX-License-Identifier: MIT OR Apache-2.0
//! **An `F32` kernel local holds an `f32`** (found writing ROADMAP U10 A1, reported to the PM 2026-11-06): the kernel oracle used to
//! keep the `f64` result of a `Let` or `Assign` in the local, so `exp(234)` stayed finite and `1 + 2^24 - 2^24` was `1`, where `float`
//! C gives `inf` and `0`. A local is now rounded to its own type when it is written, so a chain of statements rounds at every step as
//! the serial C does; an `F64` local is still held at `f64`.

use unpopped::exec::LaunchShape;
use unpopped::ir::{ScalarExpr, UnaryOp};
use unpopped::kernel::{
    Block, BufferRole, CallArg, KernelFn, LocalDecl, Module, Param, Place, Stmt,
};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::{run_launch, run_launch_module};
use unpopped_vocab::ElementKind;

const F32: ElementKind = ElementKind::F32;
const F64: ElementKind = ElementKind::F64;

fn local(i: u32) -> Box<ScalarExpr> {
    Box::new(ScalarExpr::Local(i))
}

fn out0() -> Place {
    Place::Logical {
        buffer: 0,
        coords: vec![ScalarExpr::ConstInt(0, ElementKind::U32)],
    }
}

/// Run a one-thread kernel of `compute` dtype with locals `decls`, `body` and one `F32`/`F64` output; the result.
fn run(compute: ElementKind, decls: Vec<LocalDecl>, body: Vec<Stmt>) -> f64 {
    let k = KernelFn::new(
        "f32_local",
        compute,
        vec![Param::buffer(
            "out",
            compute,
            BufferRole::ReadWrite,
            Some(1),
        )],
        decls,
        Block(body),
    )
    .with_launch(LaunchShape::new([1, 1, 1], [1, 1, 1]));
    let width = if compute == F64 { 8 } else { 4 };
    let mut args = [KernelArg::Buffer(TypedBuffer::new(
        compute,
        vec![1],
        vec![1],
        vec![0; width],
    ))];
    run_launch(&k, &mut args).expect("the oracle runs it");
    let KernelArg::Buffer(b) = &args[0] else {
        unreachable!("a buffer")
    };
    if compute == F64 {
        f64::from_le_bytes(b.bytes[..8].try_into().unwrap())
    } else {
        f64::from(f32::from_le_bytes(b.bytes[..4].try_into().unwrap()))
    }
}

fn c(v: f64) -> ScalarExpr {
    ScalarExpr::Const(v)
}

#[test]
fn an_overflowing_f32_local_is_infinite_so_inf_over_inf_is_nan() {
    // l = exp(234); r = l / l: float C gives inf / inf = NaN.
    let r = run(
        F32,
        vec![LocalDecl::scalar(F32), LocalDecl::scalar(F32)],
        vec![
            Stmt::Let {
                local: 0,
                init: ScalarExpr::Unary(UnaryOp::Exp, Box::new(c(234.0))),
            },
            Stmt::Let {
                local: 1,
                init: ScalarExpr::Div(local(0), local(0)),
            },
            Stmt::Store {
                place: out0(),
                value: ScalarExpr::Local(1),
            },
        ],
    );
    assert!(r.is_nan(), "got {r}");
}

#[test]
fn an_assigned_f32_local_loses_the_bits_an_f32_cannot_hold() {
    // b = 2^24; b = b + 1 (2^24 + 1 is not an f32: stays 2^24); c = b - 2^24 = 0. Unrounded, c would be 1.
    let r = run(
        F32,
        vec![LocalDecl::scalar(F32), LocalDecl::scalar(F32)],
        vec![
            Stmt::Let {
                local: 0,
                init: c(16_777_216.0),
            },
            Stmt::Assign {
                local: 0,
                value: ScalarExpr::Add(local(0), Box::new(c(1.0))),
            },
            Stmt::Let {
                local: 1,
                init: ScalarExpr::Sub(local(0), Box::new(c(16_777_216.0))),
            },
            Stmt::Store {
                place: out0(),
                value: ScalarExpr::Local(1),
            },
        ],
    );
    assert_eq!(r, 0.0);
}

#[test]
fn an_f64_local_is_not_rounded_to_f32() {
    // the control: rounding follows the local's own type. In an F64 kernel, d = 2^24 + 1 survives and d - 2^24 is 1 (an f32 would give 0).
    let r = run(
        F64,
        vec![LocalDecl::scalar(F64), LocalDecl::scalar(F64)],
        vec![
            Stmt::Let {
                local: 0,
                init: ScalarExpr::Add(Box::new(c(16_777_216.0)), Box::new(c(1.0))),
            },
            Stmt::Let {
                local: 1,
                init: ScalarExpr::Sub(local(0), Box::new(c(16_777_216.0))),
            },
            Stmt::Store {
                place: out0(),
                value: ScalarExpr::Local(1),
            },
        ],
    );
    assert_eq!(r, 1.0, "the F64 local kept 2^24 + 1");
}

#[test]
fn an_f32_call_argument_is_held_in_the_parameters_type() {
    // f(x) = x - 2^24, called with 2^24 + 1: the argument is an f32 in the callee, so 2^24 and the result is 0 (unrounded it was 1).
    let callee = KernelFn::new(
        "f",
        F32,
        vec![Param::scalar("x", F32, 0)],
        vec![LocalDecl::scalar(F32), LocalDecl::scalar(F32)],
        Block(vec![Stmt::Let {
            local: 1,
            init: ScalarExpr::Sub(local(0), Box::new(c(16_777_216.0))),
        }]),
    )
    .with_result(1);
    let entry = KernelFn::new(
        "entry",
        F32,
        vec![Param::buffer("out", F32, BufferRole::ReadWrite, Some(1))],
        vec![LocalDecl::scalar(F32)],
        Block(vec![
            Stmt::Call {
                callee: 1,
                args: vec![CallArg::Scalar(ScalarExpr::Add(
                    Box::new(c(16_777_216.0)),
                    Box::new(c(1.0)),
                ))],
                result: Some(0),
            },
            Stmt::Store {
                place: out0(),
                value: ScalarExpr::Local(0),
            },
        ]),
    )
    .with_launch(LaunchShape::new([1, 1, 1], [1, 1, 1]));
    let mut args = [KernelArg::Buffer(TypedBuffer::new(
        F32,
        vec![1],
        vec![1],
        vec![0; 4],
    ))];
    run_launch_module(&Module::new(vec![entry, callee]), 0, &mut args).expect("the module runs");
    let KernelArg::Buffer(b) = &args[0] else {
        unreachable!("a buffer")
    };
    assert_eq!(f32::from_le_bytes(b.bytes[..4].try_into().unwrap()), 0.0);
}
