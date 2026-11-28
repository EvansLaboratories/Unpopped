// SPDX-License-Identifier: MIT OR Apache-2.0
//! ROADMAP U50 (PM ruling Q10 A): `Stmt::Assume { cond }` is a value fact (`tl.assume`, `multiple_of`): undefined behaviour on a device
//! when it is false, a typed `KernelTrap::AssumptionViolated` in the oracle, so a fact a front end lowers is checked on every oracle
//! run. It changes no value and orders nothing: the same kernel without it computes the same buffer. Each false fact here has its
//! true control beside it.

use unpopped::exec::{ExecId, LaunchShape};
use unpopped::ir::{BinaryOp, ScalarExpr};
use unpopped::kernel::{Block, BufferRole, KernelFn, Param, Place, Stmt, check_kernel};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::{KernelArg, KernelTrap};
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_vocab::ElementKind::{F32, U32};

const THREADS: u32 = 4;

fn tid() -> ScalarExpr {
    ScalarExpr::Exec(ExecId::ThreadIdx(0))
}

fn u(v: i64) -> ScalarExpr {
    ScalarExpr::ConstInt(v, U32)
}

fn bin(op: BinaryOp, a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Binary(op, Box::new(a), Box::new(b))
}

/// `out[tid] = 1.0` after the statements `pre`.
fn kernel(pre: Vec<Stmt>) -> KernelFn {
    let mut body = pre;
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
    .with_launch(LaunchShape::new([THREADS, 1, 1], [1, 1, 1]))
}

fn assume(cond: ScalarExpr) -> Stmt {
    Stmt::Assume { cond }
}

fn run(k: &KernelFn) -> Result<Vec<f64>, KernelTrap> {
    check_kernel(k).expect("checks");
    let mut args = vec![KernelArg::Buffer(TypedBuffer::from_f32(
        &[i64::from(THREADS)],
        &[0.0; THREADS as usize],
    ))];
    run_launch(k, &mut args)?;
    let KernelArg::Buffer(o) = &args[0] else {
        panic!("buffer")
    };
    Ok(o.to_f64_vec())
}

#[test]
fn a_true_fact_changes_nothing() {
    let plain = run(&kernel(vec![])).expect("runs");
    assert_eq!(plain, vec![1.0; 4]);
    let with = run(&kernel(vec![assume(bin(
        BinaryOp::CmpLt,
        tid(),
        u(i64::from(THREADS)),
    ))]));
    assert_eq!(with, Ok(plain));
}

#[test]
fn a_false_fact_in_one_thread_is_assumption_violated() {
    // tid < 3 is false for thread 3 only
    let r = run(&kernel(vec![assume(bin(BinaryOp::CmpLt, tid(), u(3)))]));
    assert_eq!(r, Err(KernelTrap::AssumptionViolated));
}

#[test]
fn multiple_of_is_an_assume_over_a_mask() {
    // x = tid * 2 is a multiple of 2 for every thread (the fact `multiple_of(x, 2)`: x & 1 == 0) ...
    let fact = |x: ScalarExpr| assume(bin(BinaryOp::CmpEq, bin(BinaryOp::BitAnd, x, u(1)), u(0)));
    let ok = run(&kernel(vec![fact(ScalarExpr::Mul(
        Box::new(tid()),
        Box::new(u(2)),
    ))]));
    assert_eq!(ok, Ok(vec![1.0; 4]));
    // ... and the same fact about x = tid is false for thread 1
    let bad = run(&kernel(vec![fact(tid())]));
    assert_eq!(bad, Err(KernelTrap::AssumptionViolated));
}

#[test]
fn only_a_thread_that_reaches_the_fact_states_it() {
    let guarded = |who: i64| {
        kernel(vec![Stmt::If {
            cond: bin(BinaryOp::CmpEq, tid(), u(who)),
            then: Block(vec![assume(bin(BinaryOp::CmpNe, tid(), u(0)))]),
            els: Block(vec![]),
        }])
    };
    // thread 0 reaches `tid != 0` and it is false; thread 1 reaches it and it is true
    assert_eq!(run(&guarded(0)), Err(KernelTrap::AssumptionViolated));
    assert_eq!(run(&guarded(1)), Ok(vec![1.0; 4]));
}

#[test]
fn a_float_condition_is_taken_as_an_if_takes_it() {
    // NaN is true, +1.0 is true, -0.0 and 0.0 are false
    for (v, holds) in [(f64::NAN, true), (1.0, true), (-0.0, false), (0.0, false)] {
        let r = run(&kernel(vec![assume(ScalarExpr::Const(v))]));
        let want = if holds {
            Ok(vec![1.0; 4])
        } else {
            Err(KernelTrap::AssumptionViolated)
        };
        assert_eq!(r, want, "{v:?}");
    }
}

#[test]
fn an_ill_formed_condition_is_a_check_error_not_a_trap() {
    // a local that was never declared
    let k = kernel(vec![assume(ScalarExpr::Local(7))]);
    assert!(check_kernel(&k).is_err(), "{k:?}");
}
