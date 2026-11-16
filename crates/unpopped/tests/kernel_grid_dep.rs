// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Programmatic dependent launch statements** (ROADMAP U17): `GridDepWait` (`griddepcontrol.wait`) and
//! `GridDepLaunchDependents` (`griddepcontrol.launch_dependents`). One launch has no prerequisite grid, so the wait does
//! nothing and the launch hint orders nothing; the tests pin that a kernel carrying them computes what it computes
//! without them, in any control flow, that the checker accepts them, and that they survive serialisation. Cross-grid
//! ordering is **not modelled** (see CHANGELOG): nothing here tests it.

use unpopped::exec::{ExecId, LaunchShape};
use unpopped::ir::{BinaryOp, ScalarExpr};
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt, check_kernel};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_vocab::ElementKind;

const U32: ElementKind = ElementKind::U32;
const THREADS: u32 = 4;

fn tid() -> ScalarExpr {
    ScalarExpr::Exec(ExecId::ThreadIdx(0))
}
fn out() -> Place {
    Place::Logical {
        buffer: 0,
        coords: vec![tid()],
    }
}
fn store_three_tid() -> Stmt {
    Stmt::Store {
        place: out(),
        value: ScalarExpr::Mul(
            Box::new(tid()),
            Box::new(ScalarExpr::ConstInt(3, U32)),
        ),
    }
}
fn kernel(body: Vec<Stmt>) -> KernelFn {
    KernelFn::new(
        "grid_dep",
        U32,
        vec![Param::buffer(
            "out",
            U32,
            BufferRole::ReadWrite,
            Some(1),
        )],
        vec![LocalDecl::scalar(U32)],
        Block(body),
    )
    .with_launch(LaunchShape::new([THREADS, 1, 1], [1, 1, 1]))
}
fn run(k: &KernelFn) -> Vec<u32> {
    assert_eq!(check_kernel(k), Ok(()));
    let mut args = [KernelArg::Buffer(TypedBuffer::new(
        U32,
        vec![i64::from(THREADS)],
        vec![1],
        vec![0; 4 * THREADS as usize],
    ))];
    run_launch(k, &mut args).expect("the launch runs");
    let KernelArg::Buffer(b) = &args[0] else {
        unreachable!("a buffer")
    };
    b.bytes
        .chunks(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}
const EXPECTED: [u32; 4] = [0, 3, 6, 9];

#[test]
fn a_kernel_with_the_statements_computes_what_it_computes_without_them() {
    assert_eq!(run(&kernel(vec![store_three_tid()])), EXPECTED);
    let with = kernel(vec![
        Stmt::GridDepLaunchDependents,
        Stmt::GridDepWait,
        store_three_tid(),
        Stmt::GridDepLaunchDependents,
    ]);
    assert_eq!(run(&with), EXPECTED);
}

#[test]
fn the_wait_does_not_hide_a_wrong_result() {
    // control: the harness reports a difference when the kernel does compute something else
    let wrong = kernel(vec![
        Stmt::GridDepWait,
        Stmt::Store {
            place: out(),
            value: ScalarExpr::ConstInt(7, U32),
        },
    ]);
    assert_eq!(run(&wrong), [7, 7, 7, 7]);
}

#[test]
fn they_are_accepted_under_divergent_control_flow() {
    let guarded = |s: Stmt| Stmt::If {
        cond: ScalarExpr::Binary(
            BinaryOp::CmpEq,
            Box::new(tid()),
            Box::new(ScalarExpr::ConstInt(0, U32)),
        ),
        then: Block(vec![s]),
        els: Block(vec![]),
    };
    let k = kernel(vec![
        guarded(Stmt::GridDepWait),
        guarded(Stmt::GridDepLaunchDependents),
        store_three_tid(),
    ]);
    assert_eq!(run(&k), EXPECTED);
}

#[test]
fn they_round_trip_through_json_under_their_own_names() {
    let k = kernel(vec![Stmt::GridDepWait, Stmt::GridDepLaunchDependents]);
    let json = serde_json::to_string(&k).expect("serialises");
    assert!(json.contains("GridDepWait") && json.contains("GridDepLaunchDependents"));
    let back: KernelFn = serde_json::from_str(&json).expect("reads back");
    assert_eq!(back, k);
}
