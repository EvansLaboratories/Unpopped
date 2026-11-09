// SPDX-License-Identifier: MIT OR Apache-2.0
//! **A function that returns several values** (ROADMAP R5.7 D4b, slice 1; PM ruling 2026-11-09, option B of
//! `docs/options-d4b-tuple-state-combine.md`): `KernelFn::results` names the values after `result`, and
//! `Stmt::CallTuple { results }` receives them all, in order. The fixture is the argmax step of Triton's tuple combine,
//! `best(v, i, bv, bi) -> (nv, ni)`, called once per element in a serial loop and once per thread in a launch.
//!
//! The expected values are worked out by hand. `vals = [1.5, 4.0, -2.0, 4.0, 3.0, 0.5]` folds to `(4.0, 1)`: the second
//! `4.0` (index 3) ties and loses to the smaller index. The launch folds `[2, 5, 5, 1]`: thread `t` pairs `(vals[t], t)`
//! with `(vals[3 - t], 3 - t)`, giving values `[2, 5, 5, 2]` and indices `[0, 1, 1, 0]` (threads 1 and 2 tie on `5` and both
//! pick index 1).

use unpopped::exec::Scope;
use unpopped::ir::BinaryOp;
use unpopped::kernel::{BarrierKind, Block, CallArg, KernelError, Module, Stmt, check_module};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch_module;
use unpopped::oracle::run_module;

#[path = "support/call_tuple.rs"]
mod fixture;
use fixture::{F32, I32, argmax_loop, ci, cmp, local, pairwise_launch};

fn floats(v: &[f32]) -> KernelArg {
    let bytes = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    KernelArg::Buffer(TypedBuffer::new(F32, vec![v.len() as i64], vec![1], bytes))
}

fn zeros(elem: unpopped_vocab::ElementKind, n: usize) -> KernelArg {
    KernelArg::Buffer(TypedBuffer::new(
        elem,
        vec![n as i64],
        vec![1],
        vec![0; 4 * n],
    ))
}

fn read_f32(a: &KernelArg) -> Vec<f32> {
    let KernelArg::Buffer(b) = a else {
        unreachable!()
    };
    b.bytes
        .chunks(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn read_i32(a: &KernelArg) -> Vec<i32> {
    let KernelArg::Buffer(b) = a else {
        unreachable!()
    };
    b.bytes
        .chunks(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

#[test]
fn a_loop_of_two_value_calls_finds_the_argmax_and_keeps_the_first_of_a_tie() {
    let mut args = vec![
        floats(&[1.5, 4.0, -2.0, 4.0, 3.0, 0.5]),
        zeros(F32, 1),
        zeros(I32, 1),
    ];
    run_module(&argmax_loop(), 0, &mut args).expect("runs");
    assert_eq!(read_f32(&args[1]), vec![4.0]);
    assert_eq!(read_i32(&args[2]), vec![1]);
    // a strictly larger value later wins
    let mut later = vec![
        floats(&[1.5, 4.0, -2.0, 4.0, 9.0, 0.5]),
        zeros(F32, 1),
        zeros(I32, 1),
    ];
    run_module(&argmax_loop(), 0, &mut later).expect("runs");
    assert_eq!(read_f32(&later[1]), vec![9.0]);
    assert_eq!(read_i32(&later[2]), vec![4]);
}

#[test]
fn a_launched_two_value_call_agrees_with_the_hand_worked_pairs() {
    let mut args = vec![floats(&[2.0, 5.0, 5.0, 1.0]), zeros(F32, 4), zeros(I32, 4)];
    run_launch_module(&pairwise_launch(), 0, &mut args).expect("runs");
    assert_eq!(read_f32(&args[1]), vec![2.0, 5.0, 5.0, 2.0]);
    assert_eq!(read_i32(&args[2]), vec![0, 1, 1, 0]);
}

fn check(m: &Module) -> Result<(), KernelError> {
    check_module(m, 0)
}

/// `argmax_loop` with the `CallTuple` of the step replaced by `call`.
fn with_call(call: Stmt) -> Module {
    let mut m = argmax_loop();
    let Stmt::For { body, .. } = &mut m.fns[0].body.0[2] else {
        unreachable!("the loop is the third statement")
    };
    body.0[1] = call;
    m
}

fn tuple_call(results: Vec<u32>) -> Stmt {
    Stmt::CallTuple {
        callee: 1,
        args: [3, 2, 0, 1].map(|l| CallArg::Scalar(local(l))).to_vec(),
        results,
    }
}

#[test]
fn a_tuple_call_must_name_one_local_per_value_of_its_callee() {
    assert_eq!(check(&argmax_loop()), Ok(()));
    for bad in [vec![4], vec![4, 5, 6]] {
        assert!(
            matches!(
                check(&with_call(tuple_call(bad.clone()))),
                Err(KernelError::CallResult { .. })
            ),
            "{bad:?} is refused"
        );
    }
}

#[test]
fn each_result_local_must_have_the_type_of_its_value() {
    // local 4 is F32 and local 5 is I32 in the entry; swapping them gives each local the other's type
    let mut m = with_call(tuple_call(vec![5, 4]));
    let err = check(&m).expect_err("a type mismatch is refused");
    assert!(
        matches!(err, KernelError::TypeMismatch { .. }),
        "got {err:?}"
    );
    // and a callee whose extra result names a local it does not declare is refused when the module is checked
    m = argmax_loop();
    m.fns[1].results = vec![99];
    assert_eq!(check(&m), Err(KernelError::UnknownLocal { local: 99 }));
}

#[test]
fn a_tuple_call_is_an_edge_of_the_call_graph() {
    // the callee calls itself through a CallTuple: the cycle is found
    let mut m = argmax_loop();
    m.fns[1].body.0.push(Stmt::CallTuple {
        callee: 1,
        args: [0, 1, 2, 3].map(|l| CallArg::Scalar(local(l))).to_vec(),
        results: vec![4, 5],
    });
    assert!(
        matches!(check(&m), Err(KernelError::CallCycle { .. })),
        "{:?}",
        check(&m)
    );
}

#[test]
fn a_tuple_call_result_is_divergent_so_a_barrier_under_it_is_refused() {
    let barrier = || Stmt::Barrier {
        scope: Scope::Block,
        kind: BarrierKind::Full,
    };
    let guarded = |cond| Stmt::If {
        cond,
        then: Block(vec![barrier()]),
        els: Block(vec![]),
    };
    // thread-dependent arguments make the call's values differ between threads
    let mut under = pairwise_launch();
    under.fns[0]
        .body
        .0
        .push(guarded(cmp(BinaryOp::CmpGt, local(5), ci(0))));
    assert_eq!(check_module(&under, 0), Err(KernelError::BarrierDivergent));
    // the same barrier outside any condition is fine
    let mut plain = pairwise_launch();
    plain.fns[0].body.0.push(barrier());
    assert_eq!(check_module(&plain, 0), Ok(()));
}
