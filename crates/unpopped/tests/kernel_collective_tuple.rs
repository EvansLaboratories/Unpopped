// SPDX-License-Identifier: MIT OR Apache-2.0
//! **A combining collective over a tuple state** (ROADMAP R5.7 D4b, slice 2; PM ruling 2026-11-09, option B of
//! `docs/options-d4b-tuple-state-combine.md`): `Stmt::CollectiveTuple { kind: ReduceWith | ScanWith, values, results }`
//! folds the lanes' states with a module function of `2k` scalar parameters that returns `k` values, left to right in
//! lane order. The two conformance fixtures are Triton's: an argmax over `(value, index)` and Chan's Welford merge over
//! `(mean, m2, count)`. The expected values are worked out by hand.
//!
//! Argmax over `[3, 7, -1, 7, 0, 9, 9, 2]`: the largest value is `9` at lanes 5 and 6 and a tie keeps the smaller index,
//! so a block reduce gives `(9, 5)` to every lane, a subgroup reduce (groups of four) gives `(7, 1)` to lanes 0..4 and
//! `(9, 5)` to lanes 4..8, and a scan gives the running best `(3,0) (7,1) (7,1) (7,1) (7,1) (9,5) (9,5) (9,5)`.
//!
//! Welford over `[2, 4, 4, 4, 5, 5, 7, 9]` (mean 5, sum of squared deviations 32, 8 elements): the merge divides by the
//! running count, so only the first prefixes are exact in `F32` (`(2, 0, 1)` and `(3, 2, 2)`); the rest are compared
//! within `1e-4`.

use unpopped::exec::Scope;
use unpopped::kernel::{CollectiveKind, KernelError, Module, Stmt, check_module};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch_module;
use unpopped_vocab::ElementKind;

#[path = "support/collective_tuple.rs"]
mod fixture;
use fixture::{
    ARGMAX_VALS, F32, I32, N, WELFORD_VALS, argmax_module, cross_combine, local, welford_combine,
    welford_module,
};

fn floats(v: &[f32]) -> KernelArg {
    let bytes = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    KernelArg::Buffer(TypedBuffer::new(F32, vec![v.len() as i64], vec![1], bytes))
}

fn zeros(elem: ElementKind) -> KernelArg {
    KernelArg::Buffer(TypedBuffer::new(
        elem,
        vec![N as i64],
        vec![1],
        vec![0; 4 * N],
    ))
}

fn f32s(a: &KernelArg) -> Vec<f32> {
    let KernelArg::Buffer(b) = a else {
        unreachable!()
    };
    b.bytes
        .chunks(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn i32s(a: &KernelArg) -> Vec<i32> {
    let KernelArg::Buffer(b) = a else {
        unreachable!()
    };
    b.bytes
        .chunks(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

/// Run the argmax module: the `(value, index)` of every lane.
fn argmax(scan: bool, scope: Scope) -> (Vec<f32>, Vec<i32>) {
    let mut args = vec![floats(&ARGMAX_VALS), zeros(F32), zeros(I32)];
    run_launch_module(&argmax_module(scan, scope), 0, &mut args).expect("runs");
    (f32s(&args[1]), i32s(&args[2]))
}

/// Run the Welford module: the `(mean, m2, count)` of every lane.
fn welford(scan: bool) -> (Vec<f32>, Vec<f32>, Vec<i32>) {
    let mut args = vec![floats(&WELFORD_VALS), zeros(F32), zeros(F32), zeros(I32)];
    run_launch_module(&welford_module(scan), 0, &mut args).expect("runs");
    (f32s(&args[1]), f32s(&args[2]), i32s(&args[3]))
}

#[test]
fn an_argmax_reduce_keeps_the_first_of_the_largest_in_every_scope() {
    let (v, i) = argmax(false, Scope::Block);
    assert_eq!((v, i), (vec![9.0; N], vec![5; N]));
    let (v, i) = argmax(false, Scope::Subgroup);
    assert_eq!(v, vec![7.0, 7.0, 7.0, 7.0, 9.0, 9.0, 9.0, 9.0]);
    assert_eq!(i, vec![1, 1, 1, 1, 5, 5, 5, 5]);
}

#[test]
fn an_argmax_scan_is_the_running_best() {
    let (v, i) = argmax(true, Scope::Block);
    assert_eq!(v, vec![3.0, 7.0, 7.0, 7.0, 7.0, 9.0, 9.0, 9.0]);
    assert_eq!(i, vec![0, 1, 1, 1, 1, 5, 5, 5]);
    // within groups of four the scan restarts at lane 4: lanes 4..8 hold 0, 9, 9, 2
    let (v, i) = argmax(true, Scope::Subgroup);
    assert_eq!(v, vec![3.0, 7.0, 7.0, 7.0, 0.0, 9.0, 9.0, 9.0]);
    assert_eq!(i, vec![0, 1, 1, 1, 4, 5, 5, 5]);
}

fn near(got: f32, want: f32) -> bool {
    (got - want).abs() <= 1e-4 * want.abs().max(1.0)
}

#[test]
fn a_welford_reduce_gives_every_lane_the_mean_the_m2_and_the_count() {
    let (mean, m2, n) = welford(false);
    assert_eq!(n, vec![8; N]);
    assert!(
        mean.iter().all(|m| near(*m, 5.0)),
        "mean {mean:?} is within 1e-4 of 5"
    );
    assert!(
        m2.iter().all(|m| near(*m, 32.0)),
        "m2 {m2:?} is within 1e-4 of 32"
    );
}

#[test]
fn a_welford_scan_gives_each_lane_the_merge_of_the_lanes_up_to_it() {
    let (mean, m2, n) = welford(true);
    assert_eq!(n, vec![1, 2, 3, 4, 5, 6, 7, 8]);
    // lane 0 is its own element; lane 1 merges 2 and 4 exactly: mean 3, m2 = 2
    assert_eq!((mean[0], m2[0]), (2.0, 0.0));
    assert_eq!((mean[1], m2[1]), (3.0, 2.0));
    assert!(near(mean[7], 5.0) && near(m2[7], 32.0));
    // lane 3 holds 2, 4, 4, 4: mean 3.5, m2 = 2.25 + 0.25 * 3 = 3
    assert!(near(mean[3], 3.5) && near(m2[3], 3.0), "{mean:?} {m2:?}");
}

fn check(m: &Module) -> Result<(), KernelError> {
    check_module(m, 0)
}

/// The `CollectiveTuple` of the argmax module, to edit.
fn tuple_of(m: &mut Module) -> &mut Stmt {
    &mut m.fns[0].body.0[2]
}

#[test]
fn the_checker_accepts_the_fixtures_and_refuses_a_malformed_tuple_collective() {
    assert_eq!(check(&argmax_module(false, Scope::Block)), Ok(()));
    assert_eq!(check(&welford_module(true)), Ok(()));
    let refused = |m: &Module| matches!(check(m), Err(KernelError::CollectiveUnsupported { .. }));
    // one result local per value
    let mut m = argmax_module(false, Scope::Block);
    if let Stmt::CollectiveTuple { results, .. } = tuple_of(&mut m) {
        results.pop();
    }
    assert!(refused(&m), "too few result locals: {:?}", check(&m));
    // no values at all
    let mut m = argmax_module(false, Scope::Block);
    if let Stmt::CollectiveTuple {
        values, results, ..
    } = tuple_of(&mut m)
    {
        values.clear();
        results.clear();
    }
    assert!(refused(&m), "no values: {:?}", check(&m));
    // only a combining kind carries a tuple
    let mut m = argmax_module(false, Scope::Block);
    if let Stmt::CollectiveTuple { kind, .. } = tuple_of(&mut m) {
        *kind = CollectiveKind::Reduce(unpopped::ir::ReduceOp::Sum);
    }
    assert!(refused(&m), "a Reduce kind: {:?}", check(&m));
}

#[test]
fn the_combine_must_take_twice_the_components_and_return_them_with_their_types() {
    let refused = |m: &Module| matches!(check(m), Err(KernelError::CollectiveUnsupported { .. }));
    // the Welford combine behind a two-component state: 6 parameters for a state of 2 components
    let mut m = argmax_module(false, Scope::Block);
    m.fns[1] = welford_combine();
    assert!(refused(&m), "wrong parameter count: {:?}", check(&m));
    // a combine whose extra result has the wrong type (the index returned as F32)
    let mut m = argmax_module(false, Scope::Block);
    m.fns[1].locals[5] = unpopped::kernel::LocalDecl::scalar(F32);
    assert!(refused(&m), "wrong result type: {:?}", check(&m));
    // a combine whose parameter in the `b` half has the wrong type
    let mut m = argmax_module(false, Scope::Block);
    m.fns[1].params[3] = unpopped::kernel::Param::scalar("bi", F32, 3);
    assert!(refused(&m), "wrong b parameter: {:?}", check(&m));
    // a combine that returns only the first value
    let mut m = argmax_module(false, Scope::Block);
    m.fns[1].results.clear();
    assert!(refused(&m), "missing result: {:?}", check(&m));
    // a combine that reaches a barrier
    let mut m = argmax_module(false, Scope::Block);
    m.fns[1].body.0.push(Stmt::Barrier {
        scope: Scope::Block,
        kind: unpopped::kernel::BarrierKind::Full,
    });
    assert!(refused(&m), "a barrier: {:?}", check(&m));
}

#[test]
fn a_tuple_collective_needs_a_launch_a_block_or_subgroup_scope_and_a_module() {
    // a scope other than Subgroup or Block
    let mut m = argmax_module(false, Scope::Block);
    if let Stmt::CollectiveTuple { scope, .. } = tuple_of(&mut m) {
        *scope = Scope::Device;
    }
    assert_eq!(check(&m), Err(KernelError::BadBarrierScope));
    // the kernel alone has no module to resolve the combine in
    let m = argmax_module(false, Scope::Block);
    assert!(matches!(
        unpopped::kernel::check_kernel(&m.fns[0]),
        Err(KernelError::CallUnsupported { .. })
    ));
    // a result local named twice is a second declaration
    let mut m = argmax_module(false, Scope::Block);
    if let Stmt::CollectiveTuple { results, .. } = tuple_of(&mut m) {
        results[1] = results[0];
    }
    assert!(check(&m).is_err(), "{:?}", check(&m));
}

#[test]
fn a_tuple_collective_under_a_thread_dependent_condition_is_refused() {
    // a loaded value differs between lanes, so a rendezvous under a condition on it is not reached by the whole scope
    let guard = |m: &mut Module| {
        m.fns[0].body.0.truncate(3);
        let tuple = m.fns[0].body.0[2].clone();
        m.fns[0].body.0[2] = Stmt::If {
            cond: unpopped::ir::ScalarExpr::Binary(
                unpopped::ir::BinaryOp::CmpGt,
                Box::new(local(0)),
                Box::new(unpopped::ir::ScalarExpr::Const(1.0)),
            ),
            then: unpopped::kernel::Block(vec![tuple]),
            els: unpopped::kernel::Block(vec![]),
        };
    };
    let mut guarded = argmax_module(false, Scope::Block);
    guard(&mut guarded);
    assert_eq!(check(&guarded), Err(KernelError::BarrierDivergent));
    // the same body without the condition is accepted
    let mut plain = argmax_module(false, Scope::Block);
    plain.fns[0].body.0.truncate(3);
    assert_eq!(check(&plain), Ok(()));
}

#[test]
fn the_fold_applies_the_combine_with_the_running_state_first() {
    // `cross_combine` takes the value of the later state and the index of the earlier one, so it is not commutative:
    // a block reduce of `[3, 7, -1, 7, 0, 9, 9, 2]` is (last value 2, first index 0), a subgroup reduce is (7, 0) for
    // lanes 0..4 and (2, 4) for lanes 4..8, and a scan holds each lane's own value with index 0
    let run = |scan, scope| {
        let mut m = argmax_module(scan, scope);
        m.fns[1] = cross_combine();
        let mut args = vec![floats(&ARGMAX_VALS), zeros(F32), zeros(I32)];
        run_launch_module(&m, 0, &mut args).expect("runs");
        (f32s(&args[1]), i32s(&args[2]))
    };
    assert_eq!(run(false, Scope::Block), (vec![2.0; N], vec![0; N]));
    assert_eq!(
        run(false, Scope::Subgroup),
        (
            vec![7.0, 7.0, 7.0, 7.0, 2.0, 2.0, 2.0, 2.0],
            vec![0, 0, 0, 0, 4, 4, 4, 4]
        )
    );
    assert_eq!(run(true, Scope::Block), (ARGMAX_VALS.to_vec(), vec![0; N]));
}

#[test]
fn a_tuple_of_no_components_is_refused_even_for_a_combine_that_takes_none() {
    let mut m = argmax_module(false, Scope::Block);
    m.fns[1] = unpopped::kernel::KernelFn::new(
        "unit",
        F32,
        vec![],
        vec![],
        unpopped::kernel::Block(vec![]),
    );
    if let Stmt::CollectiveTuple {
        values, results, ..
    } = tuple_of(&mut m)
    {
        values.clear();
        results.clear();
    }
    assert!(
        matches!(check(&m), Err(KernelError::CollectiveUnsupported { .. })),
        "{:?}",
        check(&m)
    );
}

#[test]
fn a_tuple_collective_naming_its_own_function_is_a_call_cycle() {
    let mut m = argmax_module(false, Scope::Block);
    if let Stmt::CollectiveTuple { kind, .. } = tuple_of(&mut m) {
        *kind = CollectiveKind::ReduceWith { combine: 0 };
    }
    assert!(
        matches!(check(&m), Err(KernelError::CallCycle { .. })),
        "{:?}",
        check(&m)
    );
}
