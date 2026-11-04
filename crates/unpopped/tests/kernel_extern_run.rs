// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The oracle runs an `ExternCall` through its declaration's reference kernel** (ROADMAP U11 step 2,
//! `docs/options-u11-extern-and-inline-asm.md`). Both evaluators: `run_module` (the recursive one, an unlaunched
//! entry) and `run_launch_module` (the resumable one, one frame stack per thread). The reference is the only
//! thing the oracle can run (it cannot execute asm), so a declaration with none traps with a typed `Unsupported`,
//! and the tests pin that the *reference* is what produced the number: change the reference and the answer changes.

use unpopped::exec::{ExecId, LaunchShape};
use unpopped::ir::ScalarExpr;
use unpopped::kernel::{
    Block, BufferRole, Effects, ExternDecl, KernelFn, LocalDecl, Module, Param, Place, Stmt,
};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::{KernelArg, KernelTrap};
use unpopped::oracle::kernel_launch::run_launch_module;
use unpopped::oracle::run_module;
use unpopped_vocab::ElementKind;

const U32: ElementKind = ElementKind::U32;

fn cu(v: i64) -> ScalarExpr {
    ScalarExpr::ConstInt(v, U32)
}

fn local(i: u32) -> ScalarExpr {
    ScalarExpr::Local(i)
}

fn tid() -> ScalarExpr {
    ScalarExpr::Exec(ExecId::ThreadIdx(0))
}

fn slot(index: ScalarExpr) -> Place {
    Place::Logical {
        buffer: 0,
        coords: vec![index],
    }
}

fn ecall(decl: u32, args: Vec<ScalarExpr>, results: Vec<u32>) -> Stmt {
    Stmt::ExternCall {
        decl,
        args,
        results,
    }
}

fn store(index: ScalarExpr, value: ScalarExpr) -> Stmt {
    Stmt::Store {
        place: slot(index),
        value,
    }
}

fn add(a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Add(Box::new(a), Box::new(b))
}

fn sub(a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Sub(Box::new(a), Box::new(b))
}

/// `main(a: U32 buffer)` with `locals` U32 locals; launched on four threads when `launched`.
fn main_fn(locals: usize, launched: bool, body: Vec<Stmt>) -> KernelFn {
    let k = KernelFn::new(
        "main",
        U32,
        vec![Param::buffer("a", U32, BufferRole::ReadWrite, Some(1))],
        (0..locals).map(|_| LocalDecl::scalar(U32)).collect(),
        Block(body),
    );
    if launched {
        k.with_launch(LaunchShape::new([4, 1, 1], [1, 1, 1]))
    } else {
        k
    }
}

/// `r(x: U32) -> U32 = x + by`.
fn plus(by: i64) -> KernelFn {
    KernelFn::new(
        "plus",
        U32,
        vec![Param::scalar("x", U32, 0)],
        vec![LocalDecl::scalar(U32), LocalDecl::scalar(U32)],
        Block(vec![Stmt::Let {
            local: 1,
            init: add(local(0), cu(by)),
        }]),
    )
    .with_result(1)
}

/// `r(x: U32, y: U32) -> U32 = x - y`.
fn minus() -> KernelFn {
    KernelFn::new(
        "minus",
        U32,
        vec![Param::scalar("x", U32, 0), Param::scalar("y", U32, 1)],
        vec![
            LocalDecl::scalar(U32),
            LocalDecl::scalar(U32),
            LocalDecl::scalar(U32),
        ],
        Block(vec![Stmt::Let {
            local: 2,
            init: sub(local(0), local(1)),
        }]),
    )
    .with_result(2)
}

fn inc_decl() -> ExternDecl {
    ExternDecl::new("inc", vec![U32], vec![U32], Effects::pure())
}

fn buffer(n: usize) -> KernelArg {
    KernelArg::Buffer(TypedBuffer::new(
        U32,
        vec![n as i64],
        vec![1],
        vec![0; 4 * n],
    ))
}

fn words(a: &KernelArg) -> Vec<u32> {
    let KernelArg::Buffer(b) = a else {
        unreachable!("a buffer")
    };
    b.bytes
        .chunks(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn run(m: &Module, n: usize) -> Result<Vec<u32>, KernelTrap> {
    let mut args = [buffer(n)];
    if m.fns[0].launch.is_some() {
        run_launch_module(m, 0, &mut args)?;
    } else {
        run_module(m, 0, &mut args)?;
    }
    Ok(words(&args[0]))
}

/// `main` stores `inc(4)` at slot 0, with the reference `plus(by)`.
fn inc_of_four(by: i64) -> Module {
    Module::new(vec![
        main_fn(
            1,
            false,
            vec![ecall(0, vec![cu(4)], vec![0]), store(cu(0), local(0))],
        ),
        plus(by),
    ])
    .with_externs(vec![inc_decl().with_reference(1)])
}

#[test]
fn the_reference_is_what_the_oracle_runs() {
    assert_eq!(run(&inc_of_four(1), 1), Ok(vec![5]));
    // control: another reference gives another answer, so the number came from the reference
    assert_eq!(run(&inc_of_four(2), 1), Ok(vec![6]));
}

#[test]
fn a_declaration_with_no_reference_traps_with_a_typed_unsupported() {
    let m = Module::new(vec![main_fn(
        1,
        false,
        vec![ecall(0, vec![cu(4)], vec![0]), store(cu(0), local(0))],
    )])
    .with_externs(vec![inc_decl()]);
    match run(&m, 1) {
        Err(KernelTrap::Unsupported(what)) => assert!(what.contains("no reference body"), "{what}"),
        other => panic!("expected the no-reference trap, got {other:?}"),
    }
    // the same trap in the resumable evaluator
    let launched = Module::new(vec![main_fn(
        1,
        true,
        vec![ecall(0, vec![tid()], vec![0]), store(tid(), local(0))],
    )])
    .with_externs(vec![inc_decl()]);
    match run(&launched, 4) {
        Err(KernelTrap::Unsupported(what)) => assert!(what.contains("no reference body"), "{what}"),
        other => panic!("expected the no-reference trap, got {other:?}"),
    }
}

#[test]
fn arguments_reach_the_reference_in_declaration_order() {
    // sub(10, 3) = 7; swapped arguments would wrap to 4294967289
    let m = Module::new(vec![
        main_fn(
            1,
            false,
            vec![
                ecall(0, vec![cu(10), cu(3)], vec![0]),
                store(cu(0), local(0)),
            ],
        ),
        minus(),
    ])
    .with_externs(vec![
        ExternDecl::new("sub", vec![U32, U32], vec![U32], Effects::pure()).with_reference(1),
    ]);
    assert_eq!(run(&m, 1), Ok(vec![7]));
}

#[test]
fn a_result_feeds_the_next_call() {
    // inc(inc(4)) = 6
    let m = Module::new(vec![
        main_fn(
            2,
            false,
            vec![
                ecall(0, vec![cu(4)], vec![0]),
                ecall(0, vec![local(0)], vec![1]),
                store(cu(0), local(1)),
            ],
        ),
        plus(1),
    ])
    .with_externs(vec![inc_decl().with_reference(1)]);
    assert_eq!(run(&m, 1), Ok(vec![6]));
}

#[test]
fn every_thread_of_a_launch_runs_its_own_call() {
    let m = Module::new(vec![
        main_fn(
            1,
            true,
            vec![ecall(0, vec![tid()], vec![0]), store(tid(), local(0))],
        ),
        plus(1),
    ])
    .with_externs(vec![inc_decl().with_reference(1)]);
    assert_eq!(run(&m, 4), Ok(vec![1, 2, 3, 4]));
}

#[test]
fn a_call_outside_a_module_run_is_refused_before_it_runs() {
    use unpopped::oracle::kernel_launch::run_launch;
    let k = main_fn(1, true, vec![ecall(0, vec![cu(1)], vec![0])]);
    assert!(matches!(
        run_launch(&k, &mut [buffer(4)]),
        Err(KernelTrap::Malformed(_))
    ));
}
