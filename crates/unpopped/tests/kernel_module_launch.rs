// SPDX-License-Identifier: MIT OR Apache-2.0
//! `run_launch_module` (ROADMAP U3 R1, slice M4a of `docs/options-u3-calls.md`): a launched entry may call
//! plain functions. Each thread runs its calls on its own frames; a callee reads the calling thread's
//! execution ids (it runs inside the entry's launch); its accesses are logged against the calling thread and
//! the caller's buffer, so the race detector sees through a call. A barrier inside a callee is
//! `kernel_module_barrier.rs` (slice M4b).

use unpopped::exec::{ExecId, LaunchShape};
use unpopped::ir::ScalarExpr;
use unpopped::kernel::{
    Block, BufferRole, CallArg, KernelFn, LocalDecl, Module, Param, Place, Stmt,
};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::{KernelArg, KernelTrap};
use unpopped::oracle::kernel_launch::run_launch_module;
use unpopped_vocab::ElementKind;

const F32: ElementKind = ElementKind::F32;
const U32: ElementKind = ElementKind::U32;
const N: u32 = 8;

fn tid() -> ScalarExpr {
    ScalarExpr::Exec(ExecId::ThreadIdx(0))
}

fn buf() -> KernelArg {
    KernelArg::Buffer(TypedBuffer::new(
        F32,
        vec![i64::from(N)],
        vec![1],
        vec![0; 4 * N as usize],
    ))
}

fn read(a: &KernelArg) -> Vec<f32> {
    let KernelArg::Buffer(b) = a else {
        unreachable!()
    };
    b.bytes
        .chunks(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn rw() -> Param {
    Param::buffer("x", F32, BufferRole::ReadWrite, Some(1))
}

fn at(buffer: u32, i: ScalarExpr) -> Place {
    Place::Logical {
        buffer,
        coords: vec![i],
    }
}

fn call(callee: u32, args: Vec<CallArg>) -> Stmt {
    Stmt::Call {
        callee,
        args,
        result: None,
    }
}

fn launched(body: Vec<Stmt>) -> KernelFn {
    KernelFn::new("main", F32, vec![rw()], vec![], Block(body))
        .with_launch(LaunchShape::new([N, 1, 1], [1, 1, 1]))
}

/// `put(x, i: U32)`: `x[i] = 2.0`.
fn put_at() -> KernelFn {
    KernelFn::new(
        "put_at",
        F32,
        vec![rw(), Param::scalar("i", U32, 0)],
        vec![LocalDecl::scalar(U32)],
        Block(vec![Stmt::Store {
            place: at(0, ScalarExpr::Local(0)),
            value: ScalarExpr::Const(2.0),
        }]),
    )
}

/// `mark(x)`: `x[ThreadIdx.x] = 1.0`, reading the calling thread's id.
fn mark() -> KernelFn {
    KernelFn::new(
        "mark",
        F32,
        vec![rw()],
        vec![],
        Block(vec![Stmt::Store {
            place: at(0, tid()),
            value: ScalarExpr::Const(1.0),
        }]),
    )
}

fn run(m: &Module) -> Result<Vec<f32>, KernelTrap> {
    let mut args = vec![buf()];
    run_launch_module(m, 0, &mut args)?;
    Ok(read(&args[0]))
}

#[test]
fn every_thread_calls_with_its_own_arguments() {
    let main = launched(vec![call(
        1,
        vec![CallArg::Buffer(0), CallArg::Scalar(tid())],
    )]);
    let out = run(&Module::new(vec![main, put_at()])).expect("runs");
    assert_eq!(out, vec![2.0; N as usize], "each thread wrote its own slot");
}

#[test]
fn a_callee_reads_the_calling_threads_ids() {
    let main = launched(vec![call(1, vec![CallArg::Buffer(0)])]);
    let out = run(&Module::new(vec![main, mark()])).expect("runs");
    assert_eq!(out, vec![1.0; N as usize]);
}

#[test]
fn the_race_detector_sees_through_a_call() {
    // every thread calls put_at(x, 0): N writes to one element, made inside a callee
    let main = launched(vec![call(
        1,
        vec![
            CallArg::Buffer(0),
            CallArg::Scalar(ScalarExpr::ConstInt(0, U32)),
        ],
    )]);
    assert_eq!(
        run(&Module::new(vec![main, put_at()])),
        Err(KernelTrap::DataRace)
    );
}

#[test]
fn calls_nest_and_a_return_ends_only_the_callee() {
    // outer(x): mark(x); Return.   main: outer(x); x[tid] = 9.0 (it runs only if `Return` ended the callee)
    let outer = KernelFn::new(
        "outer",
        F32,
        vec![rw()],
        vec![],
        Block(vec![call(2, vec![CallArg::Buffer(0)]), Stmt::Return]),
    );
    let main = launched(vec![
        call(1, vec![CallArg::Buffer(0)]),
        Stmt::Store {
            place: at(0, tid()),
            value: ScalarExpr::Const(9.0),
        },
    ]);
    let out = run(&Module::new(vec![main, outer, mark()])).expect("runs");
    assert_eq!(out, vec![9.0; N as usize]);
}

#[test]
fn the_callers_frame_is_back_after_a_call() {
    // main has buffers a (0) and b (1) and a local; the call writes b, then main uses its local and its
    // buffer map again: a[tid] = the local, untouched by the callee's own locals and buffer binding.
    let main = KernelFn::new(
        "main",
        F32,
        vec![rw(), rw()],
        vec![LocalDecl::scalar(F32)],
        Block(vec![
            Stmt::Let {
                local: 0,
                init: ScalarExpr::Const(5.0),
            },
            call(1, vec![CallArg::Buffer(1), CallArg::Scalar(tid())]),
            Stmt::Store {
                place: at(0, tid()),
                value: ScalarExpr::Local(0),
            },
        ]),
    )
    .with_launch(LaunchShape::new([N, 1, 1], [1, 1, 1]));
    let mut args = vec![buf(), buf()];
    run_launch_module(&Module::new(vec![main, put_at()]), 0, &mut args).expect("runs");
    assert_eq!(
        read(&args[0]),
        vec![5.0; N as usize],
        "a: the caller's local"
    );
    assert_eq!(
        read(&args[1]),
        vec![2.0; N as usize],
        "b: the callee's write"
    );
}

#[test]
fn a_result_comes_back_to_the_calling_thread() {
    // twice(k: F32) -> k + k.   main: r = twice(tid as F32 + 1); x[tid] = r
    let twice = KernelFn::new(
        "twice",
        F32,
        vec![Param::scalar("k", F32, 0)],
        vec![LocalDecl::scalar(F32), LocalDecl::scalar(F32)],
        Block(vec![Stmt::Let {
            local: 1,
            init: ScalarExpr::Add(
                Box::new(ScalarExpr::Local(0)),
                Box::new(ScalarExpr::Local(0)),
            ),
        }]),
    )
    .with_result(1);
    let k = ScalarExpr::Cast(
        Box::new(tid()),
        unpopped::ir::CastSpec {
            to: F32,
            round: unpopped::ir::RoundMode::NearestEven,
            sat: unpopped::ir::Saturation::Wrap,
        },
    );
    let main = KernelFn::new(
        "main",
        F32,
        vec![rw()],
        vec![LocalDecl::scalar(F32)],
        Block(vec![
            Stmt::Call {
                callee: 1,
                args: vec![CallArg::Scalar(k)],
                result: Some(0),
            },
            Stmt::Store {
                place: at(0, tid()),
                value: ScalarExpr::Local(0),
            },
        ]),
    )
    .with_launch(LaunchShape::new([N, 1, 1], [1, 1, 1]));
    let out = run(&Module::new(vec![main, twice])).expect("runs");
    let want: Vec<f32> = (0..N).map(|i| 2.0 * i as f32).collect();
    assert_eq!(out, want, "each thread got its own result");
}
