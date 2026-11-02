// SPDX-License-Identifier: MIT OR Apache-2.0
//! A `Barrier` inside a callee (ROADMAP U3 R1, slice M4b of `docs/options-u3-calls.md`, Q6). The call that
//! reaches a barrier is a barrier site: it must sit in uniform control and pass uniform arguments (the callee
//! was checked assuming its scalar parameters are uniform), transitively through nested calls. A thread
//! stopped at a barrier inside a callee resumes inside that callee, with its own locals and buffer binding.

use unpopped::exec::{ExecId, LaunchShape, Scope};
use unpopped::ir::{BinaryOp, ScalarExpr};
use unpopped::kernel::{
    BarrierKind, Block, BufferRole, CallArg, KernelError, KernelFn, LocalDecl, Module, Param,
    Place, Stmt,
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

fn u(v: i64) -> ScalarExpr {
    ScalarExpr::ConstInt(v, U32)
}

fn at(buffer: u32, i: ScalarExpr) -> Place {
    Place::Logical {
        buffer,
        coords: vec![i],
    }
}

fn rw(name: &str) -> Param {
    Param::buffer(name, F32, BufferRole::ReadWrite, Some(1))
}

fn barrier() -> Stmt {
    Stmt::Barrier {
        scope: Scope::Block,
        kind: BarrierKind::Full,
    }
}

fn call(callee: u32, args: Vec<CallArg>) -> Stmt {
    Stmt::Call {
        callee,
        args,
        result: None,
    }
}

fn buf(vals: &[f32]) -> KernelArg {
    let bytes = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
    KernelArg::Buffer(TypedBuffer::new(
        F32,
        vec![vals.len() as i64],
        vec![1],
        bytes,
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

/// `exchange(x, out)`: `x[tid] = tid + 1`; [barrier if `sync`]; `out[tid] = x[N - 1 - tid]`.
fn exchange(sync: bool) -> KernelFn {
    let one = ScalarExpr::Const(1.0);
    let mine = ScalarExpr::Add(Box::new(f32_of(tid())), Box::new(one));
    let next = ScalarExpr::Sub(Box::new(u(i64::from(N) - 1)), Box::new(tid()));
    let mut body = vec![Stmt::Store {
        place: at(0, tid()),
        value: mine,
    }];
    if sync {
        body.push(barrier());
    }
    body.push(Stmt::Load {
        local: 0,
        place: at(0, next),
    });
    body.push(Stmt::Store {
        place: at(1, tid()),
        value: ScalarExpr::Local(0),
    });
    KernelFn::new(
        "exchange",
        F32,
        vec![rw("x"), rw("out")],
        vec![LocalDecl::scalar(F32)],
        Block(body),
    )
}

fn f32_of(e: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Cast(
        Box::new(e),
        unpopped::ir::CastSpec {
            to: F32,
            round: unpopped::ir::RoundMode::NearestEven,
            sat: unpopped::ir::Saturation::Wrap,
        },
    )
}

fn entry(body: Vec<Stmt>) -> KernelFn {
    KernelFn::new("main", F32, vec![rw("x"), rw("out")], vec![], Block(body))
        .with_launch(LaunchShape::new([N, 1, 1], [1, 1, 1]))
}

fn both() -> Vec<CallArg> {
    vec![CallArg::Buffer(0), CallArg::Buffer(1)]
}

fn run(m: &Module) -> Result<Vec<f32>, KernelTrap> {
    let mut args = vec![buf(&[0.0; N as usize]), buf(&[0.0; N as usize])];
    run_launch_module(m, 0, &mut args)?;
    Ok(read(&args[1]))
}

#[test]
fn a_barrier_in_a_callee_orders_the_threads_of_the_block() {
    let m = Module::new(vec![entry(vec![call(1, both())]), exchange(true)]);
    let want: Vec<f32> = (0..N).map(|i| (N - i) as f32).collect();
    assert_eq!(run(&m).expect("runs"), want, "each thread read its mirror");
}

#[test]
fn without_the_callees_barrier_the_exchange_is_a_data_race() {
    let m = Module::new(vec![entry(vec![call(1, both())]), exchange(false)]);
    assert_eq!(run(&m), Err(KernelTrap::DataRace));
}

#[test]
fn a_buffer_binding_survives_the_callee_barrier() {
    // the call passes the buffers swapped: the callee's `x` is the caller's `out`. After the barrier the
    // thread must still read through that binding, not through the identity map.
    let swapped = vec![CallArg::Buffer(1), CallArg::Buffer(0)];
    let m = Module::new(vec![entry(vec![call(1, swapped)]), exchange(true)]);
    let mut args = vec![buf(&[0.0; N as usize]), buf(&[0.0; N as usize])];
    run_launch_module(&m, 0, &mut args).expect("runs");
    let mirror: Vec<f32> = (0..N).map(|i| (N - i) as f32).collect();
    let own: Vec<f32> = (0..N).map(|i| (i + 1) as f32).collect();
    assert_eq!(
        read(&args[0]),
        mirror,
        "the callee's `out` is the caller's buffer 0"
    );
    assert_eq!(
        read(&args[1]),
        own,
        "the callee's `x` is the caller's buffer 1"
    );
}

#[test]
fn the_caller_resumes_inside_its_own_frame_after_the_callee_barrier() {
    // a local, and a store through the caller's own buffer map, after the call must survive the callee's
    // barrier phase change (`out` is the thread's own slot: the callee also wrote it before the barrier)
    let mut main = entry(vec![
        Stmt::Let {
            local: 0,
            init: ScalarExpr::Const(7.0),
        },
        call(1, both()),
        Stmt::Store {
            place: at(1, tid()),
            value: ScalarExpr::Local(0),
        },
    ]);
    main.locals.push(LocalDecl::scalar(F32));
    let mut args = vec![buf(&[0.0; N as usize]), buf(&[0.0; N as usize])];
    run_launch_module(&Module::new(vec![main, exchange(true)]), 0, &mut args).expect("runs");
    assert_eq!(read(&args[1]), vec![7.0; N as usize], "the caller's local");
}

#[test]
fn a_call_that_reaches_a_barrier_must_be_in_uniform_control() {
    let divergent = Stmt::If {
        cond: ScalarExpr::Binary(
            BinaryOp::CmpLt,
            Box::new(tid()),
            Box::new(u(i64::from(N / 2))),
        ),
        then: Block(vec![call(1, both())]),
        els: Block(vec![]),
    };
    let m = Module::new(vec![entry(vec![divergent]), exchange(true)]);
    assert_eq!(
        run(&m),
        Err(KernelTrap::Malformed(KernelError::BarrierDivergent))
    );
}

#[test]
fn a_barrier_call_must_pass_uniform_arguments() {
    // sync_at(x, k): a barrier only when k > 0; the callee was checked with k uniform
    let gated = KernelFn::new(
        "gated",
        F32,
        vec![rw("x"), Param::scalar("k", U32, 0)],
        vec![LocalDecl::scalar(U32)],
        Block(vec![Stmt::If {
            cond: ScalarExpr::Binary(
                BinaryOp::CmpGt,
                Box::new(ScalarExpr::Local(0)),
                Box::new(u(0)),
            ),
            then: Block(vec![barrier()]),
            els: Block(vec![]),
        }]),
    );
    let main = |arg: ScalarExpr| {
        entry(vec![call(
            1,
            vec![CallArg::Buffer(0), CallArg::Scalar(arg)],
        )])
    };
    // a uniform argument is fine; a per-thread one is not
    let ok = Module::new(vec![main(u(1)), gated.clone()]);
    assert_eq!(run(&ok), Ok(vec![0.0; N as usize]));
    let bad = Module::new(vec![main(tid()), gated]);
    assert_eq!(
        run(&bad),
        Err(KernelTrap::Malformed(KernelError::BarrierDivergent))
    );
}

#[test]
fn reaching_a_barrier_is_transitive_through_nested_calls() {
    // mid(x, out) calls exchange; a divergent call of mid is as bad as a divergent call of exchange
    let mid = KernelFn::new(
        "mid",
        F32,
        vec![rw("x"), rw("out")],
        vec![],
        Block(vec![call(2, both())]),
    );
    let divergent = Stmt::If {
        cond: ScalarExpr::Binary(
            BinaryOp::CmpLt,
            Box::new(tid()),
            Box::new(u(i64::from(N / 2))),
        ),
        then: Block(vec![call(1, both())]),
        els: Block(vec![]),
    };
    let bad = Module::new(vec![entry(vec![divergent]), mid.clone(), exchange(true)]);
    assert_eq!(
        run(&bad),
        Err(KernelTrap::Malformed(KernelError::BarrierDivergent))
    );
    let ok = Module::new(vec![entry(vec![call(1, both())]), mid, exchange(true)]);
    assert!(run(&ok).is_ok(), "a uniform call of mid runs");
}

#[test]
fn an_unlaunched_entry_cannot_reach_a_barrier() {
    let mut main = entry(vec![call(1, both())]);
    main.launch = None;
    let r = run(&Module::new(vec![main, exchange(true)]));
    assert!(
        matches!(
            r,
            Err(KernelTrap::Malformed(KernelError::ExecWithoutLaunch))
        ),
        "{r:?}"
    );
}
