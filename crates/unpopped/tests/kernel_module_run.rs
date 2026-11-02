// SPDX-License-Identifier: MIT OR Apache-2.0
//! `run_module` (ROADMAP U3 R1, slice M3 of `docs/options-u3-calls.md`): the oracle runs a `Stmt::Call` as a
//! call. The callee gets fresh locals, its scalar parameters bound by value, its buffer parameters bound by
//! alias to the caller's buffers, and hands its result local back; control flow inside it (`Return`) ends the
//! callee, not the caller. Unlaunched entries only (a launched module is slice M4).

use unpopped::exec::LaunchShape;
use unpopped::ir::{BinaryOp, ScalarExpr};
use unpopped::kernel::{
    Block, BufferRole, CallArg, KernelError, KernelFn, LocalDecl, Module, Param, Place, Stmt,
};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::{KernelArg, KernelTrap};
use unpopped::oracle::run_module;
use unpopped_vocab::ElementKind;

const F32: ElementKind = ElementKind::F32;
const U32: ElementKind = ElementKind::U32;
const I32: ElementKind = ElementKind::I32;

fn at(buffer: u32, i: i64) -> Place {
    Place::Logical {
        buffer,
        coords: vec![ScalarExpr::ConstInt(i, U32)],
    }
}

fn f(v: f64) -> ScalarExpr {
    ScalarExpr::Const(v)
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

fn rw(name: &str) -> Param {
    Param::buffer(name, F32, BufferRole::ReadWrite, Some(1))
}

fn kernel(name: &str, params: Vec<Param>, locals: usize, body: Vec<Stmt>) -> KernelFn {
    let decls = (0..locals).map(|_| LocalDecl::scalar(F32)).collect();
    KernelFn::new(name, F32, params, decls, Block(body))
}

fn call(callee: u32, args: Vec<CallArg>, result: Option<u32>) -> Stmt {
    Stmt::Call {
        callee,
        args,
        result,
    }
}

/// `put(x, k)`: `x[0] = k`.
fn put() -> KernelFn {
    let body = vec![Stmt::Store {
        place: at(0, 0),
        value: ScalarExpr::Local(0),
    }];
    kernel("put", vec![rw("x"), Param::scalar("k", F32, 0)], 1, body)
}

fn run(m: &Module, args: &mut [KernelArg]) -> Result<(), KernelTrap> {
    run_module(m, 0, args)
}

#[test]
fn a_scalar_result_comes_back_to_the_caller() {
    // g(k) -> k + 1.0
    let g = kernel(
        "g",
        vec![Param::scalar("k", F32, 0)],
        2,
        vec![Stmt::Let {
            local: 1,
            init: ScalarExpr::Add(Box::new(ScalarExpr::Local(0)), Box::new(f(1.0))),
        }],
    )
    .with_result(1);
    let main = kernel(
        "main",
        vec![rw("a")],
        1,
        vec![
            call(1, vec![CallArg::Scalar(f(2.5))], Some(0)),
            Stmt::Store {
                place: at(0, 0),
                value: ScalarExpr::Local(0),
            },
        ],
    );
    let mut args = vec![buf(&[0.0])];
    run(&Module::new(vec![main, g]), &mut args).expect("runs");
    assert_eq!(read(&args[0]), vec![3.5]);
}

#[test]
fn a_buffer_is_passed_by_alias_to_the_caller_buffer_it_names() {
    // main has buffers a (0) and b (1); the call passes b, so `put` writes b[0], not a[0].
    let main = kernel(
        "main",
        vec![rw("a"), rw("b")],
        0,
        vec![call(
            1,
            vec![CallArg::Buffer(1), CallArg::Scalar(f(7.0))],
            None,
        )],
    );
    let mut args = vec![buf(&[1.0]), buf(&[2.0])];
    run(&Module::new(vec![main, put()]), &mut args).expect("runs");
    assert_eq!((read(&args[0]), read(&args[1])), (vec![1.0], vec![7.0]));
}

#[test]
fn two_parameters_may_alias_one_caller_buffer() {
    // copy(x, y): x[1] = y[0]; both name the caller's `a`.
    let copy = kernel(
        "copy",
        vec![rw("x"), rw("y")],
        1,
        vec![
            Stmt::Load {
                local: 0,
                place: at(1, 0),
            },
            Stmt::Store {
                place: at(0, 1),
                value: ScalarExpr::Local(0),
            },
        ],
    );
    let main = kernel(
        "main",
        vec![rw("a")],
        0,
        vec![call(1, vec![CallArg::Buffer(0), CallArg::Buffer(0)], None)],
    );
    let mut args = vec![buf(&[3.0, 0.0])];
    run(&Module::new(vec![main, copy]), &mut args).expect("runs");
    assert_eq!(read(&args[0]), vec![3.0, 3.0]);
}

/// `early(k) -> r`: `r = k`; if `k > 0` return at once, else `r = 0`. It reuses local ids 0 and 1.
fn early() -> KernelFn {
    let positive = ScalarExpr::Binary(
        BinaryOp::CmpGt,
        Box::new(ScalarExpr::Local(0)),
        Box::new(f(0.0)),
    );
    let body = vec![
        Stmt::Let {
            local: 1,
            init: ScalarExpr::Local(0),
        },
        Stmt::If {
            cond: positive,
            then: Block(vec![Stmt::Return]),
            els: Block(vec![]),
        },
        Stmt::Assign {
            local: 1,
            value: f(0.0),
        },
    ];
    kernel("early", vec![Param::scalar("k", F32, 0)], 2, body).with_result(1)
}

/// A caller that keeps a value in local 0 across two calls and stores everything it saw.
fn keeps_local_zero() -> KernelFn {
    let store = |i, local| Stmt::Store {
        place: at(0, i),
        value: ScalarExpr::Local(local),
    };
    let body = vec![
        Stmt::Let {
            local: 0,
            init: f(5.0),
        },
        call(1, vec![CallArg::Scalar(f(4.0))], Some(1)),
        call(1, vec![CallArg::Scalar(f(-4.0))], Some(2)),
        store(0, 0),
        store(1, 1),
        store(2, 2),
    ];
    kernel("main", vec![rw("a")], 3, body)
}

#[test]
fn the_callee_has_its_own_locals_and_a_return_ends_only_the_callee() {
    // The callee reuses local ids 0 and 1; the caller's local 0 survives, and the caller keeps running
    // after a callee that `Return`s early.
    let mut args = vec![buf(&[0.0; 3])];
    run(&Module::new(vec![keeps_local_zero(), early()]), &mut args).expect("runs");
    assert_eq!(read(&args[0]), vec![5.0, 4.0, 0.0]);
}

/// `add(x, k)`: `x[0] = x[0] + k`.
fn add() -> KernelFn {
    let sum = ScalarExpr::Add(
        Box::new(ScalarExpr::Local(1)),
        Box::new(ScalarExpr::Local(0)),
    );
    let body = vec![
        Stmt::Load {
            local: 1,
            place: at(0, 0),
        },
        Stmt::Store {
            place: at(0, 0),
            value: sum,
        },
    ];
    kernel("add", vec![rw("x"), Param::scalar("k", F32, 0)], 2, body)
}

/// `twice(x, k)`: `add(x, k)` two times (callee 1).
fn twice() -> KernelFn {
    let once = call(
        1,
        vec![CallArg::Buffer(0), CallArg::Scalar(ScalarExpr::Local(0))],
        None,
    );
    let params = vec![rw("x"), Param::scalar("k", F32, 0)];
    kernel("twice", params, 1, vec![once.clone(), once])
}

/// `for i in 0..4: twice(a, i as f32)` (callee 2).
fn loops_over_twice() -> KernelFn {
    let spec = unpopped::ir::CastSpec {
        to: F32,
        round: unpopped::ir::RoundMode::NearestEven,
        sat: unpopped::ir::Saturation::Wrap,
    };
    let k = ScalarExpr::Cast(Box::new(ScalarExpr::Local(0)), spec);
    let step = call(2, vec![CallArg::Buffer(0), CallArg::Scalar(k)], None);
    let lp = Stmt::For {
        local: 0,
        start: ScalarExpr::ConstInt(0, I32),
        end: ScalarExpr::ConstInt(4, I32),
        step: ScalarExpr::ConstInt(1, I32),
        body: Block(vec![step]),
    };
    KernelFn::new(
        "main",
        F32,
        vec![rw("a")],
        vec![LocalDecl::scalar(I32)],
        Block(vec![lp]),
    )
}

#[test]
fn a_call_in_a_loop_and_a_call_inside_a_call() {
    let mut args = vec![buf(&[0.0])];
    let m = Module::new(vec![loops_over_twice(), add(), twice()]);
    run(&m, &mut args).expect("runs");
    assert_eq!(read(&args[0]), vec![12.0], "2 * (0 + 1 + 2 + 3)");
}

#[test]
fn refusals_are_typed() {
    // an unchecked module is malformed, not run
    let bad = kernel("main", vec![rw("a")], 0, vec![call(5, vec![], None)]);
    let r = run(&Module::new(vec![bad]), &mut [buf(&[0.0])]);
    assert!(matches!(
        r,
        Err(KernelTrap::Malformed(KernelError::BadCallee { callee: 5 }))
    ));
    // a launched entry is slice M4
    let launched = kernel("main", vec![rw("a")], 0, vec![])
        .with_launch(LaunchShape::new([1, 1, 1], [1, 1, 1]));
    let r = run(&Module::new(vec![launched]), &mut [buf(&[0.0])]);
    assert!(matches!(r, Err(KernelTrap::Unsupported(_))), "{r:?}");
}
