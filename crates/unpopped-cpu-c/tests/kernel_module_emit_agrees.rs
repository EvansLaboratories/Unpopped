// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The module C agrees with `run_module`** (ROADMAP U3 R1, slice M5 of `docs/options-u3-calls.md`):
//! `unpopped_cpu_c::lower_module` spells each function a module's entry reaches as a C function, a `Call`
//! as a C call (a buffer forwarded by alias, a scalar passed by value, the result through `res`, a nonzero
//! status returned at once). The source is compiled with a real host C compiler, run, and its written
//! buffers compared byte for byte with `oracle::run_module`.

use unpopped::backend::LowerError;
use unpopped::exec::{ExecId, LaunchShape, Scope};
use unpopped::ir::{BinaryOp, ScalarExpr};
use unpopped::kernel::{
    BarrierKind, Block, BufferRole, CallArg, KernelFn, LocalDecl, Module, Param, ParamKind, Place,
    Stmt,
};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::{KernelArg, KernelTrap};
use unpopped::oracle::kernel_launch::run_launch_module;
use unpopped::oracle::run_module;
use unpopped_cpu_c::lower_module;
use unpopped_vocab::ElementKind;

#[path = "support/cc.rs"]
mod cc;
// The shared compile-and-run driver; this file uses the part that runs an already generated source.
#[allow(dead_code)]
#[path = "support/kernel_run.rs"]
mod kernel_run;
use kernel_run::{run_generated, same_bytes};

const F32: ElementKind = ElementKind::F32;
const U32: ElementKind = ElementKind::U32;

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

fn store(buffer: u32, i: i64, local: u32) -> Stmt {
    Stmt::Store {
        place: at(buffer, i),
        value: ScalarExpr::Local(local),
    }
}

/// The written buffers of `args` after a run, with their dtypes.
fn written(k: &KernelFn, args: &[KernelArg]) -> (Vec<Vec<u8>>, Vec<ElementKind>) {
    k.params
        .iter()
        .zip(args)
        .filter_map(|(p, a)| match (&p.kind, a) {
            (ParamKind::Buffer { role, .. }, KernelArg::Buffer(b)) if *role != BufferRole::Read => {
                Some((b.bytes.clone(), b.dtype))
            }
            _ => None,
        })
        .unzip()
}

/// Lower and run `m` as C and as the oracle; the written buffers must agree (or both report out of bounds).
fn agree_module(tag: &str, m: &Module, args: Vec<KernelArg>) {
    agree_mutated(tag, m, args, |s| s);
}

fn agree_mutated(
    tag: &str,
    m: &Module,
    args: Vec<KernelArg>,
    mutate: impl FnOnce(String) -> String,
) {
    let g = lower_module(m, 0).unwrap_or_else(|e| panic!("{tag}: lowering refused: {e}"));
    let entry = &m.fns[0];
    let Some(c) = run_generated(tag, entry, &g, &args, mutate) else {
        return;
    };
    let mut interp = args;
    let ran = if entry.launch.is_some() {
        run_launch_module(m, 0, &mut interp)
    } else {
        run_module(m, 0, &mut interp)
    };
    match ran {
        Ok(()) => {
            assert_eq!(
                c.status, 0,
                "{tag}: the oracle finished, the C said {}",
                c.status
            );
            let (want, dtypes) = written(entry, &interp);
            same_bytes(tag, &c.outs, &want, &dtypes, 0);
        }
        Err(KernelTrap::OutOfBounds { .. }) => assert_eq!(c.status, 1, "{tag}: out of bounds"),
        Err(e) => panic!("{tag}: the oracle trapped ({e:?}); pick a module it finishes"),
    }
}

/// `g(k) -> k + 1.0`.
fn plus_one() -> KernelFn {
    let sum = ScalarExpr::Add(Box::new(ScalarExpr::Local(0)), Box::new(f(1.0)));
    let body = vec![Stmt::Let {
        local: 1,
        init: sum,
    }];
    kernel("plus_one", vec![Param::scalar("k", F32, 0)], 2, body).with_result(1)
}

/// `put(x, k)`: `x[0] = k`.
fn put() -> KernelFn {
    let body = vec![store(0, 0, 0)];
    kernel("put", vec![rw("x"), Param::scalar("k", F32, 0)], 1, body)
}

#[test]
fn a_scalar_result_comes_back_through_res() {
    let main = kernel(
        "main",
        vec![rw("a")],
        1,
        vec![
            call(1, vec![CallArg::Scalar(f(2.5))], Some(0)),
            store(0, 0, 0),
        ],
    );
    agree_module(
        "m_result",
        &Module::new(vec![main, plus_one()]),
        vec![buf(&[0.0])],
    );
}

#[test]
fn a_buffer_is_forwarded_to_the_caller_buffer_it_names() {
    let args = vec![CallArg::Buffer(1), CallArg::Scalar(f(7.0))];
    let main = kernel("main", vec![rw("a"), rw("b")], 0, vec![call(1, args, None)]);
    let m = Module::new(vec![main, put()]);
    agree_module("m_alias", &m, vec![buf(&[1.0]), buf(&[2.0])]);
}

/// `early(k) -> r`: `r = k`; if `k > 0` return at once, else `r = 0`.
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

fn keeps_local_zero() -> KernelFn {
    let body = vec![
        Stmt::Let {
            local: 0,
            init: f(5.0),
        },
        call(1, vec![CallArg::Scalar(f(4.0))], Some(1)),
        call(1, vec![CallArg::Scalar(f(-4.0))], Some(2)),
        store(0, 0, 0),
        store(0, 1, 1),
        store(0, 2, 2),
    ];
    kernel("main", vec![rw("a")], 3, body)
}

#[test]
fn an_early_return_stores_the_result_and_locals_stay_apart() {
    let m = Module::new(vec![keeps_local_zero(), early()]);
    agree_module("m_early", &m, vec![buf(&[0.0; 3])]);
}

#[test]
fn a_callee_out_of_bounds_returns_status_one_from_the_entry() {
    agree_module("m_oob", &far_reader(), vec![buf(&[1.0])]);
}

fn id(e: ExecId) -> ScalarExpr {
    ScalarExpr::Exec(e)
}

fn u(v: i64) -> ScalarExpr {
    ScalarExpr::ConstInt(v, U32)
}

fn f32_of(e: ScalarExpr) -> ScalarExpr {
    let spec = unpopped::ir::CastSpec {
        to: F32,
        round: unpopped::ir::RoundMode::NearestEven,
        sat: unpopped::ir::Saturation::Wrap,
    };
    ScalarExpr::Cast(Box::new(e), spec)
}

/// `BlockIdx.x * BlockDim.x + ThreadIdx.x`: the thread's slot in a one-dimensional launch.
fn slot() -> ScalarExpr {
    let base = ScalarExpr::Mul(
        Box::new(id(ExecId::BlockIdx(0))),
        Box::new(id(ExecId::BlockDim(0))),
    );
    ScalarExpr::Add(Box::new(base), Box::new(id(ExecId::ThreadIdx(0))))
}

/// `ids(x)`: `x[slot] = BlockIdx.x * 10 + ThreadIdx.x` (the callee reads ids).
fn ids_callee() -> KernelFn {
    let tens = ScalarExpr::Mul(Box::new(id(ExecId::BlockIdx(0))), Box::new(u(10)));
    let value = ScalarExpr::Add(Box::new(tens), Box::new(id(ExecId::ThreadIdx(0))));
    let body = vec![Stmt::Store {
        place: Place::Logical {
            buffer: 0,
            coords: vec![slot()],
        },
        value: f32_of(value),
    }];
    kernel("ids", vec![rw("x")], 0, body)
}

fn launched_entry(grid: [u32; 3], block: [u32; 3], body: Vec<Stmt>) -> KernelFn {
    kernel("main", vec![rw("a")], 1, body).with_launch(LaunchShape::new(block, grid))
}

fn ids_module() -> Module {
    let body = vec![call(1, vec![CallArg::Buffer(0)], None)];
    Module::new(vec![
        launched_entry([2, 1, 1], [4, 1, 1], body),
        ids_callee(),
    ])
}

#[test]
fn a_callee_under_a_launch_reads_the_calling_threads_ids() {
    agree_module("m_launch_ids", &ids_module(), vec![buf(&[0.0; 8])]);
}

#[test]
fn ids_are_forwarded_through_a_nested_call_and_a_result_per_thread() {
    // mid(x) calls ids(x); the entry calls mid, then a per-thread result r[slot] = plus_one(ThreadIdx.x)
    let mid = kernel(
        "mid",
        vec![rw("x")],
        0,
        vec![call(2, vec![CallArg::Buffer(0)], None)],
    );
    let arg = CallArg::Scalar(f32_of(id(ExecId::ThreadIdx(0))));
    let store_r = Stmt::Store {
        place: Place::Logical {
            buffer: 1,
            coords: vec![slot()],
        },
        value: ScalarExpr::Local(0),
    };
    let body = vec![
        call(1, vec![CallArg::Buffer(0)], None),
        call(3, vec![arg], Some(0)),
        store_r,
    ];
    let mut main = launched_entry([2, 1, 1], [4, 1, 1], body);
    main.params.push(rw("r"));
    let m = Module::new(vec![main, mid, ids_callee(), plus_one()]);
    agree_module("m_launch_nested", &m, vec![buf(&[0.0; 8]), buf(&[0.0; 8])]);
}

#[test]
fn a_callee_that_holds_a_barrier_under_a_launch_is_a_typed_refusal() {
    let barrier = Stmt::Barrier {
        scope: Scope::Block,
        kind: BarrierKind::Full,
    };
    let sync = kernel("sync", vec![rw("x")], 0, vec![barrier]);
    let body = vec![call(1, vec![CallArg::Buffer(0)], None)];
    let main = launched_entry([1, 1, 1], [4, 1, 1], body);
    let r = lower_module(&Module::new(vec![main, sync]), 0);
    // refused up front, by name, not by the callee emitter tripping over its barrier
    assert!(
        matches!(&r, Err(LowerError::UnsupportedKernel { detail }) if detail.contains("called under a launch")),
        "{r:?}"
    );
}

/// Whether the differential notices `mutate` applied to the generated C.
fn caught(tag: &str, m: &Module, args: Vec<KernelArg>, mutate: fn(String) -> String) -> bool {
    std::panic::catch_unwind(|| agree_mutated(tag, m, args, mutate)).is_err()
}

fn far_reader() -> Module {
    let far = kernel(
        "far",
        vec![rw("x")],
        1,
        vec![
            Stmt::Load {
                local: 0,
                place: at(0, 5),
            },
            store(0, 0, 0),
        ],
    );
    let args = vec![CallArg::Buffer(0)];
    let main = kernel("main", vec![rw("a")], 0, vec![call(1, args, None)]);
    Module::new(vec![main, far])
}

#[test]
fn seeded_mutants_of_the_generated_c_are_noticed() {
    let early = Module::new(vec![keeps_local_zero(), early()]);
    let res_args = || vec![buf(&[0.0; 3])];
    // the early `Return` no longer stores the result
    let m = |s: String| s.replace("{ *res = l1; return 0; }", "return 0;");
    assert!(
        caught("mut_early", &early, res_args(), m),
        "early return store"
    );
    // the end of the callee no longer stores the result
    let plus = Module::new(vec![
        kernel(
            "main",
            vec![rw("a")],
            1,
            vec![
                call(1, vec![CallArg::Scalar(f(2.5))], Some(0)),
                store(0, 0, 0),
            ],
        ),
        plus_one(),
    ]);
    let m = |s: String| {
        s.replacen(
            "    *res = l1;
",
            "",
            1,
        )
    };
    assert!(
        caught("mut_final", &plus, vec![buf(&[0.0])], m),
        "final store"
    );
    // a nonzero callee status is swallowed
    let m = |s: String| s.replace("if (st0) return st0;", "");
    assert!(
        caught("mut_status", &far_reader(), vec![buf(&[1.0])], m),
        "status"
    );
}

#[test]
fn seeded_mutants_of_the_launched_c_are_noticed() {
    let m = ids_module();
    let buffers = || vec![buf(&[0.0; 8])];
    // the thread id is passed where the block id belongs
    let swap = |s: String| s.replace("((unsigned int)blk0)", "((unsigned int)(t % 4))");
    assert!(caught("mut_ids_swap", &m, buffers(), swap), "ids swapped");
    // the block id is not passed down
    let drop_blk = |s: String| s.replace("((unsigned int)blk0)", "0u");
    assert!(
        caught("mut_blk0", &m, buffers(), drop_blk),
        "block id dropped"
    );
    // the block size constant the callee reads is wrong
    let dim = |s: String| s.replace("4u", "3u");
    assert!(caught("mut_dim", &m, buffers(), dim), "block dim");
}
