// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The module C of an `ExternCall` agrees with the oracle** (ROADMAP U11 step 2,
//! `docs/options-u11-extern-and-inline-asm.md`). cpu-c cannot spell asm, so it spells an `ExternCall` as a C call of the
//! declaration's **reference** kernel, exactly as the oracle runs it. The source is compiled with a real host C
//! compiler, run, and its written buffers compared byte for byte with `run_module` / `run_launch_module`.
//! A declaration with no reference, or with a body of target `"cpu-c"` the emitter would have to honour, is refused
//! with a typed error rather than silently replaced.

use unpopped::backend::LowerError;
use unpopped::exec::{ExecId, LaunchShape};
use unpopped::ir::ScalarExpr;
use unpopped::kernel::{
    Block, BodyText, BufferRole, Effects, ExternDecl, KernelFn, LocalDecl, Module, Param,
    ParamKind, Place, Stmt, TargetBody,
};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch_module;
use unpopped::oracle::run_module;
use unpopped_cpu_c::lower_module;
use unpopped_vocab::ElementKind;

#[path = "support/cc.rs"]
mod cc;
#[expect(
    dead_code,
    reason = "the shared driver; this file uses the part that runs a generated source"
)]
#[path = "support/kernel_run.rs"]
mod kernel_run;
use kernel_run::{run_generated, same_bytes};

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

fn ecall(decl: u32, args: Vec<ScalarExpr>, results: Vec<u32>) -> Stmt {
    Stmt::ExternCall {
        decl,
        args,
        results,
    }
}

fn store(index: ScalarExpr, value: ScalarExpr) -> Stmt {
    Stmt::Store {
        place: Place::Logical {
            buffer: 0,
            coords: vec![index],
        },
        value,
    }
}

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

/// `r(x: U32) -> U32 = x + 1`.
fn plus_one() -> KernelFn {
    KernelFn::new(
        "plus_one",
        U32,
        vec![Param::scalar("x", U32, 0)],
        vec![LocalDecl::scalar(U32), LocalDecl::scalar(U32)],
        Block(vec![Stmt::Let {
            local: 1,
            init: ScalarExpr::Add(Box::new(local(0)), Box::new(cu(1))),
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
            init: ScalarExpr::Sub(Box::new(local(0)), Box::new(local(1))),
        }]),
    )
    .with_result(2)
}

fn inc_decl() -> ExternDecl {
    ExternDecl::new("inc", vec![U32], vec![U32], Effects::pure())
}

fn buf(n: usize) -> KernelArg {
    KernelArg::Buffer(TypedBuffer::new(
        U32,
        vec![n as i64],
        vec![1],
        vec![0; 4 * n],
    ))
}

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

/// Lower and run `m` as C and as the oracle; the written buffers must agree byte for byte.
fn agree(tag: &str, m: &Module, n: usize) {
    agree_mutated(tag, m, n, |s| s);
}

fn agree_mutated(tag: &str, m: &Module, n: usize, mutate: impl FnOnce(String) -> String) {
    let g = lower_module(m, 0).unwrap_or_else(|e| panic!("{tag}: lowering refused: {e}"));
    let entry = &m.fns[0];
    let args = vec![buf(n)];
    let Some(c) = run_generated(tag, entry, &g, &args, mutate) else {
        return;
    };
    let mut interp = args;
    if entry.launch.is_some() {
        run_launch_module(m, 0, &mut interp).unwrap();
    } else {
        run_module(m, 0, &mut interp).unwrap();
    }
    assert_eq!(c.status, 0, "{tag}: the C said {}", c.status);
    let (want, dtypes) = written(entry, &interp);
    same_bytes(tag, &c.outs, &want, &dtypes, 0);
}

fn inc_of_four() -> Module {
    Module::new(vec![
        main_fn(
            1,
            false,
            vec![ecall(0, vec![cu(4)], vec![0]), store(cu(0), local(0))],
        ),
        plus_one(),
    ])
    .with_externs(vec![inc_decl().with_reference(1)])
}

#[test]
fn an_extern_call_agrees_with_the_oracle() {
    agree("x_inc", &inc_of_four(), 1);
}

#[test]
fn arguments_arrive_in_declaration_order() {
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
    agree("x_order", &m, 1);
}

#[test]
fn a_result_feeds_the_next_call() {
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
        plus_one(),
    ])
    .with_externs(vec![inc_decl().with_reference(1)]);
    agree("x_chain", &m, 1);
}

#[test]
fn a_launched_extern_call_agrees_with_the_oracle() {
    let m = Module::new(vec![
        main_fn(
            1,
            true,
            vec![ecall(0, vec![tid()], vec![0]), store(tid(), local(0))],
        ),
        plus_one(),
    ])
    .with_externs(vec![inc_decl().with_reference(1)]);
    agree("x_launch", &m, 4);
}

#[test]
fn a_declaration_with_no_reference_is_refused_with_a_typed_error() {
    let m = Module::new(vec![main_fn(
        1,
        false,
        vec![ecall(0, vec![cu(4)], vec![0]), store(cu(0), local(0))],
    )])
    .with_externs(vec![inc_decl()]);
    match lower_module(&m, 0) {
        Err(LowerError::UnsupportedKernel { detail }) => {
            assert!(detail.contains("no reference body"), "{detail}");
        }
        other => panic!("expected a typed refusal, got {other:?}"),
    }
}

#[test]
fn a_body_for_cpu_c_itself_is_refused_not_ignored() {
    let body = TargetBody::new(
        "cpu-c",
        BodyText::Symbol {
            library: "m".into(),
            symbol: "inc".into(),
        },
    );
    let m = Module::new(vec![
        main_fn(
            1,
            false,
            vec![ecall(0, vec![cu(4)], vec![0]), store(cu(0), local(0))],
        ),
        plus_one(),
    ])
    .with_externs(vec![inc_decl().with_reference(1).with_body(body)]);
    match lower_module(&m, 0) {
        Err(LowerError::UnsupportedKernel { detail }) => {
            assert!(detail.contains("body for target cpu-c"), "{detail}");
        }
        other => panic!("expected a typed refusal, got {other:?}"),
    }
    // control: a body for another target is not cpu-c's business
    let other = TargetBody::new(
        "cuda-ptx",
        BodyText::Asm {
            template: "add.u32 {1}, {0}, 1;".into(),
            constraints: "=r,r".into(),
        },
    );
    let ok = Module::new(vec![
        main_fn(
            1,
            false,
            vec![ecall(0, vec![cu(4)], vec![0]), store(cu(0), local(0))],
        ),
        plus_one(),
    ])
    .with_externs(vec![inc_decl().with_reference(1).with_body(other)]);
    agree("x_other_target", &ok, 1);
}
