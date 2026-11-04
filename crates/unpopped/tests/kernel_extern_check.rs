// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Typed `Extern` declarations and `Stmt::ExternCall`, checked** (ROADMAP U11 step 1,
//! `docs/options-u11-extern-and-inline-asm.md`). Unpopped cannot look inside an asm body, so every test here is
//! about the declaration held to its own words and a call held to the declaration: effects that contradict each
//! other, bodies of a bad shape, a reference with the wrong signature, a call with the wrong arity or types, and the
//! uniformity rule (a `sync` call is a barrier site; a call that is not pure has results that may differ between
//! threads). Each refusal is named by its text, so a refusal for another reason does not pass.

use unpopped::exec::{ExecId, LaunchShape, Scope};
use unpopped::ir::ScalarExpr;
use unpopped::kernel::{
    BarrierKind, Block, BodyText, BufferRole, CallArg, Effects, ExternDecl, KernelError, KernelFn,
    LocalDecl, Module, Param, Place, Stmt, TargetBody, check_kernel, check_module,
};
use unpopped_vocab::ElementKind;

const U32: ElementKind = ElementKind::U32;
const F32: ElementKind = ElementKind::F32;
const I32: ElementKind = ElementKind::I32;

fn cu(v: i64) -> ScalarExpr {
    ScalarExpr::ConstInt(v, U32)
}

fn tid() -> ScalarExpr {
    ScalarExpr::Exec(ExecId::ThreadIdx(0))
}

/// Non-zero where the thread index differs from 1: divergent.
fn divergent() -> ScalarExpr {
    ScalarExpr::Sub(Box::new(tid()), Box::new(cu(1)))
}

fn store(local: u32) -> Stmt {
    Stmt::Store {
        place: Place::Logical {
            buffer: 0,
            coords: vec![cu(0)],
        },
        value: ScalarExpr::Local(local),
    }
}

fn if_(cond: ScalarExpr, then: Vec<Stmt>) -> Stmt {
    Stmt::If {
        cond,
        then: Block(then),
        els: Block(vec![]),
    }
}

fn barrier() -> Stmt {
    Stmt::Barrier {
        scope: Scope::Block,
        kind: BarrierKind::Full,
    }
}

fn ecall(decl: u32, args: Vec<ScalarExpr>, results: Vec<u32>) -> Stmt {
    Stmt::ExternCall {
        decl,
        args,
        results,
    }
}

/// `main(a: U32 buffer)` launched on one block of four threads, with `locals` U32 locals.
fn main_fn(locals: usize, body: Vec<Stmt>) -> KernelFn {
    KernelFn::new(
        "main",
        U32,
        vec![Param::buffer("a", U32, BufferRole::ReadWrite, Some(1))],
        (0..locals).map(|_| LocalDecl::scalar(U32)).collect(),
        Block(body),
    )
    .with_launch(LaunchShape::new([4, 1, 1], [1, 1, 1]))
}

/// `inc(x: U32) -> U32`, pure, with a libdevice-style symbol and a PTX body.
fn inc() -> ExternDecl {
    ExternDecl::new("inc", vec![U32], vec![U32], Effects::pure())
        .with_body(TargetBody::new(
            "cuda-ptx",
            BodyText::Asm {
                template: "add.u32 {1}, {0}, 1;".into(),
                constraints: "=r,r".into(),
            },
        ))
        .with_body(TargetBody::new(
            "rocm",
            BodyText::Symbol {
                library: "ocml".into(),
                symbol: "inc".into(),
            },
        ))
}

/// `clock() -> U32`: an effect Unpopped does not model.
fn clock() -> ExternDecl {
    ExternDecl::new("clock", vec![], vec![U32], Effects::opaque())
}

/// `sync_all()`: a rendezvous of the block, no operands.
fn sync_all() -> ExternDecl {
    ExternDecl::new(
        "sync_all",
        vec![],
        vec![],
        Effects::opaque().with_sync(Scope::Block),
    )
}

fn module(main: KernelFn, rest: Vec<KernelFn>, externs: Vec<ExternDecl>) -> Module {
    let mut fns = vec![main];
    fns.extend(rest);
    Module::new(fns).with_externs(externs)
}

fn refused(m: &Module, text: &str) {
    match check_module(m, 0) {
        Err(KernelError::ExternUnsupported { what }) => {
            assert!(what.contains(text), "{what:?} does not mention {text:?}");
        }
        other => panic!("expected an ExternUnsupported mentioning {text:?}, got {other:?}"),
    }
}

/// The reference kernel of `inc`: `r(x: U32) -> U32 = x + 1`.
fn inc_reference() -> KernelFn {
    KernelFn::new(
        "inc_ref",
        U32,
        vec![Param::scalar("x", U32, 0)],
        vec![LocalDecl::scalar(U32), LocalDecl::scalar(U32)],
        Block(vec![Stmt::Let {
            local: 1,
            init: ScalarExpr::Add(Box::new(ScalarExpr::Local(0)), Box::new(cu(1))),
        }]),
    )
    .with_result(1)
}

#[test]
fn a_well_formed_call_checks_in_its_module_and_not_alone() {
    let m = module(
        main_fn(1, vec![ecall(0, vec![cu(4)], vec![0]), store(0)]),
        vec![],
        vec![inc()],
    );
    assert_eq!(check_module(&m, 0), Ok(()));
    match check_kernel(&m.fns[0]) {
        Err(KernelError::ExternUnsupported { what }) => assert!(what.contains("outside a module")),
        other => panic!("expected the outside-a-module refusal, got {other:?}"),
    }
}

#[test]
fn a_declaration_must_agree_with_its_own_effects() {
    let call = |d: ExternDecl| module(main_fn(1, vec![ecall(0, vec![], vec![0])]), vec![], vec![d]);
    let mut volatile_pure = ExternDecl::new("a", vec![], vec![U32], Effects::pure());
    volatile_pure.effects.volatile = true;
    refused(
        &call(volatile_pure),
        "pure declaration that is also volatile",
    );
    let sync_pure = ExternDecl::new(
        "b",
        vec![],
        vec![U32],
        Effects::pure().with_sync(Scope::Block),
    );
    refused(
        &call(sync_pure),
        "pure declaration that is also volatile or sync",
    );
    let dead = ExternDecl::new("c", vec![], vec![], Effects::pure());
    refused(&call(dead), "no result");
    let mut reads = ExternDecl::new("d", vec![], vec![U32], Effects::opaque());
    reads.effects.reads = vec![unpopped::exec::MemorySpace::Global];
    refused(&call(reads), "pointer operands");
    let mut writes = ExternDecl::new("e", vec![], vec![U32], Effects::opaque());
    writes.effects.writes = vec![unpopped::exec::MemorySpace::Shared];
    refused(&call(writes), "pointer operands");
    let thread = ExternDecl::new(
        "f",
        vec![],
        vec![U32],
        Effects::opaque().with_sync(Scope::Thread),
    );
    refused(&call(thread), "Thread scope");
}

#[test]
fn a_body_must_be_of_a_sane_shape() {
    let with = |b: TargetBody| {
        module(
            main_fn(1, vec![ecall(0, vec![cu(1)], vec![0])]),
            vec![],
            vec![ExternDecl::new("x", vec![U32], vec![U32], Effects::pure()).with_body(b)],
        )
    };
    let asm = |t: &str| BodyText::Asm {
        template: t.into(),
        constraints: String::new(),
    };
    // control: operands {0} and {1} are the parameter and the result
    assert_eq!(
        check_module(&with(TargetBody::new("t", asm("mov {1}, {0};"))), 0),
        Ok(())
    );
    refused(
        &with(TargetBody::new("t", asm("mov {2}, {0};"))),
        "placeholder past",
    );
    refused(&with(TargetBody::new("t", asm("  "))), "empty template");
    refused(&with(TargetBody::new("", asm("nop"))), "empty target");
    refused(
        &with(TargetBody::new(
            "t",
            BodyText::Symbol {
                library: String::new(),
                symbol: "s".into(),
            },
        )),
        "empty library or symbol",
    );
    let twice = module(
        main_fn(1, vec![ecall(0, vec![cu(1)], vec![0])]),
        vec![],
        vec![
            ExternDecl::new("x", vec![U32], vec![U32], Effects::pure())
                .with_body(TargetBody::new("t", asm("a")))
                .with_body(TargetBody::new("t", asm("b"))),
        ],
    );
    refused(&twice, "two bodies for one target");
}

#[test]
fn a_reference_must_have_the_declared_signature() {
    let call = vec![ecall(0, vec![cu(4)], vec![0])];
    let good = module(
        main_fn(1, call.clone()),
        vec![inc_reference()],
        vec![inc().with_reference(1)],
    );
    assert_eq!(check_module(&good, 0), Ok(()));

    let at = |fns: Vec<KernelFn>, r: u32| {
        module(main_fn(1, call.clone()), fns, vec![inc().with_reference(r)])
    };
    refused(&at(vec![], 5), "not a function of the module");
    let mut launched = inc_reference();
    launched.launch = Some(LaunchShape::new([1, 1, 1], [1, 1, 1]));
    refused(&at(vec![launched], 1), "launch shape");
    // a parameter of the wrong type
    let mut wrong_param = inc_reference();
    wrong_param.params = vec![Param::scalar("x", F32, 0)];
    wrong_param.locals[0] = LocalDecl::scalar(F32);
    refused(&at(vec![wrong_param], 1), "parameters are not the declared");
    // a result of the wrong type
    let mut wrong_result = inc_reference();
    wrong_result.locals[1] = LocalDecl::scalar(I32);
    refused(
        &at(vec![wrong_result], 1),
        "result type is not the declared",
    );
    // no result at all
    let mut none = inc_reference();
    none.result = None;
    refused(&at(vec![none], 1), "result presence differs");
    // a reference that calls an extern is not allowed to reach itself
    let mut recursive = inc_reference();
    recursive.body.0.push(ecall(0, vec![cu(1)], vec![]));
    refused(&at(vec![recursive], 1), "itself calls an extern");
    // a tuple result has no single-result reference yet
    let tuple = ExternDecl::new("t", vec![U32], vec![U32, U32], Effects::pure()).with_reference(1);
    let m = module(
        main_fn(2, vec![ecall(0, vec![cu(1)], vec![0, 1])]),
        vec![inc_reference()],
        vec![tuple],
    );
    refused(&m, "tuple result");
}

#[test]
fn a_call_must_match_its_declaration() {
    let m = |body| module(main_fn(2, body), vec![], vec![inc()]);
    assert_eq!(
        check_module(&m(vec![ecall(0, vec![cu(1)], vec![0])]), 0),
        Ok(())
    );
    assert_eq!(
        check_module(&m(vec![ecall(3, vec![cu(1)], vec![0])]), 0),
        Err(KernelError::BadExtern { decl: 3 })
    );
    match check_module(&m(vec![ecall(0, vec![], vec![0])]), 0) {
        Err(KernelError::ExternUnsupported { what }) => {
            assert!(what.contains("number of arguments"))
        }
        other => panic!("arity: {other:?}"),
    }
    match check_module(&m(vec![ecall(0, vec![cu(1)], vec![])]), 0) {
        Err(KernelError::ExternUnsupported { what }) => assert!(what.contains("result locals")),
        other => panic!("result count: {other:?}"),
    }
    assert!(matches!(
        check_module(
            &m(vec![ecall(0, vec![ScalarExpr::ConstInt(1, I32)], vec![0])]),
            0
        ),
        Err(KernelError::TypeMismatch { what }) if what.contains("ExternCall argument")
    ));
    // a result local of another type than the declared result
    let bad_local = {
        let mut k = main_fn(1, vec![ecall(0, vec![cu(1)], vec![0])]);
        k.locals[0] = LocalDecl::scalar(I32);
        module(k, vec![], vec![inc()])
    };
    assert!(matches!(
        check_module(&bad_local, 0),
        Err(KernelError::TypeMismatch { .. })
    ));
    // the result local is declared by the call, so a second declaration of it is refused
    assert_eq!(
        check_module(
            &m(vec![
                ecall(0, vec![cu(1)], vec![0]),
                ecall(0, vec![cu(1)], vec![0])
            ]),
            0
        ),
        Err(KernelError::Redeclared { local: 0 })
    );
}

#[test]
fn a_sync_call_is_a_barrier_site() {
    let m = |body| module(main_fn(0, body), vec![], vec![sync_all()]);
    // control: in uniform control it is fine
    assert_eq!(check_module(&m(vec![ecall(0, vec![], vec![])]), 0), Ok(()));
    // under a divergent branch it is refused, as a Barrier would be
    assert_eq!(
        check_module(
            &m(vec![if_(divergent(), vec![ecall(0, vec![], vec![])])]),
            0
        ),
        Err(KernelError::BarrierDivergent)
    );
    // control: the same branch around a non-sync extern is fine
    let quiet = module(
        main_fn(1, vec![if_(divergent(), vec![ecall(0, vec![], vec![0])])]),
        vec![],
        vec![clock()],
    );
    assert_eq!(check_module(&quiet, 0), Ok(()));
}

#[test]
fn a_sync_scope_the_launch_lacks_is_refused() {
    let cluster = ExternDecl::new(
        "cl",
        vec![],
        vec![],
        Effects::opaque().with_sync(Scope::Cluster),
    );
    let m = module(
        main_fn(0, vec![ecall(0, vec![], vec![])]),
        vec![],
        vec![cluster],
    );
    assert!(matches!(
        check_module(&m, 0),
        Err(KernelError::SyncUnsupported { what }) if what.contains("sync scope")
    ));
}

#[test]
fn a_call_that_reaches_a_sync_extern_is_a_barrier_site() {
    // helper(): sync_all(); main calls it under a divergent branch
    let helper = KernelFn::new(
        "helper",
        U32,
        vec![],
        vec![],
        Block(vec![ecall(0, vec![], vec![])]),
    );
    let call = |cond| Stmt::If {
        cond,
        then: Block(vec![Stmt::Call {
            callee: 1,
            args: Vec::<CallArg>::new(),
            result: None,
        }]),
        els: Block(vec![]),
    };
    let m = |cond| {
        module(
            main_fn(0, vec![call(cond)]),
            vec![helper.clone()],
            vec![sync_all()],
        )
    };
    assert_eq!(check_module(&m(cu(1)), 0), Ok(()));
    assert_eq!(
        check_module(&m(divergent()), 0),
        Err(KernelError::BarrierDivergent)
    );
}

#[test]
fn the_results_of_a_call_that_is_not_pure_may_differ_between_threads() {
    // r = clock(); if r { barrier }: refused, the clock differs between threads
    let non_pure = module(
        main_fn(
            1,
            vec![
                ecall(0, vec![], vec![0]),
                if_(ScalarExpr::Local(0), vec![barrier()]),
            ],
        ),
        vec![],
        vec![clock()],
    );
    assert_eq!(
        check_module(&non_pure, 0),
        Err(KernelError::BarrierDivergent)
    );
    // r = inc(4); if r { barrier }: a pure function of a uniform argument is uniform
    let pure_uniform = module(
        main_fn(
            1,
            vec![
                ecall(0, vec![cu(4)], vec![0]),
                if_(ScalarExpr::Local(0), vec![barrier()]),
            ],
        ),
        vec![],
        vec![inc()],
    );
    assert_eq!(check_module(&pure_uniform, 0), Ok(()));
    // r = inc(threadIdx); if r { barrier }: a divergent argument makes the pure result divergent
    let pure_divergent = module(
        main_fn(
            1,
            vec![
                ecall(0, vec![tid()], vec![0]),
                if_(ScalarExpr::Local(0), vec![barrier()]),
            ],
        ),
        vec![],
        vec![inc()],
    );
    assert_eq!(
        check_module(&pure_divergent, 0),
        Err(KernelError::BarrierDivergent)
    );
}

#[test]
fn a_module_round_trips_through_serde_and_a_module_without_externs_still_loads() {
    let m = module(
        main_fn(1, vec![ecall(0, vec![cu(4)], vec![0]), store(0)]),
        vec![inc_reference()],
        vec![inc().with_reference(1)],
    );
    let json = serde_json::to_string(&m).unwrap();
    let back: Module = serde_json::from_str(&json).unwrap();
    assert_eq!(back, m);
    assert_eq!(check_module(&back, 0), Ok(()));
    // a module written before `externs` existed has no such key
    let mut v: serde_json::Value = serde_json::from_str(&json).unwrap();
    v.as_object_mut().unwrap().remove("externs");
    let old: Module = serde_json::from_value(v).unwrap();
    assert!(old.externs.is_empty());
}
