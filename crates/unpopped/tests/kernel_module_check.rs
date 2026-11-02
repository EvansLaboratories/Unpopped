// SPDX-License-Identifier: MIT OR Apache-2.0
//! `check_module` (ROADMAP U3 R1, slice M2 of `docs/options-u3-calls.md`): a `Stmt::Call` is well formed
//! only inside a `Module`: the callee exists, arity and argument kinds and types match its parameters, a
//! buffer is passed by alias with a role the caller has, the result local has the callee's result type, the
//! call graph is acyclic, and a callee is a plain function (no launch, no `Shared`/`Constant` allocation,
//! no barrier). `check_kernel` alone keeps refusing a `Call`: with no module there is nothing to resolve.

use unpopped::exec::{AllocDecl, Extent, LaunchShape, MemorySpace, Scope};
use unpopped::ir::ScalarExpr;
use unpopped::kernel::{
    Block, BufferRole, CallArg, KernelError, KernelFn, LocalDecl, Module, Param, Place, Stmt,
    check_kernel, check_module,
};
use unpopped_vocab::ElementKind;

const F32: ElementKind = ElementKind::F32;
const I32: ElementKind = ElementKind::I32;
const U32: ElementKind = ElementKind::U32;

fn zero() -> Place {
    Place::Logical {
        buffer: 0,
        coords: vec![ScalarExpr::ConstInt(0, U32)],
    }
}

/// `f(buffer x: F32 `role`, scalar k: F32)`: `x[0] = k` when `role` writes, else it only reads `x[0]`.
fn callee(role: BufferRole) -> KernelFn {
    let body = if matches!(role, BufferRole::Read) {
        Stmt::Load {
            local: 1,
            place: zero(),
        }
    } else {
        Stmt::Store {
            place: zero(),
            value: ScalarExpr::Local(0),
        }
    };
    KernelFn::new(
        "f",
        F32,
        vec![
            Param::buffer("x", F32, role, Some(1)),
            Param::scalar("k", F32, 0),
        ],
        vec![LocalDecl::scalar(F32), LocalDecl::scalar(F32)],
        Block(vec![body]),
    )
}

/// A callee with a result: `g(k: F32) -> F32`, the result local 1 set to `k`.
fn callee_with_result() -> KernelFn {
    KernelFn::new(
        "g",
        F32,
        vec![Param::scalar("k", F32, 0)],
        vec![LocalDecl::scalar(F32), LocalDecl::scalar(F32)],
        Block(vec![Stmt::Let {
            local: 1,
            init: ScalarExpr::Local(0),
        }]),
    )
    .with_result(1)
}

fn c(v: f64) -> CallArg {
    CallArg::Scalar(ScalarExpr::Const(v))
}

fn caller(role: BufferRole, locals: usize, body: Vec<Stmt>) -> KernelFn {
    let decls = (0..locals).map(|_| LocalDecl::scalar(F32)).collect();
    KernelFn::new(
        "main",
        F32,
        vec![Param::buffer("a", F32, role, Some(1))],
        decls,
        Block(body),
    )
}

fn call(callee: u32, args: Vec<CallArg>, result: Option<u32>) -> Stmt {
    Stmt::Call {
        callee,
        args,
        result,
    }
}

fn module(main: KernelFn, rest: Vec<KernelFn>) -> Module {
    let mut fns = vec![main];
    fns.extend(rest);
    Module::new(fns)
}

fn ok_call() -> Stmt {
    call(1, vec![CallArg::Buffer(0), c(2.5)], None)
}

#[test]
fn a_well_formed_call_checks_in_its_module_and_not_alone() {
    let m = module(
        caller(BufferRole::ReadWrite, 0, vec![ok_call()]),
        vec![callee(BufferRole::ReadWrite)],
    );
    assert_eq!(check_module(&m, 0), Ok(()));
    assert!(matches!(
        check_kernel(&m.fns[0]),
        Err(KernelError::CallUnsupported { .. })
    ));
}

#[test]
fn the_callee_must_exist_and_so_must_the_entry() {
    let m = module(
        caller(
            BufferRole::ReadWrite,
            0,
            vec![call(9, vec![CallArg::Buffer(0), c(1.0)], None)],
        ),
        vec![callee(BufferRole::ReadWrite)],
    );
    assert_eq!(
        check_module(&m, 0),
        Err(KernelError::BadCallee { callee: 9 })
    );
    assert_eq!(
        check_module(&m, 7),
        Err(KernelError::BadCallee { callee: 7 })
    );
}

#[test]
fn arity_and_argument_kinds_must_match_the_callee_parameters() {
    let bad = |args| {
        let m = module(
            caller(BufferRole::ReadWrite, 0, vec![call(1, args, None)]),
            vec![callee(BufferRole::ReadWrite)],
        );
        check_module(&m, 0)
    };
    assert_eq!(
        bad(vec![CallArg::Buffer(0)]),
        Err(KernelError::CallArity {
            expected: 2,
            found: 1
        })
    );
    // a scalar where the callee wants a buffer, and the reverse
    assert!(matches!(
        bad(vec![c(1.0), c(1.0)]),
        Err(KernelError::CallArgMismatch { arg: 0, .. })
    ));
    assert!(matches!(
        bad(vec![CallArg::Buffer(0), CallArg::Buffer(0)]),
        Err(KernelError::CallArgMismatch { arg: 1, .. })
    ));
    // a scalar of the wrong type
    let wrong = CallArg::Scalar(ScalarExpr::ConstInt(1, I32));
    assert!(matches!(
        bad(vec![CallArg::Buffer(0), wrong]),
        Err(KernelError::CallArgMismatch { arg: 1, .. })
    ));
    // a caller buffer that does not exist
    assert!(matches!(
        bad(vec![CallArg::Buffer(4), c(1.0)]),
        Err(KernelError::CallArgMismatch { arg: 0, .. })
    ));
}

#[test]
fn a_buffer_passed_by_alias_keeps_its_element_type_and_cannot_gain_a_role() {
    // the caller's buffer is ReadOnly: a callee that writes it would write a read-only buffer
    let m = module(
        caller(BufferRole::Read, 0, vec![ok_call()]),
        vec![callee(BufferRole::ReadWrite)],
    );
    assert!(matches!(
        check_module(&m, 0),
        Err(KernelError::CallArgMismatch { arg: 0, .. })
    ));
    // a Read callee on a ReadWrite caller buffer is fine (a role may narrow)
    let m = module(
        caller(BufferRole::ReadWrite, 0, vec![ok_call()]),
        vec![callee(BufferRole::Read)],
    );
    assert_eq!(check_module(&m, 0), Ok(()));
    // the element type must be the same
    let mut other = caller(BufferRole::ReadWrite, 0, vec![ok_call()]);
    other.params[0] = Param::buffer("a", I32, BufferRole::ReadWrite, Some(1));
    let m = module(other, vec![callee(BufferRole::ReadWrite)]);
    assert!(matches!(
        check_module(&m, 0),
        Err(KernelError::CallArgMismatch { arg: 0, .. })
    ));
}

#[test]
fn a_result_has_the_callee_result_type_and_a_void_callee_has_none() {
    let with = |ty: ElementKind, args, result| {
        let mut main = caller(BufferRole::ReadWrite, 0, vec![call(1, args, result)]);
        main.locals.push(LocalDecl::scalar(ty));
        module(main, vec![callee_with_result()])
    };
    // the result local is declared by the call, as a `Load` or an atomic's `old` declares its local
    assert_eq!(check_module(&with(F32, vec![c(1.0)], Some(0)), 0), Ok(()));
    assert_eq!(check_module(&with(F32, vec![c(1.0)], None), 0), Ok(()));
    // a void callee cannot be asked for a result
    let void = module(
        caller(
            BufferRole::ReadWrite,
            1,
            vec![call(1, vec![CallArg::Buffer(0), c(1.0)], Some(0))],
        ),
        vec![callee(BufferRole::ReadWrite)],
    );
    assert!(matches!(
        check_module(&void, 0),
        Err(KernelError::CallResult { .. })
    ));
    // the wrong declared type for the result local
    assert!(matches!(
        check_module(&with(I32, vec![c(1.0)], Some(0)), 0),
        Err(KernelError::TypeMismatch { .. })
    ));
}

#[test]
fn the_call_graph_must_be_acyclic() {
    let selfcall = {
        let mut f = callee(BufferRole::ReadWrite);
        f.body
            .0
            .push(call(1, vec![CallArg::Buffer(0), c(1.0)], None));
        f
    };
    let m = module(
        caller(BufferRole::ReadWrite, 0, vec![ok_call()]),
        vec![selfcall],
    );
    assert!(matches!(
        check_module(&m, 0),
        Err(KernelError::CallCycle { .. })
    ));
    // f (1) calls h (2), h calls f
    let mut f = callee(BufferRole::ReadWrite);
    f.body
        .0
        .push(call(2, vec![CallArg::Buffer(0), c(1.0)], None));
    let mut h = callee(BufferRole::ReadWrite);
    h.body
        .0
        .push(call(1, vec![CallArg::Buffer(0), c(1.0)], None));
    let m = module(
        caller(BufferRole::ReadWrite, 0, vec![ok_call()]),
        vec![f, h],
    );
    assert!(matches!(
        check_module(&m, 0),
        Err(KernelError::CallCycle { .. })
    ));
}

#[test]
fn a_callee_is_a_plain_function() {
    let launched =
        callee(BufferRole::ReadWrite).with_launch(LaunchShape::new([1, 1, 1], [1, 1, 1]));
    let shared = callee(BufferRole::ReadWrite).with_allocs(vec![AllocDecl::new(
        "s",
        MemorySpace::Shared,
        F32,
        Extent::Static(4),
    )]);
    let mut barrier = callee(BufferRole::ReadWrite);
    barrier.body.0.push(Stmt::barrier(Scope::Block));
    for f in [launched, shared, barrier] {
        let m = module(caller(BufferRole::ReadWrite, 0, vec![ok_call()]), vec![f]);
        assert!(
            matches!(
                check_module(&m, 0),
                Err(KernelError::CallUnsupported { .. } | KernelError::ExecWithoutLaunch)
            ),
            "{:?}",
            check_module(&m, 0)
        );
    }
}

#[test]
fn every_function_of_the_module_is_checked_not_only_the_entry() {
    let mut broken = callee(BufferRole::ReadWrite);
    broken.body.0.push(Stmt::Assign {
        local: 0,
        value: ScalarExpr::Const(1.0),
    });
    let m = module(
        caller(BufferRole::ReadWrite, 0, vec![ok_call()]),
        vec![broken],
    );
    assert!(check_module(&m, 0).is_err(), "assigning a scalar parameter");
}
