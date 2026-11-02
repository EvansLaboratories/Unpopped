// SPDX-License-Identifier: MIT OR Apache-2.0
//! `Stmt::Call` and `Module` (0.41.0, PM ruling 2026-11-02 on `docs/options-u3-remainder.md`, R1a): the
//! callee is a `KernelFn` in a `Module`, and the shape rides the one `Stmt` break. Nothing checks or runs
//! a call yet: `check_kernel` refuses it with a typed error, so no producer builds a kernel an interpreter
//! would have to guess about, and the later slice that lands the semantics only lifts that refusal.

use unpopped::ir::ScalarExpr;
use unpopped::kernel::{Block, CallArg, KernelError, KernelFn, Module, Param, Stmt, check_kernel};
use unpopped_vocab::ElementKind;

const I32: ElementKind = ElementKind::I32;

fn call() -> Stmt {
    Stmt::Call {
        callee: 1,
        args: vec![
            CallArg::Scalar(ScalarExpr::ConstInt(7, I32)),
            CallArg::Buffer(0),
            CallArg::Alloc(1),
        ],
        result: None,
    }
}

fn kernel(body: Vec<Stmt>) -> KernelFn {
    KernelFn::new("k", I32, Vec::<Param>::new(), vec![], Block(body))
}

#[test]
fn a_call_is_refused_with_a_typed_error_until_its_semantics_land() {
    let r = check_kernel(&kernel(vec![call()]));
    assert!(
        matches!(r, Err(KernelError::CallUnsupported { .. })),
        "{r:?}"
    );
}

#[test]
fn the_same_kernel_without_the_call_checks() {
    assert_eq!(check_kernel(&kernel(vec![])), Ok(()));
}

#[test]
fn a_module_names_its_callees_by_index_and_round_trips() {
    let m = Module::new(vec![kernel(vec![call()]), kernel(vec![])]);
    assert_eq!(m.fns.len(), 2);
    let text = serde_json::to_string(&m).expect("a module serializes");
    let back: Module = serde_json::from_str(&text).expect("and reads back");
    assert_eq!(back, m);
    let Stmt::Call {
        callee,
        args,
        result,
    } = &back.fns[0].body.0[0]
    else {
        panic!("the call survived the round trip");
    };
    assert_eq!((*callee, args.len(), *result), (1, 3, None));
    assert_eq!(args[1], CallArg::Buffer(0));
    assert_eq!(args[2], CallArg::Alloc(1));
}

/// A callee's result is a local of the callee (`KernelFn::result`), the value it holds at `Return` or at the
/// end; an older serialized kernel without the field reads back with no result.
#[test]
fn a_callee_names_its_result_local_and_the_field_defaults_to_none() {
    let f = KernelFn::new("f", I32, Vec::<Param>::new(), vec![], Block(vec![])).with_result(0);
    assert_eq!(f.result, Some(0));
    let back: KernelFn = serde_json::from_str(&serde_json::to_string(&f).unwrap()).unwrap();
    assert_eq!(back, f);
    let mut v: serde_json::Value = serde_json::to_value(&f).unwrap();
    v.as_object_mut().unwrap().remove("result");
    let old: KernelFn = serde_json::from_value(v).expect("a kernel without `result` still reads");
    assert_eq!(old.result, None);
}
