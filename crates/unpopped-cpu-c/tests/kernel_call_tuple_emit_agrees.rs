// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The C of a call with several results agrees with the oracle** (ROADMAP R5.7 D4b, slice 1; the semantics are
//! `crates/unpopped/tests/kernel_call_tuple.rs`): `KernelFn::results` and `Stmt::CallTuple` lowered by
//! `unpopped_cpu_c::lower_module` as a C function with an out-pointer per value (`res`, `res1`, ...) and a call that
//! binds each to its local, compiled with a host C compiler, run, and compared byte for byte with `run_module` and
//! `run_launch_module`. The seeded mutants of the emitted C drop an out-pointer store or bind the wrong value, and each
//! must change the bytes.

use unpopped::kernel::{KernelFn, Module};
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
// The fixture shared with the oracle's test; each file uses part of it.
#[allow(dead_code)]
#[path = "support/call_tuple.rs"]
mod fixture;
use fixture::{F32, I32, argmax_loop, pairwise_launch};
use kernel_run::{run_generated, same_bytes};

fn floats(v: &[f32]) -> KernelArg {
    let bytes = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    KernelArg::Buffer(TypedBuffer::new(F32, vec![v.len() as i64], vec![1], bytes))
}

fn zeros(elem: ElementKind, n: usize) -> KernelArg {
    KernelArg::Buffer(TypedBuffer::new(
        elem,
        vec![n as i64],
        vec![1],
        vec![0; 4 * n],
    ))
}

fn argmax_args() -> Vec<KernelArg> {
    vec![
        floats(&[1.5, 4.0, -2.0, 4.0, 3.0, 0.5]),
        zeros(F32, 1),
        zeros(I32, 1),
    ]
}

fn pairwise_args() -> Vec<KernelArg> {
    vec![floats(&[2.0, 5.0, 5.0, 1.0]), zeros(F32, 4), zeros(I32, 4)]
}

/// The written buffers of `args` after a run, with their dtypes.
fn written(k: &KernelFn, args: &[KernelArg]) -> (Vec<Vec<u8>>, Vec<ElementKind>) {
    use unpopped::kernel::{BufferRole, ParamKind};
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

/// Lower and run `m` as C and as the oracle; the written buffers must agree.
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

#[test]
fn a_two_value_call_agrees_in_a_loop_and_in_a_launch() {
    agree_mutated("tuple_loop", &argmax_loop(), argmax_args(), |s| s);
    agree_mutated("tuple_launch", &pairwise_launch(), pairwise_args(), |s| s);
}

/// Whether the differential notices `mutate` applied to the generated C of `m`.
fn caught(tag: &str, m: &Module, args: Vec<KernelArg>, mutate: fn(String) -> String) -> bool {
    std::panic::catch_unwind(|| agree_mutated(tag, m, args, mutate)).is_err()
}

#[test]
fn seeded_mutants_of_the_emitted_c_are_noticed() {
    let m = argmax_loop();
    // anchors from the emitted C: a callee stores `*res = l4; *res1 = l5;`, the caller binds `l4 = r3`, `l5 = r3_1`
    let g = lower_module(&m, 0).expect("lowers");
    for a in ["*res1 = l5;", "int l5 = (int)(r3_1);", "&r3, &r3_1"] {
        assert!(g.source.contains(a), "the anchor `{a}` is in the C");
    }
    // the second value is never stored through its pointer
    let drop_store = |s: String| s.replacen("*res1 = l5;", "", 1);
    assert!(
        caught("mut_store", &m, argmax_args(), drop_store),
        "the extra result's store"
    );
    // the caller binds the extra local to the running best index instead of the call's second value
    let wrong_bind = |s: String| s.replacen("int l5 = (int)(r3_1);", "int l5 = (int)(l1);", 1);
    assert!(
        caught("mut_bind", &m, argmax_args(), wrong_bind),
        "the extra result's binding"
    );
    // the caller binds the first local to the running best value instead of the call's first value
    let wrong_first =
        |s: String| s.replacen("float l4 = (float)(r3);", "float l4 = (float)(l0);", 1);
    assert!(
        caught("mut_first", &m, argmax_args(), wrong_first),
        "the first result's binding"
    );
}
