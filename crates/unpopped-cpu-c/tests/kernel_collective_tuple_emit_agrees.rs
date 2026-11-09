// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The C of a tuple collective agrees with the oracle** (ROADMAP R5.7 D4b, slice 2; the semantics are
//! `crates/unpopped/tests/kernel_collective_tuple.rs`): `Stmt::CollectiveTuple` lowered by `unpopped_cpu_c::lower_module`
//! as a block-level fold that calls the module's tuple combine per lane (the state passed component by component, the new
//! state returned through one pointer per component), compiled and run with a host C compiler, and compared byte for byte
//! with `run_launch_module`. The seeded mutants of the emitted C drop a component's update, start the fold wrong or count
//! lane 0 twice, and each must change the bytes.

use unpopped::exec::Scope;
use unpopped::kernel::{BufferRole, KernelFn, Module, ParamKind};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch_module;
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
#[path = "support/collective_tuple.rs"]
mod fixture;
use fixture::{ARGMAX_VALS, F32, I32, N, WELFORD_VALS, argmax_module, welford_module};
use kernel_run::{run_generated, same_bytes};

fn buffer(elem: ElementKind, bytes: Vec<u8>) -> KernelArg {
    KernelArg::Buffer(TypedBuffer::new(elem, vec![N as i64], vec![1], bytes))
}

fn input(vals: &[f32; N]) -> KernelArg {
    buffer(F32, vals.iter().flat_map(|x| x.to_le_bytes()).collect())
}

fn zeros(elem: ElementKind) -> KernelArg {
    buffer(elem, vec![0; 4 * N])
}

fn argmax_args() -> Vec<KernelArg> {
    vec![input(&ARGMAX_VALS), zeros(F32), zeros(I32)]
}

fn welford_args() -> Vec<KernelArg> {
    vec![input(&WELFORD_VALS), zeros(F32), zeros(F32), zeros(I32)]
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

/// The oracle's output bytes and dtypes.
fn oracle(m: &Module, args: Vec<KernelArg>) -> (Vec<Vec<u8>>, Vec<ElementKind>) {
    let mut interp = args;
    run_launch_module(m, 0, &mut interp).expect("the oracle runs it");
    written(&m.fns[0], &interp)
}

/// Compile and run `m` with the C mutated by `mutate`; the written bytes, or `None` without a host C compiler.
fn c_bytes(
    tag: &str,
    m: &Module,
    args: Vec<KernelArg>,
    mutate: impl FnOnce(String) -> String,
) -> Option<Vec<Vec<u8>>> {
    let g = lower_module(m, 0).unwrap_or_else(|e| panic!("{tag}: lowering refused: {e}"));
    let c = run_generated(tag, &m.fns[0], &g, &args, mutate)?;
    assert_eq!(c.status, 0, "{tag}: the C reported status {}", c.status);
    Some(c.outs)
}

/// The C and the oracle agree, the `F32` outputs within `tol` ulps (the C evaluates a float expression in its own
/// way, so Welford's chain of products may differ in the last bit; the argmax is exact).
fn agree(tag: &str, m: &Module, args: impl Fn() -> Vec<KernelArg>, tol: u64) {
    let (want, dtypes) = oracle(m, args());
    let Some(got) = c_bytes(tag, m, args(), |s| s) else {
        return;
    };
    same_bytes(tag, &got, &want, &dtypes, tol);
}

#[test]
fn a_tuple_reduce_and_scan_agree_with_the_oracle() {
    for (sname, scope) in [("block", Scope::Block), ("subgroup", Scope::Subgroup)] {
        for (kname, scan) in [("reduce", false), ("scan", true)] {
            agree(
                &format!("tup_argmax_{kname}_{sname}"),
                &argmax_module(scan, scope),
                argmax_args,
                0,
            );
        }
    }
    for (kname, scan) in [("reduce", false), ("scan", true)] {
        agree(
            &format!("tup_welford_{kname}"),
            &welford_module(scan),
            welford_args,
            2,
        );
    }
}

/// A seeded mutant: its name, the module and its inputs, and the emitted text to replace and what replaces it.
struct Mutant {
    tag: &'static str,
    module: Module,
    args: fn() -> Vec<KernelArg>,
    from: &'static str,
    to: &'static str,
}

fn mutants() -> Vec<Mutant> {
    let argmax = |scan| argmax_module(scan, Scope::Block);
    vec![
        // the index never follows the value: the accumulator's second component stays lane 0's
        Mutant {
            tag: "tup_mut_second",
            module: argmax(false),
            args: argmax_args,
            from: "acc1_1 = cr1_1;",
            to: "acc1_1 = acc1_1;",
        },
        // the value never follows: the first component stays lane 0's
        Mutant {
            tag: "tup_mut_first",
            module: argmax(false),
            args: argmax_args,
            from: "acc1_0 = cr1_0;",
            to: "acc1_0 = acc1_0;",
        },
        // the second component starts from a constant instead of lane 0's (a scan shows lane 0's own state)
        Mutant {
            tag: "tup_mut_start",
            module: argmax(true),
            args: argmax_args,
            from: "int acc1_1 = cv1_1[g1];",
            to: "int acc1_1 = 99;",
        },
        // the fold starting at lane 0 instead of 1 counts lane 0 twice (the Welford count shows it)
        Mutant {
            tag: "tup_mut_twice",
            module: welford_module(false),
            args: welford_args,
            from: "for (int i = g1 + 1;",
            to: "for (int i = g1;",
        },
        // a scan that folds lane 0 into itself
        Mutant {
            tag: "tup_mut_scan",
            module: welford_module(true),
            args: welford_args,
            from: "if (i > g1) {",
            to: "if (i >= g1) {",
        },
        // the state's `b` half is the accumulator again instead of the lane's value
        Mutant {
            tag: "tup_mut_rhs",
            module: welford_module(false),
            args: welford_args,
            from: "(int)(cv1_2[i])",
            to: "(int)(acc1_2)",
        },
    ]
}

#[test]
fn the_corpus_catches_seeded_mutants_of_the_emitted_fold() {
    for Mutant {
        tag,
        module,
        args,
        from,
        to,
    } in mutants()
    {
        let (want, _) = oracle(&module, args());
        let src = lower_module(&module, 0).expect("lowers").source;
        assert_eq!(
            src.matches(from).count(),
            1,
            "{tag}: the anchor `{from}` is in the emitted C once:\n{src}"
        );
        let Some(got) = c_bytes(tag, &module, args(), |s| s.replacen(from, to, 1)) else {
            return;
        };
        assert_ne!(got, want, "{tag}: the mutant went unnoticed");
    }
}
