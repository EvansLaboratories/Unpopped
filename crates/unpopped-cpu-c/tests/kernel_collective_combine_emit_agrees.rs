// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The C of a combining collective agrees with the oracle** (ROADMAP R5.7 D4): `CollectiveKind::{ReduceWith, ScanWith}`
//! lowered by `unpopped_cpu_c::lower_module` as a block-level fold that calls the module's combine function per lane,
//! compiled and run with a host C compiler, and compared byte for byte with `run_launch_module`. The combines make the
//! fold order and the tie rule visible (see `crates/unpopped/tests/kernel_collective_combine.rs`); the seeded mutants of
//! the emitted C change the order, the first step or the call, and each must change the bytes.

use unpopped::exec::{ExecId, LaunchShape, Scope};
use unpopped::ir::{BinaryOp, ScalarExpr};
use unpopped::kernel::{
    Block, BufferRole, CollectiveKind, KernelFn, LocalDecl, Module, Param, ParamKind, Place, Stmt,
};
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
use kernel_run::{run_generated, same_bytes};

const I32: ElementKind = ElementKind::I32;
const N: usize = 8;
const VALS: [i32; N] = [-2, 3, -1, 4, 0, -4, 1, -3];

fn tid() -> ScalarExpr {
    ScalarExpr::Exec(ExecId::ThreadIdx(0))
}

fn at(buffer: u32) -> Place {
    Place::Logical {
        buffer,
        coords: vec![tid()],
    }
}

fn bin(op: BinaryOp, a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Binary(op, Box::new(a), Box::new(b))
}

fn abs(e: ScalarExpr) -> ScalarExpr {
    let zero = ScalarExpr::ConstInt(0, I32);
    let neg = ScalarExpr::Sub(Box::new(zero.clone()), Box::new(e.clone()));
    ScalarExpr::Select(
        Box::new(bin(BinaryOp::CmpLt, e.clone(), zero)),
        Box::new(neg),
        Box::new(e),
    )
}

fn combine_fn(name: &str, body: ScalarExpr) -> KernelFn {
    let mut f = KernelFn::new(
        name,
        I32,
        vec![Param::scalar("a", I32, 0), Param::scalar("b", I32, 1)],
        vec![LocalDecl::scalar(I32); 3],
        Block(vec![Stmt::Let {
            local: 2,
            init: body,
        }]),
    );
    f.result = Some(2);
    f
}

fn a() -> ScalarExpr {
    ScalarExpr::Local(0)
}

fn b() -> ScalarExpr {
    ScalarExpr::Local(1)
}

fn first() -> KernelFn {
    combine_fn("first", a())
}

fn last() -> KernelFn {
    combine_fn("last", b())
}

fn max_abs() -> KernelFn {
    let better = bin(BinaryOp::CmpGt, abs(b()), abs(a()));
    combine_fn(
        "max_abs",
        ScalarExpr::Select(Box::new(better), Box::new(b()), Box::new(a())),
    )
}

fn entry(kind: CollectiveKind, scope: Scope) -> KernelFn {
    KernelFn::new(
        "main",
        I32,
        vec![
            Param::buffer("vals", I32, BufferRole::Read, Some(1)),
            Param::buffer("out", I32, BufferRole::Write, Some(1)),
        ],
        vec![LocalDecl::scalar(I32), LocalDecl::scalar(I32)],
        Block(vec![
            Stmt::Load {
                local: 0,
                place: at(0),
            },
            Stmt::Collective {
                kind,
                value: ScalarExpr::Local(0),
                result: 1,
                scope,
            },
            Stmt::Store {
                place: at(1),
                value: ScalarExpr::Local(1),
            },
        ]),
    )
    .with_launch(LaunchShape::new([N as u32, 1, 1], [1, 1, 1]).with_subgroup(4, 4))
}

fn args() -> Vec<KernelArg> {
    let vals: Vec<u8> = VALS.iter().flat_map(|v| v.to_le_bytes()).collect();
    vec![
        KernelArg::Buffer(TypedBuffer::new(I32, vec![N as i64], vec![1], vals)),
        KernelArg::Buffer(TypedBuffer::new(
            I32,
            vec![N as i64],
            vec![1],
            vec![0; 4 * N],
        )),
    ]
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

fn module(kind: CollectiveKind, scope: Scope, f: KernelFn) -> Module {
    Module::new(vec![entry(kind, scope), f])
}

/// The C source of `m` (the entry's) and the oracle's output bytes.
fn oracle(m: &Module) -> Vec<Vec<u8>> {
    let mut interp = args();
    run_launch_module(m, 0, &mut interp).expect("the oracle runs it");
    written(&m.fns[0], &interp).0
}

/// Compile and run `m` with the C mutated by `mutate`; the written bytes, or `None` without a host C compiler.
fn c_bytes(tag: &str, m: &Module, mutate: impl FnOnce(String) -> String) -> Option<Vec<Vec<u8>>> {
    let g = lower_module(m, 0).unwrap_or_else(|e| panic!("{tag}: lowering refused: {e}"));
    let c = run_generated(tag, &m.fns[0], &g, &args(), mutate)?;
    assert_eq!(c.status, 0, "{tag}: the C reported status {}", c.status);
    Some(c.outs)
}

fn agree(tag: &str, m: &Module) {
    let want = oracle(m);
    let Some(got) = c_bytes(tag, m, |s| s) else {
        return;
    };
    let dtypes = [I32];
    same_bytes(tag, &got, &want, &dtypes, 0);
}

fn add() -> KernelFn {
    combine_fn("add", ScalarExpr::Add(Box::new(a()), Box::new(b())))
}

fn each_combine() -> Vec<(&'static str, KernelFn)> {
    vec![("first", first()), ("last", last()), ("max_abs", max_abs())]
}

#[test]
fn a_combining_reduce_and_scan_agree_with_the_oracle() {
    for (name, f) in each_combine() {
        for (sname, scope) in [("block", Scope::Block), ("subgroup", Scope::Subgroup)] {
            for (kname, kind) in [
                ("reduce", CollectiveKind::ReduceWith { combine: 1 }),
                ("scan", CollectiveKind::ScanWith { combine: 1 }),
            ] {
                let tag = format!("cmb_{kname}_{name}_{sname}");
                agree(&tag, &module(kind, scope, f.clone()));
            }
        }
    }
}

#[test]
fn the_corpus_catches_seeded_mutants_of_the_emitted_fold() {
    // An emitted fold step is `cs = f((int)(acc), (int)(cv1[i]), ..., &cr)` then `acc = cr`; each mutant must change the
    // bytes of a combine for which it matters, or the data could not tell a swapped, skipped or restarted step from the
    // right one. (`last` and `first` are not symmetric; `add` counts every lane once.)
    type Mutant = (
        &'static str,
        CollectiveKind,
        KernelFn,
        &'static str,
        &'static str,
    );
    let reduce = || CollectiveKind::ReduceWith { combine: 1 };
    let scan = || CollectiveKind::ScanWith { combine: 1 };
    let mutants: Vec<Mutant> = vec![
        // the arguments swapped: `last` becomes `first`
        (
            "cmb_mut_swap",
            reduce(),
            last(),
            "(int)(acc), (int)(cv1[i])",
            "(int)(cv1[i]), (int)(acc)",
        ),
        // the result of the step dropped: the accumulator never moves off lane 0
        ("cmb_mut_keep", reduce(), last(), "acc = cr1;", "acc = acc;"),
        // the accumulator started from a constant instead of lane 0 (`first` returns it)
        (
            "cmb_mut_start",
            reduce(),
            first(),
            "int acc = cv1[g1];",
            "int acc = 99;",
        ),
        // the fold starting at lane 0 instead of 1 counts lane 0 twice (a sum shows it)
        (
            "cmb_mut_twice",
            reduce(),
            add(),
            "for (int i = g1 + 1;",
            "for (int i = g1;",
        ),
        // a scan that folds lane 0 into itself
        (
            "cmb_mut_scan",
            scan(),
            add(),
            "if (i > g1) {",
            "if (i >= g1) {",
        ),
    ];
    for (tag, kind, f, from, to) in mutants {
        let m = module(kind, Scope::Block, f);
        let want = oracle(&m);
        let src = lower_module(&m, 0).expect("lowers").source;
        assert_eq!(
            src.matches(from).count(),
            1,
            "{tag}: the anchor `{from}` is in the emitted C once:
{src}"
        );
        let Some(got) = c_bytes(tag, &m, |s| s.replacen(from, to, 1)) else {
            return;
        };
        assert_ne!(got, want, "{tag}: the mutant went unnoticed");
    }
}
