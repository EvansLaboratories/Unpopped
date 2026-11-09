// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The C of masked loads and stores agrees with the oracle** (ROADMAP R5.7 C4; the semantics are
//! `crates/unpopped/tests/kernel_masked_access.rs`): `Stmt::MaskedLoad` and `Stmt::MaskedStore` lowered by
//! `unpopped_cpu_c::CpuC::lower_kernel` as a bounds-checked access inside `if (mask != 0)`, compiled with a host C
//! compiler, run, and compared byte for byte with `run_kernel` / `run_launch`. A false mask must leave an out-of-range
//! index alone (both sides finish) and a true one must report out of bounds (both sides report it). The seeded mutants of
//! the emitted C move the guard, drop it or lose the fill, and each must change the bytes or the status.

use unpopped::exec::{ExecId, LaunchShape};
use unpopped::ir::{BinaryOp, ScalarExpr};
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped_cpu_c::CpuC;
use unpopped_vocab::ElementKind;

#[path = "support/cc.rs"]
mod cc;
// The shared compile-and-run drivers; this file uses a subset of them.
#[allow(dead_code)]
#[path = "support/kernel_run.rs"]
mod kernel_run;
#[allow(dead_code)]
#[path = "support/launch_run.rs"]
mod launch_run;
use kernel_run::{agree, run_c_mutated};
use launch_run::agree_launch;

const F32: ElementKind = ElementKind::F32;
const I32: ElementKind = ElementKind::I32;
const U32: ElementKind = ElementKind::U32;
const IN_LEN: usize = 5;
const OUT_LEN: usize = 8;

fn ci(v: i64) -> ScalarExpr {
    ScalarExpr::ConstInt(v, I32)
}

fn at(buffer: u32, i: ScalarExpr) -> Place {
    Place::Logical {
        buffer,
        coords: vec![i],
    }
}

fn floats(v: &[f32]) -> KernelArg {
    let bytes = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    KernelArg::Buffer(TypedBuffer::new(F32, vec![v.len() as i64], vec![1], bytes))
}

fn args() -> Vec<KernelArg> {
    vec![
        floats(&[10.0, 11.0, 12.0, 13.0, 14.0]),
        floats(&[0.0; OUT_LEN]),
    ]
}

fn in_range() -> ScalarExpr {
    ScalarExpr::Binary(
        BinaryOp::CmpLt,
        Box::new(ScalarExpr::Local(0)),
        Box::new(ci(IN_LEN as i64)),
    )
}

/// `for i in 0..8 { out[i] = masked_load(in[i], mask, fill) }`.
fn load_loop(mask: ScalarExpr, fill: ScalarExpr) -> KernelFn {
    KernelFn::new(
        "mload",
        F32,
        vec![
            Param::buffer("in", F32, BufferRole::Read, Some(1)),
            Param::buffer("out", F32, BufferRole::Write, Some(1)),
        ],
        vec![LocalDecl::scalar(I32), LocalDecl::scalar(F32)],
        Block(vec![Stmt::For {
            local: 0,
            start: ci(0),
            end: ci(OUT_LEN as i64),
            step: ci(1),
            body: Block(vec![
                Stmt::MaskedLoad {
                    local: 1,
                    place: at(0, ScalarExpr::Local(0)),
                    mask,
                    fill,
                },
                Stmt::Store {
                    place: at(1, ScalarExpr::Local(0)),
                    value: ScalarExpr::Local(1),
                },
            ]),
        }]),
    )
}

/// `for i in 0..8 { masked_store(out[i] = 9.0, mask) }` over a buffer of `IN_LEN` elements.
fn store_loop(mask: ScalarExpr) -> KernelFn {
    KernelFn::new(
        "mstore",
        F32,
        vec![Param::buffer("out", F32, BufferRole::Write, Some(1))],
        vec![LocalDecl::scalar(I32)],
        Block(vec![Stmt::For {
            local: 0,
            start: ci(0),
            end: ci(OUT_LEN as i64),
            step: ci(1),
            body: Block(vec![Stmt::MaskedStore {
                place: at(0, ScalarExpr::Local(0)),
                value: ScalarExpr::Const(9.0),
                mask,
            }]),
        }]),
    )
}

#[test]
fn a_masked_load_agrees_for_each_fill_and_reports_a_true_out_of_range_index() {
    for (tag, fill) in [
        ("mload_neg", ScalarExpr::Const(-7.5)),
        ("mload_zero", ScalarExpr::Const(0.0)),
        ("mload_inf", ScalarExpr::Const(f64::INFINITY)),
        (
            "mload_expr",
            ScalarExpr::Cast(
                Box::new(ScalarExpr::Local(0)),
                unpopped::ir::CastSpec {
                    to: F32,
                    round: unpopped::ir::RoundMode::NearestEven,
                    sat: unpopped::ir::Saturation::Wrap,
                },
            ),
        ),
    ] {
        agree(tag, &load_loop(in_range(), fill), args());
    }
    // a mask that is true past the end: both report out of bounds (status 1)
    agree(
        "mload_oob",
        &load_loop(ci(1), ScalarExpr::Const(0.0)),
        args(),
    );
}

#[test]
fn a_masked_store_agrees_and_leaves_a_false_out_of_range_index_alone() {
    agree(
        "mstore",
        &store_loop(in_range()),
        vec![floats(&[0.0; IN_LEN])],
    );
    agree(
        "mstore_oob",
        &store_loop(ci(1)),
        vec![floats(&[0.0; IN_LEN])],
    );
}

/// A launched kernel: thread `t` of 8 does `out[t] = masked_load(in[t], t < 5, -3.0)`.
fn launched_load() -> KernelFn {
    let tid = ScalarExpr::Exec(ExecId::ThreadIdx(0));
    KernelFn::new(
        "mload_launch",
        F32,
        vec![
            Param::buffer("in", F32, BufferRole::Read, Some(1)),
            Param::buffer("out", F32, BufferRole::Write, Some(1)),
        ],
        vec![LocalDecl::scalar(F32)],
        Block(vec![
            Stmt::MaskedLoad {
                local: 0,
                place: at(0, tid.clone()),
                mask: ScalarExpr::Binary(
                    BinaryOp::CmpLt,
                    Box::new(tid.clone()),
                    Box::new(ScalarExpr::ConstInt(IN_LEN as i64, U32)),
                ),
                fill: ScalarExpr::Const(-3.0),
            },
            Stmt::MaskedStore {
                place: at(1, tid.clone()),
                value: ScalarExpr::Local(0),
                mask: ScalarExpr::ConstInt(1, U32),
            },
        ]),
    )
    .with_launch(LaunchShape::new([OUT_LEN as u32, 1, 1], [1, 1, 1]))
}

#[test]
fn a_launched_masked_load_and_store_agree() {
    agree_launch("mload_launch", &launched_load(), args());
}

#[test]
fn the_corpus_catches_seeded_mutants_of_the_emitted_guard() {
    // The emitted load is `T mv<n> = (T)(fill); if ((mask) != 0) { <bounds-checked address>; mv<n> = (T)(at); }`.
    let k = load_loop(in_range(), ScalarExpr::Const(-7.5));
    let g = <CpuC as unpopped::backend::Backend>::lower_kernel(&CpuC, &k).expect("lowers");
    let src = g.source.clone();
    let anchors = ["if ((", ") != 0) {", "mv"];
    for a in anchors {
        assert!(
            src.contains(a),
            "the anchor `{a}` is in the emitted C:\n{src}"
        );
    }
    let want = {
        let mut interp = args();
        unpopped::oracle::kernel_eval::run_kernel(&k, &mut interp).expect("the oracle runs it");
        let KernelArg::Buffer(b) = &interp[1] else {
            unreachable!()
        };
        b.bytes.clone()
    };
    // each mutant edits the C of the first masked load only
    type Mutant = (&'static str, &'static str, &'static str);
    let mutants: [Mutant; 2] = [
        // the guard always taken: the false lanes read out of bounds (status 1, no bytes to compare)
        ("mask_always_true", ") != 0) {", ") != 0 || 1) {"),
        // the guard never taken: every lane keeps the fill
        ("mask_never_true", ") != 0) {", ") != 0 && 0) {"),
    ];
    for (tag, from, to) in mutants {
        let Some(ran) = run_c_mutated(tag, &k, &args(), |s| s.replacen(from, to, 1)) else {
            return;
        };
        let got = ran.outs.first().cloned().unwrap_or_default();
        assert!(
            ran.status != 0 || got != want,
            "{tag}: the mutant went unnoticed"
        );
    }
}
