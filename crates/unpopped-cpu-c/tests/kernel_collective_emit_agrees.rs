// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The serialized thread-loop C agrees with the launch interpreter on collectives** (ROADMAP U5 P5; PM ruling
//! 2026-11-02 (4)): a launched `KernelFn` holding a `Stmt::Collective` is lowered by `CpuC::lower_kernel` to C in
//! which the collective is a block-level construct between two thread loops, compiled with a real host C compiler,
//! run, and its written buffer compared byte for byte with `oracle::kernel_launch::run_launch`.
//!
//! One 8-thread block with a subgroup of 4 (two subgroups; the oracle's default subgroup is the launch's minimum, which
//! is 4 here). The integer input has negatives and repeats; the float input has a NaN in lane 2 (subgroup 0 only), so a
//! `Max`/`Min`/`Sum` that mishandles NaN differs in one subgroup and not the other.

use unpopped::backend::Backend;
use unpopped::exec::{ExecId, LaunchShape, Scope};
use unpopped::ir::{ReduceOp, ScalarExpr};
use unpopped::kernel::{
    Block, BufferRole, CollectiveKind, KernelFn, LocalDecl, Param, Place, ShuffleMode, Stmt,
    VoteKind, check_kernel,
};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped_cpu_c::CpuC;
use unpopped_vocab::ElementKind;

#[path = "support/cc.rs"]
mod cc;
#[path = "support/kernel_data.rs"]
mod kernel_data;
// The shared compile-and-run drivers; this file uses a subset of them.
#[allow(dead_code)]
#[path = "support/kernel_run.rs"]
mod kernel_run;
#[allow(dead_code)]
#[path = "support/launch_run.rs"]
mod launch_run;
use kernel_data::zeroed;
use kernel_run::run_c_mutated;
use launch_run::agree_launch;

const I32: ElementKind = ElementKind::I32;
const U32: ElementKind = ElementKind::U32;
const F32: ElementKind = ElementKind::F32;
const N: i64 = 8;

fn tid() -> ScalarExpr {
    ScalarExpr::Exec(ExecId::ThreadIdx(0))
}

fn at_tid(buffer: u32) -> Place {
    Place::Logical {
        buffer,
        coords: vec![tid()],
    }
}

/// `v = in[tid]; r = <collective>(v); out[tid] = r`.
fn kernel(kind: CollectiveKind, scope: Scope, ty: ElementKind, out: ElementKind) -> KernelFn {
    let body = vec![
        Stmt::Load {
            local: 0,
            place: at_tid(0),
        },
        Stmt::Collective {
            kind,
            value: ScalarExpr::Local(0),
            result: 1,
            scope,
        },
        Stmt::Store {
            place: at_tid(1),
            value: ScalarExpr::Local(1),
        },
    ];
    KernelFn::new(
        "coll",
        if ty == F32 { F32 } else { I32 },
        vec![
            Param::buffer("in", ty, BufferRole::Read, Some(1)),
            Param::buffer("out", out, BufferRole::Write, Some(1)),
        ],
        vec![LocalDecl::scalar(ty), LocalDecl::scalar(out)],
        Block(body),
    )
    .with_launch(LaunchShape::new([8, 1, 1], [1, 1, 1]).with_subgroup(4, 4))
}

fn int_args(out: ElementKind) -> Vec<KernelArg> {
    let vals: [i32; 8] = [3, -1, 4, -1, 5, -9, 2, 6];
    let bytes = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
    vec![
        KernelArg::Buffer(TypedBuffer::new(I32, vec![N], vec![1], bytes)),
        KernelArg::Buffer(zeroed(out, &[N])),
    ]
}

fn float_args() -> Vec<KernelArg> {
    let vals: [f32; 8] = [1.5, -2.0, f32::NAN, 4.0, 0.5, -0.25, 8.0, 2.0];
    let bytes = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
    vec![
        KernelArg::Buffer(TypedBuffer::new(F32, vec![N], vec![1], bytes)),
        KernelArg::Buffer(zeroed(F32, &[N])),
    ]
}

/// A vote's input: zeros in an asymmetric pattern, so a mask with its bits reversed or a vote that miscounts lanes
/// differs (subgroup 0 is `1 0 2 0`, mask `0b0101`; subgroup 1 is `0 0 7 3`, mask `0b1100`).
fn vote_args() -> Vec<KernelArg> {
    let vals: [i32; 8] = [1, 0, 2, 0, 0, 0, 7, 3];
    let bytes = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
    vec![
        KernelArg::Buffer(TypedBuffer::new(I32, vec![N], vec![1], bytes)),
        KernelArg::Buffer(zeroed(U32, &[N])),
    ]
}

fn lane(v: i64) -> ScalarExpr {
    ScalarExpr::ConstInt(v, I32)
}

fn check(tag: &str, k: &KernelFn, args: Vec<KernelArg>) {
    assert_eq!(check_kernel(k), Ok(()), "{tag}: well formed");
    agree_launch(tag, k, args);
}

#[test]
fn integer_reductions_and_scans_agree() {
    for scope in [Scope::Block, Scope::Subgroup] {
        for op in [
            ReduceOp::Sum,
            ReduceOp::Max,
            ReduceOp::Min,
            ReduceOp::And,
            ReduceOp::Or,
            ReduceOp::Xor,
        ] {
            let k = kernel(CollectiveKind::Reduce(op), scope, I32, I32);
            check(&format!("red_{op:?}_{scope:?}"), &k, int_args(I32));
            let k = kernel(
                CollectiveKind::Scan {
                    op,
                    exclusive: false,
                },
                scope,
                I32,
                I32,
            );
            check(&format!("scan_{op:?}_{scope:?}"), &k, int_args(I32));
        }
        for op in [ReduceOp::Sum, ReduceOp::And, ReduceOp::Or, ReduceOp::Xor] {
            let kind = CollectiveKind::Scan {
                op,
                exclusive: true,
            };
            check(
                &format!("xscan_{op:?}_{scope:?}"),
                &kernel(kind, scope, I32, I32),
                int_args(I32),
            );
        }
    }
}

#[test]
fn float_reductions_and_scans_agree_with_a_nan_in_one_subgroup() {
    for scope in [Scope::Block, Scope::Subgroup] {
        for op in [ReduceOp::Sum, ReduceOp::Max, ReduceOp::Min] {
            let k = kernel(CollectiveKind::Reduce(op), scope, F32, F32);
            check(&format!("fred_{op:?}_{scope:?}"), &k, float_args());
            let kind = CollectiveKind::Scan {
                op,
                exclusive: false,
            };
            let k = kernel(kind, scope, F32, F32);
            check(&format!("fscan_{op:?}_{scope:?}"), &k, float_args());
        }
        let kind = CollectiveKind::Scan {
            op: ReduceOp::Sum,
            exclusive: true,
        };
        check(
            &format!("fxscan_{scope:?}"),
            &kernel(kind, scope, F32, F32),
            float_args(),
        );
    }
}

#[test]
fn shuffles_and_broadcasts_agree_including_lanes_past_the_scope() {
    for scope in [Scope::Block, Scope::Subgroup] {
        for mode in [
            ShuffleMode::Idx,
            ShuffleMode::Up,
            ShuffleMode::Down,
            ShuffleMode::Xor,
        ] {
            for operand in [0, 1, 3, 9] {
                let kind = CollectiveKind::Shuffle {
                    mode,
                    operand: lane(operand),
                };
                let tag = format!("shuf_{mode:?}_{operand}_{scope:?}");
                check(&tag, &kernel(kind, scope, I32, I32), int_args(I32));
            }
        }
        for l in [0, 2, 6, 11] {
            let kind = CollectiveKind::Broadcast { lane: lane(l) };
            let tag = format!("bcast_{l}_{scope:?}");
            check(&tag, &kernel(kind, scope, I32, I32), int_args(I32));
        }
    }
}

#[test]
fn votes_and_the_ballot_mask_agree() {
    for (v, scope) in [
        (VoteKind::All, Scope::Block),
        (VoteKind::Any, Scope::Block),
        (VoteKind::All, Scope::Subgroup),
        (VoteKind::Any, Scope::Subgroup),
        (VoteKind::Ballot, Scope::Subgroup),
    ] {
        let k = kernel(CollectiveKind::Vote(v), scope, I32, U32);
        check(&format!("vote_{v:?}_{scope:?}"), &k, vote_args());
    }
    // a float predicate: a NaN is non-zero
    let k = kernel(
        CollectiveKind::Vote(VoteKind::Any),
        Scope::Subgroup,
        F32,
        U32,
    );
    let mut args = float_args();
    args[1] = KernelArg::Buffer(zeroed(U32, &[N]));
    check("vote_float", &k, args);
}

#[test]
fn the_c_combine_is_what_the_differential_is_sensitive_to() {
    // a seeded mutant of the generated C: the Sum's `+` turned into `-` must change an answer
    let k = kernel(
        CollectiveKind::Reduce(ReduceOp::Sum),
        Scope::Block,
        F32,
        F32,
    );
    let g = CpuC.lower_kernel(&k).expect("a collective lowers");
    assert!(g.source.contains(" + "), "the fold is spelled with a C `+`");
    let honest = run_c_mutated("coll_honest", &k, &float_args(), |s| s).expect("a C compiler");
    let mutant = run_c_mutated("coll_mutant", &k, &float_args(), |s| {
        s.replace(" + ", " - ")
    });
    if let Some(m) = mutant {
        assert_ne!(
            m.outs, honest.outs,
            "the answer must not survive a `+` that became `-`"
        );
    }
}
