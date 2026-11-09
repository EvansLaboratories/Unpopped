// SPDX-License-Identifier: MIT OR Apache-2.0
//! The modules of `kernel_collective_tuple.rs` (ROADMAP R5.7 D4b, slice 2): the two tuple combines the conformance row
//! names. `argmax_combine` carries `(value, index)` as Triton's argmax does; `welford_combine` carries
//! `(mean, m2, count)` as Chan's parallel Welford merge does. Each is the combine function (function 1) of a launched
//! 8-thread entry (function 0) whose lanes contribute one element each and receive the folded state.

use unpopped::exec::{ExecId, LaunchShape, Scope};
use unpopped::ir::{BinaryOp, CastSpec, RoundMode, Saturation, ScalarExpr};
use unpopped::kernel::{
    Block, BufferRole, CollectiveKind, KernelFn, LocalDecl, Module, Param, Place, Stmt,
};
use unpopped_vocab::ElementKind;

pub(crate) const F32: ElementKind = ElementKind::F32;
pub(crate) const I32: ElementKind = ElementKind::I32;
pub(crate) const N: usize = 8;
/// The argmax inputs: the maximum `9.0` is at lanes 5 and 6, and `7.0` at lanes 1 and 3.
pub(crate) const ARGMAX_VALS: [f32; N] = [3.0, 7.0, -1.0, 7.0, 0.0, 9.0, 9.0, 2.0];
/// The Welford inputs: mean 5, sum of squared deviations 32.
pub(crate) const WELFORD_VALS: [f32; N] = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];

pub(crate) fn local(i: u32) -> ScalarExpr {
    ScalarExpr::Local(i)
}

fn ci(v: i64) -> ScalarExpr {
    ScalarExpr::ConstInt(v, I32)
}

fn cmp(op: BinaryOp, a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Binary(op, Box::new(a), Box::new(b))
}

fn to(ty: ElementKind, e: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Cast(
        Box::new(e),
        CastSpec {
            to: ty,
            round: RoundMode::NearestEven,
            sat: Saturation::Wrap,
        },
    )
}

fn tid() -> ScalarExpr {
    ScalarExpr::Exec(ExecId::ThreadIdx(0))
}

fn at(buffer: u32) -> Place {
    Place::Logical {
        buffer,
        coords: vec![tid()],
    }
}

fn add(a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Add(Box::new(a), Box::new(b))
}

fn sub(a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Sub(Box::new(a), Box::new(b))
}

fn mul(a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Mul(Box::new(a), Box::new(b))
}

fn div(a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Div(Box::new(a), Box::new(b))
}

fn assign(local: u32, value: ScalarExpr) -> Stmt {
    Stmt::Assign { local, value }
}

/// `argmax_combine(av, ai, bv, bi) -> (rv, ri)`: the state `b` replaces the state `a` when its value is larger, or equal
/// with a smaller index. Locals: 0 `av`, 1 `ai`, 2 `bv`, 3 `bi`, 4 `rv` (the result), 5 `ri`.
pub(crate) fn argmax_combine() -> KernelFn {
    let take = || vec![assign(4, local(2)), assign(5, local(3))];
    let tie = Stmt::If {
        cond: cmp(BinaryOp::CmpEq, local(2), local(0)),
        then: Block(vec![Stmt::If {
            cond: cmp(BinaryOp::CmpLt, local(3), local(1)),
            then: Block(take()),
            els: Block(vec![]),
        }]),
        els: Block(vec![]),
    };
    KernelFn::new(
        "argmax_combine",
        F32,
        vec![
            Param::scalar("av", F32, 0),
            Param::scalar("ai", I32, 1),
            Param::scalar("bv", F32, 2),
            Param::scalar("bi", I32, 3),
        ],
        vec![
            LocalDecl::scalar(F32),
            LocalDecl::scalar(I32),
            LocalDecl::scalar(F32),
            LocalDecl::scalar(I32),
            LocalDecl::scalar(F32),
            LocalDecl::scalar(I32),
        ],
        Block(vec![
            Stmt::Let {
                local: 4,
                init: local(0),
            },
            Stmt::Let {
                local: 5,
                init: local(1),
            },
            Stmt::If {
                cond: cmp(BinaryOp::CmpGt, local(2), local(0)),
                then: Block(take()),
                els: Block(vec![tie]),
            },
        ]),
    )
    .with_result(4)
    .with_results([5])
}

/// `cross_combine(av, ai, bv, bi) -> (bv, ai)`: the value of the later state with the index of the earlier one. Unlike
/// the argmax and Welford merges it is not commutative, so a fold that swaps its arguments shows. Locals as for
/// `argmax_combine`.
pub(crate) fn cross_combine() -> KernelFn {
    let mut f = argmax_combine();
    f.name = "cross_combine".to_string();
    f.body = Block(vec![
        Stmt::Let {
            local: 4,
            init: local(2),
        },
        Stmt::Let {
            local: 5,
            init: local(1),
        },
    ]);
    f
}

/// `welford_combine(ma, m2a, na, mb, m2b, nb) -> (mean, m2, n)`: Chan's merge of two partial states, with
/// `n = na + nb`, `delta = mb - ma`, `mean = ma + delta * nb / n` and `m2 = m2a + m2b + delta^2 * na * nb / n`.
/// Locals: 0..=5 the parameters, 6 `mean` (the result), 7 `m2`, 8 `n`, 9 `n` as `F32`, 10 `delta`.
pub(crate) fn welford_combine() -> KernelFn {
    let nf = |l| to(F32, local(l));
    let delta = local(10);
    KernelFn::new(
        "welford_combine",
        F32,
        vec![
            Param::scalar("ma", F32, 0),
            Param::scalar("m2a", F32, 1),
            Param::scalar("na", I32, 2),
            Param::scalar("mb", F32, 3),
            Param::scalar("m2b", F32, 4),
            Param::scalar("nb", I32, 5),
        ],
        vec![
            LocalDecl::scalar(F32),
            LocalDecl::scalar(F32),
            LocalDecl::scalar(I32),
            LocalDecl::scalar(F32),
            LocalDecl::scalar(F32),
            LocalDecl::scalar(I32),
            LocalDecl::scalar(F32),
            LocalDecl::scalar(F32),
            LocalDecl::scalar(I32),
            LocalDecl::scalar(F32),
            LocalDecl::scalar(F32),
        ],
        Block(vec![
            Stmt::Let {
                local: 8,
                init: add(local(2), local(5)),
            },
            Stmt::Let {
                local: 9,
                init: nf(8),
            },
            Stmt::Let {
                local: 10,
                init: sub(local(3), local(0)),
            },
            Stmt::Let {
                local: 6,
                init: add(local(0), mul(delta.clone(), div(nf(5), local(9)))),
            },
            Stmt::Let {
                local: 7,
                init: add(
                    add(local(1), local(4)),
                    mul(mul(delta.clone(), delta), div(mul(nf(2), nf(5)), local(9))),
                ),
            },
        ]),
    )
    .with_result(6)
    .with_results([7, 8])
}

fn buffers(outs: &[(&str, ElementKind)]) -> Vec<Param> {
    let mut p = vec![Param::buffer("vals", F32, BufferRole::Read, Some(1))];
    p.extend(
        outs.iter()
            .map(|(n, t)| Param::buffer(*n, *t, BufferRole::Write, Some(1))),
    );
    p
}

fn launch() -> LaunchShape {
    LaunchShape::new([N as u32, 1, 1], [1, 1, 1]).with_subgroup(4, 4)
}

fn kind(scan: bool) -> CollectiveKind {
    if scan {
        CollectiveKind::ScanWith { combine: 1 }
    } else {
        CollectiveKind::ReduceWith { combine: 1 }
    }
}

/// Function 0 for the argmax: lane `t` contributes `(vals[t], t)`, receives the folded `(value, index)` and stores them
/// in `out_v[t]` and `out_i[t]`. Locals: 0 `v`, 1 `t`, 2 and 3 the state.
pub(crate) fn argmax_module(scan: bool, scope: Scope) -> Module {
    let main = KernelFn::new(
        "main",
        F32,
        buffers(&[("out_v", F32), ("out_i", I32)]),
        vec![
            LocalDecl::scalar(F32),
            LocalDecl::scalar(I32),
            LocalDecl::scalar(F32),
            LocalDecl::scalar(I32),
        ],
        Block(vec![
            Stmt::Load {
                local: 0,
                place: at(0),
            },
            Stmt::Let {
                local: 1,
                init: to(I32, tid()),
            },
            Stmt::CollectiveTuple {
                kind: kind(scan),
                values: vec![local(0), local(1)],
                results: vec![2, 3],
                scope,
            },
            Stmt::Store {
                place: at(1),
                value: local(2),
            },
            Stmt::Store {
                place: at(2),
                value: local(3),
            },
        ]),
    )
    .with_launch(launch());
    Module::new(vec![main, argmax_combine()])
}

/// Function 0 for Welford: lane `t` contributes `(vals[t], 0.0, 1)` and stores the folded `(mean, m2, count)` in
/// `out_mean[t]`, `out_m2[t]` and `out_n[t]`. Locals: 0 `x`, 1..=3 the state.
pub(crate) fn welford_module(scan: bool) -> Module {
    let main = KernelFn::new(
        "main",
        F32,
        buffers(&[("out_mean", F32), ("out_m2", F32), ("out_n", I32)]),
        vec![
            LocalDecl::scalar(F32),
            LocalDecl::scalar(F32),
            LocalDecl::scalar(F32),
            LocalDecl::scalar(I32),
        ],
        Block(vec![
            Stmt::Load {
                local: 0,
                place: at(0),
            },
            Stmt::CollectiveTuple {
                kind: kind(scan),
                values: vec![local(0), ScalarExpr::Const(0.0), ci(1)],
                results: vec![1, 2, 3],
                scope: Scope::Block,
            },
            Stmt::Store {
                place: at(1),
                value: local(1),
            },
            Stmt::Store {
                place: at(2),
                value: local(2),
            },
            Stmt::Store {
                place: at(3),
                value: local(3),
            },
        ]),
    )
    .with_launch(launch());
    Module::new(vec![main, welford_combine()])
}
