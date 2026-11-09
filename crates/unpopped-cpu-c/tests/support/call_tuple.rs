// SPDX-License-Identifier: MIT OR Apache-2.0
//! The modules of `kernel_call_tuple.rs` (ROADMAP R5.7 D4b, slice 1): a function that returns a `(value, index)` pair
//! (`best`, the argmax step of Triton's `tl.reduce` with a tuple combine) called by `Stmt::CallTuple`, once per element
//! in a serial loop (`argmax_loop`) and once per thread in a launch (`pairwise_launch`).

use unpopped::exec::{ExecId, LaunchShape};
use unpopped::ir::{BinaryOp, CastSpec, RoundMode, Saturation, ScalarExpr};
use unpopped::kernel::{
    Block, BufferRole, CallArg, KernelFn, LocalDecl, Module, Param, Place, Stmt,
};
use unpopped_vocab::ElementKind;

pub(crate) const F32: ElementKind = ElementKind::F32;
pub(crate) const I32: ElementKind = ElementKind::I32;
/// The elements `argmax_loop` reads.
pub(crate) const LEN: i64 = 6;
/// The threads of `pairwise_launch`.
pub(crate) const THREADS: u32 = 4;

pub(crate) fn local(i: u32) -> ScalarExpr {
    ScalarExpr::Local(i)
}

pub(crate) fn ci(v: i64) -> ScalarExpr {
    ScalarExpr::ConstInt(v, I32)
}

pub(crate) fn cmp(op: BinaryOp, a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Binary(op, Box::new(a), Box::new(b))
}

pub(crate) fn at(buffer: u32, i: ScalarExpr) -> Place {
    Place::Logical {
        buffer,
        coords: vec![i],
    }
}

fn assign(local: u32, value: ScalarExpr) -> Stmt {
    Stmt::Assign { local, value }
}

/// The locals of a function, one scalar per listed type.
fn scalars(tys: &[ElementKind]) -> Vec<LocalDecl> {
    tys.iter().map(|t| LocalDecl::scalar(*t)).collect()
}

/// The three buffers both entries use: `vals` read, `out_v` and `out_i` written.
fn buffers() -> Vec<Param> {
    vec![
        Param::buffer("vals", F32, BufferRole::Read, Some(1)),
        Param::buffer("out_v", F32, BufferRole::Write, Some(1)),
        Param::buffer("out_i", I32, BufferRole::Write, Some(1)),
    ]
}

fn store(buffer: u32, index: ScalarExpr, value: ScalarExpr) -> Stmt {
    Stmt::Store {
        place: at(buffer, index),
        value,
    }
}

/// `CallTuple` of `best` (function 1) with four scalar arguments, its two values landing in locals 4 and 5.
fn call_best(args: [ScalarExpr; 4]) -> Stmt {
    Stmt::CallTuple {
        callee: 1,
        args: args.into_iter().map(CallArg::Scalar).collect(),
        results: vec![4, 5],
    }
}

/// `best(v, i, bv, bi) -> (nv, ni)`: the pair `(v, i)` replaces `(bv, bi)` when `v` is larger, or equal with a smaller
/// index. Locals: 0 `v`, 1 `i`, 2 `bv`, 3 `bi`, 4 `nv` (the result), 5 `ni` (the extra result).
pub(crate) fn best() -> KernelFn {
    let take = || vec![assign(4, local(0)), assign(5, local(1))];
    let tie = Stmt::If {
        cond: cmp(BinaryOp::CmpEq, local(0), local(2)),
        then: Block(vec![Stmt::If {
            cond: cmp(BinaryOp::CmpLt, local(1), local(3)),
            then: Block(take()),
            els: Block(vec![]),
        }]),
        els: Block(vec![]),
    };
    KernelFn::new(
        "best",
        F32,
        vec![
            Param::scalar("v", F32, 0),
            Param::scalar("i", I32, 1),
            Param::scalar("bv", F32, 2),
            Param::scalar("bi", I32, 3),
        ],
        scalars(&[F32, I32, F32, I32, F32, I32]),
        Block(vec![
            Stmt::Let {
                local: 4,
                init: local(2),
            },
            Stmt::Let {
                local: 5,
                init: local(3),
            },
            Stmt::If {
                cond: cmp(BinaryOp::CmpGt, local(0), local(2)),
                then: Block(take()),
                els: Block(vec![tie]),
            },
        ]),
    )
    .with_result(4)
    .with_results([5])
}

/// Function 0, run unlaunched: `(bv, bi)` starts as `(vals[0], 0)` and folds `vals[1..LEN]` through `best`; buffer 1 gets
/// `bv`, buffer 2 gets `bi`. Locals: 0 `bv`, 1 `bi`, 2 `i`, 3 `v`, 4 and 5 the call's values.
pub(crate) fn argmax_loop() -> Module {
    let step = vec![
        Stmt::Load {
            local: 3,
            place: at(0, local(2)),
        },
        call_best([local(3), local(2), local(0), local(1)]),
        assign(0, local(4)),
        assign(1, local(5)),
    ];
    let main = KernelFn::new(
        "main",
        F32,
        buffers(),
        scalars(&[F32, I32, I32, F32, F32, I32]),
        Block(vec![
            Stmt::Load {
                local: 0,
                place: at(0, ci(0)),
            },
            Stmt::Let {
                local: 1,
                init: ci(0),
            },
            Stmt::For {
                local: 2,
                start: ci(1),
                end: ci(LEN),
                step: ci(1),
                body: Block(step),
            },
            store(1, ci(0), local(0)),
            store(2, ci(0), local(1)),
        ]),
    );
    Module::new(vec![main, best()])
}

/// Function 0, launched with `THREADS` threads: thread `t` folds the pair `(vals[t], t)` into `(vals[3 - t], 3 - t)`
/// through `best` and writes the two values to `out_v[t]` and `out_i[t]`. Locals: 0 `t`, 1 `u = 3 - t`, 2 `a = vals[t]`,
/// 3 `b = vals[u]`, 4 and 5 the call's values.
pub(crate) fn pairwise_launch() -> Module {
    // the thread index is a U32; the call's index parameter is an I32
    let t = ScalarExpr::Cast(
        Box::new(ScalarExpr::Exec(ExecId::ThreadIdx(0))),
        CastSpec {
            to: I32,
            round: RoundMode::NearestEven,
            sat: Saturation::Wrap,
        },
    );
    let main = KernelFn::new(
        "main",
        F32,
        buffers(),
        scalars(&[I32, I32, F32, F32, F32, I32]),
        Block(vec![
            Stmt::Let { local: 0, init: t },
            Stmt::Let {
                local: 1,
                init: ScalarExpr::Sub(Box::new(ci(i64::from(THREADS) - 1)), Box::new(local(0))),
            },
            Stmt::Load {
                local: 2,
                place: at(0, local(0)),
            },
            Stmt::Load {
                local: 3,
                place: at(0, local(1)),
            },
            call_best([local(2), local(0), local(3), local(1)]),
            store(1, local(0), local(4)),
            store(2, local(0), local(5)),
        ]),
    )
    .with_launch(LaunchShape::new([THREADS, 1, 1], [1, 1, 1]));
    Module::new(vec![main, best()])
}
