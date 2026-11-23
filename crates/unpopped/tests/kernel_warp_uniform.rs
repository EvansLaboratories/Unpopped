// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Warp-uniform control** (PM TASK 2026-11-23, `kernel_uniform.rs`): a `Subgroup`-scope `Collective` may sit under control that is
//! uniform within each subgroup (`warp = tid >> 5; if (warp == 0) { shuffle }`, the block-reduce idiom of 15 Baracuda kernels),
//! because only its own subgroup must reach it. Every condition below is a test:
//!
//! 1. warp-uniform ONLY when the launch STATES one subgroup size (`with_subgroup(N, N)`) and the shift or divisor is exactly
//!    `2^k == N`, with the block's x extent a multiple of `N`; one mutant per case (`tid >> 3` with `N = 4`, `tid >> 1`, a range
//!    `(1, 8)`, a block of 6);
//! 2. a `Block` barrier is NEVER accepted under warp-uniform control, nor a `Block` collective;
//! 3. only `Subgroup` collectives (shuffle, vote, reduce) sit there, and the oracle's reading with ONE subgroup taking part is shown;
//! 4. the forms: `tid >> k`, `tid / 2^k`, `warp == c`, `warp < c`; NOT `lane == c` (`tid & (N-1)`, or `LaneId`);
//! 5. the oracle runs the launch at the stated subgroup size.

use unpopped::exec::{ExecId, LaunchShape, Scope};
use unpopped::ir::{BinaryOp, CastSpec, ReduceOp, RoundMode, Saturation, ScalarExpr};
use unpopped::kernel::{
    Block, BufferRole, CollectiveKind, KernelError, KernelFn, LocalDecl, Param, Place, ShuffleMode,
    Stmt, Unroll, VoteKind, check_kernel,
};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_vocab::ElementKind;

const I32: ElementKind = ElementKind::I32;
const U32: ElementKind = ElementKind::U32;
/// The stated subgroup size of the launches below.
const N: u32 = 4;

fn tid_u32() -> ScalarExpr {
    ScalarExpr::Exec(ExecId::ThreadIdx(0))
}

fn cast(e: ScalarExpr, to: ElementKind) -> ScalarExpr {
    ScalarExpr::Cast(
        Box::new(e),
        CastSpec {
            to,
            round: RoundMode::NearestEven,
            sat: Saturation::Wrap,
        },
    )
}

fn ci(v: i64) -> ScalarExpr {
    ScalarExpr::ConstInt(v, I32)
}

fn bin(op: BinaryOp, a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Binary(op, Box::new(a), Box::new(b))
}

fn add(a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Add(Box::new(a), Box::new(b))
}

fn div(a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Div(Box::new(a), Box::new(b))
}

const TID: u32 = 0;
const WARP: u32 = 1;
const VALUE: u32 = 2;
const RESULT: u32 = 3;

fn tid() -> ScalarExpr {
    ScalarExpr::Local(TID)
}

fn warp() -> ScalarExpr {
    ScalarExpr::Local(WARP)
}

/// `tid = (int)threadIdx.x; warp = <warp_def>; if (<cond>) { value = tid + 1; result = <collective>(value); out[tid] = result; }`
/// in a launch of `block` threads whose subgroup range is `subgroup`.
fn kernel(
    block: u32,
    subgroup: (u32, u32),
    warp_def: ScalarExpr,
    cond: ScalarExpr,
    kind: CollectiveKind,
    scope: Scope,
) -> KernelFn {
    // a vote's result is a `U32` mask or flag; every other collective here keeps the `I32` of its value
    let out = if matches!(kind, CollectiveKind::Vote(_)) {
        U32
    } else {
        I32
    };
    let guarded = vec![
        Stmt::Let {
            local: VALUE,
            init: bin(BinaryOp::BitAnd, tid(), ci(1)).add_one_if_reduce(&kind),
        },
        Stmt::Collective {
            kind,
            value: ScalarExpr::Local(VALUE),
            result: RESULT,
            scope,
        },
        Stmt::Store {
            place: Place::Logical {
                buffer: 0,
                coords: vec![tid_u32()],
            },
            value: ScalarExpr::Local(RESULT),
        },
    ];
    KernelFn::new(
        "warp_uniform",
        I32,
        vec![Param::buffer("out", out, BufferRole::ReadWrite, Some(1))],
        vec![
            LocalDecl::scalar(I32),
            LocalDecl::scalar(I32),
            LocalDecl::scalar(I32),
            LocalDecl::scalar(out),
        ],
        Block(vec![
            Stmt::Let {
                local: TID,
                init: cast(tid_u32(), I32),
            },
            Stmt::Let {
                local: WARP,
                init: warp_def,
            },
            Stmt::If {
                cond,
                then: Block(guarded),
                els: Block(vec![]),
            },
        ]),
    )
    .with_launch(LaunchShape::new([block, 1, 1], [1, 1, 1]).with_subgroup(subgroup.0, subgroup.1))
}

trait AddOne {
    fn add_one_if_reduce(self, kind: &CollectiveKind) -> ScalarExpr;
}
impl AddOne for ScalarExpr {
    /// A reduction contributes `tid + 1` (so `1..=8`); a vote contributes `tid & 1`.
    fn add_one_if_reduce(self, kind: &CollectiveKind) -> ScalarExpr {
        if matches!(kind, CollectiveKind::Vote(_)) {
            self
        } else {
            add(tid(), ci(1))
        }
    }
}

fn shr(k: i64) -> ScalarExpr {
    bin(BinaryOp::Shr, tid(), ci(k))
}

fn is_zero(e: ScalarExpr) -> ScalarExpr {
    bin(BinaryOp::CmpEq, e, ci(0))
}

fn sum() -> CollectiveKind {
    CollectiveKind::Reduce(ReduceOp::Sum)
}

/// The default: `warp = tid >> 2` (`N = 4`), `if (warp == 0)`, a subgroup sum, a block of 8.
fn standard() -> KernelFn {
    kernel(8, (N, N), shr(2), is_zero(warp()), sum(), Scope::Subgroup)
}

fn accepted(k: &KernelFn) {
    assert_eq!(check_kernel(k), Ok(()));
}

fn refused(k: &KernelFn) {
    assert_eq!(check_kernel(k), Err(KernelError::BarrierDivergent));
}

#[test]
fn a_subgroup_collective_under_a_condition_on_the_warp_number_is_accepted() {
    accepted(&standard());
    // `tid / 4`, `warp < 1`, the shift written inline, and a copy of a copy of the thread index
    accepted(&kernel(
        8,
        (N, N),
        div(tid(), ci(4)),
        is_zero(warp()),
        sum(),
        Scope::Subgroup,
    ));
    accepted(&kernel(
        8,
        (N, N),
        shr(2),
        bin(BinaryOp::CmpLt, warp(), ci(1)),
        sum(),
        Scope::Subgroup,
    ));
    accepted(&kernel(
        8,
        (N, N),
        shr(2),
        is_zero(bin(BinaryOp::Shr, cast(tid_u32(), I32), ci(2))),
        CollectiveKind::Shuffle {
            mode: ShuffleMode::Xor,
            operand: ci(1),
        },
        Scope::Subgroup,
    ));
}

#[test]
fn a_shift_or_divisor_that_is_not_the_stated_subgroup_size_stays_divergent() {
    // N = 4: `tid >> 3` (warps of 8), `tid >> 1` (pairs), `tid / 8`, `tid / 2`
    for def in [shr(3), shr(1), div(tid(), ci(8))] {
        refused(&kernel(
            8,
            (N, N),
            def,
            is_zero(warp()),
            sum(),
            Scope::Subgroup,
        ));
    }
    refused(&kernel(
        8,
        (N, N),
        div(tid(), ci(2)),
        is_zero(warp()),
        sum(),
        Scope::Subgroup,
    ));
}

#[test]
fn a_launch_that_does_not_state_one_size_or_whose_rows_are_not_whole_subgroups_stays_divergent() {
    // a range of sizes, even one that contains N
    refused(&kernel(
        8,
        (2, 8),
        shr(2),
        is_zero(warp()),
        sum(),
        Scope::Subgroup,
    ));
    refused(&kernel(
        8,
        (1, 4),
        shr(2),
        is_zero(warp()),
        sum(),
        Scope::Subgroup,
    ));
    // a block whose x extent is not a multiple of N: a subgroup straddles two rows
    refused(&kernel(
        6,
        (N, N),
        shr(2),
        is_zero(warp()),
        sum(),
        Scope::Subgroup,
    ));
}

#[test]
fn a_lane_condition_is_divergent_whatever_the_form() {
    // `lane = tid & 3`, `if (lane == 0)`
    refused(&kernel(
        8,
        (N, N),
        bin(BinaryOp::BitAnd, tid(), ci(i64::from(N) - 1)),
        is_zero(warp()),
        sum(),
        Scope::Subgroup,
    ));
    // the lane id leaf itself
    refused(&kernel(
        8,
        (N, N),
        cast(ScalarExpr::Exec(ExecId::LaneId), I32),
        is_zero(warp()),
        sum(),
        Scope::Subgroup,
    ));
    // the thread index compared directly
    refused(&kernel(
        8,
        (N, N),
        shr(2),
        is_zero(tid()),
        sum(),
        Scope::Subgroup,
    ));
}

#[test]
fn a_block_collective_or_barrier_is_never_accepted_under_warp_uniform_control() {
    refused(&kernel(
        8,
        (N, N),
        shr(2),
        is_zero(warp()),
        sum(),
        Scope::Block,
    ));
    let mut k = standard();
    let Stmt::If { then, .. } = &mut k.body.0[2] else {
        panic!()
    };
    then.0.insert(0, Stmt::barrier(Scope::Block));
    refused(&k);
}

#[test]
fn a_thread_index_copy_that_is_reassigned_or_comes_from_another_axis_is_not_the_warp_number() {
    // `tid` assigned again: no longer a copy of the thread index
    let mut k = standard();
    k.body.0.insert(
        2,
        Stmt::Assign {
            local: TID,
            value: add(tid(), ci(1)),
        },
    );
    refused(&k);
    // axis y
    let mut k = standard();
    k.body.0[0] = Stmt::Let {
        local: TID,
        init: cast(ScalarExpr::Exec(ExecId::ThreadIdx(1)), I32),
    };
    refused(&k);
}

fn run(k: &KernelFn, n: usize) -> Vec<i64> {
    assert_eq!(check_kernel(k), Ok(()), "accepted");
    let buf = TypedBuffer::new(
        k.locals[RESULT as usize].elem,
        vec![n as i64],
        vec![1],
        vec![0u8; 4 * n],
    );
    let mut args = vec![KernelArg::Buffer(buf)];
    run_launch(k, &mut args).expect("runs at the stated subgroup size");
    let KernelArg::Buffer(b) = &args[0] else {
        unreachable!()
    };
    b.bytes
        .chunks(4)
        .map(|c| i64::from(i32::from_le_bytes(c.try_into().unwrap())))
        .collect()
}

#[test]
fn the_oracle_runs_the_stated_size_with_only_one_subgroup_taking_part() {
    // only warp 0 reaches the reduction: its lanes (values 1..=4) get 10; warp 1 never arrives and writes nothing
    assert_eq!(run(&standard(), 8), vec![10, 10, 10, 10, 0, 0, 0, 0]);
    // only warp 1: values 5..=8 sum to 26
    let k = kernel(
        8,
        (N, N),
        shr(2),
        bin(BinaryOp::CmpEq, warp(), ci(1)),
        sum(),
        Scope::Subgroup,
    );
    assert_eq!(run(&k, 8), vec![0, 0, 0, 0, 26, 26, 26, 26]);
}

#[test]
fn a_vote_and_a_shuffle_with_one_subgroup_taking_part_read_only_that_subgroup() {
    // ballot over lanes 0..=3 contributing `tid & 1`: lanes 1 and 3 are set, mask 0b1010 = 10, only in warp 0
    let ballot = kernel(
        8,
        (N, N),
        shr(2),
        is_zero(warp()),
        CollectiveKind::Vote(VoteKind::Ballot),
        Scope::Subgroup,
    );
    assert_eq!(run(&ballot, 8), vec![10, 10, 10, 10, 0, 0, 0, 0]);
    // `any` and `all` of the same contributions
    let any = kernel(
        8,
        (N, N),
        shr(2),
        is_zero(warp()),
        CollectiveKind::Vote(VoteKind::Any),
        Scope::Subgroup,
    );
    assert_eq!(run(&any, 8), vec![1, 1, 1, 1, 0, 0, 0, 0]);
    let all = kernel(
        8,
        (N, N),
        shr(2),
        is_zero(warp()),
        CollectiveKind::Vote(VoteKind::All),
        Scope::Subgroup,
    );
    assert_eq!(run(&all, 8), vec![0, 0, 0, 0, 0, 0, 0, 0]);
    // a butterfly shuffle (xor 1) of `tid + 1` inside warp 0: lanes swap pairs
    let shuffle = kernel(
        8,
        (N, N),
        shr(2),
        is_zero(warp()),
        CollectiveKind::Shuffle {
            mode: ShuffleMode::Xor,
            operand: ci(1),
        },
        Scope::Subgroup,
    );
    assert_eq!(run(&shuffle, 8), vec![2, 1, 4, 3, 0, 0, 0, 0]);
}

// ---- the taint is tracked through assignments, loops and breaks --------------------------------------------------------------

const X: u32 = 4;
const I: u32 = 5;

/// The guarded statements of [`kernel`]: a subgroup sum of `tid + 1`, stored to `out[tid]`.
fn guarded() -> Vec<Stmt> {
    let Stmt::If { then, .. } = &standard().body.0[2] else {
        unreachable!()
    };
    then.0.clone()
}

/// `tid = (int)threadIdx.x; warp = tid >> 2; x = 0;` then `rest`, in a launch of 8 threads with the stated subgroup size 4.
fn with(rest: Vec<Stmt>) -> KernelFn {
    let mut k = standard();
    let mut body = vec![
        k.body.0[0].clone(),
        k.body.0[1].clone(),
        Stmt::Let {
            local: X,
            init: ci(0),
        },
    ];
    body.extend(rest);
    k.body = Block(body);
    k.locals.push(LocalDecl::scalar(I32));
    k.locals.push(LocalDecl::scalar(I32));
    k
}

fn if_(cond: ScalarExpr, then: Vec<Stmt>) -> Stmt {
    Stmt::If {
        cond,
        then: Block(then),
        els: Block(vec![]),
    }
}

fn set_x(v: i64) -> Stmt {
    Stmt::Assign {
        local: X,
        value: ci(v),
    }
}

fn lane_is_zero() -> ScalarExpr {
    is_zero(bin(BinaryOp::BitAnd, tid(), ci(i64::from(N) - 1)))
}

#[test]
fn a_value_assigned_under_lane_divergent_control_is_lane_divergent() {
    // `x = 0; if (lane == 0) x = 1; if (x == 0) { sum }`: x differs between the lanes of a subgroup
    refused(&with(vec![
        if_(lane_is_zero(), vec![set_x(1)]),
        if_(is_zero(ScalarExpr::Local(X)), guarded()),
    ]));
}

#[test]
fn a_value_assigned_under_warp_uniform_control_is_still_warp_uniform() {
    // `x = 0; if (warp == 0) x = 1; if (x == 1) { sum }`: x differs between warps, never within one
    accepted(&with(vec![
        if_(is_zero(warp()), vec![set_x(1)]),
        if_(bin(BinaryOp::CmpEq, ScalarExpr::Local(X), ci(1)), guarded()),
    ]));
}

fn counted(end: ScalarExpr, body: Vec<Stmt>) -> Stmt {
    Stmt::For {
        unroll: Unroll::None,
        local: I,
        start: ci(0),
        end,
        step: ci(1),
        body: Block(body),
    }
}

#[test]
fn a_loop_bounded_by_the_lane_diverges_but_one_bounded_by_the_warp_does_not() {
    refused(&with(vec![counted(tid(), guarded())]));
    accepted(&with(vec![counted(warp(), guarded())]));
    // a while on a lane-dependent condition
    refused(&with(vec![Stmt::While {
        cond: bin(BinaryOp::CmpLt, tid(), ci(2)),
        body: Block(guarded()),
    }]));
}

#[test]
fn a_collective_after_a_divergent_break_is_refused_even_when_the_break_is_warp_uniform() {
    // known over-rejection: a `break` under any thread-divergent control poisons what follows for every scope
    let mut body = vec![if_(is_zero(warp()), vec![Stmt::Break])];
    body.extend(guarded());
    refused(&with(vec![counted(ci(4), body)]));
}
