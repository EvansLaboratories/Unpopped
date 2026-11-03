// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Collectives** (ROADMAP U5 P5, PM ruling 2026-11-02 (4); `docs/options-u5-collectives-and-richer-reductions.md`):
//! `Stmt::Collective` over a `Subgroup` or a `Block`, full participation of the scope. A collective is a rendezvous
//! like a barrier that also exchanges registers: the oracle stops every thread of the scope at the statement, then
//! computes every lane's result from every lane's contribution. These tests run an 8-thread block with a subgroup of 4
//! (two subgroups) and write each lane's result to `out[ThreadIdx.x]`.
//!
//! Each lane contributes `ThreadIdx.x + 1` (so `1..=8`) unless a test says otherwise; the expected arrays are written
//! out by hand, not computed by a model of the collective.

use unpopped::exec::{ExecId, LaunchShape, Scope};
use unpopped::ir::{BinaryOp, CastSpec, ReduceOp, RoundMode, Saturation, ScalarExpr};
use unpopped::kernel::{
    Block, BufferRole, CollectiveKind, KernelError, KernelFn, LocalDecl, Param, Place, ShuffleMode,
    Stmt, VoteKind, check_kernel,
};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::{KernelArg, KernelTrap};
use unpopped::oracle::kernel_launch::run_launch_with_subgroup;
use unpopped_vocab::ElementKind;

const I32: ElementKind = ElementKind::I32;
const U32: ElementKind = ElementKind::U32;
const F32: ElementKind = ElementKind::F32;
const N: u32 = 8;

fn tid() -> ScalarExpr {
    ScalarExpr::Exec(ExecId::ThreadIdx(0))
}

fn cast(e: ScalarExpr, to: ElementKind) -> ScalarExpr {
    let spec = CastSpec {
        to,
        round: RoundMode::NearestEven,
        sat: Saturation::Wrap,
    };
    ScalarExpr::Cast(Box::new(e), spec)
}

fn c(v: i64) -> ScalarExpr {
    ScalarExpr::ConstInt(v, I32)
}

/// `ThreadIdx.x + 1` as an `I32`.
fn lane_value() -> ScalarExpr {
    ScalarExpr::Add(Box::new(cast(tid(), I32)), Box::new(c(1)))
}

/// One collective over `value`, its result stored to `out[tid]`: locals 0 (the value) and 1 (the result).
fn kernel(
    kind: CollectiveKind,
    scope: Scope,
    value: ScalarExpr,
    ty: ElementKind,
    out: ElementKind,
) -> KernelFn {
    let body = vec![
        Stmt::Let {
            local: 0,
            init: value,
        },
        Stmt::Collective {
            kind,
            value: ScalarExpr::Local(0),
            result: 1,
            scope,
        },
        Stmt::Store {
            place: Place::Logical {
                buffer: 0,
                coords: vec![tid()],
            },
            value: ScalarExpr::Local(1),
        },
    ];
    KernelFn::new(
        "coll",
        if ty == F32 { F32 } else { I32 },
        vec![Param::buffer("out", out, BufferRole::ReadWrite, Some(1))],
        vec![LocalDecl::scalar(ty), LocalDecl::scalar(out)],
        Block(body),
    )
    .with_launch(LaunchShape::new([N, 1, 1], [1, 1, 1]).with_subgroup(1, 32))
}

fn run(k: &KernelFn, out: ElementKind) -> Result<Vec<f64>, KernelTrap> {
    // every dtype used here is four bytes wide
    let buf = TypedBuffer::new(out, vec![i64::from(N)], vec![1], vec![0u8; 4 * N as usize]);
    let mut args = vec![KernelArg::Buffer(buf)];
    run_launch_with_subgroup(k, &mut args, Some(4))?;
    let KernelArg::Buffer(b) = &args[0] else {
        unreachable!()
    };
    Ok(b.bytes
        .chunks(4)
        .map(|c| match out {
            F32 => f64::from(f32::from_le_bytes(c.try_into().unwrap())),
            I32 => f64::from(i32::from_le_bytes(c.try_into().unwrap())),
            _ => f64::from(u32::from_le_bytes(c.try_into().unwrap())),
        })
        .collect())
}

fn ints(k: &KernelFn) -> Vec<i64> {
    assert_eq!(check_kernel(k), Ok(()), "well formed");
    run(k, I32)
        .expect("runs")
        .into_iter()
        .map(|v| v as i64)
        .collect()
}

fn on(kind: CollectiveKind, scope: Scope) -> KernelFn {
    kernel(kind, scope, lane_value(), I32, I32)
}

#[test]
fn a_reduction_gives_every_lane_the_fold_of_its_scope() {
    let sum = CollectiveKind::Reduce(ReduceOp::Sum);
    assert_eq!(ints(&on(sum.clone(), Scope::Block)), vec![36; 8]);
    assert_eq!(
        ints(&on(sum, Scope::Subgroup)),
        vec![10, 10, 10, 10, 26, 26, 26, 26]
    );
    let max = on(CollectiveKind::Reduce(ReduceOp::Max), Scope::Subgroup);
    assert_eq!(ints(&max), vec![4, 4, 4, 4, 8, 8, 8, 8]);
    let min = on(CollectiveKind::Reduce(ReduceOp::Min), Scope::Subgroup);
    assert_eq!(ints(&min), vec![1, 1, 1, 1, 5, 5, 5, 5]);
}

#[test]
fn the_bit_folds_act_on_integers() {
    let r = |op| ints(&on(CollectiveKind::Reduce(op), Scope::Subgroup));
    assert_eq!(
        r(ReduceOp::And),
        vec![0; 8],
        "1&2&3&4 and 5&6&7&8 are both zero"
    );
    assert_eq!(r(ReduceOp::Or), vec![7, 7, 7, 7, 15, 15, 15, 15]);
    assert_eq!(
        r(ReduceOp::Xor),
        vec![4, 4, 4, 4, 12, 12, 12, 12],
        "1^2^3^4, 5^6^7^8"
    );
}

#[test]
fn a_scan_is_the_prefix_fold_inclusive_or_exclusive() {
    let scan = |exclusive, scope| {
        ints(&on(
            CollectiveKind::Scan {
                op: ReduceOp::Sum,
                exclusive,
            },
            scope,
        ))
    };
    assert_eq!(scan(false, Scope::Block), vec![1, 3, 6, 10, 15, 21, 28, 36]);
    assert_eq!(scan(true, Scope::Block), vec![0, 1, 3, 6, 10, 15, 21, 28]);
    assert_eq!(
        scan(false, Scope::Subgroup),
        vec![1, 3, 6, 10, 5, 11, 18, 26]
    );
}

#[test]
fn a_shuffle_reads_another_lane_or_keeps_its_own() {
    let shuffle = |mode, operand: i64| {
        ints(&on(
            CollectiveKind::Shuffle {
                mode,
                operand: c(operand),
            },
            Scope::Subgroup,
        ))
    };
    assert_eq!(shuffle(ShuffleMode::Idx, 1), vec![2, 2, 2, 2, 6, 6, 6, 6]);
    assert_eq!(
        shuffle(ShuffleMode::Up, 1),
        vec![1, 1, 2, 3, 5, 5, 6, 7],
        "lane 0 has no lane below"
    );
    assert_eq!(
        shuffle(ShuffleMode::Down, 1),
        vec![2, 3, 4, 4, 6, 7, 8, 8],
        "lane 3 has no lane above"
    );
    assert_eq!(shuffle(ShuffleMode::Xor, 1), vec![2, 1, 4, 3, 6, 5, 8, 7]);
    assert_eq!(shuffle(ShuffleMode::Xor, 2), vec![3, 4, 1, 2, 7, 8, 5, 6]);
}

#[test]
fn a_broadcast_gives_every_lane_one_lanes_value() {
    let b = CollectiveKind::Broadcast { lane: c(2) };
    assert_eq!(ints(&on(b.clone(), Scope::Block)), vec![3; 8]);
    assert_eq!(ints(&on(b, Scope::Subgroup)), vec![3, 3, 3, 3, 7, 7, 7, 7]);
}

#[test]
fn votes_are_predicates_over_the_scope() {
    // each lane contributes ThreadIdx.x: lane 0 of subgroup 0 contributes zero
    let vote = |v| {
        let k = kernel(
            CollectiveKind::Vote(v),
            Scope::Subgroup,
            cast(tid(), I32),
            I32,
            U32,
        );
        assert_eq!(check_kernel(&k), Ok(()));
        run(&k, U32)
            .expect("runs")
            .into_iter()
            .map(|x| x as i64)
            .collect::<Vec<_>>()
    };
    assert_eq!(vote(VoteKind::All), vec![0, 0, 0, 0, 1, 1, 1, 1]);
    assert_eq!(vote(VoteKind::Any), vec![1; 8]);
    assert_eq!(
        vote(VoteKind::Ballot),
        vec![
            0b1110, 0b1110, 0b1110, 0b1110, 0b1111, 0b1111, 0b1111, 0b1111
        ]
    );
}

#[test]
fn a_float_sum_is_the_left_fold_in_lane_order() {
    let half = ScalarExpr::Mul(Box::new(cast(tid(), F32)), Box::new(ScalarExpr::Const(0.5)));
    let k = kernel(
        CollectiveKind::Reduce(ReduceOp::Sum),
        Scope::Block,
        half,
        F32,
        F32,
    );
    assert_eq!(check_kernel(&k), Ok(()));
    assert_eq!(
        run(&k, F32).expect("runs"),
        vec![14.0; 8],
        "0.5 * (0 + 1 + ... + 7)"
    );
}

#[test]
fn a_collective_in_divergent_control_is_refused() {
    let guard = ScalarExpr::Binary(
        BinaryOp::CmpLt,
        Box::new(tid()),
        Box::new(ScalarExpr::ConstInt(4, U32)),
    );
    let inner = on(CollectiveKind::Reduce(ReduceOp::Sum), Scope::Block);
    let mut k = inner.clone();
    k.body.0 = vec![
        inner.body.0[0].clone(),
        Stmt::If {
            cond: guard,
            then: Block(vec![inner.body.0[1].clone(), inner.body.0[2].clone()]),
            els: Block(vec![]),
        },
    ];
    assert_eq!(check_kernel(&k), Err(KernelError::BarrierDivergent));
}

#[test]
fn a_collective_does_not_order_memory() {
    // x[tid] = 1; collective; read x[7 - tid]: still a data race, because only a barrier orders memory
    let k = {
        let mut k = on(CollectiveKind::Reduce(ReduceOp::Sum), Scope::Block);
        let store = Stmt::Store {
            place: Place::Logical {
                buffer: 0,
                coords: vec![tid()],
            },
            value: ScalarExpr::ConstInt(1, I32),
        };
        let rev = ScalarExpr::Sub(Box::new(ScalarExpr::ConstInt(7, U32)), Box::new(tid()));
        let load = Stmt::Load {
            local: 2,
            place: Place::Logical {
                buffer: 0,
                coords: vec![rev],
            },
        };
        k.body.0.insert(0, store);
        k.body.0.insert(3, load);
        k.locals.push(LocalDecl::scalar(I32));
        k
    };
    assert_eq!(check_kernel(&k), Ok(()));
    assert_eq!(run(&k, I32), Err(KernelTrap::DataRace));
}

#[test]
fn the_checker_admits_only_the_v1_forms() {
    let refused = |k: &KernelFn| {
        matches!(
            check_kernel(k),
            Err(KernelError::CollectiveUnsupported { .. })
        )
    };
    // a ballot over a block, an exclusive Max, a bit fold of floats, a Mean
    let ballot = kernel(
        CollectiveKind::Vote(VoteKind::Ballot),
        Scope::Block,
        lane_value(),
        I32,
        U32,
    );
    assert!(refused(&ballot));
    let ex_max = on(
        CollectiveKind::Scan {
            op: ReduceOp::Max,
            exclusive: true,
        },
        Scope::Block,
    );
    assert!(refused(&ex_max));
    let half = ScalarExpr::Const(0.5);
    let float_and = kernel(
        CollectiveKind::Reduce(ReduceOp::And),
        Scope::Block,
        half,
        F32,
        F32,
    );
    assert!(refused(&float_and));
    assert!(refused(&on(
        CollectiveKind::Reduce(ReduceOp::Mean),
        Scope::Block
    )));
}

#[test]
fn the_checker_wants_a_launch_a_group_scope_and_an_integer_lane() {
    // a thread scope, no launch, a float lane operand
    let thread = on(CollectiveKind::Reduce(ReduceOp::Sum), Scope::Thread);
    assert_eq!(check_kernel(&thread), Err(KernelError::BadBarrierScope));
    let mut unlaunched = on(CollectiveKind::Reduce(ReduceOp::Sum), Scope::Block);
    unlaunched.launch = None;
    assert_eq!(
        check_kernel(&unlaunched),
        Err(KernelError::ExecWithoutLaunch)
    );
    let bad_lane = on(
        CollectiveKind::Broadcast {
            lane: cast(tid(), F32),
        },
        Scope::Block,
    );
    assert!(matches!(
        check_kernel(&bad_lane),
        Err(KernelError::TypeMismatch { .. })
    ));
}
