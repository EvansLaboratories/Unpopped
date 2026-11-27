// SPDX-License-Identifier: MIT OR Apache-2.0
//! The quad and wave-query idioms of HLSL (`QuadReadAcrossX/Y/Diagonal`, `QuadReadLaneAt`, `WaveReadLaneFirst`,
//! `WaveActiveAllEqual`) and Slang, built from `Stmt::Collective` shuffles and votes already on main (survey rows 18, 45 and 46 of
//! `docs/ir-union-evidence/conformance-idiom-gaps-2026-11-27.md`). A quad is four aligned lanes: across X is the shuffle with
//! XOR mask 1, across Y mask 2, diagonal mask 3, and read-lane-at is an index shuffle whose lane is the quad's base plus the
//! requested quad lane. First-lane and all-equal are an index-0 shuffle and a vote; WHICH lane is first when some lanes are
//! inactive is NOT modelled (the subgroup participates in full). One thread per lane; `ThreadIdx(0)` is the lane.

use unpopped::exec::{ExecId, LaunchShape, Scope};
use unpopped::ir::{BinaryOp, CastSpec, RoundMode, Saturation, ScalarExpr};
use unpopped::kernel::{
    Block, BufferRole, CollectiveKind, KernelFn, LocalDecl, Param, Place, ShuffleMode, Stmt,
    VoteKind,
};
use unpopped_vocab::ElementKind;

pub(crate) const I32: ElementKind = ElementKind::I32;
pub(crate) const U32: ElementKind = ElementKind::U32;
pub(crate) const LANES: usize = 8;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Op {
    AcrossX,
    AcrossY,
    Diagonal,
    /// `QuadReadLaneAt(q)` for the given quad lane.
    LaneAt(i64),
    ReadLaneFirst,
    AllEqual,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Mutant {
    None,
    /// Across X uses mask 2 (the Y mask).
    XMaskIsY,
    /// Read-lane-at forgets the quad base and reads lane `q` of the whole subgroup.
    NoQuadBase,
    /// All-equal votes `Any` instead of `All`.
    AnyForAll,
}

fn c(v: i64) -> ScalarExpr {
    ScalarExpr::ConstInt(v, I32)
}
fn l(i: u32) -> ScalarExpr {
    ScalarExpr::Local(i)
}
fn lane() -> ScalarExpr {
    ScalarExpr::Cast(
        Box::new(ScalarExpr::Exec(ExecId::ThreadIdx(0))),
        CastSpec {
            to: I32,
            round: RoundMode::NearestEven,
            sat: Saturation::Wrap,
        },
    )
}
fn at(buffer: u32) -> Place {
    Place::Logical {
        buffer,
        coords: vec![ScalarExpr::Exec(ExecId::ThreadIdx(0))],
    }
}
fn shuffle(mode: ShuffleMode, operand: ScalarExpr) -> Stmt {
    Stmt::Collective {
        kind: CollectiveKind::Shuffle { mode, operand },
        value: l(0),
        result: 1,
        scope: Scope::Subgroup,
    }
}

/// The statements between the load of local 0 and the store: the result ends up in local 1 (an `I32`).
fn middle(op: Op, mutant: Mutant) -> Vec<Stmt> {
    match op {
        Op::AcrossX => {
            let mask = if mutant == Mutant::XMaskIsY { 2 } else { 1 };
            vec![shuffle(ShuffleMode::Xor, c(mask))]
        }
        Op::AcrossY => vec![shuffle(ShuffleMode::Xor, c(2))],
        Op::Diagonal => vec![shuffle(ShuffleMode::Xor, c(3))],
        Op::LaneAt(q) => {
            let target = if mutant == Mutant::NoQuadBase {
                c(q)
            } else {
                ScalarExpr::Binary(
                    BinaryOp::BitOr,
                    Box::new(ScalarExpr::Binary(
                        BinaryOp::BitAnd,
                        Box::new(lane()),
                        Box::new(c(-4)),
                    )),
                    Box::new(c(q)),
                )
            };
            vec![shuffle(ShuffleMode::Idx, target)]
        }
        Op::ReadLaneFirst => vec![shuffle(ShuffleMode::Idx, c(0))],
        Op::AllEqual => all_equal(mutant),
    }
}

/// Lane 0's value is broadcast, each lane compares its own value to it, and the subgroup votes on the comparisons. Locals: 1 lane
/// 0's value, 2 the comparison as an `I32`, 3 the vote (`U32`), then 1 again holds the vote widened to `I32`.
fn all_equal(mutant: Mutant) -> Vec<Stmt> {
    let vote = if mutant == Mutant::AnyForAll {
        VoteKind::Any
    } else {
        VoteKind::All
    };
    vec![
        shuffle(ShuffleMode::Idx, c(0)),
        Stmt::Let {
            local: 2,
            init: ScalarExpr::Cast(
                Box::new(ScalarExpr::Binary(
                    BinaryOp::CmpEq,
                    Box::new(l(0)),
                    Box::new(l(1)),
                )),
                CastSpec {
                    to: I32,
                    round: RoundMode::NearestEven,
                    sat: Saturation::Wrap,
                },
            ),
        },
        Stmt::Collective {
            kind: CollectiveKind::Vote(vote),
            value: l(2),
            result: 3,
            scope: Scope::Subgroup,
        },
        Stmt::Assign {
            local: 1,
            value: ScalarExpr::Cast(
                Box::new(l(3)),
                CastSpec {
                    to: I32,
                    round: RoundMode::NearestEven,
                    sat: Saturation::Wrap,
                },
            ),
        },
    ]
}

/// Buffers: 0 `input` `[LANES]` (read), 1 `out` `[LANES]` (write), both `I32`. Locals: 0 the lane's value, 1 the result, 2 and 3 for
/// the all-equal vote.
pub(crate) fn wave_op(op: Op, mutant: Mutant) -> KernelFn {
    let mut body = vec![Stmt::Load {
        local: 0,
        place: at(0),
    }];
    body.extend(middle(op, mutant));
    body.push(Stmt::Store {
        place: at(1),
        value: l(1),
    });
    let params = vec![
        Param::buffer("input", I32, BufferRole::Read, Some(1)),
        Param::buffer("out", I32, BufferRole::Write, Some(1)),
    ];
    let locals = [I32, I32, I32, U32].map(LocalDecl::scalar).to_vec();
    KernelFn::new("wave_op", I32, params, locals, Block(body))
        .with_launch(LaunchShape::new([LANES as u32, 1, 1], [1, 1, 1]).with_subgroup(1, 32))
}
