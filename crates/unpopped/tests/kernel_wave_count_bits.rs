// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Wave count-bits idioms** (survey row 46, read from the HLSL Shader Model 6.0 wave table in
//! `docs/ir-union-evidence/conformance-idiom-gaps-verified-2026-11-27.md`): `WaveActiveCountBits(p)` is a sum of the predicate over
//! the subgroup, `WavePrefixCountBits(p)` the sum over lower lanes. Both are built from `Collective` `Reduce(Sum)` and
//! exclusive `Scan(Sum)` over the predicate as an `I32`, with a `Ballot` beside them so the count is checked against the mask.
//! Eight lanes, subgroups of four, predicate bits `0xB4` (lanes 0..7 = 0,0,1,0,1,1,0,1). The expected arrays are written by hand.
//! Mutants (an inclusive scan, a `Max` for the sum) change a lane by value.

use unpopped::exec::{ExecId, LaunchShape, Scope};
use unpopped::ir::{BinaryOp, CastSpec, ReduceOp, RoundMode, Saturation, ScalarExpr};
use unpopped::kernel::{
    Block, BufferRole, CollectiveKind, KernelFn, LocalDecl, Param, Place, Stmt, VoteKind,
    check_kernel,
};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch_with_subgroup;
use unpopped_vocab::ElementKind;

const I32: ElementKind = ElementKind::I32;
const U32: ElementKind = ElementKind::U32;
const N: usize = 8;

fn ci(v: i64) -> ScalarExpr {
    ScalarExpr::ConstInt(v, I32)
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

/// `(0xB4 >> lane) & 1` as an `I32`.
fn predicate() -> ScalarExpr {
    let shifted = ScalarExpr::Binary(BinaryOp::Shr, Box::new(ci(0xB4)), Box::new(lane()));
    ScalarExpr::Binary(BinaryOp::BitAnd, Box::new(shifted), Box::new(ci(1)))
}

/// One collective over the predicate (local 0), its result (local 1) stored to `out[lane]`.
fn kernel(kind: CollectiveKind, out: ElementKind) -> KernelFn {
    let at = Place::Logical {
        buffer: 0,
        coords: vec![ScalarExpr::Exec(ExecId::ThreadIdx(0))],
    };
    let body = vec![
        Stmt::Let {
            local: 0,
            init: predicate(),
        },
        Stmt::Collective {
            kind,
            value: ScalarExpr::Local(0),
            result: 1,
            scope: Scope::Subgroup,
        },
        Stmt::Store {
            place: at,
            value: ScalarExpr::Local(1),
        },
    ];
    KernelFn::new(
        "count_bits",
        I32,
        vec![Param::buffer("out", out, BufferRole::ReadWrite, Some(1))],
        vec![LocalDecl::scalar(I32), LocalDecl::scalar(out)],
        Block(body),
    )
    .with_launch(LaunchShape::new([N as u32, 1, 1], [1, 1, 1]).with_subgroup(1, 32))
}

fn run(k: &KernelFn, out: ElementKind) -> Vec<i64> {
    assert_eq!(check_kernel(k), Ok(()), "well formed");
    let buf = TypedBuffer::new(out, vec![N as i64], vec![1], vec![0u8; 4 * N]);
    let mut args = vec![KernelArg::Buffer(buf)];
    run_launch_with_subgroup(k, &mut args, Some(4)).expect("runs");
    let KernelArg::Buffer(b) = &args[0] else {
        unreachable!()
    };
    b.bytes
        .chunks(4)
        .map(|c| match out {
            U32 => i64::from(u32::from_le_bytes(c.try_into().unwrap())),
            _ => i64::from(i32::from_le_bytes(c.try_into().unwrap())),
        })
        .collect()
}

fn sum() -> CollectiveKind {
    CollectiveKind::Reduce(ReduceOp::Sum)
}

fn scan(exclusive: bool) -> CollectiveKind {
    CollectiveKind::Scan {
        op: ReduceOp::Sum,
        exclusive,
    }
}

#[test]
fn active_count_bits_is_the_subgroup_sum_of_the_predicate() {
    assert_eq!(run(&kernel(sum(), I32), I32), [1, 1, 1, 1, 3, 3, 3, 3]);
}

#[test]
fn prefix_count_bits_is_the_exclusive_sum_over_lower_lanes() {
    assert_eq!(run(&kernel(scan(true), I32), I32), [0, 0, 0, 1, 0, 1, 2, 2]);
}

#[test]
fn the_ballot_mask_agrees_with_the_count() {
    let ballot = run(&kernel(CollectiveKind::Vote(VoteKind::Ballot), U32), U32);
    assert_eq!(
        ballot,
        [
            0b0100, 0b0100, 0b0100, 0b0100, 0b1011, 0b1011, 0b1011, 0b1011
        ]
    );
    let counts = run(&kernel(sum(), I32), I32);
    for (mask, count) in ballot.iter().zip(&counts) {
        assert_eq!(i64::from(mask.count_ones()), *count);
    }
}

#[test]
fn the_mutants_change_a_lane_by_value() {
    let want = [0, 0, 0, 1, 0, 1, 2, 2];
    let inclusive = run(&kernel(scan(false), I32), I32);
    assert_eq!(inclusive, [0, 0, 1, 1, 1, 2, 2, 3]);
    assert_ne!(inclusive, want);
    let max = run(&kernel(CollectiveKind::Reduce(ReduceOp::Max), I32), I32);
    assert_ne!(max, [1, 1, 1, 1, 3, 3, 3, 3]);
}
