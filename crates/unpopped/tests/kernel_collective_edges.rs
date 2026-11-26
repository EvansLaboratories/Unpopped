// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The edge rules of subgroup collectives** (`docs/design-kernel-collectives-edges.md`; vulkane's `collective-edge-rules` request):
//! one subgroup of 8 lanes, lane `i` contributes `i + 1`, and the expected array of each case is written by hand. Shuffle `Up`,
//! `Down` and `Xor` give a lane with no source lane its OWN value (PTX ISA 9.5 section 9.7.10.6 `shfl.sync`); `Idx` reads lane
//! `operand mod n` (a Euclidean remainder); a ballot is defined for 32 lanes and a typed trap for 33.

use unpopped::exec::{ExecId, LaunchShape, Scope};
use unpopped::ir::{BinaryOp, CastSpec, RoundMode, Saturation, ScalarExpr};
use unpopped::kernel::{
    Block, BufferRole, CollectiveKind, KernelFn, LocalDecl, Param, Place, ShuffleMode, Stmt,
    VoteKind, check_kernel,
};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::{KernelArg, KernelTrap};
use unpopped::oracle::kernel_launch::run_launch_with_subgroup;
use unpopped_vocab::ElementKind;

const I32: ElementKind = ElementKind::I32;
const U32: ElementKind = ElementKind::U32;
const N: u32 = 8;

fn tid() -> ScalarExpr {
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

fn c(v: i64) -> ScalarExpr {
    ScalarExpr::ConstInt(v, I32)
}

/// A collective over `value`, each lane's result stored to `out[tid]`, in `lanes` threads that are one subgroup.
fn kernel(
    kind: CollectiveKind,
    value: ScalarExpr,
    ty: ElementKind,
    out: ElementKind,
    lanes: u32,
) -> KernelFn {
    KernelFn::new(
        "edge",
        I32,
        vec![Param::buffer("out", out, BufferRole::ReadWrite, Some(1))],
        vec![LocalDecl::scalar(ty), LocalDecl::scalar(out)],
        Block(vec![
            Stmt::Let {
                local: 0,
                init: value,
            },
            Stmt::Collective {
                kind,
                value: ScalarExpr::Local(0),
                result: 1,
                scope: Scope::Subgroup,
            },
            Stmt::Store {
                place: Place::Logical {
                    buffer: 0,
                    coords: vec![tid()],
                },
                value: ScalarExpr::Local(1),
            },
        ]),
    )
    .with_launch(LaunchShape::new([lanes, 1, 1], [1, 1, 1]).with_subgroup(lanes, lanes))
}

fn run(k: &KernelFn, lanes: u32, out: ElementKind) -> Result<Vec<i64>, KernelTrap> {
    let buf = TypedBuffer::new(
        out,
        vec![i64::from(lanes)],
        vec![1],
        vec![0u8; 4 * lanes as usize],
    );
    let mut args = vec![KernelArg::Buffer(buf)];
    run_launch_with_subgroup(k, &mut args, Some(lanes))?;
    let KernelArg::Buffer(b) = &args[0] else {
        unreachable!()
    };
    Ok(b.bytes
        .chunks(4)
        .map(|c| match out {
            I32 => i64::from(i32::from_le_bytes(c.try_into().unwrap())),
            _ => i64::from(u32::from_le_bytes(c.try_into().unwrap())),
        })
        .collect())
}

fn lane_plus_one() -> ScalarExpr {
    ScalarExpr::Add(Box::new(cast(tid(), I32)), Box::new(c(1)))
}

fn shuffle(mode: ShuffleMode, operand: i64) -> Vec<i64> {
    let k = kernel(
        CollectiveKind::Shuffle {
            mode,
            operand: c(operand),
        },
        lane_plus_one(),
        I32,
        I32,
        N,
    );
    assert_eq!(check_kernel(&k), Ok(()), "well formed");
    run(&k, N, I32).expect("runs")
}

const OWN: [i64; 8] = [1, 2, 3, 4, 5, 6, 7, 8];

#[test]
fn up_reads_its_own_value_when_the_source_is_below_lane_0() {
    // delta 0: every lane reads itself; 1: lane 0 has no source; n - 1: only lane 7 has one (lane 0); n and n + 1: none
    assert_eq!(shuffle(ShuffleMode::Up, 0), OWN);
    assert_eq!(shuffle(ShuffleMode::Up, 1), [1, 1, 2, 3, 4, 5, 6, 7]);
    assert_eq!(shuffle(ShuffleMode::Up, 7), [1, 2, 3, 4, 5, 6, 7, 1]);
    assert_eq!(shuffle(ShuffleMode::Up, 8), OWN);
    assert_eq!(shuffle(ShuffleMode::Up, 9), OWN);
}

#[test]
fn down_reads_its_own_value_when_the_source_is_above_the_last_lane() {
    assert_eq!(shuffle(ShuffleMode::Down, 0), OWN);
    assert_eq!(shuffle(ShuffleMode::Down, 1), [2, 3, 4, 5, 6, 7, 8, 8]);
    assert_eq!(shuffle(ShuffleMode::Down, 7), [8, 2, 3, 4, 5, 6, 7, 8]);
    assert_eq!(shuffle(ShuffleMode::Down, 8), OWN);
    assert_eq!(shuffle(ShuffleMode::Down, 9), OWN);
}

#[test]
fn xor_reads_its_own_value_when_the_partner_does_not_exist() {
    assert_eq!(shuffle(ShuffleMode::Xor, 0), OWN);
    assert_eq!(shuffle(ShuffleMode::Xor, 1), [2, 1, 4, 3, 6, 5, 8, 7]);
    assert_eq!(shuffle(ShuffleMode::Xor, 7), [8, 7, 6, 5, 4, 3, 2, 1]);
    // mask n and n + 1: every `i ^ mask` is at least 8, so no lane has a partner
    assert_eq!(shuffle(ShuffleMode::Xor, 8), OWN);
    assert_eq!(shuffle(ShuffleMode::Xor, 9), OWN);
}

#[test]
fn idx_reads_lane_operand_mod_n_with_a_euclidean_remainder() {
    // operand 8 = n reads lane 0 (value 1), 9 reads lane 1 (value 2), -1 reads lane 7 (value 8); 3 is in range (value 4)
    assert_eq!(shuffle(ShuffleMode::Idx, 8), [1; 8]);
    assert_eq!(shuffle(ShuffleMode::Idx, 9), [2; 8]);
    assert_eq!(shuffle(ShuffleMode::Idx, -1), [8; 8]);
    assert_eq!(shuffle(ShuffleMode::Idx, 3), [4; 8]);
}

fn ballot(lanes: u32) -> Result<Vec<i64>, KernelTrap> {
    // the predicate is `lane & 1`: odd lanes vote 1
    let odd = ScalarExpr::Binary(BinaryOp::BitAnd, Box::new(cast(tid(), I32)), Box::new(c(1)));
    let k = kernel(CollectiveKind::Vote(VoteKind::Ballot), odd, I32, U32, lanes);
    assert_eq!(check_kernel(&k), Ok(()), "well formed");
    run(&k, lanes, U32)
}

#[test]
fn a_ballot_is_defined_at_32_lanes_and_a_typed_trap_at_33() {
    // 32 lanes: bits 1, 3, 5 .. 31 set = 0xAAAAAAAA for every lane
    assert_eq!(ballot(32).expect("32 lanes"), vec![0xAAAA_AAAA_i64; 32]);
    // the control: the same kernel at 8 lanes gives 0xAA
    assert_eq!(ballot(8).expect("8 lanes"), vec![0xAA_i64; 8]);
    // 33 lanes cannot be a U32 mask: a typed trap, not a silent truncation
    let r = ballot(33);
    assert!(
        matches!(r, Err(KernelTrap::Unsupported(_))),
        "33 lanes must be a typed trap, got {r:?}"
    );
}
