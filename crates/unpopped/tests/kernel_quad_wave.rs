// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Quad reads, read-lane-first and all-equal from shuffles and votes already on main** (survey rows 18, 45, 46). Eight lanes in
//! one subgroup (two quads) hold distinct boundary values; each operation is run on the oracle and compared with a reference
//! written from the quad's definition (a lane's quad base and its position in the quad), not from XOR masks. Integers have no NaN,
//! so the non-blind checks are that the quads differ from each other and that each mutant changes the result. Which lane is
//! "first" among INACTIVE lanes is not modelled: the whole subgroup participates.

use unpopped::kernel::{KernelFn, check_kernel};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch_with_subgroup;

#[path = "support/quad_wave.rs"]
mod fixture;
use fixture::{I32, LANES, Mutant, Op, wave_op};

fn buf(v: &[i32]) -> KernelArg {
    KernelArg::Buffer(TypedBuffer::new(
        I32,
        vec![v.len() as i64],
        vec![1],
        v.iter().flat_map(|x| x.to_le_bytes()).collect(),
    ))
}

fn run(op: Op, mutant: Mutant, input: &[i32]) -> Vec<i32> {
    let k: KernelFn = wave_op(op, mutant);
    assert_eq!(check_kernel(&k), Ok(()));
    let mut args = vec![buf(input), buf(&[0; LANES])];
    run_launch_with_subgroup(&k, &mut args, Some(LANES as u32)).expect("the oracle runs it");
    let KernelArg::Buffer(o) = &args[1] else {
        unreachable!("a buffer")
    };
    o.bytes
        .chunks(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

const MIXED: [i32; LANES] = [5, -3, i32::MIN, 7, -1, 0, i32::MAX, 42];

/// A lane's quad base and its position within the quad, from the definition of a quad of four aligned lanes.
fn quad(lane: usize) -> (usize, usize) {
    (lane / 4 * 4, lane % 4)
}

/// Across X swaps the two lanes of a row of the 2x2 quad (positions 0 and 1, 2 and 3); Y swaps the rows (0 and 2, 1 and 3);
/// diagonal swaps 0 with 3 and 1 with 2.
fn reference(op: Op, v: &[i32]) -> Vec<i32> {
    (0..LANES)
        .map(|lane| {
            let (base, pos) = quad(lane);
            match op {
                Op::AcrossX => v[base + [1, 0, 3, 2][pos]],
                Op::AcrossY => v[base + [2, 3, 0, 1][pos]],
                Op::Diagonal => v[base + [3, 2, 1, 0][pos]],
                Op::LaneAt(q) => v[base + q as usize],
                Op::ReadLaneFirst => v[0],
                Op::AllEqual => i32::from(v.iter().all(|x| *x == v[0])),
            }
        })
        .collect()
}

#[test]
fn every_quad_and_wave_operation_matches_its_definition() {
    let all_nine = [-9; LANES];
    let mut one_off = all_nine;
    one_off[5] = -8;
    let ops = [
        Op::AcrossX,
        Op::AcrossY,
        Op::Diagonal,
        Op::LaneAt(0),
        Op::LaneAt(3),
        Op::ReadLaneFirst,
        Op::AllEqual,
    ];
    for op in ops {
        for input in [MIXED, all_nine, one_off] {
            assert_eq!(
                run(op, Mutant::None, &input),
                reference(op, &input),
                "{op:?} {input:?}"
            );
        }
    }
}

#[test]
fn the_all_equal_corpus_distinguishes_equal_from_one_off() {
    let all_nine = [-9; LANES];
    let mut one_off = all_nine;
    one_off[5] = -8;
    assert_eq!(reference(Op::AllEqual, &all_nine), vec![1; LANES]);
    assert_eq!(reference(Op::AllEqual, &one_off), vec![0; LANES]);
    // the two quads of the mixed input hold different values, so a quad base that is dropped shows
    assert_ne!(reference(Op::LaneAt(1), &MIXED), vec![MIXED[1]; LANES]);
}

#[test]
fn the_mutants_fail_by_value() {
    let mut one_off = [-9; LANES];
    one_off[5] = -8;
    let cases = [
        (Op::AcrossX, Mutant::XMaskIsY, MIXED),
        (Op::LaneAt(1), Mutant::NoQuadBase, MIXED),
        (Op::AllEqual, Mutant::AnyForAll, one_off),
    ];
    for (op, mutant, input) in cases {
        assert_ne!(
            run(op, mutant, &input),
            reference(op, &input),
            "{op:?} {mutant:?}"
        );
    }
}
