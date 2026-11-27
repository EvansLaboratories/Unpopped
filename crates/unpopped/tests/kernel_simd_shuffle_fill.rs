// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Metal `simd_shuffle_and_fill_up/down` and `simd_shuffle_rotate_up/down`** (survey row 40, read from the Metal Shading Language
//! Specification 4.1, footer 2026-06-04, section 6.9.2 "SIMD-group functions": `docs/ir-union-evidence/conformance-idiom-gaps-verified-2026-11-27.md`).
//! The IR has one-operand `Shuffle` modes only, so these are built from two of them and a `Select`: `fill_up` by `delta` reads data
//! from lane `i - delta` and, for the lower `delta` lanes, `filling` from lane `(i - delta) mod n` (the IR's Euclidean `Idx` rule);
//! `fill_down` is the mirror; rotate is an `Idx` shuffle at `(i -+ delta) mod n`. Eight lanes, `delta` 2, data `lane + 1`, filling
//! `10 * (lane + 1)`; the expected arrays are written by hand from the spec's own worked example (lower lanes take the filling's
//! upper lanes, upper lanes take the filling's lower lanes). The one-operand `modulo` overloads are not built. Mutants change a lane
//! by value.

use unpopped::exec::{ExecId, LaunchShape, Scope};
use unpopped::ir::{BinaryOp, CastSpec, RoundMode, Saturation, ScalarExpr};
use unpopped::kernel::{
    Block, BufferRole, CollectiveKind, KernelFn, LocalDecl, Param, Place, ShuffleMode, Stmt,
    check_kernel,
};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch_with_subgroup;
use unpopped_vocab::ElementKind;

const I32: ElementKind = ElementKind::I32;
const N: i64 = 8;
const DELTA: i64 = 2;

#[derive(Clone, Copy, PartialEq)]
enum Op {
    FillUp,
    FillDown,
    RotateUp,
    RotateDown,
    /// Mutant: a fill that never reads the filling operand (a plain no-wrap shuffle).
    FillUpIgnoresFilling,
}

fn ci(v: i64) -> ScalarExpr {
    ScalarExpr::ConstInt(v, I32)
}
fn l(i: u32) -> ScalarExpr {
    ScalarExpr::Local(i)
}
fn bin(op: BinaryOp, a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Binary(op, Box::new(a), Box::new(b))
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
fn shuffle(mode: ShuffleMode, operand: ScalarExpr, value: u32, result: u32) -> Stmt {
    Stmt::Collective {
        kind: CollectiveKind::Shuffle { mode, operand },
        value: l(value),
        result,
        scope: Scope::Subgroup,
    }
}

/// Locals: 0 data, 1 filling, 2 shuffled data, 3 shuffled filling, 4 the result.
fn body(op: Op) -> Vec<Stmt> {
    let lane_plus = |d: i64| ScalarExpr::Add(Box::new(lane()), Box::new(ci(d)));
    let mut v = vec![
        Stmt::Let {
            local: 0,
            init: lane_plus(1),
        },
        Stmt::Let {
            local: 1,
            init: ScalarExpr::Mul(Box::new(lane_plus(1)), Box::new(ci(10))),
        },
    ];
    let result = match op {
        Op::FillUp | Op::FillUpIgnoresFilling => {
            v.push(shuffle(ShuffleMode::Up, ci(DELTA), 0, 2));
            v.push(shuffle(ShuffleMode::Idx, lane_plus(-DELTA), 1, 3));
            let from = if op == Op::FillUp { l(3) } else { l(2) };
            ScalarExpr::Select(
                Box::new(bin(BinaryOp::CmpGe, lane(), ci(DELTA))),
                Box::new(l(2)),
                Box::new(from),
            )
        }
        Op::FillDown => {
            v.push(shuffle(ShuffleMode::Down, ci(DELTA), 0, 2));
            v.push(shuffle(ShuffleMode::Idx, lane_plus(DELTA), 1, 3));
            ScalarExpr::Select(
                Box::new(bin(BinaryOp::CmpLt, lane(), ci(N - DELTA))),
                Box::new(l(2)),
                Box::new(l(3)),
            )
        }
        Op::RotateUp => {
            v.push(shuffle(ShuffleMode::Idx, lane_plus(-DELTA), 0, 2));
            l(2)
        }
        Op::RotateDown => {
            v.push(shuffle(ShuffleMode::Idx, lane_plus(DELTA), 0, 2));
            l(2)
        }
    };
    v.push(Stmt::Let {
        local: 4,
        init: result,
    });
    v.push(Stmt::Store {
        place: Place::Logical {
            buffer: 0,
            coords: vec![ScalarExpr::Exec(ExecId::ThreadIdx(0))],
        },
        value: l(4),
    });
    v
}

fn run(op: Op) -> Vec<i32> {
    let k = KernelFn::new(
        "fill",
        I32,
        vec![Param::buffer("out", I32, BufferRole::ReadWrite, Some(1))],
        vec![LocalDecl::scalar(I32); 5],
        Block(body(op)),
    )
    .with_launch(LaunchShape::new([N as u32, 1, 1], [1, 1, 1]).with_subgroup(1, 32));
    assert_eq!(check_kernel(&k), Ok(()), "well formed");
    let buf = TypedBuffer::new(I32, vec![N], vec![1], vec![0u8; 4 * N as usize]);
    let mut args = vec![KernelArg::Buffer(buf)];
    run_launch_with_subgroup(&k, &mut args, Some(N as u32)).expect("runs");
    let KernelArg::Buffer(b) = &args[0] else {
        unreachable!()
    };
    b.bytes
        .chunks(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

#[test]
fn fill_up_shifts_data_up_and_takes_the_fillings_upper_lanes() {
    assert_eq!(run(Op::FillUp), [70, 80, 1, 2, 3, 4, 5, 6]);
}

#[test]
fn fill_down_shifts_data_down_and_takes_the_fillings_lower_lanes() {
    assert_eq!(run(Op::FillDown), [3, 4, 5, 6, 7, 8, 10, 20]);
}

#[test]
fn rotate_wraps_the_data_itself() {
    assert_eq!(run(Op::RotateUp), [7, 8, 1, 2, 3, 4, 5, 6]);
    assert_eq!(run(Op::RotateDown), [3, 4, 5, 6, 7, 8, 1, 2]);
}

#[test]
fn a_fill_that_ignores_its_filling_fails_by_value() {
    let bad = run(Op::FillUpIgnoresFilling);
    assert_eq!(bad, [1, 2, 1, 2, 3, 4, 5, 6]);
    assert_ne!(bad, run(Op::FillUp));
}
