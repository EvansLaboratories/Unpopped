// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Float `Max`/`Min`/`Sum` collectives over 32 lanes: NaN, signed zero, infinities** (task from `unpopped`, 2026-11-27, rules from
//! its `docs/design-float-collective-nan.md` on branch `lane/doc-float-collective-nan`; this file read the rule from the code,
//! `crates/unpopped/src/oracle/kernel_collective.rs` `combine` and `nan_first` at `origin/main` 30d04900, and the doc matches it).
//! The rule: `Max`/`Min` propagate NaN (once a lane or the running fold is NaN the result is NaN) and keep the FIRST of equal
//! values (a tie never replaces, so `Max(-0.0, +0.0)` is the first zero's sign); `Sum` is the lane-order left fold rounded to `F32`
//! at each step, and `+inf` plus `-inf` is NaN. The expected vectors are written by hand or by a plain Rust `f32` fold, not by the
//! oracle's combine. One subgroup of 32 lanes; lane `i` contributes `in[i]` and stores the reduction to `out[i]`.
//!
//! Mutants, each of which must fail the vector: a NaN-ignoring `Max`/`Min` (the `fmaxf` behaviour; `MaxNum` is not admitted as a
//! collective fold, so it is Rust's `f32::max`), the tie going to the last lane, and the `Sum` in reverse or pairwise-tree order
//! are Rust folds shown to differ from the oracle's vector by value.

use unpopped::exec::{ExecId, LaunchShape, Scope};
use unpopped::ir::{ReduceOp, ScalarExpr};
use unpopped::kernel::{
    Block, BufferRole, CollectiveKind, KernelFn, LocalDecl, Param, Place, Stmt, check_kernel,
};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch_with_subgroup;
use unpopped_vocab::ElementKind;

const F32: ElementKind = ElementKind::F32;
const LANES: usize = 32;

fn at(buffer: u32) -> Place {
    Place::Logical {
        buffer,
        coords: vec![ScalarExpr::Exec(ExecId::ThreadIdx(0))],
    }
}

/// Load `in[lane]` (local 0), reduce it over the subgroup (local 1), store `out[lane]`.
fn kernel(op: ReduceOp) -> KernelFn {
    let body = vec![
        Stmt::Load {
            local: 0,
            place: at(0),
        },
        Stmt::Collective {
            kind: CollectiveKind::Reduce(op),
            value: ScalarExpr::Local(0),
            result: 1,
            scope: Scope::Subgroup,
        },
        Stmt::Store {
            place: at(1),
            value: ScalarExpr::Local(1),
        },
    ];
    KernelFn::new(
        "float_reduce",
        ElementKind::I32,
        vec![
            Param::buffer("in", F32, BufferRole::Read, Some(1)),
            Param::buffer("out", F32, BufferRole::ReadWrite, Some(1)),
        ],
        vec![LocalDecl::scalar(F32), LocalDecl::scalar(F32)],
        Block(body),
    )
    .with_launch(LaunchShape::new([LANES as u32, 1, 1], [1, 1, 1]).with_subgroup(1, 32))
}

/// The 32 results of `op` over `input`, as raw `f32` bits.
fn run(op: ReduceOp, input: &[f32; LANES]) -> Vec<u32> {
    let k = kernel(op);
    assert_eq!(check_kernel(&k), Ok(()), "well formed");
    let bytes: Vec<u8> = input.iter().flat_map(|v| v.to_le_bytes()).collect();
    let shape = vec![LANES as i64];
    let a = TypedBuffer::new(F32, shape.clone(), vec![1], bytes);
    let b = TypedBuffer::new(F32, shape, vec![1], vec![0u8; 4 * LANES]);
    let mut args = vec![KernelArg::Buffer(a), KernelArg::Buffer(b)];
    run_launch_with_subgroup(&k, &mut args, Some(LANES as u32)).expect("runs");
    let KernelArg::Buffer(out) = &args[1] else {
        unreachable!()
    };
    out.bytes
        .chunks(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn lanes_of(base: f32, set: &[(usize, f32)]) -> [f32; LANES] {
    let mut v = [base; LANES];
    for &(i, x) in set {
        v[i] = x;
    }
    v
}

fn all_nan(bits: &[u32]) -> bool {
    bits.iter().all(|b| f32::from_bits(*b).is_nan())
}

fn all_bits(bits: &[u32], want: f32) -> bool {
    bits.iter().all(|b| *b == want.to_bits())
}

#[test]
fn a_nan_in_one_lane_makes_every_lane_nan_for_max_and_min() {
    let v = lanes_of(1.0, &[(13, f32::NAN)]);
    assert!(all_nan(&run(ReduceOp::Max, &v)));
    assert!(all_nan(&run(ReduceOp::Min, &v)));
    let first = lanes_of(1.0, &[(0, f32::NAN)]);
    assert!(all_nan(&run(ReduceOp::Max, &first)), "NaN in lane 0");
    let last = lanes_of(1.0, &[(31, f32::NAN)]);
    assert!(all_nan(&run(ReduceOp::Min, &last)), "NaN in lane 31");
}

#[test]
fn a_zero_tie_keeps_the_first_lanes_sign() {
    // every other lane is below (Max) or above (Min) both zeros, so the zeros are the whole contest
    let neg_first = lanes_of(-1.0, &[(0, -0.0), (5, 0.0)]);
    assert!(
        all_bits(&run(ReduceOp::Max, &neg_first), -0.0),
        "-0.0 first"
    );
    let pos_first = lanes_of(-1.0, &[(0, 0.0), (5, -0.0)]);
    assert!(all_bits(&run(ReduceOp::Max, &pos_first), 0.0), "+0.0 first");
    let neg_first_min = lanes_of(1.0, &[(2, -0.0), (9, 0.0)]);
    assert!(all_bits(&run(ReduceOp::Min, &neg_first_min), -0.0));
    let pos_first_min = lanes_of(1.0, &[(2, 0.0), (9, -0.0)]);
    assert!(all_bits(&run(ReduceOp::Min, &pos_first_min), 0.0));
}

#[test]
fn infinities_in_a_sum_follow_ieee() {
    let both = lanes_of(1.0, &[(3, f32::INFINITY), (20, f32::NEG_INFINITY)]);
    assert!(all_nan(&run(ReduceOp::Sum, &both)), "+inf plus -inf is NaN");
    let up = lanes_of(1.0, &[(3, f32::INFINITY)]);
    assert!(all_bits(&run(ReduceOp::Sum, &up), f32::INFINITY));
    let down = lanes_of(1.0, &[(20, f32::NEG_INFINITY)]);
    assert!(all_bits(&run(ReduceOp::Sum, &down), f32::NEG_INFINITY));
}

/// Values for which the order of an `f32` sum changes the bits: `1e8` in lane 0, `-1e8` in lane 2, ones elsewhere. Left to right
/// the first one is absorbed by `1e8` (an `f32` spacing of 8 there) and 29 ones survive after the cancellation: 29. Reversed gives
/// 32, and the pairwise tree gives another value; the test asserts all three differ.
fn order_sensitive() -> [f32; LANES] {
    let mut v = [1.0_f32; LANES];
    v[0] = 1.0e8;
    v[2] = -1.0e8;
    v
}

fn left_fold(v: &[f32; LANES]) -> f32 {
    v.iter().fold(0.0_f32, |a, b| a + b)
}

#[test]
fn a_sum_is_the_lane_order_left_fold_in_f32() {
    let v = order_sensitive();
    let want = left_fold(&v);
    assert!(all_bits(&run(ReduceOp::Sum, &v), want));
    assert_eq!(want, 29.0, "one 1.0 is absorbed by 1e8 before it cancels");
}

#[test]
fn the_mutants_fail_the_vectors_by_value() {
    // NaN-ignoring Max/Min (`fmaxf`/`fminf`, Rust's `f32::max`/`min`): 1.0 where the oracle's vector is NaN
    let nan = lanes_of(1.0, &[(13, f32::NAN)]);
    let max_ignoring = nan.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let min_ignoring = nan.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    assert_eq!((max_ignoring, min_ignoring), (1.0, 1.0));
    assert!(!run(ReduceOp::Max, &nan).contains(&max_ignoring.to_bits()));
    assert!(!run(ReduceOp::Min, &nan).contains(&min_ignoring.to_bits()));
    // the tie going to the last lane (`>=`): lanes 0 and 5, -0.0 then +0.0, picks +0.0 where the oracle gives -0.0
    let tie = lanes_of(-1.0, &[(0, -0.0), (5, 0.0)]);
    let last_wins = tie
        .iter()
        .fold(f32::NEG_INFINITY, |a, &b| if b >= a { b } else { a });
    assert_ne!(run(ReduceOp::Max, &tie)[0], last_wins.to_bits());
    // a Sum in the wrong order: reversed, and a pairwise tree
    let v = order_sensitive();
    let reversed = v.iter().rev().fold(0.0_f32, |a, b| a + b);
    let mut level: Vec<f32> = v.to_vec();
    while level.len() > 1 {
        level = level.chunks(2).map(|p| p[0] + p[1]).collect();
    }
    let got = run(ReduceOp::Sum, &v)[0];
    assert_ne!(got, reversed.to_bits());
    assert_ne!(got, level[0].to_bits());
}
