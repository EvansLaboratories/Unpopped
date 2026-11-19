// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Scan with a strided operand declines, typed** (pm-kernels `[FINDING]` strided-reduction-scan-gate, from rockfish,
//! 2026-11-19, who read the source and did not run it). Run, it was real: the plan gate requires a Scan's streamed input
//! to be contiguous and unflipped, and `try_build_plan` PANICKED on a transposed or row-strided operand where
//! `check_row_reduce` already returned `PlanError::InadmissibleOpInCell`. KISS-EMIT 6.8-0004: no panic on any
//! `(OpDef, structure_key)`. The census (`adversarial_input_panic_census.rs`) has no Scan, so it never saw it.
//!
//! Fixed: the Scan operand-layout checks are `require!` in `validate_scan`, hoisted into `try_build_plan` like
//! `check_row_reduce`. Strided Scan/RowReduce itself is NOT supported (ROADMAP E16, demand-gated); this pins the refusal.

use std::panic::{AssertUnwindSafe, catch_unwind};

use unpopped::ir::{OpDef, ReduceOp, ReduceStage, input, reduced};
use unpopped::plan::{PlanError, try_build_plan};
use unpopped_vocab::{ArchSku, ElementKind, OpCategory, OperandDesc, structure_key};

const F32: ElementKind = ElementKind::F32;
const SHAPE: [i64; 3] = [2, 3, 4];

#[derive(Debug, PartialEq)]
enum Outcome {
    Plans,
    Declines,
    Panics,
}

fn outcome(cat: OpCategory, op: &OpDef, strides: [i64; 3]) -> Outcome {
    let a = OperandDesc::new(3, &SHAPE, &strides, F32, 256);
    let key = structure_key(cat, &[a, a], ArchSku::Sm89);
    match catch_unwind(AssertUnwindSafe(|| try_build_plan(op, &key).map(|_| ()))) {
        Ok(Ok(())) => Outcome::Plans,
        Ok(Err(PlanError::InadmissibleOpInCell { .. })) => Outcome::Declines,
        Ok(Err(other)) => panic!("an unexpected decline: {other:?}"),
        Err(_) => Outcome::Panics,
    }
}

const DENSE: [i64; 3] = [12, 4, 1];
const TRANSPOSED: [i64; 3] = [1, 2, 6];
const ROW_STRIDED: [i64; 3] = [24, 8, 1];

fn scan(axis: u8) -> OpDef {
    OpDef::scan_simple("s", &[F32], ReduceOp::Sum, axis, false, false)
}

#[test]
fn a_dense_scan_plans() {
    for axis in 0..3 {
        assert_eq!(
            outcome(OpCategory::Scan, &scan(axis), DENSE),
            Outcome::Plans
        );
    }
}

#[test]
fn a_transposed_or_row_strided_scan_declines_rather_than_panics() {
    for axis in 0..3 {
        for layout in [TRANSPOSED, ROW_STRIDED] {
            assert_eq!(
                outcome(OpCategory::Scan, &scan(axis), layout),
                Outcome::Declines,
                "axis {axis}, strides {layout:?}"
            );
        }
    }
}

#[test]
fn row_reduce_is_the_control_that_already_declined() {
    let op = OpDef::row_reduce(
        "rr",
        1,
        &[F32],
        vec![ReduceStage {
            pre: input(0).0,
            op: ReduceOp::Sum,
        }],
        reduced(0),
    );
    // the control: the dense layout plans, the transposed one declines (a decline, not a panic)
    assert_eq!(
        outcome(OpCategory::Normalization, &op, DENSE),
        Outcome::Plans
    );
    assert_eq!(
        outcome(OpCategory::Normalization, &op, TRANSPOSED),
        Outcome::Declines
    );
}
