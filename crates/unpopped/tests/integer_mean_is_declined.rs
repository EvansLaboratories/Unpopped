// SPDX-License-Identifier: MIT OR Apache-2.0
//! PM ruling 2026-11-01 (option b): an integer `Mean` reduction has no defined semantics, so the plan
//! gate declines it with a typed `PlanError` instead of building a plan that only the oracle's and the
//! emitters' asserts later refuse. No new semantics until an emitter lane needs them.

use unpopped::ir::{OpDef, ReduceOp, input};
use unpopped::plan::{PlanError, try_build_plan};
use unpopped_vocab::{ArchSku, ElementKind, OpCategory, OperandDesc, structure_key};

const INTS: [ElementKind; 8] = [
    ElementKind::I8,
    ElementKind::U8,
    ElementKind::I16,
    ElementKind::U16,
    ElementKind::I32,
    ElementKind::U32,
    ElementKind::I64,
    ElementKind::U64,
];

fn plan_of(dtype: ElementKind, op: ReduceOp) -> Result<(), PlanError> {
    let def = OpDef::reduction("r", 1, &[dtype], input(0), op);
    let d_in = OperandDesc::new(2, &[2, 8], &[8, 1], dtype, 256);
    let d_out = OperandDesc::new(1, &[2], &[1], dtype, 256);
    let key = structure_key(OpCategory::Reduction, &[d_in, d_out], ArchSku::Sm89);
    try_build_plan(&def, &key).map(|_| ())
}

#[test]
fn an_integer_mean_is_a_typed_decline_at_every_integer_dtype() {
    for dt in INTS {
        let r = plan_of(dt, ReduceOp::Mean);
        match r {
            Err(PlanError::InadmissibleOpAtDtype { dtype, detail }) => {
                assert_eq!(dtype, dt);
                assert!(detail.contains("Mean"), "{dt:?}: names the op: {detail}");
            }
            other => panic!("{dt:?}: expected a typed decline, got {other:?}"),
        }
    }
}

#[test]
fn the_other_integer_folds_and_a_float_mean_still_plan() {
    for dt in INTS {
        for op in [ReduceOp::Sum, ReduceOp::Prod, ReduceOp::Max, ReduceOp::Min] {
            plan_of(dt, op).unwrap_or_else(|e| panic!("{dt:?} {op:?} must plan: {e}"));
        }
    }
    for dt in [ElementKind::F32, ElementKind::F64] {
        plan_of(dt, ReduceOp::Mean).unwrap_or_else(|e| panic!("{dt:?} Mean must plan: {e}"));
    }
}
