// SPDX-License-Identifier: MIT OR Apache-2.0
//! `try_build_plan` types every refusal of a contraction cell
//! ([`PlanError::UnservableContraction`]), so every emitter inherits it and none panics.
//!
//! Until 0.20.1 the contraction arm of the plan gate `assert!`ed. That was right while the
//! only keys were the canonical rank-2/3 ones; a key derived from declared M/N/K and batch
//! roles (`unpopped_vocab::ContractionRoles`, 0.19.0) can name a cell the op cannot pair
//! with, so the refusal is now reachable by *requesting* a cell. One case per
//! [`ContractionDecline`], each beside the control that derives, so a decline cannot be
//! a gate that refuses everything.

use unpopped::ir::{ContractionAxes, OpDef, input, reduced};
use unpopped::plan::{ContractionDecline, PlanError, try_build_plan};
use unpopped_vocab::{
    ArchSku, CellFacts, ContractionRoles, ElementKind, OperandDesc, StructureKey,
    derive_structure_key,
};

fn od(shape: &[i64], strides: &[i64]) -> OperandDesc {
    OperandDesc::new(shape.len(), shape, strides, ElementKind::F32, 256)
}

/// `[8,16] . [16,4] -> [8,4]`, dense row-major.
fn rank2() -> [OperandDesc; 3] {
    [
        od(&[8, 16], &[16, 1]),
        od(&[16, 4], &[4, 1]),
        od(&[8, 4], &[4, 1]),
    ]
}

/// `[4,8,16] . [4,16,4] -> [4,8,4]`, dense row-major.
fn rank3() -> [OperandDesc; 3] {
    [
        od(&[4, 8, 16], &[128, 16, 1]),
        od(&[4, 16, 4], &[64, 4, 1]),
        od(&[4, 8, 4], &[32, 4, 1]),
    ]
}

fn facts(rank: usize, batch: usize) -> CellFacts {
    let mut f = CellFacts::new();
    f.contraction_declared = true;
    f.weight_operand = Some(1);
    f.contraction_roles = Some(ContractionRoles::gemm(rank, batch));
    f
}

fn key(ops: &[OperandDesc], rank: usize, batch: usize) -> StructureKey {
    derive_structure_key(ops, &facts(rank, batch), ArchSku::Sm89).expect("the key derives")
}

fn matmul() -> OpDef {
    OpDef::contraction(
        "mm",
        &[ElementKind::F32],
        ContractionAxes::matmul(),
        reduced(0),
    )
}

fn batched() -> OpDef {
    OpDef::contraction(
        "bmm",
        &[ElementKind::F32],
        ContractionAxes::batched_matmul(),
        reduced(0),
    )
}

fn declined(op: &OpDef, k: &StructureKey) -> ContractionDecline {
    match try_build_plan(op, k) {
        Err(PlanError::UnservableContraction(d)) => d,
        other => panic!(
            "expected an unservable-contraction decline, got {:?}",
            other.map(|_| ())
        ),
    }
}

#[test]
fn the_canonical_rank_2_and_rank_3_cells_still_plan() {
    assert!(try_build_plan(&matmul(), &key(&rank2(), 2, 0)).is_ok());
    assert!(try_build_plan(&batched(), &key(&rank3(), 3, 1)).is_ok());
}

#[test]
fn rank_four_and_above_declines_by_its_roles() {
    // [B0,B1,M,K] . [B0,B1,K,N]: two batch axes, which v1 lowers neither as a matmul nor
    // as a batched matmul. The control above is the same call at the rank it can serve.
    let ops = [
        od(&[2, 3, 8, 16], &[384, 128, 16, 1]),
        od(&[2, 3, 16, 4], &[192, 64, 4, 1]),
        od(&[2, 3, 8, 4], &[96, 32, 4, 1]),
    ];
    let k = key(&ops, 4, 2);
    for op in [matmul(), batched()] {
        assert!(matches!(
            declined(&op, &k),
            ContractionDecline::RolesInvalid { .. }
        ));
    }
}

#[test]
fn a_key_with_no_contraction_facts_declines() {
    // The same operands keyed as a non-contraction.
    let k = derive_structure_key(&rank2(), &CellFacts::new(), ArchSku::Sm89).unwrap();
    assert!(k.contraction.is_none());
    assert_eq!(
        declined(&matmul(), &k),
        ContractionDecline::KeyHasNoContractionFacts
    );
}

#[test]
fn an_epilogue_that_reads_an_operand_declines() {
    let op = OpDef::contraction(
        "bad",
        &[ElementKind::F32],
        ContractionAxes::matmul(),
        reduced(0) + input(0),
    );
    assert_eq!(
        declined(&op, &key(&rank2(), 2, 0)),
        ContractionDecline::EpilogueInadmissible
    );
}

#[test]
fn a_rank_three_key_with_no_declared_batch_disagrees_with_the_batched_op() {
    // Rank 3, but the caller declared no batch axis: the key is a non-batched cell.
    let k = key(&rank3(), 3, 0);
    assert!(k.contraction.unwrap().batch.is_none());
    assert_eq!(
        declined(&batched(), &k),
        ContractionDecline::BatchDisagrees {
            op_batched: true,
            key_batched: false
        }
    );
}

#[test]
fn a_real_batch_axis_stored_inside_the_matrix_axes_declines_and_a_broadcast_one_does_not() {
    // lhs stored [M,B,K]: logical [B,M,K] with the batch stride inside M's.
    let mut ops = rank3();
    ops[0] = od(&[4, 8, 16], &[16, 64, 1]);
    assert_eq!(
        declined(&batched(), &key(&ops, 3, 1)),
        ContractionDecline::BatchNotOutermost { operand: 0 }
    );
    // rhs stored [K,B,N].
    let mut ops = rank3();
    ops[1] = od(&[4, 16, 4], &[4, 16, 1]);
    assert_eq!(
        declined(&batched(), &key(&ops, 3, 1)),
        ContractionDecline::BatchNotOutermost { operand: 1 }
    );
    // Control: the same lhs layout with the batch axis broadcast (stride 0) occupies no
    // storage, so there is nothing for it to be inside of.
    let mut ops = rank3();
    ops[0] = od(&[4, 8, 16], &[0, 16, 1]);
    assert!(try_build_plan(&batched(), &key(&ops, 3, 1)).is_ok());
}
