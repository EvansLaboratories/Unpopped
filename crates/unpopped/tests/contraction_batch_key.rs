// SPDX-License-Identifier: MIT OR Apache-2.0
//! **CuTe survey row 39, key half: the structure key of a rank-3 batched contraction.** The value half (a rank-3 batched matmul run
//! through the converter against kiss-ref, bit-exact) is `kiss_ref_diff::batched_matmul_oracle_eq_kiss_ref`; this is the key half.
//! The expected tokens are KISS's own: its conformance corpus `structure_key_derivation_vectors.json` at KISS `origin/main`
//! `8828b194`, entries `ctr_rank3_batch_declared` (line 43) and the controls on lines 44 to 47, and the declines on lines 66 to
//! 70, under KISS-CLASSIFY-6.6-0023 (the batch set, keyed by the PRODUCT of its extents, a `b<class>` coordinate if and only if
//! the set is non-empty, no inference from rank, position or extent) and 6.6-0023a, as reported by the KISS lane on 2026-11-27
//! (UNVERIFIED by this lane against the JSON file or the spec, which are in KISS's repository and were not read here). The
//! derivation under test is this repository's `unpopped-vocab::derive_structure_key_token`, an independent implementation.
//! The cells are `f16 [64, 8, 512] . f16 [64, 512, 4096] -> f16 [64, 8, 4096]` on `cuda:sm89`, roles M = (operand 0, axis 1),
//! N = (operand 1, axis 2), K = (operand 0, axis 2), weight slot 1.

use unpopped::ir::{ContractionAxes, OpDef, reduced};
use unpopped::plan::try_build_plan;
use unpopped_vocab::{
    ArchSku, AxisRef, BatchRoleDecline, CellFacts, ContractionRoles, DeriveDecline, ElementKind,
    OperandDesc, derive_structure_key, derive_structure_key_token,
};

const F16: ElementKind = ElementKind::F16;

/// Dense row-major strides for `shape`.
fn strides(shape: &[i64]) -> Vec<i64> {
    let mut s = vec![1; shape.len()];
    for d in (0..shape.len().saturating_sub(1)).rev() {
        s[d] = s[d + 1] * shape[d + 1];
    }
    s
}

fn desc(shape: &[i64]) -> OperandDesc {
    OperandDesc::new(shape.len(), shape, &strides(shape), F16, 256)
}

/// The three operands of `lhs . rhs -> out` and the cell facts with the given roles (a declared batch axis, or none).
fn facts_and_operands(
    lhs: &[i64],
    rhs: &[i64],
    out: &[i64],
    roles: ContractionRoles,
) -> ([OperandDesc; 3], CellFacts) {
    let mut facts = CellFacts::new();
    facts.contraction_declared = true;
    facts.weight_operand = Some(1);
    facts.contraction_roles = Some(roles);
    ([desc(lhs), desc(rhs), desc(out)], facts)
}

/// The token of `lhs . rhs -> out` under `roles`.
fn token(
    lhs: &[i64],
    rhs: &[i64],
    out: &[i64],
    roles: ContractionRoles,
) -> Result<String, DeriveDecline> {
    let (ops, facts) = facts_and_operands(lhs, rhs, out, roles);
    derive_structure_key_token(&ops, &facts, ArchSku::Sm89)
}

fn rank3_roles(batch: &[usize]) -> ContractionRoles {
    ContractionRoles::new(AxisRef::new(0, 1), AxisRef::new(1, 2), AxisRef::new(0, 2))
        .with_batch_axes(batch)
}

fn rank3(batch: &[usize], b: i64) -> Result<String, DeriveDecline> {
    token(
        &[b, 8, 512],
        &[b, 512, 4096],
        &[b, 8, 4096],
        rank3_roles(batch),
    )
}

fn rank4(batch: &[usize]) -> Result<String, DeriveDecline> {
    let roles = ContractionRoles::new(AxisRef::new(0, 2), AxisRef::new(1, 3), AxisRef::new(0, 3))
        .with_batch_axes(batch);
    token(
        &[4, 64, 8, 512],
        &[4, 64, 512, 4096],
        &[4, 64, 8, 4096],
        roles,
    )
}

const HEAD: &str = "sk8|ctr|f16|cuda:sm89|ix32|grid|";
const COORDS: &str = "co/00/v8/d16/f/f16;co/00/v8/d16/f/f16;co/00/v8/d16/f/f16|-|ctlm/d16/";

#[test]
fn the_declared_rank_3_batch_derives_the_kiss_token_with_a_small_class_batch_coordinate() {
    let want = format!("{HEAD}r3|{COORDS}bs/f16/f16/f16/st");
    assert_eq!(rank3(&[0], 64).as_deref(), Ok(want.as_str()));
}

#[test]
fn batched_is_a_declaration_not_a_rank_or_an_extent() {
    // the same operands with no batch axis declared: no batch coordinate (6.6-0023: no inference from rank)
    let plain = format!("{HEAD}r3|{COORDS}f16/f16/f16/st");
    assert_eq!(rank3(&[], 64).as_deref(), Ok(plain.as_str()));
    // a declared batch of extent 1 is still batched, in the tiny class
    let tiny = format!("{HEAD}r3|{COORDS}bt/f16/f16/f16/st");
    assert_eq!(rank3(&[0], 1).as_deref(), Ok(tiny.as_str()));
}

#[test]
fn the_batch_class_is_that_of_the_product_of_every_declared_batch_extent() {
    // rank 4, batch axes 0 and 1: the product 4 * 64 = 256 is the medium class
    let both = rank4(&[0, 1]).expect("derives");
    assert!(both.contains("|r4|") && both.contains("/bm/"), "{both}");
    // only axis 1: the extent 64 alone is the small class, as at rank 3
    let one = rank4(&[1]).expect("derives");
    assert!(one.contains("/bs/"), "{one}");
    assert_ne!(both, one);
}

#[test]
fn an_unusable_batch_declaration_is_declined_by_name_not_keyed() {
    // KISS's declines are BatchAxisOutOfRange, BatchAxisDuplicate and BatchAxisOverlapsContractionRole (and two more,
    // NotInOutput and NonPositiveExtent, whose inputs this test does not build)
    assert_eq!(
        rank3(&[3], 64),
        Err(DeriveDecline::Batch(BatchRoleDecline::AxisOutOfRange))
    );
    assert_eq!(
        rank3(&[0, 0], 64),
        Err(DeriveDecline::Batch(BatchRoleDecline::DuplicateAxis))
    );
    // axis 1 of the frame is the M role and axis 2 the N role: a batch axis may not overlap a contraction role
    for axis in [1, 2] {
        assert_eq!(
            rank3(&[axis], 64),
            Err(DeriveDecline::Batch(
                BatchRoleDecline::OverlapsContractionRole
            )),
            "batch axis {axis}"
        );
    }
}

#[test]
fn the_kiss_batched_key_plans_a_batched_matmul_at_the_plan_layer() {
    // the key whose token is asserted above is the one the plan layer consumes: a batched matmul op plans against it
    let (ops, facts) = facts_and_operands(
        &[64, 8, 512],
        &[64, 512, 4096],
        &[64, 8, 4096],
        rank3_roles(&[0]),
    );
    let key = derive_structure_key(&ops, &facts, ArchSku::Sm89).expect("the key derives");
    let op = OpDef::contraction("bmm", &[F16], ContractionAxes::batched_matmul(), reduced(0));
    let planned = try_build_plan(&op, &key);
    assert!(planned.is_ok(), "{:?}", planned.err());
}
