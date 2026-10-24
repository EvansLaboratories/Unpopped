// SPDX-License-Identifier: MIT OR Apache-2.0
//! The role-driven contraction derivation (`CellFacts::contraction_roles`, 0.19.0):
//! what KISS's vectors do not pin, and the controls that make each null mean something.
//!
//! The vectors (`kiss_derivation_vectors.rs`) are the byte-match leg. These tests hold
//! the seams around it: the positional path stays byte-identical, the declared precision
//! facts reach the token, and the list-capacity and rank limits decline by name.

use unpopped_vocab::{
    ArchSku, AxisRef, BatchRoleDecline, CellFacts, ContractionRoles, DeriveDecline, ElementKind,
    MpCode, OperandDesc, OperandRole, derive_structure_key_token,
};

fn od(shape: &[i64], strides: &[i64], dt: ElementKind) -> OperandDesc {
    OperandDesc::new(shape.len(), shape, strides, dt, 256)
}

/// `[x f16 [64,128], W f16 [128,32], out f16 [64,32]]`, no operand states a role.
fn gemm() -> [OperandDesc; 3] {
    [
        od(&[64, 128], &[128, 1], ElementKind::F16),
        od(&[128, 32], &[32, 1], ElementKind::F16),
        od(&[64, 32], &[32, 1], ElementKind::F16),
    ]
}

fn axes() -> ContractionRoles {
    ContractionRoles::new(AxisRef::new(0, 0), AxisRef::new(1, 1), AxisRef::new(0, 1))
}

fn facts(a: Option<ContractionRoles>) -> CellFacts {
    let mut f = CellFacts::new();
    f.contraction_declared = true;
    f.weight_operand = Some(1);
    f.contraction_roles = a;
    f
}

fn tail(token: &str) -> String {
    token
        .split('|')
        .nth(9)
        .expect("contraction field")
        .to_owned()
}

fn key(ops: &[OperandDesc], f: &CellFacts) -> Result<String, DeriveDecline> {
    derive_structure_key_token(ops, f, ArchSku::Sm89)
}

#[test]
fn roles_are_required_and_the_accumulator_defaults_to_the_compute_dtype() {
    // No roles, no key: they are never inferred from position.
    assert_eq!(
        key(&gemm(), &facts(None)),
        Err(DeriveDecline::ContractionRolesNotDeclared)
    );
    // Declared roles, no accumulator: the compute dtype, operand 0's (§6.7-0006).
    let roles = key(&gemm(), &facts(Some(axes()))).unwrap();
    assert_eq!(tail(&roles), "csss/d16/f16/f16/f16/st", "{roles}");
}

#[test]
fn the_legacy_op_bridge_declares_its_own_roles_and_keeps_its_bytes() {
    // The bridge declares positional GEMM roles, an `f32` accumulator for `f16`
    // and `st`: the bytes 0.18 derived, now from the one role-driven path.
    let t = unpopped_vocab::try_structure_key_token(
        unpopped_vocab::OpCategory::Gemm,
        &gemm(),
        ArchSku::Sm89,
    )
    .unwrap();
    assert_eq!(tail(&t), "csss/d16/f16/f32/f16/st", "{t}");
    // ... and the same cell through the conformant entry, declaring the same facts,
    // is byte-identical: one behaviour, reached two ways.
    let mut f = facts(Some(ContractionRoles::gemm(2, 0)));
    f.accumulator = Some(ElementKind::F32);
    assert_eq!(key(&gemm(), &f).unwrap(), t);
    // An f32 contraction declares `rm` at the bridge (a permission to round its
    // inputs, never a requirement), as the GEMM routing policy does; through the
    // conformant entry that is a declaration.
    let mut g = gemm();
    for o in &mut g {
        o.dtype = ElementKind::F32;
    }
    let bridged = unpopped_vocab::try_structure_key_token(
        unpopped_vocab::OpCategory::Gemm,
        &g,
        ArchSku::Sm89,
    )
    .unwrap();
    assert_eq!(tail(&bridged), "csss/d16/f32/f32/f32/rm", "{bridged}");
    assert_eq!(
        tail(&key(&g, &facts(Some(axes()))).unwrap()),
        "csss/d16/f32/f32/f32/st"
    );
}

#[test]
fn declared_accumulator_and_math_fidelity_reach_the_token() {
    let mut f = facts(Some(axes()));
    f.accumulator = Some(ElementKind::F32);
    f.math_fidelity = MpCode::Rm;
    let t = key(&gemm(), &f).unwrap();
    assert_eq!(tail(&t), "csss/d16/f16/f32/f16/rm", "{t}");
}

#[test]
fn the_output_is_the_last_operand_when_no_role_is_stated_and_a_stated_output_is_required() {
    // Control: no roles stated, the last operand is the output (the pre-sk5 convention).
    assert!(key(&gemm(), &facts(Some(axes()))).is_ok());
    // Roles stated, none an output.
    let ops = gemm().map(|o| o.with_role(OperandRole::Input));
    assert_eq!(
        key(&ops, &facts(Some(axes()))),
        Err(DeriveDecline::ContractionNoOutput)
    );
}

fn batched() -> [OperandDesc; 3] {
    [
        od(&[4, 8, 16], &[128, 16, 1], ElementKind::F16),
        od(&[4, 16, 8], &[128, 8, 1], ElementKind::F16),
        od(&[4, 8, 8], &[64, 8, 1], ElementKind::F16),
    ]
}

fn batched_axes() -> ContractionRoles {
    ContractionRoles::new(AxisRef::new(0, 1), AxisRef::new(1, 2), AxisRef::new(0, 2))
}

#[test]
fn a_batch_list_longer_than_its_capacity_declines_as_its_first_bad_entry_does() {
    // Control: one declared batch axis derives and is batched.
    let ok = key(
        &batched(),
        &facts(Some(batched_axes().with_batch_axes(&[0]))),
    )
    .unwrap();
    assert!(tail(&ok).contains("/bt/"), "{ok}");
    // 40 entries (capacity 16): the second repeats the first, whatever follows.
    let long = [0usize; 40];
    assert_eq!(
        key(
            &batched(),
            &facts(Some(batched_axes().with_batch_axes(&long)))
        ),
        Err(DeriveDecline::Batch(BatchRoleDecline::DuplicateAxis))
    );
    // An axis no frame has, however large, is out of range and not wrapped into one.
    for bad in [3, 255, 256, usize::MAX] {
        assert_eq!(
            key(
                &batched(),
                &facts(Some(batched_axes().with_batch_axes(&[bad])))
            ),
            Err(DeriveDecline::Batch(BatchRoleDecline::AxisOutOfRange)),
            "axis {bad}"
        );
    }
}

#[test]
fn an_order_the_token_cannot_spell_at_rank_four_declines_and_its_identity_control_derives() {
    // [B0,B1,M,K] x [B0,B1,K,N] -> [B0,B1,M,N], batch axis 0 only.
    let rhs = od(&[2, 3, 16, 4], &[192, 64, 4, 1], ElementKind::F16);
    let out = od(&[2, 3, 8, 4], &[96, 32, 4, 1], ElementKind::F16);
    let roles = ContractionRoles::new(AxisRef::new(0, 2), AxisRef::new(1, 3), AxisRef::new(0, 3))
        .with_batch_axes(&[0]);
    // Control: lhs row-major, identity order.
    let lhs = od(&[2, 3, 8, 16], &[384, 128, 16, 1], ElementKind::F16);
    assert!(key(&[lhs, rhs, out], &facts(Some(roles))).is_ok());
    // The same lhs with its last two axes stored swapped: a packed, non-identity order.
    let lhs_t = od(&[2, 3, 8, 16], &[384, 128, 1, 8], ElementKind::F16);
    assert_eq!(
        key(&[lhs_t, rhs, out], &facts(Some(roles))),
        Err(DeriveDecline::ContractionOutOfScope)
    );
}
