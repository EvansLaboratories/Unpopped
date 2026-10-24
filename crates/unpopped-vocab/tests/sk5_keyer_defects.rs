// SPDX-License-Identifier: MIT OR Apache-2.0
//! The four keyer defects Baracuda measured against `unpopped-vocab` 0.16.0
//! (sk4), each pinned through **both** entry points: the conformant
//! [`derive_structure_key`] (explicit facts) and the legacy
//! [`try_structure_key`] (`op: OpCategory`). The KISS derivation vectors cover
//! the first; nothing else covers the second, and the second is what this
//! workspace's ~200 callers use.
//!
//! | | clause | sk4 behaviour |
//! |---|---|---|
//! | (a) | §6.5-0009(b) | a reduced innermost axis derived `v4`/`v8` |
//! | (b) | §6.6-0009 | a collapsed reduction output was keyed as a non-reduction |
//! | (c) | §6.6-0009 | a reduced axis at/past the rank was keyed |
//! | (d) | §6.1-0001 | a reserved dtype at operand 0 was keyed |

use unpopped_vocab::{
    ArchSku, AxisMask, CellClass, CellFacts, ContractionRoles, DeriveDecline, ElementKind,
    OpCategory, OperandDesc, OperandRole, derive_structure_key, structure_key, try_structure_key,
};

fn od(shape: &[i64], strides: &[i64], dt: ElementKind) -> OperandDesc {
    OperandDesc::new(shape.len(), shape, strides, dt, 256)
}

fn reduced(axes: &[u8]) -> CellFacts {
    let mut f = CellFacts::new();
    for a in axes {
        f.reduced_axes.set(*a);
    }
    f
}

// ---- (a) §6.5-0009(b): reduced innermost axis derives v1 -------------------

#[test]
fn a_reduced_innermost_axis_derives_v1_on_every_operand() {
    // [4,8] -> [4,1], axis 1 reduced. Input is f32, forward-unit, 256-aligned,
    // E=8: sk4 derived `v4` for it.
    let ops = [
        od(&[4, 8], &[8, 1], ElementKind::F32),
        od(&[4, 1], &[1, 1], ElementKind::F32),
    ];
    let k = derive_structure_key(&ops, &reduced(&[1]), ArchSku::Sm89).unwrap();
    assert_eq!(k.cell_class, CellClass::Rdc);
    assert_eq!(k.operands[0].vec_width, unpopped_vocab::VecWidth::Scalar);

    // Control: the same input with the *outer* axis reduced keeps its width, so
    // the rule is about the reduced innermost axis and not about reductions.
    let outer = [
        od(&[4, 8], &[8, 1], ElementKind::F32),
        od(&[1, 8], &[8, 1], ElementKind::F32),
    ];
    let k = derive_structure_key(&outer, &reduced(&[0]), ArchSku::Sm89).unwrap();
    assert_eq!(k.cell_class, CellClass::Rdc);
    assert_eq!(k.operands[0].vec_width, unpopped_vocab::VecWidth::V4);
}

#[test]
fn the_legacy_reduction_entry_point_derives_v1_too() {
    let ops = [
        od(&[4, 8], &[8, 1], ElementKind::F32),
        od(&[4, 1], &[1, 1], ElementKind::F32),
    ];
    let k = try_structure_key(OpCategory::Reduction, &ops, ArchSku::Sm89).unwrap();
    assert_eq!(k.cell_class, CellClass::Rdc);
    assert_eq!(k.operands[0].vec_width, unpopped_vocab::VecWidth::Scalar);
    assert_eq!(k.reduce_axes, AxisMask(0b10));
}

// ---- (b) §6.6-0009: a collapsed reduction output declines -------------------

#[test]
fn a_collapsed_reduction_output_declines() {
    let ops = [
        od(&[4, 8], &[8, 1], ElementKind::F32),
        od(&[4], &[1], ElementKind::F32),
    ];
    assert_eq!(
        derive_structure_key(&ops, &reduced(&[1]), ArchSku::Sm89),
        Err(DeriveDecline::CollapsedReduction)
    );
    // The legacy path used to key this as a non-reduction with an empty
    // reduce set; that is the defect.
    assert_eq!(
        try_structure_key(OpCategory::Reduction, &ops, ArchSku::Sm89),
        Err(DeriveDecline::CollapsedReduction)
    );
}

#[test]
fn a_collapsed_output_that_is_not_a_reduction_still_keys() {
    // Control: the decline is about a reduction, not about any rank mismatch.
    // A rank-reducing view under a non-reduction op has no declared reduce set.
    let ops = [
        od(&[4, 8], &[8, 1], ElementKind::F32),
        od(&[8], &[1], ElementKind::F32),
    ];
    assert!(try_structure_key(OpCategory::Normalization, &ops, ArchSku::Sm89).is_ok());
}

// ---- (c) §6.6-0009: a reduced axis at or past the rank declines -------------

#[test]
fn a_reduced_axis_past_the_rank_declines() {
    let ops = [
        od(&[4, 8], &[8, 1], ElementKind::F32),
        od(&[4, 1], &[1, 1], ElementKind::F32),
    ];
    assert_eq!(
        derive_structure_key(&ops, &reduced(&[2]), ArchSku::Sm89),
        Err(DeriveDecline::ReducedAxisOutOfRange { axis: 2 })
    );
    // Control: axis rank-1 is in range.
    assert!(derive_structure_key(&ops, &reduced(&[1]), ArchSku::Sm89).is_ok());
}

// ---- (d) §6.1-0001: a reserved dtype at operand 0 declines -------------------

#[test]
fn a_reserved_dtype_at_operand_0_declines() {
    for dt in [ElementKind::Fp8E4M3FNUZ, ElementKind::Fp8E5M2FNUZ] {
        let ops = [od(&[8], &[1], dt), od(&[8], &[1], dt), od(&[8], &[1], dt)];
        assert_eq!(
            try_structure_key(OpCategory::BinaryElementwise, &ops, ArchSku::Sm89),
            Err(DeriveDecline::ReservedDtypeAtOperand0 { dtype: dt }),
            "{dt:?}"
        );
    }
}

#[test]
fn a_reserved_dtype_at_a_non_primary_operand_still_keys() {
    // §6.6-0015: only operand 0 is a key position. KISS ruled this case on
    // 2026-10-14: our keyer was right, and a "fix" would be the defect.
    let ops = [
        od(&[8], &[1], ElementKind::F16),
        od(&[8], &[1], ElementKind::Fp8E4M3FNUZ),
        od(&[8], &[1], ElementKind::F16),
    ];
    assert!(try_structure_key(OpCategory::BinaryElementwise, &ops, ArchSku::Sm89).is_ok());
}

// ---- the class is derived, and roles reorder --------------------------------

#[test]
fn the_class_is_derived_from_operands_and_declared_facts() {
    let a = od(&[128, 256], &[256, 1], ElementKind::F32);
    let k = |op, ops: &[OperandDesc]| structure_key(op, ops, ArchSku::Sm89).cell_class;
    assert_eq!(k(OpCategory::UnaryElementwise, &[a, a]), CellClass::Ew1);
    assert_eq!(k(OpCategory::BinaryElementwise, &[a, a, a]), CellClass::Ew2);
    assert_eq!(
        k(OpCategory::TernaryElementwise, &[a, a, a, a]),
        CellClass::Ew3
    );
    // Four inputs is not pointwise-by-arity: `oth`.
    assert_eq!(
        k(OpCategory::BinaryElementwise, &[a, a, a, a, a]),
        CellClass::Oth
    );
    // The class does not name the op: a softmax over a same-shape in/out is the
    // same cell class as a unary activation.
    assert_eq!(k(OpCategory::Softmax, &[a, a]), CellClass::Ew1);
    assert_eq!(k(OpCategory::Indexing, &[a, a, a]), CellClass::Ndx);
}

#[test]
fn explicit_roles_put_operands_in_canonical_order() {
    let a = od(&[128, 256], &[256, 1], ElementKind::F32);
    let wide = od(&[128, 256], &[256, 1], ElementKind::F32);
    let narrow_stride = od(&[128, 256], &[512, 2], ElementKind::F32);
    // Output listed first; the strided input must land at sub-key position 1.
    let ops = [
        wide.with_role(OperandRole::Output),
        a.with_role(OperandRole::Input),
        narrow_stride.with_role(OperandRole::Input),
    ];
    let k = derive_structure_key(&ops, &CellFacts::new(), ArchSku::Sm89).unwrap();
    assert_eq!(k.cell_class, CellClass::Ew2);
    assert_eq!(k.n_operands, 3);
    assert_eq!(k.operands[0].contig, unpopped_vocab::Contiguity::Contig);
    assert_eq!(k.operands[1].contig, unpopped_vocab::Contiguity::Strided);
    assert_eq!(k.operands[2].contig, unpopped_vocab::Contiguity::Contig);
}

// ---- the class rules are ordered, and pointwise needs the frame rank --------

#[test]
fn an_index_role_outranks_a_reduced_axis_set() {
    // §6.5-0015: contraction, then index, then reduction. A cell that declares
    // both an index role and a reduced axis is `ndx`, and (§6.6-0017) its reduce
    // field is then `-`, because the field is non-`-` iff the class is `rdc`.
    let ops = [
        od(&[4, 8], &[8, 1], ElementKind::F32),
        od(&[4, 1], &[1, 1], ElementKind::F32),
    ];
    let mut f = reduced(&[1]);
    f.index_declared = true;
    let k = derive_structure_key(&ops, &f, ArchSku::Sm89).unwrap();
    assert_eq!(k.cell_class, CellClass::Ndx);
    assert_eq!(k.reduce_axes, AxisMask::EMPTY);
    assert!(k.to_token().ends_with("|-"), "{}", k.to_token());
}

#[test]
fn a_written_operand_must_have_the_iteration_rank_to_be_pointwise() {
    // `[1,8]` in, `[8]` out: every extent is 1 or the frame extent, but the
    // output is not the iteration rank, so §6.5-0015(4) does not hold.
    let ops = [
        od(&[1, 8], &[8, 1], ElementKind::F32),
        od(&[8], &[1], ElementKind::F32),
    ];
    let k = derive_structure_key(&ops, &CellFacts::new(), ArchSku::Sm89).unwrap();
    assert_eq!(k.cell_class, CellClass::Oth);
    // Control: the same extents at the iteration rank are pointwise.
    let same = [
        od(&[1, 8], &[8, 1], ElementKind::F32),
        od(&[1, 8], &[8, 1], ElementKind::F32),
    ];
    let k = derive_structure_key(&same, &CellFacts::new(), ArchSku::Sm89).unwrap();
    assert_eq!(k.cell_class, CellClass::Ew1);
}

// ---- the weight role is declared, never inferred (§6.6-0019, issue #31) -----

fn gemm_cell(weight_left: bool) -> [OperandDesc; 3] {
    let f = |shape: &[i64], st: &[i64], dt| OperandDesc::new(2, shape, st, dt, 256);
    let y = f(&[64, 32], &[32, 1], ElementKind::F16);
    if weight_left {
        // W[64,128] @ x[128,32]
        [
            f(&[64, 128], &[128, 1], ElementKind::Fp8E4M3FN),
            f(&[128, 32], &[32, 1], ElementKind::F16),
            y,
        ]
    } else {
        // x[64,128] @ W[128,32]
        [
            f(&[64, 128], &[128, 1], ElementKind::F16),
            f(&[128, 32], &[32, 1], ElementKind::Fp8E4M3FN),
            y,
        ]
    }
}

fn wdt_of(token: &str) -> String {
    // field 9 is `c<mnk>/<kdiv>[/b..]/<wdt>/<acc>/<out>/<mp>`; <wdt> is 4th from last.
    let f9 = token.split('|').nth(9).expect("contraction field");
    let parts: Vec<&str> = f9.split('/').collect();
    parts[parts.len() - 4].to_string()
}

#[test]
fn a_declared_weight_operand_names_the_weight_dtype_wherever_it_sits() {
    // The cell the sk5 keyer got wrong: weight on the LEFT derived `<wdt>` = f16,
    // the activation, because it read operand 1.
    let left = gemm_cell(true);
    let mut facts = CellFacts::new();
    facts.contraction_declared = true;
    facts.weight_operand = Some(0);
    facts.contraction_roles = Some(ContractionRoles::gemm(2, 0));
    let t = unpopped_vocab::derive_structure_key_token(&left, &facts, ArchSku::Sm89).unwrap();
    assert_eq!(wdt_of(&t), "f8e4m3fn", "{t}");

    let right = gemm_cell(false);
    facts.weight_operand = Some(1);
    let t = unpopped_vocab::derive_structure_key_token(&right, &facts, ArchSku::Sm89).unwrap();
    assert_eq!(wdt_of(&t), "f8e4m3fn", "{t}");

    // Control: the declaration is what is read. The SAME weight-left operands
    // with operand 1 declared name the activation, so a keyer that ignored the
    // hint and happened to read slot 0 would fail the line above, and one that
    // read slot 1 would fail this one's counterpart.
    facts.weight_operand = Some(1);
    let t = unpopped_vocab::derive_structure_key_token(&left, &facts, ArchSku::Sm89).unwrap();
    assert_eq!(wdt_of(&t), "f16", "{t}");
}

#[test]
fn a_declared_contraction_without_a_weight_operand_declines() {
    let mut facts = CellFacts::new();
    facts.contraction_declared = true;
    assert_eq!(
        derive_structure_key(&gemm_cell(true), &facts, ArchSku::Sm89),
        Err(DeriveDecline::WeightRoleNotDeclared)
    );
    // The weight alone is not enough: the M/N/K axes are never inferred from position.
    facts.weight_operand = Some(0);
    assert_eq!(
        derive_structure_key(&gemm_cell(true), &facts, ArchSku::Sm89),
        Err(DeriveDecline::ContractionRolesNotDeclared)
    );
    // A weight slot past the operand list names no operand.
    facts.contraction_roles = Some(ContractionRoles::gemm(2, 0));
    facts.weight_operand = Some(7);
    assert_eq!(
        derive_structure_key(&gemm_cell(true), &facts, ArchSku::Sm89),
        Err(DeriveDecline::WeightRoleOutOfRange { index: 7 })
    );
    // Control: the same declarations with the weight in range derive.
    facts.weight_operand = Some(0);
    assert!(derive_structure_key(&gemm_cell(true), &facts, ArchSku::Sm89).is_ok());
}

#[test]
fn the_legacy_gemm_bridge_declares_operand_1_and_changes_no_token() {
    // Weight on the right: the bridge's declaration matches the cell, so the
    // token is what 0.17.0 derived.
    let right = gemm_cell(false);
    let t =
        unpopped_vocab::try_structure_key_token(OpCategory::Gemm, &right, ArchSku::Sm89).unwrap();
    assert_eq!(wdt_of(&t), "f8e4m3fn", "{t}");
    // Weight on the left through the bridge still reads operand 1: documented,
    // and the reason such callers move to `derive_structure_key`.
    let left = gemm_cell(true);
    let t =
        unpopped_vocab::try_structure_key_token(OpCategory::Gemm, &left, ArchSku::Sm89).unwrap();
    assert_eq!(wdt_of(&t), "f16", "{t}");
}
