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
    ArchSku, AxisMask, CellClass, CellFacts, DeriveDecline, ElementKind, OpCategory, OperandDesc,
    OperandRole, derive_structure_key, structure_key, try_structure_key,
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
