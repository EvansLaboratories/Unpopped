//! `sk6`: the scale dtype of a quantized dense-contraction cell is keyed
//! (KISS-CLASSIFY-6.6-0022), as the optional last contraction coordinate
//! `q<dtype>`.
//!
//! At `sk5` an `f8e8m0` scale and an `f16` scale derived the same key although
//! they decode the same stored values to different numbers: a wrong cache hit,
//! not a slow one. These tests pin the derivation, the codec round trip, and the
//! declines, and each has the control that makes its null mean something.

use unpopped_vocab::{
    ArchSku, CellFacts, ContractionRoles, DeriveDecline, ElementKind, OperandDesc, StructureKey,
    TokenDecline, derive_structure_key, derive_structure_key_token,
};

fn od(shape: &[i64], strides: &[i64], dt: ElementKind) -> OperandDesc {
    OperandDesc::new(shape.len(), shape, strides, dt, 256)
}

/// `[x bf16 [64,128], W i4 [128,32], out bf16 [64,32]]`, weight at slot 1.
fn cell() -> [OperandDesc; 3] {
    [
        od(&[64, 128], &[128, 1], ElementKind::Bf16),
        od(&[128, 32], &[32, 1], ElementKind::I4),
        od(&[64, 32], &[32, 1], ElementKind::Bf16),
    ]
}

/// The same cell, plus a dense per-column `[N]` operand of dtype `scale` at slot
/// 2 — the one four-operand shape the contraction deriver admits (§6.6-0016).
fn cell_with_column_operand(scale: ElementKind) -> [OperandDesc; 4] {
    let [x, w, out] = cell();
    [x, w, od(&[32], &[1], scale), out]
}

fn facts(scale: Option<usize>) -> CellFacts {
    let mut f = CellFacts::new();
    f.contraction_declared = true;
    f.weight_operand = Some(1);
    f.scale_operand = scale;
    f.contraction_roles = Some(ContractionRoles::gemm(2, 0));
    // The sk5-era accumulator these pinned bytes were taken at; the spec default is
    // operand 0's dtype (§6.7-0006), which would be bf16 here.
    f.accumulator = Some(ElementKind::F32);
    f
}

fn tail(token: &str) -> String {
    token
        .split('|')
        .nth(9)
        .expect("contraction field")
        .to_owned()
}

#[test]
fn a_cell_with_no_designated_scale_omits_the_coordinate() {
    let t = derive_structure_key_token(&cell(), &facts(None), ArchSku::Sm90).unwrap();
    assert!(t.starts_with("sk6|ctr|"), "{t}");
    assert_eq!(tail(&t), "csss/d16/i4/f32/bf16/st", "{t}");
}

#[test]
fn the_designated_scale_operands_dtype_is_the_last_coordinate() {
    // Control for the whole test: the SAME cell with the scale role declared on
    // two operands of different dtype must differ, and only in the tail.
    let mut seen = Vec::new();
    for (dt, want) in [
        (ElementKind::F8E8M0, "csss/d16/i4/f32/bf16/st/qf8e8m0"),
        (ElementKind::F16, "csss/d16/i4/f32/bf16/st/qf16"),
    ] {
        let ops = cell_with_column_operand(dt);
        let t = derive_structure_key_token(&ops, &facts(Some(2)), ArchSku::Sm90);
        let t = t.unwrap_or_else(|e| panic!("{dt:?}: {e}"));
        assert_eq!(tail(&t), want, "{t}");
        seen.push(t);
    }
    assert_ne!(
        seen[0], seen[1],
        "an f8e8m0 scale and an f16 scale are one cell"
    );
}

#[test]
fn the_scale_coordinate_round_trips_and_is_unambiguous_from_position_alone() {
    for dt in [ElementKind::F8E8M0, ElementKind::F16, ElementKind::F32] {
        let ops = cell_with_column_operand(dt);
        let key = derive_structure_key(&ops, &facts(Some(2)), ArchSku::Sm90).unwrap();
        let token = key.to_token();
        let back = StructureKey::parse_token(&token).unwrap_or_else(|e| panic!("{token}: {e:?}"));
        assert_eq!(back, key, "{token}");
        assert_eq!(back.to_token(), token, "byte-unstable: {token}");
        assert_eq!(back.contraction.unwrap().scale, Some(dt));
    }
}

#[test]
fn a_scale_role_past_the_operand_list_is_a_decline_not_a_silent_omission() {
    assert_eq!(
        derive_structure_key(&cell(), &facts(Some(9)), ArchSku::Sm90),
        Err(DeriveDecline::ScaleRoleOutOfRange { index: 9 })
    );
    // Control: the in-range spelling of the same hint derives.
    assert!(derive_structure_key(&cell(), &facts(Some(1)), ArchSku::Sm90).is_ok());
}

#[test]
fn a_reserved_scale_dtype_declines_instead_of_keying_what_the_codec_would_refuse() {
    let ops = cell_with_column_operand(ElementKind::F8E6M2);
    assert_eq!(
        derive_structure_key(&ops, &facts(Some(2)), ArchSku::Sm90),
        Err(DeriveDecline::ReservedScaleDtype {
            dtype: ElementKind::F8E6M2
        })
    );
}

#[test]
fn the_reader_declines_what_the_clause_forbids() {
    let base = "sk6|ctr|i4|cuda:sm90|ix32|grid|r2|co/00/v4/d16/f;co/00/v4/d16/f;co/00/v4/d16/f|-|ctll/d16/i4/f32/bf16/st";
    // Controls: the bare and the well-formed spellings parse.
    assert!(StructureKey::parse_token(base).is_ok());
    assert!(StructureKey::parse_token(&format!("{base}/qf16")).is_ok());

    assert_eq!(
        StructureKey::parse_token(&format!("{base}/qzzz")),
        Err(TokenDecline::UnknownDtype)
    );
    assert!(matches!(
        StructureKey::parse_token(&format!("{base}/qf8e6m2")),
        Err(TokenDecline::ReservedDtype { .. })
    ));
    // A second coordinate in the same field.
    assert!(StructureKey::parse_token(&format!("{base}/qf16/qf16")).is_err());
    // Outside a contraction field.
    let ew = "sk6|ew2|f32|cuda:sm89|ix32|grid|r2|co/00/v4/d16/f;co/00/v4/d16/f;co/00/v4/d16/f|-|f32/st/qf16";
    assert!(StructureKey::parse_token(ew).is_err());
    // The retired schema is a versioned decline, not a parse.
    assert_eq!(
        StructureKey::parse_token(&base.replacen("sk6", "sk5", 1)),
        Err(TokenDecline::UnsupportedSchemaVersion { version: 5 })
    );
}
