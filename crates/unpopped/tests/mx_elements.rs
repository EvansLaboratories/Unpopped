// SPDX-License-Identifier: MIT OR Apache-2.0
//! **OCP MX element dtypes** (`f4e2m1`, `f6e2m3`, `f6e3m2`; KISS-CLASSIFY §6.1-0014).
//!
//! They are `ElementKind` rows for storage and key identity. Nothing computes with
//! them yet, so every compute entry point must decline with a TYPED error rather
//! than panic or quietly treat them as a byte type. Each refusal has a control: the
//! same call with a computed dtype succeeds.

use unpopped::ir::{OpDef, input};
use unpopped::plan::try_build_plan;
use unpopped_vocab::{
    ArchSku, ElementKind, OpCategory, OperandDesc, dtype_storage_bits, dtype_token, structure_key,
};

const MX: [(ElementKind, &str, u32); 3] = [
    (ElementKind::F4E2M1, "f4e2m1", 4),
    (ElementKind::F6E2M3, "f6e2m3", 6),
    (ElementKind::F6E3M2, "f6e3m2", 6),
];

#[test]
fn the_mx_elements_are_members_with_their_token_and_width() {
    for (dt, token, bits) in MX {
        assert!(ElementKind::ALL.contains(&dt), "{dt:?} missing from ALL");
        assert_eq!(dtype_token(dt), token);
        assert_eq!(dtype_storage_bits(dt), bits);
        assert!(dt.is_mx_element() && !dt.is_reserved(), "{dt:?}");
    }
    // Control: a scale and a computed float are neither.
    assert!(!ElementKind::F8E8M0.is_mx_element() && !ElementKind::F32.is_mx_element());
}

#[test]
fn the_plan_gate_declines_an_mx_element_op_with_a_typed_error() {
    for (dt, _, _) in MX {
        let op = OpDef::elementwise("mx_add", 2, &[dt], input(0) + input(1));
        let a = OperandDesc::new(2, &[7, 3], &[3, 1], dt, 256);
        let key = structure_key(OpCategory::BinaryElementwise, &[a, a, a], ArchSku::Sm89);
        let planned = std::panic::catch_unwind(|| try_build_plan(&op, &key).map(|_| ()));
        assert!(
            matches!(planned, Ok(Err(_))),
            "{dt:?}: want a typed plan error, got {planned:?}"
        );
    }
    // Control: the identical op at f32 plans.
    let op = OpDef::elementwise("add", 2, &[ElementKind::F32], input(0) + input(1));
    let a = OperandDesc::new(2, &[7, 3], &[3, 1], ElementKind::F32, 256);
    let key = structure_key(OpCategory::BinaryElementwise, &[a, a, a], ArchSku::Sm89);
    assert!(try_build_plan(&op, &key).is_ok());
}
