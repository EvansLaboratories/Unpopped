// SPDX-License-Identifier: MIT OR Apache-2.0
//! ROADMAP C5 (PM ruling A + D, 2026-11-01): `compare_for_plan` derives the band a cell is entitled to
//! (`required_fidelity`) instead of taking one, and `compare_for_plan_within` accepts a caller's band
//! only if it is no looser than the entitlement. A band tighter than the entitlement is harmless; a
//! looser one is the defect (`docs/options-compelled-fidelity.md`).

use unpopped::build_plan;
use unpopped::ir::{OpDef, UnaryOp, input};
use unpopped::oracle::{
    Fidelity, TypedBuffer, compare_for_plan, compare_for_plan_within, required_fidelity,
};
use unpopped_vocab::{ArchSku, ElementKind, OpCategory, OperandDesc, structure_key};

fn f32s(v: &[f32]) -> TypedBuffer {
    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    TypedBuffer::new(ElementKind::F32, vec![v.len() as i64], vec![1], bytes)
}

fn i32s(v: &[i32]) -> TypedBuffer {
    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    TypedBuffer::new(ElementKind::I32, vec![v.len() as i64], vec![1], bytes)
}

/// An `exp` cell on a CUDA target: float, a real finite band.
fn exp_cell() -> (OpDef, Vec<OperandDesc>, unpopped_vocab::StructureKey) {
    let d = OperandDesc::new(1, &[4], &[1], ElementKind::F32, 4);
    let ops = vec![d, d];
    let op = OpDef::elementwise("e", 1, &[ElementKind::F32], input(0).unary(UnaryOp::Exp));
    let key = structure_key(OpCategory::UnaryElementwise, &ops, ArchSku::Sm89);
    (op, ops, key)
}

#[test]
fn the_derived_band_accepts_a_result_inside_it_and_rejects_one_outside() {
    let (op, ops, key) = exp_cell();
    let plan = build_plan(&op, &key);
    let Some(Fidelity::Tolerant { rel, .. }) = required_fidelity(&plan, &ops) else {
        panic!("CONTROL: an exp cell on CUDA is entitled to a tolerant band");
    };
    let want = f32s(&[1.0, 2.0, 4.0, 8.0]);
    let inside = f32s(&[1.0 * (1.0 + rel as f32 / 2.0), 2.0, 4.0, 8.0]);
    let outside = f32s(&[1.0 * (1.0 + rel as f32 * 8.0), 2.0, 4.0, 8.0]);
    compare_for_plan(&want, &inside, &plan, &ops).expect("inside the entitled band");
    assert!(compare_for_plan(&want, &outside, &plan, &ops).is_err(), "outside it");
}

#[test]
fn a_caller_band_looser_than_the_entitlement_is_refused() {
    let (op, ops, key) = exp_cell();
    let plan = build_plan(&op, &key);
    let loose = Fidelity::Tolerant { rel: 0.5, abs: 0.5 };
    let want = f32s(&[1.0, 2.0, 4.0, 8.0]);
    // The buffers differ by far more than the entitlement but less than the caller's band: a plain
    // `compare` at `loose` passes, which is exactly the defect.
    let off = f32s(&[1.1, 2.0, 4.0, 8.0]);
    assert!(unpopped::oracle::compare(&want, &off, loose).is_ok(), "CONTROL: compare accepts it");
    let r = compare_for_plan_within(&want, &off, &plan, &ops, loose);
    assert!(r.unwrap_err().contains("looser"), "named as a looser band");
}

#[test]
fn a_tighter_caller_band_is_accepted_and_applied() {
    let (op, ops, key) = exp_cell();
    let plan = build_plan(&op, &key);
    let want = f32s(&[1.0, 2.0, 4.0, 8.0]);
    compare_for_plan_within(&want, &want, &plan, &ops, Fidelity::BitExact)
        .expect("bit-exact on equal buffers");
    let off = f32s(&[1.0000001, 2.0, 4.0, 8.0]);
    assert!(
        compare_for_plan_within(&want, &off, &plan, &ops, Fidelity::BitExact).is_err(),
        "the tighter band is the one applied"
    );
}

#[test]
fn an_integer_cell_refuses_any_tolerant_band() {
    let d = OperandDesc::new(1, &[3], &[1], ElementKind::I32, 4);
    let ops = vec![d, d, d];
    let op = OpDef::elementwise("add", 2, &[ElementKind::I32], input(0) + input(1));
    let key = structure_key(OpCategory::BinaryElementwise, &ops, ArchSku::Sm89);
    let plan = build_plan(&op, &key);
    let a = i32s(&[1, 2, 3]);
    let tol = Fidelity::Tolerant { rel: 0.0, abs: 0.0 };
    let r = compare_for_plan_within(&a, &a, &plan, &ops, tol);
    assert!(r.unwrap_err().contains("looser"), "BitExact is the entitlement for integers");
    compare_for_plan(&a, &a, &plan, &ops).expect("the derived band is bit-exact");
}

#[test]
fn a_target_with_no_entitlement_is_an_error_not_a_guess() {
    let d = OperandDesc::new(1, &[4], &[1], ElementKind::F32, 4);
    let ops = vec![d, d];
    let op = OpDef::elementwise("e", 1, &[ElementKind::F32], input(0).unary(UnaryOp::Exp));
    let vk = unpopped_vocab::TargetId::parse("vulkan:st16").expect("a valid target token");
    let key = structure_key(OpCategory::UnaryElementwise, &ops, vk);
    let plan = build_plan(&op, &key);
    let a = f32s(&[1.0, 2.0, 4.0, 8.0]);
    let r = compare_for_plan(&a, &a, &plan, &ops);
    assert!(r.unwrap_err().contains("no entitlement"));
    let r = compare_for_plan_within(&a, &a, &plan, &ops, Fidelity::BitExact);
    assert!(r.unwrap_err().contains("no entitlement"), "even a tight band: nothing to certify against");
}
