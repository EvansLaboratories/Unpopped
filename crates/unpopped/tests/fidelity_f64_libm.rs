// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The accuracy budget of a `double` libm op scales with the dtype's epsilon** (PM TASK 2026-11-23, gap (a) of the libm slice). The
//! per-op ULP table (`contract::unary_ulp`) counts ULPs of the compute dtype; `required_fidelity` multiplies the count by the
//! dtype's unit roundoff (`oracle/fidelity.rs`, `unit_roundoff`: `2^-53` for `F64`, `2^-24` otherwise). So an `F64` `exp`, `erfc` or
//! `lgamma` whose result carries only `f32` precision (compute in `f32`, widen) is OUTSIDE the entitled band and must fail, while the
//! accurate result passes. The control is an `F32` plan of the same op: a result one `f32` ulp off passes there, so the dtype scaling is what
//! separates the two.

use unpopped::build_plan;
use unpopped::ir::{OpDef, UnaryOp, input};
use unpopped::oracle::{TypedBuffer, compare_for_plan};
use unpopped_vocab::{ArchSku, ElementKind, OpCategory, OperandDesc, structure_key};

fn f64s(v: &[f64]) -> TypedBuffer {
    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    TypedBuffer::new(ElementKind::F64, vec![v.len() as i64], vec![1], bytes)
}

/// `op(x)` for `x = 0.5, 1.5, 2.5, 3.5`, as published reference values (not computed by a model of the lowering).
fn reference(op: UnaryOp) -> [f64; 4] {
    match op {
        UnaryOp::Exp => [
            1.648_721_270_700_128_2,
            4.481_689_070_338_064_5,
            12.182_493_960_703_473,
            33.115_451_958_692_31,
        ],
        UnaryOp::Erfc => [
            0.479_500_122_186_953_5,
            0.033_894_853_524_689_274,
            0.000_406_952_017_444_958_66,
            7.430_983_723_414_128e-7,
        ],
        UnaryOp::Lgamma => [
            0.572_364_942_924_700_1,
            -0.120_782_237_635_245_22,
            0.284_682_870_472_919_2,
            1.200_973_602_347_074_3,
        ],
        _ => unreachable!(),
    }
}

fn f32s(v: &[f64]) -> TypedBuffer {
    let bytes: Vec<u8> = v.iter().flat_map(|x| (*x as f32).to_le_bytes()).collect();
    TypedBuffer::new(ElementKind::F32, vec![v.len() as i64], vec![1], bytes)
}

fn plan_for(
    op: UnaryOp,
    dt: ElementKind,
) -> (OpDef, Vec<OperandDesc>, unpopped_vocab::StructureKey) {
    let d = OperandDesc::new(
        1,
        &[4],
        &[1],
        dt,
        if dt == ElementKind::F64 { 8 } else { 4 },
    );
    let ops = vec![d, d];
    let def = OpDef::elementwise("e", 1, &[dt], input(0).unary(op));
    let key = structure_key(OpCategory::UnaryElementwise, &ops, ArchSku::Sm89);
    (def, ops, key)
}

#[test]
fn an_f32_precision_result_fails_an_f64_exp_erfc_or_lgamma_and_the_accurate_one_passes() {
    for op in [UnaryOp::Exp, UnaryOp::Erfc, UnaryOp::Lgamma] {
        let (def, ops, key) = plan_for(op, ElementKind::F64);
        let plan = build_plan(&def, &key);
        let want = reference(op);
        let degraded = want.map(|v| f64::from(v as f32));
        compare_for_plan(&f64s(&want), &f64s(&want), &plan, &ops)
            .expect("the accurate result passes");
        assert!(
            compare_for_plan(&f64s(&want), &f64s(&degraded), &plan, &ops).is_err(),
            "{op:?}: a result with only f32 precision must fail an F64 kernel"
        );
        // control: at F32 a result one f32 ulp off is inside the entitled band
        let (def32, ops32, key32) = plan_for(op, ElementKind::F32);
        let plan32 = build_plan(&def32, &key32);
        let off_by_one_ulp = want.map(|v| f64::from(f32::from_bits((v as f32).to_bits() + 1)));
        compare_for_plan(&f32s(&want), &f32s(&off_by_one_ulp), &plan32, &ops32)
            .expect("control: one f32 ulp is inside the F32 band");
    }
}
