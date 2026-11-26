// SPDX-License-Identifier: MIT OR Apache-2.0
//! Plan E4 (KISS 0.24.0 §6.6-0023a, union batch 1): **K is a role set.** Asked red first (pm-kernels): does a contraction whose
//! contracted extent is declared as TWO axes (`lhs [M, K1, K2] . rhs [K1, K2, N] -> [M, N]`) check and run on the oracle?
//! **Measured answer:** the KEY derives (K class and divisibility from the product `K1 * K2`, `unpopped-vocab`, the four KISS
//! vectors), but the OP-LAYER plan gate REFUSES two `ContractedK` axes by name (`UnservableContraction(RolesInvalid)`, "v1: exactly one
//! ContractedK per operand (multi-group deferred)": `plan/contraction.rs`), so nothing runs. That refusal is pinned here, beside
//! its one-K control, so lifting it is a deliberate act: plan-layer multi-K is ROADMAP D2 ("multi-K", CUDA rows 44, 55, 81), whose
//! emitters index K as one axis and would otherwise miscompile silently. A kernel-layer two-K contraction (two nested loops) needs
//! no plan gate and already runs on the oracle.

use unpopped::ir::{AxisRole, ContractionAxes, OpDef, reduced};
use unpopped::oracle::{TypedBuffer, evaluate};
use unpopped::plan::try_build_plan;
use unpopped_vocab::{
    ArchSku, AxisRef, CellFacts, ContractionRoles, ElementKind, OperandDesc, derive_structure_key,
};

const F32: ElementKind = ElementKind::F32;
const M: usize = 3;
const K1: usize = 4;
const K2: usize = 5;
const N: usize = 2;

fn lhs_value(m: usize, a: usize, b: usize) -> f32 {
    (1 + m * 7 + a * 3 + b) as f32 * 0.5
}
fn rhs_value(a: usize, b: usize, n: usize) -> f32 {
    ((a * 5 + b) as f32 - 6.0) * 0.25 + n as f32
}

fn f32_bytes(v: impl Iterator<Item = f32>) -> Vec<u8> {
    v.flat_map(f32::to_le_bytes).collect()
}

/// `lhs [M, K1, K2] . rhs [K1, K2, N] -> [M, N]`, dense row-major, K declared as two axes.
fn run(k_more: &[AxisRef]) -> Result<Vec<f32>, String> {
    let lhs = TypedBuffer::new(
        F32,
        vec![M as i64, K1 as i64, K2 as i64],
        vec![(K1 * K2) as i64, K2 as i64, 1],
        f32_bytes(
            (0..M)
                .flat_map(|m| (0..K1).flat_map(move |a| (0..K2).map(move |b| lhs_value(m, a, b)))),
        ),
    );
    let rhs = TypedBuffer::new(
        F32,
        vec![K1 as i64, K2 as i64, N as i64],
        vec![(K2 * N) as i64, N as i64, 1],
        f32_bytes(
            (0..K1)
                .flat_map(|a| (0..K2).flat_map(move |b| (0..N).map(move |n| rhs_value(a, b, n)))),
        ),
    );
    let ops = [
        OperandDesc::new(
            3,
            &[M as i64, K1 as i64, K2 as i64],
            &[(K1 * K2) as i64, K2 as i64, 1],
            F32,
            256,
        ),
        OperandDesc::new(
            3,
            &[K1 as i64, K2 as i64, N as i64],
            &[(K2 * N) as i64, N as i64, 1],
            F32,
            256,
        ),
        OperandDesc::new(2, &[M as i64, N as i64], &[N as i64, 1], F32, 256),
    ];
    let mut facts = CellFacts::new();
    facts.contraction_declared = true;
    facts.weight_operand = Some(1);
    facts.contraction_roles = Some(
        ContractionRoles::new(AxisRef::new(0, 0), AxisRef::new(1, 2), AxisRef::new(0, 1))
            .with_k_more_axes(k_more),
    );
    let key =
        derive_structure_key(&ops, &facts, ArchSku::Sm89).map_err(|e| format!("key: {e:?}"))?;
    let axes = ContractionAxes {
        lhs: vec![
            AxisRole::FreeM,
            AxisRole::ContractedK,
            AxisRole::ContractedK,
        ],
        rhs: vec![
            AxisRole::ContractedK,
            AxisRole::ContractedK,
            AxisRole::FreeN,
        ],
    };
    let op = OpDef::contraction("mm2k", &[F32], axes, reduced(0));
    let plan = try_build_plan(&op, &key).map_err(|e| format!("plan: {e:?}"))?;
    let out = evaluate(&plan, &ops, &[lhs, rhs], &[]).remove(0);
    Ok(out
        .bytes
        .chunks(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect())
}

#[test]
fn a_two_k_cell_derives_a_key_but_the_plan_gate_refuses_two_contracted_k_by_name() {
    let r = run(&[AxisRef::new(0, 2)]);
    let Err(msg) = r else {
        panic!(
            "the plan gate now accepts two ContractedK axes: update this pin and the D2 row, and make every emitter decline or lower it"
        )
    };
    assert!(
        msg.starts_with("plan: UnservableContraction(RolesInvalid"),
        "the key derives, so the refusal is the plan gate's: {msg}"
    );
    assert!(msg.contains("exactly one ContractedK per operand"), "{msg}");
}

#[test]
fn the_key_of_a_two_k_cell_differs_from_the_one_k_declaration_of_the_same_operands() {
    use unpopped_vocab::{ElementKind, derive_structure_key};
    // the same operands, K declared as the first axis only versus as both: two different keys (K class from K1 versus from K1 * K2)
    let ops = [
        OperandDesc::new(3, &[3, 4, 5], &[20, 5, 1], ElementKind::F32, 256),
        OperandDesc::new(3, &[4, 5, 2], &[10, 2, 1], ElementKind::F32, 256),
        OperandDesc::new(2, &[3, 2], &[2, 1], ElementKind::F32, 256),
    ];
    let key = |k_more: &[AxisRef]| {
        let mut facts = CellFacts::new();
        facts.contraction_declared = true;
        facts.weight_operand = Some(1);
        facts.contraction_roles = Some(
            ContractionRoles::new(AxisRef::new(0, 0), AxisRef::new(1, 2), AxisRef::new(0, 1))
                .with_k_more_axes(k_more),
        );
        derive_structure_key(&ops, &facts, ArchSku::Sm89).expect("derives")
    };
    assert_ne!(key(&[]), key(&[AxisRef::new(0, 2)]), "K = 4 versus K = 20");
}

#[test]
fn the_one_k_control_still_plans_and_runs() {
    // the control for the refusal above: the canonical matmul with one ContractedK per operand is accepted by the same gate
    let ops = [
        OperandDesc::new(2, &[3, 4], &[4, 1], ElementKind::F32, 256),
        OperandDesc::new(2, &[4, 2], &[2, 1], ElementKind::F32, 256),
        OperandDesc::new(2, &[3, 2], &[2, 1], ElementKind::F32, 256),
    ];
    let mut facts = CellFacts::new();
    facts.contraction_declared = true;
    facts.weight_operand = Some(1);
    facts.contraction_roles = Some(ContractionRoles::gemm(2, 0));
    let key = derive_structure_key(&ops, &facts, ArchSku::Sm89).expect("derives");
    let op = OpDef::contraction("mm", &[F32], ContractionAxes::matmul(), reduced(0));
    let plan = try_build_plan(&op, &key).expect("one ContractedK plans");
    let lhs = TypedBuffer::new(
        F32,
        vec![3, 4],
        vec![4, 1],
        f32_bytes((0..12).map(|i| i as f32)),
    );
    let rhs = TypedBuffer::new(
        F32,
        vec![4, 2],
        vec![2, 1],
        f32_bytes((0..8).map(|i| (i % 3) as f32)),
    );
    let out = evaluate(&plan, &ops, &[lhs, rhs], &[]).remove(0);
    let got: Vec<f32> = out
        .bytes
        .chunks(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    // row 0 of lhs is [0,1,2,3], columns of rhs [0,2,1,0] and [1,0,2,1]: 0*0+1*2+2*1+3*0 = 4 and 0*1+1*0+2*2+3*1 = 7
    assert_eq!(&got[..2], &[4.0, 7.0]);
}
