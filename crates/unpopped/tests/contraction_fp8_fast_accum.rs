// SPDX-License-Identifier: MIT OR Apache-2.0
//! ROADMAP U56 (`docs/accum-policies.md`): `AccumSpec::Fp8FastAccum { k_step, promote_every, mantissa_bits }`, a policy
//! stated by the software-documented parameters of fast-accumulate FP8 kernels (a promotion interval and an imprecise
//! accumulator width), never as a claim about a chip.
//!
//! An FP8 contraction stores its result in the operand dtype, so this file checks planning, the declines and a few
//! results FP8 holds exactly; the number model over a broad corpus is the unit test in `oracle/accum_policy.rs`.

use unpopped::ir::{AccumSpec, ContractionAxes, OpDef, reduced};
use unpopped::oracle::{TypedBuffer, evaluate};
use unpopped::plan::{ContractionDecline, PlanError, try_build_plan};
use unpopped_vocab::{
    ArchSku, CellFacts, ContractionRoles, ElementKind, OperandDesc, derive_structure_key,
};

const E4M3: ElementKind = ElementKind::Fp8E4M3FN;
const E5M2: ElementKind = ElementKind::Fp8E5M2;

/// The value of an FP8 code, decoded here (not by the library): `(sign, exponent, mantissa)` fields.
fn decode(dtype: ElementKind, code: u8) -> f64 {
    let (ebits, mbits, bias) = if dtype == E4M3 { (4, 3, 7) } else { (5, 2, 15) };
    let s = if code >> 7 == 1 { -1.0 } else { 1.0 };
    let e = i32::from((code >> mbits) & ((1 << ebits) - 1));
    let m = f64::from(code & ((1 << mbits) - 1)) / f64::from(1 << mbits);
    if e == 0 {
        s * m * f64::powi(2.0, 1 - bias)
    } else {
        s * (1.0 + m) * f64::powi(2.0, e - bias)
    }
}

/// The code of a finite value the dtype holds exactly: found by decoding all 256 codes.
fn encode(dtype: ElementKind, x: f64) -> u8 {
    (0..=255u8)
        .find(|&c| {
            let d = decode(dtype, c);
            d.is_finite() && d == x && !(dtype == E4M3 && c & 0x7f == 0x7f)
        })
        .unwrap_or_else(|| panic!("{x} is not exactly an {dtype:?} value"))
}

fn run(dtype: ElementKind, accum: AccumSpec, a: &[f64], b: &[f64]) -> Result<f64, PlanError> {
    let bytes = |v: &[f64]| -> Vec<u8> { v.iter().map(|&x| encode(dtype, x)).collect() };
    let k = a.len() as i64;
    let lhs = TypedBuffer::new(dtype, vec![1, k], vec![k, 1], bytes(a));
    let rhs = TypedBuffer::new(dtype, vec![k, 1], vec![1, 1], bytes(b));
    let desc = |t: &TypedBuffer| OperandDesc::new(2, &t.shape, &t.strides, dtype, 256);
    let ops = [
        desc(&lhs),
        desc(&rhs),
        OperandDesc::new(2, &[1, 1], &[1, 1], dtype, 256),
    ];
    let mut facts = CellFacts::new();
    facts.contraction_declared = true;
    facts.weight_operand = Some(1);
    facts.contraction_roles = Some(ContractionRoles::gemm(2, 0));
    let key = derive_structure_key(&ops, &facts, ArchSku::Sm89).expect("a key");
    let o = OpDef::contraction("mm", &[dtype], ContractionAxes::matmul(), reduced(0)).with_accum(accum);
    let plan = try_build_plan(&o, &key)?;
    let out = evaluate(&plan, &ops, &[lhs, rhs], &[]).remove(0);
    // the output is stored in the operand dtype (a wider FP8 GEMM output is U46's `D` distinct from the operands)
    assert_eq!(out.dtype, dtype);
    Ok(decode(dtype, out.bytes[0]))
}

// ---- end to end: results an FP8 output holds exactly ---------------------------------------------------------------

fn policy(k_step: u32, promote_every: u32, bits: u8) -> AccumSpec {
    AccumSpec::Fp8FastAccum {
        k_step,
        promote_every,
        mantissa_bits: bits,
    }
}

/// `2*2 + 1*1 + 1*1 = 6` exactly; a 1-bit imprecise accumulator (2 significant bits: 4, 6, 8) ties `4 + 1 = 5` to even
/// (4), twice, so one promotion at the end gives 4; a promotion after every step gives 6 (the promoted `f32` sum is exact).
#[test]
fn the_policy_is_planned_and_evaluated_for_both_fp8_types() {
    let (a, b) = ([2.0, 1.0, 1.0], [2.0, 1.0, 1.0]);
    for dtype in [E4M3, E5M2] {
        assert_eq!(run(dtype, policy(1, 3, 1), &a, &b).unwrap(), 4.0, "{dtype:?} absorbed");
        assert_eq!(run(dtype, policy(1, 1, 1), &a, &b).unwrap(), 6.0, "{dtype:?} promoted every step");
        // a 23-bit imprecise accumulator is an f32 sum here
        assert_eq!(run(dtype, policy(1, 3, 23), &a, &b).unwrap(), 6.0, "{dtype:?} wide");
        // k_step 3 sums the three products exactly inside the step: 6, which 1 bit also holds
        assert_eq!(run(dtype, policy(3, 1, 1), &a, &b).unwrap(), 6.0, "{dtype:?} one step");
    }
}

#[test]
fn a_policy_its_dtype_or_its_parameters_cannot_serve_declines_with_a_reason() {
    let (a, b) = ([0.5f64; 8], [0.5f64; 8]);
    let declined = |dtype: ElementKind, p: AccumSpec| {
        match run(dtype, p, &a, &b) {
            Err(PlanError::UnservableContraction(ContractionDecline::AccumPolicyMismatch {
                dtype: d,
                k_step,
            })) => (d, k_step),
            other => panic!("expected AccumPolicyMismatch, got {other:?}"),
        }
    };
    assert_eq!(declined(E4M3, policy(0, 1, 3)).1, 0, "k_step 0");
    assert_eq!(declined(E4M3, policy(4, 0, 3)).1, 0, "promote_every 0");
    assert_eq!(declined(E4M3, policy(4, 1, 0)).1, 0, "0 mantissa bits");
    assert_eq!(declined(E4M3, policy(4, 1, 24)).1, 0, "24 mantissa bits");
    // an F16 operand is not an FP8 policy's operand
    let fits = |dtype| run(dtype, policy(4, 2, 3), &[0.5; 4], &[0.5; 4]);
    assert!(matches!(
        fits(ElementKind::F16),
        Err(PlanError::UnservableContraction(
            ContractionDecline::AccumPolicyMismatch { .. }
        ))
    ));
}
