// SPDX-License-Identifier: MIT OR Apache-2.0
//! ROADMAP U7 S1a (`docs/accum-policies.md`): a contraction may name a reduced-precision accumulation policy. The oracle
//! models the structure of a tensor-core multiply-accumulate: inputs converted exactly as the policy says, exact products,
//! and a narrow accumulator rounded after every `k_step` contracted elements. It is an Unpopped extension, not KISS.
//!
//! The reference below is written here, from the policy text, by different means than the oracle's: TF32 rounding with
//! `f64::round` (ties away from zero) where the oracle edits bits, bf16 and f16 through the independent `half` codec. The
//! corpus is built so each policy differs from the plain `f64` sum, and so `k_step` matters: a ties-away TF32 input, a
//! value only the remainder term of the `x3` forms recovers, a large value that absorbs small ones in a narrow accumulator.

use half::{bf16, f16};
use unpopped::ir::{AccumSpec, ContractionAxes, OpDef, reduced};
use unpopped::kernel_lift::{LiftDecline, lift_contraction_opdef};
use unpopped::oracle::{TypedBuffer, evaluate};
use unpopped::plan::{ContractionDecline, PlanError, try_build_plan};
use unpopped_vocab::{
    ArchSku, CellFacts, ContractionRoles, ElementKind, OperandDesc, derive_structure_key,
};

const F32: ElementKind = ElementKind::F32;
const F16: ElementKind = ElementKind::F16;
const K: usize = 16;

fn op(dtype: ElementKind, accum: AccumSpec) -> OpDef {
    OpDef::contraction("mm", &[dtype], ContractionAxes::matmul(), reduced(0)).with_accum(accum)
}

/// A `[1, K]` row times a `[K, 1]` column, as raw `f64` values the dtype holds exactly.
fn run(dtype: ElementKind, accum: AccumSpec, a: &[f64], b: &[f64]) -> Result<f64, PlanError> {
    let bytes = |v: &[f64]| -> Vec<u8> {
        v.iter()
            .flat_map(|&x| match dtype {
                ElementKind::F16 => f16::from_f64(x).to_le_bytes().to_vec(),
                _ => (x as f32).to_le_bytes().to_vec(),
            })
            .collect()
    };
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
    let o = op(dtype, accum);
    let plan = try_build_plan(&o, &key)?;
    let out = evaluate(&plan, &ops, &[lhs, rhs], &[]).remove(0).bytes;
    Ok(match dtype {
        ElementKind::F16 => f16::from_le_bytes(out[..2].try_into().unwrap()).to_f64(),
        _ => f64::from(f32::from_le_bytes(out[..4].try_into().unwrap())),
    })
}

// ---- the reference, from the policy text ----------------------------------------------------------------------

/// TF32: 10 explicit mantissa bits, ties away from zero, by scaling (not by editing bits).
fn ref_tf32(x: f64) -> f64 {
    if x == 0.0 || !x.is_finite() {
        return x;
    }
    let e = x.abs().log2().floor();
    let q = (e - 10.0).exp2();
    (x / q).round() * q
}

fn ref_bf16(x: f64) -> f64 {
    f64::from(bf16::from_f32(x as f32))
}

fn ref_split(round: fn(f64) -> f64, a: f64, b: f64) -> f64 {
    let (ah, bh) = (round(a), round(b));
    let (al, bl) = (round(a - ah), round(b - bh));
    ah * bh + ah * bl + al * bh
}

fn ref_f32(x: f64) -> f64 {
    f64::from(x as f32)
}

/// Through the `f32` path of the `half` codec, which is trustworthy for a value an `f32` holds exactly (its `from_f64` is
/// not: it rounds `0.04518128032270489` down although the value is above the tie). The corpus sums are small integers.
fn ref_f16(x: f64) -> f64 {
    assert_eq!(
        f64::from(x as f32),
        x,
        "the reference needs an f32-exact value"
    );
    f16::from_f32(x as f32).to_f64()
}

fn reference(accum: AccumSpec, a: &[f64], b: &[f64]) -> f64 {
    let (step, acc_round, prod): (usize, fn(f64) -> f64, Box<dyn Fn(f64, f64) -> f64>) = match accum
    {
        AccumSpec::Tf32 { k_step } => (
            k_step as usize,
            ref_f32,
            Box::new(|x, y| ref_tf32(x) * ref_tf32(y)),
        ),
        AccumSpec::Tf32x3 { k_step } => (
            k_step as usize,
            ref_f32,
            Box::new(|x, y| ref_split(ref_tf32, x, y)),
        ),
        AccumSpec::Bf16x3 { k_step } => (
            k_step as usize,
            ref_f32,
            Box::new(|x, y| ref_split(ref_bf16, x, y)),
        ),
        AccumSpec::F16Accum { k_step } => (k_step as usize, ref_f16, Box::new(|x, y| x * y)),
        _ => unreachable!("a policy"),
    };
    let mut acc = 0.0;
    for (ca, cb) in a.chunks(step).zip(b.chunks(step)) {
        let sum: f64 = ca.iter().zip(cb).map(|(&x, &y)| prod(x, y)).sum();
        acc = acc_round(acc + sum);
    }
    acc
}

// ---- the corpus -------------------------------------------------------------------------------------------------

/// K = 16 pairs with more mantissa than TF32 keeps, a tie, a negative tie, and a value only a remainder term recovers.
fn f32_corpus() -> (Vec<f64>, Vec<f64>) {
    let u = |p: i32| f64::from(p).exp2();
    let a: Vec<f64> = (0..K)
        .map(|i| match i {
            0 => 1.0 + u(-12),          // below TF32's resolution: rounds to 1.0
            1 => 1.0 + u(-11),          // the tie: ties away from zero gives 1 + 2^-10
            2 => -(1.0 + u(-11)),       // the negative tie
            3 => 1.0 + u(-12) + u(-20), // the remainder carries 2^-12 and 2^-20
            _ => 0.75 + (i as f64) * u(-9),
        })
        .collect();
    let b: Vec<f64> = (0..K)
        .map(|i| match i {
            3 => 1.0 + u(-13),
            _ => 1.0 - (i as f64) * u(-10) + u(-14),
        })
        .collect();
    (a, b)
}

/// 2048 followed by fifteen ones: a binary16 accumulator absorbs the ones when it rounds often.
fn f16_corpus() -> (Vec<f64>, Vec<f64>) {
    let mut a = vec![1.0; K];
    a[0] = 2048.0;
    (a, vec![1.0; K])
}

/// 2^24 followed by fifteen ones: an `f32` accumulator absorbs the ones when it rounds often.
fn big_corpus() -> (Vec<f64>, Vec<f64>) {
    let mut a = vec![1.0; K];
    a[0] = 16_777_216.0;
    (a, vec![1.0; K])
}

const POLICIES: [fn(u32) -> AccumSpec; 3] = [
    |k_step| AccumSpec::Tf32 { k_step },
    |k_step| AccumSpec::Tf32x3 { k_step },
    |k_step| AccumSpec::Bf16x3 { k_step },
];

#[test]
fn the_oracle_matches_the_reference_for_every_f32_policy_and_step() {
    for (corpus, name) in [(f32_corpus(), "mantissa"), (big_corpus(), "absorption")] {
        let (a, b) = corpus;
        for make in POLICIES {
            for k_step in [1, 4, 8, 16] {
                let accum = make(k_step);
                let got = run(F32, accum, &a, &b).expect("plans");
                assert_eq!(
                    got,
                    reference(accum, &a, &b),
                    "{accum:?} on the {name} corpus"
                );
            }
        }
    }
}

#[test]
fn the_f16_accumulator_matches_the_reference_for_every_step() {
    let (a, b) = f16_corpus();
    for k_step in [1, 2, 4, 16] {
        let accum = AccumSpec::F16Accum { k_step };
        assert_eq!(
            run(F16, accum, &a, &b).expect("plans"),
            reference(accum, &a, &b),
            "k_step {k_step}"
        );
    }
}

#[test]
fn each_policy_is_a_real_departure_from_the_plain_sum_and_k_step_matters() {
    let (a, b) = f32_corpus();
    let wide = run(F32, AccumSpec::WideFloat, &a, &b).unwrap();
    let r: Vec<f64> = POLICIES
        .iter()
        .map(|m| run(F32, m(8), &a, &b).unwrap())
        .collect();
    // plain TF32 and bf16x3 depart from the f64 sum on this corpus
    assert_ne!(r[0], wide, "TF32 must differ from the f64 sum");
    assert_ne!(r[2], wide, "bf16x3 must differ from the f64 sum");
    // 3xTF32 exists to recover what TF32 loses: it lands much closer to the sum than TF32 does
    assert!(
        (r[1] - wide).abs() * 10.0 < (r[0] - wide).abs(),
        "tf32 {} x3 {} wide {wide}",
        r[0],
        r[1]
    );
    // a narrow accumulator that rounds after every element absorbs what one rounding at the end keeps
    let (a, b) = big_corpus();
    let often = run(F32, AccumSpec::Tf32 { k_step: 1 }, &a, &b).unwrap();
    let once = run(F32, AccumSpec::Tf32 { k_step: 16 }, &a, &b).unwrap();
    assert_eq!((often, once), (16_777_216.0, 16_777_232.0));
    let (a, b) = f16_corpus();
    let often = run(F16, AccumSpec::F16Accum { k_step: 1 }, &a, &b).unwrap();
    let once = run(F16, AccumSpec::F16Accum { k_step: 16 }, &a, &b).unwrap();
    assert_eq!((often, once), (2048.0, 2064.0));
}

#[test]
fn tf32_rounds_a_tie_away_from_zero_and_the_hand_values_hold() {
    // 1 + 2^-11 sits exactly between two TF32 values (spacing 2^-10): away from zero is the larger magnitude
    let one = [1.0];
    let tie = [1.0 + 2f64.powi(-11)];
    let got = run(F32, AccumSpec::Tf32 { k_step: 1 }, &tie, &one).unwrap();
    assert_eq!(got, 1.0 + 2f64.powi(-10));
    let neg = [-(1.0 + 2f64.powi(-11))];
    assert_eq!(
        run(F32, AccumSpec::Tf32 { k_step: 1 }, &neg, &one).unwrap(),
        -(1.0 + 2f64.powi(-10))
    );
    // below the tie it rounds down to 1
    let below = [1.0 + 2f64.powi(-12)];
    assert_eq!(
        run(F32, AccumSpec::Tf32 { k_step: 1 }, &below, &one).unwrap(),
        1.0
    );
    // the x3 form recovers what the head loses: (1 + 2^-12) * 1 = 1 + 2^-12 exactly
    assert_eq!(
        run(F32, AccumSpec::Tf32x3 { k_step: 1 }, &below, &one).unwrap(),
        1.0 + 2f64.powi(-12)
    );
}

#[test]
fn non_finite_inputs_follow_the_arithmetic_of_each_form() {
    let a = [f64::INFINITY, 1.0];
    let b = [1.0, 1.0];
    // plain TF32 converts infinity to infinity
    assert_eq!(
        run(F32, AccumSpec::Tf32 { k_step: 1 }, &a, &b).unwrap(),
        f64::INFINITY
    );
    // the x3 forms take a remainder `a - head`, which is `inf - inf = NaN`: the algorithm's own result, stated here
    for make in [POLICIES[1], POLICIES[2]] {
        assert!(run(F32, make(1), &a, &b).unwrap().is_nan());
    }
    let nan = run(F32, AccumSpec::Tf32 { k_step: 1 }, &[f64::NAN, 1.0], &b).unwrap();
    assert!(nan.is_nan());
}

#[test]
fn a_policy_that_does_not_fit_the_key_is_a_typed_plan_decline() {
    let (a, b) = f32_corpus();
    for (dtype, accum) in [
        (F16, AccumSpec::Tf32 { k_step: 8 }),
        (F32, AccumSpec::F16Accum { k_step: 8 }),
        (F32, AccumSpec::Tf32 { k_step: 0 }),
        (F32, AccumSpec::Bf16x3 { k_step: 0 }),
    ] {
        let r = run(dtype, accum, &a, &b);
        assert!(
            matches!(
                r,
                Err(PlanError::UnservableContraction(
                    ContractionDecline::AccumPolicyMismatch { .. }
                ))
            ),
            "{dtype:?} {accum:?}: {r:?}"
        );
    }
}

#[test]
fn the_kernel_lift_declines_a_policy_it_does_not_model() {
    let o = op(F32, AccumSpec::Tf32 { k_step: 8 });
    assert!(matches!(
        lift_contraction_opdef(&o, F32),
        Err(LiftDecline::UnsupportedForm(m)) if m.contains("accumulation policy")
    ));
    assert!(lift_contraction_opdef(&op(F32, AccumSpec::WideFloat), F32).is_ok());
}
