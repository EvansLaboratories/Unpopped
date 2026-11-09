// SPDX-License-Identifier: MIT OR Apache-2.0
//! ROADMAP U7 S1b, integer part (`docs/accum-policies.md`; PM 2026-11-09: Int32 and Int32Sat only): an `s8`/`u8` contraction
//! into an `s32` accumulator. `Int32` is the exact sum wrapped to 32 bits; `Int32Sat { k_step }` clamps into the `i32` range after
//! every `k_step` contracted elements (PTX `.satfinite` per `mma`), so a later negative product can bring a saturated value down.
//! The expected values below are worked out by hand from those two sentences, not taken from the oracle's own helper.

use unpopped::ir::{AccumSpec, ContractionAxes, OpDef, reduced};
use unpopped::kernel_lift::{LiftDecline, lift_contraction_opdef};
use unpopped::oracle::{TypedBuffer, evaluate};
use unpopped::plan::{ContractionDecline, PlanError, try_build_plan};
use unpopped_vocab::{
    ArchSku, CellFacts, ContractionRoles, ElementKind, OperandDesc, derive_structure_key,
};

const I8: ElementKind = ElementKind::I8;
const U8: ElementKind = ElementKind::U8;
const I32: ElementKind = ElementKind::I32;

fn raw(t: ElementKind, v: &[i64]) -> Vec<u8> {
    v.iter()
        .map(|&x| match t {
            ElementKind::I8 => (x as i8) as u8,
            _ => x as u8,
        })
        .collect()
}

/// A `[1, K]` row times a `[K, 1]` column; `tweak` may spoil the op or the key facts to reach a decline.
fn run_with(
    ta: ElementKind,
    tb: ElementKind,
    accum: AccumSpec,
    a: &[i64],
    b: &[i64],
    tweak: impl Fn(&mut OpDef, &mut CellFacts),
) -> Result<i32, PlanError> {
    let k = a.len() as i64;
    let lhs = TypedBuffer::new(ta, vec![1, k], vec![k, 1], raw(ta, a));
    let rhs = TypedBuffer::new(tb, vec![k, 1], vec![1, 1], raw(tb, b));
    let ops = [
        OperandDesc::new(2, &[1, k], &[k, 1], ta, 256),
        OperandDesc::new(2, &[k, 1], &[1, 1], tb, 256),
        OperandDesc::new(2, &[1, 1], &[1, 1], I32, 256),
    ];
    let mut facts = CellFacts::new();
    facts.contraction_declared = true;
    facts.weight_operand = Some(1);
    facts.contraction_roles = Some(ContractionRoles::gemm(2, 0));
    facts.accumulator = Some(I32);
    let mut op =
        OpDef::contraction("mm", &[ta], ContractionAxes::matmul(), reduced(0)).with_accum(accum);
    op.out_dtype = Some(I32);
    tweak(&mut op, &mut facts);
    let key = derive_structure_key(&ops, &facts, ArchSku::Sm89).expect("a key");
    let plan = try_build_plan(&op, &key)?;
    let out = evaluate(&plan, &ops, &[lhs, rhs], &[]).remove(0);
    assert_eq!(out.dtype, I32, "the result is stored as i32");
    Ok(i32::from_le_bytes(out.bytes[..4].try_into().unwrap()))
}

fn run(
    ta: ElementKind,
    tb: ElementKind,
    accum: AccumSpec,
    a: &[i64],
    b: &[i64],
) -> Result<i32, PlanError> {
    run_with(ta, tb, accum, a, b, |_, _| {})
}

/// `m` copies of `x`.
fn rep(x: i64, m: usize) -> Vec<i64> {
    vec![x; m]
}

fn cat(parts: &[Vec<i64>]) -> Vec<i64> {
    parts.concat()
}

#[test]
fn int32_is_the_exact_sum_for_small_values_and_reads_the_sign() {
    let a = [127, -128, 5, -1, 0, 100, -100, 3];
    let b = [127, 127, -3, 1, 9, -2, 2, -128];
    // 16129 - 16256 - 15 - 1 + 0 - 200 - 200 - 384 = -927
    assert_eq!(run(I8, I8, AccumSpec::Int32, &a, &b), Ok(-927));
    // the same bytes read as unsigned give a different sum: the signed read is what the model says
    let ua = [200, 255, 5, 1, 0, 100, 100, 3];
    let ub = [127, 255, 3, 1, 9, 2, 2, 128];
    // 25400 + 65025 + 15 + 1 + 0 + 200 + 200 + 384 = 91225
    assert_eq!(run(U8, U8, AccumSpec::Int32, &ua, &ub), Ok(91225));
}

#[test]
fn int32_wraps_at_two_to_the_31() {
    // 131072 products of (-128)(-128) = 16384 sum to exactly 2^31, which wraps to i32::MIN
    let k = 131_072;
    assert_eq!(
        run(I8, I8, AccumSpec::Int32, &rep(-128, k), &rep(-128, k)),
        Ok(i32::MIN)
    );
    // one product fewer stays in range: the control that the wrap, not the data, makes the difference
    assert_eq!(
        run(
            I8,
            I8,
            AccumSpec::Int32,
            &rep(-128, k - 1),
            &rep(-128, k - 1)
        ),
        Ok(2_147_467_264)
    );
    // 16 more positive products past 2^31 wrap to a negative number
    let a = rep(-128, k + 16);
    assert_eq!(
        run(I8, I8, AccumSpec::Int32, &a, &a),
        Ok(i32::MIN + 16 * 16384)
    );
}

#[test]
fn int32_sat_clamps_after_every_step_and_the_step_length_matters() {
    // 131088 products of +16384 (2^31 + 262144), then 32 products of -16256: K = 131120, a multiple of 16
    let a = cat(&[rep(-128, 131_088), rep(127, 32)]);
    let b = cat(&[rep(-128, 131_088), rep(-128, 32)]);
    // every step of 16 is positive until the sum passes 2^31 - 1, where it sticks; then 32 negative products
    // (two steps of 16 x -16256 = -260096 each) bring it down from i32::MAX
    assert_eq!(
        run(I8, I8, AccumSpec::Int32Sat { k_step: 16 }, &a, &b),
        Ok(i32::MAX - 32 * 16256)
    );
    // one step for the whole row: the total 2147745792 - 520192 = 2147225600 fits, so nothing clamps
    assert_eq!(
        run(I8, I8, AccumSpec::Int32Sat { k_step: 131_120 }, &a, &b),
        Ok(2_147_225_600)
    );
    // wrapping is a different model again: the same exact total, which fits, so it agrees with the long step only
    assert_eq!(run(I8, I8, AccumSpec::Int32, &a, &b), Ok(2_147_225_600));
    // a lone positive overflow clamps to i32::MAX (not to the wrapped value)
    let p = rep(-128, 131_088);
    assert_eq!(
        run(I8, I8, AccumSpec::Int32Sat { k_step: 16 }, &p, &p),
        Ok(i32::MAX)
    );
    // and a negative one clamps to i32::MIN
    let n = rep(127, 140_000);
    let m = rep(-128, 140_000);
    assert_eq!(
        run(I8, I8, AccumSpec::Int32Sat { k_step: 16 }, &n, &m),
        Ok(i32::MIN)
    );
}

#[test]
fn an_i8_by_u8_product_is_allowed_and_each_operand_keeps_its_own_signedness() {
    // (-128) * 255 = -32640 per pair; the unsigned reading of the first operand would give a positive number
    let a = rep(-128, 4);
    let b = rep(255, 4);
    assert_eq!(run(I8, U8, AccumSpec::Int32, &a, &b), Ok(-4 * 32640));
}

#[test]
fn a_policy_that_does_not_fit_the_cell_is_a_typed_decline_and_the_control_builds() {
    let a = [1, 2, 3, 4];
    let b = [4, 3, 2, 1];
    assert_eq!(
        run(I8, I8, AccumSpec::Int32, &a, &b),
        Ok(20),
        "control: the valid cell builds"
    );
    let sat = AccumSpec::Int32Sat { k_step: 2 };
    assert_eq!(
        run(I8, I8, sat, &a, &b),
        Ok(20),
        "control: the saturating form builds"
    );
    let declined = |r: Result<i32, PlanError>, needle: &str| match r {
        Err(PlanError::UnservableContraction(ContractionDecline::IntegerPolicyMismatch {
            detail,
        })) => {
            assert!(detail.contains(needle), "{detail}");
        }
        other => panic!("expected a decline mentioning {needle:?}, got {other:?}"),
    };
    declined(
        run_with(I8, I8, AccumSpec::Int32, &a, &b, |o, _| o.out_dtype = None),
        "out_dtype",
    );
    declined(
        run_with(I8, I8, AccumSpec::Int32, &a, &b, |_, f| {
            f.accumulator = None
        }),
        "accumulator",
    );
    declined(
        run_with(I8, I8, AccumSpec::Int32, &a, &b, |o, _| {
            *o = OpDef::contraction(
                "mm",
                &[I8],
                ContractionAxes::matmul(),
                reduced(0) + reduced(0),
            )
            .with_accum(AccumSpec::Int32);
            o.out_dtype = Some(I32);
        }),
        "epilogue",
    );
    declined(
        run(I8, I8, AccumSpec::Int32Sat { k_step: 0 }, &a, &b),
        "k_step",
    );
}

#[test]
fn a_float_cell_under_an_integer_policy_is_declined() {
    let ops = [
        OperandDesc::new(2, &[1, 4], &[4, 1], ElementKind::F32, 256),
        OperandDesc::new(2, &[4, 1], &[1, 1], ElementKind::F32, 256),
        OperandDesc::new(2, &[1, 1], &[1, 1], ElementKind::F32, 256),
    ];
    let mut facts = CellFacts::new();
    facts.contraction_declared = true;
    facts.weight_operand = Some(1);
    facts.contraction_roles = Some(ContractionRoles::gemm(2, 0));
    let key = derive_structure_key(&ops, &facts, ArchSku::Sm89).expect("a key");
    let op = OpDef::contraction(
        "mm",
        &[ElementKind::F32],
        ContractionAxes::matmul(),
        reduced(0),
    )
    .with_accum(AccumSpec::Int32);
    assert!(matches!(
        try_build_plan(&op, &key),
        Err(PlanError::UnservableContraction(
            ContractionDecline::IntegerPolicyMismatch { .. }
        ))
    ));
}

#[test]
fn the_kernel_lift_declines_an_integer_policy_it_does_not_model() {
    let o = OpDef::contraction("mm", &[I8], ContractionAxes::matmul(), reduced(0))
        .with_accum(AccumSpec::Int32);
    assert!(matches!(
        lift_contraction_opdef(&o, I8),
        Err(LiftDecline::UnsupportedForm(m)) if m.contains("accumulation policy")
    ));
}
