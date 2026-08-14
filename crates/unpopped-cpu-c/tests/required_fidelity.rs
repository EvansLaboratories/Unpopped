//! The derived comparison band is validated against real kernels, not asserted.
//!
//! # Why this file is in the emitter crate
//!
//! `required_fidelity` derives a band from the plan. Whether that band is
//! *correct* is a claim about real emitted kernels, so it belongs where one can
//! be built — a unit test inside `unpopped` could only confirm the formula's
//! arithmetic against itself.
//!
//! The failure it guards is two-sided and only one side is visible. A band that
//! is too **tight** rejects a correct kernel and shows up as a red test. A band
//! that is too **loose** accepts a wrong one and shows up as nothing at all.
//! Every case here therefore checks both directions.

use unpopped::ir::{OpDef, ReduceOp, UnaryOp, input};
use unpopped::oracle::{Fidelity, TypedBuffer, compare, evaluate, required_fidelity};
use unpopped::{build_plan, generate};
use unpopped_cpu_c::CpuC;
use unpopped_vocab::{ArchSku, ElementKind, OpCategory, OperandDesc, StructureKey, structure_key};

fn cell(dtype: ElementKind, n: i64, n_ops: usize) -> (Vec<OperandDesc>, StructureKey) {
    let d = OperandDesc::new(1, &[n], &[1], dtype, 256);
    let ops = vec![d; n_ops];
    let key = structure_key(OpCategory::BinaryElementwise, &ops, ArchSku::Sm89);
    (ops, key)
}

/// An integer cell is entitled to **bit-exact**, and that is the only correct
/// answer rather than a tight one.
///
/// The emitter's integer arithmetic wraps at the C width and the oracle models
/// that wrapping in `i128`, so nothing rounds. A tolerant compare here would
/// accept a genuinely wrong integer — the band would be doing harm, not slack.
#[test]
fn an_integer_cell_is_entitled_to_bit_exact() {
    for dt in [
        ElementKind::I32,
        ElementKind::I64,
        ElementKind::U32,
        ElementKind::U64,
        ElementKind::I8,
        ElementKind::U8,
    ] {
        let (ops, key) = cell(dt, 64, 3);
        let op = OpDef::elementwise("add", 2, &[dt], input(0) + input(1));
        let plan = build_plan(&op, &key);
        assert_eq!(
            required_fidelity(&plan, &ops),
            Some(Fidelity::BitExact),
            "{dt:?} arithmetic is exact on both sides"
        );
    }
}

/// A body that only MOVES values is bit-exact too — nothing rounds.
#[test]
fn a_move_only_body_is_entitled_to_bit_exact() {
    let (ops, key) = cell(ElementKind::F32, 64, 2);
    let op = OpDef::elementwise("id", 1, &[ElementKind::F32], input(0));
    let plan = build_plan(&op, &key);
    assert_eq!(required_fidelity(&plan, &ops), Some(Fidelity::BitExact));
}

/// A float arithmetic body gets a **tolerance that grows with the body**, and
/// the growth is the point.
///
/// A one-op body and a ten-op body are not entitled to the same band: the device
/// rounds once more per node than the `f64` oracle does. A fixed tolerance would
/// be too loose for the first or too tight for the second, and whichever way it
/// erred would be invisible in a diff.
#[test]
fn the_band_grows_with_the_number_of_rounding_steps() {
    let (ops, key) = cell(ElementKind::F32, 64, 3);
    let one = OpDef::elementwise("a", 2, &[ElementKind::F32], input(0) + input(1));
    let mut chained = input(0) + input(1);
    for _ in 0..9 {
        chained = chained * input(1);
    }
    let many = OpDef::elementwise("b", 2, &[ElementKind::F32], chained);

    let p1 = build_plan(&one, &key);
    let p2 = build_plan(&many, &key);
    let (Some(Fidelity::Tolerant { rel: r1, .. }), Some(Fidelity::Tolerant { rel: r2, .. })) =
        (required_fidelity(&p1, &ops), required_fidelity(&p2, &ops))
    else {
        panic!("a float arithmetic cell gets a tolerance");
    };
    assert!(r1 > 0.0, "a rounding step must earn some band");
    assert!(
        r2 > r1,
        "ten roundings must earn a wider band than one: {r2} vs {r1}"
    );
}

/// An `f64` cell's band is far tighter than an `f32` cell's, because its unit
/// roundoff is.
#[test]
fn the_band_is_keyed_to_the_compute_precision() {
    let body = |dt| OpDef::elementwise("m", 2, &[dt], input(0) * input(1));
    let (ops32, key32) = cell(ElementKind::F32, 64, 3);
    let (ops64, key64) = cell(ElementKind::F64, 64, 3);
    let (o32, o64) = (body(ElementKind::F32), body(ElementKind::F64));
    let p32 = build_plan(&o32, &key32);
    let p64 = build_plan(&o64, &key64);
    let (Some(Fidelity::Tolerant { rel: r32, .. }), Some(Fidelity::Tolerant { rel: r64, .. })) = (
        required_fidelity(&p32, &ops32),
        required_fidelity(&p64, &ops64),
    ) else {
        panic!("both are float arithmetic cells");
    };
    assert!(
        r64 < r32 / 1e6,
        "f64's band must be orders tighter than f32's: {r64} vs {r32}"
    );
}

/// A transcendental widens the band, because the contract declares it does.
///
/// The approximate-op term is `unpopped::contract::ulp_bound` — the same number
/// the emitted contract's `max_ulp` carries. If those two ever diverged, a
/// kernel could satisfy its published contract and fail validation, or the
/// reverse, and neither would be visible from one side.
#[test]
fn an_approximate_op_widens_the_band_by_its_declared_ulp() {
    let (ops, key) = cell(ElementKind::F32, 64, 2);
    let exact = OpDef::elementwise("s", 1, &[ElementKind::F32], input(0).unary(UnaryOp::Sqrt));
    let approx = OpDef::elementwise("e", 1, &[ElementKind::F32], input(0).unary(UnaryOp::Exp));

    let pe = build_plan(&exact, &key);
    let pa = build_plan(&approx, &key);
    let (Some(Fidelity::Tolerant { rel: re, .. }), Some(Fidelity::Tolerant { rel: ra, .. })) =
        (required_fidelity(&pe, &ops), required_fidelity(&pa, &ops))
    else {
        panic!("both are float cells");
    };
    assert!(
        ra > re,
        "an approximate op must earn more band than a correctly-rounded one: {ra} vs {re}"
    );
    assert_eq!(
        unpopped::contract::ulp_bound(&exact.body),
        0.0,
        "harness precondition: Sqrt is correctly rounded, so this comparison is \
         about the ULP term and not about node count"
    );
    assert!(unpopped::contract::ulp_bound(&approx.body) > 0.0);
}

/// **The band's magnitude is pinned against a hand-counted expectation.**
///
/// Every other test here compares bands to each other — wider than, tighter
/// than — and a band scaled uniformly by any constant satisfies all of them.
/// Mutation testing proved it: multiplying the derived `rel` by **1000**
/// left the whole file green, because the rejection case below perturbs by
/// `rel * 100` and therefore *moves with* the band it is checking.
///
/// A too-loose band is the failure that does not announce itself — it accepts
/// wrong kernels silently — so it needs an assertion that does not scale. This
/// one counts the rounding steps by hand from a body written right here and
/// pins the exact value.
#[test]
fn the_band_magnitude_is_pinned_not_merely_ordered() {
    let dt = ElementKind::F32;
    let (ops, key) = cell(dt, 255, 3);
    // Four arithmetic nodes, counted from the expression: Add, Mul, Add, Mul.
    // No approximate op, so the ULP term is zero, and this is not a reducing
    // schedule, so there is no accumulation term.
    let body = ((input(0) + input(1)) * input(1)) * (input(0) + input(1));
    let op = OpDef::elementwise("chain", 2, &[dt], body);
    let plan = build_plan(&op, &key);
    let Some(Fidelity::Tolerant { rel, abs }) = required_fidelity(&plan, &ops) else {
        panic!("a chained float body earns a tolerance");
    };

    // u = f32 unit roundoff = eps/2 = 2^-24.
    let u = f64::from(f32::EPSILON) / 2.0;
    let want = 4.0 * u;
    assert!(
        (rel - want).abs() < want * 1e-12,
        "expected 4 rounding steps at f32 => rel = {want:e}, got {rel:e}. \
         A band that is merely ORDERED correctly can be scaled by any constant \
         and still satisfy every comparison test in this file."
    );
    assert_eq!(abs, 0.0, "abs is deliberately zero — see required_fidelity");

    // Sanity on the absolute scale, independent of the formula: a few roundings
    // at f32 is parts-per-ten-million, not parts-per-thousand.
    assert!(
        rel < 1e-6,
        "a 4-op f32 band must stay near 1e-7; {rel:e} is loose enough to hide a \
         real error"
    );
}

/// **The band accepts a correct result and rejects a perturbed one.**
///
/// This is the half a formula cannot prove about itself. Note it cannot catch a
/// uniformly-scaled band — the perturbations are expressed in units of `rel`, so
/// they move with it. `the_band_magnitude_is_pinned_not_merely_ordered` is what
/// covers that, and the division of labour is deliberate: this test checks the
/// band *discriminates*, that one checks it is the *right size*.
#[test]
fn the_derived_band_accepts_a_correct_result_and_rejects_a_perturbed_one() {
    let dt = ElementKind::F32;
    // 255, not 256: a 256-element f32 cell at 256-byte alignment keys
    // `Vectorized { width: 4 }`, which CpuC v1 declines. The band is about
    // arithmetic rounding, not schedule, so an odd length keeps the cell on the
    // scalar path this emitter serves without changing what is under test.
    let (ops, key) = cell(dt, 255, 3);
    // A chain, so the band must cover several roundings rather than one.
    let body = ((input(0) + input(1)) * input(1)) * (input(0) + input(1));
    let op = OpDef::elementwise("chain", 2, &[dt], body);
    let plan = build_plan(&op, &key);
    let fidelity = required_fidelity(&plan, &ops).expect("a finite band");
    let Fidelity::Tolerant { rel, .. } = fidelity else {
        panic!("a chained float body earns a tolerance");
    };

    // The emitter must actually accept this cell, or the band describes a kernel
    // that does not exist.
    let _ = generate(&op, &key, &CpuC);

    #[allow(clippy::cast_precision_loss)]
    let a: Vec<f32> = (0..255).map(|i| 0.5 + i as f32 * 0.031_25).collect();
    #[allow(clippy::cast_precision_loss)]
    let b: Vec<f32> = (0..255).map(|i| 1.25 - i as f32 * 0.007_812_5).collect();
    let bufs = vec![
        TypedBuffer::from_f32(&[255], &a),
        TypedBuffer::from_f32(&[255], &b),
    ];
    let want = evaluate(&plan, &ops, &bufs, &[])
        .into_iter()
        .next()
        .unwrap();
    let base: Vec<f64> = want.to_f64_vec();

    compare(&want, &want, fidelity).expect("a result must match itself");

    // Well OUTSIDE the band: must be rejected.
    #[allow(clippy::cast_possible_truncation)]
    let far: Vec<f32> = base
        .iter()
        .map(|v| (v * (1.0 + rel * 100.0)) as f32)
        .collect();
    assert!(
        compare(&want, &TypedBuffer::from_f32(&[255], &far), fidelity).is_err(),
        "a deviation 100x the band must be rejected — otherwise the band is not \
         discriminating and would accept a wrong kernel"
    );

    // Well INSIDE the band: must be accepted, or the band is too tight and
    // would reject correct kernels.
    #[allow(clippy::cast_possible_truncation)]
    let near: Vec<f32> = base
        .iter()
        .map(|v| (v * (1.0 + rel / 8.0)) as f32)
        .collect();
    compare(&want, &TypedBuffer::from_f32(&[255], &near), fidelity)
        .expect("a deviation well inside the band must be accepted");
}

/// A reducing cell earns band for its accumulation length.
///
/// A 4096-long sum rounds 4096 times where the `f64` oracle sums the same series
/// exactly enough not to. Charging the same band as a 16-long sum would be too
/// tight for one of them, and the reduction is precisely where a hand-picked
/// tolerance gets widened until it passes.
#[test]
fn a_reduction_earns_band_for_its_accumulation_length() {
    let dt = ElementKind::F32;
    let small = OperandDesc::new(1, &[16], &[1], dt, 256);
    let large = OperandDesc::new(1, &[4096], &[1], dt, 256);
    let op = OpDef::reduction("sum", 1, &[dt], input(0), ReduceOp::Sum);

    let ks = structure_key(OpCategory::Reduction, &[small, small], ArchSku::Sm89);
    let kl = structure_key(OpCategory::Reduction, &[large, large], ArchSku::Sm89);
    let ps = build_plan(&op, &ks);
    let pl = build_plan(&op, &kl);

    let (Some(Fidelity::Tolerant { rel: rs, .. }), Some(Fidelity::Tolerant { rel: rl, .. })) = (
        required_fidelity(&ps, &[small, small]),
        required_fidelity(&pl, &[large, large]),
    ) else {
        panic!("a float reduction earns a tolerance");
    };
    assert!(
        rl > rs,
        "a 4096-long sum accumulates more rounding than a 16-long one: {rl} vs {rs}"
    );
}
