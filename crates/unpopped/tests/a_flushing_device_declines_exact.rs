// SPDX-License-Identifier: MIT OR Apache-2.0
//! **A device whose measured result flushes subnormals does not get an Exact claim.**
//!
//! The `unpopped-harness` run of 2026-10-21 compared the Slang `relu` with the cpu-c reference over
//! all 2^32 f32 inputs: by default an AMD Radeon 610M and an RTX 4070 Laptop both returned a negative
//! subnormal flushed to zero, and with `-denorm-mode-fp32 preserve` both were exact. The Vulkan
//! property bits do not say which (NVIDIA reports `shaderDenormPreserveFloat32 = false` yet honours
//! the mode), so the only gate is the measurement. These tests drive that measurement into the two
//! places an Exact claim is made: the emitted contract's `precision` block and the oracle's
//! `required_fidelity`.
//!
//! Each assertion has its known-bad control beside it: the flushing device is declined, the
//! preserving one admitted, an unmeasured one declined, and a record for another driver, another
//! argument set, another body or another dtype changes nothing.

mod common;

use common::StubBackend;
use unpopped::backend::GeneratedKernel;
use unpopped::contract::{
    AccuracyKey, MeasuredOn, contract, contract_with_exactness, precision_of, precision_of_measured,
};
use unpopped::exactness::{
    Coverage, DeviceStamp, ExactnessLedger, ExactnessRecord, ExactnessVerdict, Measured,
};
use unpopped::ir::{BinaryOp, OpDef, ScalarExpr, UnaryOp, input};
use unpopped::oracle::{Fidelity, required_fidelity, required_fidelity_with_exactness};
use unpopped::plan::build_plan;
use unpopped_vocab::structure_key;
use unpopped_vocab::{ElementKind, HwStamp, OpCategory, OperandDesc, StructureKey, TargetId};

const VK: &str = "vulkan:sg64.ops-abr.arith-f16.cm-none";
const PRESERVE: [&str; 2] = ["-denorm-mode-fp32", "preserve"];
/// The hash of the kernel under test. Every record here names it, so these tests keep asking
/// the question they always asked; `exactness_admission.rs` varies it.
const HASH: &str = "sha256:00";

fn vk() -> TargetId {
    TargetId::parse(VK).unwrap()
}

fn stamp(target: TargetId, driver: &str, args: &[&str]) -> DeviceStamp {
    DeviceStamp::new(
        HwStamp {
            target,
            device_name: "AMD Radeon 610M".into(),
            runtime_version: "1.3.280".into(),
            captured_unix_s: 1,
        },
        driver,
        "slangc 2026.4",
        args.iter().map(|s| (*s).to_string()).collect(),
    )
}

fn relu_body() -> ScalarExpr {
    ScalarExpr::Unary(UnaryOp::Relu, Box::new(ScalarExpr::Input(0)))
}

fn record(
    stamp: &DeviceStamp,
    body: &ScalarExpr,
    dtype: ElementKind,
    verdict: ExactnessVerdict,
) -> ExactnessRecord {
    ExactnessRecord::new(
        stamp.clone(),
        body,
        dtype,
        verdict,
        HASH,
        "unpopped-slang 0.21.0",
        Coverage::exhaustive(1 << 32),
    )
}

fn ledger_of(
    stamp: &DeviceStamp,
    body: &ScalarExpr,
    dtype: ElementKind,
    verdict: ExactnessVerdict,
) -> ExactnessLedger {
    let mut l = ExactnessLedger::new();
    l.push(record(stamp, body, dtype, verdict));
    l
}

fn relu_op() -> OpDef {
    OpDef::elementwise(
        "relu",
        1,
        &[ElementKind::F32],
        input(0).unary(UnaryOp::Relu),
    )
}

fn cell(target: TargetId) -> StructureKey {
    let d = OperandDesc::new(1, &[64], &[1], ElementKind::F32, 64);
    structure_key(OpCategory::UnaryElementwise, &[d, d], target)
}

fn kernel() -> GeneratedKernel {
    GeneratedKernel::new("relu_f32".into(), "/* stub */".into())
}

/// The `mode:` the contract's precision block declares.
fn declared_mode(contract_text: &str) -> &'static str {
    if contract_text.contains("correctly_rounded") {
        "correctly_rounded"
    } else if contract_text.contains("approximate") {
        "approximate"
    } else {
        panic!("no precision mode in:\n{contract_text}")
    }
}

fn key_for(t: TargetId) -> AccuracyKey {
    AccuracyKey::for_target(t)
}

// --- the bound ----------------------------------------------------------------------------

#[test]
fn an_unmeasured_vulkan_float_body_declines_and_a_preserving_measurement_admits_it() {
    let body = relu_body();
    let s = stamp(vk(), "24.1", &PRESERVE);
    let exact = ledger_of(&s, &body, ElementKind::F32, ExactnessVerdict::Exact);
    let m = MeasuredOn {
        stamp: &s,
        kernel_hash: HASH,
        ledger: &exact,
        dtype: ElementKind::F32,
    };
    assert_eq!(
        precision_of(&body, &key_for(vk())),
        ("approximate", None),
        "control: with nothing measured the body declines"
    );
    assert_eq!(
        precision_of_measured(&body, &key_for(vk()), Some(&m)),
        ("correctly_rounded", Some(0)),
        "an Exact run on this stamp admits it"
    );
}

#[test]
fn a_flushing_measurement_declines_even_where_the_table_says_exact() {
    let body = relu_body();
    let cuda = AccuracyKey::for_target(unpopped_vocab::ArchSku::Sm89.into());
    let s = stamp(unpopped_vocab::ArchSku::Sm89.into(), "555.1", &[]);
    let flush = ledger_of(&s, &body, ElementKind::F32, ExactnessVerdict::Flushes);
    let m = MeasuredOn {
        stamp: &s,
        kernel_hash: HASH,
        ledger: &flush,
        dtype: ElementKind::F32,
    };
    assert_eq!(
        precision_of(&body, &cuda),
        ("correctly_rounded", Some(0)),
        "control: the CUDA table calls relu exact"
    );
    assert_eq!(
        precision_of_measured(&body, &cuda, Some(&m)),
        ("approximate", None)
    );
}

#[test]
fn a_flushing_measurement_outweighs_an_exact_one_on_the_same_stamp() {
    let body = relu_body();
    let s = stamp(vk(), "24.1", &PRESERVE);
    let mut l = ledger_of(&s, &body, ElementKind::F32, ExactnessVerdict::Exact);
    l.push(record(
        &s,
        &body,
        ElementKind::F32,
        ExactnessVerdict::Flushes,
    ));
    let m = MeasuredOn {
        stamp: &s,
        kernel_hash: HASH,
        ledger: &l,
        dtype: ElementKind::F32,
    };
    assert_eq!(
        precision_of_measured(&body, &key_for(vk()), Some(&m)),
        ("approximate", None)
    );
}

/// A record is a measurement of a driver and a flag set. Each of these is a different device, so
/// the Exact record must not admit it.
#[test]
fn a_record_for_another_driver_args_body_or_dtype_admits_nothing() {
    let body = relu_body();
    let measured = stamp(vk(), "24.1", &PRESERVE);
    let exact = ledger_of(&measured, &body, ElementKind::F32, ExactnessVerdict::Exact);
    let ask = |stamp: &DeviceStamp, body: &ScalarExpr, dtype: ElementKind| {
        let m = MeasuredOn {
            stamp,
            kernel_hash: HASH,
            ledger: &exact,
            dtype,
        };
        precision_of_measured(body, &key_for(stamp.hw.target), Some(&m))
    };
    // Control: the matching question is admitted, or the rows below prove nothing.
    assert_eq!(
        ask(&measured, &body, ElementKind::F32),
        ("correctly_rounded", Some(0))
    );
    let declined = ("approximate", None);
    assert_eq!(
        ask(&stamp(vk(), "24.2", &PRESERVE), &body, ElementKind::F32),
        declined,
        "driver"
    );
    assert_eq!(
        ask(&stamp(vk(), "24.1", &[]), &body, ElementKind::F32),
        declined,
        "args"
    );
    assert_eq!(ask(&measured, &body, ElementKind::F16), declined, "dtype");
    let neg = ScalarExpr::Unary(UnaryOp::Neg, Box::new(ScalarExpr::Input(0)));
    assert_eq!(ask(&measured, &neg, ElementKind::F32), declined, "body");
}

/// Agreeing with a reference that has an error budget is not correct rounding: an Exact record on
/// `exp` must not turn into a `correctly_rounded` claim.
#[test]
fn an_exact_record_does_not_make_an_operator_with_an_error_budget_exact() {
    let exp = ScalarExpr::Unary(UnaryOp::Exp, Box::new(ScalarExpr::Input(0)));
    let s = stamp(vk(), "24.1", &PRESERVE);
    let l = ledger_of(&s, &exp, ElementKind::F32, ExactnessVerdict::Exact);
    let m = MeasuredOn {
        stamp: &s,
        kernel_hash: HASH,
        ledger: &l,
        dtype: ElementKind::F32,
    };
    assert_eq!(
        precision_of_measured(&exp, &key_for(vk()), Some(&m)),
        ("approximate", None)
    );
}

/// Binary bodies are not covered by the unary exhaustive run: no record, no claim.
#[test]
fn a_binary_body_with_no_record_is_declined_whatever_the_unary_records_say() {
    let add = ScalarExpr::Binary(
        BinaryOp::Max,
        Box::new(ScalarExpr::Input(0)),
        Box::new(ScalarExpr::Input(1)),
    );
    let s = stamp(vk(), "24.1", &PRESERVE);
    let l = ledger_of(&s, &relu_body(), ElementKind::F32, ExactnessVerdict::Exact);
    let m = MeasuredOn {
        stamp: &s,
        kernel_hash: HASH,
        ledger: &l,
        dtype: ElementKind::F32,
    };
    assert_eq!(
        precision_of_measured(&add, &key_for(vk()), Some(&m)),
        ("approximate", None)
    );
}

// --- the two places an Exact claim is made -------------------------------------------------

/// The mode the relu cell's contract declares, given one device's stamp and ledger.
fn relu_contract_mode(s: &DeviceStamp, l: &ExactnessLedger) -> &'static str {
    let c = contract_with_exactness(
        &relu_op(),
        &cell(vk()),
        &kernel(),
        &StubBackend,
        Some(Measured::new(s, HASH, l)),
    )
    .expect("a contract");
    declared_mode(&c)
}

#[test]
fn the_emitted_contract_withdraws_correct_rounding_from_a_flushing_device() {
    let unmeasured =
        contract(&relu_op(), &cell(vk()), &kernel(), &StubBackend).expect("a contract");
    let s = stamp(vk(), "24.1", &PRESERVE);
    let flush = ledger_of(
        &s,
        &relu_body(),
        ElementKind::F32,
        ExactnessVerdict::Flushes,
    );
    let exact = ledger_of(&s, &relu_body(), ElementKind::F32, ExactnessVerdict::Exact);

    assert_eq!(
        declared_mode(&unmeasured),
        "approximate",
        "unmeasured declines"
    );
    assert_eq!(
        relu_contract_mode(&s, &flush),
        "approximate",
        "a flushing device declines"
    );
    assert_eq!(
        relu_contract_mode(&s, &exact),
        "correctly_rounded",
        "control: a preserving measurement admits it, or the rows above prove nothing"
    );
}

/// A stamp for another target is not a measurement of this cell.
#[test]
fn the_emitted_contract_ignores_a_stamp_for_another_target() {
    let other = stamp(unpopped_vocab::ArchSku::Sm89.into(), "555.1", &PRESERVE);
    let exact = ledger_of(
        &other,
        &relu_body(),
        ElementKind::F32,
        ExactnessVerdict::Exact,
    );
    assert_eq!(relu_contract_mode(&other, &exact), "approximate");
}

#[test]
fn required_fidelity_makes_no_claim_for_a_flushing_device() {
    let op = relu_op();
    let k = cell(vk());
    let plan = build_plan(&op, &k);
    let d = OperandDesc::new(1, &[64], &[1], ElementKind::F32, 64);
    let ops = [d, d];
    let s = stamp(vk(), "24.1", &PRESERVE);
    let exact = ledger_of(&s, &relu_body(), ElementKind::F32, ExactnessVerdict::Exact);
    let flush = ledger_of(
        &s,
        &relu_body(),
        ElementKind::F32,
        ExactnessVerdict::Flushes,
    );

    assert_eq!(required_fidelity(&plan, &ops), None, "unmeasured: no claim");
    assert!(
        matches!(
            required_fidelity_with_exactness(&plan, &ops, Some(Measured::new(&s, HASH, &exact))),
            Some(Fidelity::Tolerant { .. })
        ),
        "control: an Exact record admits the body (a finite band), or the next row proves nothing"
    );
    assert_eq!(
        required_fidelity_with_exactness(&plan, &ops, Some(Measured::new(&s, HASH, &flush))),
        None,
        "a flushing device: no claim"
    );
}
