// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Measured exactness for kernel-layer kernels** (ROADMAP U59, `docs/design-kernel-fidelity.md`): the ledger that
//! `a_flushing_device_declines_exact.rs` exercises for an op body answers the same questions for a `KernelFn`, through
//! `kernel_key`, `ExactnessRecord::for_kernel` and `ExactnessLedger::{find,lookup,admit}_kernel`.
//!
//! The case that started it: a Metal device flushes f32 subnormals and preserves f16. Each assertion has its known-bad
//! control beside it, as there: the flushing stamp is declined, a preserving exhaustive record admitted, an unmeasured
//! one declined, and a record for another stamp, hash, kernel, dtype or an op body changes nothing.

use unpopped::exactness::{
    Admission, Coverage, DeviceStamp, ExactnessLedger, ExactnessRecord, ExactnessVerdict, Finding,
    Requirement, Withheld, body_key, kernel_key,
};
use unpopped::ir::ScalarExpr;
use unpopped::kernel::{Block, KernelFn};
use unpopped_vocab::{ElementKind, HwStamp, TargetId};

const F32: ElementKind = ElementKind::F32;
const F16: ElementKind = ElementKind::F16;
const HASH: &str = "sha256:aa";

fn stamp(args: &[&str]) -> DeviceStamp {
    DeviceStamp::new(
        HwStamp {
            target: TargetId::parse("vulkan:sg64.ops-abr.arith-f16.cm-none").unwrap(),
            device_name: "Apple M-class".into(),
            runtime_version: "msl 3.1".into(),
            captured_unix_s: 1,
        },
        "driver 1",
        "metal 31.0",
        args.iter().map(|s| (*s).to_string()).collect(),
    )
}

fn kernel(name: &str) -> KernelFn {
    KernelFn::new(name, F32, vec![], vec![], Block(vec![]))
}

fn rec(
    st: &DeviceStamp,
    k: &KernelFn,
    dtype: ElementKind,
    verdict: ExactnessVerdict,
    coverage: Coverage,
    hash: &str,
) -> ExactnessRecord {
    ExactnessRecord::for_kernel(
        st.clone(),
        k,
        dtype,
        verdict,
        hash,
        "unpopped-msl 0",
        coverage,
    )
}

fn ledger(r: ExactnessRecord) -> ExactnessLedger {
    let mut l = ExactnessLedger::new();
    l.push(r);
    l
}

fn exact_required(
    l: &ExactnessLedger,
    st: &DeviceStamp,
    k: &KernelFn,
    dt: ElementKind,
) -> Admission {
    l.admit_kernel(st, HASH, k, dt, Requirement::Exact)
}

#[test]
fn a_flushing_device_is_declined_and_a_preserving_exhaustive_record_admits() {
    let (k, st) = (kernel("relu_like"), stamp(&[]));
    // measured: this kernel flushes f32 subnormals on this stamp (one mismatch, sampled is enough)
    let flushes = ledger(rec(
        &st,
        &k,
        F32,
        ExactnessVerdict::Flushes,
        Coverage::sampled(1000, Some(7)),
        HASH,
    ));
    assert_eq!(
        exact_required(&flushes, &st, &k, F32),
        Admission::Declined(Withheld::Flushes)
    );
    assert_eq!(
        flushes.lookup_kernel(&st, HASH, &k, F32),
        Some(ExactnessVerdict::Flushes)
    );
    // control: an exhaustive Exact record on the same stamp admits
    let exact = ledger(rec(
        &st,
        &k,
        F32,
        ExactnessVerdict::Exact,
        Coverage::exhaustive(1 << 32),
        HASH,
    ));
    assert_eq!(exact_required(&exact, &st, &k, F32), Admission::Admitted);
    // control: no record claims nothing
    assert_eq!(
        exact_required(&ExactnessLedger::new(), &st, &k, F32),
        Admission::Declined(Withheld::Unmeasured)
    );
    // BestEffort keeps the kernel but withholds the claim
    assert_eq!(
        flushes.admit_kernel(&st, HASH, &k, F32, Requirement::BestEffort),
        Admission::ClaimWithheld(Withheld::Flushes)
    );
}

#[test]
fn a_sampled_exact_admits_nothing_and_a_flush_outweighs_an_exact() {
    let (k, st) = (kernel("k"), stamp(&[]));
    let sampled = ledger(rec(
        &st,
        &k,
        F32,
        ExactnessVerdict::Exact,
        Coverage::sampled(10, None),
        HASH,
    ));
    assert_eq!(
        sampled.find_kernel(&st, HASH, &k, F32),
        Finding::Withheld(Withheld::SampledOnly)
    );
    let mut both = ExactnessLedger::new();
    both.push(rec(
        &st,
        &k,
        F32,
        ExactnessVerdict::Exact,
        Coverage::exhaustive(1 << 32),
        HASH,
    ));
    both.push(rec(
        &st,
        &k,
        F32,
        ExactnessVerdict::Flushes,
        Coverage::sampled(1, None),
        HASH,
    ));
    assert_eq!(
        both.find_kernel(&st, HASH, &k, F32),
        Finding::Withheld(Withheld::Flushes),
        "a Flushes dominates, sampled or not"
    );
}

#[test]
fn a_record_for_another_stamp_hash_kernel_dtype_or_an_op_body_changes_nothing() {
    let (k, st) = (kernel("k"), stamp(&[]));
    let l = ledger(rec(
        &st,
        &k,
        F32,
        ExactnessVerdict::Flushes,
        Coverage::sampled(1, None),
        HASH,
    ));
    let unmeasured = Admission::Declined(Withheld::Unmeasured);
    // another dtype (the f16 path preserves: nothing was measured for it)
    assert_eq!(exact_required(&l, &st, &k, F16), unmeasured);
    // another kernel
    assert_eq!(exact_required(&l, &st, &kernel("other"), F32), unmeasured);
    // another compiler argument set is another device
    assert_eq!(
        exact_required(&l, &stamp(&["-fpreserve-denormals"]), &k, F32),
        unmeasured
    );
    // the same stamp and kernel under another hash is stale, not a mismatch
    assert_eq!(
        l.admit_kernel(&st, "sha256:bb", &k, F32, Requirement::Exact),
        Admission::Declined(Withheld::StaleKernel)
    );
    // an op-layer record never answers a kernel question, and a kernel record never answers an op body
    let body = ScalarExpr::Input(0);
    let op = ledger(ExactnessRecord::new(
        st.clone(),
        &body,
        F32,
        ExactnessVerdict::Flushes,
        HASH,
        "unpopped-msl 0",
        Coverage::sampled(1, None),
    ));
    assert_eq!(exact_required(&op, &st, &k, F32), unmeasured);
    assert_eq!(
        l.admit(&st, HASH, &body, F32, Requirement::Exact),
        unmeasured
    );
}

#[test]
fn a_preserving_argument_set_is_offered_for_a_flushing_one() {
    // the same device and compiler, exact with different arguments: the op layer's UseArgs offer works for kernels too
    let k = kernel("k");
    let (flushing, preserving) = (stamp(&[]), stamp(&["-fpreserve-denormals"]));
    let mut l = ExactnessLedger::new();
    l.push(rec(
        &flushing,
        &k,
        F32,
        ExactnessVerdict::Flushes,
        Coverage::sampled(1, None),
        HASH,
    ));
    l.push(rec(
        &preserving,
        &k,
        F32,
        ExactnessVerdict::Exact,
        Coverage::exhaustive(1 << 32),
        "sha256:cc",
    ));
    let got = exact_required(&l, &flushing, &k, F32);
    let Admission::UseArgs(offer) = got else {
        panic!("expected an offer, got {got:?}");
    };
    assert_eq!(
        offer.compiler_args,
        vec!["-fpreserve-denormals".to_string()]
    );
    assert_eq!(offer.kernel_hash, "sha256:cc");
}

#[test]
fn the_kernel_key_is_pinned_and_apart_from_the_body_key() {
    let k = kernel("k");
    let key = kernel_key(&k, F32);
    assert!(key.starts_with("kernel:f32:{"), "{key}");
    assert!(
        key.contains(r#""name":"k""#),
        "the kernel's own JSON follows the prefix: {key}"
    );
    assert_ne!(key, kernel_key(&k, F16), "the dtype is part of the key");
    assert_ne!(
        key,
        kernel_key(&kernel("k2"), F32),
        "the kernel is part of the key"
    );
    let body = body_key(&ScalarExpr::Input(0), F32);
    assert!(
        !body.starts_with("kernel:"),
        "a body key never collides with a kernel key: {body}"
    );
    // the records survive the ledger's JSONL file
    let l = ledger(rec(
        &stamp(&[]),
        &k,
        F32,
        ExactnessVerdict::Flushes,
        Coverage::sampled(1, None),
        HASH,
    ));
    let read = ExactnessLedger::parse_jsonl(&l.to_jsonl());
    assert_eq!(
        read.ledger.find_kernel(&stamp(&[]), HASH, &k, F32),
        Finding::Withheld(Withheld::Flushes)
    );
}
