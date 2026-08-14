//! A backend can answer `supports_dtype` *differently per target*.
//!
//! # Why this test exists at all
//!
//! `Backend::supports_dtype` grew a `TargetId` parameter so a backend could stop
//! answering "always" or "never" for dtypes whose availability is
//! target-conditional. Neither in-tree backend exercises that: `CpuC` emits
//! portable C99 (target-independent by nature) and `Slang` still declines its
//! narrow integers everywhere for want of capability data.
//!
//! So every existing test would pass if the parameter were **accepted and
//! ignored** — if `generate` never threaded a target through, or threaded the
//! wrong one. A parameter no caller can act on is worse than no parameter,
//! because it advertises a capability that is not there.
//!
//! This file supplies the backend that does act on it, and checks the answer
//! changes with the target and nothing else.

use unpopped::backend::{Backend, GeneratedKernel, LowerError};
use unpopped::ir::{OpDef, input};
use unpopped::plan::KernelPlan;
use unpopped::try_generate;
use unpopped_vocab::{ArchSku, ElementKind, OpCategory, OperandDesc, TargetId, structure_key};

/// A backend that spells `f16` only where the target's capability set says so.
///
/// The rule it applies is deliberately trivial — "the capability-set contains
/// `arith-f16`" — because the *rule* is not what is under test. What is under
/// test is that the target reaching this method is the one the kernel is keyed
/// to. A real backend's rule would come from a capability manifest rather than a
/// substring, which is the KISS #171 dependency the Slang backend documents.
#[derive(Copy, Clone, Debug, Default)]
struct CapabilityAware;

impl Backend for CapabilityAware {
    fn name(&self) -> &str {
        "capability_aware"
    }
    fn provider(&self) -> &str {
        "unpopped-test"
    }

    fn supports_dtype(&self, dtype: ElementKind, target: TargetId) -> bool {
        match dtype {
            ElementKind::F16 => target.capability_set().contains("arith-f16"),
            ElementKind::F32 => true,
            _ => false,
        }
    }

    fn lower(&self, plan: &KernelPlan<'_>) -> Result<GeneratedKernel, LowerError> {
        if !self.supports_dtype(plan.dtype, plan.key.target) {
            return Err(LowerError::UnsupportedDtype {
                dtype: plan.dtype,
                detail: format!("no {:?} on target {}", plan.dtype, plan.key.target.as_str()),
            });
        }
        Ok(GeneratedKernel::new(
            format!("k_{}", plan.op_name),
            format!("/* {} on {} */", plan.op_name, plan.key.target.as_str()),
        ))
    }
}

fn cell(dtype: ElementKind, target: TargetId) -> unpopped_vocab::StructureKey {
    let d = OperandDesc::new(1, &[64], &[1], dtype, 64);
    structure_key(OpCategory::BinaryElementwise, &[d, d, d], target)
}

/// The same dtype, two targets, two answers — and the difference is the target.
#[test]
fn one_dtype_two_targets_two_answers() {
    let capable = TargetId::parse("vulkan:sg64.ops-abr.arith-f16.cm-none").unwrap();
    let incapable = TargetId::parse("vulkan:sg64.ops-abr.cm-none").unwrap();

    assert!(CapabilityAware.supports_dtype(ElementKind::F16, capable));
    assert!(!CapabilityAware.supports_dtype(ElementKind::F16, incapable));

    // And `f32` is unconditional, so the two targets agree on it — which shows
    // the difference above is about the DTYPE-target pair, not about one of the
    // targets being rejected wholesale.
    assert!(CapabilityAware.supports_dtype(ElementKind::F32, capable));
    assert!(CapabilityAware.supports_dtype(ElementKind::F32, incapable));
}

/// The target the backend sees is the one the KEY carries.
///
/// This is the half a unit test on `supports_dtype` cannot reach. `generate`
/// could plausibly build a plan whose `key.target` is not the target the caller
/// asked for — defaulted, or stamped from somewhere else — and every direct call
/// to `supports_dtype` would still pass. Going through `try_generate` is what
/// proves the thread is connected end to end.
#[test]
fn the_target_the_backend_sees_is_the_one_the_key_carries() {
    let op = OpDef::elementwise("add", 2, &[ElementKind::F16], input(0) + input(1));
    let capable = TargetId::parse("vulkan:sg64.ops-abr.arith-f16.cm-none").unwrap();
    let incapable = TargetId::parse("vulkan:sg64.ops-abr.cm-none").unwrap();

    let ok = try_generate(&op, &cell(ElementKind::F16, capable), &CapabilityAware)
        .expect("the capable target must lower");
    assert!(
        ok.source.contains("arith-f16"),
        "the emitter saw the capable target: {}",
        ok.source
    );

    let err = try_generate(&op, &cell(ElementKind::F16, incapable), &CapabilityAware)
        .expect_err("the incapable target must decline");
    let msg = err.to_string();
    assert!(
        msg.contains("F16"),
        "the decline must name the dtype it refused: {msg}"
    );
    assert!(
        !msg.contains("arith-f16"),
        "and it must have been asked about the INCAPABLE target: {msg}"
    );
}

/// A CUDA target reaches the backend as its `cuda:` token, not as an enum.
///
/// The `From<ArchSku>` conversion is what keeps existing call sites compiling,
/// and it would be easy for it to hand across something the backend cannot
/// inspect. It hands across the real token.
#[test]
fn an_arch_sku_call_site_still_delivers_an_inspectable_target() {
    let key = cell(ElementKind::F32, ArchSku::Sm89.into());
    assert_eq!(key.target.namespace(), "cuda");
    assert_eq!(key.target.capability_set(), "sm89");

    let op = OpDef::elementwise("add", 2, &[ElementKind::F32], input(0) + input(1));
    let k = try_generate(&op, &key, &CapabilityAware).expect("f32 is unconditional");
    assert!(k.source.contains("cuda:sm89"), "{}", k.source);
}
