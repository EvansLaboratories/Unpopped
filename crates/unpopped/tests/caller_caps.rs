// SPDX-License-Identifier: MIT OR Apache-2.0
//! `CallerCaps` (`docs/options-caller-caps.md`): a caller states what it can do with a lowering's result, and a backend
//! that would emit a shape needing more refuses a legacy caller instead of letting it bind blindly.
//!
//! The load-bearing claims: (1) a backend that overrides only `lower` (Baracuda's situation) produces the **same**
//! kernel through `try_generate`, `try_generate_with(LEGACY)` and `try_generate_with(with_interface())`; (2) a backend that
//! DOES override `lower_with` is refused under `LEGACY` and served under `with_interface()`, so claim (1) can see a
//! difference when there is one; (3) `LEGACY` is the default.

mod common;
use common::StubBackend;
use unpopped::backend::{Backend, CallerCaps, DtypeSupport, GeneratedKernel, LowerError};
use unpopped::ir::{OpDef, input};
use unpopped::kernel_interface::KernelInterface;
use unpopped::plan::KernelPlan;
use unpopped::{try_generate, try_generate_with};
use unpopped_vocab::{
    ArchSku, ElementKind, OpCategory, OperandDesc, StructureKey, TargetId, structure_key,
};

/// The control: needs a caller that binds the interface, so it overrides `lower_with` and its `lower` is the LEGACY case.
struct NeedsInterface;

impl Backend for NeedsInterface {
    fn name(&self) -> &str {
        "needs_interface"
    }
    fn provider(&self) -> &str {
        "unpopped-test"
    }
    fn supports_dtype(&self, _dtype: ElementKind, _target: TargetId) -> DtypeSupport {
        DtypeSupport::Supported
    }
    fn lower(&self, plan: &KernelPlan<'_>) -> Result<GeneratedKernel, LowerError> {
        self.lower_with(plan, &CallerCaps::LEGACY)
    }
    fn lower_with(
        &self,
        plan: &KernelPlan<'_>,
        caps: &CallerCaps,
    ) -> Result<GeneratedKernel, LowerError> {
        if !caps.interface {
            return Err(LowerError::UnsupportedPlanShape {
                detail: "needs a caller that binds `GeneratedKernel::interface`; call `try_generate_with` with \
                         `CallerCaps::LEGACY.with_interface()`"
                    .to_string(),
            });
        }
        Ok(GeneratedKernel::new(
            format!("ni_{}", plan.op_name),
            "/* needs interface */".to_string(),
        )
        .with_interface(KernelInterface::default()))
    }
}

fn key_for(n: i64, dtype: ElementKind) -> StructureKey {
    let d = OperandDesc::new(1, &[n], &[1], dtype, 256);
    structure_key(OpCategory::BinaryElementwise, &[d, d, d], ArchSku::Sm89)
}

fn ops() -> Vec<OpDef> {
    vec![
        OpDef::elementwise("add", 2, &[ElementKind::F32], input(0) + input(1)),
        OpDef::elementwise("mul", 2, &[ElementKind::F32], input(0) * input(1)),
    ]
}

/// Field-by-field equality, since `GeneratedKernel` is not `PartialEq`.
fn same(a: &GeneratedKernel, b: &GeneratedKernel) -> bool {
    a.name == b.name
        && a.source == b.source
        && a.interface() == b.interface()
        && a.provenance() == b.provenance()
        && a.words() == b.words()
        && a.revision_hash() == b.revision_hash()
}

#[test]
fn legacy_is_the_default_and_without_the_interface() {
    assert_eq!(CallerCaps::default(), CallerCaps::LEGACY);
    assert!(!std::hint::black_box(CallerCaps::LEGACY).interface);
    assert!(CallerCaps::LEGACY.with_interface().interface);
    assert_ne!(CallerCaps::LEGACY, CallerCaps::LEGACY.with_interface());
}

#[test]
fn a_lower_only_backend_is_byte_identical_through_all_three_paths() {
    let key = key_for(7, ElementKind::F32);
    for op in ops() {
        let plain = try_generate(&op, &key, &StubBackend).expect("plain");
        let legacy =
            try_generate_with(&op, &key, &StubBackend, &CallerCaps::LEGACY).expect("legacy");
        let with = try_generate_with(
            &op,
            &key,
            &StubBackend,
            &CallerCaps::LEGACY.with_interface(),
        )
        .expect("with");
        assert!(
            same(&plain, &legacy),
            "{}: try_generate vs LEGACY differ",
            op.name
        );
        assert!(
            same(&plain, &with),
            "{}: try_generate vs with_interface differ",
            op.name
        );
        assert!(
            plain.provenance().is_some(),
            "{}: the stamp is applied on every path",
            op.name
        );
    }
}

/// The control that lets the test above see a difference: a backend that overrides `lower_with` differs by caps.
#[test]
fn a_backend_that_needs_the_interface_refuses_legacy_and_serves_the_opted_in_caller() {
    let key = key_for(7, ElementKind::F32);
    let op = &ops()[0];

    let refused = try_generate(op, &key, &NeedsInterface).expect_err("try_generate is LEGACY");
    let LowerError::UnsupportedPlanShape { detail } = &refused else {
        panic!("expected UnsupportedPlanShape, got {refused:?}");
    };
    assert!(
        detail.contains("with_interface"),
        "the refusal names the capability: {detail}"
    );
    assert!(matches!(
        try_generate_with(op, &key, &NeedsInterface, &CallerCaps::LEGACY),
        Err(LowerError::UnsupportedPlanShape { .. })
    ));

    let served = try_generate_with(
        op,
        &key,
        &NeedsInterface,
        &CallerCaps::LEGACY.with_interface(),
    )
    .expect("opted in");
    assert!(
        served.interface().is_some(),
        "the opted-in caller gets the interface"
    );
    assert!(served.provenance().is_some(), "and the stamp");
    // `Backend::lower` on its own is the LEGACY case too.
    assert!(
        NeedsInterface
            .lower(&unpopped::plan::try_build_plan_for(op, &key, &StubBackend).expect("plan"))
            .is_err()
    );
}
