//! A minimal [`Backend`] for tests that need *a* backend, not a *particular* one.
//!
//! # Why core's tests use a double now
//!
//! The reference emitters moved into their own crates (`unpopped-cpu-c`,
//! `unpopped-slang`), so `unpopped` has no backend of its own — which is the
//! point: the core is the standard, the IR, the plan and the traits, and it
//! should be provable without any target's spelling.
//!
//! A test that reaches for a real emitter to test a *core* property is testing
//! two things and blaming the wrong one when it fails. `generate` stamping
//! provenance has nothing to do with C; a double makes that independence
//! visible rather than incidental.
//!
//! Deliberately the *smallest* thing that satisfies the trait. It emits a stub
//! source string, not compilable code — anything that needs real C belongs in
//! `unpopped-cpu-c` with the emitter that produces it.

// A `tests/common/` module is compiled INTO each test binary that declares it,
// so an item only one binary uses is genuinely unreachable from the others'
// perspective. That is the nature of a shared test helper, not a mistake.
//
// `dead_code` is allowed for the same reason and was added when the first
// binary to use only ONE of the two doubles appeared: `OtherStub` exists so a
// test can prove a stamp REPORTS its backend rather than returning a constant,
// which needs two backends. A binary that needs one of them is not evidence the
// other is dead — it is evidence this module is shared, which is its purpose.
#![allow(unreachable_pub, dead_code)]

use unpopped::backend::{Backend, GeneratedKernel, LowerError};
use unpopped::plan::KernelPlan;
use unpopped_vocab::{ElementKind, TargetId};

/// Lowers anything to a stub, so a core test can exercise the machinery around
/// lowering without depending on a real emitter.
#[derive(Copy, Clone, Debug, Default)]
pub struct StubBackend;

impl Backend for StubBackend {
    fn name(&self) -> &str {
        "stub"
    }
    fn provider(&self) -> &str {
        "unpopped-test"
    }
    fn supports_dtype(&self, _dtype: ElementKind, _target: TargetId) -> bool {
        true
    }
    fn lower(&self, plan: &KernelPlan<'_>) -> Result<GeneratedKernel, LowerError> {
        // The name embeds the dtype so a test can tell two cells apart without
        // this double growing an emitter's worth of behaviour.
        Ok(GeneratedKernel::new(
            format!("stub_{}_{:?}", plan.op_name, plan.dtype),
            format!("/* stub: {} */", plan.op_name),
        ))
    }
}

/// A second double whose only difference from [`StubBackend`] is its identity.
///
/// Exists so a test can prove a stamp *reports* the backend rather than
/// returning a constant — a check that needs two backends and cannot be written
/// with one.
#[derive(Copy, Clone, Debug, Default)]
pub struct OtherStub;

impl Backend for OtherStub {
    fn name(&self) -> &str {
        "other_stub"
    }
    fn provider(&self) -> &str {
        "someone-else"
    }
    fn supports_dtype(&self, _dtype: ElementKind, _target: TargetId) -> bool {
        true
    }
    fn lower(&self, plan: &KernelPlan<'_>) -> Result<GeneratedKernel, LowerError> {
        Ok(GeneratedKernel::new(
            format!("other_{}", plan.op_name),
            format!("/* other stub: {} */", plan.op_name),
        ))
    }
}
