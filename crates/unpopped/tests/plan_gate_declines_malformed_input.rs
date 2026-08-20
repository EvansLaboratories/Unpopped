//! **CLOSED, and now pinned the other way.** `try_generate` returns a typed
//! decline on a malformed `OpDef`, as KISS-EMIT §6.8-0004 requires.
//!
//! > On **any** input, including malformed, empty, truncated, or adversarial
//! > input, an emitter MUST NOT panic, abort, crash, hang, or read outside its
//! > input buffers; it MUST instead return the typed decline of §6.8-0001.
//!
//! §6.1-0001 fixes what "input" means: *"The normative input to a conforming
//! emitter MUST be exactly the pair `(OpDef, structure_key)`."* §6.1-0002 adds
//! that an emitter MUST NOT take a schedule-resolved plan as that input — so
//! [`unpopped::backend::Backend::lower`], which takes a `KernelPlan`, is not the
//! conforming interface. `try_generate` is.
//!
//! # This file was a `#[should_panic]` gap pin for about an hour
//!
//! It landed green — asserting the violation — with the note that it would go
//! **red the moment the gap closed**, so the fix could not land silently and
//! could not be quietly satisfied. It went red on the first run of the 0.6.0
//! conversion, which is the only proof that shape is worth anything.
//!
//! Kept rather than deleted, and inverted rather than rewritten from scratch:
//! the same input, the same clause, the opposite assertion. A test that once
//! pinned a defect is the cheapest possible regression test for it.
//!
//! # What changed underneath
//!
//! `try_generate` called `build_plan`, whose thirteen gates were `assert!`s.
//! `plan::try_build_plan` existed and typed **four** of them, then called
//! `build_plan`, which re-ran all thirteen — so the documented "when both
//! refusals must be typed" path aborted on nine.
//!
//! 0.6.0: two more gates gained `check_*` forms (coord admissibility, the
//! RowReduce validator — the only two the census found a caller could trip),
//! `try_build_plan` stopped calling `build_plan`, and `build_plan` became its
//! panicking wrapper. `try_generate` now calls `try_build_plan` and maps
//! `PlanError` into `LowerError::InadmissiblePlan`.
//!
//! The seven remaining gates still panic **on purpose**: they are structural
//! invariants about how an `OpDef` was constructed, no input reaches them
//! (measured over 1280 pairs in `adversarial_input_panic_census.rs`), and an
//! internal invariant that returns `Err` teaches its caller to handle a case
//! that cannot happen.
//!
//! # How the defect was found, which is still the part worth keeping
//!
//! By diffing an adopter's migration. Their `rowreduce_forward_reduced_ref_panics`
//! is `#[should_panic]` — an accurate name for accurate behaviour, pinning *our*
//! panic from *their* crate. They asked a scoping question about their own
//! backstops; the answer applied harder to the generator than to the emitter
//! that asked. That test had been naming this defect from inside another
//! project's suite for months, correctly, and nobody read it as a bug report.

use unpopped::backend::{Backend, GeneratedKernel, LowerError};
use unpopped::ir::{OpDef, ReduceOp, ReduceStage, input, reduced};
use unpopped::plan::KernelPlan;
use unpopped::try_generate;
use unpopped_vocab::{ArchSku, ElementKind, OpCategory, OperandDesc, TargetId, structure_key};

/// Spells everything, so the panic cannot be mistaken for a backend decline.
struct Spells;

impl Backend for Spells {
    fn name(&self) -> &str {
        "spells"
    }
    fn provider(&self) -> &str {
        "unpopped-tests"
    }
    fn supports_dtype(&self, _dtype: ElementKind, _target: TargetId) -> bool {
        true
    }
    fn lower(&self, _plan: &KernelPlan<'_>) -> Result<GeneratedKernel, LowerError> {
        Ok(GeneratedKernel::new(
            "k".to_string(),
            "/* spelled */".to_string(),
        ))
    }
}

fn key() -> unpopped_vocab::StructureKey {
    let d = OperandDesc::new(1, &[7], &[1], ElementKind::F32, 4);
    structure_key(OpCategory::Normalization, &[d, d], ArchSku::Sm89)
}

/// A stage-0 `pre` referencing `Reduced(0)` — its own not-yet-produced result.
/// Malformed by construction: the op names a value that does not exist yet.
fn forward_referencing_row_reduce() -> OpDef {
    OpDef::row_reduce(
        "malformed",
        1,
        &[ElementKind::F32],
        vec![ReduceStage {
            pre: reduced(0).0,
            op: ReduceOp::Sum,
        }],
        input(0) * reduced(0),
    )
}

#[test]
fn try_generate_declines_a_malformed_opdef_typed_per_kiss_emit_6_8_0004() {
    let err = try_generate(&forward_referencing_row_reduce(), &key(), &Spells)
        .expect_err("a malformed OpDef must be declined, never lowered");

    assert!(
        matches!(err, LowerError::InadmissiblePlan { .. }),
        "the plan gate refused, so this must be InadmissiblePlan and not a          backend variant — the backend was never consulted. Got {err:?}"
    );

    // The refusal still SAYS what was wrong. A typed decline whose payload lost
    // the reason would satisfy the clause and help nobody, and this exact string
    // is what `build_plan` still panics with, so an adopter's
    // `#[should_panic(expected = ...)]` against the panicking path keeps
    // matching.
    assert!(
        err.to_string()
            .contains("references a stage not yet produced"),
        "the decline must carry the gate's own explanation, got {err}"
    );
}

/// The panicking path is unchanged, and that is deliberate.
///
/// `build_plan` is infallible by signature, so it cannot express a decline; it
/// stays the trusted-AOT convenience and panics with `PlanError`'s `Display`.
/// An adopter has 34 `#[should_panic(expected = ...)]` tests against it that
/// this crate cannot run — so the message text is load-bearing across a repo
/// boundary, and this pins it here where it can fail.
#[test]
fn build_plan_still_panics_with_the_gates_own_words() {
    let op = forward_referencing_row_reduce();
    let k = key();
    let caught = std::panic::catch_unwind(|| {
        let prev = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let r = std::panic::catch_unwind(|| unpopped::build_plan(&op, &k));
        std::panic::set_hook(prev);
        r
    })
    .expect("outer");
    let msg = *caught
        .expect_err("build_plan must still panic on a malformed OpDef")
        .downcast::<String>()
        .expect("panic payload is a String");
    assert!(
        msg.contains("references a stage not yet produced"),
        "build_plan's panic text changed; an adopter matches on it. Got {msg:?}"
    );
}

/// The control: `try_generate` **does** decline other malformed input typed,
/// so the pin above is a specific gap and not a blanket "we never decline".
///
/// Without this, the pin could be read as "the plan gate has no decline path at
/// all", which would be a different and much larger claim than the one being
/// made.
#[test]
fn the_same_entry_point_declines_an_unspellable_dtype_typed() {
    struct SpellsNothing;
    impl Backend for SpellsNothing {
        fn name(&self) -> &str {
            "spells-nothing"
        }
        fn provider(&self) -> &str {
            "unpopped-tests"
        }
        fn supports_dtype(&self, _dtype: ElementKind, _target: TargetId) -> bool {
            false
        }
        fn lower(&self, plan: &KernelPlan<'_>) -> Result<GeneratedKernel, LowerError> {
            Err(LowerError::UnsupportedDtype {
                dtype: plan.dtype,
                detail: "declines everything".to_string(),
            })
        }
    }

    let ok = OpDef::elementwise("f", 2, &[ElementKind::F32], input(0) + input(1));
    let d = OperandDesc::new(1, &[7], &[1], ElementKind::F32, 4);
    let k = structure_key(OpCategory::BinaryElementwise, &[d, d, d], ArchSku::Sm89);

    let err = try_generate(&ok, &k, &SpellsNothing)
        .expect_err("a backend that spells nothing must produce a typed decline");
    assert!(
        matches!(err, LowerError::UnsupportedDtype { .. }),
        "expected a typed decline, got {err:?}"
    );
}
