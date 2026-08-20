//! **GAP PIN.** `try_generate` panics on a malformed `OpDef`, and KISS-EMIT
//! §6.8-0004 says it must not.
//!
//! > On **any** input, including malformed, empty, truncated, or adversarial
//! > input, an emitter MUST NOT panic, abort, crash, hang, or read outside its
//! > input buffers; it MUST instead return the typed decline of §6.8-0001.
//!
//! And §6.1-0001 fixes what "input" means: *"The normative input to a conforming
//! emitter MUST be exactly the pair `(OpDef, structure_key)`."* There is no
//! trusted-authoring carve-out in either clause.
//!
//! # Why this is `unpopped`'s problem and not a backend's
//!
//! §6.1-0002 goes further: an emitter MUST NOT take a **schedule-resolved plan**
//! as its normative input. So [`unpopped::backend::Backend::lower`], which takes
//! a `KernelPlan`, is *not* the conforming emitter interface — `try_generate` is,
//! because it has the `(OpDef, structure_key)` signature the clause names.
//!
//! The panic sits between the two: `build_plan` runs a twelve-check
//! admissibility layer (`assert_valid_out_dtype`, `assert_coord_admissibility`,
//! `validate_row_reduce`, …) that asserts rather than declines. A backend never
//! sees the input that trips it, so no backend can be at fault, and no backend
//! can fix it.
//!
//! # How it was found, which is the part worth keeping
//!
//! By diffing an adopter's migration. Their test
//! `rowreduce_forward_reduced_ref_panics` is `#[should_panic]` — an accurate
//! name for accurate behaviour, pinning *our* panic from *their* crate. They
//! were asking a scoping question about their own remaining backstops; the
//! answer applied harder to the generator than to the emitter that asked.
//!
//! The `backend.rs` stance that *"AOT op authoring is trusted, so `lower` itself
//! may still panic on a dtype it can't spell"* is **ours, not the clause's**.
//! Editing KISS-Emit does not entitle its editor to read a carve-out into it
//! that the text does not contain, and this crate is the reference
//! implementation, so the gap is visible to anyone who runs it.
//!
//! # Why a pin rather than a fix
//!
//! Closing it changes `build_plan` from `-> KernelPlan` to a fallible return,
//! which is a **breaking change** to a `pub fn` in a crate published hours ago,
//! cascading through `generate` / `try_generate` and every `Backend`. That is a
//! release decision and an adopter-coordination decision, not a cleanup.
//!
//! So this pin is `#[should_panic]`: **green while the gap is open, red the
//! moment it closes.** It cannot be quietly satisfied, and the fix cannot land
//! without editing this file — which is the only mechanism that reliably makes a
//! fix announce itself.

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
#[should_panic(expected = "references a stage not yet produced")]
fn try_generate_panics_on_a_malformed_opdef_and_kiss_emit_6_8_0004_says_it_must_not() {
    let _ = try_generate(&forward_referencing_row_reduce(), &key(), &Spells);
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
