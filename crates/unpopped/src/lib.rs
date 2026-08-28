//! # unpopped
//!
//! Build-time generator that turns an op's **abstract IR** (the algorithm) plus
//! a [`unpopped_vocab::StructureKey`] cell (the schedule) into a
//! specialized kernel — and its FKC contract.
//!
//! ## Stability posture (published as of alpha.76)
//!
//! This crate is published so Fuel can construct the live JIT synthesizer from
//! crates.io. The **supported surface is the `seam` feature** — the
//! `fuel_kernel_seam::Synthesizer` impl (`BaracudaSynthesizer`) with its
//! two-step `synthesize`/`take_kernel` handover, frozen with Fuel (2026-07-04).
//! Everything else (the IR, plan, emitters, contracts, dispatch artifacts) is
//! **alpha-fluid generator internals**: the `0.0.1-alpha.N` lockstep implies no
//! cross-version API stability anywhere — pin exact versions.
//!
//! The crate is **language-agnostic, and now holds no emitter at all**:
//!
//! - [`ir`] — the op IR (a [`ScalarExpr`] DAG). Backend-neutral.
//! - [`plan`] — the schedule decision (`StructureKey` → [`KernelPlan`]). Neutral.
//! - [`backend`] — the [`Backend`] trait + the neutral [`backend::lower_expr`].
//! - [`cfamily`] — C-family spelling helpers and emitted software codecs, shared
//!   by every C-shaped emitter so a per-op spelling cannot drift between them.
//!
//! # Where the emitters went
//!
//! Every reference emitter now lives in its own crate: `unpopped-cpu-c` (the
//! portable-C99 one), `unpopped-slang`. The CUDA emitter has always been
//! external, in `baracuda-cuda-emit`.
//!
//! This is the umbrella model — **Unpopped is a standard with a normative
//! reference emitter per target**, and a standard that ships one target's
//! spelling in its core is quietly privileging it. Consequences worth knowing:
//!
//! * A consumer wanting a working kernel needs two crates (`unpopped` plus an
//!   emitter). That is the deliberate cost; the benefit is that no emitter is
//!   the default by accident of location.
//! * This crate's own tests use a **test double**, not a real backend — see
//!   `tests/common/mod.rs`. A core property proven against a real emitter is
//!   proven against two things at once, and blames the wrong one when it fails.
//! * The split is load-bearing rather than cosmetic. It immediately surfaced a
//!   [`backend::Lowering`] seam that had no builder setter, so it was
//!   unreachable from outside this crate — invisible while the emitters lived
//!   in here and wrote the struct literal directly.
//!
//! Cross-emitter evidence (the dtype coverage matrix) lives in
//! `unpopped-conformance`, which depends on every emitter so that no emitter has
//! to depend on its siblings.
//!
//! Op logic is described as IR rather than opaque CUDA precisely so the emitter
//! can *see the dataflow* and transform it (vectorize, hoist, fuse) — and so the
//! same op can be lowered to any backend. A hand-written escape hatch is
//! reserved for bespoke ops the IR can't yet express.
//!
//! This is a dev/build tool (`publish = false`); the artifacts it emits are
//! committed and ship inside `baracuda-kernels-sys`.
//!
//! ## Status (v1)
//!
//! Pilot scope: f32 elementwise ops, contiguous operands, emitted `float4`-
//! vectorized when the cell says V4 and scalar otherwise, lowered to CUDA.
//! Other dtypes, strided / broadcast / reduction schedules, additional
//! backends, FKC emission, and the algebraic optimizer are the growth path.

pub mod backend;
pub mod capability;
pub mod cfamily;
pub mod contract;
#[cfg(feature = "convert")]
pub mod convert;
pub mod dispatch_artifact;
pub mod ir;
pub mod jit;
pub mod kisc;
pub mod lift;
pub mod link;
pub mod optimize;
pub mod oracle;
pub mod pattern;
pub mod plan;
pub mod recipe;
pub mod shape;
pub mod telemetry;
mod text;

// The cross-backend IR fuzzer (`fuzz.rs`) moved to `baracuda-cuda-emit`'s
// integration tests during the Unpopped carve — it drives `generate(&Cuda)`
// (plus CpuC/Slang), so it lives with the CUDA backend crate.

// The in-tree Baracuda ↔ kiss-ref differential converter + comparators (the
// oracle→kiss-ref consolidation). `#[cfg(test)]`-only: migrated numerical tests
// build kiss-ref DAGs from Baracuda OpDefs and assert against `eval_recipe`.
#[cfg(test)]
mod kiss_ref_diff;

pub use backend::{Backend, GeneratedKernel, Variant, VariantFidelity};
pub use contract::{bundle, bundle_kisc, contract, front_matter};
pub use dispatch_artifact::{emit_dispatch_table, parse_dispatch_table};
pub use ir::{
    Access, AccumSpec, AxisRole, ContractionAxes, DagNode, Expr, ExprDag, NodeId, OpDef, ReduceOp,
    ReduceStage, ReductionAccum, ScalarExpr, SortOrder, SortOut, UnaryOp, coord, input, konst,
    param, reduced,
};
pub use jit::{
    ArtifactKind, Compiler, JitBudget, JitError, JitRequest, JitResponse, Recipe, StubCompiler,
    SynthKernel, synthesize,
};
pub use lift::{ConsumeRefusal, LiftError, Lifted, lift_elementwise};
pub use link::{LinkEntry, emit_link_registry, link_entry};
pub use optimize::{optimize, optimize_top_k};
pub use oracle::{Fidelity, TypedBuffer, compare, evaluate};
pub use pattern::{PatternError, PatternNode, derive_pattern, to_fkc};
pub use plan::{KernelPlan, PlanError, Schedule, build_plan};
pub use shape::{
    SYMBOLIC, ShapeError, ShapeRuleForm, output_shape, pooled_axis_dim_expr, shape_rule_form,
    windowed_extent,
};
pub use telemetry::{
    Candidate, DispatchRecord, HwFingerprint, ImplId, Ingest, MissRecord, RankedCell,
    TELEMETRY_SCHEMA_MAX, VariantVote, arch_sku_of, ingest_jsonl, merge_reports, rank_matrix,
    resolve_impl, variant_votes,
};
pub use text::{op_from_text, op_to_text};

use unpopped_vocab::StructureKey;

/// The validity key stamped onto everything this crate generates.
///
/// Built here rather than in the backend because this is the one place that holds
/// both the requested cell and the backend identity, so a backend cannot stamp
/// the wrong cell or forget to stamp at all. See [`backend::Provenance`].
///
/// Honest limit worth stating: this asserts *"produced in response to this
/// request by this generator"*, not *"implements this cell"*. A backend that
/// returned a kernel it built for some other plan would still get the requested
/// key stamped on it. Verifying the kernel matches the cell is the oracle's job,
/// not the stamp's.
fn provenance_for(key: &StructureKey, backend: &dyn Backend) -> backend::Provenance {
    backend::Provenance {
        structure_key: key.to_token(),
        backend: backend.name().to_string(),
        provider: backend.provider().to_string(),
        generator: concat!("unpopped ", env!("CARGO_PKG_VERSION")).to_string(),
    }
}

/// Generate a specialized kernel for `op` at structure cell `key`, lowered by
/// `backend`. Convenience over [`build_plan`] followed by [`Backend::lower`].
///
/// The result carries a [`backend::Provenance`]; a kernel a backend builds
/// directly does not. Anything that caches or ships a kernel should require it.
///
/// # This function PANICS on an inadmissible op, and that is pinned across a
/// # repo boundary
///
/// `generate` is infallible by signature, so it cannot express a decline: it is
/// the trusted-AOT convenience, and it panics with [`plan::PlanError`]'s
/// `Display` where [`try_generate`] returns
/// [`backend::LowerError::InadmissiblePlan`]. **That difference is load-bearing
/// outside this crate.**
///
/// An adopter's §6.8-0004 reachability guard uses the pair as a **born-red
/// anchor**: it asserts that `generate` still aborts on an inadmissible op while
/// `try_generate` declines it. So the guard goes red the moment the two stop
/// differing — which makes it a regression detector, in another repository, on
/// this crate's `try_generate` -> `plan::try_build_plan` wiring.
///
/// Two consequences worth stating rather than discovering:
///
/// - Making `generate` fallible, or making it decline instead of abort, breaks
///   that anchor. It would look like their bug and would not be.
/// - Making `try_generate` panic again — a 0.6.0 regression — also breaks it,
///   which is the direction the anchor exists to catch and is welcome.
///
/// The mirror of this constraint points the other way and lives here too: their
/// 34 `#[should_panic(expected = ...)]` tests match this crate's panic TEXT, and
/// `tests/plan_gate_declines_malformed_input.rs` pins it because they cannot.
#[must_use]
pub fn generate(op: &OpDef, key: &StructureKey, backend: &dyn Backend) -> GeneratedKernel {
    try_generate(op, key, backend)
        .unwrap_or_else(|e| panic!("{} backend declined to lower: {e}", backend.name()))
}

/// [`generate`], but returns the backend's refusal instead of panicking.
///
/// This is the form a JIT or any untrusted-input caller wants: a backend that
/// cannot lower the plan says so in-band, and nothing unwinds across the caller's
/// boundary. [`generate`] is the trusted-AOT convenience over it — op authoring
/// is trusted, so a refusal there is a program error worth panicking on.
///
/// # Errors
///
/// Returns [`backend::LowerError`] when the backend has no lowering for this
/// plan. Note this does **not** cover plan construction: an op/dtype combination
/// the plan gate rejects panics inside [`build_plan`].
///
/// [`plan::try_build_plan`] with [`Backend::lower`] gets you **closer, not all
/// the way** — read its `# Honest scope` before relying on it. It types four of
/// the thirteen gates and then calls [`build_plan`], which re-asserts all
/// thirteen; two of the untyped nine are caller-trippable and measured as such.
/// Full typing of both refusals lands in 0.6.0.
pub fn try_generate(
    op: &OpDef,
    key: &StructureKey,
    backend: &dyn Backend,
) -> Result<GeneratedKernel, backend::LowerError> {
    let mut k = backend.lower(&plan::try_build_plan(op, key)?)?;
    k.stamp(provenance_for(key, backend));
    Ok(k)
}

/// Generate the full **variant set** for a cell: the default lowering (tag
/// `"base"`, bit-identical by definition) plus every backend schedule variant.
/// Ship-top-K policy (variants backlog doc): all validated variants ship, each
/// with its own contract; the item-07 bench gate ranks them per arch and the
/// dispatch table records Baracuda's default — Fuel stays the runtime selector.
#[must_use]
pub fn generate_variants(op: &OpDef, key: &StructureKey, backend: &dyn Backend) -> Vec<Variant> {
    let plan = build_plan(op, key);
    // BASE_OFFSET SLICE: an offsetted kernel's OOB guarantee is a CALLER
    // PRECONDITION (the k/n_out trust model, the gext/sext + RowSort-k<=1024
    // class) — no per-element bounds branch is emitted, so the base variant's
    // launch_note carries the contract. Offset-free plans keep the empty note
    // (byte-stable for every pre-increment op).
    let has_offset =
        plan.base_offsets.iter().any(|b| !b.is_zero()) || !plan.out_base_offset.is_zero();
    let launch_note = if has_offset {
        "offset launch contract: each `long long off*` argument is a runtime base \
         ELEMENT offset added to its operand's base pointer at kernel entry. Caller \
         precondition (no per-element bounds check is emitted): for every offsetted \
         operand, off + <maximal element address reachable via the declared \
         shape/strides (and gext/sext window, if composed)> must lie within the \
         allocated buffer; only off >= 0 is validated."
            .to_string()
    } else {
        String::new()
    };
    let base = backend
        .lower(&plan)
        .unwrap_or_else(|e| panic!("{} backend declined to lower: {e}", backend.name()));
    let mut vs = vec![Variant {
        tag: "base",
        kernels: vec![base],
        fidelity: VariantFidelity::BitIdentical,
        launch_note,
    }];
    vs.extend(
        backend
            .lower_variants(&plan)
            .unwrap_or_else(|e| panic!("{} backend declined a variant: {e}", backend.name())),
    );
    // Stamp EVERY kernel, including those a backend produced via `lower_variants`.
    // A split-K pair ships two kernels under one cell; both are cacheable
    // artifacts and both need to name what they were baked against, or the
    // guarantee holds only for whichever one the caller happened to look at.
    let p = provenance_for(key, backend);
    for v in &mut vs {
        for k in &mut v.kernels {
            k.stamp(p.clone());
        }
    }
    vs
}
