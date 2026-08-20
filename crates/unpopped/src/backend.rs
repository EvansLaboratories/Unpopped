//! Backend abstraction — the one language-specific seam.
//!
//! Everything else in the crate (the [`crate::ir`] op IR and the schedule
//! decision in [`crate::plan`]) is language-agnostic. A [`Backend`] lowers a
//! neutral [`crate::plan::KernelPlan`] to concrete kernel source. CUDA is the
//! first impl (the CUDA backend); Slang / SPIR-V / Metal / CPU backends
//! slot in as additional impls without touching the core — which is what lets
//! this generator eventually target backends beyond CUDA (and move out of
//! Baracuda) without a rewrite.

use crate::ir::{ArithOp, BinaryOp, DagNode, ExprDag, NodeId, ScalarExpr, UnaryOp};
use unpopped_vocab::{ElementKind, TargetId};

/// What a generated kernel was baked against — its validity key.
///
/// A cached or persisted kernel outlives the call that produced it, and nothing
/// in `(name, source)` says which cell it implements. A cache keyed on the side,
/// by whatever the caller remembered, is correct only by convention — and
/// [`crate::jit::Compiler`] explicitly anticipates "a pre-built-variant cache"
/// slotting in, which is exactly where that convention would be relied on.
///
/// Fuel arrived at the general rule the expensive way — four silent, full-speed
/// wrong-answer incidents in two days, every one an artifact outliving something
/// it was baked against (model identity twice, a recorded CUDA graph, a KV
/// allocation). Their `14-lifecycle.md` states it as: *a held artifact's validity
/// key MUST name every such thing, or that thing MUST be unable to change for the
/// artifact's lifetime — enforced by construction, not by convention.* They
/// offered it as cross-project text; this is Unpopped honouring the "by
/// construction" half.
///
/// Which is why this is stamped by [`crate::generate`] and not by the backend.
/// The core holds the key and the backend identity at the one point both are
/// known, so a backend cannot stamp the wrong cell — or forget.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct Provenance {
    /// The cell identity token this kernel was generated for.
    ///
    /// Carried as the token rather than a parsed key, byte-exact and never
    /// interpreted here — KISS-CLASSIFY-6.9-0002 opaque carry. Comparing it is a
    /// byte comparison (6.8-0002), never a subset or implication test.
    pub structure_key: String,
    /// [`Backend::name`] at generation time — the target language.
    pub backend: String,
    /// [`Backend::provider`] at generation time — who supplied the backend.
    pub provider: String,
    /// The generator that baked this kernel — `"unpopped <version>"`.
    ///
    /// [`Self::structure_key`] names **what was asked for**, not **what was
    /// produced**. Those come apart: change a lowering rule or an optimizer
    /// rewrite and the same request yields different code under an unchanged
    /// token. A consumer caching on the key alone would then serve a kernel baked
    /// by a generator that no longer exists, with nothing to detect it.
    ///
    /// Scope, stated precisely because it is easy to overclaim: caching on the key
    /// alone is *already* forbidden — KISS-Synth §6.7-0004 requires a cache hit to
    /// match `(structure_key, revision_hash)`, and this crate's
    /// `kernel_revision_hash` is FNV-1a over the emitted **source**, i.e. over the
    /// product rather than the recipe. A lowering change therefore already yields
    /// a different hash. So this field is **not** what closes that hole, and it is
    /// not a substitute for the revision hash.
    ///
    /// What it adds: an identity for the producer that survives where a
    /// source-text hash does not. When [`GeneratedKernel`] grows a non-source
    /// artifact — a SPIR-V word stream, a cubin — a hash over `source` covers
    /// nothing, and the same holds across the *compile* step, where identical
    /// source through a different toolchain version yields different device code
    /// under an unchanged source hash. It is also comparable without hashing.
    ///
    /// So the producer must version what it bakes, because only the producer knows
    /// what it bakes — a consumer cannot enumerate that, and the baked set changes
    /// on the producer's release cadence, not the consumer's. (Fuel reached this
    /// from the consumer side: an operand-identity check cannot detect a producer
    /// that started baking one more thing.)
    ///
    /// Deliberately coarse. The crate version over-invalidates — a release that
    /// changed no lowering still re-mints — and that is the intended trade: it is
    /// stamped automatically, so it cannot be forgotten, and it fails toward a
    /// needless rebuild rather than toward a stale kernel. A precise
    /// rules-digest could replace it later without changing this field's meaning.
    ///
    /// Coarse is right **here**, not universally, and the deciding variable is the
    /// cost of a *false* invalidation. For a generator that cost is a recompile,
    /// so over-invalidating is nearly free and unforgettable beats precise. For a
    /// consumer holding a live artifact it can be ruinous — Fuel measured a 223×
    /// reuse factor on a held decode plan, where a version-granularity stamp would
    /// fire every step and destroy the thing it protects. They needed precise
    /// never-recycled identity for the same rule. Do not copy this granularity
    /// across the seam; copy the obligation.
    pub generator: String,
}

/// A generated kernel: its exported symbol name and source text.
///
/// `#[non_exhaustive]`: a backend artifact grows a structured payload later (the
/// Vulkane review's #1/#2 — a SPIR-V `[u32]` word stream + an ABI/binding manifest
/// can't ride a `String`). Reserving the shape now lets that land in a minor
/// version. Out-of-crate backends build kernels through [`GeneratedKernel::new`].
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct GeneratedKernel {
    /// The exported (`extern "C"` or backend-equivalent) symbol name.
    pub name: String,
    /// The kernel source text, in the backend's language.
    pub source: String,
    /// Set by [`crate::generate`]; `None` on a fragment a backend built directly.
    /// Private so only the core can stamp it — see [`GeneratedKernel::provenance`].
    provenance: Option<Provenance>,
}

impl GeneratedKernel {
    /// The canonical constructor — the only way an out-of-crate [`Backend`] builds
    /// a `GeneratedKernel` (the struct is `#[non_exhaustive]`).
    ///
    /// The result carries no [`Provenance`]: a backend is producing a fragment,
    /// and it is [`crate::generate`] that knows which cell was asked for. That
    /// asymmetry is deliberate — see [`Provenance`].
    #[must_use]
    pub fn new(name: String, source: String) -> Self {
        Self {
            name,
            source,
            provenance: None,
        }
    }

    /// What this kernel was baked against, or `None` if it is a raw fragment.
    ///
    /// `Some` exactly when the kernel came through [`crate::generate`] or
    /// [`crate::generate_variants`]. **Anything that caches, persists or ships a
    /// kernel should require `Some` and treat `None` as unfit** — a `None` kernel
    /// is one whose cell identity is known only to whoever happens to be holding
    /// it, which is the convention this type exists to replace.
    #[must_use]
    pub fn provenance(&self) -> Option<&Provenance> {
        self.provenance.as_ref()
    }

    /// Stamp the validity key. Crate-internal so the core is the only stamper.
    pub(crate) fn stamp(&mut self, p: Provenance) {
        self.provenance = Some(p);
    }
}

/// How a schedule variant's computed bits relate to the cell's default lowering.
/// Drives the selection policy (variants backlog doc): a [`VariantFidelity::BitIdentical`]
/// variant may be selected silently; anything else is selectable only through an
/// honest FKC contract (the caller's precision policy decides), never silently.
/// `#[non_exhaustive]`: the 3-way accumulator/order/compensated split (the
/// deferred fidelity work) adds variants; core matches them, backends only
/// produce them, so reserving room here is free.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum VariantFidelity {
    /// Same result bits as the default lowering for every input.
    BitIdentical,
    /// Deterministic (fixed order for a fixed launch configuration), but a
    /// different operation *association* than the default — e.g. a split-K
    /// partial-sum tree vs the sequential fold.
    ReassociatedDeterministic,
    /// **Run-to-run non-deterministic** (increment 5, SCATTER): the result bits
    /// vary *between launches of the same configuration* because the schedule
    /// accumulates through order-varying floating-point atomics (`atomicAdd` on
    /// an FP cell whose completion order the hardware does not fix), and FP add
    /// is non-associative. This is strictly weaker than
    /// [`Self::ReassociatedDeterministic`] (which is at least stable for a fixed
    /// launch): a `Nondeterministic` variant can differ from ITSELF run to run.
    ///
    /// Per the house variant-selection rule this may **never** be selected
    /// silently — only through an honest FKC contract whose determinism block
    /// flips to Fuel's `nondeterministic` spelling (and, per Fuel's precision
    /// coherence rule `fuel-dispatch fkc/validate.rs`, that contract must also
    /// carry `bit_stable_on_same_hardware: false` + `audited: true`). The
    /// deterministic default (a gather-sum or sorted-segment sweep — the bespoke
    /// `segment_sorted_kernel` precedent) stays the base route.
    Nondeterministic,
    /// **Strictly more accurate** than the default lowering, and deterministic.
    /// The default f32 reduction accumulates in `float` (error growing with the
    /// reduced length); this variant forces a `double` accumulator and a
    /// no-reassociation serial fold, yielding ~0.5 ULP(f32) of the correctly-
    /// rounded reduction — a *directed* "closer to the true reduction" guarantee.
    /// That directedness is the whole selection signal, and it is why this is
    /// neither [`Self::BitIdentical`] (it differs from the default bits, so it
    /// must never be chosen silently) nor [`Self::ReassociatedDeterministic`]
    /// (which is same-accuracy-different-rounding, undirected).
    ///
    /// The serial double fold is bitwise-reproducible on any IEEE-754 double
    /// hardware (fixed order, per-op-deterministic), so its determinism spelling
    /// is the strongest — `bitwise`, not `same_hardware_bitwise`. Selectable only
    /// through an honest FKC contract whose precision block advertises the tighter
    /// bound; the caller's precision policy decides.
    MorePrecise,
}

impl VariantFidelity {
    /// The Fuel FKC `determinism:` block spelling for this fidelity — the exact
    /// string Fuel's contract schema accepts (`fuel-dispatch fkc/schema.rs`:
    /// `bitwise` | `same_hardware_bitwise` | `nondeterministic`). Used when a
    /// variant's contract is emitted so the determinism block flips **honestly**
    /// with the schedule's numeric class (never a hardcoded `bitwise`).
    ///
    /// `Nondeterministic` maps to `nondeterministic`, which — per Fuel's
    /// precision coherence rule (`fkc/validate.rs` Rule 9) — additionally
    /// obligates the emitted precision block to declare
    /// `bit_stable_on_same_hardware: false` + `audited: true`.
    #[must_use]
    pub fn determinism_str(self) -> &'static str {
        match self {
            VariantFidelity::BitIdentical => "bitwise",
            VariantFidelity::ReassociatedDeterministic => "same_hardware_bitwise",
            VariantFidelity::Nondeterministic => "nondeterministic",
            // A serial double fold is reproducible across IEEE-754 hardware.
            VariantFidelity::MorePrecise => "bitwise",
        }
    }
}

/// One alternative schedule for a cell: a tagged set of kernels (most variants
/// are a single kernel; a split-K pair is two, in launch order) plus the launch
/// protocol the contract must carry. The ship-top-K policy: every validated
/// variant ships with its own contract under the same `accept.structure_key`;
/// the dispatch table records Baracuda's measured default, and Fuel remains the
/// runtime selector.
///
/// `#[non_exhaustive]`: a richer per-variant artifact (the Vulkane review's #1/#2
/// binding/ABI manifest) attaches here later; out-of-crate backends build variants
/// through [`Variant::new`] so that addition stays a minor version.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct Variant {
    /// Short stable tag (`"base"`, `"splitk"`, `"unroll4"`, …). Rides in the
    /// generated symbol names and, eventually, contract front-matter (opaque on
    /// the wire — the entry point stays the true identity).
    pub tag: &'static str,
    /// The kernels implementing this variant, in launch order.
    pub kernels: Vec<GeneratedKernel>,
    /// Bit relationship to the default lowering.
    pub fidelity: VariantFidelity,
    /// Launch protocol (grids, workspace sizing, chunking) — contract-facing.
    pub launch_note: String,
}

impl Variant {
    /// The canonical constructor — the only way an out-of-crate [`Backend`] builds
    /// a `Variant` (the struct is `#[non_exhaustive]`).
    #[must_use]
    pub fn new(
        tag: &'static str,
        kernels: Vec<GeneratedKernel>,
        fidelity: VariantFidelity,
        launch_note: String,
    ) -> Self {
        Self {
            tag,
            kernels,
            fidelity,
            launch_note,
        }
    }
}

/// Why a [`Backend`] declined to lower a plan.
///
/// # Why lowering refuses in-band rather than panicking
///
/// A backend's legality surface is **its own**, and it is the only thing that
/// knows it. Before this existed, `lower` panicked, and the JIT kept the
/// `Synthesizer` boundary safe by pre-screening requests against
/// `jit::dtype_compatible` — a mirror of the neutral AOT plan gate, maintained
/// so a plan-gate `assert!` could not unwind into the caller.
///
/// That mirror covered *dtype and op* legality. It did not cover **schedule**
/// legality, because nothing in it asked whether the backend could emit the cell
/// the planner chose. A `u8` `Add` over 256 aligned elements plans to
/// `Vectorized { width: 8 }`, which the CpuC v1 emitter does not serve — so the
/// simplest region anyone would write panicked straight through `synthesize`.
/// Pinned by `tests/jit_never_unwinds.rs`.
///
/// The lesson is about *dimension*, not about vendors: a pre-screen can only
/// cover the axes someone thought to enumerate, whereas the backend refusing
/// in-band covers every axis by construction, because it is the thing that would
/// otherwise have failed.
///
/// The variants are **derived from the refusals that already existed** in
/// `cpu_c` and `slang` rather than invented, so every previous panic has a
/// natural home.
///
/// `detail` carries the human-readable reason the panic used to carry; it is for
/// diagnostics, never for matching. Callers that need to branch match the
/// variant.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum LowerError {
    /// No scalar spelling for this dtype at all (`cpu_c` declining f16/bf16).
    UnsupportedDtype {
        /// The dtype the backend cannot spell.
        dtype: ElementKind,
        /// Why, in the backend's own words.
        detail: String,
    },
    /// The backend cannot emit this schedule (a scalar-only emitter handed a
    /// `Vectorized`/`Strided` cell).
    UnsupportedSchedule {
        /// Why, in the backend's own words.
        detail: String,
    },
    /// No spelling for an operation or expression node — an op with no intrinsic
    /// on this target, or one used at a dtype it has no lowering for.
    UnsupportedOp {
        /// Why, in the backend's own words.
        detail: String,
    },
    /// The plan's *shape* is outside what this backend emits — multi-output, a
    /// runtime scalar param, a hetero output dtype.
    UnsupportedPlanShape {
        /// Why, in the backend's own words.
        detail: String,
    },
    /// **No plan was built.** The op is inadmissible in the requested cell, so
    /// no backend was consulted.
    ///
    /// Distinct from every variant above, which are a *backend* refusing a plan
    /// that exists. This one is the plan gate refusing to make one. Folding it
    /// into [`Self::UnsupportedPlanShape`] would put "the backend cannot emit
    /// this" and "there is nothing to emit" under one label, leaving `detail` —
    /// diagnostics, never matched on — as the only way to tell them apart.
    ///
    /// Before 0.6.0 this case did not exist because the gate *panicked*, which
    /// KISS-EMIT §6.8-0004 forbids: on any input an emitter must return the
    /// typed decline instead.
    InadmissiblePlan {
        /// The gate's refusal, typed.
        source: crate::plan::PlanError,
    },
}

impl From<crate::plan::PlanError> for LowerError {
    fn from(source: crate::plan::PlanError) -> Self {
        LowerError::InadmissiblePlan { source }
    }
}

/// What a [`Lowering`] seam returns: a spelling, **or a typed refusal to spell**.
///
/// # Why a decline is a SUCCESS value and not an error variant
///
/// The obvious design is one error enum with `Declined` and `Failed` variants.
/// **It is wrong, and the reason is `?`.** With a single error type,
/// `let s = spell(..)?;` compiles, reads naturally, and silently converts a
/// *decline* into a propagated *failure* — every call site has to remember to
/// discriminate, and the one that forgets is indistinguishable from the one that
/// didn't.
///
/// With `Result<Spelling, LowerError>`, `?` propagates only real failures and a
/// decline **forces a `match`**. That makes the lazy path the correct one, which
/// matters because the lazy path is the one that ships.
///
/// Owed to Vulkane, who had solved the same shape three times in their own crate:
/// `cooperative_matrix_properties()` returned a bare `Vec` where empty meant both
/// *"the device supports none"* and *"the query failed"*. It is now `Result<Vec<_>>`
/// — extension absent is `Ok(empty)`, missing entry point is `Err`. **"Supports
/// none" is an answer; "I could not ask" is a failure.**
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Spelling {
    /// The target-language text for this node.
    Spelled(String),
    /// This backend does not spell this — a **capability statement**, not a fault.
    Declined(Decline),
}

/// Why a [`Lowering`] seam declined, as a **matchable** value.
///
/// KISS-EMIT §6.8-0001 requires a **typed** decline. A `String` reason cannot be
/// matched on, so it cannot be the type — it can only ride along for diagnostics.
///
/// # This carries WHAT was declined, deliberately
///
/// Baracuda's cross-backend suite asserts *"Slang declines `Copysign`"* by
/// inspecting a panic payload, so that an unrelated crash cannot satisfy it. When
/// the panic became a typed value that discrimination had to survive, or their test
/// would have got **weaker at exactly the moment this API got better**.
/// `matches!(d, Decline::UnsupportedOp { op: BinaryOp::Copysign, .. })` is
/// strictly stronger than payload-string matching and is structurally
/// unsatisfiable by an unrelated failure.
///
/// `#[non_exhaustive]`: `NotExpressible` is deliberately here from the start so the
/// `supports_dtype` third state — supported / unsupported / **not expressible** —
/// has a home when it is designed, without a second breaking change.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Decline {
    /// The backend has no spelling for this op at all.
    UnsupportedOp {
        /// The op, so a caller can match on the specific refusal.
        op: DeclinedOp,
        /// Diagnostics only. Never match on this.
        why: String,
    },
    /// The backend spells this op, but not at this dtype.
    UnsupportedDtypeForOp {
        /// The op it otherwise spells.
        op: DeclinedOp,
        /// The dtype it will not spell it at.
        dtype: ElementKind,
        /// Diagnostics only. Never match on this.
        why: String,
    },
    /// The target language cannot express this at all, at any dtype — distinct
    /// from "this backend has not implemented it". Reserved for the
    /// `supports_dtype` third state.
    NotExpressible {
        /// Diagnostics only. Never match on this.
        why: String,
    },
}

/// Which op a [`Decline`] is about, so the refusal is matchable rather than
/// described.
///
/// Deliberately **not** a `String`: a declined op named by text is a decline a
/// caller can only recognise by spelling, which is the thing this type exists to
/// stop.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum DeclinedOp {
    /// A [`UnaryOp`] the backend will not spell.
    Unary(UnaryOp),
    /// A [`BinaryOp`] the backend will not spell.
    Binary(BinaryOp),
    /// An [`ArithOp`] the backend will not spell.
    Arith(ArithOp),
    /// The ternary select.
    Select,
    /// A constant literal.
    Constant,
    /// An input operand leaf the backend will not spell, by **operand index**.
    ///
    /// Carries the index because a leaf refusal is usually *positional* — "this
    /// emitter spells at most two operands" — and an index is matchable where a
    /// sentence about one is not.
    Leaf(u8),
    /// A per-row reduced-scalar leaf ([`ScalarExpr::Reduced`]), by index.
    Reduced(u8),
    /// An output-coordinate leaf ([`ScalarExpr::Coord`]), by axis.
    Coord(u8),
}

impl Spelling {
    /// The spelled text, or the decline.
    ///
    /// # Errors
    ///
    /// Returns the [`Decline`] unchanged when this is not a spelling. Provided so
    /// a caller that genuinely wants to treat a decline as terminal can say so in
    /// one place, rather than each call site inventing the conversion.
    pub fn spelled(self) -> Result<String, Decline> {
        match self {
            Spelling::Spelled(s) => Ok(s),
            Spelling::Declined(d) => Err(d),
        }
    }
}

impl From<Decline> for LowerError {
    /// A decline that reaches the top of a lowering becomes the emitter's typed
    /// decline (KISS-EMIT §6.8-0001/-0002), preserving which op refused.
    fn from(d: Decline) -> Self {
        match d {
            Decline::UnsupportedOp { op, why } => LowerError::UnsupportedOp {
                detail: format!("{op:?}: {why}"),
            },
            Decline::UnsupportedDtypeForOp { op, dtype, why } => LowerError::UnsupportedDtype {
                dtype,
                detail: format!("{op:?}: {why}"),
            },
            Decline::NotExpressible { why } => LowerError::UnsupportedOp {
                detail: format!("not expressible in this target language: {why}"),
            },
        }
    }
}

impl std::fmt::Display for LowerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedDtype { dtype, detail } => {
                write!(f, "unsupported dtype {dtype:?}: {detail}")
            }
            Self::UnsupportedSchedule { detail } => write!(f, "unsupported schedule: {detail}"),
            Self::UnsupportedOp { detail } => write!(f, "unsupported op: {detail}"),
            Self::UnsupportedPlanShape { detail } => write!(f, "unsupported plan shape: {detail}"),
            Self::InadmissiblePlan { source } => write!(f, "inadmissible plan: {source}"),
        }
    }
}

impl std::error::Error for LowerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InadmissiblePlan { source } => Some(source),
            _ => None,
        }
    }
}

/// Lowers a neutral [`crate::plan::KernelPlan`] to concrete kernel source.
pub trait Backend {
    /// Short backend identifier / target language (e.g. `"cuda"`, `"cpu_c"`,
    /// `"slang"`). Canonicalized into the FKC `backend:` field.
    fn name(&self) -> &str;
    /// The **provider identity** — WHO supplies this backend, emitted verbatim as
    /// the FKC contract's `kernel_source:` provenance field (a field consumers may
    /// parse). This is distinct from [`Backend::name`] (the target language): the
    /// same neutral generator can be provided by different vendors, and a neutral
    /// core cannot know who its provider is, so each backend declares it. Required
    /// (no default) precisely so a new backend can't silently inherit the wrong
    /// provenance — the identity must be stated, not defaulted. Baracuda's CUDA
    /// backend returns `"baracuda"`; the generator's own in-tree reference backends
    /// (CpuC / Slang) return the generator's name.
    fn provider(&self) -> &str;
    /// Lower a kernel plan to source, or say why it cannot.
    ///
    /// # Errors
    ///
    /// Returns [`LowerError`] when this backend has no lowering for the plan —
    /// an unspellable dtype, a schedule it does not emit, an op with no
    /// intrinsic on this target, or a plan shape outside its scope.
    ///
    /// **Refuse rather than approximate.** A backend that cannot spell something
    /// exactly must return `Err`, never fall through to a default that compiles
    /// but computes something else. That trades a visible refusal for a silent
    /// numerical bug, which is strictly worse — see the "decline, do not fall
    /// through" note in `tests/neutral_spelling.rs`.
    fn lower(&self, plan: &crate::plan::KernelPlan<'_>) -> Result<GeneratedKernel, LowerError>;
    /// Whether the backend can lower `dtype` **on `target`**. The JIT trust
    /// boundary checks this *before* [`Backend::lower`] so an unlowerable dtype
    /// is a typed decline, not a lowering panic. (AOT op authoring is trusted,
    /// so `lower` itself may still panic on a dtype it can't spell.)
    ///
    /// # Why `target` is here
    ///
    /// This used to be `fn supports_dtype(&self, dtype) -> bool` — no target —
    /// so a backend could only answer **always** or **never**. For a dtype whose
    /// availability is target-conditional the sole *sound* unconditional answer
    /// is "never", because claiming it would emit a type the target cannot
    /// compile, which the "decline, do not fall through" rule forbids.
    ///
    /// That cost real coverage. Slang's own conformance docs say only
    /// `int`/`uint` (32-bit) are universally supported and *"the others depend on
    /// target + capabilities"* — so Slang **can** spell `i8`/`i16`/`u8`/`u16` on a
    /// capable target, and this crate was refusing them everywhere for want of a
    /// parameter.
    ///
    /// # Two blockers, and only one of them was ours
    ///
    /// Adding the parameter removes the **API** blocker. It does not by itself
    /// close the Slang gap, and saying otherwise would be the kind of claim this
    /// codebase measures rather than asserts: a backend still needs to know what
    /// a given `target` is *capable of*, and that vocabulary belongs to the
    /// namespace's maintainer (KISS-CLASSIFY §6.8-0004), not to us. Reading
    /// `vulkan:` capability sets by transcribing what we think they mean is
    /// exactly the coupling KISS #171 (a machine-readable per-namespace
    /// capability manifest) exists to remove.
    ///
    /// So: the mechanism is here now; a backend that has capability data may
    /// condition on it today, and the remaining declines are honest ones about
    /// missing *data* rather than a missing *parameter*.
    fn supports_dtype(&self, dtype: ElementKind, target: TargetId) -> bool;
    /// The count-unit width the backend's emitter actually builds for `plan`:
    /// `1` means the kernel's `n` argument counts **elements** (a scalar
    /// lowering); `w > 1` means it counts `w`-element **vectors** (a vectorized
    /// or packed lowering, `n = elements / w`). Feeds the FKC contract's
    /// `count_unit:` line. Default `1` (scalar / element-counted); a vectorizing
    /// backend overrides it to mirror its emitter's dispatch exactly. CPU-C and
    /// Slang emit scalar kernels today, so they inherit the default.
    fn effective_count_width(&self, _plan: &crate::plan::KernelPlan<'_>) -> u32 {
        1
    }
    /// Alternative schedule variants for this plan's cell, beyond the default
    /// [`Backend::lower`] kernel. Default: none. Every returned variant must
    /// pass the same validation gate as the default (nvrtc/nvcc compile +
    /// numeric oracle + sanitizer where the schedule warrants) before it is
    /// shipped or ranked by the bench gate.
    ///
    /// # Errors
    ///
    /// Returns [`LowerError`] on the same terms as [`Backend::lower`]. Note that
    /// a backend with no variants returns `Ok(vec![])` — an empty variant set is
    /// a valid answer, not a refusal.
    fn lower_variants(
        &self,
        _plan: &crate::plan::KernelPlan<'_>,
    ) -> Result<Vec<Variant>, LowerError> {
        Ok(Vec::new())
    }
}

/// Backend-injected lowering closures for the **non-universal** parts of the
/// math. Parenthesization is universal and inlined directly; everything that
/// renders a target-language surface is a seam:
///
/// - `leaf` — how input operand `i`'s value is named (`in0[i]` scalar, `v0.x`
///   for a vector lane);
/// - `unary` — spells a [`UnaryOp`] over an already-lowered inner string
///   (`expf(...)` is CUDA-specific);
/// - `binary` — spells a non-infix [`BinaryOp`] over two operand strings
///   (`fmaxf(a, b)`, `powf(a, b)`);
/// - `arith` — spells the four **infix** arithmetic nodes.
///
/// # Infix is a seam because it is not universal
///
/// This doc used to say infix `+ - * /` was "universal across
/// CUDA/Slang/HLSL/Metal/GLSL and inlined directly". That is **false**, and the
/// counterexample is in this crate's own dtype set: for `c64`/`c128` the C
/// carrier is a struct, and a C compiler rejects operators on structs (MSVC
/// `C2088`; it rejects `_Complex` outright with `C2440`). `a + b` does not
/// render. That is why `arith` exists, and the claim is corrected here rather
/// than deleted because the sentence outlived the code that disproved it.
///
/// The universality of an operator is therefore a property of the operator
/// **and the dtype**, not of the operator alone — a distinction worth keeping in
/// view for anyone auditing which spellings may be driver-side.
///
/// `#[non_exhaustive]`: the seam set grows as non-C-family backends land (SPIR-V
/// needs no textual spelling for several of these, and per-storage-class access
/// is a known gap). Out-of-crate backends build one through [`Lowering::builder`],
/// so a new seam with a sensible default is a minor version rather than a break
/// of every emitter at once.
#[non_exhaustive]
pub struct Lowering<'a> {
    /// Operand-access spelling.
    pub leaf: &'a dyn Fn(u8) -> Result<Spelling, LowerError>,
    /// Per-row reduced-scalar spelling ([`ScalarExpr::Reduced`]). Only the
    /// `RowReduce` emitter produces a body containing a `Reduced` leaf; every other
    /// emitter passes a closure that panics (its bodies never contain one).
    pub reduced: &'a dyn Fn(u8) -> Result<Spelling, LowerError>,
    /// Output-coordinate spelling ([`ScalarExpr::Coord`], increment 0d): the
    /// per-axis coordinate of the output element, cast to the compute dtype
    /// (`(float)c{d}` in the CUDA strided emitter). Only the strided
    /// elementwise emitter materializes coordinates; every other emitter
    /// passes a panicking closure — the plan gate routes Coord bodies to
    /// `Schedule::Strided`, and those closures are the per-emitter backstop.
    pub coord: &'a dyn Fn(u8) -> Result<Spelling, LowerError>,
    /// Unary-op spelling.
    pub unary: &'a dyn Fn(UnaryOp, String) -> Result<Spelling, LowerError>,
    /// Binary-function-op spelling.
    pub binary: &'a dyn Fn(BinaryOp, String, String) -> Result<Spelling, LowerError>,
    /// Spelling for the four **infix arithmetic** nodes.
    ///
    /// Defaults to the C operator (`(a + b)`), which is right for every scalar
    /// dtype. A backend overrides it where the compute type is not something a C
    /// operator applies to — a complex struct being the case that forced the seam
    /// to exist.
    ///
    /// # Both walkers must honour it
    ///
    /// [`lower_expr`] and `lower_node` lower the same four nodes, and every
    /// real emitter goes through `lower_node` (via [`lower_dag`]) — `lower_expr`
    /// is the simple path. Wiring this seam into `lower_expr` alone left the
    /// production path spelling `(a * b)` on a struct, which C rejects. Anything
    /// added here must be added to both, and the differential end-to-end tests
    /// are what notice when it is not: a unit test that greps the emitted source
    /// for a helper NAME matches the helper's own definition and passes while the
    /// body still says `*`.
    pub arith: &'a dyn Fn(ArithOp, String, String) -> Result<Spelling, LowerError>,
    /// Ternary select spelling ([`ScalarExpr::Select`]) over the three
    /// already-lowered operand strings `(cond, a, b)`. Its own seam — the
    /// 2-operand `binary` closure cannot carry three operands, and the select
    /// spelling has its own bitwise contract (the arms must move raw bits, so
    /// it must NEVER route through a promote-demote wrapper; see
    /// `cuda::cuda_select`). Emitters whose bodies can never contain a Select
    /// (the packed f16/bf16 pair path — `body_packs` excludes it) pass a
    /// panicking closure, the `coord` precedent.
    pub select: &'a dyn Fn(String, String, String) -> Result<Spelling, LowerError>,
    /// Constant-literal spelling ([`ScalarExpr::Const`]).
    ///
    /// A seam rather than a fixed call to [`const_lit`] because the correct
    /// spelling is **dtype-dependent and not universal**. [`const_lit`] renders an
    /// `f64` as a C double literal; that is wrong for an integer dtype (`3.0` in
    /// an `int` context), wrong for a backend whose literal syntax is not C's, and
    /// carries `NAN`/`INFINITY` macros that only exist in C.
    ///
    /// The default is [`const_lit`], which is what every in-tree emitter uses
    /// today, so opening this seam changed no emitted byte. The dtype-aware
    /// spelling is a deliberate follow-up that WILL move goldens — see the
    /// `const_lit` note about the optimizer's bit-preservation proofs, which are
    /// stated against the current double-promoted semantics and must be
    /// re-verified before the spelling changes.
    ///
    /// Backends capture the dtype the same way `unary`/`binary` do.
    pub constant: &'a dyn Fn(f64) -> Result<Spelling, LowerError>,
}

/// Defaults for the seams [`LoweringBuilder`] does not require.
///
/// The three panicking ones mirror what every in-tree emitter already passes by
/// hand: a body containing one of these leaves has been routed to an emitter that
/// cannot spell it, which is a plan-gate bug, not a user error. Defaulting them
/// means a backend that never sees such a body writes nothing, while one that
/// does still fails loudly instead of emitting something plausible.
mod default_seam {
    use super::{ArithOp, Decline, DeclinedOp, LowerError, Spelling};

    /// These three USED TO PANIC. They now return a typed decline, which is the
    /// KISS-EMIT §6.8-0004 fix: an emitter must not panic on any input, and a
    /// mis-routed body IS input.
    ///
    /// The reasoning that justified the panic is preserved and still true — a body
    /// containing one of these leaves has been routed to an emitter that cannot
    /// spell it, which is a plan-gate bug rather than a user error. **But "this is
    /// a bug" and "abort the process" are different claims.** A typed decline is
    /// strictly more informative than a panic: a caller can still treat it as
    /// fatal, and one that would rather report it can now do so. The panic took
    /// that choice away from every caller in order to make a point to one.
    pub(super) static REDUCED: fn(u8) -> Result<Spelling, LowerError> = |i| {
        Ok(Spelling::Declined(Decline::UnsupportedOp {
            op: DeclinedOp::Reduced(i),
            why: format!(
                "a Reduced({i}) leaf reached an emitter with no `reduced` seam. Only a                  row-reduction emitter produces such a body; either supply                  `.reduced(..)` or route this body to a reduction schedule."
            ),
        }))
    };
    pub(super) static COORD: fn(u8) -> Result<Spelling, LowerError> = |d| {
        Ok(Spelling::Declined(Decline::UnsupportedOp {
            op: DeclinedOp::Coord(d),
            why: format!(
                "a Coord({d}) leaf reached an emitter with no `coord` seam. Coord bodies                  lower via a strided schedule only (a linear-index loop has no per-axis                  coordinates); either supply `.coord(..)` or route this body to                  Schedule::Strided."
            ),
        }))
    };
    pub(super) static SELECT: fn(String, String, String) -> Result<Spelling, LowerError> =
        |_c, _a, _b| {
            Ok(Spelling::Declined(Decline::UnsupportedOp {
                op: DeclinedOp::Select,
                why: "a Select reached an emitter with no `select` seam. Select has its own                       bitwise contract (its arms must move raw bits and must never route                       through a promote-demote wrapper), which is why it is not expressible                       through the 2-operand `binary` seam. Supply `.select(..)`."
                    .to_string(),
            }))
        };

    /// The C infix operator — right for every dtype whose compute type is a C
    /// scalar, which is all of them except the complex structs. Unlike the three
    /// above this is a working spelling rather than a refusal, because "spell it
    /// infix" is the correct answer almost everywhere and a backend should only
    /// have to say so when it is not.
    pub(super) static ARITH: fn(ArithOp, String, String) -> Result<Spelling, LowerError> =
        |op, a, b| Ok(Spelling::Spelled(format!("({a} {} {b})", op.c_operator())));

    pub(super) static CONSTANT: fn(f64) -> Result<Spelling, LowerError> =
        |v| Ok(Spelling::Spelled(super::const_lit(v)));
}

/// Builds a [`Lowering`] — the only way an out-of-crate backend constructs one,
/// since [`Lowering`] is `#[non_exhaustive]`.
///
/// `leaf`, `unary` and `binary` are required because every backend has a real
/// answer for them. The rest default: `reduced`/`coord`/`select` to a panic naming
/// the missing seam (a body needing them has been mis-routed), and `constant` to
/// [`const_lit`].
pub struct LoweringBuilder<'a> {
    inner: Lowering<'a>,
}

impl<'a> Lowering<'a> {
    /// Start building a `Lowering` from the three seams every backend must answer.
    #[must_use]
    pub fn builder(
        leaf: &'a dyn Fn(u8) -> Result<Spelling, LowerError>,
        unary: &'a dyn Fn(UnaryOp, String) -> Result<Spelling, LowerError>,
        binary: &'a dyn Fn(BinaryOp, String, String) -> Result<Spelling, LowerError>,
    ) -> LoweringBuilder<'a> {
        LoweringBuilder {
            inner: Lowering {
                leaf,
                reduced: &default_seam::REDUCED,
                coord: &default_seam::COORD,
                unary,
                binary,
                arith: &default_seam::ARITH,
                select: &default_seam::SELECT,
                constant: &default_seam::CONSTANT,
            },
        }
    }
}

impl<'a> LoweringBuilder<'a> {
    /// Per-row reduced-scalar spelling ([`ScalarExpr::Reduced`]).
    #[must_use]
    pub fn reduced(mut self, f: &'a dyn Fn(u8) -> Result<Spelling, LowerError>) -> Self {
        self.inner.reduced = f;
        self
    }
    /// Output-coordinate spelling ([`ScalarExpr::Coord`]).
    #[must_use]
    pub fn coord(mut self, f: &'a dyn Fn(u8) -> Result<Spelling, LowerError>) -> Self {
        self.inner.coord = f;
        self
    }
    /// Ternary select spelling ([`ScalarExpr::Select`]).
    #[must_use]
    pub fn select(
        mut self,
        f: &'a dyn Fn(String, String, String) -> Result<Spelling, LowerError>,
    ) -> Self {
        self.inner.select = f;
        self
    }
    /// Infix-arithmetic spelling ([`Lowering::arith`]). Defaults to the C
    /// operator, which is right for every dtype whose compute type is a C
    /// scalar.
    ///
    /// # This setter was missing, and that made the seam unreachable
    ///
    /// `Lowering` gained `arith` so a backend could spell arithmetic on a
    /// compute type no C operator applies to — a complex struct, where `a * b`
    /// is `error C2088`. The field landed; this method did not. Since `Lowering`
    /// is `#[non_exhaustive]`, the builder is *the only way* an out-of-crate
    /// backend constructs one, so for that whole window the newest seam existed
    /// and could not be reached from outside this crate.
    ///
    /// Nothing caught it because both in-tree emitters lived in this crate and
    /// wrote the struct literal directly. It surfaced the moment they moved out —
    /// as a compile error, immediately, which is the argument for the move.
    #[must_use]
    pub fn arith(
        mut self,
        f: &'a dyn Fn(ArithOp, String, String) -> Result<Spelling, LowerError>,
    ) -> Self {
        self.inner.arith = f;
        self
    }
    /// Constant-literal spelling ([`ScalarExpr::Const`]). Defaults to [`const_lit`].
    #[must_use]
    pub fn constant(mut self, f: &'a dyn Fn(f64) -> Result<Spelling, LowerError>) -> Self {
        self.inner.constant = f;
        self
    }
    /// Finish.
    #[must_use]
    pub fn build(self) -> Lowering<'a> {
        self.inner
    }
}

impl std::fmt::Debug for Lowering<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Lowering").finish_non_exhaustive()
    }
}

impl std::fmt::Debug for LoweringBuilder<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LoweringBuilder").finish_non_exhaustive()
    }
}

/// Spell an `f64` constant as a valid C literal. `{v:?}` emits `inf`/`NaN`, which
/// aren't valid C literals; map the non-finite cases to the standard macros.
///
/// # The f-suffix question is now a codegen concern, not a correctness one
///
/// This used to warn that the `f32` `f`-suffix vs double-promotion question was
/// "**not** purely a perf concern", because the optimizer's bit-preservation
/// contract was proven against double-promoted semantics. That reasoning has
/// been overtaken, in the right direction, by a fix one layer up.
///
/// [`crate::optimize::optimize`] now rounds every constant to the kernel's
/// **compute precision** — at ingest and after every fold (see
/// `optimize::Compute`, and the 1-ULP divergence it closes). So the value
/// reaching this function from an `f32` kernel is already `f32`-representable,
/// and `(float)(double)v == v` exactly for such a value. Emitting it as a plain
/// decimal that the C compiler converts to `float` is therefore lossless, with
/// or without the suffix.
///
/// What that leaves is a genuine codegen question — whether the emitted
/// arithmetic happens at `float` or gets double-promoted by C's usual
/// conversions — which affects instruction selection and speed, not the bits of
/// the constant itself.
///
/// The `--use_fast_math` warning still stands unchanged: that changes the
/// *arithmetic*, not the literal, and it invalidates the rule-set proofs
/// regardless of how constants are spelled.
#[must_use]
pub fn const_lit(v: f64) -> String {
    if v.is_nan() {
        "NAN".to_string()
    } else if v.is_infinite() {
        if v > 0.0 {
            "INFINITY".to_string()
        } else {
            "-INFINITY".to_string()
        }
    } else {
        format!("{v:?}")
    }
}

/// Unwrap a child's [`Spelling`], propagating a **decline** as this node's decline
/// and a **failure** via `?`.
///
/// This is the whole propagation rule in one place, and it exists so no walker arm
/// can get it subtly wrong. A declined child makes the parent declined -- it does
/// NOT become an error -- which is the distinction the type was introduced to
/// preserve: *"this backend does not spell that"* is an answer, not a fault.
macro_rules! spelled_or_return_decline {
    ($e:expr) => {
        match $e? {
            Spelling::Spelled(s) => s,
            declined @ Spelling::Declined(_) => return Ok(declined),
        }
    };
}

/// Lower a [`ScalarExpr`] tree to a backend expression string via `lo`'s seams.
///
/// Structural: a subtree reachable by two paths is re-rendered once per path. For
/// shared-interior dedup (emit a value once as a `tmp`), lower an [`ExprDag`] via
/// [`lower_dag`] instead — this remains the inlining primitive both paths share
/// (single-use nodes lower identically through either).
///
/// # Errors
///
/// Propagates a [`LowerError`] from a seam that genuinely failed. A seam that
/// **declined** is not an error: it surfaces as `Ok(Spelling::Declined(..))`, so a
/// caller must match rather than `?` past a capability statement.
pub fn lower_expr(e: &ScalarExpr, lo: &Lowering<'_>) -> Result<Spelling, LowerError> {
    macro_rules! sp {
        ($x:expr) => {
            spelled_or_return_decline!(lower_expr($x, lo))
        };
    }
    Ok(match e {
        ScalarExpr::Input(i) => return (lo.leaf)(*i),
        ScalarExpr::Reduced(i) => return (lo.reduced)(*i),
        ScalarExpr::Coord(d) => return (lo.coord)(*d),
        ScalarExpr::Param(i) => Spelling::Spelled(format!("p{i}")),
        ScalarExpr::Const(v) => return (lo.constant)(*v),
        ScalarExpr::Unary(op, x) => {
            let x = sp!(x);
            return (lo.unary)(*op, x);
        }
        ScalarExpr::Binary(op, a, b) => {
            let (a, b) = (sp!(a), sp!(b));
            return (lo.binary)(*op, a, b);
        }
        ScalarExpr::Select(c, a, b) => {
            let (c, a, b) = (sp!(c), sp!(a), sp!(b));
            return (lo.select)(c, a, b);
        }
        ScalarExpr::Add(a, b) => {
            let (a, b) = (sp!(a), sp!(b));
            return (lo.arith)(ArithOp::Add, a, b);
        }
        ScalarExpr::Sub(a, b) => {
            let (a, b) = (sp!(a), sp!(b));
            return (lo.arith)(ArithOp::Sub, a, b);
        }
        ScalarExpr::Mul(a, b) => {
            let (a, b) = (sp!(a), sp!(b));
            return (lo.arith)(ArithOp::Mul, a, b);
        }
        ScalarExpr::Div(a, b) => {
            let (a, b) = (sp!(a), sp!(b));
            return (lo.arith)(ArithOp::Div, a, b);
        }
    })
}

/// Lower an [`ExprDag`] to `(prelude, root_ref)`.
///
/// `prelude` is the block of `<ctype> tmpN = <expr>;` statements — one per shared
/// non-leaf node, in topological order (a `tmp`'s RHS references only earlier
/// `tmp`s / inlined leaves) — that the caller emits before the use site.
/// `root_ref` names the DAG's output value.
///
/// A node with `consumers <= 1`, or any leaf, is **inlined** at its use site;
/// only a shared *interior* (`consumers > 1`, non-leaf) is hoisted. So for a body
/// with no shared interior the prelude is empty and `root_ref` is byte-identical
/// to [`lower_expr`] — the DAG is transparent for every single-use body, which is
/// the no-regression guarantee for existing goldens.
///
/// # Errors
///
/// Propagates a [`LowerError`] from a seam that genuinely failed. A seam that
/// **declined** is not an error — it surfaces as `Ok(Spelling::Declined(..))`, so
/// a caller must match rather than `?` its way past a capability statement.
pub fn lower_dag(
    dag: &ExprDag,
    ctype: &str,
    lo: &Lowering<'_>,
) -> Result<(Vec<String>, String), LowerError> {
    let mut refs: Vec<Option<String>> = vec![None; dag.len()];
    let mut prelude: Vec<String> = Vec::new();
    let policy = HoistPolicy {
        hoist_all: false,
        hoist_shared_leaves: false,
        extra_uses: &[],
    };
    // A decline becomes a LowerError HERE and not at the seam. That is the whole
    // boundary: at a seam, "I do not spell that" is an answer and must not be
    // `?`-able into a failure. At the top of a lowering it is terminal -- the body
    // cannot be emitted -- and LowerError's Unsupported* variants ARE the emitter's
    // typed-decline vocabulary (KISS-EMIT 6.8-0002). So the conversion happens once,
    // in one place, rather than each backend inventing it.
    let root = lower_node(dag, dag.root(), ctype, lo, &mut refs, &mut prelude, &policy)?
        .spelled()
        .map_err(LowerError::from)?;
    Ok((prelude, root))
}

/// Hoisting policy for [`lower_node`] — how aggressively a value is bound to a
/// named `tmp` vs inlined at its use site.
struct HoistPolicy<'u> {
    /// Hoist EVERY non-leaf (the packed pair-split path — see [`lower_dag_all`]).
    hoist_all: bool,
    /// Also hoist a **shared `Input` leaf** (a memory load referenced by more
    /// than one use) — the multi-output path, so the shared `dy` load appears
    /// once. Single-output paths keep leaves inlined (a leaf ref is free).
    hoist_shared_leaves: bool,
    /// Per-node "extra use" count beyond the intra-DAG `consumers` edges — the
    /// multi-output root multiplicity (a node that is the output root of `k`
    /// bodies has `k` extra uses). Empty ⇒ all zero (single-output). Combined
    /// with `consumers` this is the node's total use count, which decides
    /// sharing.
    extra_uses: &'u [u32],
}

impl HoistPolicy<'_> {
    /// Total uses of `id` = intra-DAG consumer edges + multi-output root uses.
    fn total_uses(&self, dag: &ExprDag, id: NodeId) -> u32 {
        dag.consumers(id) + self.extra_uses.get(id as usize).copied().unwrap_or(0)
    }
}

/// Lower a **multi-root** DAG (one root per output body, from
/// [`ExprDag::from_exprs`]) to `(prelude, root_refs)` — the cross-body-CSE core
/// of the multi-output emitter. All roots are lowered against ONE shared `refs`
/// memo and ONE shared `prelude`, so a subexpression shared between outputs is
/// emitted once and referenced by each store. Beyond the intra-body sharing
/// [`lower_dag`] already hoists, this additionally hoists:
///
/// - a value used by more than one output body (a shared interior, or a node
///   that is one body's root and another body's interior), via the
///   root-multiplicity `extra_uses`;
/// - a **shared `Input` leaf** — the shared `dy` load — so it appears in the
///   source exactly once (the "strictly fewer global loads" win).
///
/// `hoist_all` mirrors [`lower_dag_all`] for the packed pair-split path (every
/// non-leaf a `tmp`). `root_refs[j]` names output body `j`'s value; a node that
/// is the sole use of its value inlines exactly as the single-output path does.
pub fn lower_dag_multi(
    dag: &ExprDag,
    ctype: &str,
    lo: &Lowering<'_>,
    hoist_all: bool,
) -> Result<(Vec<String>, Vec<String>), LowerError> {
    // Root multiplicity: how many output bodies name each node as their root.
    // A node that is a body root AND referenced by another node (or the root of
    // two bodies) has total_uses > 1, so it hoists once rather than re-emitting.
    let mut extra_uses = vec![0u32; dag.len()];
    for &r in dag.roots() {
        extra_uses[r as usize] += 1;
    }
    let policy = HoistPolicy {
        hoist_all,
        hoist_shared_leaves: true,
        extra_uses: &extra_uses,
    };
    let mut refs: Vec<Option<String>> = vec![None; dag.len()];
    let mut prelude: Vec<String> = Vec::new();
    // One at a time rather than `collect()`: a decline is a SUCCESS value, so
    // collecting would hand back a Vec containing one and a caller could emit it.
    // The first decline aborts the whole multi-body lowering, which is the only
    // correct answer -- a kernel with one unspellable output is not a kernel with
    // n-1 outputs.
    let mut root_refs = Vec::with_capacity(dag.roots().len());
    for &r in dag.roots() {
        let one = lower_node(dag, r, ctype, lo, &mut refs, &mut prelude, &policy)?
            .spelled()
            .map_err(LowerError::from)?;
        root_refs.push(one);
    }
    Ok((prelude, root_refs))
}

/// [`lower_dag`], but hoisting **every** non-leaf node to a named `tmp` (not
/// just shared ones). For lowerings whose op spellings reference an operand
/// string more than once (e.g. the packed f16/bf16 pair-scalarization, which
/// splits one operand into `__low2half(x)` / `__high2half(x)`), inlining would
/// duplicate whole subexpression *text* per reference — exponential in depth.
/// Hoist-all makes every operand a `tmp` name, so a duplicate is a name, never
/// an expression, and the emitted source stays linear. Values are unchanged.
///
/// # Errors
///
/// As [`lower_dag`]: a genuine seam failure propagates; a decline surfaces as
/// `Ok(Spelling::Declined(..))`.
pub fn lower_dag_all(
    dag: &ExprDag,
    ctype: &str,
    lo: &Lowering<'_>,
) -> Result<(Vec<String>, String), LowerError> {
    let mut refs: Vec<Option<String>> = vec![None; dag.len()];
    let mut prelude: Vec<String> = Vec::new();
    let policy = HoistPolicy {
        hoist_all: true,
        hoist_shared_leaves: false,
        extra_uses: &[],
    };
    // A decline becomes a LowerError HERE and not at the seam. That is the whole
    // boundary: at a seam, "I do not spell that" is an answer and must not be
    // `?`-able into a failure. At the top of a lowering it is terminal -- the body
    // cannot be emitted -- and LowerError's Unsupported* variants ARE the emitter's
    // typed-decline vocabulary (KISS-EMIT 6.8-0002). So the conversion happens once,
    // in one place, rather than each backend inventing it.
    let root = lower_node(dag, dag.root(), ctype, lo, &mut refs, &mut prelude, &policy)?
        .spelled()
        .map_err(LowerError::from)?;
    Ok((prelude, root))
}

/// Post-order, memoized lowering of one DAG node. Emits a shared interior once
/// (into `prelude`) and returns the string every use site references (a `tmpN`
/// name for a hoisted node, the inlined expression otherwise).
fn lower_node(
    dag: &ExprDag,
    id: NodeId,
    ctype: &str,
    lo: &Lowering<'_>,
    refs: &mut Vec<Option<String>>,
    prelude: &mut Vec<String>,
    policy: &HoistPolicy<'_>,
) -> Result<Spelling, LowerError> {
    if let Some(r) = &refs[id as usize] {
        return Ok(Spelling::Spelled(r.clone()));
    }
    // Copy the node out (all fields are `Copy`) so the immutable borrow of `dag`
    // is released before the `&mut refs`/`&mut prelude` recursion.
    let node = dag.node(id).clone();
    macro_rules! child {
        ($n:expr) => {
            spelled_or_return_decline!(lower_node(dag, $n, ctype, lo, refs, prelude, policy))
        };
    }
    let rhs = match node {
        DagNode::Input(i) => spelled_or_return_decline!((lo.leaf)(i)),
        DagNode::Reduced(i) => spelled_or_return_decline!((lo.reduced)(i)),
        DagNode::Coord(d) => spelled_or_return_decline!((lo.coord)(d)),
        DagNode::Param(i) => format!("p{i}"),
        DagNode::Const(v) => spelled_or_return_decline!((lo.constant)(v)),
        DagNode::Unary(op, x) => {
            let x = child!(x);
            spelled_or_return_decline!((lo.unary)(op, x))
        }
        DagNode::Binary(op, a, b) => {
            let (a, b) = (child!(a), child!(b));
            spelled_or_return_decline!((lo.binary)(op, a, b))
        }
        DagNode::Select(c, a, b) => {
            let (c, a, b) = (child!(c), child!(a), child!(b));
            spelled_or_return_decline!((lo.select)(c, a, b))
        }
        DagNode::Add(a, b) => {
            let (a, b) = (child!(a), child!(b));
            spelled_or_return_decline!((lo.arith)(ArithOp::Add, a, b))
        }
        DagNode::Sub(a, b) => {
            let (a, b) = (child!(a), child!(b));
            spelled_or_return_decline!((lo.arith)(ArithOp::Sub, a, b))
        }
        DagNode::Mul(a, b) => {
            let (a, b) = (child!(a), child!(b));
            spelled_or_return_decline!((lo.arith)(ArithOp::Mul, a, b))
        }
        DagNode::Div(a, b) => {
            let (a, b) = (child!(a), child!(b));
            spelled_or_return_decline!((lo.arith)(ArithOp::Div, a, b))
        }
    };
    // Hoisting decision:
    // - a non-leaf hoists when `hoist_all` (packed pair-split), or when it is
    //   shared — total uses > 1 (intra-body consumer edges + multi-output root
    //   multiplicity). A single-use root has total_uses <= 1, so it inlines
    //   into its store, byte-identical to `lower_expr`.
    // - a leaf normally inlines (a leaf ref is free); under `hoist_shared_leaves`
    //   (multi-output) a shared `Input` LEAF — a memory load referenced by more
    //   than one output — hoists so the load appears once. Const/Param/Coord/
    //   Reduced leaves stay inlined (no load to dedup; multi-output bodies carry
    //   no Coord/Reduced anyway).
    let shared = policy.total_uses(dag, id) > 1;
    let hoist = if node.is_leaf() {
        policy.hoist_shared_leaves && matches!(node, DagNode::Input(_)) && shared
    } else {
        policy.hoist_all || shared
    };
    let r = if hoist {
        let name = format!("tmp{}", prelude.len());
        prelude.push(format!("{ctype} {name} = {rhs};"));
        name
    } else {
        rhs
    };
    refs[id as usize] = Some(r.clone());
    Ok(Spelling::Spelled(r))
}
