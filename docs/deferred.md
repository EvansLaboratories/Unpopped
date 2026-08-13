# Deferred work

Everything knowingly left undone, and — more importantly — **why**, and **what
unblocks it**.

This file exists because the reasoning is the perishable part. A deferral whose
justification is lost becomes indistinguishable from an oversight, and gets
either re-litigated or silently done wrong. Several entries below are deferred
*because doing them naively is worse than not doing them*, and that is not
recoverable from the code.

Grouped by what unblocks each, since that is the actionable axis.

> **Keeping this honest.** `docs/conformance.md`'s scope list rotted once — it
> claimed the oracle could not evaluate contractions long after it could, and a
> reader would have skipped writing a test that already worked. An over-cautious
> register is not harmless. When an item lands, delete it here in the same
> commit.

---

## A. Blocked on an external event

Nothing to do but be ready. Each names its trigger.

| Item | Trigger |
|---|---|
| ~~**sk4 regen** of `unpopped-vocab`~~ — **DONE and PUBLISHED as 0.2.0** (`4cefb6a`, tag `unpopped-vocab-v0.2.0`, checksum `4c66b4a9f48b888f…`) | Closed. Verification preceded publication throughout: the byte-match leg ran and passed against merged KISS `a43a96f` (spec `19c3ad7`) **before** the version existed. That ordering — push → byte-match → publish — cost one round-trip and bought a version number that is permanently correct; publishing first would have made a divergence unrecoverable, since crates.io versions are immutable. `unpopped` (the generator) stays at 0.1.0 and is explicitly NOT gated on this cut. |
| ~~**Final byte-match leg**~~ — **DONE** (`6bc8bea`), committed as a permanent test | Ran against merged KISS `a43a96f` (spec `19c3ad7`): **19/19 claimed positives byte-exact + 1 earned capability exclusion (vulkan namespace), 10/10 declines with exact verdict and payload.** Skips are earned by substituting an implemented target and requiring a byte-exact round-trip, so an exclusion cannot hide a divergence. **Known bound, recorded in code as `this_leg_is_not_dtype_coverage`:** the vectors exercise 3 dtypes in the dtype position and 5 anywhere, against 22 usable — a dtype-spelling divergence on the other 17 is invisible here and is caught by `kiss_dtype_manifest.rs` instead. The two tests are complementary; neither is sufficient alone. |
| ~~**Vendor the KISS dtype manifest**~~ — **DONE** (`4d72bcb`) | Unblocked by KISS #131: `conformance/corpus/dtype_manifest.json` carries `structure_key_schema_version` / `token_prefix`. Vendored verbatim at `19c3ad7` under `crates/unpopped-vocab/kiss/`, with equality asserted **both ways** plus a schema-version assertion, so a copy taken from a newer schema fails loudly instead of quietly widening the set. |
| **`unpopped-cuda` sub-crate**: CUDA emitter donated by Baracuda, plus the `convert.rs` CUDA parser moving out of neutral core | The 0.2 trait freeze. The crate has **two** inbound streams — IR→`.cu` (Baracuda's donation) and `.cu`→IR (our parser) — so it must not be designed emit-only. |
| **Make the target namespace pluggable**; move the `cuda:` vocabulary out of `unpopped-vocab` | Same carve. `ArchSku` stays Baracuda-owned content sourced from KISS's SSOT, generated-and-committed, never `build.rs`. |

**Publication is no longer held.** `unpopped-vocab` shipped 0.2.0 on the sk4 cut.
The rule that produced the hold still stands for the next schema event: per sk4
§6 (Eric-ratified) that clause binds the *token-deriving* crate, so its breaking
changes ride the coordinated cut — and the ordering within a cut is **push →
byte-match → publish**, because a registry version is immutable and cannot be
un-published if the match then finds a divergence. The generator (`unpopped`) is
explicitly **not** gated and ships on its own schedule; it remains at 0.1.0 with
its own breaking batch pending.

---

## B. Gated on a coordinated golden regen

These rewrite emitted text. Doing any of them quietly breaks byte-identity
goldens *including Baracuda's physical CUDA corpus*, so they ride a regen event
rather than a cleanup commit.

- **The f16/bf16 spelling seam.** `cfamily::scalar_ctype` spells `__half` /
  `__nv_bfloat16` and `cast_scalar` emits `__half2float`-class intrinsics from
  the *neutral* module. Tripwired in `tests/neutral_spelling.rs`, which fails in
  both directions.
  **Trap for the implementer, found by running the seam as a mutation:** making
  `scalar_ctype`/`half_load_intrinsic` return `None` does **not** make
  `cast_scalar` decline — it drops into the arithmetic arm and emits `(float)v`
  on a `__half`, trading a visible vendor leak for a silent numerical bug. The
  neutral core must *refuse*, never fall through.
- **The temp-binding pass.** `Sqr`/`Relu` reference their operand twice,
  `Gelu`/`Silu`/`Sign` three times, `Max`/`Min` four.
  **It is only safe because of the op set, not because hoisting is safe.** All of
  those are float-only (the plan gate rejects every `UnaryOp`, the float binary
  fns and `Cmp*` at integer dtypes), and a float temp round-trips exactly. At
  sub-`int` widths hoisting *changes results* — C promotes `char`/`short` to
  `int`, so an inlined compound operand is un-truncated while a hoisted one is
  truncated by its temp: `(in0+in1)>>in2` at u8 is `150` inlined, `22` hoisted.
  Anyone extending these ops to 8/16-bit must settle truncation **first**.
- **`const_lit` dtype-correctness**, welded to threading dtype into the
  optimizer. These must land together: `exact_pow2_recip` reasons in f64
  normality and never sees the kernel dtype, so the `x / 2^k → x * 2^-k` rule is
  sound *only because* `const_lit` emits a bare double and C promotes. Measured:
  of 2045 constants the f64 predicate accepts, **44 are wrong under f32**; an
  f32-aware guard admits 253 with 0 wrong. Pinned by
  `optimize::pow2_rule_soundness_is_coupled_to_double_promotion`.

---

## C. Needs a ruling — not ours to decide alone

- **`required_fidelity(plan)`** — nothing derives which fidelity a cell is
  *entitled* to; `Fidelity` is picked by whoever calls `compare`, so the standard
  can only say "passed at a tolerance someone chose". Note for whoever takes it:
  **existing usage cannot seed the table** — there are six construction sites and
  five are the oracle testing its own comparator, so it must be derived from the
  numerics. Plausibly KISS-Conform's rather than ours. (`conformance.md` OPEN-1.)
- **What `VariantFidelity::BitIdentical` means across backends** — bit-identical
  to *what*? Local-to-this-backend is real but says nothing cross-backend; to a
  normative reference makes accumulation order normative, contradicting the
  incidental list. (OPEN-2.)
- **Whether NaN propagation (rule N2) is universally required** — on Vulkan it
  may be unachievable *by any lowering*:
  `shaderSignedZeroInfNanPreserveFloat32` is a `returnedonly` **property**, not a
  feature you enable. Either gate on the capability and declare non-advertising
  devices out of conformance, or weaken the rule. (OPEN-4.)
- **The KISS cost model** (#125) — vector-authoritative vs optional sibling. Our
  position is recorded: we already emit a two-axis vector with provenance, and a
  generator can only ever say `declared`.
- ~~**Whether a reader must decline a non-canonical `x<hh>` reduce field**~~ —
  **RULED, and implemented** (`4d72bcb`). I read §6.6-0009/§6.7-0005 as silent on
  the reader and shipped accept-and-normalize. It is not silent: the `x<hh>`
  value is *domain-restricted* to sets that are neither all-axes nor the lone
  trailing axis, so such a spelling is outside all four values and §6.7-0005's
  "reject any other field-8 spelling" applies. Worth keeping as a record of the
  reasoning: **the lenient reading was self-defeating on its own terms** — two
  accepted spellings for one set make `from_token` → `to_token` byte-unstable,
  and a key with two spellings for one meaning is not an identity. When a clause
  looks silent, check whether an adjacent value's *domain* already answers it.
  (KISS #160.)

---

## D. Ordinary engineering — unblocked, just not done

- **The catalog / server mode** — designed in [`catalog.md`](catalog.md), not
  built. Phase 1 is the library (op registry, emitter registry, `resolve`);
  phase 2 is a server over it, deliberately sequenced after the PROVISIONAL wire
  formats are co-pinned. The design names one gap that must be closed *in* the
  build rather than after it: a catalog entry is baked against caller-supplied op
  logic, which no current validity field names (§7).

- **Dtype lowering coverage.** The numbers are no longer here: they are
  **measured** by `crates/unpopped/tests/dtype_lowering_coverage.rs`, which
  carries the per-dtype × per-backend table and fails when it goes stale. The
  prose figures this entry used to give ("CpuC 9, Slang 5") were **wrong** — the
  measurement is **CpuC 8/22, Slang 4/22** — which is the whole argument for
  moving them: a coverage claim decays silently, because nothing about adding a
  dtype arm forces the sentence describing it to change.

  Remaining work, roughly by cost. `bool` is listed first but **is not simply a
  missing arm** — read `plan.rs`'s admissibility doc first: the logical ops
  already narrow to `U8`, which *is* the bespoke Bool surface, so the open
  question is whether a `Bool`-keyed cell should route to that same `uint8_t`
  path or whether `ElementKind::Bool` is deliberately not a plan dtype. Decide
  that before writing code; it is a naming question wearing a coverage question's
  clothes.
  Then: `u64` (needs an unsigned-wrap audit — `wrap_bits` is two's-complement
  *signed* and f64 cannot represent every u64, which is why it is named but
  deliberately not lowered) · `i4`/`u4`/`b1` (sub-byte pack/unpack) · both FP8s
  (software codec — and the oracle needs an *independent* one or it stops being a
  differential) · `c64`/`c128` (struct ABI + complex arithmetic in the IR).

  **Slang's `i8`/`i16`/`u8`/`u16` are blocked on a missing mechanism, not on
  Slang.** Slang's conformance docs say *"Only `int`/`int32_t` and
  `uint`/`uint32_t` are universally supported; the others depend on target +
  capabilities"* — which means Slang **can** spell them on a capable target. The
  gap is ours: `Backend::supports_dtype(&self, dtype) -> bool` has no target
  parameter, so a backend can only answer "always" or "never", and for a
  conditionally-available type the sole *sound* unconditional answer is "never"
  (claiming it would emit `int8_t` for a target that cannot compile it — the
  fall-through the backend contract forbids).

  The capability data already exists: `KernelPlan.key` carries the
  `StructureKey`, and KISS §6.8 target tokens encode capabilities directly —
  `vulkan:sg64.ops-abr.arith-f16.cm-none` names an `arith-f16` capability. Only
  the admissibility gate cannot see it.

  **This is the same root cause as the one vulkan vector excluded from the
  byte-match**, where `ArchSku` — a closed CUDA-only enum — cannot represent a
  `vulkan:` target at all. Capability-aware dtype admission and the pluggable
  target namespace are one piece of work, not two, and doing them together turns
  19/19-plus-an-exclusion into 20/20 *and* unlocks these four dtypes.

  The reserved `fnuz` pair must **never** be lowered at this schema version, and
  that is asserted separately from the table: "forbidden" and "not done yet" are
  different facts and should not share a column of `false`s.
- **Oracle coverage**: `RowSort` (NaN-greatest `key_lt`, stable ties, TopK), and
  gather/scatter — the latter needs a *different notion of correct*, since under
  nondeterministic FP `atomicAdd` only an order-independent invariant is
  checkable at all. Pinned by the exhaustive `Coverage` classifier in
  `oracle.rs`'s tests.
- **Quant: adopt the scale-sibling-operand model.** `QuantFacts` is the wrong
  *shape*, not the right shape un-keyed — removing it is breaking, so it rides
  the sk4 cut. Residual it does **not** close: blk32 vs blk128 both contribute
  one same-rank scale sibling and still collide. That part is a genuine sk5 item.
- ~~**Constructors for `ContractionKey`/`OperandKey`/`QuantFacts`, then
  `#[non_exhaustive]`**~~ — **DONE in 0.2.0** (`a0c8207`), plus `AccMp`.
  Constructors and the reservation landed in one commit, which is the ordering
  the earlier backed-out attempt got wrong: reserved-first breaks cross-crate
  struct literals with no constructor to fall back on.
  **`SymExtent` was in that list and I missed it — it is still unreserved, and
  0.2.0 has shipped.** The cost is exactly what the item exists to prevent:
  reserving it now is a breaking change. It should ride the **next** cut rather
  than justify a 0.3.0 of its own, and the natural home is the quant/scale-sibling
  rework below, which touches the same two vestigial `OperandDesc` fields
  (`quant`, `symbolic`). Cheap when it happens — one construction site, in this
  crate's own tests. Recorded here because the whole argument for doing this
  before a cut was that the window closes when the *next* break is designed, and
  this is now waiting on one.
- **Per-target N2 verification.** NaN-propagation surviving the toolchain is a
  property of *one optimizer*, re-measured per target, never inherited. Verified
  on portable C (`/O2`) and CUDA (nvrtc→PTX→driver JIT, RTX 4070). Any new target
  needs its own; a source-text golden is structurally blind to it.

---

## Not deferred, just worth knowing

The PR-gating box **has GPU hardware**. This crate's tests are CPU-only because
it holds no device backend, not because the hardware is absent — so the device
legs become runnable in-repo the moment `unpopped-cuda` lands.
