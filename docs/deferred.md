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
| **`cuda.md` SSOT appendix needs an `Sm90` row** — ours to flag, Baracuda's to edit | We wired `ArchSku::Sm90` in 0.2.0 to close five `cuda:sm90` byte-match vectors. `spec/namespaces/cuda.md`'s appendix says a row is added "when the emitter wires that arch", and lists only Sm80/Sm89/Sm90a — so the SSOT and its own named reference implementation (`unpopped-vocab`) are out of sync now. Row sent to Baracuda; the annex is maintainer-owned so we do not edit it. **Nothing but a human reading the annex noticed** — which is the live evidence for the manifest proposal in section C. |
| **`unpopped-cuda` sub-crate**: CUDA emitter donated by Baracuda, plus the `convert.rs` CUDA parser moving out of neutral core | The 0.2 trait freeze. The crate has **two** inbound streams — IR→`.cu` (Baracuda's donation) and `.cu`→IR (our parser) — so it must not be designed emit-only. |
| ~~**Make the target namespace pluggable**~~ — **DONE**; the `cuda:` token eviction is what remains | Pluggable landed: `StructureKey.arch: ArchSku` → `target: TargetId`, an interned §6.8 token validated for **grammar only**. The CUDA-only `arch_code`/`arch_from_code` pair was **deleted**, not adapted — measured, the codec has *zero* non-test `ArchSku` code, and Vulkane's v3→v4 four-to-five-field bump cost **zero changes** (byte-match 19/20-with-an-exclusion → **20/20, zero exclusions**), which is what proves it opaque rather than opaque-looking. **Remaining is four sites and two decisions, no codec work:** the enum (`layout.rs`), the re-export (`lib.rs`), `KernelSku.arch` (`sku.rs`), and `From<ArchSku> for TargetId` (`target.rs`). Decided 2026-08-15 with Baracuda: **drop the reserved block** and convert via `TargetId::parse`, fully evicting CUDA from neutral core; `KernelSku.arch` is a *separate* call pending Fuel, since it turns on who constructs it. **Trigger: the registry repoint.** §6.8-0003 names `unpopped-vocab` as `cuda`'s `reference_implementation`, so evicting before the pointer moves to `baracuda-cuda-vocab` breaks a PR-gated file. Landing crate is `baracuda-cuda-vocab` (Eric). |

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

---

## C. Needs a ruling — not ours to decide alone

- **Whether `required_fidelity` belongs to us at all.** The function now exists
  (`oracle::required_fidelity`, derived from unit roundoff + rounding steps +
  `contract::ulp_bound`, so validation and the declared `max_ulp` cannot drift).
  What is NOT settled is ownership: the original note said "plausibly
  KISS-Conform's rather than ours", and **Fuel already checks kernel contracts at
  runtime**. Asked them 2026-08-14 whether this duplicates machinery they hold —
  if so, ours belongs behind theirs rather than beside it. Answer pending.
  (`conformance.md` OPEN-1.)
- **What `VariantFidelity::BitIdentical` means across backends** — bit-identical
  to *what*? Local-to-this-backend is real but says nothing cross-backend; to a
  normative reference makes accumulation order normative, contradicting the
  incidental list. (OPEN-2.)
- **Whether NaN propagation (rule N2) is universally required** — on Vulkan it
  may be unachievable *by any lowering*:
  `shaderSignedZeroInfNanPreserveFloat32` is a `returnedonly` **property**, not a
  feature you enable. Either gate on the capability and declare non-advertising
  devices out of conformance, or weaken the rule. (OPEN-4.)
- **A standard machine-readable form for §6.8-0004 namespace vocabularies** —
  proposed to the KISS architect with Vulkane, Baracuda and Fuel holding it
  (2026-08-13). The two registered namespaces publish in incompatible shapes:
  `cuda`'s annex ends with a TSV `Appendix: machine-readable capability set (SSOT
  seed)`; `vulkan`'s has no machine-readable form at all, its four-field grammar
  living in prose. A consumer supporting both transcribes one and hand-parses the
  other — **the same structure as the §6.1 dtype problem that hid a 22-vs-24 set
  mismatch here for weeks with a green suite on both sides.**

  The design crux is that the vocabularies differ **in kind**, not merely in
  format: `cuda`'s is a closed enumeration, `vulkan`'s is generated over an open
  product space and can only be validated, never listed. Strawman is therefore
  discriminated (`kind: enumerated` with members, or `kind: generated` with a
  field spec), on the claim that a consumer only ever asks *can I recognize a
  well-formed token?* and *can I enumerate what exists, if anything?*

  Legitimate for KISS to pin despite §6.8-0004 delegating vocabulary *content*,
  because KISS already owns this axis's meta-level: §6.8-0001 grammar, -0002
  byte-exact matching, -0005 charset, -0006 fixed-width juxtaposition, -0007 the
  digest form. An annex format is the same shape — KISS pins the envelope, the
  maintainer fills it.

  **Filed as KISS #171**, architect ruling: KISS's venue, strawman adopted, and
  `kind` MUST be an **open** set — an unrecognized `kind` typed-declines rather
  than guessing, because a union whose cases came from exactly two examples is a
  closed list built from n=2. **Cannot be Accepted without Baracuda and Vulkane
  cosigning**, since a convention imposed on a maintainer's annex by a consumer
  is not a convention.

  Three refinements from the round, each better than what was proposed:

  - **Vulkane — the strawman was missing a question, and it is the load-bearing
    one.** Not just *recognize* and *enumerate* but **"can I produce the
    canonical token?"** §6.8-0002 is byte-exact with no implication logic, so a
    schema answering only recognition lets a consumer believe it is
    interoperable while emitting non-matching tokens: green on both sides, cache
    never hits, no error anywhere. So `generated` must carry **canonicalization**
    — sort orders, dedup, digest algorithm, threshold, and *which string* is
    hashed. Their blocker: `<coop>`'s third form is chosen by a **length-
    conditional switch** (>512 bytes → FNV-1a digest), which no alphabet or
    regex decides — §6.8-0007's own digest mechanism is the thing a grammar-only
    schema cannot express. And the manifest must be **generated from the owner's
    crate and byte-compared**, never authored: half of `kiss-vulkan-vocab` is
    structural code, so a hand-written file would transcribe the hard half and
    reproduce the defect it removes.
  - **Fuel — `vocabulary_version` must be ASSERTED, not merely present** ("a
    field a consumer reads is a field; one the consumer asserts is a gate"), and
    **injectivity is mandatory only where the output is an IDENTITY**, optional
    where it is a classification — a blanket rule would force implementations to
    invent distinctions their hardware lacks. Also the line the RFC should be
    drawn on: **byte-exact matching covers the token; it does not cover
    producing one. A consumer that only compares tokens can be opaque; one that
    also emits them cannot.**
  - **Both, independently: `generated` entries need worked examples plus
    negatives.** Vulkane because interop failures live in canonicalization, not
    recognition; Fuel because a consumer will treat `generated` as
    validate-only and never test it — "two implementations that agree by never
    disagreeing."

  **Explicit non-goal** (Vulkane): do not bind `vulkan:` component types to
  `dtype_manifest.json`. Different vocabularies, different axes, one derives no
  `structure_key` at all — importing would manufacture exactly the false
  divergence the 25-variants-against-24-tokens case warns about. **The format
  composes namespaces; it does not unify them.**

  **Blocks the open target model** below: whether we import a table, a grammar,
  or both changes what `ArchSku`'s replacement has to hold.
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

- **The catalog / server mode — RETIRED, not deferred.** It had no consumer.
  Measured 2026-08-14: [`catalog.md`](catalog.md) never named a requesting party
  ("a consumer", abstractly, throughout), `unpopped` has exactly one code
  consumer (Baracuda, 6 crates; Fuel zero, Lightbulb zero at any depth), and
  Baracuda holds both halves already — routing as the `DispatchTable` they
  populate *from this repo's own `unpopped-vocab::dispatch` types*, inventory as
  their emitter's cell enumeration, with Fuel doing runtime selection on top.
  The dispatch **types** stay in `unpopped-vocab` as shared vocabulary; the
  populated registry was never core's to hold. See the retirement banner in
  `catalog.md` for the full reasoning — it is kept because several of its
  findings about the *generator* survive the catalog being dropped.


- **Dtype lowering coverage.** The numbers are no longer here: they are
  **measured** by `crates/unpopped/tests/dtype_lowering_coverage.rs`, which
  carries the per-dtype × per-backend table and fails when it goes stale. The
  prose figures this entry used to give were **wrong twice** — first "CpuC 9,
  Slang 5" against a measured 8, then "CpuC 8/22" against a measured 18 once the
  work below landed. Twice in the same entry, in both directions, is the whole
  argument for moving them: a coverage claim decays silently, because nothing
  about adding a dtype arm forces the sentence describing it to change. **Run the
  test.** Any number written here is a number that will be wrong.

  **The list below is done.** `bool`, `u32`, `u64`, both FP8s, `i4`/`u4`/`b1` and
  `c64`/`c128` all lower and are differentially tested through a real C compiler.
  It is kept rather than deleted because each entry records *why the dtype was
  hard*, and those reasons outlived the work — the promotion rule below is still
  the reason `u8`/`u16` are correct today, and someone will need it again.

  `bool` was **not simply a missing arm** — the logical ops already narrow to
  `U8`, which *is* the bespoke Bool surface, so the question was whether a
  `Bool`-keyed cell routes to that same `uint8_t` path or whether
  `ElementKind::Bool` is deliberately not a plan dtype. A naming question wearing
  a coverage question's clothes. (Resolved: it routes there, and arithmetic on a
  truth value is refused.)
  **`u32` and `u64` together** shared one blocker, sharper than "unsigned-wrap
  audit" suggests: C's integer promotions lift `unsigned char`/`unsigned short`
  to **signed** `int`, so `u8`/`u16` genuinely compute at 32-bit signed width —
  which is what the oracle's `op_width` (32) and sign-extending `wrap_bits`
  model, and why those two were already correct. `unsigned int` has the same rank
  as `int` and does **not** promote, so `u32` arithmetic is unsigned modulo
  2³² and the old model read `3_000_000_000u32` as negative. Both needed an
  unsigned width/wrap path in the oracle and the emitter. (`u32`'s index/address
  role is additive and was previously — wrongly — recorded as the reason it
  cannot compute.)
  `i4`/`u4`/`b1` needed sub-byte pack/unpack — and the store is a
  **read-modify-write**, safe only because `cpu_c`'s loop is serial; a threaded
  backend copying it races. Both FP8s needed a software codec, with an
  *independent* one in the oracle or it stops being a differential.
  `c64`/`c128` needed a struct ABI and complex arithmetic in the IR — see the
  MSVC finding under "worth knowing".

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

**MSVC does not implement C99 `_Complex`.** Measured, not assumed: `float
_Complex x = 1;` is `error C2440: cannot convert from 'int' to '_Fcomplex'`.
Microsoft's `<complex.h>` ships opaque `_Fcomplex`/`_Dcomplex` structs
constructed with `_FCbuild` and multiplied with `_FCmulcc` — a real API, but an
MSVC-specific one that would need a `#if defined(_MSC_VER)` fork against the
Clang/GCC spelling in every emitted kernel.

So `c64`/`c128` lower as a struct this crate defines and emits
(`cfamily::complex_helpers`), with arithmetic by call rather than by operator.
No conditional compilation, no vendor's name in the output, identical on every C
compiler. The same answer FP8 and the sub-byte dtypes reached, for the same
reason.

Two things follow that are easy to miss:

- **Clang-on-Windows is not the fix.** It would compile `_Complex` happily,
  which is precisely why it is the wrong test compiler: the portability
  constraint belongs to the *emitted* C, not to our harness, and picking a more
  permissive compiler hides the constraint rather than satisfying it. (nvcc and
  Clang interoperate poorly anyway, so it was moot.) A genuinely useful variant
  would be testing against *every* compiler found rather than the first —
  unimplemented.
- **The portable struct is a fallback, not the answer.** A backend whose target
  has native complex would spell it its own way — the CUDA peer reports it would
  use `cuFloatComplex` + `cuCaddf`/`cuCmulf` from `cuComplex.h`, not this
  struct. That makes complex structurally identical to f16/bf16: a dtype whose
  neutral spelling must be overridable per backend. Complex only *looked*
  settled because its portable default already works. See `fp8_helpers` for what
  that override has to carry — including arity, since a packed-pair override
  reshapes the emit loop rather than renaming anything in it.
