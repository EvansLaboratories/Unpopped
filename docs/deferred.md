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
| ~~**`cuda.md` SSOT appendix needs an `Sm90` row**~~ — **DONE**, verified 2026-08-20 against `origin/main`: the annex now lists `cuda:sm90` (Hopper) beside `sm80`/`sm89`/`sm90a`. | We wired `ArchSku::Sm90` in 0.2.0 to close five `cuda:sm90` byte-match vectors. `spec/namespaces/cuda.md`'s appendix says a row is added "when the emitter wires that arch", and lists only Sm80/Sm89/Sm90a — so the SSOT and its own named reference implementation (`unpopped-vocab`) are out of sync now. Row sent to Baracuda; the annex is maintainer-owned so we do not edit it. **Nothing but a human reading the annex noticed** — which is the live evidence for the manifest proposal in section C. **And nothing but a human re-reading this row noticed it had been fixed**, which is the same defect one level up: the flag had no closing condition anyone would trip over. |
| ~~**Publish `unpopped-cpu-c` / `unpopped-slang`**~~ — **DONE**; both have been on crates.io since **0.1.0, 2026-08-19**, and are at 0.3.0. Verified against the registry index 2026-08-20, not inferred from tags. | **The `Backend` trait settling** — concretely, the two gaps the crate README still names as open: no structured binding/ABI manifest, and an artifact type that is source text rather than an arbitrary word stream (**the editor's position, not an external requirement.** Recorded since `be8a04e`, 2026-08-06 pre-extraction, as "the Vulkane review's #1/#2" — but **all three plausible parties have been checked and none raised them**: Vulkane and Fuel consume nothing from this lineage, and Baracuda, who does consume it and does have the stake, is explicit they did not ask. Likeliest reading: there was never an external requester and our own design judgment was recorded as someone else's. The engineering stands on its merits; what changed is that it is **arguable again**. **The trigger is a condition, not a party**: these close when a backend emitting a *word stream* rather than source text needs them — and since that is now known to be our own judgment, **this trigger is ours to move rather than something we are waiting on.**). **Ruled 2026-08-15: hold.** The recruiting argument for publishing is real — KISS-EMIT §8.2-0002's freeze gate needs an emitter whose surface spellings differ from the reference's, and a dependable published emitter is what makes an outsider's life easier. But it inverts once you ask what they would be implementing *against*: a third party who adopts `unpopped-slang` and then eats two breaking `Backend` changes in a month concludes the umbrella is not ready, and is right. **Unreachable is a better first impression than unstable.** Note the trigger is an artifact (those two gaps closed), not an event ("when someone asks") — a demand-shaped trigger would fire exactly when the churn is most expensive.<br><br>**The hold was lifted four days later and this row did not know for a day.** What lifted it is recorded — in the `unpopped-cpu-c-v0.1.0` tag, in the right words: *"It publishes now because a CONCRETE consumer needs it: Baracuda's three-way cross-backend agreement test (CpuC x Slang x Cuda) cannot be assembled without it."*<br><br>**And the trigger this row named is not the one that fired.** It said these close "when a backend emitting a *word stream* rather than source text needs them". No such backend appeared. A different, unanticipated consumer did. So the artifact-shaped trigger rule (section B) is necessary and not sufficient: an artifact-shaped trigger is checkable, but it still only fires for the future you imagined. |
| **`unpopped-cuda` sub-crate**: CUDA emitter donated by Baracuda, plus the `convert.rs` CUDA parser moving out of neutral core | ~~The 0.2 trait freeze.~~ **That trigger passed unmet and must be restated.** 0.2 shipped 2026-08-15 and `Backend`/`Lowering` has taken two breaking changes since (0.4.0 `Result<Spelling, LowerError>`, 0.5.0 the `DeclinedOp` split), so "the 0.2 trait freeze" now names an event that happened without the condition being true — the failure mode section B's rule exists to prevent, in a row written before that rule. Restate against an artifact when the next freeze is designed. The crate has **two** inbound streams — IR→`.cu` (Baracuda's donation) and `.cu`→IR (our parser) — so it must not be designed emit-only. |
| ~~**Make the target namespace pluggable**~~ — **DONE**; the `cuda:` token eviction is what remains | Pluggable landed: `StructureKey.arch: ArchSku` → `target: TargetId`, an interned §6.8 token validated for **grammar only**. The CUDA-only `arch_code`/`arch_from_code` pair was **deleted**, not adapted — measured, the codec has *zero* non-test `ArchSku` code, and Vulkane's v3→v4 four-to-five-field bump cost **zero changes** (byte-match 19/20-with-an-exclusion → **20/20, zero exclusions**), which is what proves it opaque rather than opaque-looking. **Remaining is four sites and two decisions, no codec work:** the enum (`layout.rs`), the re-export (`lib.rs`), `KernelSku.arch` (`sku.rs`), and `From<ArchSku> for TargetId` (`target.rs`). Decided 2026-08-15 with Baracuda: **drop the reserved block** and convert via `TargetId::parse`, fully evicting CUDA from neutral core; `KernelSku.arch` is a *separate* call pending Fuel, since it turns on who constructs it. **Trigger: the registry repoint.** §6.8-0003 names `unpopped-vocab` as `cuda`'s `reference_implementation`, so evicting before the pointer moves to `baracuda-cuda-vocab` breaks a PR-gated file. Landing crate is `baracuda-cuda-vocab` (Eric). |

**Publication is no longer held; every crate here has shipped.** The first cut
was 2026-08-15 — `unpopped-vocab` 0.3.0 and `unpopped` 0.2.0, release commit
`8a242e8`.

**No live version number is recorded in this file.** A paragraph naming the
published and in-tree versions was here and went stale twice while nothing
forced it to move; that is the same defect as every other stale row in this
register, in the one field where the answer is a `curl` away. The manifests are
in-tree truth and `https://index.crates.io/un/po/<crate>` is registry truth;
`git tag -l` lists what was cut and its annotation says why.

**The vocab went to 0.3.0, not 0.2.0, and the reason is worth keeping.** 0.2.0
was already published, and the in-tree code had moved materially past it while
the manifest still said `0.2.0`. The workspace dep was
`{ path = ..., version = "0.2.0" }` — and **cargo strips `path` on publish and
keeps `version`**, so a published `unpopped` would have resolved against the OLD
registry vocab and failed to compile for every downloader, unrecoverably, since
registry versions are immutable. Caught by a `cargo package` dry-run, which
resolves the verification build against the registry exactly as a downloader
does. **That leg is now permanent, between byte-match and publish**, and it makes
dependency-first publish order a precondition the tooling enforces rather than a
convention someone remembers: once the dep says `0.3.0`, `cargo package -p
unpopped` cannot pass until vocab 0.3.0 is actually on the registry.

The rule that produced the original hold still stands for the next schema event:
per sk4 §6 (Eric-ratified) that clause binds the *token-deriving* crate, so its
breaking changes ride the coordinated cut — and the ordering within a cut is
**push → byte-match → cargo package → publish**, because a registry version is
immutable and cannot be un-published if verification then finds a divergence.
The generator is explicitly **not** gated on a schema cut and ships on its own
schedule.

**Corollary now enforced by habit: bump on landing, not on shipping.** Leaving a
manifest at a published version while `main` holds breaking changes past it
recreates exactly the state above. `unpopped` went to 0.3.0 the moment the
`JitRequest` break landed, not at its future publish — and to 0.6.0 the moment
the plan gate began declining, likewise before any publish.

That habit is why the in-tree version is a *fact about the tree* rather than a
plan, and it is the reason this file no longer needs to carry one.

---

## B. Gated on a coordinated golden regen

These rewrite emitted text. Doing any of them quietly breaks byte-identity
goldens *including Baracuda's physical CUDA corpus*, so they ride a regen event
rather than a cleanup commit.

- **The f16/bf16 spelling seam.** `cfamily::scalar_ctype` spells `__half` /
  `__nv_bfloat16` and `cast_scalar` emits `__half2float`-class intrinsics from
  the *neutral* module. Tripwired in `crates/unpopped/tests/neutral_spelling.rs`,
  which fails in both directions.

  **That tripwire lived in `unpopped-cpu-c/tests/` until 2026-08-20, and it
  imports nothing from that crate** — only `unpopped::cfamily` and
  `unpopped-vocab`. So the guard on `unpopped`'s own neutrality did not run when
  you tested `unpopped`: `cargo test -p unpopped` went 7 suites / 433 tests →
  **8 / 438** on moving it, measured before and after. A contributor iterating
  with `-p unpopped`, or anyone depending on `unpopped` alone, got a green with
  these five never executed.

  Four separate doc references already gave the path as `unpopped`'s. **The
  prose was right and the file was in the wrong place** — the inverse of the
  three stale-claim defects found the same day, and a reminder that a
  disagreement between a doc and the tree does not tell you which one moved.
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
- ~~**What `VariantFidelity::BitIdentical` means across backends**~~ — **ANSWERED
  2026-08-15: bit-identical to the default lowering of the same cell, in the same
  backend, at the same version.** The two horns were the right pair and the left
  one is the answer: local-to-this-backend is the only referent that is both real
  and non-normative, and "says nothing cross-backend" is a **feature** — the
  moment the referent becomes a normative reference, accumulation order becomes
  normative, which contradicts the incidental list. (OPEN-2.)

  **The axis is ours, and KISS has no home for it — measured, not inferred.**
  `git grep -iE "bit.identical|variant fidelity"` across `spec/` at KISS
  `efe111c` returns seven hits and **every one is cross-language prose** (Slang
  `tanh` vs CUDA `tanh`: `emit.md:252/:302/:816/:1143`, `consume.md:311/:831`,
  `synth.md:306`). Not one is about two schedule variants of one cell in one
  language. So there are two different axes:

  - KISS-EMIT §6.6-0002 fidelity = determinism class + MathPrecision — the kernel
    against **its own semantics**;
  - `VariantFidelity` = a variant against **the default lowering of the same
    cell**.

  **§6.6-0005 therefore does not reach this field, and it MUST NOT be moved into
  the contract's Guarantees section.** That clause forbids declaring the kernel's
  fidelity (determinism class or MathPrecision) off-schema; this is neither.
  Filing a variant-relative claim in a section whose other contents are absolute
  would make the next reader take `BitIdentical` as a claim against the semantics
  rather than against a sibling — strictly worse than leaving it where it is.
  Confirmed with the KISS architect, who is filing the missing home as a
  KISS-Emit category-(c) gap.

  **Status of the KISS side is a filing, not a clause** — per this file's own
  trigger rule, do not treat it as ruled until it lands as a clause ID. What is
  settled and needs no clause is the local half: the referent is the same-cell
  default lowering, and the field stays on `Variant`.
- **Whether NaN propagation (rule N2) is universally required** — on Vulkan it
  may be unachievable *by any lowering*:
  `shaderSignedZeroInfNanPreserveFloat32` is a `returnedonly` **property**, not a
  feature you enable. Either gate on the capability and declare non-advertising
  devices out of conformance, or weaken the rule. (OPEN-4.)
- ~~**A standard machine-readable form for §6.8-0004 namespace vocabularies**~~ —
  **RULED AND MERGED; this entry was stale.** See the re-anchoring note at the
  end of this item. Originally
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

  **RESOLVED — and this entry pointed at the question instead of the answer.**
  The ruling is merged as **`KISS-CLASSIFY-6.8-0008` … `-0013`** (verified at KISS
  `efe111c`): schema `kiss-namespace-vocabulary-v1`, the required-field envelope,
  a typed decline on a missing field or unrecognized `schema`, `kind` open exactly
  as ruled, and mapped conformance tests. KISS pins the envelope; the maintainer
  fills it — as proposed.

  **How this sat here wrong, because the mechanism matters more than the entry.**
  The trigger was an **issue number** (#171). An issue is a *question* identifier:
  it stays open until a human closes it, and closing it is nobody's build step. A
  clause is an *answer* identifier and cannot drift out from under you. So this
  read "needs a ruling" long after the ruling shipped.

  It is worse than a stale note, and the sharp part is the bit that should have
  caught it: **this crate already consumes those clauses.** `unpopped-vocab`'s
  `the_namespace_vocabulary_versions_are_asserted` is named after §6.8-0009's own
  `test_namespace_vocabulary_version_is_asserted`. The implementation had moved on
  and the register had not, with nothing connecting them — a green suite on one
  side and a stale claim on the other, which is *precisely* the 22-vs-24 dtype
  failure this very entry cites as its motivating example.

  **Rule for this file going forward:** an unblocking trigger names the artifact
  that will exist when it is met — a clause ID, a published version, a file — not
  the issue where it was asked. Mechanism owed to kiss-ref, who found it in a
  well-built guard of their own and passed it on via the portfolio PM.

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

  ~~**Blocks the open target model** below: whether we import a table, a grammar,
  or both changes what `ArchSku`'s replacement has to hold.~~ **It did not.** The
  open target model landed first and independently: `TargetId` validates the §6.8
  **grammar only** and holds no vocabulary, so what a manifest might carry never
  constrained `ArchSku`'s replacement. Measured, not argued — Vulkane's v3→v4
  four-to-five-field vocabulary bump cost **zero** changes here.

  Kept rather than deleted because the prediction was reasonable and wrong in a
  useful direction: **the coupling dissolved because the replacement declined to
  hold vocabulary at all.** A dependency between two pieces of work is a claim
  about a design that has not been made yet, and it expires when the design does.
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
- ~~**Quant: adopt the scale-sibling-operand model.**~~ — **DONE**; `QuantFacts`
  removed, the sibling model measured. The residual sentence that used to sit
  here — *"blk32 vs blk128 both contribute one same-rank scale sibling and still
  collide"* — was **wrong about the mechanism and is corrected in
  `tests/scale_sibling_model.rs`.** It is bucket saturation, not rank:
  divisibility saturates at `d16` and vector width at `v8`, so any two block
  counts ≥ 16 collide while smaller ones genuinely differ. The stated reason
  predicts collision at every granularity; the real one predicts it only above
  saturation, and those imply different sk5 fixes.

  **This sentence was mine, not KISS's** — I described it to the KISS architect
  as sk4's own text and they filed it into #210 as a fact about the
  specification. `git grep`/`git log -S` across all KISS refs finds it nowhere.
  Two attribution failures in one hop: I mis-sourced my own note, they relayed
  without measuring. Corrected on both sides; recorded here because the sentence
  lived in a file and the correction lived in a conversation, which is how it
  survived the first fix (`structure_key.rs` was corrected the same day and this
  copy was missed).
- ~~**Constructors for `ContractionKey`/`OperandKey`/`QuantFacts`, then
  `#[non_exhaustive]`**~~ — **DONE in 0.2.0** (`a0c8207`), plus `AccMp`.
  Constructors and the reservation landed in one commit, which is the ordering
  the earlier backed-out attempt got wrong: reserved-first breaks cross-crate
  struct literals with no constructor to fall back on.
  ~~**`SymExtent` was in that list and I missed it — it is still unreserved, and
  0.2.0 has shipped.**~~ — **DONE**, and it rode exactly the cut this entry
  predicted: `734915d` (*adopt sk4's scale-sibling model*) reserved it and added
  `SymExtent::new`, because that rework touched the same two vestigial
  `OperandDesc` fields (`quant`, `symbolic`).

  **The plan worked and this entry did not know it for two releases.** Left as
  written it said "still unreserved" and "should ride the next cut" about a type
  that had already ridden one — a worklist entry describing work that was done,
  which is worse than a stale note because it is an instruction to redo it.
  Found 2026-08-20 by scanning every backticked identifier in this file against
  the sources; the *identifier* check could not catch it (`SymExtent` still
  exists), only reading the claim could.
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
