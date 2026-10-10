# Changelog

**Why this file exists, and it is not bookkeeping.**

Measured 2026-09-06 by diffing each published `.crate` tarball against the tree —
a practice relayed by baracuda as *"keep the last release's consumer code and run
it against the new artifact; it answers what did this fix break, which no gate
written for the fix can ask."*

⚠️ **A public-API diff of `unpopped` 0.9.0 vs the tree reported PURELY ADDITIVE:
292 items → 293, nothing removed. Underneath were TWO behaviour changes in
`unpopped-vocab`, one of them NUMERIC — and both sat at a version number
IDENTICAL to the published one.** A signature diff cannot see a same-signature
behaviour change, and a version check cannot see a tree that never bumped.

---

## Unreleased — `0.15.0` (breaking)

### f16/bf16 take FP8's shape: a portable carrier and an emitted codec

The last vendor spelling in the neutral `cfamily` module is gone. Ships only
after `baracuda-cuda-emit` 0.14.4, whose local shadow (baracuda#154) keeps every
CUDA byte it emits unchanged.

- **`cfamily::scalar_ctype(F16 | Bf16)` returns `"unsigned short"`**, the
  storage carrier, instead of `"__half"` / `"__nv_bfloat16"`.
- **New `cfamily::half_helpers(ElementKind)`**: portable-C99 f16/bf16 codecs
  (`unpopped_f16_load`/`_store`, `unpopped_bf16_load`/`_store`), emitted with
  the kernel. Round to nearest, ties to even; NaN encodes as the canonical
  quiet NaN. Checked bit for bit through a real C compiler against the oracle's
  codec on all 65536 patterns and on 524288 products that need rounding; the
  oracle's codec is itself checked against the `half` crate.
- **`narrow_load_fn` / `narrow_store_fn` name the codec for the halves**, so
  `promote_load_f32`, `demote_store_f32` and `cast_scalar` emit
  `unpopped_f16_load(x)` where they emitted `__half2float(x)`.
- **Removed: `cfamily::half_load_intrinsic` and `half_store_intrinsic`.**
- **`store_expr_of` encodes a uniform half store once, at the store**, as for
  FP8: a half body root is now a plain f32 expression.
- **`unpopped-cpu-c` computes f16 and bf16.**
- **Fix: a narrow-float predicate stored a wrong mask.** The U8 store of a
  `Cmp*` body over FP8 decoded the f32 `0.0`/`1.0` root as if it were a storage
  pattern, so every true lane stored `0`. Live in `unpopped-cpu-c` since FP8
  support; `store_expr_of` now converts from what the root is.

---

## Unreleased — `0.14.4`

### `HalfArith`: the first plan decision that differs by sm (additive)

U1 of the joint P40 (sm_61) / RTX 4070 (sm_89) plan
(`docs/joint-gpu-milestone-plan.md`). No emitted byte or golden moves.

- **`TargetCapabilities` gains `fp16_results_per_clk_per_sm` and
  `fp32_results_per_clk_per_sm`** (`Option<u32>`), transcribed from NVIDIA's
  12.9.1 "Throughput of Native Arithmetic Instructions" table (page sha256
  `06499a0c…`). sm_61 does 2 fp16 results per clock per SM against 128 for fp32;
  sm_89 does 128 for both. The table has no column for 8.7, 10.3, 11.0 or 12.1,
  so those are `None`. `from_queried` leaves both `None`, and
  `with_arith_throughput` attaches sourced figures.
- **`KernelPlan::half_arith()` and `half_arith_for(TargetId)` return `HalfArith`
  (`Native` / `ViaF32`).** `ViaF32` is returned only when both rates are sourced
  and fp16 < fp32, which today means sm_61 only. Everything else is `Native`,
  the behaviour every emitter already has.
- It is a method, not a `KernelPlan` field, because `KernelPlan` is not
  `#[non_exhaustive]` and emitters build it by struct literal.
- Consumer: baracuda-cuda-emit gates its packed native `__h*2` path on
  `Native`. That path is bit-identical to the scalar float path by design, so on
  the P40 this changes speed, not numerics.

### Also in this change set (no API or emitted-byte change)

These merged after `0.14.3` with no version change of their own, and ship under
`0.14.4` (the PM's ruling, 2026-10-07: an unpublished number covers everything
merged under it). The change set is #37–#51, except #49.

- **Docs in the crate:** `telemetry::cuda_target_of` documents that its
  `sm{major}{minor}` format is not injective in principle (#37).
- **Tests:** `unpopped-vocab` pins what the frame broadcast mask says about
  collapsed outputs (`a_collapsed_output_is_absent_not_broadcast.rs`, #38).
  `unpopped` checks that every Qwen3 dense op plans and generates for
  `cuda:sm61` (`qwen3_dense_ops_plan_for_sm61.rs`, U2, #42).
  `unpopped-conformance`'s `golden_regen_checkpoint.rs` follows the
  `docs/deferred.md` split and the 2026-11-05 checkpoint (#43, #45).
- **Repository docs (not in any crate):** the joint P40 (sm_61) / RTX 4070
  (sm_89) milestone plan, `docs/joint-gpu-milestone-plan.md` (#39, kept current
  in #41, #46, #47, #48, #50, #51); `docs/deferred.md` Section B split, with the
  f16/bf16 shadow surface counted as eleven functions (#43, #45); and the
  qwen3_5 linear-attention assessment (#44).

## Released

⚠️ **What reached crates.io, read from its API on 2026-10-03:** `0.11.0`
(2026-09-11), `0.12.0` (2026-10-01), `0.13.0` (2026-10-01), `0.14.0` (2026-10-02), `0.14.1`
(2026-10-03), for all four published crates. `0.14.3` (2026-10-03) was confirmed
from the same API on 2026-10-07; `0.14.2` was never published. **`0.11.1`–`0.11.6` existed only in the workspace and were
never published.** A consumer went straight from `0.11.0` to `0.12.0`, so
everything those intermediate numbers carried is listed under `0.12.0`.


## 2026-10-03 — everything at `0.14.3`

### Fixed: telemetry from any compute capability now merges, and every drop is counted

**Behaviour change: H100 telemetry now counts.** `telemetry::merge_reports` used to
stamp a record through `arch_sku_of`, which has answers only for 8.x and 9.x, and
dropped every other capability before the merge saw it (sm_61, sm_70, sm_75,
Blackwell). Its 9.x answer was `cuda:sm90a`, but Fuel keys a Hopper record
`cuda:sm90`, so `merge`'s arch gate (stamp target == key target) rejected every
H100 report as well. Both drops were silent.

- **New `cuda_target_of(major, minor)`:** `cuda:sm{major}{minor}`, digits
  concatenated, no `a` suffix. This is baracuda's `cuda:` rule (it owns the
  namespace, §6.8-0004), and it is the same rule Fuel keys records with.
  `merge_reports` stamps with it.
- **Changed rows, on purpose:** 9.0 now stamps `cuda:sm90` (was `cuda:sm90a`), so
  H100 reports keyed `cuda:sm90` now merge into the `cuda:sm90` cells, where
  before they changed nothing. 8.6/8.7 now stamp `cuda:sm86`/`cuda:sm87` (was
  `cuda:sm80`); Fuel builds no key for those today, so no live row changes.
  6.x, 7.x, 10.x and 12.x now merge into their own targets. 8.0 and 8.9 are
  byte-identical.
- **New `merge_reports_counted`** returns a `MergeReport`: merged records per
  full target token, plus counts for no stamp, no capability, an unformable
  target, an unknown backend, a declined entry, and an arch-gate reject.
  `merge_reports` is the same fold with the report discarded.
- `arch_sku_of` is unchanged. It still answers "which cutlass dispatch cell".

## `0.14.2` (docs only, never published; its content shipped in `0.14.3`)

### Docs only: `TargetId` is the architecture identity; `ArchSku` is a cutlass dispatch SKU

No code, signature or behaviour change. This release records board #106 option C
(2026-10-03), which every affected lane approved with CireSnave's prior authorisation:

- **`TargetId` names every target, CUDA architectures included.** Any
  architecture is spelled as a token (`cuda:sm61`, `cuda:sm100`, …) whether or not
  `ArchSku` has a variant for it. The `unpopped-vocab::target` module docs and
  `TargetId`'s doc now say so.
- **`ArchSku` is re-scoped to a baracuda-cutlass dispatch SKU.** It stays closed
  and exhaustive, with no new variants and no freeze. A missing variant is not a
  missing architecture. Whether to add one (Blackwell `Sm100a`, say) is baracuda's
  call, made for its cutlass dispatch. Its doc also lost a pointer to a ROADMAP that
  this repo doesn't have.
- **`docs/idiom-lifting-design.md` §6:** the planned `ArchSku::Sm61` breaking
  release is marked superseded. No variant is added. The capability row shipped in `0.14.1`.

The `telemetry::merge_reports` drop this entry listed as a follow-up is fixed in `0.14.3`.

## 2026-10-03 — everything at `0.14.1`

### `cuda_capabilities` / `capabilities_for` cover sm_61 through sm_121 (additive)

These rows were missing, so they returned `None`. Each is now transcribed from
NVIDIA's own tables, parsed from the HTML rather than copied by hand. The
sources and the verification state are recorded on `cuda_capabilities`:

| New rows | Source | Toolkit accepts the arch | Verified on hardware |
|---|---|---|---|
| 6.1 | CUDA C++ Programming Guide **12.9.1** | CUDA 12.9 (NVRTC); not 13.x | no (awaits a P40) |
| 10.0, 10.3, 11.0, 12.0, 12.1 | CUDA Programming Guide (13.x edition), Sep 10, 2026 | CUDA 13.3 | no, compile-level only |

**Every existing row was re-checked against the same sources and is
unchanged.**

**Deliberately absent:**
- **10.7**: NVIDIA tabulates it, but CUDA 13.3 doesn't accept it.
- **8.8**: CUDA 13.3 accepts it, but neither table has it.
- Anything below 6.1.

**The two NVIDIA editions disagree on 12.x resident blocks per SM.** The CUDA
12.9.1 edition says 32 and the current edition says 24. The row uses **24**,
the later figure. A launcher must query the device regardless.

This is part of CireSnave's ruling that baracuda supports sm_61 through the
latest architecture via Unpopped's CUDA parse/emit. Nothing in Unpopped's IR is
arch-specific. These rows only inform which schedule variants are worth
offering.

## 2026-10-02 — everything at `0.14.0`

One breaking release that conforms `unpopped-vocab`'s structure-key derivation
to KISS-Classify as of **KISS#517 + KISS#519** (`KISS@bc16715` plus the #519
amendment). Tracking: #31.

⚠️ **Wire-visible. Tokens change for the inputs below.** If you cache on the
token, re-derive the affected entries rather than migrating them.

**Every same-rank cell with ordinary strides keys exactly as in 0.13.0.** The
KISS byte-match vectors and every pre-existing token test pass unchanged.

### Which tokens change

| Input | `0.13.0` | `0.14.0` | KISS-CLASSIFY |
|---|---|---|---|
| A lower-rank operand in a higher-rank cell, e.g. `[256]` with `[128,256]` | mask `00` | mask `01` (in frame coordinates) | §6.5-0014 (as amended by #519), §6.6-0008, §6.6-0013 |
| An own extent-1, stride-0 axis where the frame is wider, e.g. `[1,256]` strides `[0,1]` | mask `00` | mask `01` | §6.6-0008 |
| A trailing unit axis, e.g. `[4,1]` | `d4` (and possibly a packed width) | `da` / `v1` | §6.3-0011 |
| A zero-extent innermost axis, e.g. `[8,0]` | `d8` | `da` / `v1` | §6.3-0011, §6.5-0012 |
| A mixed-rank work class, e.g. `[2,64]` with `[64]` | `grid` (left-aligned frame: 4096) | `block` (right-aligned: 128) | §6.6-0013, §6.5-0010 |
| A token naming `f8e6m2` | parsed and keyed | **typed decline** (`ReservedDtype`) | §6.1-0013 |

**Unchanged on purpose:** a rank-deficient operand is still laid out from its
**own** axes. It is not `br` and not forced to `v1` merely for lacking a frame
axis. KISS#517's first text padded the layout too. That gave im2col's dense and
strided rank-3 outputs the same token, which Unpopped's im2col gate caught.
KISS#519 amended the clause.

### New API: `try_structure_key` / `try_structure_key_token`

§6.6-0021: a scale-type dtype (`f8e8m0`, `f8e6m2`) at operand 0 has no defined
primary dtype, and derivation MUST decline. The new functions return
`Result<_, DeriveDecline>` with `DeriveDecline::ScaleDtypeAtOperand0 { dtype }`.
On every other input they return exactly what the infallible pair does.

**`structure_key` / `structure_key_token` stay, undeprecated for now.** They key
a scale-first list with the scale's dtype, which is non-conformant for that one
input, and their docs say so. `#[deprecated]` arrives in the release that
migrates this workspace's own ~200 call sites.

A scale in any *other* operand slot (the sk4 sibling model) is unaffected.

### Consumers in this workspace

- **The im2col plan gate** now admits mask bit 0, the frame axis a rank-3 output
  cannot span, and still requires the output's own layout to be dense.
- **`contract::index_is_1d`:** a rank-deficient index tensor gathered along the
  **last** axis now reads as a 1-D (broadcast) index. Its frame mask has the
  leading bits set, where its empty own mask used to read as full-shape. This
  is the right reading, but it changes the emitted contract for that input. No
  in-tree test exercises a rank-deficient index.

### Vendored KISS artifacts

- Re-vendored from `KISS@bc16715`, verbatim; the files hash to KISS's blob IDs.
- The Vulkan vocabulary moved 4 → 5. It was re-verified, not just re-pinned: v5
  only adds `i16`/`i64`/`f64` arith names, and no vendored vector names them.
- KISS's §6.8-0002 discriminating target-match pairs are now asserted.

### Not in this release

- The §6.6-0019 weight-role `<wdt>` (the `gem_weight_role_discriminator`
  vector). It needs role hints in the derivation API, and has a separate ticket.
- The stride-0 vs nonzero-stride unit-axis mask question (two tokens for one
  broadcast meaning). The derivation follows the text literally, and KISS has
  raised the question with the PM.

**Cross-checked against an independent implementation:** the `[4,1]` token and
the right-aligned work class match Fuel's own deriver, run on the same inputs.
The #519 examples are pinned as goldens.

> **About the merge commit.** `9574050`, the squash of #32, is titled *"(DRAFT) …
> awaiting baracuda + Fuel sign-off"*. The squash took the PR's title as it stood
> before the sign-offs arrived. Both had landed before the merge, and both are
> recorded on #32: Fuel's by its own re-run (fuel#285), baracuda's as relayed by
> the portfolio PM. Published 2026-10-02 07:43–07:44Z from that commit, in the order
> `unpopped-vocab` → `unpopped` → `unpopped-cpu-c` → `unpopped-slang`.

## 2026-10-01 — everything at `0.13.0`

### ⚠️ Structure keys change for two zero-valued inputs (wire-visible)

Two derivations read zero as "maximally divisible", because zero passes every
`x % n == 0` test. Both were live in `0.12.0` and both are fixed (#29):

| Input | `0.12.0` derived | `0.13.0` derives | KISS-CLASSIFY |
|---|---|---|---|
| A contraction with `K = 0` (rank 2 or 3) | `k_div` `d16` | `da` | §6.5-0012, §6.6-0016 |
| An operand with `align_bytes = 0` (unspecified alignment) | a packed width, e.g. `v4` | `v1` | §6.5-0009 |

**Who is affected:** only callers who build those two inputs. Every other key is
byte-identical; the full workspace and its KISS byte-match vectors pass
unchanged. **If you cache on the token, a cached entry for either input was
keyed under the wrong cell.** Re-derive it rather than migrating it.

**Why it went unnoticed:** the per-operand sub-key reaches the divisibility
ladder only through `inner_axis`, which picks an axis of extent `> 1`, so that
path never passes `0`. The contraction path calls it on the raw K extent. An
outside report of the K case was retracted on the strength of the per-operand
path, and measuring through the public `structure_key` reversed the
retraction. Pinned by `unpopped-vocab/tests/zero_is_not_maximally_divisible.rs`,
whose three tests fail on `0.12.0`.

**Deliberately not changed:** an operand whose innermost non-unit axis has
extent `0` (e.g. `[8, 0]`) still derives from the outer axis (`d8`). Whether
that is wrong depends on how KISS resolves its open "active axis" question for
§6.3-0011.

## 2026-10-01 — everything at `0.12.0`

Covers `0.11.1`–`0.12.0`, none of which was published before this.

### ⚠️ BREAKING — the `seam` feature and `unpopped::jit::seam` are removed

`0.11.0` shipped an optional `seam` feature (`seam =
["dep:fuel-kernel-seam-types"]`) and `#[cfg(feature = "seam")] pub mod seam` in
`jit.rs`, which converted Fuel's `PatternNode` into this crate's. Both are gone,
along with the `fuel-kernel-seam-types` dependency (#18).

**Why:** this crate's public API named another project's type, so every major
release of `fuel-kernel-seam-types` forced a major release of `unpopped`. It had
already happened once. A `0.10.3` → `0.11.2` bump put two incompatible
`PatternNode` types into `baracuda-cuda-emit`'s dependency graph.

**Migrate:**
- To synthesize from this crate's own `unpopped::pattern::PatternNode`, call the
  native entry point `unpopped::jit::synthesize(&JitRequest, ..)` directly.
- If you start from Fuel's `PatternNode`, the conversion now lives in
  `baracuda-cuda-emit` (`src/seam.rs`, relocated near-verbatim). That crate
  already depends on `fuel-kernel-seam-types` directly.
- `--all-features` no longer turns on anything but `convert`.

The five tests that reached synthesis only through the seam (happy path,
recursion, three typed declines) now call the native entry point, in
`tests/native_synthesize_reaches_core.rs`. Before this change, no test in the
crate called that entry point directly.

### `convert` (the CUDA/Slang lifter, off by default) lifts more

- **Local variables resolve (#22).** A local that is assigned exactly once is
  substituted into the lifted body. Compound assignments (`+=`) and `++`/`--` count
  as further assignments, so a running accumulator is never mistaken for a
  constant. The crate's own scan-vs-elementwise test caught a first draft that
  missed them.
- **Comparisons and the ternary operator lift (#25)** to `Cmp*` and `Select`.

Both target constructs the lifter previously *refused*: an unresolved
identifier, or an unrecognized comparison token. They are meant to widen what is
accepted, not to change what an already-accepted kernel lifts to. That is the
design intent. No before/after comparison over a corpus has been run to measure
it.

### `unpopped-vocab`: the `cuda:` reserved id block is gone (#15)

The four `cuda:sm*` target tokens were interned at fixed ids `0`–`3`, the last
CUDA-specific vocabulary in this neutral crate. They now intern like every
other namespace's tokens, via `TargetId::parse`. **Only code that relied on a
`TargetId`'s numeric value could notice, and that was always forbidden:** ids
are process-local and never serialized. Tokens and `==` are unchanged.

### Also in this range, with no behaviour change

- **Doc comment correction (#8):** the `VariantFidelity::DeterministicallyDivergent`
  doc comment no longer cites the retracted measurement (see the 0.11.0
  correction below). The fix landed after `0.11.0` was packaged, so `0.12.0` is
  the first published version to carry it.
- **New tests that pin existing behaviour:**
  - the one-version rule (#7);
  - dtype `storage_bits`/`kind` checked against KISS's manifest (#16);
  - `Complex64`'s width (#19).
- **Dependencies:**
  - lockfile refreshed to the latest in-range versions (#20);
  - `kiss-ref-core`/`kiss-ops-vocab`/`kiss-classify-vocab` `0.3.4` (#24), a
    dev-dependency only, never reaching consumers.
- **Design and sizing docs:** #10–#12, #14, #26, #27. The
  `golden_regen_checkpoint` date moved to 2026-10-22 (#28). CI's checkout
  action moved to v7 (#9).

## 2026-09-09 — everything at `0.11.0`

⚠️ **`unpopped-vocab 0.4.0 → 0.11.0`, `unpopped-cpu-c 0.9.0 → 0.11.0`,
`unpopped-slang 0.7.0 → 0.11.0`, `unpopped 0.10.0 → 0.11.0` — the skipped version
numbers are a DELIBERATE UNIFICATION, not a mistake, and this line exists so
nobody re-derives it later.**

**CireSnave's standing rule, quoted:** *"I always want all crates within a project
to use the same version number so developers consuming them know which go with
which."* The exception in that rule is for a **cross-project** pair — an emitter
crate in another project whose number answers *"which Unpopped does this work
with"*. `unpopped-cpu-c` and `unpopped-slang` are **inside** this project, so
their numbers answered nothing and the exception does not reach them.

**`unpopped-conformance` moves too, although it is `publish = false` and has no
external consumer.** Leaving one crate off the shared number recreates the
question the rule exists to close.

### ⚠️ For consumers: one number, written four times, and all four must move together

Before this release a consumer wrote **three different numbers** for four crates.
Now they write `0.11.0` four times.

**Every current pin excludes `0.11.0`, so nothing arrives on a `cargo update`:**

    ^0.10.0  = >=0.10.0, <0.11.0     excludes 0.11.0
    ^0.4.0   = >=0.4.0,  <0.5.0      excludes 0.11.0
    ^0.9.0   = >=0.9.0,  <0.10.0     excludes 0.11.0
    ^0.7.0   = >=0.7.0,  <0.8.0      excludes 0.11.0

⚠️ **A PARTIAL edit resolves two versions of a crate into one graph** — e.g.
moving `unpopped` to `0.11.0` while leaving `unpopped-vocab` at `^0.4.0` gives
`0.4.0` (the direct pin) *and* `0.11.0` (what `unpopped 0.11.0` requires), which
are different types with the same name. **Unification makes that mistake more
visible, not less: four identical numbers make a missed line obvious, where three
different numbers made it look plausible.**


**Remedies issue #4 for TODAY'S INSTANCE ONLY.** ⚠️ Publishing makes published ==
workspace **at one moment**; the member keeps evolving and the gap reopens on the
next unpublished commit. **A green publish is not the class being closed** —
baracuda's argument, and the portfolio gate vulkane is building is the durable
half.

### ⚠️ BREAKING — `VariantFidelity::ReassociatedDeterministic` → `DeterministicallyDivergent`

The old name asserted **reassociation** for every member of the class. The new one
names the class extensionally — deterministic on fixed hardware, different bits
from the default, undirected — and **asserts no mechanism**.

**One selection policy, so one variant** — a fifth with identical semantics is a
distinction no consumer can act on. **The FKC determinism spelling is unchanged
(`same_hardware_bitwise`)**, asserted by `the_fidelity_rename_is_wire_invisible`.

> 🔴 **CORRECTION, 2026-09-11 — this entry originally cited a RETRACTED
> measurement, and the retracted text is immutable in the published `0.11.0`.**
>
> It read: *"baracuda measured a member whose reduction tree was provably
> unchanged and whose bits differed anyway — cache-vs-recompute of an equal
> `expf`, 12,283,172 of 16,777,216 elements, worst 11 ULP, reproducible."*
>
> **That measurement was withdrawn on 2026-09-09 (baracuda#99, closed
> `not_planned`), two days before `0.11.0` shipped.** It had been taken against
> `variants[1]` — which is `prec`, declared `MorePrecise`, whose entire purpose is
> to differ from the base. Measured against `smemrow`'s own kernel: **0 ULP over
> 16,777,216 elements.** `smemrow` is `BitIdentical` and always was.
>
> ⚠️ **So the second mechanism this rename was argued from has NO measured
> instance.** It is recorded in `backend.rs` as a hypothesis rather than an
> observation.
>
> **The name is kept.** The old name asserted a mechanism for every member; this
> one asserts none, and **removing an unsupported claim is not the same act as
> adding one.** Reverting would re-assert *"every member is a reassociation"* on
> evidence no better than what was withdrawn, and would cost the one adopter a
> second 19-site rename. **If the class turns out to be reassociation-only, this
> name is less specific than it could be, which is not wrong.**
>
> **How it happened:** the retraction reached baracuda's issue and two of their
> code comments, and never reached here. ⚠️ **A RETRACTION MUST TRAVEL AS FAR AS
> THE CLAIM DID — and a correction is *more* discoverable than the original while
> still not travelling, because discoverability is pull and propagation is push.
> Nobody goes looking at a number they have already written down.**

### `ulp_bound` no longer declines an expression whose exactness is by construction

An expression built entirely from `BinaryOp::is_int_only` operators has no
rounding step on any hardware, so a **non-CUDA** target now gets `0.0` where it got
`INFINITY`, and `precision_of` rates it `("correctly_rounded", Some(0))` instead
of `("approximate", None)`.

⚠️ **Keyed on `is_int_only`, NOT on `ulp_sum(e) == 0`.** `unary_ulp` rates
`Sqrt`/`Recip`/`Floor` at 0.0 and `binary_ulp` rates
`Copysign`/`Nextafter`/`FmaxIeee` at 0.0 — **those zeros are IEEE claims about a
target's float unit**, exactly the borrowed assertion the namespace gate exists to
refuse. **A CUDA target is untouched**, asserted across four rating tiers.

Found by baracuda while adopting 0.10.0, applying this repo's own rule: *a
prescription that errs conservatively has no complainant.*

### `Access::Contraction` is documented ALWAYS COMPUTED, and its premise is guarded

"No predicate applies" is an answer; its absence read as an oversight. The K-fold
is a sum over products and the variant carries **no operator field at all**.
Guarded by an exhaustive match on `AccumSpec` — **0 exhaustive matches existed
before, so a variant that falsifies the ruling used to compile with 0 errors, 0
clippy warnings and 0 test failures.**

### ⚠️ Why the emitters move at all

`unpopped-cpu-c 0.9.0` and `unpopped-slang 0.7.0` **both require `unpopped =
"0.10.0"`** — measured from their served manifests. For a 0.x crate `^0.10.0`
**excludes 0.11.0**, so publishing `unpopped` alone would strand both emitters on
the old line while every sibling moved. **`baracuda-cuda-emit` declares all four**,
so it would then resolve two `unpopped` versions into one graph — the two-artifacts
defect this release exists to close.

**Source of both emitters is byte-identical to their published versions. They move
because a dependency did, and then again because of the unification above.**

⚠️ **This section was headed *"Why three crates and not one"* until the
unification, and the count was correct when written.** The unification made it
five, and **a heading carrying a stale number is the artifact a reader trusts
most** — it was found by a reviewer's nitpick about an unrelated count two
sections above, which is the only reason anyone re-read this one.

## 2026-09-06 — `unpopped-vocab 0.4.0` · `unpopped 0.10.0` · `unpopped-cpu-c 0.9.0` · `unpopped-slang 0.7.0`

⚠️ **Everything below shipped. It sat under "Unreleased" for forty minutes after
being published** — caught while adding the next entry, which is the only reason
it was caught at all. **A changelog whose "Unreleased" section describes released
work is worse than no changelog: it is the same object as a version number that
no longer matches its code, one level up.**

**Published and verified against the served tarballs; `unpopped-vocab 0.4.0` and
`unpopped 0.10.0` src trees are byte-identical to the merge commit.**

**Version cascade applied 2026-09-06 on the PM's ruling: `unpopped-vocab 0.4.0`,
never `0.3.3`.**

| crate | was | now | why it moved |
|---|---|---|---|
| `unpopped-vocab` | 0.3.2 | **0.4.0** | two behaviour changes, one numeric |
| `unpopped` | 0.9.0 | **0.10.0** | new public fn, a deprecation, and its vocab requirement moved |
| `unpopped-cpu-c` | 0.8.0 | **0.9.0** | ⚠️ **source BYTE-IDENTICAL to the published 0.8.0** — bumped ONLY because a dependency did |
| `unpopped-slang` | 0.6.0 | **0.7.0** | ⚠️ **source BYTE-IDENTICAL to the published 0.6.0** — bumped ONLY because a dependency did |
| `unpopped-conformance` | 0.1.0 | 0.1.0 | `publish = false`; not on the registry |

⚠️ **The two "byte-identical" rows are stated because a reader who sees a version
move with no behaviour change will otherwise assume there was one they cannot
find.** Both were verified by diffing the served `.crate` tarball against the
tree, not by inspection.

### Downstream, measured rather than reasoned

**Registry reverse-dependencies of `unpopped-vocab`: 6, all in this portfolio**
(control: `serde` returns 120,294, so the query works).

    baracuda-cuda-emit / -kernels-types / -types   require ^0.1.0  (an OLD line;
                                                    0.3.x never reached them)
    unpopped / unpopped-cpu-c / unpopped-slang     require ^0.3.2

⚠️ **And the hop that mattered was the one nobody had measured.** baracuda's
**working tree** declares `unpopped-vocab = "0.3.2"` **directly** — measured at
their `origin/main` `2b3292ca`, after fetching, because the local ref was stale.
**So the reassurance "baracuda is caret-pinned `^0.8.7` on `unpopped`, therefore
cannot reach the new vocab" was FALSE**: their direct vocab dependency would have
taken `0.3.3` on the next `cargo update`, bypassing the `unpopped` pin entirely.

**Their exposure to the numeric change is nonetheless zero, measured on their
side:** `Fp8E5M2::from_f32` has **0** call sites in baracuda. **`Fp8E4M3FN` is
unchanged by this release** — 0 diff lines between published 0.3.2 and the tree
mention E4M3, against a control of 5 mentioning E5M2 — so their two E4M3 call
sites are unaffected.


### `unpopped-vocab` 0.3.2 → 0.4.0 — the divergence that forced this release

**Two behaviour changes. Neither could ship as `0.3.3`.** The then-published
`unpopped 0.9.0` requires `unpopped-vocab = "0.3.2"`, a caret range that
**accepts 0.3.3** — so a patch bump reaches every existing consumer on their next
`cargo update`, with no compile error and no version signal.

- **`ElementKind` E5M2 `from_f32` overflow now produces INFINITY, not a saturated
  max-finite.** Overflow at `|x| >= 61440.0` (the midpoint between max-finite
  `57344` and `2^16`; the midpoint itself rounds to infinity, because
  ties-to-even picks the significand ending `00`) returns `sign | 0x7C`.
  Previously it delegated to `float8` 0.7.0, whose E5M2 encoder saturates **every**
  overflow to `0x7B` — including a literal `f32::INFINITY` — while **its own
  decoder maps `0x7C` to infinity**. ⚠️ **This is a NUMERIC change to emitted fp8
  bytes**: a value that encoded as `57344.0` now encodes as `inf`.
- **`winner_of` resolves an exact timing tie deterministically.** Previously
  `sort_by` stability made the winner, its `entry_point`, and the `margin` depend
  on the order the caller pushed candidates. The tiebreak is
  median → implementor code → entry point and carries **no** performance or
  preference meaning. ⚠️ **Emitted dispatch tables can differ for tied cells**,
  which is generated code a consumer has committed.

### `unpopped` 0.9.0 → 0.10.0

- **Added `ir::is_bit_move_row_reduce_output(stages, epilogue)`** — the
  §6.16-0011 classification for the `Access::RowReduce` shape, which had no
  helper at all. Purely additive.
- **Deprecated `ir::is_bit_move_reduce`** — it answers a fold question for shapes
  that may have several folds. Non-breaking; the two replacements have signatures
  that cannot be satisfied without the whole input→output path.
- **Corrected the `RowReduce` gating prescription in `ir.rs`'s field table.** The
  old advice classified an all-move multi-stage fold as *computed*. ⚠️ **It erred
  CONSERVATIVELY, so a consumer following it emitted pessimised-but-conforming
  code and had no symptom to report.**

### `unpopped-cpu-c` 0.8.0 → 0.9.0 · `unpopped-slang` 0.6.0 → 0.7.0

- **Published 0.8.0 source is IDENTICAL to the tree.** No bump needed on its own
  account; it moves only if its `unpopped` requirement does.

---


Versions at or before `unpopped 0.9.0` / `unpopped-cpu-c 0.8.0` /
`unpopped-vocab 0.3.2` predate this file. **Their contents were verified against
the served tarballs rather than reconstructed from memory**, which is the only
claim this file makes about them.
