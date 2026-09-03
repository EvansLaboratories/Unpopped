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

> **⚠️ THIS SECTION NOW HAS A DATE AND AN OWNER.** It previously had neither, and
> that is a defect rather than sequencing: *a gate whose trigger nobody is
> responsible for pulling is not a deferral, it is a permanent hold wearing a
> deferral's clothes.* An item held on *"when the regen happens"* never fires if
> the regen never happens.
>
> **Checkpoint: 2026-10-01. Owner: the portfolio PM** (accepts the regen; this
> workspace drafts and implements it). **Enforced, not remembered** —
> `crates/unpopped-conformance/tests/golden_regen_checkpoint.rs` goes red on that
> date and says what to do. Proven to fire by moving the date into the past.
>
> When it reds, exactly one of: **schedule the regen** and do all three together,
> or **move the date in a commit that says why it slipped and who agreed.** A
> moved date with a reason is a live deferral; a moved date without one is this
> defect returning.
>
> The window opened when `unpopped 0.7.0` published on 2026-09-02.

### Why FP8's shape is the RIGHT answer and not merely a working one

**Offered by vulkane 2026-09-02 as a data point from their namespace, explicitly
not as advice about ours. It applies, one layer over.**

Their finding: five tokens — `f16 f64 i16 i64 i8` — appear in **both**
`arith_names` and `component_types` in their published manifest, and they are
**different capabilities with different witnesses** (`Arith::FLOAT16` is
`shaderFloat16`; `ComponentType::F16` is a cooperative-matrix element type read
from another query). **So a bare token is not a key there; `(token, class)` is.**
They solved it by making the field part of the identity rather than by inventing
globally unique names.

**The same shape is in `scalar_ctype`, and it is the row's actual defect:**

| dtype | `scalar_ctype` returns | which question does it answer? |
|---|---|---|
| `F32` | `float` | storage **and** compute — they coincide |
| `Fp8E4M3FN` | `unsigned char` | **storage only**; compute goes via `promote_load_f32` to f32 |
| `F16` | `__half` | storage **and** native compute — conflated, vendor-named |

**`scalar_ctype` is a bare name where the real key is `(dtype, storage-or-compute)`.**
The two coincide for wide types and diverge for narrow ones, which is why the
conflation is invisible until a narrow dtype arrives — and why this workspace has
already fixed one storage-vs-compute trap in CpuC.

⚠️ **So adopting FP8's shape is not "a portable spelling that happens to work."
It is making `scalar_ctype` answer exactly one question** — storage — and routing
compute through the promote/demote path where it already lives for FP8. That is
vulkane's remedy stated in this crate's terms: **they made the field part of the
identity; the analogue here is making the function answer one question.**

**And their finding confirmed a guard here was better than it knew** —
**verified against the manifest rather than relayed**, at vulkane `a9f89e5`,
`vocabulary_version: 5`, on their advice that a token colliding today may not be
the one that collided at v4:

```
arith_names      dot8 f16 f64 i16 i64 i8 st16 st8            (8)
component_types  f16 f32 f64 bf16 i8 i16 i32 i64 u8 u16
                 u32 u64 f8e4m3fn f8e5m2 i8packed u8packed   (16)
IN BOTH          f16  f64  i16  i64  i8
```

`unpopped-slang`'s gate test asserts
`!can("vulkan:sg32.arith-none.cm-i8", ElementKind::U8)` — an `i8` in some *other*
field must not answer for `arith`. **Written as a generic substring-collision
guard; `cm-` is the cooperative-matrix field and `i8` is confirmed present in both
alphabets**, so it defends a real documented collision rather than a hypothetical.

⚠️ **One precision the manifest adds that the test's comment should not overstate:**
a `cm-` tuple is *M-N-K plus four component types*, so **`cm-i8` is not a
well-formed cooperative-matrix value** — it is a minimal probe for the substring
class, not a realistic token. **The hazard it guards is real; the token it uses is
synthetic**, and those are different claims.

### ⚠️ f16 and bf16 are NOT parallel in the vulkan namespace

**Offered by vulkane 2026-09-02, verified in the packaged `kiss-vulkan-vocab
0.4.0` (`vocabulary_version: 5`) rather than taken from their message:**

```
       arith   component_type
f16    yes     yes
bf16   NO      yes
f32    NO      yes
f64    yes     yes
```

**`arith_names` only names capabilities a device can LACK.** `f16` is there
because `shaderFloat16` is an optional feature bit; `f32` is absent because it is
baseline; **`bf16` is absent because Vulkan has no bfloat16 scalar-arithmetic
feature bit at all.** Cooperative-matrix support for bf16 elements exists; a
scalar *"can this device do bf16 arithmetic"* capability does not. **It is a gap
in Vulkan, not a choice of theirs.**

**This tree treats the two as parallel everywhere** — `scalar_ctype` gives
`__half`/`__nv_bfloat16`, `half_load_intrinsic` gives
`__half2float`/`__bfloat162float`, and `unpopped-slang` declines both together.

**For CUDA that parallelism is correct** — both types exist with matching
intrinsics, so the storage-shape half of this row (spell storage, emit software
helpers) is symmetric and unaffected.

⚠️ **It stops being correct the moment a vulkan-family backend tries to GATE
them**, which is the step immediately after this row lands. `f16` gates on
`arith` containing `f16`. **`bf16` has nothing to gate on** — the honest answer is
a permanent decline for scalar bf16 arithmetic, or a route through the
cooperative-matrix surface, which is a different mechanism entirely.

**So "f16/bf16" is one row for the storage change and two cases for the
capability gate.** Recorded now because the asymmetry is invisible from this side:
nothing in this workspace distinguishes them, and the first thing to notice would
have been a gate that silently never satisfies — **the failure mode with no error
message.**

### The f16/bf16 shadow surface, for a consumer that must pin current bytes

**Measured at HEAD 2026-09-02**, because baracuda asked for the *surface* rather
than the change — the difference between shadowing 2 functions and discovering in
October that it was 4.

**MOVES when the halves adopt FP8's shape — must be shadowed:**

| fn | why it moves |
|---|---|
| `scalar_ctype` | F16/Bf16 arms return `__half` / `__nv_bfloat16` directly |
| `promote_load_f32` | chains `narrow_load_fn` → `half_load_intrinsic` |
| `demote_store_f32` | chains `narrow_store_fn` → `half_store_intrinsic` |
| **`cast_scalar`** | **delegates to all three above** |

⚠️ **`cast_scalar` is the one a grep would miss: it contains no `F16` literal at
all.** It moves purely by delegation, so *"which functions mention F16"* returns
three and the true answer is four. **The surface is defined by the call graph,
not by the token.**

**DOES NOT move:**

- `store_expr_of` — its match routes **only the FP8 pair** to `demote_store_f32`,
  written as an explicit match precisely so it does not inherit whatever
  `narrow_store_fn` happens to do. That deliberate choice is what keeps it still.
- `dtype_tag` — spells the NAME (`"f16"`), not the storage.
- `half_load_intrinsic` / `half_store_intrinsic` / `narrow_load_fn` /
  `narrow_store_fn` — the leaves that actually change, but a consumer shadowing
  the four wrappers above never calls them.

⚠️ **Caveat on the usage counts**, which matter to a consumer sizing the work:
they come from `baracuda-cuda-emit-0.0.1-alpha.79`, **which pins
`unpopped = "0.1.0"` — five minor versions back.** The *set* of functions is from
this tree at HEAD and is authoritative; the *counts* are indicative only and the
consumer must re-derive them on their own main.

⚠️ **BUT THE OVERRIDE ANSWER DOES NOT CLOSE THEM — it answers a different
question than the rows ask.** Recorded 2026-09-02 after the PM read the
consumer-side shadow ruling as disposing of the f16/bf16 row.

**The shadow ruling answers "how does a backend get a DIFFERENT spelling."** That
was real and it is settled. **The f16/bf16 row asks something else: should the
neutral module spell a vendor type AT ALL.** `cfamily::scalar_ctype(F16)` returns
`__half` to anyone who calls it, from a module documented "deliberately
backend-neutral" and published to crates.io. A consumer shadowing does not remove
that.

**And the answer is already in the tree, working, with its own tripwire comment
naming it:**

> *"THE SEAM, WORKING. `f8e4m3fn`/`f8e5m2` spell `unsigned char` — the STORAGE
> type — and their conversions are software helpers emitted into the kernel
> (`cfamily::fp8_helpers`), not vendor intrinsics. **That is precisely the shape
> the `F16`/`Bf16` arms below still need**, and FP8 got it first because **it had
> no existing goldens to rewrite**."*

**So the row is: adopt FP8's proven shape for f16/bf16 — spell the storage type,
emit software conversion helpers — and the ONLY thing gating it is the goldens,
exactly as Section B says.** Not an unanswered design question. **A known target
shape with a rewrite cost.**

**Slang complex is independent of all of it:** `unpopped-slang` names `Complex`
nowhere, so it needs a prelude whether or not any override mechanism exists.

**All three rows survive. The checkpoint is more justified, not less.**

⚠️ **THESE THREE ARE ONE MECHANISM, not three items.** Found 2026-09-02 by
auditing this file's own row count, and it changes what the regen has to be.

The "Not deferred, just worth knowing" section already says it about complex —
*"structurally identical to f16/bf16: **a dtype whose neutral spelling must be
overridable per backend.** Complex only LOOKED settled because its portable
default already works"* — and the same is true of FP8 and the sub-byte types,
which reached the portable-struct answer for the same reason. So:

| row | what it needs |
|---|---|
| the f16/bf16 spelling seam | a per-backend override for `scalar_ctype` |
| a Slang complex prelude | the same override, plus a Slang-side default |
| FP8 / sub-byte (from section D) | the same override, plus Slang-side codecs |

**One override mechanism discharges all of them.** The regen is what the
*overrides* cost, not what the *seam* costs.

⚠️ **And that suggests a split worth pricing before the regen is scheduled:**
adding the seam with today's spelling as its default changes **no emitted byte**,
exactly as the `temp` seam did in `9123b84` — so the seam is not gated on the
regen at all, and only a backend actually exercising it is.

✅ **ANSWERED 2026-09-02, and the answer is that there is nothing to build here.**
Three shapes were offered — a new additive `cfamily` function, a `Lowering` field
the emitters refactor onto, or something keyed off the target. Baracuda priced
all three and returned **none of them**: the override belongs in the **consumer**,
via a **local function that shadows the name and delegates the rest**.

```rust
// baracuda-cuda-emit/src/cuda.rs — `unary_f32` is NOT in their cfamily import
fn unary_f32(op, x) {
    match op { Rsqrt => "rsqrtf({x})",
               other => unpopped::cfamily::unary_f32(other, x) }
}
```

Their reasons for declining each shape are the part worth keeping. The additive
function is **redundant with shadowing, and two public names on a four-consumer
crate IS the divergence generator**. The `Lowering` field would ripple across
~24 free `emit_*(plan, ..)` sites that have no `Lowering` in scope, **for zero
benefit to the consumer doing the refactor**. And the target-keyed option — the
one *cheapest for them* — they turned down because **it would force this crate to
model targets**, a concept it does not want. A consumer declining the option that
costs them least because of what it would cost us is the right instinct from the
right side.

**Both in-tree emitters had already resolved it independently, in two different
ways, and neither needed shared API:**

| emitter | shape | why it is right there |
|---|---|---|
| `unpopped-cpu-c` | **calls `cfamily::scalar_ctype` directly** (5 sites) | CpuC *is* portable C — the neutral default is exactly what it wants |
| `unpopped-slang` | **zero references; a full local `slang_ctype`** | a non-C-family target needs different spellings for most dtypes, so delegating to a C table would be wrong more often than right |

**Three consumers, three shapes, no shared mechanism** — and Slang's is stronger
evidence than a delegating shadow, because it replaced the table outright and
that was simply the natural thing to do without anyone designing for it.

⚠️ **So the collapse goes further than "the seam is not gated on the regen": there
is no seam.** Each backend spells its own; `cfamily` keeps the neutral default;
the regen shrinks to only the backends that actually deviate — **and CUDA's
default already is the spelling CUDA wants.**

✅ **CLOSED 2026-09-02. Every arm measured; nobody assessed willingness.**

```
vulkane    NOT A PARTY   0 of 258 packages; 0 in *.rs; control 48
fuel       NOT A PARTY   0 real hits; controls dtype_token 40, baracuda 3097
baracuda   YES, A PARTY  ~52 sites across the four functions
```

**The conclusion stands stronger than a 3-0 would have.** Two lanes measured
themselves OUT of the question and asked not to be counted; the one lane that is
a party measured itself IN and committed to a plan. **A tally where the arms
disagree about their own membership is evidence; one where everybody says yes is
a headcount.**

⚠️ **CARRY THIS ARTEFACT WITH THE ZERO, at fuel's architect's specific request.**
Their `grep cast_scalar` over fuel returns **1**, and it is
`fuel-ir/src/shape.rs:1090` — `fn broad`**`cast_scalar`**`_with_matrix()`. **A
substring, not a call site. The real count is zero.**

**The reason it must travel:** the finding *above* — that `cast_scalar` moves by
**delegation** and contains no `F16` literal — is exactly what will send someone
to re-grep fuel for it. **They will get `1` and read it as a call site.** A
recorded zero with no explanation loses to a fresh grep returning one.

**Baracuda's plan, recorded so the row says what actually happens:** a
consumer-side local shadow, ~10 lines, byte-identical at adoption, extended to
all four functions above — since `promote_load_f32`/`demote_store_f32` are the
op-emission half rather than just the ctype string.

⚠️ **Their design note is the part worth keeping: the four shadows must form a
CLOSED set** — local `cast_scalar` calling local `promote`/`demote`/`scalar_ctype`,
never cfamily's moving versions. **"A shadow that delegates back into the thing it
is shadowing away from is not a shadow."** That is the failure mode a partial
shadow has, and it would look correct right up until the leaf moved.

**Ordering is load-bearing and they took it as such: the shadow lands BEFORE the
publish, not with it** — the `8f42471`-after-#46 precedent. Timing is theirs.

*(Closing the loop: the idiom baracuda cites as proven is the shadow they built
for PR #46 — the precondition of this workspace's own `rsqrt` fix in `8f42471`.
Forcing that fix created the pattern that later answered this question.)*

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
- **A Slang complex prelude (`c64`/`c128`).** CpuC implements complex as an
  emitted **C prelude** — `typedef struct { float re, im; } unpopped_c64;` plus a
  library of `unpopped_c64_add`/`_sub`/… statics. Slang needs a parallel prelude
  in its own syntax. **That is a new emitted-text surface, so it changes bytes
  for every complex kernel and belongs to the regen event rather than to dtype
  coverage.** Filed here on the PM's ruling 2026-08-27, bundled with the
  f16/bf16 seam so one dated regen discharges three held items rather than two.

---

## C. Needs a ruling — not ours to decide alone

- ~~**Whether `required_fidelity` belongs to us at all**~~ — **ANSWERED
  2026-09-02: it is OURS, and the two are complementary rather than duplicated.**
  **Askee: fuel (architect). Asked 2026-08-14, answered 2026-09-02 — 19 days.**
  (`conformance.md` OPEN-1.)

  Fuel measured their side at `1fb2e9db`: **0 occurrences of
  `required_fidelity`/`RequiredFidelity` in any spelling**; "fidelity" appears 4
  times, all prose in doc comments. Their runtime layer is
  `fuel-dispatch/src/fkc/verify/` (14 files), which **empirically verifies a
  kernel contract's `precision` claims and ledgers `(kernel, backend, dtypes,
  claim)` tuples, downgrading at import any claim the ledger does not cover.**

  **That layer is SUPPLY-SIDE — "has this kernel EARNED what it CLAIMS?"** Ours
  is neither that nor a caller's demand: `required_fidelity(plan, operands)`
  derives, from the body's own structure, **the band a CORRECT implementation
  must land in** — integer cells bit-exact because wrapping is modelled exactly,
  otherwise `(arith_steps + reduction_len + 2·ulp_bound) · unit_roundoff`. It is
  the threshold `compare` judges an emitted kernel against its oracle with.

  ⚠️ **The decisive evidence is structural, not the naming argument: the input is
  `plan.body`, an Unpopped IR node. Fuel does not have it and cannot compute this
  band.** Their ledger records what was *measured after the fact*; this derives
  what *must be true a priori*. A kernel can pass ours and still lack an entry in
  theirs, and both statements are worth having.

  **Fuel's own limit, stated by them: they answered about FUEL's surface only and
  did not read `oracle::required_fidelity`** — the half only they could supply,
  handing back the half only we could.

  **Carried forward as a live defect rather than closed clean:** `ulp_bound` sums
  a **CUDA** per-op table and neither it nor `required_fidelity` takes a target,
  so the band is CUDA's accuracy for every backend. Vulkan's `exp` is 3 ULP
  against CUDA `expf`'s 2, so **a conforming Vulkan kernel fails a comparison it
  should pass.** Latent today (the only caller is `unpopped-cpu-c`'s test), live
  the moment a Vulkan backend compares through it. Needs a per-target accuracy
  seam — see `ulp_bound`'s own KNOWN LIMIT. **Owner: this workspace. Party:
  vulkane.**
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
- ~~**Whether NaN propagation (rule N2) is universally required**~~ —
  **ANSWERED: no. Propagation is a per-op pinned semantic, not a global rule.**
  Verified in `spec/ops.md` at `ef5efec`, not relayed: **`max_prop`/`min_prop`**
  (NaN-*propagating*, matching `torch.maximum`/`torch.minimum`) and
  **`fmax_ieee`/`fmin_ieee`** (IEEE-754 `maxNum`/`minNum`, NaN-*suppressing*) are
  **four distinct ops**, and §80-81 lists *"pinned per-op numeric semantics — for
  each op: NaN propagation, signed-zero behavior, IEEE versus NaN-propagating
  min/max"*. **An implementation does not choose a propagation policy; it
  implements whichever op it was asked for.**

  The Vulkan worry that opened this entry is *not* refuted — it is a different
  question. `shaderSignedZeroInfNanPreserveFloat32` being a `returnedonly`
  property still means a device may not deliver what the op pins, and **whether a
  given toolchain delivers it must be measured per target and never inherited.**
  That half is live and is item **D — per-target N2 verification**, where it
  belongs. The two claims were being answered as one.
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
- ~~**The KISS cost model** (#125)~~ — **ANSWERED: it is KISS-Contract's, not a
  separate model.** `contract.md:390` defines `cost_provenance` (`declared` or
  `measured`, authored in Guarantees, mirrored in Provenance);
  **KISS-CONTRACT-6.7-0006** requires the Capabilities `cost` to carry a cost
  *class* plus cost expressions; **KISS-CONTRACT-6.8-0007** names the cost model
  itself. Our position is unchanged and now has a home: we emit a two-axis vector
  with provenance, and a generator can only ever say `declared`.

  ⚠️ **Read the citation with its limit, which I found by checking rather than
  quoting.** The conformance table maps both clauses to test names —
  `test_contract_cost_expressions` and `test_contract_cost_single_home` — and
  **only the second one exists** (`conformance/tests/contract_schema.rs:402`, at
  `ef5efec`; positive control: the same grep finds other tests). **The row for
  6.7-0006 names a checker that is not there**, which reads as covered. So
  6.8-0007 is conformance-checked, 6.7-0006 is normative-only, and *"the cost
  model is checkable"* is true of exactly half of it. Reported to the KISS
  architect; the table is theirs, not ours.
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
- **A schedule language, and whether Unpopped should have one.** Tiling, shared-
  memory staging and unroll factor have **no representation** in the IR: a
  `StructureKey` carries `contig`/`bcast`/`vec_width`/`inner_div`/`flipped` per
  operand, `WorkClass` is three values, and `Schedule::Contraction` is the one
  variant of ten carrying no parameters — precisely the one that would need tile
  dimensions. Seven of the other nine already carry real coordinates (`Window`
  holds `size`/`stride`/`dilation`/`pad_lo`/`pad_hi`), **so the gap is a specific
  hole in a pattern the design already uses, not a missing concept.**

  **The cost is not in the key, it is downstream.** `Lowering`'s eight seams are
  expression spellers; there is no seam for *"stage this into shared memory"*, and
  adding one changes what a `Backend` is. So this is priced as **"give Unpopped a
  schedule language"**, not as a field — and a schedule language with nothing
  searching it is a more elaborate way to hardcode a number, so it also implies a
  search. **CireSnave's spend, not ours**; recorded because it was designed in
  conversation on 2026-08-27 and would otherwise exist only in a transcript.

  **What it is NOT:** a reason to lift hand-tuned kernels. A lifter drops what it
  cannot represent; PTX bakes the schedule in; the round trip would re-enter the
  competition having discarded the advantage. `LiftError::Inexpressible` already
  refuses the population that matters (ten CUDA residue markers).

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


- ~~**Dtype lowering coverage.**~~ — **CLOSED 2026-09-02: nothing remains in this
  section.** Measured at HEAD rather than declared: **CpuC 18 Lowers / 2 NotYet**
  (f16+bf16, Section B) and **Slang 6 Lowers / 5 Blocked / 9 NotYet / 2 ByDesign**.

  **The five Slang `Blocked` cells are implemented and correctly refusing** —
  `i8`/`u8`/`i16`/`u16`/`bool` all lower when the target advertises the matching
  arithmetic capability, and this table probes `cuda:sm89`, which advertises none.
  **A gated refusal is not missing work.**

  **All nine Slang `NotYet` cells are Section B**, not this section: `f16`/`bf16`
  are the spelling seam, and both FP8s, `i4`/`u4`/`b1` and `c64`/`c128` need
  emitted preludes — a new emitted-text surface, so they ride the regen.

  ⚠️ **This row stayed open after its content had emptied**, which is the same
  defect as counting notes as work: **a row is not a unit of work, it is a
  container, and a container empties without announcing it.** Found by asking what
  was still IN it rather than by reading its title. The original entry follows,
  kept because its lesson outlives its content.

  **The original entry, kept for its lesson.** The numbers are no longer here: they are
    **measured** by `crates/unpopped-conformance/tests/dtype_lowering_coverage.rs`, which
    carries the per-dtype × per-backend table and fails when it goes stale. The
    prose figures this entry used to give were **wrong twice** — first "CpuC 9,
    Slang 5" against a measured 8, then "CpuC 8/22" against a measured 18 once the
    work below landed. Twice in the same entry, in both directions, is the whole
    argument for moving them: a coverage claim decays silently, because nothing
    about adding a dtype arm forces the sentence describing it to change. **Run the
    test.** Any number written here is a number that will be wrong.

    **Split by what actually blocks it, ruled by the portfolio PM 2026-08-27**,
    because filing the whole thing as available made it look actionable when most
    of it was not — the mirror of Section B's ownerless gate, one section down.

    **CpuC is 18/22 and its remaining two are f16/bf16 (Section B).** Slang is
    **6/22** after `u32`/`u64` landed (`6818679`); the rest sorts as:

    - **UNBLOCKED, this section:** `i8`/`u8`, `bool`, both FP8s, `i4`/`u4`/`b1`.
      Vulkane answered the gating question on 2026-08-28 and it is recorded here
      rather than left in a transcript:
      - **Grammar is `arith-f16-i8`** — hyphen-separated tuples, `.` between
        fields, `arith-none` for empty. **Juxtaposition (`arith-f16i8`) is
        MALFORMED**, not merely unusual (`spec/namespaces/vulkan.md` V-6).
      - ⚠️ **Order is load-bearing.** The set is spelled in **lexicographic** name
        order (`dot8, f16, f64, i16, i64, i8, st16, st8`), and §6.8-0002 matching
        is **byte-exact** — so `arith-i8-f16` matches *nothing*. **Sort, then join.**
        A wrong order is a well-formed string that silently never satisfies.
      - ⚠️ **There is no `u8` token.** `i8` names `shaderInt8`, which is
        **signedness-agnostic** — 8-bit integers usable in shader code. So `u8`
        arithmetic gates on `i8`, and that absence is not an omission. Signedness
        lives in the component-type vocabulary (`cm-`/`cv-`), a different alphabet.
      - ⚠️ **`st8` and `i8` answer different questions, and the question is about
        the KERNEL, not the dtype.** `st8` is `storageBuffer8BitAccess` (8-bit data
        in a buffer); `i8` is 8-bit *arithmetic*. A kernel that only loads and
        stores `Bool`-as-`U8` bytes needs `st8`; one that computes in 8-bit needs
        `i8`. Reading one as the other is a silently wrong lowering on hardware
        that is behaving correctly (V-15).
    - ✅ **`i16`/`u16` — DONE.** They spell `int16_t`/`uint16_t` and gate on `i16`
      (`shaderInt16`), which became expressible on **2026-09-02** when
      kiss-vulkan-vocab 0.4.0 / vulkane 0.14.0 shipped capability vocabulary v5.
      This entry once recorded it as *not expressible*; that was true of published
      v4 and was **an unshipped vocabulary, never a missing one.**

      ⚠️ **Re-verified against the PUBLISHED CRATE 2026-09-02, not a peer's tree.**
    My first check read `vulkan-vocabulary.json` from vulkane's working tree at
    `a9f89e5`. **I recorded that as "a topic branch" and it was `main`** — their
    correction, verified here: `git branch --contains a9f89e5` lists `main`, and
    `git ls-remote` has it as `origin/main`'s head. **My error was reading
    `git rev-parse --abbrev-ref HEAD`, which answers *where is the worktree
    pointed*, not *what branch is this commit on*.** Their checkout sat on a topic
    branch that had zero commits at the time, so it pointed at main's head.
    **Branch membership is `--contains`; `--abbrev-ref HEAD` is a different
    question that happens to agree most of the time.**

    ⚠️ **And the correction makes the lesson stronger, not weaker.** A topic branch
    being unreliable is unsurprising. **`main` being unreliable as a source for
    what shipped is the actual finding** — measured: **9 commits** between the
    published `64001e79` and `a9f89e5`. Both are "the repository"; only one is what
    anyone consumes. Pulled
    `kiss-vulkan-vocab-0.4.0.crate` from crates.io instead — 54,602 bytes,
    `.cargo_vcs_info.json` sha1 `64001e79`:

    ```
    vocabulary_version : 5
    arith_names        : dot8 f16 f64 i16 i64 i8 st16 st8
      i16 in arith?  true       u16 in arith?  FALSE
      i8  in arith?  true       u8  in arith?  FALSE
      i16 and u16 both in component_types: true
    ```

    **`u16` is not an arith capability and is not planned to be one** — Vulkan's
    feature bit is `shaderInt16`, signedness-agnostic, exactly as `shaderInt8` is
    for the 8-bit pair. **So `u16` gating on `i16` is not a convenience; it is the
    only spelling that exists**, and this crate's gate does that. The absence of
    `u8`/`u16` from `arith_names` is confirmed in the shipped artifact rather than
    inferred from the pattern.

    ⚠️ **Adopt `.cargo_vcs_info.json` with its own limit, which vulkane supplied:**
    it records the commit `cargo package` **ran from**, which is not necessarily a
    commit anyone can fetch — it can name a dirty tree or an unpushed commit.
    **It tells you what shipped; it does not guarantee you can check it out.**
    Resolve the sha before treating it as a ref. Here it does resolve and is an
    ancestor of `main` (verified), but that is a property of this release rather
    than of the mechanism.

    ⚠️ **And a correction to how I said I would detect the unblock.** I told the
      PM I would *"see `arith-i16` appear in the coverage table without being
      told."* **That could never have happened.** The coverage table probes
      `ArchSku::Sm89` — a `cuda:` token, which carries no `<arith>` field at all —
      so no vulkan capability can ever surface there. **The detector I named was
      fictional, and I would have waited on it indefinitely.** The real signal was
      the publish itself, which only the PM could see.
    - **SECTION B:** `f16`/`bf16` (the spelling seam), **complex `c64`/`c128`**,
      and — **corrected 2026-09-02** — **both FP8s and `i4`/`u4`/`b1`**. These were
      listed as unblocked on the strength of `bool` being one, and that was wrong:
      `bool` needed only a ctype because FKC §5 stores it as U8 and its ops route
      through `binary_int`. **FP8 and sub-byte need CODECS.** `cfamily::fp8_helpers`
      returns emitted C prelude text, and `sub_byte_load_fn` names
      `unpopped_i4_load` / `unpopped_b1_load`, defined as `static int
      unpopped_i4_load(const unsigned char* p, ...)`. Slang needs a parallel prelude
      in its own syntax — the same shape as complex, the same new emitted-text
      surface, and therefore the same regen. **"Follows bool's test" was true of the
      gate and false of the work.**

    **Why I could not answer the `arith` question myself, recorded because it is
    their finding not my gap:** all eight of vulkane's normative vectors carry
    `arith-none`, so the machine-readable artifact a consumer validates against
    never exercises the multi-value form. The only multi-value example in their
    tree is one string in two unit tests and a README table. They are adding a
    normative multi-value vector — **found by asking rather than guessing.**

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
- **Oracle coverage**: ~~`RowSort`~~ — **DONE**; `eval_row_sort` evaluates it
  (NaN-greatest in both directions, stable index ties, TopK read off the output
  operand's width), and the `Coverage` classifier moved `Deferred` -> `Evaluated`.
  **That also removed a `panic!` from a `pub` module**: `oracle` is
  `pub mod oracle`, so `evaluate` on a `RowSort` plan was a reachable panic on
  valid input — the same class 0.6.0 removed from the plan gate, in a corner
  nobody had swept.

  **Still open: gather/scatter.** The entry used to say it *"needs a different
  notion of correct"* as though that were true of the whole of it. **Measured
  2026-09-02, it is true of exactly ONE cell**, and the plan gate had already
  done most of the narrowing:

  | write combine | dtypes the gate admits | order-independent? |
  |---|---|---|
  | `Assign` | any | **only when the indices are unique** — otherwise last-writer-wins with no defined order |
  | `AtomicAdd` | `i32`/`i64` | **yes** — integer addition is associative and commutative, wrapping included |
  | `AtomicAdd` | `f32`/`f64` | **NO. This is the whole open question.** |
  | `AtomicMax` / `AtomicMin` | **integer only** (`combine_legal_for_dtype`) | **yes** — and the float case that would have been genuinely subtle (`max(-0.0, +0.0)` is implementation-defined, so it could have been order-dependent) **is not admitted**, so it does not arise |

  **And gather is not part of the question at all.** It is a `ReadIndex` — an
  indexed *read*, no contention, fully deterministic, exactly checkable.

  **Recommendation for the one open cell, cheapest sound answer first:** do not
  weaken the check, **characterise the subset where the nondeterminism cannot
  manifest.** Float addition is exact — and therefore associative and
  order-independent — when every scattered value is an integer exactly
  representable in the type and every running partial stays inside the exact
  range (2^24 for `f32`, 2^53 for `f64`). A corpus built that way makes
  `AtomicAdd@f32/f64` bit-exact and needs no new notion of correct. Only if
  someone needs coverage *outside* that subset does a weaker invariant (an
  acceptance envelope over several accumulation orders) become necessary — and
  that is a separate, later question.

  ✅ **BUILT.** The PM routed the ask and it came back a yes with a **named,
  not-yet-built consumer**: `tools/kiss-ref-diff/main.rs` scopes *"gather/scatter
  with `IndexRef` are step 2c"*, and the converter already carries the `IndexMap`
  scaffolding. **That named consumer is the whole difference between this and the
  variant predicate** — which stayed unbuilt for exactly the reason this one did
  not.

  Gather lives inside `eval_elementwise` because the gate admits it nowhere else,
  and OOB is handled as a **store predicate** rather than a load behaviour (the
  policy docs are explicit that no OOB load occurs — the emitter clamps the
  address and guards the store). Scatter mirrors it on the store side, with the
  policy pinned to `Skip` by the gate.

  ⚠️ **The bug worth recording: a scatter iterates its SOURCE, not its
  destination.** Every other elementwise op produces one output element per
  iteration, so walking the output shape is the same walk — **a scatter's
  destination can be far smaller than its source** (bincount is the extreme), and
  walking the output silently drops everything past the first few elements. Found
  by asking what a bincount would do, not by a test; then pinned by one with
  deliberately unequal extents (6 → 2), and mutation-proven — the
  destination-shaped walk yields `[1, 2]` where the source-shaped one yields
  `[9, 12]`. **Every same-extent test passes under both.**

  ⚠️ **And one limitation of the oracle recorded rather than tested around:**
  `Skip` and `ZeroFill` are indistinguishable here, because `alloc_output` zeroes
  the destination and so "leave the cell alone" and "write zero" produce the same
  bytes. They differ on device whenever the caller pre-filled the buffer. **A test
  asserting they agree would be pinning the oracle's limitation as if it were the
  contract**, so there isn't one — the note is the artifact.

  The deterministic-subset recommendation is confirmed **proven rather than
  proposed**: baracuda's fold differentials already use exactly-representable
  integer-valued floats so any fold order gives identical bits, and
  `AtomicAdd` on the 2²⁴/2⁵³ subset is that same construction one op-class over.
  `scatter_add_f32_is_exact_on_the_integer_valued_corpus` is that corpus.

  Pinned by the exhaustive `Coverage` classifier in `oracle.rs`'s tests.
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
- ~~**Sweep for exhaustive checks over the wrong axis.**~~ — **DONE 2026-09-02.**
  Swept all 39 test files; 8 declare a corpus and therefore make a coverage claim.

  ⚠️ **The class was already known here and I did not know that.** The remedy is
  in the tree, written after **baracuda** caught `..._and_never_panics` claiming
  more than it checked: *"the loop iterates `ElementKind::ALL` with a fixed op, so
  'never panics' was measured on the dtype axis only."* **Rather than rename
  around the gap, it was pinned as an INEQUALITY** — a test asserting the op-axis
  panic *existed*, so the day it became a typed decline the test fails and forces
  a deliberate deletion. **So the finding is not a new class; it is that the
  remedy was applied once and never systematised.**

  **What the sweep changed:**

  - **`adversarial_input_panic_census`** — its dtype axis was a hand-listed 20
    while `ElementKind::ALL` has 25. Four omitted were the non-compute rows; **the
    fifth was `F32Strict`, a live compute dtype.** `KNOWN_PANICKING = 0` read as a
    statement about all input and was measured over 20 of 25. Now derived from
    `ALL`; surface 1920 → **2400 inputs, still 0 panicking**, declines
    1008 → 1428.
  - **`oracle_reachable_on_every_admitted_dtype`** — **I hand-listed 25 variants
    in a test written that same hour to close a coverage gap.** The list would not
    have grown with a 26th dtype. Now derives from `ALL`.

  **What the sweep found and deliberately did NOT change**, because a hand list is
  not a defect when no authoritative axis exists:

  - `CHANNELS` (9 output-channel names) — there is no enum of *ways to print*.
  - `VENDOR_MARKERS` (6 prefixes) — no authoritative list of vendor spellings.
  - `SPECIALS` (13 floats), `A` (7 u32s) — honestly corpora, and named as such.
  - `ADMISSION_API` (6 names) — ⚠️ **the interesting one: its axis is ANOTHER
    REPO'S API surface.** A rename in fuel silently disarms the guard, and no
    local mechanism can see it. Same shape as [[panic-text-is-a-cross-repo-api]]
    inverted: there the guard had to live on the side that could not run the
    tests; here it has to name a subject it cannot observe.

  **The transferable rule, which is narrower than "check your axis":**
  **derive a corpus from a guaranteed-complete source where one exists, and where
  none exists, say so in the guard.** `ElementKind::ALL` is complete by
  *mechanism* — the enum is exhaustive so a new variant breaks every match site,
  and `kiss_dtype_manifest` fails if a §6.1-token variant is missing from it.
  **Deriving inherits that guarantee; copying does not**, and the two look
  identical at the call site.

- ~~**External-axis sweep** — which guards depend on a fact this repo cannot
  observe?~~ — **DONE 2026-09-02.** Space: 39 test files + 22 production sources.
  Discriminator: **if the external fact changed, would anything here go red?**

  **Three outcomes, not two — and the third turned out to be the common one.**

  **1 · EXTERNAL WITH A MECHANISM (no action).** `kiss_byte_match` and
  `kiss_dtype_manifest` read **vendored** KISS artifacts and pin their **length
  and content**. Re-vendoring reds them, forcing a deliberate update. ⚠️ **Their
  own comment makes the distinction that matters:** *"this is the check
  `source_commit` cannot perform, because that field records the spec commit and
  does not move when the artifact is regenerated."* **Pinning the provenance field
  would have been a false mechanism.** This is the mechanism worth copying.

  **2 · EXTERNAL, DOMINATED (annotate, do not remedy).** `ADMISSION_API`. Fuel
  measured that its six names are **private internals** — `mod jit_ingest` is not
  `pub mod`, `adopt_verified` is not `pub fn`, zero re-exports — of a crate this
  workspace does not depend on. **So compiled code cannot name them without first
  declaring the dependency the guard's OTHER axis already checks.** The manifest
  scan strictly dominates; the name list adds coverage only for non-compiled
  mentions.

  ⚠️ **A guard with two axes where one dominates reads as two checks and is one.
  The dominated axis costs maintenance and its rot is INVISIBLE, because the
  surviving axis keeps the test green.** That is a different finding from *"this
  guard will silently stop checking"* — and it needed the other repo's
  **visibility modifiers**, which are exactly what cannot be seen from here.

  **3 · EXTERNAL WITHOUT A MECHANISM — one, and it is small.**
  `KISS-CLASSIFY-6.8-0002`, asserted by `operand_facts_reach_the_key.rs` and
  covered by **no vendored vector**. Of 6 clause IDs asserted across the vocab
  tests, 5 appear in the 37 vendored vectors and this one does not. **Remedy is
  cheap and not local: ask KISS for a vector exercising it**, which converts a
  paraphrase into the same content-pinned mechanism the other five already have.

  ⚠️ **Three self-inflicted measurement errors during this sweep, all the same
  shape — a label that does not depend on what it labels.** An `echo "(empty…)"`
  that printed unconditionally over a six-hit result; a `grep -E "a\|b"` that
  searched for a literal pipe and returned a false zero; and guessing JSON key
  names and reading the miss as "0 vectors" when the file had 37 under different
  keys. **Each produced a plausible number from a query that had not asked the
  question**, which is the class this whole sweep is about, committed three times
  by the sweeper inside one hour.
- **Per-target N2 verification.** NaN-propagation surviving the toolchain is a
  property of *one optimizer*, re-measured per target, never inherited. Verified
  on portable C (`/O2`) and CUDA (nvrtc→PTX→driver JIT, RTX 4070). Any new target
  needs its own; a source-text golden is structurally blind to it.
- **Hardware-aware candidate generation.** Unpopped **chooses nothing** today:
  the caller supplies the whole `StructureKey`, `vec_width` included, and
  `plan.rs` only honours it. There is no hardware-resource module in the tree.

  The unblocked increment is a `TargetCapabilities` (per-compute-capability
  static limits, plus per-device facts queried at runtime — `baracuda_driver`'s
  `Device::attribute` is public and generic over `CUdevice_attribute`, so the
  whole table is reachable) and a `candidates()` returning a **small set of legal
  `StructureKey`s** worth racing. **Candidates as key variations, not as tile
  parameters** — `vec_width`, `WorkClass` and `IdxWidth` are already carried
  end-to-end and honoured by every emitter, so this needs no new seam and is
  independent of the schedule-language question in C.

  **It has a consumer on day one**, which is the test it must pass: Fuel's Judge
  races siblings at one `structure_key`, distinguished by `kernel_revision_hash`.
  Bound the count — **4–6 survivors per cell, not 40** — by filtering
  spec-legal, then by `ptxas -v` (registers/spills) *before* anything reaches the
  race. Judge cost is ops × dtypes × sizes × devices × **siblings**, and that last
  factor is ~1 today.

  **Precondition:** Fuel's GAP-244 — `"unpopped"` is not in `kernel_source_intern`,
  and an unknown tag interns to `""`, which collapses our candidates into
  `portable-cpu`'s cell rather than merging them with each other.

  **Increment 1 landed:** `crate::capability` — `TargetCapabilities`, the CUDA
  per-compute-capability table, and `capabilities_for(TargetId)`. An unknown
  capability returns `None` rather than a neighbouring row, and a `vulkan:` token
  is refused rather than answered from the CUDA table.

  ⚠️ **The design in this entry was WRONG about where candidates live, and the
  correction is measured.** "Candidates as `StructureKey` variations" is **not
  expressible**: every field of a key is *derived* from `(op, operands, target)` —
  `idx` from the largest offset, `work` from `frame_work_class`, `vec_width` from
  the operand's own alignment and extent via `classify_vec_width`. **A key is a
  pure function of the request, so two candidates for one request cannot differ
  in it.** That is the identity property working correctly, and it means
  variation belongs to **`Backend::lower_variants`** — which already exists,
  already returns `Vec<Variant>` with a `tag` and a `VariantFidelity`, and is
  already implemented by baracuda's CUDA backend. Capabilities inform *which
  variants are worth emitting*, not which key to build.

  **Also found while measuring, and not fixed here:** `classify_vec_width` caps
  vector accesses at a hardcoded `vbytes <= 16` in `unpopped-vocab`
  (`structure_key.rs:1282`). That is CUDA's `float4` limit living in the neutral
  vocabulary — the same shape as the `backend.rs:1199` declaration leak, one
  crate over, and it is what `TargetCapabilities::max_vector_bytes` should
  eventually feed.

  ⚠️ **Scoped 2026-09-03, and it is NOT the solo unblocked fix it reads as.**
  `classify_vec_width` feeds `structure_key`, so changing it changes **normative
  key derivation** — the identity property, cache keys, and any corpus keyed by
  them. Measured over the rule itself (`align = 64`, `ext = 256`):

  | dtype | today (cap 16) | uncapped | per-target cap 64 |
  |---|---|---|---|
  | `f32` | V4 | **V8** | **V8** |
  | `f64` | V2 | **V8** | **V8** |
  | `f16` | V8 | V8 | V8 |
  | `i8` | V8 | V8 | V8 |

  **Removing the cap changes CUDA keys too** — 3 of 5 sampled rows move, and
  `cuda:` is exactly where 16 is the *correct* limit. So "delete the vendor
  constant from the neutral crate" is not a neutrality fix; it would make the
  vocabulary wrong for the one target it is currently right for.

  **Three designs, different blast radii, and this is a decision rather than an
  implementation:**

  1. **Per-target cap.** Only non-CUDA keys move. Needs a target→`max_vector_bytes`
     table *in vocab* — and vocab is BELOW `unpopped`, so it cannot read
     `TargetCapabilities`. `structure_key(op, operands, target)` does take the
     target, so the fact is derivable there; where the table lives is the open
     part.
  2. **No cap in vocab; the backend caps.** Cleanest neutrality — *the vocabulary
     says what the DATA permits, the backend says what the DEVICE permits* — and
     the 16 conflates those two. **But it moves CUDA keys**, so it is a
     coordinated change on baracuda's corpus, i.e. section B in all but location.
  3. **Leave it, document it as target-conditional.** Costs nothing today because
     both in-tree emitters are `Schedule::Scalar` only.

  ⚠️ **AND THE UNIT IS WRONG, WHICH RETIRES ALL THREE OPTIONS ABOVE AS STATED.**
  Vulkane measured Vulkan's actual constraint in `vk.xml` 2026-09-03 rather than
  recalling it:

  ```
  vk.xml:11351  VkPhysicalDeviceShaderLongVectorPropertiesEXT
                  maxVectorComponents : uint32_t   limittype="max"
  vk.xml:30792  extension VK_EXT_shader_long_vector (636, device, EXT)
  vk.xml:33729  VkPhysicalDeviceShaderLongVectorFeaturesEXT.longVector
  ```

  **A byte cap is wrong on three axes at once:**

  1. **Wrong unit.** `maxVectorComponents` is a **component count**. Bytes =
     count × element size, and Vulkan has 8/16/32/64-bit components, so **no
     single byte number expresses it** — a 16-byte cap admits `vec4<f32>` and
     wrongly rejects `vec4<f64>`.
  2. **Wrong lifetime.** Runtime-queried per physical device via
     `vkGetPhysicalDeviceProperties2`. A vocabulary compiled ahead of time
     cannot know it.
  3. **Wrong scope.** Per-**device**, not per-target. Two Vulkan devices on one
     machine can differ.

  **So the parameter is `(component_count, element_type)`, derived to bytes at
  the call site where the element type is known — not a byte cap.** That also
  expresses CUDA's limit without privileging it: `float4` is *"4 components of 4
  bytes"*, which is a fact about CUDA rather than a constant in the vocabulary.

  ⚠️ **If anyone wires a real device query to feed this, gate it on BOTH the
  extension AND the API version.** An ungated `pNext` property read on a device
  lacking the extension **reads back zeroed and looks like an answer** —
  `maxVectorComponents: 0` means *"no vectors at all"*, a plausible number and a
  false one. (Vulkane's scar tissue, not ours; their property queries gate on
  `min(instance, device)` version and check extension presence rather than
  trusting a populated struct.)

  **PARTY CORRECTION — the urgency I attached to this was mine and it was never
  measured.** I told the PM this was *"wrong for the first non-CUDA backend that
  vectorises — vulkane, on day one"*, and wrote **"Party: vulkane (holds the
  non-CUDA corpus)"** into this file. **Vulkane measured their own tree at
  `origin/main` `8425770`: zero `unpopped` in any `Cargo.toml`, zero SPIR-V
  emission** (`OpTypeVector|rspirv|spirv_headers` = 0; control: `"SPIR-V"`
  appears 60 times in prose), **and all 13 `structure_key`/`vbytes` hits are doc
  comments about KISS's concept.** They hand shader source to `naga` / `shaderc`
  / `slang` and receive words back, **so for a Vulkan target the vector-width
  decision belongs to those compilers and vulkane makes no such choice.**

  **There is no day-one consumer. The item stands; the deadline does not.**
  Vulkane holds no corpus keyed by our `structure_key`, so option 2's blast
  radius is **baracuda alone**, not baracuda-and-vulkane.

  **Not startable solo** — it was listed as unblocked on 2026-09-03 and that was
  wrong; the blast-radius table corrected the design, and vulkane's registry read
  corrected the unit and the party.

  ⚠️ **Increment 2 has NO IN-TREE SUBJECT, measured 2026-09-02 before building
  it.** A variant needs an axis to vary, and both in-tree emitters serve
  `Schedule::Scalar` only — there is no second thing for CpuC or Slang to emit.
  The real variant axis is **algorithmic**, not tiling: `plan.rs` names
  `cuda::scan_blockscan_variant` and `cuda::row_sort_bitonic_variant` as
  `lower_variants` filters, and those are baracuda's. `unpopped/src/lib.rs:249`
  does call `lower_variants`, so the seam is wired — it is the *producers* that
  are all out of tree.

  **So the next real increment is a capability-aware legality predicate** (is a
  block-cooperative variant viable for this plan on this target — block size,
  shared memory, warp width), whose only consumer today is baracuda. **Not built
  speculatively:** an API with no in-tree caller is the exact shape this
  workspace keeps finding defects in, and building one *for* an adopter who has
  not asked is worse than waiting. **Ask baracuda whether they want it before
  writing it.**

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

---

**A test count that does not reconcile is the cheapest defect detector here.**
Found 2026-09-02 during a vacuous-pass sweep, by an arithmetic mismatch and
nothing cleverer: **734 `#[test]` functions existed in this workspace and 712
ran.** The suite was green, every gate passed, and no output anywhere said that
22 tests had not executed.

The cause was two optional features. `convert` (20 tests) and `seam` are off by
default, and `verify.yml` explicitly excluded `--all-features` — the exclusion
was *documented*, which is why it survived: it read as a considered decision
rather than a hole. What the note did not say is what it cost.

⚠️ **The expensive half was not the 20 unrun tests. It was that
`#[cfg(feature = "seam")] pub mod seam` — the entry point Fuel and Baracuda
actually call — was never COMPILED by CI at all.** Not linted, not type-checked,
zero tests. A breaking change to its frozen grammar dep, or a refactor of
`region_to_op`, would have gone green here and failed in a consumer's tree, where
it costs *them* a debugging session to find out it was ours. **The least-verified
code in the repository was the cross-repo seam**, and the default-features
convention is what put it there.

**An optional feature is not optional to the consumer who enables it.** Fixed by
an `all-features` CI job (clippy + test) and `tests/seam_reaches_core_synthesis.rs`
— 5 tests covering the grammar walk, its recursion, and its three typed declines.

Two transferable pieces:

- **`cargo test` reports what it RAN, never what it SKIPPED COMPILING.** A
  feature-gated test contributes zero to both the numerator and the denominator,
  so no ratio anywhere goes down. Count `#[test]` on disk and compare it to the
  count that executed; the gap is the whole finding, and it needs no judgment to
  read. Reconciling it here also turned up a `#[cfg(feature = "convert")]` test
  inside `unpopped-slang` that nothing had ever run.
- **A documented exclusion still has to be re-priced when what it excludes
  changes.** The `--all-features` note was accurate when written and the `seam`
  module landed behind it later. Nothing re-read the exclusion in light of the
  new module, because an exclusion with a stated reason stops looking like a gap.
