# What Unpopped DECIDES that KISS might intend to own

**Volunteered, not requested.** The seam tables being assembled are built from
queries against each tree, and every such query so far has been **citation-shaped**
— it finds the clause numbers a repo names. ⚠️ **A grep for `KISS-` cannot see a
function that encodes a normative rule and never mentions the standard**, and this
crate's strongest seam is exactly that: `is_bit_or_sign_move` decides a §6.16
question, in Unpopped's core, for consumers in other repositories, and names no
clause in its signature.

**Written 2026-09-05 against `origin/main` at the ref each row states.** The point
is not to claim KISS *should* own these. It is to put the decisions where someone
who owns that question can see them.

**Every row is labelled `MEASURED`, `REPORTED` (someone told me, I did not verify),
or `JUDGED` (my reading, not a measurement).** ⚠️ A row's label matters more than
its content: an unlabelled list invites a reader to treat my judgement as a
finding.

---

## 1 · The move/compute boundary — the strongest row, and the one least visible from outside

| | |
|---|---|
| **What we decide** | Whether an op MOVES bits (KISS-OPS-§6.16-0009, bits preserved exactly) or COMPUTES (§6.16-0010, MUST quiet a signalling NaN) |
| **Where** | `unpopped::ir::{is_bit_move, is_bit_or_sign_move, is_bit_move_reduce, is_bit_move_fold_output}` |
| **Status** | **MEASURED.** Public API of `unpopped` 0.8.6. |

**This is a normative predicate living in a dependency.** A backend that routes on
it inherits our reading of §6.16 without ever citing it.

**In-tree consumer:** `unpopped-cpu-c/src/lib.rs:278` (`is_bit_or_sign_move`).
**Out-of-tree:** baracuda's CUDA emitter — **REPORTED by them and by the PM; I have
not measured their tree.**

⚠️ **`is_bit_move_reduce` has ZERO consumers, in-tree or out, as of this writing.**
It exists because baracuda asked for the shape rather than writing a second copy of
a normative predicate in a backend, and the KISS architect ruled the truth
(#416: `Max`/`Min` fold → move; `Sum`/`Prod`/`Mean` → computed). **An API with no
in-tree caller is a shape this workspace has been burned by, and it is recorded
here rather than hidden.**

### The history is the argument for the row

- **August:** `is_bit_move` covered `Input`/`Max`/`Min`/`Select` — the no-sign-edit
  subset. **Correct, and short.** `Neg`/`Abs`/`Copysign` fell through to the
  arithmetic path and shipped broken in `unpopped-cpu-c` 0.6.0.
- **2026-09-05:** fixed, published as 0.7.0. kiss-ref found **the same three ops,
  same mechanism, different dtype, hours apart, with no coordination** — which is
  what makes this structural rather than one lane's slip.
- **The same day:** baracuda found the predicate's *subject* is wrong at a
  reduction site. ⚠️ **I then got the correction wrong twice more, and the second
  time was in THIS FILE** — see below.
- **Same day, third round:** the element expression lives in **opposite fields per
  `Access` shape**. `Access::Reduction` puts it in `plan.body` (its `post` is
  separate); `Access::RowReduce` puts the *epilogue* in `plan.body` and the
  elements in `stages[i].pre`. **There is no single safe field**, and every
  statement of the form "`plan.body` is safe" is true of one shape and false of
  the other.
- **Same day, fourth round:** §6.16-0009 attaches to the **observable output**,
  not the fold — so a *moving* post (`Neg(Reduced(0))`) keeps -0009 while an
  arithmetic one (`Sqrt(Reduced(0))`) does not. **`is_bit_or_sign_move` could not
  express that at all**: an epilogue's leaf is `Reduced`, which scores `false`, so
  the case was inexpressible rather than unimplemented. Closed by
  `is_bit_move_fold_output(fold, element, post)`.

  ⚠️ **The obvious fix was rejected as dangerous:** admitting `Reduced` as a leaf
  in the *public* predicate would make a bare identity epilogue read `true`, and a
  caller checking only the epilogue would then route an identity post over a
  **Sum** fold as a move. **So the leaf policy lives in a function whose signature
  cannot be satisfied without naming the fold** — structural safety rather than a
  documented warning.

⚠️ **AND THE ROUNDS ARE THE ROW'S REAL CONTENT.** Four corrections in one day on
one predicate, and **the code was wrong only in the first.** Rounds 2–4 were
*prose and expressiveness*: a claim true of one constructor stated unqualified, the
same claim surviving in a second doc after the first was fixed, and a case the
vocabulary could not name. **A seam is not just what a predicate decides — it is
what its documentation causes consumers to believe**, and this file was itself
round 4's carrier until 0.8.6.

**KISS-side question, if there is one:** whether the move/compute boundary is a
property KISS states and implementations derive, or one each implementation
derives independently. **Today it is the second, and three repos have derived it
from the same clause and reached three different coverages.**

---

## 2 · `vec_width` — a vendor constant deciding a field of a normative token

| | |
|---|---|
| **What we decide** | The `vec_width` field of the KISS-Classify `structure_key` token |
| **Where** | `unpopped-vocab/src/structure_key.rs:1282`, `classify_vec_width` (**private**) |
| **Status** | **MEASURED** |

```rust
if vbytes <= 16 && align % vbytes == 0 && ext % v == 0 {
```

⚠️ **`16` is CUDA's `float4` type limit, hardcoded in the vendor-neutral
vocabulary**, and it decides a field of a token that is byte-matched against KISS
vectors.

**Measured by vulkane 2026-09-05 on hardware, and it retires the premise the number
rests on:** one machine, two physical devices — AMD Radeon 610M reports
`VK_EXT_shader_long_vector` **absent**; NVIDIA RTX 4070 reports
`maxVectorComponents` **= 1024**. ⚠️ **The same vendor whose `float4` is where our
16 comes from reports 1024 components through Vulkan, so 16 was never a hardware
limit — it is a CUDA vector-TYPE limit.**

**JUDGED:** it is a *language* fact in the wrong crate, not a *device* fact in the
wrong crate — which changes the fix. Also measured by vulkane: the constraint is a
**component count**, runtime-queried **per physical device**, so **no static
per-target byte number is expressible at all.**

**KISS-side question:** whether `vec_width`'s derivation is normative (KISS says
how to compute it) or free (KISS pins only the spelling). **We currently derive it,
and our derivation carries one vendor's type-system limit.**

---

## 3 · What "conformant output" means numerically

| | |
|---|---|
| **What we decide** | The comparison band a cell is entitled to, and what counts as equal |
| **Where** | `unpopped::oracle::{required_fidelity, Fidelity}`, `unpopped::contract::ulp_bound` |
| **Status** | **MEASURED** |

`required_fidelity(plan, operands)` derives, from the body's structure, the band a
**correct** implementation must land in. Ownership was open for 19 days and is
settled: **OURS** — fuel measured **zero occurrences** of the name in any spelling
at `1fb2e9db`, and their `fkc/verify/` layer is supply-side ("has this kernel
EARNED its claim?"). **The deciding evidence is structural: the input is
`plan.body`, an Unpopped IR node fuel does not have.**

⚠️ **Two live constraints inside it:**

- **`ulp_bound` sums a CUDA per-op table and takes no target**, so the band is
  CUDA's accuracy **for every backend**. Vulkan's `exp` is 3 ULP against CUDA
  `expf`'s 2 → **a conforming Vulkan kernel fails a comparison it should pass.**
  Latent today (only caller is a CpuC test).
- **REPORTED by vulkane, labelled by them as hypothesis not measurement:** Vulkan
  states precision requirements **per instruction AND per precision mode**, some
  ops *correctly rounded* with no ULP bound at all. **`ulp_bound` is one scalar per
  op.** If that holds, the shape is wrong before the values matter.
- **MEASURED by vulkane:** the numbers exist in **no machine-readable source** —
  `vk.xml` and `vulkaninfo` carry zero hits for ULP/accuracy (control:
  `maxImageDimension2D` = 1), and the SDK ships no spec document.

**`Fidelity::Tolerant` also decides what EQUAL means:** both-NaN compares equal
**payload-agnostically**, and ±0 compare equal (`oracle.rs:3119`). **JUDGED:** that
is a conformance-semantics decision, not an implementation detail, and it sits
beside §6.16-0009's requirement that a *moved* NaN keep its exact bits — the two
answer different questions and a reader could easily think they conflict.

---

## 4 · Which dtypes are COMPUTABLE, as distinct from spellable

| | |
|---|---|
| **What we decide** | That four §6.1 dtypes are storage/scale-only and decline at the plan gate |
| **Where** | `unpopped::plan`, `check_dtype_is_a_compute_dtype` (`plan.rs:1728`) |
| **Status** | **MEASURED** |

`Fp8E4M3FNUZ` / `Fp8E5M2FNUZ` decline as **reserved**; `F8E8M0` / `F8E6M2` decline
as **MX block scales**. **KISS §6.1 gives all four tokens** — recognition and
usability are already distinguished there (24 recognized, 22 usable) — **but
"usable" and "has arithmetic semantics" are our reading, not a clause we cite.**

---

## 5 · fp8 codec behaviour at the boundaries

| | |
|---|---|
| **What we decide** | E5M2 overflow → **infinity**; E4M3FN overflow → **saturate to 448**; a moved NaN keeps sign and payload |
| **Where** | `unpopped/src/cfamily.rs` (emitted codecs), `unpopped/src/oracle.rs` (reference) |
| **Status** | **MEASURED**, fixed 2026-09-05 |

**Opposite rules for the two formats, deliberately:** `e5m2` is IEEE-shaped and HAS
infinities, so overflow produces one; `e4m3fn` defines none, so saturation is
correct. ⚠️ **A single rule for "FP8 overflow" would read as coverage while
asserting one of them wrong.**

**Related, and it belongs on the table because it crosses a repo boundary:** the
upstream `float8` crate 0.7.0 has `F8E5M2::INFINITY = 0x7B` (max-finite) and
`MAX` one encoding low on **both** types. Filed as `EricLBuehler/float8#14`.
**We work around it locally; the constants are theirs.**

---

## 6 · Panic text as a cross-repo API

| | |
|---|---|
| **What we decide** | The exact strings our declines and panics carry |
| **Status** | **REPORTED** (recorded in this workspace's memory; not re-measured for this document) |

Adopter tests match our panic/decline strings. **JUDGED:** that makes message text
a compatibility surface whether or not anyone intended it, and the guard has to
live on the side that cannot run the adopter's tests.

---

## 7 · `2*pad <= span` is a CROSS-EMITTER contract, not an internal validation

| | |
|---|---|
| **What we decide** | That every window overlaps the input by ≥1 tap — so `Max`/`Min` folds **peel the first element** and no `±inf` identity is ever materialised |
| **Where** | `unpopped::plan` `plan.rs:3399`/`3405`; `ReduceOp`'s doc states the consequence |
| **Status** | **MEASURED**, here and independently by baracuda |

```rust
assert!(2 * u32::from(pad_lo) <= span, "Window pad_lo {pad_lo} exceeds half the window span {span}");
```

Pad taps are **SKIPPED** for `Max`/`Min` (*"padding never wins"*), so this
assertion is what guarantees a fold always has a real element — **the peel is not
an optimisation, it rests on this.**

### ⚠️ I FIRST WROTE THIS ROW AS A DIVERGENCE. THERE IS NO DIVERGENCE.

**The first version said one of two models admits a shape the other forbids, and
offered two horns.** ⚠️ **Both horns assumed two independent gates. There is ONE,
and it is in the crate baracuda pins — so it governs their CUDA emitter too.**

**They measured it rather than reading it:** `window_simple(size=3, pad_lo=3)`
panics on this assertion; a legal shape builds (their control). **Their all-pad
branch is unreachable, exactly as ours is.**

**And their `narrow_extreme_lit` was never an all-pad guard** — it is a **type**
requirement: the literal initialises an `{acc}`-typed register, and the generic
`scan_identity` yields a `float` ±inf bit-cast, wrong once `acc` is `__half`.
**The code was right and the stated reason was wrong**, in a comment they had
written hours after relaying that exact rule to three lanes.

⚠️ **My error here is the one worth keeping: I built a two-horned seam question
out of a peer's stated REASON without checking whether the two systems were
actually independent.** **They share a dependency — mine — and one grep of my own
`Cargo.toml` consumers would have shown it.**

### The coupling, which is the real content

**Relaxing `2*pad <= span` un-deadens code in a repository this one cannot see.**
⚠️ **Nothing in baracuda's tests would notice their dead arm going live**, and
until 2026-09-06 **this gate had no test on our side either** — a relaxation was a
one-line edit with no signal anywhere.

**Now pinned:** `the_window_pad_gate_is_a_cross_emitter_contract.rs`, with a
control so the `should_panic` cannot pass on an unrelated assertion, and a header
naming who goes live if it is relaxed. **Their standing ask: ping in the same
window as the change, not after.**

**Still deliberately NOT in core: an identity table.** Encoding a constant for a
case the gate forbids would invite a consumer to read its presence as permission.
⚠️ **And the argument got stronger, not weaker:** the consumer being protected is
baracuda, and **they had already made the error the constant would have licensed
— not by reading a table of ours, but by writing their own and justifying it with
a shape this gate forbids.** **The artefact would not have created the
misconception; it would have CONFIRMED one already held**, which is the more
dangerous case.

## What I did NOT check

**State this loudly, because a seam list is the kind of document a reader widens
silently.**

- **KISS-Contract: I bind ZERO clauses as a producer.** MEASURED with a positive
  control — 0 of the 7 section headers emitted, while `cost:` (an FKC key I do
  emit) greps 1. `crates/unpopped/src/contract.rs` emits an **FKC**, citing Fuel's
  §4.4 symbol vocabulary. ⚠️ **That nil is about Contract specifically and is NOT a
  statement that Unpopped is outside the regulated surface** — rows 1–6 above are
  the evidence it is not.
- **I read the KISS-Contract clauses my FKC mapping touches, plus the §6.6/§6.11
  expression clauses — not all 112.** A vacuous trigger elsewhere would not have
  crossed my path.
- **I have not measured any consumer's tree.** baracuda's use of
  `is_bit_or_sign_move` is REPORTED. I can see my own re-exports; I cannot see
  theirs.
- ~~**I did not audit `unpopped-slang` separately.**~~ **Audited 2026-09-05 after
  baracuda enumerated FOUR defective §6.16-0009 lowering surfaces in their own
  emitter and sent the mechanism.** MEASURED through `supports_dtype`, with an
  `F32` control so the falses are real:

  | | F16 | Bf16 | fp8 | F32 (control) |
  |---|---|---|---|---|
  | `unpopped-slang` | false | false | **false** | true |
  | `unpopped-cpu-c` | false | false | **true** | true |

  **slang serves no narrow float at all**, so the §6.16 rows do not reach it —
  and it has **zero** `bit_move` references. ⚠️ **That count alone is
  indistinguishable from "the surface exists and is unhandled"**; only the dtype
  measurement separates them, and baracuda flagged that asymmetry rather than me.

  **cpu-c serves fp8 only, and the guard's dtype range equals the served range**
  (`Fp8E4M3FN`, `Fp8E5M2`, plus `F32` as a wide control) — the trap baracuda hit,
  where a live and correctly-aimed guard stayed green because its dtype
  population excluded exactly the buggy dtypes.

  **Population taken from each emitter's own `Schedule::` dispatch, not a grep:**
  one arm each, everything else a named typed decline. ⚠️ **A grep's completeness
  is unknowable from inside the grep.**

  ⚠️ **AND THE EXPIRY IS ENFORCED RATHER THAN WRITTEN.** This nil is about
  **today's dispatch**, not a property of the crates.
  `unpopped-conformance/tests/one_lowering_path_per_emitter.rs` asserts each
  emitter still dispatches exactly one `Schedule` arm and fails with what to do —
  sweep the new path for an accumulator register whose type was chosen for
  arithmetic convenience, then update this nil. It carries a control, so a
  renamed enum reds instead of passing by measuring nothing. **baracuda's
  prescription: a nil about a dispatch decays silently; a test that asserts the
  arm count fails loudly.**

  **JUDGED, not measured:** the sub-byte and complex sub-paths inside that one arm
  were reasoned out (sub-byte types are integers — no NaN, no rounding; complex is
  not `narrow`, so it never reaches the move path) rather than emitted.
- **I have not asked whether KISS WANTS any of these.** ⚠️ Listing a decision here
  is not a request that KISS adopt it. Several are probably correctly ours; row 3's
  ownership was already ruled ours after being asked.

## The shape of the nil

**JUDGED, not measured:** I believe rows 1 and 2 are the ones a KISS-side query
could not have found — row 1 because the predicate names no clause, row 2 because
the constant is private and numeric. **The others cite clauses somewhere nearby and
a citation-shaped sweep would plausibly surface them.** That belief is why this
document leads with those two, and it is a judgement about someone else's query,
which is the weakest kind of claim in here.
