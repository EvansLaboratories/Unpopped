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

## Unreleased

*Nothing. Everything that was here shipped in 0.11.0, below.*

## Released

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
