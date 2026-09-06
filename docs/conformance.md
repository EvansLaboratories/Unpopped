# Backend conformance: what the oracle actually requires

**Status: classification, not yet a ratified contract.** The NORMATIVE and
INCIDENTAL sections below are a faithful translation of behavior already
implemented and tested in `crates/unpopped/src/oracle.rs` — they change nothing,
they only restate obligations that were written down as observations about the
CUDA emitter. The OPEN section lists questions that are genuinely undecided and
need a ruling before this becomes a contract; they are not translations and I
have not answered them here.

## Why this document exists

Unpopped's CPU oracle is an independent interpreter that recomputes what a
kernel should produce, from the same plan, via a separate code path. It is the
mechanism by which a backend is judged correct.

Its semantics, however, are stated throughout as *"the emitter's inner loop
order"*, *"mirroring the emitter"*, *"the emitter tests `((float)(c)) != 0.0f`"*.
That was accurate and sufficient while CUDA was the only backend: there was one
emitter, and "what it does" and "what is required" were the same sentence.

They are no longer the same sentence. A second backend author reading the oracle
cannot tell which of its behaviors they must reproduce and which are artifacts
of CUDA having been first. Both kinds of statement look identical in the source.
Getting that distinction wrong is expensive in both directions — reproducing an
incidental choice wastes effort and blocks legitimate optimizations, while
missing a normative one produces silently wrong numerics that the tolerant
comparator will happily accept.

The distinguishing question used throughout: **would a backend that did this
differently produce an answer the oracle rejects?** If yes, it is normative. If
the tolerant comparator absorbs the difference, it is incidental.

---

## NORMATIVE — a conforming backend MUST reproduce these

### N1. Discontinuities are decided in the compute dtype

`Cmp*` (all comparisons), `Sign`, `Step`, and the condition of `Select` must be
evaluated on operands **rounded to the kernel's compute dtype**, not in wider
precision.

This is the most important rule in the document and the easiest to miss, because
it is the one place where "compute more accurately" is *wrong*. A tolerance
cannot absorb a discontinuity. Decide `x > 0` in f64 where the kernel decides in
f32 and you do not get a slightly different number — you get the other branch,
and the output is wrong by however far apart the two arms are.

The oracle implements this explicitly (`round_to_compute` before the comparison
in `eval_binary`, before `Sign`/`Step` in `eval`, and on the `Select` condition),
*deliberately breaking* its own "accumulate in f64" posture to do so. Continuous
ops (`Max`/`Min`/`Pow`/`Atan2`/`Rem`, the transcendentals) stay in accurate f64
and rely on tolerance — a flipped near-tie there returns a within-tolerance
operand, so it is harmless.

### N2. `Max` / `Min` propagate NaN

If either operand is NaN, the result is NaN. This applies to the binary ops and
to `Max`/`Min` reductions, where a NaN anywhere in the reduced extent makes the
result NaN.

**This is deliberately not IEEE-754 `maxNum` / C `fmax` semantics**, which return
the non-NaN operand. It is the trap most likely to catch a new backend, because
every C-family target offers a convenient built-in with the *other* behavior, and
the difference is invisible on all NaN-free input.

The deliberateness is settled rather than inferred: the IR carries
**`BinaryOp::FmaxIeee` / `FminIeee` as separate ops** for exactly the
NaN-suppressing semantics, and those are the only ops that lower to
`fmaxf`/`fminf`. `Max`/`Min` emit the compare-select and match
`torch.maximum`/`minimum`. So a backend that collapses the two is not making a
defensible precision trade — it is merging two distinct ops in the vocabulary.

The reference CUDA emitter does not use `fmaxf`. It spells the ternary out:

```c
(a != a ? a : (b != b ? b : (a >= b ? a : b)))
```

A SPIR-V/GLSL backend must not lower this to `max()`. Two things below that are
invisible from where this rule is written, both found from the Vulkan side and
verified here against sources:

**(a) The obvious "safe" opcode is not safe.** GLSL.std.450 `FMax` is
**undefined** when either operand is NaN — not NaN-suppressing, *undefined* —
so it fails this rule non-reproducibly across drivers, which is worse than
failing it deterministically. `NMax`/`NMin` are the explicitly NaN-suppressing
variants and are equally wrong. And a backend routing through a shader frontend
does not get to choose: naga 29.0.1 lowers float `max` to `GlslStd450Op::FMax`
(`src/back/spv/block.rs:1313`) and subgroup max to `OpGroupNonUniformFMax`
(`src/back/spv/subgroup.rs:82`). So a WGSL/GLSL → naga → SPIR-V path violates
this rule today regardless of what the backend author writes.

**(b) Emitting the longhand ternary at source level may not survive — now
measured on both reference targets, not assumed.** The rule is implemented by
writing the ternary and assuming no compiler in the chain contracts it back into
a max instruction. Every golden in this repo compares *source text*, which is
exactly the layer at which the ternary is still present, so the goldens are
structurally blind to the assumption they rest on.

Both reference targets have now been checked by compiling and running:

| target | chain | result |
|---|---|---|
| portable C | host `cl` at `/O2` | NaN propagates — `tests/cpu_end_to_end.rs` |
| CUDA | nvrtc → PTX → driver JIT at `-O3` | NaN propagates — verified on RTX 4070, 2026-08-08 |

The CUDA leg matters because PTX `max.f32` *is* NaN-suppressing without the
`.NaN` modifier, so the contraction was a live possibility rather than a
theoretical one. It does not occur. Baracuda holds that check as a standing
guard (`baracuda-kernels-bench/tests/max_nan_propagation.rs`), and it migrates
into the `unpopped-cuda` suite with the emitter.

**Any new target needs its own.** This is not a property that transfers: it is a
statement about one toolchain's optimizer, and it must be re-measured per target
rather than inherited. A source-text golden cannot carry it.

Two harness notes worth reusing, one from each leg. The never-wrote sentinel
**must be finite** when NaN is the expected output — NaN cannot serve as both the
answer and the not-written marker (CUDA leg used `-777.0`; the C leg instead
mixes NaN and finite lanes in one kernel). And assert the emitted source contains
no `fmaxf` *first*, so the run is known to be testing the ternary rather than the
intrinsic.

**Achievability is target-conditional — see OPEN-4.** On Vulkan this rule may be
unsatisfiable *by any lowering*, so it is a requirement a backend must meet where
it can and declare where it cannot.

### N3. Integer arithmetic wraps two's-complement at the C promotion width

`Add`/`Sub`/`Mul` on integer compute dtypes wrap at the dtype's C promotion
width, not the storage width, and never saturate. The oracle computes this
exactly (`wrap_bits(r, op_width(dtype))`) and compares bit-exactly, so a
saturating or differently-widening backend fails outright. This is the IR's
definition of the op, not a CUDA artifact.

### N4. `Cmp*` yields exactly 1.0 / 0.0

Not "some true value". The exactness is load-bearing: a U8 mask output stores the
predicate directly, and a `Select` may test the same value.

### N5. `Select` moves the chosen arm verbatim

The chosen arm is copied bit-for-bit. `-0.0`, NaN payloads, and signaling NaNs
survive; no arithmetic may touch an arm. A backend that implements `Select` as
arithmetic blending (`c*a + (1-c)*b`) is non-conformant even where the values
agree numerically.

### N6. `F32Strict` steps and stores on the f32 lattice

`Nextafter` under `F32Strict` uses the f32 lattice, not f64. This is the entire
purpose of `F32Strict` existing as a dtype distinct from `F32`.

### N7. Reduction scope refusals are part of the contract

Integer reductions are `I32`/`I64` only, and integer `Mean` is rejected. A
backend must **refuse** these rather than emit something plausible. The oracle
panics rather than returning an undivided sum, deliberately, so that an
out-of-scope plan cannot quietly acquire a wrong meaning.

---

## INCIDENTAL — currently mirrored from CUDA, NOT required

### I1. Accumulation order

The oracle sums reductions and contractions in ascending index order (K ascending
for matmul) and its doc says *"the SAME accumulation ORDER"*. That phrase
describes how the oracle is written; it is **not** an obligation on backends.

The oracle accumulates in `f64` and the comparator is `Tolerant`, so a different
association passes. A split-K or tree-reduction backend does not accumulate in
ascending order and is conformant. See OPEN-2 for the one case where this may not
hold.

### I2. The "double-then-round-once f32" accumulation convention

The CUDA emitter's convention. The oracle deliberately does *not* mirror it — it
accumulates in accurate f64 precisely so that it does not become a bit-for-bit
mirror of one emitter. A backend may use any accumulator width whose result lands
within tolerance.

### I3. Transcendental implementations

`Erf`/`Erfc`/`Gelu`/`Lgamma`/`exp`/`ln`/`tanh` are computed by the oracle in
accurate f64 (libm, in-house Lanczos), deliberately independent of the device
`erff` etc. Backends supply their own; only the tolerance is binding. None of
these is used in a bit-exact path.

---

## OPEN — needs a ruling before this is a contract

> ⚠️ **TWO OF THE FOUR ENTRIES BELOW WERE STALE, AND BOTH FAILED THE SAME WAY.**
> Answers to these questions land in `deferred.md` — that is where the work and
> the rulings are recorded — and **nothing propagated them back here.** OPEN-2
> was answered 2026-08-15 and sat open here for 19 days; OPEN-1 was answered in
> two halves, the second on 2026-09-02, and was still describing
> `oracle::required_fidelity` as a function that did not exist. Both were
> corrected 2026-09-03.
>
> **This file asks the questions; that file records the answers; the link is one
> parenthetical `(OPEN-n)` pointing the wrong way.** A reader arriving here — the
> natural place to look for what is unsettled — gets a confident description of a
> gap that has been closed. ⚠️ **An answered question left open is worse than an
> unasked one: it invites work that has already been done, and it argues from a
> premise the repository itself has retired.**
>
> **When an OPEN-n is answered, edit BOTH files in the same commit.** The
> `(OPEN-n)` tags in `deferred.md` are what make that mechanical — grep for the
> tag before recording a ruling.


### OPEN-1. ~~Nothing determines which fidelity a cell is entitled to~~ — **ANSWERED**

**Answered in two halves, on two dates. The derivation landed first; the
ownership question that was blocking anyone from maintaining it closed
2026-09-02.**

`oracle::required_fidelity(plan, operands) -> Option<Fidelity>` reads the
entitlement off the plan, exactly as the original entry asked:

```text
integer in AND integer out          -> BitExact   (wrapping is modelled exactly, so a
                                                   tolerant compare would ACCEPT a wrong
                                                   integer rather than merely be loose)
steps == 0 && ulp == 0              -> BitExact   (a body that only MOVES rounds nothing)
otherwise (arith_steps + reduction_len + 2*ulp_bound) * unit_roundoff(dtype)
ulp_bound not finite                -> None       (declines rather than guessing)
```

Derived from the numerics, not read off existing usage — which is what the
original entry's second consequence said it would have to be.

**Ownership: OURS, settled 2026-09-02.** The entry above supposed it "plausibly
belongs to KISS-Conform rather than to Unpopped alone". Fuel measured their side
at `1fb2e9db`: **zero occurrences of `required_fidelity`/`RequiredFidelity` in
any spelling**; their `fuel-dispatch/src/fkc/verify/` layer is **supply-side** —
*has this kernel EARNED what it CLAIMS?* — and ledgers verified claims,
downgrading unearned ones at import. ⚠️ **The deciding evidence is structural,
not the naming argument: the input is `plan.body`, an Unpopped IR node. Fuel does
not have it and cannot compute this band.** Their ledger records what was
measured after the fact; this derives what must be true a priori. Complementary,
not duplicated. (`deferred.md` C-1 carries the askee and the 19-day latency.)

**The exercise count in the original entry is stale and the correction is worth
recording, because the number was the entry's main evidence.** It said *"exactly
six sites construct a `Fidelity`, and five of them are inside the oracle's own
unit tests"*. Measured 2026-09-03 at `2d83a731`: **37 sites across five files.**

| file | sites |
|---|---|
| `unpopped-cpu-c/tests/complex_domain.rs` | 9 |
| `unpopped/src/oracle.rs` (production) | 8 |
| `unpopped/tests/wide_integer_comparison.rs` | 8 |
| `unpopped-cpu-c/tests/required_fidelity.rs` | 8 |
| `unpopped/src/oracle.rs` (`#[cfg(test)]`) | 5 |
| `unpopped-cpu-c/tests/cpu_end_to_end.rs` | 1 |

**Five of six inside the comparator's own tests has become five of
thirty-seven**, and there is now a test file dedicated to the derivation itself.
The original entry's *"the mechanism is barely exercised"* was true when written
and is no longer.

#### What is still open, narrowed

1. **Nothing COMPELS a caller to use it.** `compare` still takes whatever
   `Fidelity` it is handed, so the original consequence — *"did it pass at a
   tolerance whoever wrote the test chose?"* — survives for any caller that does
   not ask. The derivation exists; using it is not enforced.
2. ⚠️ **The band is CUDA's for every backend.** `contract::ulp_bound` sums a
   **CUDA** per-op ULP table and neither it nor `required_fidelity` takes a
   target. Vulkan's `exp` is 3 ULP against CUDA `expf`'s 2, so **a conforming
   Vulkan kernel fails a comparison it should pass.** Latent today — the only
   caller is `unpopped-cpu-c`'s test — and live the moment a Vulkan backend
   compares through it.

   **This is [OPEN-4](#open-4-n2-is-not-achievable-on-every-target-and-the-standard-must-say-so)
   on a second axis.** OPEN-4 is *NaN propagation is target-conditional*; this is
   *accuracy is target-conditional*. **Two instances of one shape: the rule is
   target-conditional and the code assumes it is universal.** They probably want
   one per-target seam rather than two. **Owner: this workspace.**

   ⚠️ **The numbers are NOT AVAILABLE from any machine-readable source, measured
   2026-09-03 by vulkane rather than assumed:**

   ```
   vk.xml       ULP 0 · ulp 0 · accuracy 0   (control: maxImageDimension2D = 1)
   vulkaninfo   ULP 0 · accuracy 0
   Vulkan SDK   Bin / Include / Lib — no spec document present
   ```

   **Vulkan's precision requirements live in the specification's prose appendix,
   not in the registry and not in any runtime query.** Vulkane declined to supply
   recalled figures, for the right reason: a number sourced from memory and
   seeded into a normative-adjacent table is recall wearing a measurement's
   clothes.

   ⚠️ **AND THE TABLE'S SHAPE MAY BE WRONG BEFORE ITS VALUES MATTER.** Flagged by
   vulkane as **hypothesis, not measurement**, to be checked against the spec
   before anything is built: Vulkan states precision requirements **per
   instruction AND per precision mode** — some operations are required to be
   *correctly rounded* rather than carrying a ULP bound at all, and the
   requirement differs between 32-bit and 16-bit and again under
   `RelaxedPrecision`.

   **`contract::ulp_bound` is one scalar per op.** If that hypothesis holds, this
   seam is **the same category error as the `vbytes <= 16` cap in
   `unpopped-vocab`** — right question, wrong dimensionality — and a per-target
   table of scalars would encode the error one level deeper rather than fix it.
   **Check the dimensionality before collecting values.**

   ### ⚠️ MEASURED 2026-09-06: THE SHAPE IS WRONG, AND BY THREE AXES NOT ONE

   **vulkane's hypothesis is confirmed, and it understated it.** The whole ULP
   path is keyed on the **operator alone**:

   ```text
   contract.rs:1296  pub fn ulp_bound(e: &ScalarExpr) -> f64      expression only
   contract.rs:1332  fn unary_ulp(op: UnaryOp)  -> f64            op only
   contract.rs:1389  fn binary_ulp(op: BinaryOp) -> f64           op only
   ```

   **No dtype. No target. No precision mode.** They hypothesised a missing
   precision-mode axis; there are **three** missing:

   - **dtype** — `exp` at `f32` and at `f16` are different functions with
     different accuracy, and get the same number here.
   - **target** — the KNOWN LIMIT above, now located: this is *where* the CUDA
     table is baked in, because there is no parameter for anything else.
   - **precision mode** — vulkane's `RelaxedPrecision` point.

   **One half of their concern does NOT apply, and it is worth saying so:**
   *correctly rounded* **is** expressible — it is `0.0`, and **twelve ops already
   use it** (`Neg`, `Abs`, `Sqr`, `Sqrt`, `Recip`, `Relu`, `Floor`, `Ceil`,
   `Round`, `Sign`, `Step`, `Trunc`). The table is not missing a way to say "no
   bound"; it is missing the axes that decide *when* that value is true.

   ### The concrete bite, since a dimensionality argument is easy to wave through

   **`Sqrt` is rated `0.0` — asserted correctly-rounded.** `arith_steps` counts a
   unary as 1, so `sqrt(x)` gets `rel = (1 + 2·0)·unit_roundoff` = **one unit
   roundoff**, `2^-24 ≈ 5.96e-8` at `f32`.

   ⚠️ **AND THE `3 ULP` FIGURE THAT WAS HERE WAS FABRICATED — MINE, 2026-09-06.**
   This said *"Vulkan's `sqrt` under `RelaxedPrecision` is not correctly rounded;
   a 3-ULP result is ≈ 3.6e-7, about six times the band."* **I took the `exp`
   figure from two paragraphs up and applied it to `sqrt`, a different op I had no
   number for.** A specific quantity, for the wrong function, in a document whose
   subject is being precise.

   **The argument does not need it, and is stronger stated without:** `Sqrt` is
   rated `0.0` **unconditionally — on every target, at every dtype, in every
   precision mode — and the table has no axis on which that could be
   conditional.** **If any target's `sqrt` is not correctly rounded, a conforming
   implementation is compared against a band of one unit roundoff and fails.**

   ⚠️ **Whether Vulkan's is, NOBODY IN THIS PORTFOLIO CAN SAY.** vulkane measured
   at `c1ce1af`: `vk.xml` — the registry every Vulkan implementation is generated
   from — carries **zero** hits for `ulp`, `ULP`, `accuracy` or
   `RelaxedPrecision`, and its four `precision` hits are all subpixel/subtexel/
   mipmap **bit counts**. Control: `maxImageDimension2D` = 1. **The requirements
   live in the spec's prose appendix, which neither of us holds.**

   **That absence is itself the strongest evidence for the key argument:** ⚠️ **the
   `mode` axis is not merely missing from this table — it is UNRECOVERABLE from
   the artifact a `vulkan:` target would be generated from.** **A table keyed on
   `(op)` cannot be repaired into a Vulkan-correct one by anyone reading `vk.xml`,
   however carefully; whoever fills it is reading a prose appendix by hand.**
   **That is a cost fact, and it belongs in the decision about the key.** **A conforming implementation fails, and the rating that
   fails it is the one asserting the op is EXACT.**

   ### So: do not collect values

   ⚠️ **The key is undecided, and a table with the wrong number of axes cannot be
   fixed by better entries.** Deciding whether the key is `(op)`, `(op, dtype)`,
   `(op, dtype, target)` or `(op, dtype, target, mode)` is the work; filling it is
   the expensive half and is wasted until then.

   **Measuring per-op ULP on real hardware is possible and is a DIFFERENT CLAIM:**
   it yields *this device and this driver*, not *Vulkan's requirement*, which is
   what a conformance band needs. Offered by vulkane at a cost of hours; **not
   taken, because the gap is more useful than a plausible constant.**

   **Party: NOT vulkane.** They hold no `unpopped` dependency and consume no band
   from this crate (measured at their `origin/main` `8425770`). The earlier
   attribution here was mine and was never checked — the same error, in the same
   hour, as the one corrected in `deferred.md`'s `vec_width` row.

### OPEN-2. ~~What does `VariantFidelity::BitIdentical` mean across backends?~~ — **ANSWERED 2026-08-15**

**Bit-identical to the default lowering of the same cell, in the same backend, at
the same version.**

The entry below framed two readings and asked which. **The two horns were the
right pair and the left one is the answer:** local-to-this-backend is the only
referent that is both real and non-normative. *"Says nothing across backends"* is
a **feature**, not the cost of the choice — the moment the referent becomes a
normative reference, accumulation order becomes normative for the base variant,
which contradicts I1.

**The axis is ours and KISS has no home for it — measured, not inferred.**
`deferred.md` carries the evidence: a search across KISS `spec/` at `efe111c`
returns seven hits for bit-identity language and **every one is cross-language
prose** (Slang `tanh` vs CUDA `tanh`), never a within-backend variant claim.

<details>
<summary>The original question, kept because the framing is what produced the answer</summary>

`VariantFidelity` describes a variant's fidelity *relative to the base kernel*.
Bit-identical to **what**, exactly?

- If it means "to this backend's own base variant", the claim is real but local,
  and it says nothing across backends.
- If it means "to a normative reference", then accumulation order becomes
  normative for the base variant, contradicting I1.

</details>

### OPEN-4. N2 is not achievable on every target, and the standard must say so

Rule N2 assumes a backend can *choose* to propagate NaN. On Vulkan it may not be
able to, and this is not a lowering question.

Verified against the `vk.xml` pinned by the Vulkane repo (1.4, header 348):
`VkPhysicalDeviceFloatControlsProperties` carries
`shaderSignedZeroInfNanPreserveFloat16/32/64` as `VkBool32` members (line 4401 for
the Float32 one), and the struct is marked `returnedonly="true"` — decisively a
**property the implementation reports, not a feature you enable**. The matching
SPIR-V execution mode `SignedZeroInfNanPreserve` is gated on it: vk.xml:33459
enables that capability only when the property is `VK_TRUE`.

So on a device advertising `VK_FALSE`, no lowering makes NaN propagate — not the
longhand ternary, not hand-emitted SPIR-V, not avoiding `FMax`. The
implementation is permitted not to preserve NaN at all, so a NaN operand may
never reach the comparison. A backend author looking for a switch to turn this on
will not find one.

That leaves the standard with a choice it has not made:

1. **Gate**: N2 holds only on targets that can advertise NaN preservation;
   devices that cannot are out of conformance for ops whose semantics depend on
   it. Costs coverage, keeps the rule meaningful.
2. **Weaken**: restate N2 as conditional, and accept that "conformant" means
   something different per target. Keeps coverage, and means a consumer cannot
   rely on NaN semantics without inspecting the target.

This is a ruling, not an implementation detail, and it is the kind of
target-capability question KISS-Classify already has vocabulary for. It should
not be settled inside Unpopped by default.

**Attached trap, for whoever implements the gate.** The property is read through
`vkGetPhysicalDeviceProperties2`'s pNext chain. Chaining it without first
checking `min(instance, device apiVersion) >= 1.2` leaves the struct untouched on
a pre-1.2 implementation, so the read returns zeroed memory —
indistinguishable from a genuine `VK_FALSE`. The failure is silent and yields a
plausible answer rather than an error. Here it happens to fail safe (wrongly
excluding a good device rather than admitting a broken one), but the same read
pattern inverted does not, and a conformance gate built on it would quietly
misclassify hardware.

### OPEN-3. Comparator semantics for the v2 deferrals

`RowSort`'s NaN-greatest `key_lt`, stable index ties, and TopK, plus gather /
scatter OOB policy and FP `atomicAdd` nondeterminism, are v2 oracle deferrals.
For the nondeterministic cases only an order-independent invariant is checkable
at all, so "conformance" needs a different definition there rather than a
tolerance. Out of scope until the oracle covers them.

---

## Relationship to the other backend-neutrality gaps

This is one of three known places where the neutral core encoded one backend's
behavior. The other two:

1. **f16/bf16 spelling** — the neutral `cfamily` module spells `__half` /
   `__nv_bfloat16` and emits `__half2float`-class intrinsics. Tripwired in
   `crates/unpopped/tests/neutral_spelling.rs`; the seam lands in the 0.2 batch
   with the `unpopped-cuda` carve.
2. **`effective_count_width`** — closed. It is a `Backend` trait method now
   (`backend.rs`), so `contract.rs` reaches it through the trait rather than
   through the CUDA emitter.

Treat all three as one theme: *the neutral core must encode no single backend's
spelling or semantics*.
