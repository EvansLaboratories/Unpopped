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

The reference CUDA emitter does not use `fmaxf`. It spells the ternary out:

```c
(a != a ? a : (b != b ? b : (a >= b ? a : b)))
```

A SPIR-V/GLSL backend must not lower this to `max()` — GLSL's `max` does not
propagate NaN. Vulkane specifically should read this paragraph twice.

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

### OPEN-1. Nothing determines which fidelity a cell is entitled to

This is the gap that most limits the document. `Fidelity` (`BitExact` vs
`Tolerant { rel, abs }`) is chosen **by the caller of `compare`**, not derived
from the plan. Nothing in `plan.rs`, `backend.rs` or `contract.rs` computes it.

The mechanism is also barely exercised, which is worth stating plainly because it
is easy to assume otherwise from a 500-test suite. There are exactly **six**
sites that construct a `Fidelity` and pass it to `compare`, and **five of them
are inside the oracle's own unit tests** — the oracle checking its comparator.
The sixth is `tests/cpu_end_to_end.rs`. Tolerances used: `rel: 1e-5`,
`abs: 1e-6`, and `0.0`.

Two consequences follow, and the second is the uncomfortable one:

1. The standard cannot currently answer *"did this backend pass?"* — only *"did
   it pass at a tolerance whoever wrote the test chose?"* A backend could be
   certified against a tolerance loose enough to hide a real defect.
2. There is **no established body of practice to derive a tolerance table from**.
   I had assumed the hand-picked values across the suite would serve as evidence
   for what a `required_fidelity(plan)` should return. They will not; there are
   too few, and they are almost all the comparator testing itself rather than
   real cells being judged. This makes OPEN-1 more open than it first appears —
   the table has to be *derived from the numerics*, not read off existing usage.

The missing piece is still a derivation — roughly
`required_fidelity(plan) -> Fidelity` — that reads the entitlement off the plan:
which cells must be bit-exact (identity bodies, movement/permutation, integer
arithmetic, `Im2Col`, `Select` arm moves) and what error bound the rest are
allowed given accumulation depth and dtype.

I have deliberately not invented the tolerance table. It is a standard-defining
decision with consequences for Baracuda and Vulkane, and it plausibly belongs to
KISS-Conform rather than to Unpopped alone.

### OPEN-2. What does `VariantFidelity::BitIdentical` mean across backends?

`VariantFidelity` describes a variant's fidelity *relative to the base kernel*.
Bit-identical to **what**, exactly?

- If it means "to this backend's own base variant", the claim is real but local,
  and it says nothing across backends.
- If it means "to a normative reference", then accumulation order becomes
  normative for the base variant, contradicting I1.

The two readings are not distinguishable from the current code, and they imply
different obligations. This needs answering before a second backend can claim
`BitIdentical` about anything.

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
