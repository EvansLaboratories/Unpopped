# Accumulation policies for a contraction (ROADMAP U7 S1a)

Status: **built in the oracle** (0.42.23): `AccumSpec::{Tf32, Tf32x3, Bf16x3, F16Accum}`, each with a `k_step`. Rulings: PM, 2026-11-03
(`docs/options-u7-mma-atoms.md`, all six as recommended). **Unpopped extension, not KISS**: KISS pins no accumulator width
(KISS-OPS-6.0-0004 makes a float contraction order-invariant, compared under a tolerance) and has no policy beyond the two-valued
`MathFidelity`. To be proposed upstream, as KISS#768/#769 were.

## What a policy is

A **precision policy**, never an operator choice: the contraction is still a sum over `K` (`accumspec_is_a_precision_policy_and_never_an_operator_choice`
in `ir.rs` pins that). A policy states how a tensor core differs from an `f64` sum, as a *model of the structure*, not of a chip:

1. **Input conversion**, exact and stated per policy.
2. **Products** of converted inputs, exact in `f64`.
3. **A narrow accumulator** rounded after every `k_step` contracted elements:
   `acc = round(acc + sum_of_the_step's_products)`, steps ascending, the last step possibly shorter, the sum inside a step in
   ascending order in `f64`. The bytes of the oracle are therefore deterministic.

| Policy | Operands | Conversion | Product | Accumulator `acc` (unit roundoff `u`) |
|---|---|---|---|---|
| `Tf32 { k_step }` | `F32` | each input to TF32: 10 explicit mantissa bits, ties away from zero (`cvt.rna.tf32.f32`) | `tf32(a) * tf32(b)` | `f32` (`2^-24`) |
| `Tf32x3 { k_step }` | `F32` | head `h = tf32(x)`, remainder `l = tf32(x - h)` | `ah*bh + ah*bl + al*bh` (the `al*bl` term is dropped) | `f32` |
| `Bf16x3 { k_step }` | `F32` | as `Tf32x3` with `bf16` (nearest even) for head and remainder | same | `f32` |
| `F16Accum { k_step }` | `F16` | none (the operands are `f16`) | `a * b` | `f16` (`2^-11`), nearest even, overflow beyond 65520 is infinite |

`k_step` is the atom's K (8 for an `m16n8k8` TF32 atom, 16 for `m16n8k16` in `f16`); at least 1 (the plan gate refuses 0, and a policy
whose operand dtype does not fit the key: `ContractionDecline::AccumPolicyMismatch`). `WideFloat` (the default) is unchanged.

Edge cases, all consequences of the arithmetic and tested: a NaN input gives NaN; an infinite input stays infinite under `Tf32` and
becomes NaN under the `x3` forms (the remainder `x - head` is `inf - inf`, as the published 3xTF32 and bf16x3 algorithms compute it); an
accumulator that overflows its type is infinite; a value that rounds up past the largest finite `f32` while converting to TF32 is infinite.

**Not here yet (S1b, with the mixed-dtype `StructureKey` change the PM deferred):** `Int32`/`Int32Sat` accumulation (an `s8` input with an `s32`
output has two dtypes) and `Fp8FastAccum`. **Not modelled by the kernel lift**: `lift_contraction_opdef` declines every policy but `WideFloat`
(typed), so `unpopped-cpu-c` has no policy reference yet.

## The declared bound, per emitter (PM, 2026-11-03: stated before any emitter claims conformance)

An emitter's result `r_E` is compared with the oracle's model result `r_O` under the **declared bound**

```
|r_E - r_O|  <=  B(P) = ( gamma_K + gamma_s ) * S  +  d(P) * S          gamma_n = n*u / (1 - n*u)
```

where `K` is the contracted length, `s = ceil(K / k_step)` the number of steps, `u` the accumulator's unit roundoff (table above),
`S = sum |ã_i b̃_i|` over the *converted* inputs, and `d(P)` the term the model drops: `0` for `Tf32` and `F16Accum`, `2^-22` for `Tf32x3`
(`al*bl <= (2^-11)^2 |a||b|`), `2^-18` for `Bf16x3`. The reasoning: the oracle rounds `s` times, so it is within `gamma_s S` of the exact
sum of converted products; an emitter may associate inside a step in any order and in the accumulator type, so up to `K` roundings,
`gamma_K S`; the two bounds add; the `x3` forms add what the model drops. The bound is a **ceiling**, not a promise that any chip is that
bad. Non-finite results must agree in kind: a NaN where the oracle has NaN, the same infinity where it has an infinity.

| Emitter | Policies it may claim | Bound it states today | Evidence required to claim |
|---|---|---|---|
| The oracle (`oracle/accum_policy.rs`) | all four | `0` (it is the definition) | `tests/contraction_accum_policy.rs`: a reference written independently of the oracle |
| `unpopped-cpu-c`, via the kernel lift | none yet (the lift declines policies) | none | when the lift models a policy it reproduces the arithmetic, so `B = 0`, byte for byte, and the claim needs a compiled-C differential against the oracle |
| A device emitter (Baracuda, Fuel, Vulkane, Bonefish) | none today | none | a test, in that emitter's repo, showing results within `B(P)` of the oracle on the corpus of `tests/contraction_accum_policy.rs` and on its own shapes, naming the K range and the chip; a policy it does not run is not claimed |

An emitter that claims `P` without a stated `B` is not conformant; an emitter with a tighter bound states it in its row.

## Mapping to KISS

KISS-OPS-6.17 owns `MathFidelity = {bit-stable, reduced-mantissa-permitted}`. Every policy above except `WideFloat` is
`reduced-mantissa-permitted` (a reduced-width input conversion or a narrow accumulator); `WideFloat` computes in a wider type than the
operands and is, for KISS's purposes, what a conforming reader gets by default. KISS has no way to say *which* reduced form, so the
policies are the finer vocabulary, and the bound above is the finer comparator. The proposal to KISS waits for the first slice that
carries an emitter's claim.

## A finding recorded here

`half::f16::from_f64` (half 2.7.1) mis-rounds some values that sit just above a tie: `0.04518128032270489` is 1480.5002 quanta of
`2^-15` from zero, so it rounds up to `0.045196533203125`, and `half` returns `0.045166015625`. The oracle's `f64` to `f16` rounding is
tested against exact integer arithmetic over a 200 000-value sweep and the boundaries, and against `half`'s `from_f32` path (trustworthy
for a value an `f32` holds exactly) over 200 000 `f32` bit patterns. Any emitter or consumer that converts `f64` to `f16` through `half::f16::from_f64` should know.
