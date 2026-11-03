# Options: accumulate-into-C for a contraction (ROADMAP U7, S1a-2)

Status: **options, no code.** Asked for by `docs/options-u7-mma-atoms.md` (S1a-2, after S1a). Read at `origin/main` `29a388e`.

## What exists

A contraction's epilogue may read `Reduced(0)` (the K-sum) and `Input(i)` for `2 <= i < n_inputs`, which is a **per-column `[N]` bias** read at column `n` (oracle `eval_contraction`, `plan.rs` `contraction_epilogue_admissible`, the lift's rank-1 `bias<j>` buffers; a batched contraction carries none). The `StructureKey` (`ContractionKey`) has no fact saying what an extra input is, so "bias" is a convention of rank and role.

`D = alpha*A*B + beta*C` is already an epilogue expression over `Reduced(0)` and an input. What is missing is a **full `[M,N]` (or `[B,M,N]`) C input**, and, for a precision policy, the question of **where C enters the rounding**.

## Decision 1: how an extra input says it is a full C

| Option | Cost | Note |
|---|---|---|
| **A. By operand rank (recommended).** An `Input(i>=2)` of rank 2 (rank 3 batched) is a full C read at `[m,n]`; rank 1 stays the `[N]` bias | oracle, plan gate, lift and cpu-c read the operand's rank; no key change | the plan gate must check C's shape equals the output's; one more decline: `CShapeMismatch` |
| B. A key fact (`c_form: None / Bias / Full`) | a `StructureKey` change and a version of the codec | the cache then separates the two shapes, which A cannot; heavier than the evidence asks |
| C. A new `Access` field `c: Option<..>` on `Contraction` | an IR change visible to every `match` | clearest, widest blast radius (the whole-workspace exhaustive-match hazard) |

## Decision 2: where C enters a precision policy (S1a)

Hardware MMA computes `D = A*B + C` with `C` as the **initial accumulator**, so under `Tf32`/`F16Accum` the first rounding already includes `C`. The epilogue form adds `C` after the narrow accumulation, in the wide type, which is a different number.

| Option | Meaning |
|---|---|
| **1. Epilogue only now (recommended).** `C` is read by the epilogue in the wide type, under `WideFloat` and under a policy alike; the doc says so | smallest slice; correct for every non-MMA use (`beta*C` residual, bias-add) |
| 2. Initial accumulator under a policy | `acc0 = round(C)`, then the steps as today; the bound gains `u*|C|`. Needed by `Stmt::Mma` (S2), where C is a fragment | build it in S2, where the consumer (the atom) exists, and state the bound then |

## Recommendation

A + 1: a full C by operand rank, read in the epilogue, no key change, plus a typed decline for a shape mismatch; the initial-accumulator form is S2's. Evidence plan: an independent reference for `alpha*AB + beta*C` (plain and batched), seeded mutants (C indexed `[n]` instead of `[m,n]`, `[m,m]`, transposed; beta dropped), a compiled-C differential through the lift and cpu-c, and the shape-mismatch decline.

**For the PM:** rule Decision 1 (A/B/C) and Decision 2 (1/2). Both are Unpopped extensions only in the sense that KISS has no C operand; if KISS-OPS has a bias/residual form, KISS wins.
