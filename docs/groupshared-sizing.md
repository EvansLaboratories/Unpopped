# Sizing the `groupshared`/`__shared__` residue bucket

Design note only — **no committed code changes.** Answers "what is the
largest remaining error bucket in the Fuel corpus and what would it take to
close it," measured against `origin/main` at `b91e4d6` (post-#25, unpopped
0.11.4), using the same `slang_corpus_report` harness and the naming-fixed
147-file corpus as every prior sizing pass this session.

## Result up front

**This is not an unbuilt feature. It is a standing, already-documented
policy decision hitting its intended target.** `convert.rs`'s `SLANG_RESIDUE`
/ `CUDA_RESIDUE` lists (`crates/unpopped/src/convert.rs:61-88`) refuse
`groupshared`/`__shared__`, barriers, and atomics by design:

> "CUDA constructs that aren't IR-expressible — their presence means the
> kernel is hand-optimized (shared mem, atomics, barriers, library calls)
> and belongs in the source language. ... a *naive* reduction (plain
> accumulator loop) has none of these; the generator supplies the
> cooperative/block machinery."

So sizing this bucket is a scope question — should Unpopped's neutral IR
ever model workgroup-cooperative execution — not an engineering estimate
for a parser addition. Recommending against building anything here without
that scope call, same as the seam/scale-role decisions escalated earlier
this session.

## The population: 36/147 files, 12 distinct kernel families

Generalized error messages, largest bucket in the corpus:

| bucket | files |
|---|---|
| `Inexpressible("groupshared")` | 36 |
| `Unrecognized("non-elementwise index [*] (expected [*])")` | 22 |
| `Unrecognized("identifier '*'")` | 22 |
| `Inexpressible("InterlockedCompareExchange")` | 16 |
| `NotElementwise` | 14 |

Stripping dtype suffixes (`_f32`/`_f16`/`_bf16`/`_f64`), the 36 groupshared
files are 12 distinct kernels, each shipped in up to 4 dtype variants:
`arg_reduce_last_dim`, `flash_attention`, `layer_norm_last_dim`,
`layer_norm_last_dim_backward`, `matmul_q4_0_tiled`, `qmatvec_q4_0`,
`reduce`, `reduce_last_dim`, `rms_norm_last_dim`,
`rms_norm_last_dim_backward`, `softmax`, `softmax_last_dim_backward`.

All twelve share one shape: a single workgroup does a multi-phase
computation — per-thread partial accumulation, a `groupshared` scratch
array, a tree reduction gated by `GroupMemoryBarrierWithGroupSync()`
barriers between phases, then a single thread writes the result. `reduce.slang`
is the simplest instance and was read in full to confirm this (256-wide
tree reduction over a `groupshared float wg_data[256]` array, two barrier
points).

## Evidence the naive/cooperative split is intentional in Fuel's own corpus

`arg_reduce_any_dim_f32.slang` — one of the 4 dtype variants in the
**22-file `identifier` bucket**, not the groupshared bucket — states in its
own header comment:

> "Argmax / Argmin along ARBITRARY dim — f32. Slow path counterpart to
> `arg_reduce_last_dim_f32` (which dedicates a workgroup per row and uses a
> tree reduction). For non-last dims a workgroup-per-row dispatch is
> awkward because the row's elements are strided, so instead we use one
> thread per OUTPUT element and let each thread scan the `d_dim` values
> serially."

So Fuel already ships a naive, per-thread-serial twin for `arg_reduce`
specifically — and that naive twin is *not* blocked by the residue policy at
all; it currently fails on the unrelated, smaller `identifier` bucket (same
family as PR #22's local-variable-resolution hardening, not groupshared).
Fixing that would be a same-class, PR-#22-sized lift, not a new IR feature —
but it rescues at most 4 files (the `arg_reduce_any_dim` variants), not the
36 in this bucket.

**No other one of the 11 remaining families has a naive twin present in this
corpus** — `ls` on the corpus shows no `_naive`, `_any_dim`, or otherwise
unoptimized counterpart for `softmax`, `layer_norm`, `rms_norm`,
`flash_attention`, `reduce` (plain), `matmul_q4_0_tiled`, or `qmatvec_q4_0`.
For those, closing this bucket means the IR modeling the cooperative kernel
itself, or nothing.

## What closing it for real would require (rough shape, not measured)

Not prototyped — this is a structural sizing, same caveat as the
strided/broadcast row in `docs/slang-second-blocker-sizing.md`. A
groupshared/tree-reduction kernel is a different *category* from the
current three (elementwise / reduction / scan), all of which assume each
output is computed independently with no cross-thread synchronization:

- A new IR shape modeling a fixed-size cooperative scratch buffer, a
  parametrized reduction tree, and an explicit barrier/phase ordering —
  none of `ScalarExpr`/`OpDef`'s current variants represent
  cross-invocation communication at all.
- `Backend`/`Lowering` changes in **both** `unpopped-cpu-c` (emit an
  equivalent to `__shared__`/barriers in portable C, or fall back to a
  single-threaded oracle semantics) and `unpopped-slang` (emit
  `groupshared`/`GroupMemoryBarrierWithGroupSync` faithfully) — not a
  `convert.rs`-only change like #22 or #25.
- A decision on whether the *workgroup size* (256 in `reduce.slang`) is a
  kernel-baked constant, a dispatch parameter, or something the IR must
  parametrize — none of which the current single-thread-semantics model has
  an opinion on.

This is comparable in scope to the earlier seam-decoupling work (#18), not
a same-night PR, and plausibly larger since it touches both backends' code
generation rather than one crate's parsing.

## Recommendation

Do not build. Two independent, smaller items are available if wanted
instead, both same-class as work already merged this session:

1. Fix the `arg_reduce_any_dim` naive twin's `identifier` blocker (≤4 files,
   PR-#22-sized).
2. Ask CireSnave/PM whether workgroup-cooperative kernels are in scope for
   Unpopped's neutral IR at all, before sizing further. If the answer is no,
   this 36-file bucket is a correct, permanent refusal and the corpus's
   ceiling for `convert.rs`-only work is effectively the remaining
   `Unrecognized` buckets (90 files), not 147.
