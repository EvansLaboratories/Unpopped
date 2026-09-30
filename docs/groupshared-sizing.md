# Sizing the `groupshared`/`__shared__` residue bucket

> **Superseded in part (2026-09-29).** The claim that closing this bucket
> needs the IR to model cooperation is withdrawn, and the scope question is
> answered by `docs/idiom-lifting-design.md` §1a. See "What closing it
> requires" below. The measurements in this note stand.

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
For those, closing this bucket needs the parser to recognize the cooperative
kernel and lift it to the intent it implements. It does **not** need the IR to
model cooperation. See the next section.

## What closing it requires: corrected by `docs/idiom-lifting-design.md`

This section first said that closing the bucket needs *"a new IR shape
modeling a fixed-size cooperative scratch buffer, a parametrized reduction
tree, and an explicit barrier/phase ordering"*, because *"none of
`ScalarExpr`/`OpDef`'s current variants represent cross-invocation
communication"*. **That was wrong for most of the bucket, and the claim is
withdrawn.** The second half is true, and it doesn't matter. The IR represents
the *semantics* of these kernels without representing their cooperation, and
the emitters already supply the cooperation as a schedule:

- `Access::Reduction` / `Access::RowReduce` (`crates/unpopped/src/ir.rs:1616`,
  `:1656`) already express reduce, softmax and rmsnorm. The `RowReduce` doc
  names softmax and rmsnorm as instances.
- `plan::Schedule::RowReduce` is documented as *"one block per output row
  (warp-shuffle + shared-memory tree reduce)"*.
- baracuda's CUDA emitter emits exactly that: `__shfl_down_sync` and a
  `__shared__ smem[32]` tree, at `baracuda@ab2e0bf`
  `crates/baracuda-cuda-emit/src/cuda.rs:4660-4700`.

So for `reduce`, `reduce_last_dim`, `rms_norm_last_dim`, `softmax`,
`arg_reduce_last_dim` and `layer_norm_last_dim`, the missing piece is a
**parse-side idiom recognizer**, subject to the IR gaps named in the design
doc §3.2 (e.g. RowReduce's single-input limit for the γ weight).
- Emitter work remains for `unpopped-slang` and `unpopped-cpu-c`, which don't
  yet emit a `RowReduce` schedule.
- `flash_attention`, `matmul_q4_0_tiled` and `qmatvec_q4_0` need the design
  doc's additions: `Access::Attention`, the block-dequant view, and mixed-dtype
  `Contraction`.
- The backward kernels were not assessed.

## Recommendation, as superseded

The recommendation was *do not build; ask whether workgroup-cooperative
kernels are in scope for the neutral IR*. That question is answered by the
design in `docs/idiom-lifting-design.md` §1a, once it passes the PM gate:
**the IR never models cooperation, and parsers lift cooperative source into
the existing non-cooperative intent**. The `__shared__`/`__shfl`/barrier/
`groupshared` residue refusal then means "no recognizer claims this", not
"hand-optimized, therefore residue".

The standalone item remains: the `arg_reduce_any_dim` naive twin's
`identifier` blocker (≤4 files, PR-#22-sized).
