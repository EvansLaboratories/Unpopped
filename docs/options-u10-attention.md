# Options note: attention at the kernel layer (ROADMAP U10, issue #121)

Status: **proposal, for the PM.** Same shape as the U9 note (`docs/options-u9-gemm-family.md`, scope A approved 2026-11-04): what
the kernel layer can already state and prove, one conformance kernel per PR, no new IR; and what a plan-layer `Access::Attention`
needs from other lanes. Nothing here touches `StructureKey`. U10 was "open, after U5, U7, U9"; U5 (collectives), U7 (MMA atoms) and
U9 (the GEMM family, K1 to K7) are built, so the row is unblocked.

## Why no new IR for the kernel layer

Attention is a program, not a statement: a score contraction (`S = Q K^T * scale`), a masked row softmax, a value contraction
(`O = P V`). Everything it needs exists at `origin/main` 8333176: `Load`/`Store`/`Let`/`For`, `Exp`, `Max`, `Div`, `Select` and the
comparisons (the mask), `Shared` allocations and `Block` barriers (the K/V tile), `Collective` or `AtomicRmw` (row max and sum
across threads), `Stmt::Mma` (the two GEMMs on a tensor-core path), `CopyAsync` (the K/V pipeline, S1 of U8), and `I32` index
arithmetic for a block table. The online softmax is two running locals per row, `(m, l)`, rescaled when a new tile raises `m`.
What the IR cannot say is the intent "this is attention with this mask, choose the tiling", and that is the plan layer.

## What each U10 item becomes

| Item | Kernel-layer form | Kernel (one PR each) |
|---|---|---|
| Score contraction, softmax, value contraction | three phases over materialized `S` in global memory, the reference everything else is compared to | **A1** naive attention (`F32`, small shapes) |
| Online-softmax state `(max, sum)` without materializing scores | per-row `m`, `l` and the output accumulator rescaled by `exp(m_old - m_new)` while streaming K/V tiles through `Shared` | **A2** flash-style forward, compared with A1 and a host loop within a stated ulp bound |
| Masks: causal, sliding window, ALiBi bias | the mask is a `Select` or an additive term on the score before the max | **A3** causal and ALiBi variants of A2; a fully masked row is defined (all `-inf` gives a zero row, not NaN) and tested |
| Paged KV (block-table addressing over `ReadIndex`) | K/V live in a pool of fixed-size pages; a block table maps `(sequence, page)` to a pool page; the kernel loads the page index, then the page | **A4** paged decode kernel (one query token per sequence), page tables of different lengths |
| GQA / MQA (several query heads per KV head) | a head-index division, no new construct | folded into A2 and A4 as a parameter |
| Backward | recompute `P` from the saved row log-sum-exp, then `dV = P^T dO`, `dP = dO V^T`, `dS = P * (dP - rowsum(dO * O))`, `dQ`, `dK` | **A5**, last, only if the PM wants it at this layer |
| MMA path for the two GEMMs | `Stmt::Mma` over `F16`/`BF16` fragments with an `F32` accumulator | **A6**, after A2, using the K6 chain pattern |

Float data throughout, so the reference is a host loop and equality is within a stated ulp bound (`exp` differs between the Rust
oracle and the host C library; the cpu-c differential uses the existing `agree_within` tolerance). Each kernel ships an oracle test
and a cpu-c differential, with a seeded-mutation pass: the scale dropped, the max subtraction dropped (overflow on large scores
must be visible in the data), the rescale factor wrong, the mask off by one, a page index off by one, a block barrier dropped. The
data must make each mutant observable (large-magnitude scores, a row whose maximum arrives in the last tile, a masked-out maximum).

## What stays blocked, and on whom

| Plan-layer item | Needs | Owner |
|---|---|---|
| `Access::Attention` as an intent node | a KISS contract clause (the op, its online-softmax accumulation policy, the mask grammar) and a `StructureKey` field set | KISS and Fuel (propose-first, ROADMAP D2) |
| Attention accuracy bands (what tolerance a fused kernel owes the naive one) | the accuracy key (ROADMAP C1, parked) | this lane after C1 |
| Paged KV as a plan operand role | a block-table operand kind, and the `ReadIndex` generalization to a two-level gather | Fuel and KISS |
| Autotuning tile sizes | a schedule language (ROADMAP P9) | CireSnave's spend, parked |

None is started here. A1 to A4 are the evidence of what the plan-layer node will have to mean, and the differential against a host
loop is what its lowering is later checked against.

## Decisions for the PM

| # | Question | Recommendation |
|---|---|---|
| Q1 | Scope A for U10 (kernels A1 to A4, A6, one PR each, tests only, no new IR) | **Yes** |
| Q2 | Backward (A5) at the kernel layer now, or held until a consumer asks | **Held**: the forward kernels settle the conventions first; schedule it, do not decline it |
| Q3 | Each kernel is tests only, so a patch bump each (PM allocates) | Yes |
| Q4 | A fully masked row yields zeros (not NaN) | **Yes**: it is what the common kernels do and what a test can pin; a NaN is a consumer-visible fault |
| Q5 | Paged KV pages are addressed by an `I32` block table in a buffer, not by pointers (`Place::Ptr` stays held, as for U9c) | Yes |

## Build order, once approved

A1, A2, A3, A4, A6; then A5 on request. Each with its cpu-c differential and mutation pass; the changelog fragment of each names
what the kernel pins.
