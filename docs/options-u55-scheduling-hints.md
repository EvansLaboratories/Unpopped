<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
# Options: non-semantic scheduling hints (ROADMAP U55, PM Q14 ruled A)

Written 2026-11-28 by `unpopped-leaves` at pm-kernels' [TASK] `next-rows` (item 3). **Options only: no code.** Q14 A (docs/ir-union-rulings.md row 14): carry
scheduling hints (AMD `schedule_barrier`, `schedule_group_barrier`) **as hints with no semantics**, "one field batched with the next breaking release", on the precedent of
the `Unroll` hint (U18). Source for the construct: `docs/ir-union-evidence/rescore-2026-11-25/mojo.md` row 17 and `conformance-idiom-gaps-verified-2026-11-27.md` row 10
("compiler instruction-scheduling controls: no value semantics"). The AMD documentation was **not** opened for this note (no patent-lane reading is claimed).

## What a hint is here

A request to the emitter about how it orders its own instructions, never a meaning: the oracle ignores it, `check_kernel` checks only that it is well formed, an emitter may lower it
(`__builtin_amdgcn_sched_barrier(mask)` and the like) **or ignore it**, and ignoring it is always correct. It is **not a fence**: it orders no memory access, synchronises no thread,
defines no local, and is uniform-safe at any depth (as `Assume` is). Nothing may rely on it for correctness.

## The carrier: three options, and what each breaks

| | Option | Shape | Breaks (measured proxy) |
|---|---|---|---|
| **A** | one **new `Stmt` variant per hint** | `Stmt::SchedBarrier { mask }`, `Stmt::SchedGroupBarrier { mask, size, id }` | `Stmt` is exhaustive: **every exhaustive `match` on `Stmt` stops compiling, twice over as hints are added**. Proxy: files naming `Stmt::Assume` at each repo's origin/main (2026-11-28): Unpopped 9, vulkane 12, baracuda 2, rockfish 2, fishstringer 2, bonefish 1, openclownfish 1 = **29 files, per new variant** |
| **B (recommended)** | **one new `Stmt` variant with a `non_exhaustive` kind** | `Stmt::ScheduleHint { hint: ScheduleHint }`, `#[non_exhaustive] enum ScheduleHint { Barrier { mask: u32 }, GroupBarrier { mask: u32, size: u32, id: u32 } }` | the same 29-file take **once**; a later hint is a new `ScheduleHint` case, which `non_exhaustive` makes additive (a match on `ScheduleHint` needs a wildcard) |
| **C** | a **field on an existing `Stmt`** (as `Unroll` is on `For`) | e.g. `Stmt::For { .., sched: SchedHint }` | **no new `Stmt` variant**, so no take chain for exhaustive matches; but a field on `For` breaks every pattern `Stmt::For { local, start, end, step, unroll, body }` written **without `..`** and every struct literal, and, more important, **it cannot say what the AMD forms say**: `schedule_barrier(mask)` is a *position* in the instruction stream (between two statements), and `schedule_group_barrier` groups the instructions around it; a hint hung on a loop is a different request. |
| D | an additive `KernelFn::hints: Vec<..>` keyed by a statement path | no `Stmt` change | **no break at all**, but a path into the body is fragile under every lowering and rewrite that moves statements, and there is no precedent for a hint that is not carried by the construct it concerns |

**Recommendation: B.** It is the one option that says what the construct says (a position in the stream), costs the exhaustive-match take **once** (not once per hint), and keeps later
hints additive. C avoids the take but cannot carry a positional hint; D avoids the break but loses the position. If the PM weighs the 29-file take above the fidelity of the position,
the fallback is **C restricted to what a loop-attached hint can say** (not these two AMD forms), which would be a smaller row than U55 as written: say so before choosing it.

## Semantics fixed by B (if chosen)

- **Oracle:** a no-op. No memory effect, no ordering, no race-log entry, no step beyond the statement count.
- **`check_kernel`:** the hint is well formed (`GroupBarrier { size }` greater than 0; the mask is an opaque `u32` passed through, never interpreted here). It may appear in any control flow, in divergent control too (it has no uniformity requirement).
- **cpu-c:** ignores it (a comment line in the C at most), as it ignores `Assume`; no decline needed because nothing depends on it.
- **Other emitters:** MAY lower to the target's scheduling intrinsic, MAY ignore; MUST NOT turn it into a barrier or a fence.
- **Serde:** new tagged `Stmt` variant; a document without it is unchanged.

## Release and tests

**Breaking** (a new `Stmt` variant): it ships with **the next breaking release**; U49 (device-side launch, BREAKING) is the other READY row waiting for one. Pre-1.0 a breaking change moves the second number; the version is the PM's. Red-first tests: a kernel with a hint
runs on the oracle to the same bytes as without it (both options of the mask); `check_kernel` refuses `size == 0`; cpu-c lowers the kernel and its C agrees with the oracle; a mutant that makes the oracle trap on the hint, and one that makes cpu-c emit a barrier for it, are caught. The migration note goes into `docs/emitter-migration-b1.md` with the same "ignore it is correct" line.

## Patent

**No patent search has been run for this row.** The carrier is a statement with no semantics and no hardware; but the hint's *lowering* names an AMD-specific scheduling intrinsic, and the AMD ISA and patents are the patent lane's to read (this lane does not open AMD documentation). Before any emitter lowers a hint, the patents lane reads the relevant AMD scheduling claims; the IR carrier itself needs none beyond that note. Findings, not advice, not a clearance.

— lane: unpopped-leaves
