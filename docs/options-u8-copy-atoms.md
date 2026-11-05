# Options note: copy atoms, pipelines and roles (ROADMAP U8, issue #119)

Status: **Q1-Q6 approved as recommended by the PM 2026-11-04 (S2 and S3 scheduled as ROADMAP rows U8b and U8c, not declined); nothing built yet.** Closes gap rows 3.8 (async data movement) and 3.9 (pipelines and
roles) of `docs/ir-union-design.md`. U4, U6 and U7 are far enough along (barriers, atomics, the layout algebra, MMA atoms
with their lane tables) that the first slice is unblocked; the rest waits on named things below.

## What exists (read at `origin/main` 77d39bff)

- `Stmt::{Load, Store, Barrier, Fence, Mma, Call, ExternCall}`; `Place::{Logical, Alloc}` (arrays by id, no pointer values,
  decision D-R4-3); `Shared`/`Private` allocations; `BarrierKind` has only `Full`.
- The oracle's resumable evaluator runs a block's threads in lockstep phases between barriers and records a stamp per memory
  access for race detection; a copy that completes *later* than it is issued has no model yet.
- `MmaAtom` has per-lane id tables (`MmaAtom::id(which, lane, value)`) that say which fragment element a lane holds. `ldmatrix`
  and `stmatrix` distribute data to lanes by exactly such a table, so they can reuse the mechanism.
- Nothing for: `cp.async`, commit/wait groups, `mbarrier`, TMA, descriptors, named barriers, stages, roles, schedulers.

## The line between IR and idiom

Row 3.9 mixes two things. **Machine operations** (a copy that completes asynchronously, a barrier object with an expected byte
count) need new statements, because no composition of existing statements has their semantics. **Patterns** (a ring of stages,
producer and consumer warps, a persistent tile loop, rasterization order) are ordinary control flow over those operations plus
`BlockIdx` arithmetic, and a front end recovers them as idioms (`docs/idiom-lifting.md`); an IR node per pattern would freeze one
library's choices (CUTLASS's `PipelineTmaAsync` is not Mojo's). **Recommendation: statements for the first group only; the
second is a documented recipe plus a conformance kernel, not an IR type.**

## Slices

### S1, `cp.async` and async groups (recommended first)

`Stmt::CopyAsync { dst: Place, src: Place, bytes: 4 | 8 | 16, zfill: Option<ScalarExpr> }` copies `bytes` from a global-space
place to a `Shared` place for the calling thread; `zfill` (when `Some`) is the number of source bytes actually read, the rest
being zero (PTX `src-size`). `Stmt::AsyncCommit` closes the thread's open copies into a group; `Stmt::AsyncWait { pending: u32 }`
blocks until at most `pending` groups remain. Both new statement kinds are barrier-free (per thread).

*Oracle:* each thread keeps a queue of groups. A copy is **recorded, not performed**; it lands when its group is waited on (or at
the end of the kernel, which `check` refuses as an unwaited copy). A read of the destination before the wait that covers it, or
a second copy into bytes still in flight, is the typed `AsyncRace` trap. That is stricter than hardware (where the early read
returns old or new data) and it is the right strictness for a reference. *cpu-c:* the sequential reference performs the copy at
issue, so commit and wait are no-ops; the differential test is then "kernels that wait correctly agree", and a kernel that reads
early is refused by the checker's ordering rule, not run.
*Checker:* destination in `Shared`, source in a global buffer, `bytes` a legal size, the destination aligned to `bytes`
(the oracle already knows allocation element sizes), `AsyncWait` only after a commit that can reach it on every path.
*Cost:* three statements, one trap, one queue in the resumable evaluator, a cpu-c arm each. One breaking minor.

### S2, `ldmatrix` / `stmatrix`

`Stmt::MatrixLoad { shape, trans, count, dst: Vec<u32> (locals), src: Place }`: a warp-collective load of 8x8 b16 tiles from
`Shared` into per-lane fragment locals, the lane distribution being the one PTX fixes (lane `l` receives row `l/4`, columns
`2*(l%4)..+2` of each tile). Rather than hand-code it, define it as a `MmaAtom`-style lane table in one place and test it against
the same independent source the MMA tables used. `MatrixStore` is the mirror. *Open:* the second lane-table source
(U7's `mma.sp` hold has the same need, a source other than our own memory of the PTX manual); S2 should not start before one is
found, so the two holds are cleared together. *Oracle and cpu-c:* gather and scatter by the table. One breaking minor.

### S3, `mbarrier`, TMA and descriptors (wait for a consumer)

`mbarrier` is a `Shared` object with a phase, an arrival count and an `expect_tx` byte count; TMA is a bulk copy completing on an
`mbarrier` and driven by a **descriptor** (a host-built tensor map: base, dims, strides, box, swizzle, interleave, fill). This is
the part of U8 that needs a value kind the IR does not have (a descriptor operand, which U8 of the roadmap says it brings, plus
pointer values or an opaque handle). It is also the part only sm_90+ hardware runs, so the oracle model would be written
before any GPU evidence exists. **Recommendation: record the design shape (an `mbarrier` as an allocation kind `Alloc{space:
Shared, elem: Mbarrier}`, TMA as `CopyBulk{ dst, desc, coords, barrier }`) and build nothing until a consumer (Baracuda's
sm_90 emitter) asks.** U4's S6b-2 (named barriers) and S7 (`mbarrier`) wait on the same event.

### S4, pipelines, roles, schedulers (no IR node)

Stages are an `Alloc` with a leading stage axis plus S1 groups (`AsyncWait { pending: stages - 2 }`); roles are a branch on
`threadIdx / group` once named barriers exist (S6b-2, above); a persistent scheduler is a `While` over a tile counter with the
swizzle from `layout::swizzle`; stream-K is index arithmetic. Deliverable: one conformance kernel per pattern (a double-buffered
copy-and-sum using S1; later a producer/consumer one), plus a paragraph in `docs/idiom-lifting.md` naming them as the lift
targets. `setmaxnreg` and cluster launch control are launch attributes, not statements; they ride on `LaunchShape` fields when a
backend needs them.

## What this is not

- Not a promise of performance: the oracle and cpu-c prove meaning, not overlap. Whether copies overlap compute is the target
  emitter's and the device leg's business.
- Not a second memory model: `AsyncRace` is a trap in the existing race machinery, not a new consistency level.

## Decisions for the PM

| # | Question | Recommendation |
|---|---|---|
| Q1 | Build S1 (`CopyAsync`, `AsyncCommit`, `AsyncWait`) now | **Yes**, one breaking minor |
| Q2 | Is the oracle's "read before wait is a typed trap" the right strictness | Yes (hardware returns either value; a reference should not) |
| Q3 | S2 (`MatrixLoad`/`MatrixStore`) | Hold until a second lane-table source is found; clear with U7's `mma.sp` hold |
| Q4 | S3 (`mbarrier`, TMA, descriptors) | Design shape recorded above; build when Baracuda's sm_90 emitter asks |
| Q5 | S4 as recipes and conformance kernels, no IR node | Yes |
| Q6 | Version | PM allocates at the gate (S1 is a new exhaustive `Stmt` variant set, a breaking minor) |

## Build order, once approved

1. S1: statements, serde and text form, checker (placement, size, alignment, group ordering), tests naming each refusal, a
   mutant per check.
2. S1: oracle queue and `AsyncRace`; cpu-c arms; differential tests for correct kernels, trap tests for early reads.
3. S4 conformance kernel (double-buffered copy-and-sum) over S1.
