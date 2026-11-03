# Options note: `Cluster` scope and named barriers (ROADMAP U4 S6b)

Status: **proposal, nothing built.** S6a (the `Device` barrier) is built in the oracle (0.42.13) and the cpu-c reference
(0.42.14). What is left of the S6 row of `docs/options-u4-sync-and-atomics.md` is two things the IR already has the shape for and
`check_kernel` still refuses (`BadBarrierScope`): the `Cluster` scope, and a barrier that a **subset** of a block's threads
reach, with a participant count (a named barrier: `bar.sync id, count` in PTX, `barrier_arrive`/`barrier_wait` pairs in SYCL
and HIP cooperative groups). KISS is silent on both (no sync clause in `spec/ops.md`; checked at `origin/main`).

## What exists

- `Scope::Cluster` ("the blocks of one cluster") and `Scope::Device`; `BarrierKind` is `#[non_exhaustive]` with only `Full`.
- `LaunchShape { block, grid, subgroup }` has **no cluster dimensions**, so a cluster has no extent; `LaunchShape` is
  `#[non_exhaustive]`, so a field is additive.
- The oracle's block driver runs a block's threads in lockstep phases between barriers (`release` groups threads by scope), and
  since 0.42.13 runs the grid in phases between `Device` barriers (`advance`, `grid_released`). The uniformity rule
  (`kernel_uniform.rs`) rejects a barrier under control that differs between the threads of its scope.

## S6b-1, the `Cluster` scope

A cluster is a group of blocks launched together (`cluster_dims` on Hopper-and-later; a thread-block cluster in CUDA, a
workgroup cluster in some SYCL extensions). Its barrier is the `Device` barrier restricted to the blocks of one cluster.

**Proposal.** Add `cluster: Option<[u32; 3]>` to `LaunchShape` (`with_cluster(dims)`; the grid must be a multiple of it, the
checker says so typed). A `Cluster` barrier then needs a launch with cluster dims (`ExecWithoutLaunch`-style typed error
otherwise). Oracle: the same phase driver as S6a, but a block stops at the cluster barrier and the **cluster** (not the grid)
is released when all its blocks wait at the same barrier; accesses are ordered by a per-cluster epoch (`Stamp` gains the
cluster epoch next to `dev_epoch`). A `BlockIdx` read is divergent for a `Cluster` barrier within the cluster only, but the
conservative rule (as for `Device`: any `BlockIdx` is divergent in a kernel holding one) is simplest and stays sound.
Atomics: the `Cluster` atomic scope (refused since 0.42.7) becomes admissible with the same "narrower scope must contain both
threads" rule. cpu-c: the same phase loop, with the pass over the blocks of a cluster inside a pass over clusters; it is a
small step from S6a's emitter.
*Cost:* one `LaunchShape` field, a checker rule, a per-cluster epoch, the driver generalized from "the grid" to "a group of
blocks"; about the size of S6a. *Risk:* none to existing kernels (a launch without cluster dims is unchanged).

## S6b-2, named barriers

A named barrier lets a **subset** of a block (e.g. one warp group) synchronize without the rest: `Barrier { scope: Block, kind:
Named { id, count } }`, where `count` threads must arrive at barrier `id`. Two things change that are not additive:

1. **The uniformity rule.** "Every thread of the scope reaches the barrier" no longer holds by design; the rule must become
   "exactly `count` threads reach barrier `id` and they are the same set each time". A static check can only prove a subset
   rule for simple shapes (a branch on `threadIdx / group_size`), so the checker would accept a narrow, documented form and
   refuse the rest, or defer to a run-time `BarrierMismatch`.
2. **The oracle's rendezvous.** `release` groups by scope today; a named barrier groups by `(id, at)` and releases when `count`
   threads are waiting there, regardless of the others.

**Recommendation: do S6b-1 first and leave S6b-2 until a consumer needs it** (the ROADMAP's "each when a consumer needs it"
rule, as for the effects model). Named barriers are what warp-specialized producer/consumer kernels use (CUTLASS, Mojo,
ThunderKittens in `docs/ir-union-surveys/`); the survey rows are the consumer evidence, but the producer/consumer pattern is
only useful together with the async-copy and `mbarrier` payload of S7 (U8's), so building named barriers alone would be shape
without a user.

## Needs a ruling

1. S6b-1 (`Cluster` scope with `LaunchShape::cluster`): build now? (recommended: yes.)
2. S6b-2 (named barriers): wait for a consumer / S7? (recommended: wait.)
3. Or skip S6b for the U1 remainder (packed values, wider index dtypes), which waits on other lanes' seams.
