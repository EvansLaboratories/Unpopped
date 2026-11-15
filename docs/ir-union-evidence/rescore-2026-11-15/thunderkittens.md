# ThunderKittens re-score at 78c7d38e (Unpopped 0.47.3)

Survey: `docs/ir-union-surveys/thunderkittens.md`.

Status at a634b09 (Yes/Partial/No/n/a): **9 / 19 / 30 / 2** (total 60).
Status at 78c7d38e (Yes/Partial/No/n/a): **17 / 24 / 17 / 2** (total 60).

Row-count note: the survey table has 60 rows in file order, not 58. The first re-score numbered 58: it dropped the two rows whose status cell carries a parenthetical (`Yes (tensor level)`, the tril/triu row, and `No (UNVERIFIED API)`, the cross-device signal-wait row), so its numbers are shifted. This file numbers all 60 rows in file order (brief rule). Mapping to the first re-score's numbers: mine 1-33 = same; mine 35-49 = first re-score N-1; mine 51-60 = N-2. The two dropped rows are mine #34 (Yes, tril/triu, unchanged) and #50 (No at a634b09; moved below). The first re-score's 10 changed rows are mine 4, 10, 12, 13, 14, 29, 37, 39, 54, 55 (its 36 = my 37, its 38 = my 39, its 52/53 = my 54/55). Its counts (8/19/29) plus those two dropped rows (1 Yes, 1 No) give my 9/19/30.

Changed rows: **15** (8 rows end at Yes, 7 at Partial).

## Changed rows

| # | Capability | Status | Evidence at 78c7d38e |
|---|---|---|---|
| 1 | Register tile `rt<T,rows,cols,layout>` (16x16 base, packed per-thread distribution) | No -> **Partial** | Per-thread distributed fragments now exist: `AllocDecl` in `MemorySpace::Private` (`exec.rs`) plus `MmaAtom` thread-value layouts checked as bijections (`mma.rs:MmaAtom::new`/`check_bijection`), gathered/scattered by `oracle/kernel_mma.rs`; `tests/kernel_mma.rs` (PTX m16n8k16 fragment tables), `kernel_mma_flags.rs`. Still no tile value type (rows, cols, layout as a property of a value); a fragment is an untyped `Private` array and the layout lives on the atom. |
| 5 | Shared vector `sv`; `st::col_vec/row_vec` | No -> **Partial** | `AllocDecl { space: Shared, extent: Static/Param }` (`exec.rs`) is a 1-D shared array, read/written by `Place::Alloc`; `tests/kernel_launch.rs`, `kernel_atomic_shared.rs`, `kernel_async_copy.rs`. No vector orientation, alignment or swizzle attribute. Also understated at a634b09 (see wrong-status list); not a delta. |
| 11 | Distributed shared memory (cluster) | No -> **Yes** | `Place::ClusterPeer { alloc, block, index }` (`kernel.rs`, ROADMAP U12 slice 1): another block's `Shared` array by cluster rank, taken by `Load`/`Store`/`AtomicRmw`/`AtomicCas`; checked (rank type, Shared only, needs cluster dims) and raced by the oracle. `tests/kernel_cluster_peer.rs` (peer read after a `Cluster` barrier, peer write races without one, release/acquire on a peer flag, 2-D cluster rank order). |
| 15 | Warp-level `mma_AB/ABt/AtB/AtBt` (register A,B,C; mma.sync) | Partial -> **Yes** | `Stmt::Mma { atom, a, b, c, d, extra }` (`kernel.rs`), `MmaAtom` with per-operand dtypes `(a, b, acc)` and A/B/C layouts as data (`mma.rs`), checker `kernel_check/mma.rs`, oracle `oracle/kernel_mma.rs` (exact products, one rounding; `D` may alias `C` so `D += A.B` accumulates). `tests/kernel_mma.rs` (f16, bf16, s8 atoms vs PTX fragment tables, overflow wrap), `kernel_gemm_family.rs` (K6: a K loop accumulating in the fragment). Transposed forms (`ABt` etc.) are an atom with a different A/B layout, not a flag; no test builds a transposed-operand atom of its own (judgement call, see below). |
| 19 | Block-scaled MMA (mxfp8, nvfp4): scale operands SA/SB, `mma_k_bytes`, `a_sf128` | No -> **Partial** | `MmaAtom::with_block_scale(block, dtype, (sfa, sfb))` and `Stmt::Mma::extra` scale operands (`mma.rs`, `kernel.rs`), applied per K block by `oracle/kernel_mma.rs`; `tests/kernel_mma_flags.rs::a_block_scaled_atom_applies_the_scale_of_each_k_block`, `a_sparse_and_scaled_atom_composes_both`. Tested only with `F32` operands and `F32` scales on a toy m4n4k8 atom: no `F8E8M0` scale, no fp8/fp4 operand atom, no tensor-memory operands (grep of `kernel_mma*.rs` and `kernel_check/mma.rs` for `F8E8M0`/`F4E2M1`: 0 hits; `unpopped-cpu-c` still declines `F4E2M1`). |
| 21 | `cp.async` global->shared, `load_async_wait<N>` | No -> **Yes** | `Stmt::CopyAsync { dst, src, elems }`, `Stmt::AsyncCommit`, `Stmt::AsyncWait { pending }` (`kernel.rs`), checker `kernel_check/async_copy.rs`, oracle `oracle/kernel_async.rs` (issue and landing are logged writes; `KernelTrap::AsyncRace`, `AsyncUnwaited`). `tests/kernel_async_copy.rs` (commit groups, `wait pending` lands only older groups, empty commit is a group, race and unwaited traps), plus the double-buffered cp.async kernel (U8 S4). |
| 26 | TMA multicast (cluster mask) and DSMEM arrive/expect with `dst_cta` | No -> **Partial** | The DSMEM half exists: a block writes a peer block's `Shared` element by cluster rank (`Place::ClusterPeer`, `tests/kernel_cluster_peer.rs`), ordered by a `Cluster` barrier or cluster-scope release/acquire. Not there: TMA (no tensor-descriptor copy, no `cluster_mask` multicast), no remote mbarrier arrive/expect_bytes (`BarrierKind` has only `Full`; `CopyAsync` refuses a `ClusterPeer` destination). |
| 36 | Reduction: `row_max/min/sum/prod`, `col_*`, with `reset=false` accumulate-into (`src_accum`) | Partial -> **Yes** | The survey's missing piece was a loop-carried accumulator. `Stmt::Let`/`Assign` inside `Stmt::For` carry a running max/sum/output across iterations; `tests/kernel_attention_flash.rs` (U10 A2: running max `m`, sum `l`, output rescaled by `exp(m - m')`, no scores materialized; scores past 89, row max arriving last; checked against the naive kernel and a host loop) and `kernel_attention_mma.rs` (A6, row softmax between two `Stmt::Mma`). `Collective { Reduce(op) }` gives the in-warp fold. Not a tile op: a tile-shaped `row_max(dst, src, src_accum)` is scalar code over locals. |
| 44 | Persistent grid / task iteration, SM-count-sized launches | No -> **Partial** | Grid-stride loops over `ExecId::{BlockIdx, GridDim}` with `For`/`While` (`tests/kernel_parallel_grid_stride_agrees.rs`, `kernel_launch.rs`) and a work-queue ticket by `AtomicRmw` with `old` (`tests/kernel_atomic_fetch.rs`); the launch grid is whatever the host picks (`LaunchShape::grid`). Missing: any SM-count / `sm_id` query and a test of a persistent task loop with a `num_iters < 0` stop. |
| 45 | Cluster launch (`cluster_mask`, CTA pairs for cta_group 2) | No -> **Yes** | `LaunchShape::cluster: Option<[u32;3]>` / `with_cluster` (`exec.rs`), `Scope::Cluster` barrier/atomic/fence; `tests/kernel_cluster.rs` (4 blocks in clusters of 2; clusters independent). A CTA pair is a cluster of 2. The `cluster_mask` of TMA multicast is not here (row 26, 22). The cluster launch itself was in the tree at a634b09 (0.42.15/0.42.18, the evidence the first re-score cited for row 12), so this row was understated then; not a delta of the 89 commits. |
| 46 | Parallel global layout `pgl` (peer address + multicast address + TMA descriptors) | No -> **Yes** | `LaunchShape::devices`, `ExecId::{DeviceRank, DeviceCount}`, `Place::Peer { buffer, device, coords }` (a buffer's copy on another device) and `Place::Multicast { buffer, coords }` (the same element on every device) in `kernel.rs`; oracle `run_launch_devices`. `tests/kernel_device_peer.rs` (all-gather ordered by a `System` barrier, races without it), `kernel_multicast.rs`. The TMA-descriptor leg of `pgl` is not modelled (it belongs to row 22), so "Yes" is for the peer and multicast addressing. |
| 47 | `multimem::ld_reduce<ADD>`, `multimem::red`, `multimem::st` | No -> **Yes** | `Place::Multicast` with `Stmt::Store` (`multimem.st`), `Stmt::AtomicRmw` at `System` scope, `Release` (`multimem.red`; `Add/Min/Max` on I32/U32, `Add` on F32/F16/Bf16) and `Stmt::MulticastLoadReduce { local, place, op, order }` (`multimem.ld_reduce`; F16/Bf16 accumulate in f32). `tests/kernel_multicast.rs`, `kernel_multicast_load_reduce.rs`, `kernel_atomic_half.rs`. |
| 48 | TMA load from peer / TMA store to multicast / pgl reduce-stores (`store_add_async(PGL)`) | No -> **Partial** | The reduce-store to multicast is `multimem.red` (row 47); a peer read/write is `Place::Peer`. There is no TMA op in the IR, so the async tile-copy forms are absent. |
| 50 | Cross-device synchronization / signal-wait | No -> **Yes** | `Stmt::Barrier { scope: Scope::System }`, and `System`-scope `Fence`/`AtomicRmw` release/acquire (`kernel.rs`, `exec.rs::Scope::System`); the oracle orders accesses across devices only through them. `tests/kernel_device_peer.rs::a_system_release_and_acquire_chain_a_reduction_across_devices`, `an_all_gather_is_ordered_by_a_system_barrier`, `a_release_or_acquire_that_is_relaxed_or_below_system_scope_orders_nothing`. Survey marked the row UNVERIFIED API; the IR row asked for a `DeviceBarrier`, which this is. |
| 60 | ThunderMittens (Metal): 8x8 base tile on `simdgroup_matrix`, smaller role for shared memory | No -> **Partial** | "Base-tile-size as a target parameter" is what `MmaAtom` is: `(m, n, k)`, lane count and thread-value layouts are data (`MmaAtom::new`, public; `tests/kernel_mma_flags.rs` builds a toy `m4n4k8` atom on 4 lanes). No simdgroup-matrix atom is defined or tested, and the per-target tile abstraction above atoms (rows 1-3) is still thin. |

## Examined, unchanged (<= 25)

- 2: layout is atom data; no per-value layout property or legal swap-layout op.
- 4: `layout_algebra` swizzle and its text/parser exist; no swizzled shared alloc attribute.
- 6: base offsets and index arithmetic into an alloc; still no subtile view value.
- 10: Global/Shared/Private/Constant plus `ClusterPeer`; still no tensor memory.
- 12: `Scope::{Subgroup,Block,Cluster,Device,System}`; no warpgroup scope (4 warps).
- 13: `SubgroupId`/`LaneId` branching works; no `setmaxnreg`, no named barriers, no producer/consumer test.
- 14: `if LaneId == 0` is the leader; unchanged, but arguably Yes (see below).
- 16: `MmaAtom::new` accepts only `Scope::Subgroup` (`BadScope`); no async token or fence/commit/wait.
- 17: per-operand atom dtypes exist, but warpgroup-only legality and ABt rule not expressible.
- 18: tensor-memory operands, `cta_group`, semaphore completion: nothing in the tree.
- 20: complex MMA; no complex atom or complex contraction anywhere.
- 22: no tensor-descriptor copy or expect_bytes; `CopyAsync` is per-thread 4/8/16 bytes.
- 23: `AtomicRmw` combines per element; no async store, commit or wait-read groups.
- 25: `MaskedLoad` fill and kernel padding modes give zero-fill; no TMA tile load to tie it to.
- 27: `BarrierKind` has only `Full`; no mbarrier phases, arrive counts or tx bytes.
- 28: no named barrier with id and participant count.
- 29: block/warp/cluster/device/system barriers and `Fence`; no proxy fence.
- 30: only `cp.async` async-vs-plain ordering is modelled (`AsyncRace`); no proxy fence.
- 35: per-thread scalar locals broadcast fine; no tile-vector type or multi-input RowReduce.
- 38: `Cast` rounding modes cover dtype change; layout swap is hand-written index code.
- 39: element-level `Load`/`Store` between all spaces; no tile-granular cooperative move.
- 42: no stage ring, role-conditioned regions or semaphores; only a two-buffer cp.async kernel.
- 54: `F4E2M1` is storage/decode only; kernel sub-byte loads are `I4`/`U4`; cpu-c declines it.
- 55: `Packed` locals, `Pack`/`Extract` and lane-wise typing exist but no test evaluates packed arithmetic.
- 58: hardware row; target registry/emitters not touched by these commits.

Coverage: 60 rows. Skipped 11: 9 already Yes at a634b09 (31-34, 37, 41, 51-53) and 2 n/a (56, 57). Examined 49 (every non-Yes row). Of those, 15 changed and 25 are listed above; the other 9 were ruled quickly: 49 (host-side runtime concern, skip class: host), and 3, 7, 8, 9, 24, 43, 59 (need a construct the IR has no name for: tile-value vectors, TMEM, complex tiles, descriptor-carrying global layout, prefetch hint, megakernel interpreter, register pinning; grep of `kernel.rs`, `exec.rs`, `mma.rs` for them finds nothing, while the same grep finds `ClusterPeer` and `CopyAsync`), plus 8/9 which stay Partial unchanged.

## Rows where the status at a634b09 looks wrong

- 45 (Cluster launch), at a634b09 No: `LaunchShape::cluster` and `tests/kernel_cluster.rs` were already in (the first re-score cites them for row 12). Should have been Yes then.
- 5 (Shared vector): a `Shared` `AllocDecl` was in the tree and counted as the evidence for row 10; the vector row should have been Partial then.
- 14 (Leader election), Partial: IR addition asked was an `ElectOne` predicate; `if ExecId::LaneId == 0` with full participation runs in the oracle and is what the survey row quotes ("only lane 0 issues"). Arguably Yes; `elect.sync`'s any-active-lane choice is not modelled.
- 13 (Warp role), Partial: `warpid()`/`laneid()` are `SubgroupId`/`LaneId`, and `warpgroupid` is `SubgroupId / 4`; the real missing piece is synchronization between roles (mbarrier, row 27), not the predicate. Stays Partial; the reason in the first re-score ("no named barriers, no setmaxnreg") is only part of it.
- 55 (Packed 2-element types): the first re-score says "no vector arithmetic ops beyond lane-wise Select"; CHANGELOG 0.27.2 and `tests/packed_values.rs::lane_wise_arithmetic_keeps_the_packed_type` show lane-wise `Add/Sub/Mul/Fma/Min/Max` typed over `Packed`. But the tests are type-inference tests and the plan gate declines packed bodies, so no end-to-end run: Partial is right, the stated reason is not.
- 15 (judgement call): moved to Yes on `Stmt::Mma` plus a run test; if the lead wants "transposes as a flag" to be the bar, keep Partial.

## Remaining No rows (17), grouped by what they take

- Tile value types and a tile-granular memory class: 2 (register-tile layout as a property plus swap/transpose op), 3 (register vectors rv/col_vec/row_vec), 7 (tensor-memory tile `tt` and TMEM alloc/dealloc), 40 (TMEM<->register load/store).
- Async-proxy bulk copy and mbarrier family (TMA, semaphores, named barriers, proxy fence): 22 (TMA tensor load with expect_bytes), 24 (TMA/tensormap prefetch hint), 27 (mbarrier init/arrive/wait/expect_bytes, phase parity), 28 (named barriers `bar.sync id,count`), 30 (async-proxy `Fence{proxy}`).
- Warpgroup-scope and Blackwell MMA: 16 (warpgroup async `wgmma` with commit/wait token; `Scope` has no warpgroup, `MmaAtom` refuses non-Subgroup), 17 (wgmma operand legality table: fp8 only ABt, same A/B dtype), 18 (tcgen05: TMEM operands, `cta_group`, semaphore completion), 20 (complex MMA).
- Pipeline templates built on the above (need 27, 22, 12-scope and role-conditioned regions first): 42 (LCSF producer/consumer ring), 43 (LCF/LCSC and the interpreter megakernel).
- Host-side runtime, not IR: 49 (copy-engine transfers, IPC/VMM, `TKParallelTensor`).
- Cross-vendor scheduling hints: 59 (HipKittens register pinning for AGPR, wave scheduling/ping-pong).
