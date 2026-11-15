Mojo / MAX (`docs/ir-union-surveys/mojo.md`), re-score at 78c7d38e (0.47.3). N = 76 table rows in file order (not 73; see "Count and numbering discrepancy").
- Status at a634b09: Yes 21 / Partial 27 / No 24 / n/a 4 (total 76)
- Status at 78c7d38e:  Yes 28 / Partial 27 / No 17 / n/a 4 (total 76)

Row numbers below are FILE ORDER 1..76 (first table row = #1). They differ from the numbers in `docs/ir-union-rescore.md` for rows after #41; see the mapping at the end.

## Changed rows (11)

| # | Capability | Status | Evidence at 78c7d38e |
|---|---|---|---|
| 4 | Launch config via DeviceContext.enqueue_function (grid/block dim, dynamic shared bytes, launch attributes) | Partial -> **Yes** | `exec.rs:LaunchShape{block,grid,subgroup,cluster,devices}`; dynamic shared = `Extent::Param`; `kernel_interface.rs:KernelInterface::of` (`DynamicShared`, `launch`, `LaunchBounds`, `max_regs`, `cooperative`); tests `kernel_interface.rs`, `kernel_exec.rs`. All four fields of the row's `LaunchSpec{grid,block,cluster,dyn_smem}` exist. PDL / cluster-launch-control stay in row 5 (No) |
| 8 | Dynamic/external shared memory (`external_memory`) | Partial -> **Yes** | `Extent::Param` runs in the launch oracle (`kernel_exec.rs`, `kernel_launch.rs`), `KernelInterface::dynamic_shared` carries the launch-time size (`kernel_interface.rs`), cpu-c spells it (`unpopped-cpu-c/tests/kernel_dynamic_shared_emit_agrees.rs`, ROADMAP U16). The "no size attribute" gap of the first re-score is closed |
| 12 | multimem ld_reduce / st (NVLink multicast memory) | No -> **Yes** | `kernel.rs:Place::Multicast` (`multimem.st` via `Store`, `multimem.red` via `AtomicRmw` at `System` scope), `Stmt::MulticastLoadReduce` (`multimem.ld_reduce`), `LaunchShape::devices`, `Scope::System`; tests `kernel_multicast.rs`, `kernel_multicast_load_reduce.rs`, cpu-c `kernel_multicast_emit_agrees.rs` |
| 18 | Device-wide `Semaphore` (NVIDIA) | No -> **Yes** | Built from statement-level atomics, as the row says it can be: `AtomicCas`/`AtomicRmw` with `order` and `Device` scope plus `While`; the oracle runs a spin-on-flag with release/acquire (`kernel_atomic_order.rs`: `spin_then_publish`) and a CAS loop (`kernel_atomic_cas.rs`: `a_cas_loop_reaches_an_atomic_maximum`). The atomics predate a634b09, so the earlier `No` was understated |
| 24 | Cluster collectives: cluster_allreduce/allgather, load/store_cluster_smem, cluster_remote_smem_addr (DSMEM), cluster_sync | No -> **Partial** | `kernel.rs:Place::ClusterPeer` (DSMEM load/store/atomic on another block's `Shared`), `Barrier{Scope::Cluster}` = cluster_sync; tests `kernel_cluster_peer.rs`, `kernel_cluster.rs`, cpu-c `kernel_cluster_peer_emit_agrees.rs`. Not confirmed: a cluster-scope `Collective` (collective checker admits only `Subgroup`/`Block`, `kernel_check/collective.rs:37,111`); an allreduce/allgather is hand-written peer loads |
| 25 | warp-level `mma` (m16n8k*, WMMA; NVIDIA, AMD MFMA/WMMA) | Partial -> **Yes** | `kernel.rs:Stmt::Mma` + `mma.rs:MmaAtom` (atom = data: shape, A/B/acc dtypes, per-lane fragment layouts, bijection-checked); shipped PTX m16n8k16 f16/bf16->f32 and m16n8k32 s8->s32; checker `kernel_check/mma.rs`, oracle `oracle/kernel_mma.rs`; tests `kernel_mma.rs`, `mma_atoms.rs`, `kernel_mma_flags.rs` (2:4 sparse, block-scaled), cpu-c `kernel_mma_emit_agrees.rs`, `kernel_mma_extras_emit_agrees.rs`; K-loop `kernel_gemm_family.rs`, `kernel_attention_mma.rs`. AMD atoms are not shipped, but an atom is a constructor, not an enum arm |
| 32 | cp.async: async_copy, commit_group, wait_group, wait_all (gmem->smem) | No -> **Yes** | `Stmt::{CopyAsync, AsyncCommit, AsyncWait{pending}}` (`wait_all` = `pending: 0`), checker `kernel_check/async_copy.rs`, oracle `oracle/kernel_async.rs` (`AsyncRace`, `AsyncUnwaited` traps); tests `kernel_async_copy.rs`, `kernel_pipeline_double_buffer.rs`, cpu-c `kernel_async_copy_emit_agrees.rs`, `kernel_pipeline_emit_agrees.rs` |
| 40 | ldmatrix-aware smem tile copy (`tile_async_copy`), `tile_io` | No -> **Partial** | Row says "derivable from AsyncCopy + Layout + Swizzle": `CopyAsync` (tests above) and `unpopped-vocab` `Layout`/`Swizzle`/`ComposedLayout` (`layout_algebra*.rs`) exist. Not confirmed: any test composing a swizzled `Shared` tile with `CopyAsync`; no `ldmatrix` (ROADMAP U8b, scheduled, not built) |
| 66 | Arg-reductions / custom combine functions as user lambdas (`warp.reduce` generic op) | Partial -> **Yes** | `CollectiveKind::{ReduceWith, ScanWith}{combine}` = module function; `Stmt::CollectiveTuple` + `KernelFn::results` + `Stmt::CallTuple` for tuple/pair state (value,index; Welford); arg reductions `ReduceOp::{ArgMax,ArgMin}`; tests `kernel_collective_combine.rs`, `kernel_collective_tuple.rs`, `kernel_call_tuple.rs`, cpu-c `kernel_collective_combine_emit_agrees.rs`, `kernel_collective_tuple_emit_agrees.rs`. Associativity is the author's claim, as the row asks; no exclusive `ScanWith` |
| 72 | AMD-specific: permlane_shuffle/swap, readfirstlane, ds_read_tr, AMDBufferResource, cvt_pk_fp8_f32, byte_permute, `lop` | No -> **Partial** | `Stmt::ExternCall` + `ExternDecl` (typed params/results, declared effects, per-target `BodyText::{Asm,Symbol}` bodies, a `reference` kernel the oracle runs): `kernel_extern.rs`, tests `kernel_extern_check.rs`, `kernel_extern_run.rs`, cpu-c `kernel_extern_emit_agrees.rs`; `CollectiveKind::Broadcast{lane}` covers readfirstlane. Opaque only: no named permlane / ds_read_tr / buffer-resource op, and a memory effect cannot be declared (v1 has no pointer operands) |
| 73 | Inline MLIR / `__mlir_op` / `inlined_assembly` escape hatch; `compile` package | No -> **Partial** | Same `ExternDecl` with `BodyText::Asm{template}` (placeholders checked) and `BodyText::Symbol`: this is the row's "opaque `Intrinsic{name,target}`". Not covered: inline MLIR, runtime function compilation and asm/IR introspection (host side) |

## Examined, unchanged (<= 25)
- 2: `ExecId::{LaneId,SubgroupId,SubgroupSize}`; no `sm_id`, no clock/special registers.
- 3: cluster rank only by arithmetic on `BlockIdx` (`kernel_cluster_peer.rs`); no cluster id leaves.
- 5: PDL scheduled as ROADMAP U17; no `clusterlaunchcontrol`; grid-stride persistence is only a user pattern.
- 6: `Place::ClusterPeer` adds shared-cluster, but no TMEM space, no generic space, no pointers.
- 9: `Private` arrays, now spelled by cpu-c; no warpgroup register realloc (`max_regs` is per kernel).
- 11: no cache-hint or eviction field anywhere (U13 scheduled).
- 14: `Barrier` at `Subgroup` scope only; no lane mask on barrier or collective.
- 19: `Stmt::Fence` order+scope only; no proxy, mbarrier-init or tensormap fences.
- 22: `VoteKind::{All,Any,Ballot}`; no match_any/match_all/elect.
- 26: no `MatrixLoad`/`MatrixStore` (U8b); fragments are plain `Private` arrays.
- 27: `Stmt::Mma` is synchronous; async groups exist for cp.async only.
- 31: `Int32` accum plus f16/bf16/s8 atoms and I4 dequant GEMM; no fp8-operand test, plan `Contraction` still one key dtype.
- 33: `CopyAsync` is 4/8/16 bytes only; no bulk copy, prefetch or reduce-on-store.
- 37: `Swizzle`/`ComposedLayout` and canonical text are vocabulary only; no kernel or IR consumer.
- 39: `slice`/`tile_to_shape` and `CopyAsync` exist; no tile/distribute/vectorize/copy_from ops.
- 44: no `F8E3M4` in `ElementKind`.
- 52: packed locals are 2 or 4 lanes only (`LocalDecl.lanes`); U15 (8/16 lanes) not built, cpu-c refuses packed locals.
- 53: only `Pack`/`Extract`, type-checked in `packed_values.rs`; no oracle or emitter run of a shuffle.
- 55: `MaskedLoad`/`MaskedStore` (`kernel_masked_access.rs`) are per-element; no vector-typed masked access.
- 57: no unroll hint on `Stmt::For` (U18 not built).
- 61: Rmw/CAS/order/scope/old value built; missing atomic load/store, weak CAS, separate failure order.
- 69: `Stmt::Call`/`CallTuple`/`ExternCall` exist; no closures.
- 71: cp.async, DSMEM, Mma, Extern now in; TMA, wgmma, cache hints, named barriers still absent.

## Rows where the status at a634b09 looks WRONG
- 18 (No, understated): atomics + `While` + release/acquire already ran at a634b09 (`kernel_atomic_order.rs`, `kernel_atomic_cas.rs`). Corrected above.
- 45 (Yes, arguably overstated): `F4E2M1/F6E2M3/F6E3M2` are storage, key identity and `Cast` (`oracle/value.rs:decode_mx`) only; `mx_elements.rs` shows the plan gate declines compute and the doc comment says "nothing computes with them". Partial is defensible.
- 42 (Yes with 2 reserved): `Fp8E4M3FNUZ`/`Fp8E5M2FNUZ` are RESERVED with no semantics (`reserved_dtypes.rs`); half the row is not expressible. Partial is defensible.
- 60 (Partial, maybe understated): arbitrary-body kernels, `BufferRole::ReadWrite` in-place and multiple outputs are all expressible at the kernel layer since before a634b09; the remaining gap is only a shape-inference function / ABI (frontend). Yes is defensible.
- 53 (No): `Pack`/`Extract` give 2- and 4-lane shuffle/slice/join as type rules, but nothing runs them in the oracle or cpu-c; left No.

## Remaining No rows (17), grouped by what they would take
- Async-MMA and tensor memory (new `MemorySpace::Tensor` + async issue/commit/wait for MMA + descriptors): 10, 27, 28, 29, 30.
- Mbarrier and named barriers (new `BarrierKind` variants / phase-parity object, id-addressed subset barriers): 15, 16, 17 (17 is advisory only, may be dropped).
- TMA and bulk copy (tensor-map descriptor, `TmaLoad/Store/Reduce`, multi-stage pipeline-state object, bulk copy + reduce): 33, 34, 35.
- Cache hints / eviction / consistency on loads, stores and `CopyAsync` (ROADMAP U13): 11.
- Launch-level dependency and persistence (PDL, cluster launch control; ROADMAP U17): 5.
- Matrix fragment moves (`MatrixLoad`/`MatrixStore`, ROADMAP U8b): 26.
- Dtypes (`F8E3M4`; int128/256): 44, 48.
- Vector-lane ops (shuffle/slice/join on wider vectors; needs U15 plus an oracle run): 53.

## Count and numbering discrepancy
`docs/ir-union-rescore.md` counts this survey as 73 rows / 3 n/a. The file has 76 table rows and 4 n/a (rows 41, 56, 75, 76). The first re-score dropped three rows whose status cell has a qualifier: 42 "Yes (2 reserved)", 56 "n/a (frontend)", 65 "Yes (single axis, inner only v1)". Its row numbers therefore lag file order after #41. Mapping from the first re-score's number to file order: #1-#41 identical; #44 -> 45 (float4/6); #60 -> 62 (fence standalone); #63 -> 66 (arg-reductions); #64 -> 67; #65 -> 68; #66 -> 69; #67 -> 70. Corrected counts at a634b09 are 21/27/24/4 (the first re-score's totals of 19 Yes / 27 Partial / 24 No / 3 n/a omit rows 42 and 65 as Yes and row 56 as n/a).

## Rows examined versus skipped
Examined 51 of the 72 non-n/a rows (every row not already Yes, 51 of 51: rows 2-6, 8-12, 14-19 (excl. none), 22, 24-35, 37-40, 44, 48, 49, 52-55, 57-61, 66, 69, 71-74), plus 42, 45 (Yes rows re-read for overstatement) and 53. Skipped: 21 rows already Yes at a634b09 (1, 7, 13, 20, 21, 23, 36, 43, 46, 47, 50, 51, 62-65, 67, 68, 70), because none of the 89 commits touches them (a re-ask could only confirm); and 4 n/a rows (41, 56, 75, 76, frontend or host-only). Rows 48 and 49 were checked only for a new dtype or index type and found none.
