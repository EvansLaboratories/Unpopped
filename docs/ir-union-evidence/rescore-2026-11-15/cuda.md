# Re-score of CUDA C++ (`docs/ir-union-surveys/cuda.md`) at 78c7d38e (Unpopped 0.47.3)

Status at a634b09: Yes 18 / Partial 30 / No 47 / n/a 0 (total 95; the raw survey has no n/a row).
Status at 78c7d38e: Yes 27 / Partial 40 / No 28 / n/a 0 (total 95).

Changed rows: 24 (Partial->Yes 5, No->Yes 4, No->Partial 15; none moved down). Examined: all 77 rows not Yes at a634b09. Skipped: the 18 rows already Yes at a634b09 (rows 2,5,10,18,28,30,32,33,52,53,57,58,60,66,67,73,74,84), no downgrade check beyond the tests the first re-score cited. No row was skipped as a host-side concern; rows 8, 27, 92 (graphs/streams) were read and judged by their kernel-visible part only.

| # | Capability | Status | Evidence at 78c7d38e |
|---|---|---|---|
| 3 | dynamic `extern __shared__` + opt-in >48KB (`cudaFuncAttributeMaxDynamicSharedMemorySize`): No | Partial -> **Yes** | `DynamicShared` + `KernelInterface::of` list an `Extent::Param` `Shared` allocation and say the consumer raises the dynamic limit (`kernel_interface.rs`, `kernel_interface.rs` test); cpu-c spells it (U16, 0.45.5). Arch limit table is row 95, still No. |
| 6 | distributed shared memory (`cluster.map_shared_rank`, `shared::cluster`): No | No -> **Yes** | `Place::ClusterPeer{alloc,block,index}` (`kernel.rs`), oracle `oracle/kernel_mem_ops.rs` (`ClusterRankOutOfRange`), cpu-c spells it; `tests/kernel_cluster_peer.rs` (peer read after a `Cluster` barrier, peer write, 2-D rank order). |
| 12 | cooperative launch / `grid.sync()`: No | Partial -> **Yes** | `Barrier` at `Device` scope (`tests/kernel_device_barrier.rs`) plus `KernelInterface::cooperative` / `.cooperative()` (`kernel_interface.rs`, test `kernel_interface.rs`). The flag is emitter-set, not derived from the kernel. |
| 14 | `__launch_bounds__`, `__maxnreg__`, `-maxrregcount`, occupancy hints: No | No -> **Partial** | `KernelInterface::{launch_bounds: LaunchBounds{max_threads,min_blocks_per_sm}, max_regs}` and `LaunchShape::block` carry them (`tests/kernel_interface.rs`); emitter-set data, no hint on `KernelFn`, no oracle meaning, no occupancy model. |
| 25 | async groups: `cp.async.commit_group/wait_group`, bulk groups, `wgmma.commit_group/wait_group`: No | No -> **Partial** | `Stmt::{AsyncCommit,AsyncWait{pending}}` with oracle `AsyncRace`/`AsyncUnwaited` (`tests/kernel_async_copy.rs`: `wait_pending_lands_only_the_older_groups`, empty commit). Only the `cp.async` family: no bulk groups, no wgmma groups. |
| 26 | `cuda::pipeline` (acquire/commit/wait/release, stages count, roles, `quit`): No | No -> **Partial** | Stage buffers + `CopyAsync`/`AsyncCommit`/`AsyncWait` form a pipeline as a recipe, no IR node (`tests/kernel_pipeline_double_buffer.rs`, `support/pipeline.rs`); no acquire/release/roles/`quit` object. |
| 36 | `wmma` fragments (load_matrix_sync/mma_sync/store): m16n16k16, m8n32k16, m32n8k16 (f16/bf16/u8/s8),  | No -> **Partial** | `Stmt::Mma{atom: MmaAtom,..}` over thread-value layouts, run by the oracle (`tests/kernel_mma.rs`); shipped atoms are PTX m16n8k16/k32 only (`mma.rs`); no wmma m16n16k16/m8n32k16 atom, no tf32 atom (no tf32 `ElementKind`). |
| 37 | `mma.sync` PTX (m16n8k8/16/32/64, row/col, accumulators f16/f32/s32): No | No -> **Partial** | `Stmt::Mma` with `MmaAtom::nvidia_m16n8k16_{f16,bf16}_f32`, `nvidia_m16n8k32_s8_s32`, fragment tables checked against PTX (`tests/mma_atoms.rs`, `kernel_mma.rs`; s32 wrap tested). m16n8k8 tf32, fp8 and f16-accumulate atoms not shipped. |
| 38 | `mma.sp` structured 2:4 sparsity with metadata: No | No -> **Partial** | `MmaAtom::with_sparse_24` + `ExtraRole::Metadata`, oracle expands A through the metadata (`tests/kernel_mma_flags.rs`: `a_2_4_sparse_atom_expands_a_through_the_metadata`, unordered-position trap). Only toy atoms: the PTX `mma.sp` lane tables are not shipped, `ordered_metadata` not modelled. |
| 42 | block-scaled MMA: `.kind::mxf8f6f4`, `.kind::mxf4`, `.kind::mxf4nvf4`, `.block_scale`, `scale_vec::1 | No -> **Partial** | `MmaAtom::with_block_scale` (`ScaleA`/`ScaleB`, block length) run by the oracle (`tests/kernel_mma_flags.rs`: `a_block_scaled_atom_applies_the_scale_of_each_k_block`); toy atoms only, no `.kind::mxf4nvf4`/`scale_vec` constructor, no checked scale-factor lane table. |
| 43 | accumulate policy: tensor-core internal rounding, TF32 inputs, fp16 accumulate vs fp32, int32 accumu | No -> **Yes** | `AccumSpec::{Tf32,Tf32x3,Bf16x3,F16Accum,Int32,Int32Sat}` (`ir.rs`), oracle `oracle/accum_policy.rs`; `tests/contraction_accum_policy.rs` (every policy vs an independent reference) and `tests/contraction_int32_accum.rs` (wrap and per-k_step sat). |
| 44 | higher-rank/mixed-dtype/multi-K contractions (A,B different dtypes): No | No -> **Partial** | Kernel layer only: mixed-input dequant GEMM (`tests/kernel_dequant_gemm.rs`), grouped GEMM by offsets (`kernel_gemm_grouped.rs`), I8xU8 contraction (`contraction_int32_accum.rs`). Plan-layer per-operand dtype lands as the sk8 key (0.47.0) without a mixed-dtype `Access::Contraction`; pointer-array GEMM needs `Place::Ptr` (U9c). |
| 45 | `cuda::memcpy_async` / `cp.async` (4/8/16B, L1 bypass at 16B, global->shared): No | No -> **Partial** | `Stmt::CopyAsync{dst,src,elems}` global to `Shared`, 4/8/16 B, oracle + cpu-c (`tests/kernel_async_copy.rs`). No cache mode (`.ca/.cg`), no zero-fill/`src-size`, not bindable to an mbarrier (ROADMAP U13). |
| 49 | swizzle modes NONE/32B/64B/128B (TMA) and manual XOR swizzles for bank conflicts: No | No -> **Partial** | `Swizzle` in the layout algebra with canonical text and parser (`unpopped-vocab/src/layout_algebra/swizzle.rs`, `text.rs`; `tests/layout_algebra_properties.rs`, `layout_text_round_trip.rs`); not an addressing mode of a kernel `Place`, no TMA swizzle enum; a manual XOR is integer index math only. |
| 51 | `red.async`, `multimem.*` (NVLS multicast memory), `st.async`: No | No -> **Partial** | The `multimem.*` part is built: `Place::Multicast`, `Stmt::MulticastLoadReduce`, multicast `AtomicRmw` (`tests/kernel_multicast.rs`, `kernel_multicast_load_reduce.rs`). `red.async`/`st.async` (mbarrier-completed remote ops) are not. |
| 54 | hierarchical CuTe-style layouts (nested shape:stride, composition, logical divide), tiled/blocked la | No -> **Partial** | Layout algebra with nested shape:stride, composition, logical divide/product, complement (`unpopped-vocab/src/layout_algebra.rs`; `tests/layout_algebra_examples.rs`, `layout_algebra_properties.rs`); `View::Layout` (flat only; nested is declined) agrees with the oracle (`tests/view_layout_oracle.rs`). |
| 63 | TF32 as storage-in-f32 with truncated mantissa: Partial | Partial -> **Yes** | `AccumSpec::Tf32{k_step}` rounds inputs to TF32 (ties away, as `cvt.rna.tf32.f32`) and accumulates in f32, oracle-run (`tests/contraction_accum_policy.rs`: `tf32_rounds_a_tie_away_from_zero_and_the_hand_values_hold`). There is no standalone TF32 element type. |
| 68 | `atomicMin/Max`: Partial | Partial -> **Yes** | `AtomicOp::{Max,Min}` with `old`, order and scope in the oracle and cpu-c (`tests/kernel_atomic_rmw.rs`, `kernel_atomic_fetch.rs`); the integer forms `atomicMin/Max` names. |
| 71 | cluster-scope atomics / DSM atomics: No | Partial -> **Yes** | `AtomicRmw`/`AtomicCas` at `Scope::Cluster` on a `Place::ClusterPeer` element (`tests/kernel_cluster_peer.rs`: `a_cluster_scope_atomic_on_a_peer_element_counts_every_thread`), plus `Scope::System` across devices (`kernel_device_peer.rs`). |
| 78 | multi-stage software pipelining (stage count, prologue/epilogue): No | No -> **Partial** | The double-buffered `cp.async` conformance kernel with a typed trap for each wrong pipeline (`tests/kernel_pipeline_double_buffer.rs`); stages are a recipe (an allocation with a stage axis), no `Schedule::Pipelined`, no prologue/epilogue notion. |
| 81 | split-K / stream-K / sliced-K reduction: No | No -> **Partial** | Kernel-layer split-K in a block, across blocks by atomics, and with a device-barrier fix-up (the stream-K shape) equal the unsplit GEMM (`tests/kernel_gemm_family.rs`: `split_k_in_a_block_and_across_blocks_equal_the_unsplit_product`, `split_k_with_a_device_barrier_fixup_equals_the_unsplit_product`). No plan-level `k_split` axis, no stream-K work division. |
| 89 | inline PTX `asm volatile("..." : "=r"(x) : "r"(y))` and `cuda::ptx::*` wrappers: No | No -> **Yes** | `Module::externs: ExternDecl{params,results,Effects{pure,sync,volatile},bodies: BodyText::Asm{template,constraints}|Symbol,reference}` and `Stmt::ExternCall` (`kernel.rs`, `kernel_extern.rs`); checker and oracle (runs the `reference`) tested in `tests/kernel_extern_check.rs`, `kernel_extern_run.rs`. Scalar operands only: asm that touches memory is not expressible (needs descriptors). |
| 92 | CUDA graphs, streams, events, green contexts, multi-GPU peer access: No | No -> **Partial** | Multi-GPU peer access: `LaunchShape::devices`, `Place::Peer`, `Scope::System`, `ExecId::{DeviceRank,DeviceCount}` (`tests/kernel_device_peer.rs`). Graphs, streams, events and green contexts are host-side and absent. |
| 94 | cooperative-groups `memcpy_async` with `cg::wait/wait_prior`: No | No -> **Yes** | Cooperative `cg::memcpy_async` + `cg::wait` is each thread's `CopyAsync` slice then `AsyncCommit`/`AsyncWait{0}` + `Barrier` (`tests/kernel_async_copy.rs`, `kernel_pipeline_double_buffer.rs`); the row's own IR addition (AsyncCopy + AsyncGroup) is built. 4/8/16-byte copies only. |

## Examined, unchanged

- 1: `BufferGuarantee::restrict` is interface data only; pointer values (`Place::Ptr`) still absent.
- 4: `AllocDecl` still has no initializer; `ParamsBlock::passed_by_address` is emitter-set data.
- 11: no cluster-rank or cluster-dims leaf; the rank is arithmetic over `BlockIdx` with hand-written dims.
- 20: `Barrier` at `Subgroup` has no lane mask.
- 24: `Scope::System` fence added; proxy, async and tensormap fences absent.
- 29: ballot/any/all only; no `activemask`, no `match`.
- 31: `Subgroup`/`Block` scope only; no labeled or binary partition.
- 34: `invoke_one` is an `If` on `LaneId == 0`; no elect leaf, no mask-aware election.
- 35: block reduce/scan run; block sort does not.
- 50: packed locals lanes 2 and 4 only; 8/16-lane vectors are ROADMAP U15.
- 55: atoms carry A/B layouts, but no operand major-ness on `Access::Contraction`.
- 56: `I4`/`U4` load nibble order is fixed (low first); no named interleave layouts.
- 59: Pack/Extract exist; still no vector arithmetic beyond lane-wise `Select`.
- 61, 62: `F4E2M1`/`F6*` decode (`oracle/value.rs`); no kernel load/store of packed fp4/fp6 buffers confirmed.
- 69: CAS/Exchange/And/Or/Xor run; no native `atomicInc/Dec` wrap op (a CAS loop could).
- 70: order and scope (now `System`) on RMW; no atomic load/store statements.
- 75: `Stmt::Call`, `CallTuple`, `ExternCall`; still no recursion, templates or `constexpr`.
- 76: no pointer values or arbitrary address arithmetic (U9c).
- 82, 83, 85: `Qualified` untouched by the 89 commits; `NearestAway` is a `Cast` mode only.
- 87: `RoundMode::NearestAway` and float-to-int `Cast` (`kernel_cast_float_to_int.rs`) added; no `cvt.pack`.
- 88: no `Popc/Clz/Brev/Funnel/Dot4` op in `ir.rs`; `AccumSpec::Int32` is a contraction, not `dp4a`.
- 90: `DeviceRank`/`DeviceCount` added; no `%smid`, clock or cluster registers.

## Rows whose status at a634b09 looks wrong

- 49, 54 (No, should have been at least Partial): the layout algebra with swizzle, composition and `View::Layout` predates the first re-score (ROADMAP U6 "done", #143, #152); the first re-score listed neither row. Moved here only because the row wording asks for the algebra.
- 68 (Partial, should have been Yes): `AtomicOp::Max/Min` with `old`, order and scope were already tested when row 67 moved to Yes.
- 63 (Partial) and 43 (No) are consistent with the first re-score's date: `AccumSpec::Tf32/Int32` were built in 0.42.23 and later, after a634b09.
- No row looked overstated (no downward move).

## Remaining No rows (28), by what they would take

- Barrier kinds and mbarrier (BarrierKind variants, a barrier object with arrive/wait/expect_tx): 19, 21, 22, 23.
- TMA, bulk copies and tensor-map descriptors (ROADMAP U8c, descriptor operand kind): 46, 47, 48.
- Warpgroup / TMEM matrix units and ldmatrix (an atom scope above `Subgroup`, `Space::TensorMem`, U8b lane tables): 7, 39, 40, 41.
- Launch-level scheduling and roles (persistent/tile schedulers, stream-K work division, PDL U17, `setmaxnreg`, warp-specialized roles, device-side launch, a tile-shape schedule): 13, 15, 16, 17, 79, 80.
- Memory attributes and per-arch tables (managed memory, cache hints U13, sync domains, shared-memory limit table, atomic denormal note): 8, 9, 27, 72, 95.
- Hints and device side effects (unroll U18, `__nanosleep`/`clock`, `printf`/`assert`/`trap`): 77, 91, 93.
- Wide scalar dtypes (`F128`, `I128`/`U128`): 64, 65.
- Leader election as a first-class predicate: 34.
