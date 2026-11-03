# The IR union, re-scored: where the seven surveys stand now

Status: **a dated measurement**, taken at `origin/main` `a634b09` (Unpopped 0.42.14, 2026-11-03; 0.42.15, the `Cluster` scope, is in review and is counted here only where stated). Docs only; no code changed. Asked by CireSnave (2026-11-03: "Are we getting closer to a proper union of the kernel languages in our IR?"), approved by the PM.

## What was measured, and what was not

The seven surveys in `docs/ir-union-surveys/` (519 capability rows) were written on 2026-10-24/25, when the IR had no kernel layer. They are raw snapshots and stay untouched. This note re-asks each row against the code at the ref above and records **only the rows whose answer changed**, with the test or code that carries the new answer. Every other row is unchanged **as scored then**; this pass did not re-verify the unchanged rows, and a row the surveys call "No" may be wrong in either direction. A row moves up only on evidence in the tree (an enum variant, a checker rule, a test that runs); an unconfirmed piece is named in the Evidence column and keeps the row at `Partial`.

"Yes" means the IR can express it and the oracle runs it; it does not mean an emitter for that vendor exists. The only emitter in this repo is the serial C reference (`unpopped-cpu-c`); real targets live in other repos. The union-coverage figure below counts a `Partial` as half a `Yes` over the rows that are not `n/a`; it is a rough index, not a promise.

## Totals

| Survey | Rows | Yes | Partial | No | n/a | Changed |
|---|---|---|---|---|---|---|
| CUDA C++ | 95 | 7 -> **18** | 15 -> **30** | 73 -> **47** | 0 | 31 |
| CuTe / CuTe DSL | 66 | 0 -> **7** | 15 -> **29** | 50 -> **29** | 1 | 22 |
| CUTLASS | 96 | 6 -> **10** | 19 -> **23** | 69 -> **61** | 2 | 9 |
| Mojo / MAX | 73 | 7 -> **19** | 18 -> **27** | 45 -> **24** | 3 | 25 |
| Slang | 61 | 0 -> **17** | 14 -> **19** | 44 -> **22** | 3 | 26 |
| ThunderKittens | 58 | 7 -> **8** | 11 -> **19** | 38 -> **29** | 2 | 10 |
| Triton | 70 | 2 -> **14** | 16 -> **21** | 51 -> **34** | 1 | 21 |
| **All seven** | 519 | 29 -> **93** | 108 -> **168** | 370 -> **246** | 12 | 144 |

**Union coverage index (Yes + half Partial, over applicable rows): 16% -> 35%.**

## What moved, by theme

- **Kernel body** (loops, branches, early exit, locals, calls and modules): CUDA, Mojo, Slang, Triton and CuTe control-flow rows moved to `Yes`/`Partial` (U2, U3).
- **Execution model**: thread/block/grid and lane/subgroup ids, launch geometry, subgroup size, static and dynamic shared memory, per-thread arrays (U3).
- **Synchronization**: `Subgroup`/`Block`/`Device` barriers with a uniformity rule, `Cluster` scope (0.42.15), fences, atomics with order, scope, old value and compare-and-swap (U4).
- **Collectives**: shuffle, vote, ballot, reduce, scan and broadcast at subgroup and block scope (U5).
- **Layout algebra**: the CuTe layout rows moved wholesale (U6, done): hierarchical layouts, composition, complement, divide, product, coalesce, inverses.
- **Numerics and dtypes**: `Fma`, `Cast`/`Bitcast` with rounding and saturation, rounding/ftz/approx qualifiers, fp4/fp6 storage, packed values (U1).
- **Reductions and scans**: bitwise, NaN-ignoring and arg-reductions; scans and windows over any axis; a running mean and a product window (U5, U2; four of these are Unpopped extensions, proposed in KISS#766-769).

## The remaining gaps, as a counted list

Rows still `No` after this pass: **246** of 519. They are not independent (one capability often appears in several surveys). Grouped by what it would take:

1. **Asynchronous data movement, pipelines and mbarrier-style synchronization**: 71 rows.
2. **Tensor-core and matrix-unit operations**: 37 rows.
3. **Resources, address spaces, cache policy and host-side or vendor concerns**: 36 rows.
4. **Scheduling, launch policy and whole-kernel recipes (GEMM/conv/attention schedules)**: 36 rows.
5. **Tile, fragment and partition abstractions**: 22 rows.
6. **Smaller numerics, dtypes and mixed-precision policy**: 19 rows.
7. **Escape hatches, metaprogramming and user-defined types**: 17 rows.
8. **Unclassified**: 8 rows.

Where the groups live in the ROADMAP: asynchronous movement is U4 S7 and U8 (open; named barriers are U4 S6b-2, waiting for a consumer); matrix units are U7 (open, now unblocked: U3 and U6 are done) and U8; tile and fragment abstractions are U7/U8 on top of the finished layout algebra; scheduling and whole-kernel recipes are mostly plan-level hints, not IR statements; resources are U3's open rows and partly out of scope for a kernel IR; numerics are the U1 remainder (packed arithmetic, wider index dtypes); escape hatches are mostly deliberately out of scope; custom-combine reductions and the effects model are U5/U2 D6.

The groups are assigned by keyword over each remaining `No` row's capability text (`docs/ir-union-rescore.md` generator logic: first matching group wins), so a row can land in a neighbouring group and the numbers are indicative; the 'Unclassified' line shows what the keywords missed. Where one capability appears in several surveys it is counted once per survey.

## Extensions awaiting a KISS ruling

Everything in the union that KISS does not yet have is marked "Unpopped extension, not KISS" and proposed upstream; none blocks Unpopped: KISS#766 (running mean scan), #767 (product pool), #768 (bitwise and NaN-ignoring reductions, collectives), #769 (synchronization and memory ordering).

## The changes, row by row

### CUDA C++ (`docs/ir-union-surveys/cuda.md`): 31 rows changed of 95

| # | Capability | Status | Evidence at `a634b09` |
|---|---|---|---|
| 2 | Memory spaces: static `__shared__` array | No -> **Yes** | `Place::Alloc` in `MemorySpace::Shared` with a static `AllocDecl`; `tests/kernel_launch.rs`, `kernel_atomic_shared.rs`; cpu-c spells it |
| 3 | Memory spaces: dynamic `extern __shared__` + opt-in >48KB (`cudaFuncAttributeMaxDynamicSharedMemorySize`) | No -> **Partial** | `Extent::Param`: a dynamic extent from a scalar argument (`kernel_exec.rs`); no opt-in size attribute |
| 4 | Memory spaces: `__constant__` memory, `__grid_constant__` params | No -> **Partial** | `MemorySpace::Constant` can be declared; `AllocDecl` has no initializer, so the oracle refuses to read one (shape only) |
| 5 | Memory spaces: local memory / per-thread arrays, register arrays with dynamic index | No -> **Yes** | `MemorySpace::Private` per-thread arrays (oracle: `UninitializedPrivate`, `tests/kernel_launch.rs`); cpu-c refuses them; oracle only; cpu-c refuses |
| 10 | Execution: thread/block/grid indexing (`threadIdx/blockIdx/blockDim/gridDim`) | Partial -> **Yes** | `ExecId::{ThreadIdx,BlockIdx,BlockDim,GridDim}` on three axes (`exec.rs`); `tests/kernel_exec.rs`, `kernel_launch.rs` |
| 11 | Execution: thread-block clusters (`__cluster_dims__`, `cudaLaunchKernelEx` cluster attr, cluster rank | No -> **Partial** | `LaunchShape::cluster` + `Cluster` barrier/atomic/fence scope (0.42.15); `tests/kernel_cluster.rs`; no cluster ids, no distributed shared memory |
| 12 | Execution: cooperative launch / `grid.sync()` | No -> **Partial** | `Barrier` at `Device` scope, every block resident (0.42.13; cpu-c phases 0.42.14); `tests/kernel_device_barrier.rs`, cpu-c `kernel_device_barrier_emit_agrees.rs`; grid.sync semantics; no cooperative-launch attribute |
| 18 | Sync: `__syncthreads()` / `cg::sync` | No -> **Yes** | `Stmt::Barrier` at `Subgroup` and `Block` scope + the uniformity rule (`kernel_uniform.rs`); `tests/kernel_barrier.rs`, `kernel_barriers_run.rs` |
| 20 | Sync: `__syncwarp(mask)` / `bar.warp.sync` | No -> **Partial** | `Barrier` at `Subgroup` scope; no lane mask |
| 24 | Sync: fences: `__threadfence{_block,_system}`, `fence.proxy.async`, `fence.proxy.tensormap`, `tc | No -> **Partial** | `Stmt::Fence` with order and scope (0.42.8); `tests/kernel_fence.rs`; no proxy/async/tensormap fences |
| 28 | Collectives: warp shuffle `__shfl_{,up,down,xor}_sync` | No -> **Yes** | `Stmt::Collective` shuffle/vote/ballot/reduce/scan/broadcast (0.42.3; cpu-c 0.42.4); `tests/kernel_collective.rs`, cpu-c `kernel_collective_emit_agrees.rs` |
| 29 | Collectives: vote/ballot/any/all, `__activemask`, `match_any/all` | No -> **Partial** | `Stmt::Collective` shuffle/vote/ballot/reduce/scan/broadcast (0.42.3; cpu-c 0.42.4); `tests/kernel_collective.rs`, cpu-c `kernel_collective_emit_agrees.rs`; ballot/any/all; no `activemask`/`match` |
| 30 | Collectives: `redux.sync` / `__reduce_{add,min,max,and,or,xor}_sync` (sm_80+) | Partial -> **Yes** | `Stmt::Collective` shuffle/vote/ballot/reduce/scan/broadcast (0.42.3; cpu-c 0.42.4); `tests/kernel_collective.rs`, cpu-c `kernel_collective_emit_agrees.rs`; Reduce Sum/Max/Min/And/Or/Xor |
| 31 | Collectives: cooperative_groups partitions: `tiled_partition<N>`, `labeled_partition`, `binary_partitio | No -> **Partial** | `Stmt::Collective` shuffle/vote/ballot/reduce/scan/broadcast (0.42.3; cpu-c 0.42.4); `tests/kernel_collective.rs`, cpu-c `kernel_collective_emit_agrees.rs`; `Subgroup` scope = a tiled partition of the launch subgroup size only |
| 32 | Collectives: `cg::reduce` w/ plus/less/greater/bit_and/bit_or/bit_xor; `reduce_update_async` | Partial -> **Yes** | `Stmt::Collective` shuffle/vote/ballot/reduce/scan/broadcast (0.42.3; cpu-c 0.42.4); `tests/kernel_collective.rs`, cpu-c `kernel_collective_emit_agrees.rs` |
| 59 | Dtypes: `half2` / `__nv_bfloat162` / `float2` / `float4` packed | No -> **Partial** | `ScalarExpr::{Pack,Extract,BitcastPacked}` and the packed `Select` mask (0.31.x); no vector arithmetic ops beyond lane-wise `Select` |
| 61 | Dtypes: `__nv_fp6_e2m3`, `__nv_fp6_e3m2` | No -> **Partial** | `ElementKind::{F4E2M1,F6E2M3,F6E3M2}`: storage and key identity (0.29.0), `Cast` rows (0.31.7); stored-result layout still open |
| 62 | Dtypes: `__nv_fp4_e2m1` (and NVFP4 = e2m1 + ue4m3 block scale) | No -> **Partial** | `ElementKind::{F4E2M1,F6E2M3,F6E3M2}`: storage and key identity (0.29.0), `Cast` rows (0.31.7); stored-result layout still open |
| 67 | Atomics/ordering: `atomicAdd` | Partial -> **Yes** | `Stmt::AtomicRmw`/`AtomicCas` with `old`, order and scope (0.41.x-0.42.7); `tests/kernel_atomic_{rmw,fetch,cas,order,shared}.rs` |
| 69 | Atomics/ordering: `atomicCAS`, `atomicExch`, `atomicAnd/Or/Xor`, `atomicInc/Dec` | No -> **Partial** | `Stmt::AtomicRmw`/`AtomicCas` with `old`, order and scope (0.41.x-0.42.7); `tests/kernel_atomic_{rmw,fetch,cas,order,shared}.rs`; CAS, exchange, and/or/xor; no `atomicInc/Dec` wrap ops |
| 70 | Atomics/ordering: `cuda::atomic<T,scope>`, `atomic_ref`, `fetch_add`, load/store with acquire/release; scope | No -> **Partial** | order+scope RMW (an `Or` of 0 reads); no atomic load/store statements; a spin on one is a named `CrossBlockWait` (0.42.7) |
| 71 | Atomics/ordering: cluster-scope atomics / DSM atomics | No -> **Partial** | `LaunchShape::cluster` + `Cluster` barrier/atomic/fence scope (0.42.15); `tests/kernel_cluster.rs`; no cluster ids, no distributed shared memory; cluster-scope atomics; no distributed shared memory |
| 73 | Control flow: data-dependent `while`/`for` loops, early `break/continue/return` | No -> **Yes** | `Stmt::{If,For,While,Break,Continue,Return}` and `Let`/`Assign` locals; `tests/kernel_layer.rs`, `kernel_resume_agrees.rs`, `kernel_eval_semantics.rs` |
| 74 | Control flow: predication / divergence (`if (tid<n)`), masks | Partial -> **Yes** | `Stmt::{If,For,While,Break,Continue,Return}` and `Let`/`Assign` locals; `tests/kernel_layer.rs`, `kernel_resume_agrees.rs`, `kernel_eval_semantics.rs` |
| 75 | Control flow: recursion, function calls, templates, `constexpr` | No -> **Partial** | `Stmt::Call`, `Module`, `check_module`, run in the oracle and spelled in cpu-c; `tests/kernel_module_run.rs`, `kernel_call_shape.rs`; no recursion, templates or `constexpr` |
| 82 | Numerics: `--use_fast_math` bundle: `-ftz`, `-prec-div`, `-prec-sqrt`, `-fmad` | No -> **Partial** | `ScalarExpr::Qualified` rounding/ftz/approx qualifiers (0.25.0); `docs/ir-numerics-qualifiers.md` |
| 83 | Numerics: explicit rounding-mode intrinsics (`_rn/_rz/_ru/_rd`: `__fadd_rn`, `__fmaf_rz`, `__fma_rd` | No -> **Partial** | `ScalarExpr::Qualified` rounding/ftz/approx qualifiers (0.25.0); `docs/ir-numerics-qualifiers.md` |
| 84 | Numerics: explicit FMA (`fma`, `__fmaf_rn`, `__fmaf_ieee_*`) vs contraction freedom (`-fmad`) | No -> **Yes** | `ScalarExpr::Fma` (0.23.0) |
| 85 | Numerics: approximate transcendental intrinsics (`__expf`, `__sinf`, `__cosf`, `__powf`, `__fdividef | No -> **Partial** | `ScalarExpr::Qualified` rounding/ftz/approx qualifiers (0.25.0); `docs/ir-numerics-qualifiers.md` |
| 87 | Numerics: explicit conversions with rounding/saturation (`cvt.rn.satfinite`, `cvt.pack`, `__float2ha | No -> **Partial** | `ScalarExpr::Cast`/`Bitcast` with `RoundMode` and `Saturation` (0.24.0); `docs/ir-typed-expressions.md`; no `cvt.pack` |
| 90 | Unique to CUDA: special registers (`%laneid`, `%warpid`, `%smid`, `%clock`, `%globaltimer`, cluster regs ` | No -> **Partial** | `ExecId::{LaneId,SubgroupId,SubgroupSize}`; `tests/kernel_exec.rs` (no `sm_id`, clock or other special registers) |

### CuTe / CuTe DSL (`docs/ir-union-surveys/cute.md`): 22 rows changed of 66

| # | Capability | Status | Evidence at `a634b09` |
|---|---|---|---|
| 1 | Layout: Layout = (Shape, Stride) pair with explicit integer strides as first-class IR value | No -> **Yes** | `unpopped-vocab/src/layout_algebra.rs`: hierarchical `IntTuple`/`Layout`, `composition`, `complement`, `logical_divide`, `logical_product`, `coalesce`, inverses, `swizzle`, `select/take/slice/tile_to_shape` with a property-test suite (U6, ROADMAP: done, issue #117 closed 2026-11-02) |
| 2 | Layout: Hierarchical/nested IntTuple shapes and strides (`(_5,(_2,_2)):(_16,(_80,_4))`), depth, co | No -> **Yes** | `unpopped-vocab/src/layout_algebra.rs`: hierarchical `IntTuple`/`Layout`, `composition`, `complement`, `logical_divide`, `logical_product`, `coalesce`, inverses, `swizzle`, `select/take/slice/tile_to_shape` with a property-test suite (U6, ROADMAP: done, issue #117 closed 2026-11-02) |
| 4 | Layout: Layout as function idx->offset; size/rank/depth/cosize | No -> **Yes** | `unpopped-vocab/src/layout_algebra.rs`: hierarchical `IntTuple`/`Layout`, `composition`, `complement`, `logical_divide`, `logical_product`, `coalesce`, inverses, `swizzle`, `select/take/slice/tile_to_shape` with a property-test suite (U6, ROADMAP: done, issue #117 closed 2026-11-02) |
| 5 | Layout: crd2idx / idx2crd (colexicographic, natural, R-D, h-D coordinates) | Partial -> **Yes** | `unpopped-vocab/src/layout_algebra.rs`: hierarchical `IntTuple`/`Layout`, `composition`, `complement`, `logical_divide`, `logical_product`, `coalesce`, inverses, `swizzle`, `select/take/slice/tile_to_shape` with a property-test suite (U6, ROADMAP: done, issue #117 closed 2026-11-02) |
| 6 | Layout: Structural ops: get/select/take/flatten/group/append/prepend/replace/slice/dice/filter/fil | No -> **Partial** | `unpopped-vocab/src/layout_algebra.rs`: hierarchical `IntTuple`/`Layout`, `composition`, `complement`, `logical_divide`, `logical_product`, `coalesce`, inverses, `swizzle`, `select/take/slice/tile_to_shape` with a property-test suite (U6, ROADMAP: done, issue #117 closed 2026-11-02); select/take/slice/flatten/tile_to_shape; group/append/prepend/replace/dice/filter not confirmed |
| 8 | Layout algebra: composition (A o B), by-mode composition with tilers | No -> **Yes** | `unpopped-vocab/src/layout_algebra.rs`: hierarchical `IntTuple`/`Layout`, `composition`, `complement`, `logical_divide`, `logical_product`, `coalesce`, inverses, `swizzle`, `select/take/slice/tile_to_shape` with a property-test suite (U6, ROADMAP: done, issue #117 closed 2026-11-02) |
| 9 | Layout algebra: complement(layout, cotarget) | No -> **Yes** | `unpopped-vocab/src/layout_algebra.rs`: hierarchical `IntTuple`/`Layout`, `composition`, `complement`, `logical_divide`, `logical_product`, `coalesce`, inverses, `swizzle`, `select/take/slice/tile_to_shape` with a property-test suite (U6, ROADMAP: done, issue #117 closed 2026-11-02) |
| 10 | Layout algebra: logical_divide / zipped / tiled / flat divide (tiling with tile + rest modes, round-up res | No -> **Partial** | `unpopped-vocab/src/layout_algebra.rs`: hierarchical `IntTuple`/`Layout`, `composition`, `complement`, `logical_divide`, `logical_product`, `coalesce`, inverses, `swizzle`, `select/take/slice/tile_to_shape` with a property-test suite (U6, ROADMAP: done, issue #117 closed 2026-11-02); logical_divide; the zipped/tiled/flat variants not confirmed |
| 11 | Layout algebra: logical_product / blocked / raked / zipped / tiled / flat product | No -> **Partial** | `unpopped-vocab/src/layout_algebra.rs`: hierarchical `IntTuple`/`Layout`, `composition`, `complement`, `logical_divide`, `logical_product`, `coalesce`, inverses, `swizzle`, `select/take/slice/tile_to_shape` with a property-test suite (U6, ROADMAP: done, issue #117 closed 2026-11-02); logical_product; the variants not confirmed |
| 12 | Layout algebra: left_inverse / right_inverse; max_common_layout/vector (basis of auto-vectorisation) | No -> **Partial** | `unpopped-vocab/src/layout_algebra.rs`: hierarchical `IntTuple`/`Layout`, `composition`, `complement`, `logical_divide`, `logical_product`, `coalesce`, inverses, `swizzle`, `select/take/slice/tile_to_shape` with a property-test suite (U6, ROADMAP: done, issue #117 closed 2026-11-02); left/right inverse; `max_common_layout/vector` not built |
| 13 | Layout algebra: make_ordered_layout, make_identity_layout, tile_to_shape, recast_layout (element bit-width | No -> **Partial** | `unpopped-vocab/src/layout_algebra.rs`: hierarchical `IntTuple`/`Layout`, `composition`, `complement`, `logical_divide`, `logical_product`, `coalesce`, inverses, `swizzle`, `select/take/slice/tile_to_shape` with a property-test suite (U6, ROADMAP: done, issue #117 closed 2026-11-02); `col_major`, `tile_to_shape`; `recast_layout`/`make_ordered_layout` not confirmed |
| 15 | Swizzle: `Swizzle<B,M,S>` XOR functor; ComposedLayout(swizzle o offset o layout); smem swizzle atom | No -> **Partial** | `swizzle` in `layout_algebra` (a functor over layouts); no shared-memory tile consumer, no `ComposedLayout` with smem atoms |
| 18 | Memory space: Pointer tagged with address space: gmem/smem/rmem/tmem/dsmem/cmem/generic | No -> **Partial** | `Place::Alloc` in `MemorySpace::Shared` with a static `AllocDecl`; `tests/kernel_launch.rs`, `kernel_atomic_shared.rs`; cpu-c spells it; `MemorySpace` on allocations; no tagged pointers |
| 19 | Memory: Static shared-memory allocation / `cute.struct` SharedStorage with MemRange and Align; dyn | No -> **Partial** | `Place::Alloc` in `MemorySpace::Shared` with a static `AllocDecl`; `tests/kernel_launch.rs`, `kernel_atomic_shared.rs`; cpu-c spells it; no `cute.struct`/alignment |
| 20 | Memory: Register fragments: make_fragment_like / make_rmem_tensor / TensorSSA (immutable vector va | No -> **Partial** | `MemorySpace::Private` per-thread arrays (oracle: `UninitializedPrivate`, `tests/kernel_launch.rs`); cpu-c refuses them |
| 28 | Index: Thread / block / cluster / warp identity values (thread_idx, block_idx, cluster_idx, lane_ | No -> **Partial** | `ExecId::{LaneId,SubgroupId,SubgroupSize}`; `tests/kernel_exec.rs` (no `sm_id`, clock or other special registers); thread/block/lane/subgroup ids; no cluster id |
| 49 | Sync: CTA barrier `sync_threads`, named barriers (IDs 0-15, arrive / arrive_and_wait), warp sync | No -> **Partial** | `Stmt::Barrier` at `Subgroup` and `Block` scope + the uniformity rule (`kernel_uniform.rs`); `tests/kernel_barrier.rs`, `kernel_barriers_run.rs`; CTA/warp barrier; no named barriers |
| 50 | Sync: Warp collectives: shuffle (idx/up/down/bfly), vote (ballot/any/all/uni), match, warp_redux | No -> **Partial** | `Stmt::Collective` shuffle/vote/ballot/reduce/scan/broadcast (0.42.3; cpu-c 0.42.4); `tests/kernel_collective.rs`, cpu-c `kernel_collective_emit_agrees.rs`; shuffle idx/up/down/xor, ballot/any/all; no match/uni |
| 51 | Sync: Cluster sync (arrive/wait/relaxed), cluster/CTA-in-cluster ids, distributed shared memory | No -> **Partial** | `LaunchShape::cluster` + `Cluster` barrier/atomic/fence scope (0.42.15); `tests/kernel_cluster.rs`; no cluster ids, no distributed shared memory; cluster barrier; no cluster ids or DSM |
| 53 | Sync: Memory fences: acq_rel at cta/cluster/gpu/sys; proxy fences (async/tensormap/generic); tme | No -> **Partial** | `Stmt::Fence` with order and scope (0.42.8); `tests/kernel_fence.rs`; no proxy/async/tensormap fences |
| 60 | Control: Per-thread control flow: if/for/while with loop-carried values; tile loops (k-tile mainloo | No -> **Yes** | `Stmt::{If,For,While,Break,Continue,Return}` and `Let`/`Assign` locals; `tests/kernel_layer.rs`, `kernel_resume_agrees.rs`, `kernel_eval_semantics.rs` |
| 62 | Launch: grid/block/cluster, fallback_cluster, dynamic smem, PDL (`use_pdl`), cooperative launch, m | No -> **Partial** | `LaunchShape::cluster` + `Cluster` barrier/atomic/fence scope (0.42.15); `tests/kernel_cluster.rs`; no cluster ids, no distributed shared memory; grid/block/cluster dims, dynamic shared extent; no PDL/fallback cluster |

### CUTLASS (`docs/ir-union-surveys/cutlass.md`): 9 rows changed of 96

| # | Capability | Status | Evidence at `a634b09` |
|---|---|---|---|
| 58 | Scheduling: Thread block clusters, cluster shape, preferred+fallback cluster, dynamic cluster | No -> **Partial** | `LaunchShape::cluster` + `Cluster` barrier/atomic/fence scope (0.42.15); `tests/kernel_cluster.rs`; no cluster ids, no distributed shared memory; cluster dims; no preferred/fallback or dynamic cluster |
| 66 | Data movement: Shared-memory tiles with swizzled layouts (Crosswise/Congruous, CuTe Swizzle<B,M,S>) | No -> **Partial** | `swizzle` in `layout_algebra` (a functor over layouts); no shared-memory tile consumer, no `ComposedLayout` with smem atoms |
| 74 | Compute: Saturating (satfinite) int output | No -> **Yes** | `ScalarExpr::Cast`/`Bitcast` with `RoundMode` and `Saturation` (0.24.0); `docs/ir-typed-expressions.md`; `Saturation` on integer outputs |
| 78 | Compute: fp6 (e2m3, e3m2), fp4 (e2m1) | No -> **Partial** | `ElementKind::{F4E2M1,F6E2M3,F6E3M2}`: storage and key identity (0.29.0), `Cast` rows (0.31.7); stored-result layout still open |
| 83 | Compute: Half2/bfloat162/vectorized packed math (HFMA2, FP32x2 packed) | No -> **Partial** | `ScalarExpr::{Pack,Extract,BitcastPacked}` and the packed `Select` mask (0.31.x); no vector arithmetic ops beyond lane-wise `Select` |
| 84 | Compute: Fast-math/approx transcendentals (`__expf`) as explicit "Approx" | No -> **Partial** | `ScalarExpr::Qualified` rounding/ftz/approx qualifiers (0.25.0); `docs/ir-numerics-qualifiers.md` |
| 85 | Compute: Warp/block collectives (shuffle, reduce) | Partial -> **Yes** | `Stmt::Collective` shuffle/vote/ballot/reduce/scan/broadcast (0.42.3; cpu-c 0.42.4); `tests/kernel_collective.rs`, cpu-c `kernel_collective_emit_agrees.rs` |
| 86 | CuTe/DSL: CuTe layouts algebra (composition, complement, logical divide/product, hierarchical shapes | No -> **Yes** | `unpopped-vocab/src/layout_algebra.rs`: hierarchical `IntTuple`/`Layout`, `composition`, `complement`, `logical_divide`, `logical_product`, `coalesce`, inverses, `swizzle`, `select/take/slice/tile_to_shape` with a property-test suite (U6, ROADMAP: done, issue #117 closed 2026-11-02) |
| 89 | CuTe/DSL: Control flow with dynamic loops/ifs inside kernel | No -> **Yes** | `Stmt::{If,For,While,Break,Continue,Return}` and `Let`/`Assign` locals; `tests/kernel_layer.rs`, `kernel_resume_agrees.rs`, `kernel_eval_semantics.rs` |

### Mojo / MAX (`docs/ir-union-surveys/mojo.md`): 25 rows changed of 73

| # | Capability | Status | Evidence at `a634b09` |
|---|---|---|---|
| 1 | Exec hierarchy: thread_idx / block_idx / block_dim / grid_dim / global_idx (x,y,z) | No -> **Yes** | `ExecId::{ThreadIdx,BlockIdx,BlockDim,GridDim}` on three axes (`exec.rs`); `tests/kernel_exec.rs`, `kernel_launch.rs` |
| 2 | Exec hierarchy: lane_id, warp_id, sm_id | No -> **Partial** | `ExecId::{LaneId,SubgroupId,SubgroupSize}`; `tests/kernel_exec.rs` (no `sm_id`, clock or other special registers) |
| 3 | Exec hierarchy: thread blocks clusters: cluster_idx, cluster_dim, block_id_in_cluster, block_rank_in_clust | No -> **Partial** | `LaunchShape::cluster` + `Cluster` barrier/atomic/fence scope (0.42.15); `tests/kernel_cluster.rs`; no cluster ids, no distributed shared memory; cluster dims and barrier; no cluster_idx/rank leaves |
| 4 | Exec hierarchy: Launch config via DeviceContext.enqueue_function (grid/block dim, dynamic shared bytes, la | No -> **Partial** | `Extent::Param`: a dynamic extent from a scalar argument (`kernel_exec.rs`); no opt-in size attribute; grid/block dims in `LaunchShape`; no dynamic shared bytes attribute |
| 6 | Memory spaces: AddressSpace (generic/global/shared/constant/local/shared-cluster) on pointers | No -> **Partial** | `Place::Alloc` in `MemorySpace::Shared` with a static `AllocDecl`; `tests/kernel_launch.rs`, `kernel_atomic_shared.rs`; cpu-c spells it; `MemorySpace` on allocations, not on pointers |
| 7 | Memory spaces: Static shared memory via stack_allocation / `unsafe_stack_allocation` | No -> **Yes** | `Place::Alloc` in `MemorySpace::Shared` with a static `AllocDecl`; `tests/kernel_launch.rs`, `kernel_atomic_shared.rs`; cpu-c spells it |
| 8 | Memory spaces: Dynamic/external shared memory (`external_memory`) | No -> **Partial** | `Extent::Param`: a dynamic extent from a scalar argument (`kernel_exec.rs`); no opt-in size attribute |
| 9 | Memory spaces: Registers/local arrays; register realloc (`warpgroup_reg_alloc/dealloc`) | No -> **Partial** | `MemorySpace::Private` per-thread arrays (oracle: `UninitializedPrivate`, `tests/kernel_launch.rs`); cpu-c refuses them |
| 13 | Sync: Block barrier (`barrier()`) | No -> **Yes** | `Stmt::Barrier` at `Subgroup` and `Block` scope + the uniformity rule (`kernel_uniform.rs`); `tests/kernel_barrier.rs`, `kernel_barriers_run.rs` |
| 14 | Sync: syncwarp(mask) | No -> **Partial** | `Barrier` at `Subgroup` scope; no lane mask |
| 19 | Sync: Memory fences: threadfence, `atomic.fence`, fence_async_view_proxy, fence_mbarrier_init, t | No -> **Partial** | `Stmt::Fence` with order and scope (0.42.8); `tests/kernel_fence.rs`; no proxy/async/tensormap fences |
| 20 | Warp collectives: shuffle_idx/up/down/xor | No -> **Yes** | `Stmt::Collective` shuffle/vote/ballot/reduce/scan/broadcast (0.42.3; cpu-c 0.42.4); `tests/kernel_collective.rs`, cpu-c `kernel_collective_emit_agrees.rs` |
| 21 | Warp collectives: warp sum/max/min/reduce, lane_group_{sum,max,min,reduce}, broadcast, prefix_sum | Partial -> **Yes** | `Stmt::Collective` shuffle/vote/ballot/reduce/scan/broadcast (0.42.3; cpu-c 0.42.4); `tests/kernel_collective.rs`, cpu-c `kernel_collective_emit_agrees.rs` |
| 22 | Warp collectives: vote (ballot mask), match_any, match_all, elect_one_sync | No -> **Partial** | `Stmt::Collective` shuffle/vote/ballot/reduce/scan/broadcast (0.42.3; cpu-c 0.42.4); `tests/kernel_collective.rs`, cpu-c `kernel_collective_emit_agrees.rs` |
| 23 | Block collectives: block.sum/max/min/broadcast/prefix_sum | Partial -> **Yes** | `Stmt::Collective` shuffle/vote/ballot/reduce/scan/broadcast (0.42.3; cpu-c 0.42.4); `tests/kernel_collective.rs`, cpu-c `kernel_collective_emit_agrees.rs`; Block-scope reduce/scan/broadcast |
| 36 | Layouts: Layout algebra: Layout(shape,stride) nested/hierarchical; composition, coalesce, complemen | No -> **Yes** | `unpopped-vocab/src/layout_algebra.rs`: hierarchical `IntTuple`/`Layout`, `composition`, `complement`, `logical_divide`, `logical_product`, `coalesce`, inverses, `swizzle`, `select/take/slice/tile_to_shape` with a property-test suite (U6, ROADMAP: done, issue #117 closed 2026-11-02) |
| 37 | Layouts: Swizzle / ComposedLayout, make_swizzle, make_ldmatrix_swizzle (bank-conflict avoidance, TM | No -> **Partial** | `swizzle` in `layout_algebra` (a functor over layouts); no shared-memory tile consumer, no `ComposedLayout` with smem atoms |
| 39 | Layouts: LayoutTensor ops: tile, distribute(thread layout), vectorize, slice, copy_from, async copy | No -> **Partial** | `unpopped-vocab/src/layout_algebra.rs`: hierarchical `IntTuple`/`Layout`, `composition`, `complement`, `logical_divide`, `logical_product`, `coalesce`, inverses, `swizzle`, `select/take/slice/tile_to_shape` with a property-test suite (U6, ROADMAP: done, issue #117 closed 2026-11-02); `slice`, `tile_to_shape`; `distribute`/`vectorize`/`copy_from` not built |
| 44 | Dtypes: float4_e2m1fn, float6_e2m3fn, float6_e3m2fn as DType | Partial -> **Yes** | `ElementKind::{F4E2M1,F6E2M3,F6E3M2}`: storage and key identity (0.29.0), `Cast` rows (0.31.7); stored-result layout still open |
| 60 | Atomics: fence(ordering) standalone | No -> **Yes** | `Stmt::Fence` with order and scope (0.42.8); `tests/kernel_fence.rs`; no proxy/async/tensormap fences |
| 63 | Reductions: Arg-reductions / custom (non-monoid) combine functions as user lambdas (`warp.reduce` take | No -> **Partial** | `ReduceOp::{Sum,Mean,Max,Min,Prod,And,Or,Xor,MaxNum,MinNum,ArgMax,ArgMin}`; `tests/reduce_bit_ops.rs`, `reduce_maxnum.rs`, `reduce_argmax.rs`, `kernel_reduction_agrees.rs`; arg-reductions yes; a user-lambda combine is not |
| 64 | Control flow: Data-dependent loops, early exit/return, while, runtime if/else (not select) with side eff | No -> **Yes** | `Stmt::{If,For,While,Break,Continue,Return}` and `Let`/`Assign` locals; `tests/kernel_layer.rs`, `kernel_resume_agrees.rs`, `kernel_eval_semantics.rs` |
| 65 | Control flow: Local mutable variables / registers, in-kernel arrays, scratch state across loop iteration | No -> **Yes** | `MemorySpace::Private` per-thread arrays (oracle: `UninitializedPrivate`, `tests/kernel_launch.rs`); cpu-c refuses them; oracle only for arrays; locals everywhere |
| 66 | Control flow: Function calls / `@always_inline` helper kernels, closures captured in `foreach`/`vectoriz | No -> **Partial** | `Stmt::Call`, `Module`, `check_module`, run in the oracle and spelled in cpu-c; `tests/kernel_module_run.rs`, `kernel_call_shape.rs`; no closures |
| 67 | Control flow: Bounds guards / predicated tails (tile remainder, `if tid < n`) | Partial -> **Yes** | `Stmt::{If,For,While,Break,Continue,Return}` and `Let`/`Assign` locals; `tests/kernel_layer.rs`, `kernel_resume_agrees.rs`, `kernel_eval_semantics.rs` |

### Slang (`docs/ir-union-surveys/slang.md`): 26 rows changed of 61

| # | Capability | Status | Evidence at `a634b09` |
|---|---|---|---|
| 1 | Memory: `groupshared` / workgroup array (named, sized, per-block) | No -> **Yes** | `Place::Alloc` in `MemorySpace::Shared` with a static `AllocDecl`; `tests/kernel_launch.rs`, `kernel_atomic_shared.rs`; cpu-c spells it |
| 2 | Memory: Dynamic shared memory size / opaque scratch | No -> **Partial** | `Extent::Param`: a dynamic extent from a scalar argument (`kernel_exec.rs`); no opt-in size attribute |
| 5 | Memory: Append/Consume structured buffer (atomic counter queue) | No -> **Partial** | fetch-add hands each thread a distinct slot (`tests/kernel_atomic_fetch.rs`): an append-queue counter, no `Append/Consume` resource |
| 11 | Memory: Coherence/cache qualifiers (`LoadCoherent`, `globallycoherent`, Vulkan memory model `vk_me | No -> **Partial** | `Stmt::AtomicRmw`/`AtomicCas` with `old`, order and scope (0.41.x-0.42.7); `tests/kernel_atomic_{rmw,fetch,cas,order,shared}.rs`; memory order and scope; no coherence qualifiers |
| 13 | Sync: Workgroup barrier `GroupMemoryBarrierWithGroupSync` (Fuel: 36 files) | No -> **Yes** | `Stmt::Barrier` at `Subgroup` and `Block` scope + the uniformity rule (`kernel_uniform.rs`); `tests/kernel_barrier.rs`, `kernel_barriers_run.rs` |
| 14 | Sync: Memory-only barriers: `GroupMemoryBarrier`, `DeviceMemoryBarrier[WithGroupSync]`, `AllMemo | No -> **Yes** | `Stmt::Fence` with order and scope (0.42.8); `tests/kernel_fence.rs`; no proxy/async/tensormap fences; `GroupMemoryBarrier` = Fence(Block), `DeviceMemoryBarrier` = Fence(Device), `...WithGroupSync` = Barrier |
| 15 | Sync: Per-thread control flow with barriers (uniformity analysis, `Treat Values as Uniform`) | No -> **Yes** | `kernel_uniform.rs` (barrier-uniformity analysis); `tests/kernel_barrier.rs` |
| 16 | Sync: Execution-model queries: `SV_GroupThreadID`, `SV_GroupID`, `SV_DispatchThreadID`, `SV_Grou | Partial -> **Yes** | `ExecId::{ThreadIdx,BlockIdx,BlockDim,GridDim}` on three axes (`exec.rs`); `tests/kernel_exec.rs`, `kernel_launch.rs`; SV_* ids are arithmetic over thread/block ids |
| 17 | Sync: `[numthreads(x,y,z)]`, generic numthreads, `[WaveSize(n)]` | No -> **Yes** | `LaunchShape { block, grid }` (the `[numthreads]` equivalent) and the subgroup range |
| 18 | Wave: Reductions `WaveActiveSum/Product/Min/Max/BitAnd/BitOr/BitXor/CountBits/AllTrue/AnyTrue/Al | No -> **Partial** | `Stmt::Collective` shuffle/vote/ballot/reduce/scan/broadcast (0.42.3; cpu-c 0.42.4); `tests/kernel_collective.rs`, cpu-c `kernel_collective_emit_agrees.rs`; Sum/Max/Min/And/Or/Xor and any/all; no product, no `CountBits` |
| 19 | Wave: Prefix: `WavePrefixSum/Product/Min/Max/BitAnd/..`, `WavePrefixCountBits` | No -> **Partial** | `Stmt::Collective` shuffle/vote/ballot/reduce/scan/broadcast (0.42.3; cpu-c 0.42.4); `tests/kernel_collective.rs`, cpu-c `kernel_collective_emit_agrees.rs`; scan Sum/Max/Min/And/Or/Xor; no product |
| 20 | Wave: Shuffles/broadcast: `WaveReadLaneAt/First`, `WaveBroadcastLaneAt`, `WaveShuffle`, `WaveGet | No -> **Yes** | `Stmt::Collective` shuffle/vote/ballot/reduce/scan/broadcast (0.42.3; cpu-c 0.42.4); `tests/kernel_collective.rs`, cpu-c `kernel_collective_emit_agrees.rs` |
| 21 | Wave: Ballot/vote/match: `WaveActiveBallot`, `WaveMatch`, lane masks (`WaveGetLaneEq/Ge/Gt/Le/Lt | No -> **Partial** | `Stmt::Collective` shuffle/vote/ballot/reduce/scan/broadcast (0.42.3; cpu-c 0.42.4); `tests/kernel_collective.rs`, cpu-c `kernel_collective_emit_agrees.rs`; ballot yes, `WaveMatch` no |
| 24 | Wave: `WaveGetNumWaves`/`WaveGetWaveIndex` | No -> **Yes** | `ExecId::{LaneId,SubgroupId,SubgroupSize}`; `tests/kernel_exec.rs` (no `sm_id`, clock or other special registers) |
| 28 | Atomics: `Interlocked{Add,And,Or,Xor,Min,Max,Exchange,CompareExchange,CompareStore}` on int32/int64 | Partial -> **Yes** | `Stmt::AtomicRmw`/`AtomicCas` with `old`, order and scope (0.41.x-0.42.7); `tests/kernel_atomic_{rmw,fetch,cas,order,shared}.rs` |
| 29 | Atomics: `Atomic<T>` with MemoryOrder {Relaxed,Acquire,Release,AcquireRelease,SeqCst}, load/store/e | No -> **Partial** | order+scope RMW (an `Or` of 0 reads); no atomic load/store statements; a spin on one is a named `CrossBlockWait` (0.42.7) |
| 31 | Atomics: 64-bit atomics (HLSL SM6.6, SPIR-V Int64Atomics) | No -> **Yes** | `Stmt::AtomicRmw`/`AtomicCas` with `old`, order and scope (0.41.x-0.42.7); `tests/kernel_atomic_{rmw,fetch,cas,order,shared}.rs`; integer atomics on `I64`/`U64` elements |
| 32 | Atomics: Atomics on groupshared | No -> **Yes** | `Stmt::AtomicRmw`/`AtomicCas` with `old`, order and scope (0.41.x-0.42.7); `tests/kernel_atomic_{rmw,fetch,cas,order,shared}.rs`; `tests/kernel_atomic_shared.rs` |
| 35 | Dtypes: Vector and matrix types (`float4`, `float3x3`, swizzles, vector ops, `mul`, `dot`, `cross` | No -> **Partial** | `ScalarExpr::{Pack,Extract,BitcastPacked}` and the packed `Select` mask (0.31.x); no vector arithmetic ops beyond lane-wise `Select` |
| 39 | Dtypes: Bitcast (`asfloat/asuint/asint/asdouble`), `reinterpret<T>` | Partial -> **Yes** | `ScalarExpr::Bitcast` (0.24.0) |
| 42 | Control flow: Loops (`for/while/do`), dynamic trip counts, `break/continue`, multi-level break, early re | No -> **Yes** | `Stmt::{If,For,While,Break,Continue,Return}` and `Let`/`Assign` locals; `tests/kernel_layer.rs`, `kernel_resume_agrees.rs`, `kernel_eval_semantics.rs` |
| 43 | Control flow: Divergent `if/else` with side effects (stores/atomics in branches) | Partial -> **Yes** | `Stmt::{If,For,While,Break,Continue,Return}` and `Let`/`Assign` locals; `tests/kernel_layer.rs`, `kernel_resume_agrees.rs`, `kernel_eval_semantics.rs` |
| 44 | Control flow: Function calls, recursion-free helper functions, `inline` | No -> **Yes** | `Stmt::Call`, `Module`, `check_module`, run in the oracle and spelled in cpu-c; `tests/kernel_module_run.rs`, `kernel_call_shape.rs` |
| 50 | Numerics: Fast vs precise (`-fp-mode`/`FloatingPointMode`), per-function precision; `precise` qualif | No -> **Partial** | `ScalarExpr::Qualified` rounding/ftz/approx qualifiers (0.25.0); `docs/ir-numerics-qualifiers.md` |
| 51 | Numerics: Rounding mode per op (RTZ/RTE on conversions, `OpFAdd` rounding decoration, cvt.rn) | No -> **Yes** | `ScalarExpr::Cast`/`Bitcast` with `RoundMode` and `Saturation` (0.24.0); `docs/ir-typed-expressions.md`; rounding on `Cast` and `Qualified` |
| 55 | Vulkan/SPIR-V: Subgroup size control / required subgroup size, `[WaveSize]` | No -> **Yes** | `LaunchShape::subgroup` range (`with_subgroup`) and `SubgroupSize`; `tests/kernel_exec.rs` |

### ThunderKittens (`docs/ir-union-surveys/thunderkittens.md`): 10 rows changed of 58

| # | Capability | Status | Evidence at `a634b09` |
|---|---|---|---|
| 4 | Tile types: Shared tile `st<T,R,C,swizzle,swizzle_bytes>`; 0/32/64/128B swizzle, auto-selected | No -> **Partial** | `swizzle` in `layout_algebra` (a functor over layouts); no shared-memory tile consumer, no `ComposedLayout` with smem atoms |
| 10 | Memory spaces: Register / shared / tensor-memory / global as distinct allocation classes | No -> **Partial** | `Place::Alloc` in `MemorySpace::Shared` with a static `AllocDecl`; `tests/kernel_launch.rs`, `kernel_atomic_shared.rs`; cpu-c spells it; global/shared/private/constant allocation classes; no tensor memory |
| 12 | Scope: thread / warp / warpgroup (4 warps) / block / grid scope on each op | No -> **Partial** | `LaunchShape::cluster` + `Cluster` barrier/atomic/fence scope (0.42.15); `tests/kernel_cluster.rs`; no cluster ids, no distributed shared memory; scopes `Subgroup`/`Block`/`Cluster`/`Device`; no warpgroup |
| 13 | Scope: Warp role/specialization (consumer = low warps, producer = high warps; `warpid()`, `warpgr | No -> **Partial** | `ExecId::{LaneId,SubgroupId,SubgroupSize}`; `tests/kernel_exec.rs` (no `sm_id`, clock or other special registers); branch on `SubgroupId`; no `setmaxnreg`, no named barriers |
| 14 | Scope: Leader election (`elect_leader`, "only lane 0 issues") | No -> **Partial** | `ExecId::{LaneId,SubgroupId,SubgroupSize}`; `tests/kernel_exec.rs` (no `sm_id`, clock or other special registers); "only lane 0 issues" is `if LaneId == 0` |
| 29 | Sync: Block / warp sync, cluster barrier, fences (`fence.proxy.async`) | No -> **Partial** | `LaunchShape::cluster` + `Cluster` barrier/atomic/fence scope (0.42.15); `tests/kernel_cluster.rs`; no cluster ids, no distributed shared memory; block/warp/cluster barrier + `Fence`; no proxy fences |
| 36 | Reduction: Warp-level reduction across lanes inside the group op (shuffle) is implicit | Partial -> **Yes** | `Stmt::Collective` shuffle/vote/ballot/reduce/scan/broadcast (0.42.3; cpu-c 0.42.4); `tests/kernel_collective.rs`, cpu-c `kernel_collective_emit_agrees.rs` |
| 38 | Conversion: shared<->register, global<->register direct loads/stores | No -> **Partial** | `Place::Alloc` in `MemorySpace::Shared` with a static `AllocDecl`; `tests/kernel_launch.rs`, `kernel_atomic_shared.rs`; cpu-c spells it; element-level loads/stores between spaces; no tile-level moves |
| 52 | Dtypes: fp4 e2m1 (packed `fp4e2m1_2`; raw rejected on SM10X) | No -> **Partial** | `ElementKind::{F4E2M1,F6E2M3,F6E3M2}`: storage and key identity (0.29.0), `Cast` rows (0.31.7); stored-result layout still open |
| 53 | Dtypes: Packed 2-element register types (`T2`: bf16_2, half_2, float2) | No -> **Partial** | `ScalarExpr::{Pack,Extract,BitcastPacked}` and the packed `Select` mask (0.31.x); no vector arithmetic ops beyond lane-wise `Select` |

### Triton (`docs/ir-union-surveys/triton.md`): 21 rows changed of 70

| # | Capability | Status | Evidence at `a634b09` |
|---|---|---|---|
| 1 | Program model: `tl.program_id(axis)`, `tl.num_programs(axis)` (grid of SPMD block programs) | No -> **Yes** | `ExecId::{ThreadIdx,BlockIdx,BlockDim,GridDim}` on three axes (`exec.rs`); `tests/kernel_exec.rs`, `kernel_launch.rs` |
| 4 | Program model: Device function calls (`tt.call`, `@triton.jit` helper fns, `combine_fn` lambdas) | No -> **Partial** | `Stmt::Call`, `Module`, `check_module`, run in the oracle and spelled in cpu-c; `tests/kernel_module_run.rs`, `kernel_call_shape.rs`; helper functions yes; `combine_fn` lambdas no |
| 20 | Reduce: `tl.sum/max/min/prod? (no prod; reduce combine_fn)/argmax/argmin/any/all/reduce_or/xor_sum | Partial -> **Yes** | `ReduceOp::{Sum,Mean,Max,Min,Prod,And,Or,Xor,MaxNum,MinNum,ArgMax,ArgMin}`; `tests/reduce_bit_ops.rs`, `reduce_maxnum.rs`, `reduce_argmax.rs`, `kernel_reduction_agrees.rs`; sum/max/min/prod/argmax/argmin/any/all/or/xor |
| 27 | Atomics: `atomic_add/max/min` (returns OLD value) with `sem` (acquire/release/acq_rel/relaxed) and  | Partial -> **Yes** | `Stmt::AtomicRmw`/`AtomicCas` with `old`, order and scope (0.41.x-0.42.7); `tests/kernel_atomic_{rmw,fetch,cas,order,shared}.rs` |
| 28 | Atomics: `atomic_cas`, `atomic_xchg`, `atomic_and`, `atomic_or`, `atomic_xor` | No -> **Yes** | `Stmt::AtomicRmw`/`AtomicCas` with `old`, order and scope (0.41.x-0.42.7); `tests/kernel_atomic_{rmw,fetch,cas,order,shared}.rs` |
| 29 | Atomics: `atomic_load`, `atomic_store`, `atomic_poll(expected, sem, scope, timeout_ns)` (inter-prog | No -> **Partial** | order+scope RMW (an `Or` of 0 reads); no atomic load/store statements; a spin on one is a named `CrossBlockWait` (0.42.7) |
| 32 | Dtypes: int1/int8/16/32/64, uint8/16/32/64 | Partial -> **Yes** | `ElementKind::{I8,I16,I32,I64,U8,U16,U32,U64,Bool}` |
| 36 | Dtypes: fp4: `float4e2m1x2` (two e2m1 packed in a byte) | No -> **Partial** | `ElementKind::{F4E2M1,F6E2M3,F6E3M2}`: storage and key identity (0.29.0), `Cast` rows (0.31.7); stored-result layout still open |
| 37 | Casts: `tl.cast(x, dtype, fp_downcast_rounding={"rtne","rtz"}, bitcast=False)` | No -> **Yes** | `ScalarExpr::Cast`/`Bitcast` with `RoundMode` and `Saturation` (0.24.0); `docs/ir-typed-expressions.md` |
| 38 | Casts: `tt.fp_to_fp` rounding attr, `tl.convert saturation / overflow_mode` (UNVERIFIED: overflow | No -> **Yes** | `ScalarExpr::Cast`/`Bitcast` with `RoundMode` and `Saturation` (0.24.0); `docs/ir-typed-expressions.md` |
| 40 | Packed/vector: `tl.map_elementwise(scalar_fn, *args, pack)`, `ttng.packed_arith` | No -> **Partial** | `ScalarExpr::{Pack,Extract,BitcastPacked}` and the packed `Select` mask (0.31.x); no vector arithmetic ops beyond lane-wise `Select` |
| 41 | Control: `for i in tl.range(...)` / `tl.static_range` loops with loop-carried tiles (the heart of e | No -> **Yes** | `Stmt::{If,For,While,Break,Continue,Return}` and `Let`/`Assign` locals; `tests/kernel_layer.rs`, `kernel_resume_agrees.rs`, `kernel_eval_semantics.rs` |
| 42 | Control: `if`/`while` on scalar (program-uniform) predicates, early return, data-dependent loop bou | No -> **Yes** | `Stmt::{If,For,While,Break,Continue,Return}` and `Let`/`Assign` locals; `tests/kernel_layer.rs`, `kernel_resume_agrees.rs`, `kernel_eval_semantics.rs` |
| 44 | Schedule: `num_warps` (default 4) | No -> **Partial** | `LaunchShape { block, grid }` (the `[numthreads]` equivalent) and the subgroup range; block size is in `LaunchShape`; `num_warps` as a compile hint is not |
| 46 | Schedule: `num_ctas` (cluster size, SM90+) | No -> **Partial** | `LaunchShape::cluster` + `Cluster` barrier/atomic/fence scope (0.42.15); `tests/kernel_cluster.rs`; no cluster ids, no distributed shared memory |
| 52 | Coop: Shared-memory buffers (`ttg.local_alloc/load/store/dealloc`, memdesc_index/subslice/trans/ | No -> **Partial** | `Place::Alloc` in `MemorySpace::Shared` with a static `AllocDecl`; `tests/kernel_launch.rs`, `kernel_atomic_shared.rs`; cpu-c spells it; `local_alloc/load/store`; no memdesc views |
| 53 | Coop: Shared-mem gather/scatter/atomic scatter (`ttg.local_gather/local_scatter/local_atomic_sca | No -> **Yes** | `Place::Alloc` in `MemorySpace::Shared` with a static `AllocDecl`; `tests/kernel_launch.rs`, `kernel_atomic_shared.rs`; cpu-c spells it; indexed shared loads/stores/atomics |
| 54 | Coop: CTA barriers `tl.debug_barrier`, `ttg.barrier` (CTA or warp scope), `__syncthreads` | No -> **Yes** | `Stmt::Barrier` at `Subgroup` and `Block` scope + the uniformity rule (`kernel_uniform.rs`); `tests/kernel_barrier.rs`, `kernel_barriers_run.rs` |
| 60 | Coop: Clusters / multi-CTA (`num_ctas`, cluster_barrier, multicast TMA, DSMEM, CLC launch contro | No -> **Partial** | `LaunchShape::cluster` + `Cluster` barrier/atomic/fence scope (0.42.15); `tests/kernel_cluster.rs`; no cluster ids, no distributed shared memory |
| 61 | Coop: Layouts: blocked / mma / dot_operand / shared (swizzle, linear layouts), `ttg.convert_layo | No -> **Partial** | `unpopped-vocab/src/layout_algebra.rs`: hierarchical `IntTuple`/`Layout`, `composition`, `complement`, `logical_divide`, `logical_product`, `coalesce`, inverses, `swizzle`, `select/take/slice/tile_to_shape` with a property-test suite (U6, ROADMAP: done, issue #117 closed 2026-11-02); layout vocabulary; no mma/dot-operand/linear-layout conversion op |
| 62 | Coop: Warp/block collectives (shuffle, ballot, vote) | Partial -> **Yes** | `Stmt::Collective` shuffle/vote/ballot/reduce/scan/broadcast (0.42.3; cpu-c 0.42.4); `tests/kernel_collective.rs`, cpu-c `kernel_collective_emit_agrees.rs` |

