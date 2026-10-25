> **Raw survey output, 2026-10-24/25. Read `docs/ir-union-design.md` first; where it and this file disagree, the design doc wins.** Written by a read-only Sonnet agent from the official documentation (every row carries its URL; rows it could not confirm from a primary page are marked UNVERIFIED) and from `ir.rs` at `origin/main` `a575e66`. The design doc re-checked the load-bearing IR claims against the source; this file as a whole has not been reviewed line by line.

# CuTe (C++ 3.x and CuTe DSL 4.x) vs Unpopped IR: gap survey

IR read at origin/main (fetched 2026-10-24): `crates/unpopped/src/ir.rs` (4300 lines), `docs/idiom-lifting-design.md`, unpopped-vocab `layout.rs`, `shape_expr.rs`, `structure_key.rs`, `element.rs`. Line numbers refer to those blobs.

Base URLs (primary, NVIDIA/cutlass + docs.nvidia.com):
- L1 = https://raw.githubusercontent.com/NVIDIA/cutlass/main/media/docs/cpp/cute/01_layout.md
- L2 = .../cute/02_layout_algebra.md ; T3 = .../cute/03_tensor.md ; A4 = .../cute/04_algorithms.md
- MMA = .../cute/0t_mma_atom.md ; PRED = .../cute/0y_predication.md ; TMAT = .../cute/0z_tma_tensors.md ; GEMM = .../cute/0x_gemm_tutorial.md ; PIPE = .../cpp/pipeline.md
- DSLcore = https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute.html
- DSLarch = .../cute_dsl_api/cute_arch.html ; DSLtc5 = .../cute_dsl_api/cute_nvgpu_tcgen05.html ; DSLcp = .../cute_dsl_api/cute_nvgpu_cpasync.html ; DSLwg = .../cute_dsl_api/cute_nvgpu_warpgroup.html ; DSLwarp = .../cute_dsl_api/cute_nvgpu_warp.html ; DSLpipe = .../cute_dsl_api/pipeline.html
- DSLintro = .../cute_dsl_general/dsl_introduction.html ; DSLopts = .../cute_dsl_general/dsl_jit_compilation_options.html

What the IR is today (verified): `OpDef` = one `body: ScalarExpr` (leaves Input/Const/Param/Reduced/Coord, Add/Sub/Mul/Div/Unary/Binary/Select; ir.rs:63-176) + one `Access` (Elementwise, Reduction, RowReduce, Contraction, Scan, Window, RowSort, Im2Col; ir.rs:1607-1863) + per-input `View` (Identity/Permute/Broadcast/Reshape; ir.rs:1973), `ReadIndex` (Direct/Indexed gather; 2099), `BaseOffset` (Zero/Runtime; 2158), single-output `WriteIndex` (Direct/ScatterIndexed with Assign/AtomicAdd/Max/Min; 2256). Rank <= MAX_RANK 8 (structure_key.rs:48). Layout facts in the key are classes (Contiguity Contig/InnerContig/Strided/Broadcast, broadcast mask, flip, alignment, VecWidth), NOT stride values. Numeric extents are deliberately abstracted away from the key (ir.rs:2005-2009). `layout.rs` is only GEMM SKU tags (Rcr/Rrr, ArchSku, Epilogue). `shape_expr.rs` DimExpr = Extent/Const/Param/+,-,*,floor-div; ShapeExpr = `SameAs(u8)` only (shape_expr.rs:86-108).
Grep controls (ir.rs, same blob): `__syncthreads|Barrier|Shuffle|Ballot|Vote|Fence|Swizzle|AddressSpace|MemSpace|gmem|smem` as IR constructs = 0 hits (smem/atomic words appear only in doc comments and WriteCombine). Positive control: same grep finds `AtomicAdd` (ir.rs:2206) and `Reduced` leaf.
Doc conflict: idiom-lifting-design.md section 1a (lines 69-85) says "the IR never models cooperation". The owner ruling reverses this; that section must be replaced (not caveated).

## 1. Capability table

Legend: Yes / Partial / No. "absent: X = 0" means grep of ir.rs (or named file) found nothing.

### 1a. Layout algebra

| Category | Capability | IR | Evidence | Source | IR addition needed |
|---|---|---|---|---|---|
| Layout | Layout = (Shape, Stride) pair with explicit integer strides as first-class IR value | No | key carries stride CLASSES only (structure_key.rs:98-107, 961-970); View has no strides | L1 | `Layout{shape: IntTuple, stride: IntTuple}` value type usable on every operand and in kernel-internal buffers |
| Layout | Hierarchical/nested IntTuple shapes and strides (`(_5,(_2,_2)):(_16,(_80,_4))`), depth, congruence, profiles | No | View::Permute/Broadcast/Reshape are flat rank-N (ir.rs:1973-2000); DimExpr scalar only; MAX_RANK 8 flat | L1, L2 | Recursive `IntTuple<T>` for shape/stride/coord; keep rank cap per leaf, not per depth |
| Layout | Static (compile-time `Int<N>`) vs dynamic integers mixed per mode | Partial | Param vs Const exist for scalars; numeric extents and strides not in IR/key (ir.rs:2005-2009 "caller precondition") | L1, L2 | Per-mode `Static(i64)|Dynamic(param)` in shape and stride; static strides enter the key |
| Layout | Layout as function idx->offset; size/rank/depth/cosize | No | no layout value, hence no queries | L1 | `size/rank/depth/cosize` as DimExpr nodes over layout values |
| Layout | crd2idx / idx2crd (colexicographic, natural, R-D, h-D coordinates) | Partial | only row-major unravel inside emitter; `Coord(axis)` leaf is float, Elementwise only (ir.rs:96-131) | L1 | Integer-typed `CrdToIdx/IdxToCrd(layout, coord)` index-expression nodes |
| Layout | Structural ops: get/select/take/flatten/group/append/prepend/replace/slice/dice/filter/filter_zeros | No | absent: no layout value | L1, DSLcore | Pure layout-expression nodes evaluated at plan time |
| Layout algebra | coalesce (and by-mode coalesce) | Partial | key does "size-1 squeeze + adjacent-contiguous merge" as canonicalisation (structure_key.rs:40 note), not an IR op | L2 | `Coalesce(layout, profile?)` |
| Layout algebra | composition (A o B), by-mode composition with tilers | No | absent: composition/Compose = 0; View only permutes/broadcasts | L2 | `Compose(A, B)` with divisibility side conditions checked at plan time |
| Layout algebra | complement(layout, cotarget) | No | absent | L2 | `Complement(layout, cotarget)` |
| Layout algebra | logical_divide / zipped / tiled / flat divide (tiling with tile + rest modes, round-up residue) | No | Window/Im2Col are fixed-function; no generic divide | L2 | `LogicalDivide(A, tiler)` + 3 reshaped forms (Zipped/Tiled/Flat) |
| Layout algebra | logical_product / blocked / raked / zipped / tiled / flat product | No | absent | L2 | `LogicalProduct(A, B)` + Blocked/Raked |
| Layout algebra | left_inverse / right_inverse; max_common_layout/vector (basis of auto-vectorisation) | No | VecWidth is derived from alignment+innermost stride only (structure_key.rs:111-125) | DSLcore | `RightInverse/LeftInverse`, `MaxCommonVector(src,dst)` |
| Layout algebra | make_ordered_layout, make_identity_layout, tile_to_shape, recast_layout (element bit-width change) | No | `View::Reshape` is carried for keying only, no emit (ir.rs:1968-1971) | DSLcore | `MakeOrdered`, `TileToShape`, `RecastLayout(bits_in,bits_out)` |
| Layout | View::Reshape across non-contiguous producers (CuTe composes) | Partial | Reshape = contiguous pass-through, rank-change emit deferred (ir.rs:1968-1971) | L2 | Reshape as Compose with a shape-only layout |
| Swizzle | `Swizzle<B,M,S>` XOR functor; ComposedLayout(swizzle o offset o layout); smem swizzle atoms (32/64/128B, SmemLayoutAtomKind) | No | absent: Swizzle = 0 | DSLcore, DSLwg, DSLtc5 | `Swizzle{bits,base,shift}` as an address-function stage in a layout; `ComposedLayout` |
| Shapes | Tensor shape expressions beyond SameAs (concat, tile-ceil-div, hierarchical) | Partial | ShapeExpr only `SameAs` (shape_expr.rs:105-108); DimExpr floor-div only, no ceil-div / min / max / mod | L2, PRED | Add `CeilDiv`, `Mod`, `Min`, `Max`, `Tuple`, `Select` ShapeExpr nodes |

### 1b. Tensor, partitioning, memory spaces

| Category | Capability | IR | Evidence | Source | IR addition needed |
|---|---|---|---|---|---|
| Tensor | Tensor = iterator/engine + layout; ArrayEngine (owning, register) vs ViewEngine | No | operands are bare strided buffers; no kernel-local tensor values (single OpDef body is per-coordinate scalar) | T3 | `Tensor{space, elem, layout, base}` SSA value incl. owning local tensors |
| Memory space | Pointer tagged with address space: gmem/smem/rmem/tmem/dsmem/cmem/generic | No | absent: AddressSpace = 0 | T3, DSLcore (AddressSpace enum) | `Space` enum on every tensor/pointer: Global, Shared, Reg, Tmem, Dsmem, Const, Generic |
| Memory | Static shared-memory allocation / `cute.struct` SharedStorage with MemRange and Align; dynamic smem size | No | no smem object in IR; cooperative smem schedules are emitter-internal (plan.rs:118-120, 690) | DSLcore, DSLarch (dynamic_smem_size) | `SharedStruct{fields, align}` and `AllocShared` |
| Memory | Register fragments: make_fragment_like / make_rmem_tensor / TensorSSA (immutable vector value) | No | no vector/fragment value type | T3, DSLcore | Fragment/vector value types with static layout |
| Memory | Register-level cache/eviction hints on load/store (ld.global cache modes, evict policies, memory ordering) | No | absent | DSLarch (`load`, `store`), DSLcp (LoadCacheMode) | `CacheHint`, `MemOrder` attributes on read/write |
| Tensor ops | Slicing with `_` (keep mode), domain_offset, recast_tensor/recast_ptr (type punning) | Partial | `BaseOffset::Runtime` is a flat element offset only (ir.rs:2158); no slice-with-keep, no recast | T3, DSLcore | `Slice(tensor, coord-with-underscore)`, `Recast(tensor, elem)` |
| Partition | local_tile (CTA tiling with Step projections, e.g. `Step<_1,X,_1>`) | No | absent; iteration space is implicit per-coordinate | T3, GEMM | `LocalTile(tensor, tiler, coord, proj)` |
| Partition | local_partition / outer_partition / inner_partition by thread layout | No | absent | T3 | `LocalPartition(tensor, thr_layout, thr_idx, proj)` |
| Partition | TV-layout (thread,value)->coord and `composition(tensor, tv)` then slice by thread | No | absent; make_layout_tv | T3, DSLcore | `TvLayout` and thread-index IR value |
| Partition | thr_mma.partition_A/B/C, partition_fragment_*, make_fragment_C | No | absent | GEMM, DSLtc5 | `PartitionMma{A,B,C}` bound to a TiledMma value |
| Partition | thr_copy.partition_S/D, retile (reshape a register fragment to a copy atom's expectation) | No | absent | GEMM (partition_S/D), retile UNVERIFIED (not on fetched page) | `PartitionCopy{S,D}`, `Retile` |
| Index | Thread / block / cluster / warp identity values (thread_idx, block_idx, cluster_idx, lane_idx, warp_idx, grid_dim, smid ...) | No | iteration coordinates are implicit; only `Coord(axis)` float leaf | DSLarch | `ThreadIdx/BlockIdx/ClusterIdx/LaneId/WarpId/GridDim` integer leaves |

### 1c. Atoms (copy, MMA)

| Category | Capability | IR | Evidence | Source | IR addition needed |
|---|---|---|---|---|---|
| MMA | MMA_Atom + MMA_Traits (ValTypeA/B/C/D, Shape_MNK, ThrID, A/B/C layouts) | No | only `Access::Contraction` rank-2 single-K uniform dtype (ir.rs:1675); `AccumSpec` has 1 variant WideFloat, non_exhaustive (ir.rs:1928) | MMA | `MmaAtom{shape_mnk, dtypes, thr_layout, a/b/c layouts, target-feature}` |
| MMA | TiledMMA = atom layout x permutation tiler; ThrMMA | No | absent | MMA | `TiledMma{atom, atom_layout, perm_tiler}` |
| MMA | cute.gemm over mode patterns: (V),(M)x(N),(M,K)x(N,K),(V,M,K)x(V,N,K) | No | Contraction is whole-op, not an operation on tensor values | A4 | `Gemm(acc, a, b, atom?)` statement on tensor values |
| MMA | SM80 `mma.sync` (F16/BF16/TF32/FP8 sm89/INT8) | Partial | element kinds exist incl. Tf32 via MathPrecision, Fp8, I8 (element.rs:1050-1110, 1314); doc says Contraction is SIMT only (idiom doc line 58) and mma "lifts to" Contraction | DSLwarp | add `AccumSpec` tensor-core variants + explicit MMA atom as above |
| MMA | SM90 WGMMA (warpgroup of 128 thr, M=64 N=8..256, A from RMEM or SMEM, smem descriptors, fence/commit_group/wait_group, async) | No | absent; idiom doc line 716 lists wgmma as "not transferable" | DSLwg, MMA | `WarpgroupMma` atom with `OperandSource{Reg,Smem}` and explicit async-group fence/commit/wait |
| MMA | SM100 tcgen05/UMMA (single-thread issue, CtaGroup 1/2, accumulator in TMEM, A from TMEM or SMEM, UMMA smem descriptors, commit to mbarrier) | No | absent: Tmem = 0 | DSLtc5 | `UmmaAtom{cta_group, a_src}`, `Tmem` space, `UmmaCommit(mbar)` |
| MMA | Runtime-settable MMA fields (negate A/B, accumulate flag, scale factors, sparsity metadata, disable-output-lane, collector op) | No | absent | DSLtc5 (Field, CollectorOp) | MMA atom with mutable-field set |
| MMA | Block-scaled MMA: MXF8/F6/F4, NVF4 (UE4M3, 16-elt scale vectors), SM120 warp-level variants | Partial | F8E8M0 scale dtype exists; F8E6M2 reserved; MX element formats f4/f6e2m3/f6e3m2 deliberately NOT in ElementKind (element.rs:1144-1154); no scale operand on Contraction | DSLtc5, DSLwarp | Block-scaled operand type (element fmt + scale tensor + block size) in IR; element formats in an encoding vocabulary |
| MMA | 2:4 structured sparse MMA with metadata operand `[a, e]` | No | absent: Sparse = 0 | DSLtc5 | `Sparse24` operand with metadata tensor |
| MMA | Mixed A/B dtypes (e.g. E2M1 x E4M3) | No | Contraction uniform dtype (ir.rs:1675; OpDef.out_dtype hetero only for outputs) | DSLtc5 (MmaMXF8F6F4Op) | Per-operand dtype in Contraction/MmaAtom |
| MMA | Rank > 2 / multi-K / multi-batch / transposed contraction | Partial | rank-2 single-K; `batched_matmul()` exists (ir.rs:1906) but v1 per task statement is rank-2 | MMA, A4 | general einsum role vectors already planned (ir.rs:1880 note) |
| Copy atom | Copy_Atom/CopyTraits (Src/Dst/Ref layouts), TiledCopy via make_tiled_copy(atom, thr_layout, val_layout), make_tiled_copy_A/B/C, `copy`, `copy_if`, `autovec_copy`, axpby/fill/clear | No | only fixed gather/scatter/elementwise access; no copy statement | GEMM (make_tiled_copy), A4, DSLcore | `Copy(src, dst, atom?, pred?)`, `Fill`, `Axpby` statements and `TiledCopy` value |
| Copy atom | SM80 `cp.async` (G2S, 4/8/16 B, cache mode), `cp.async.commit_group`/`wait_group` | No | absent; idiom doc line 715 "not transferable" | GEMM (cp_async_fence/wait), DSLcp (CopyG2SOp, LoadCacheMode), DSLarch | `AsyncCopyG2S{bytes,cache}` + `AsyncGroupCommit/Wait(n)` |
| Copy atom | `ldmatrix` (8x8x16b, 16x8x8b, 16x16x8b, transpose, x1/x2/x4, unpack) and `stmatrix` | No | absent | DSLwarp | Warp-level `LdMatrix/StMatrix` atoms (or generic `WarpCopy` with fragment layout) |
| Copy atom | TMA tile load/store/reduce (G2S, S2G, S2G reduce), multicast, gather4/scatter4, im2col TMA, descriptor prefetch/update/fence, tensormap copy | No | absent: Tma = 0; Im2Col is a SIMT gather Access, not a TMA op | DSLcp, TMAT | `TmaDescriptor{tensor,smem_layout,box}`, `TmaLoad/Store/Reduce/Multicast/Gather4/Im2col` + descriptor ops |
| Copy atom | TMA coordinate tensors: ArithmeticTuple + ScaledBasis `E<i>` strides (strides whose codomain is a coordinate module, not an integer) | No | strides are conceptually integers; absent | TMAT | Stride type generalised to integer-module (basis elements) |
| Copy atom | Bulk (non-tensor) copies G2S, S2G, S2S (cluster), byte-masked S2G, DSMEM store | No | absent | DSLcp | `BulkCopy{src_space,dst_space,mask?}` |
| Copy atom | TMEM copies: Ld/St 16x64b,16x128b,16x256b,16x32bx2,32x32b, repetition x1..x128, pack/unpack, load-with-reduce (MAX/MIN/ABS), SMEM->TMEM (S2T) | No | absent | DSLtc5 | `TmemLoad/Store/S2T` atoms; TMEM allocate/dealloc ops (alloc not on the DSLtc5 page: UNVERIFIED location) |
| Predication | Identity tensor + comparison with shape -> predicate tensor; stride-0 predicate broadcast over K; `copy_if`; round-up divide with residue masking | Partial | OobPolicy Skip/Clamp/ZeroFill on Indexed gather/scatter only (ir.rs:2061-2078); Window/Im2Col have fixed OOB; Select expression exists | PRED | First-class `Predicate` tensors; `IdentityTensor`; predicated copy/gemm; residue-tile handling |
| Dtype | Sub-byte packed types int4/uint4/bit and fp4/fp6 as MMA operands with pack/unpack, tf32 operand register, packed f16x2/bf16x2/f32x2 arithmetic | Partial | I4/U4/B1 present, Fp8 variants present (element.rs:~1110-1143, enum tail), TF32 via MathPrecision; no vector/packed (x2/x4) value types; f4/f6 absent by ruling | DSLarch (cvt_*, *_packed_*) | Vector value types `Vec<T,N>` + packed ops; fp4/fp6 element encodings |

### 1d. Cooperation, sync, pipelines

| Category | Capability | IR | Evidence | Source | IR addition needed |
|---|---|---|---|---|---|
| Sync | CTA barrier `sync_threads`, named barriers (IDs 0-15, arrive / arrive_and_wait), warp sync with mask | No | absent: Barrier = 0 | DSLarch, DSLpipe (NamedBarrier) | `Barrier{scope,id,count}` statements |
| Sync | Warp collectives: shuffle (idx/up/down/bfly), vote (ballot/any/all/uni), match, warp_redux, elect_one, make_warp_uniform | No | warp reduce only as emitter schedule for Reduction/RowReduce (idiom doc lines 56-57) | DSLarch | Explicit collective nodes `Shuffle/Vote/Match/Redux/ElectOne` with participation mask |
| Sync | Cluster sync (arrive/wait/relaxed), cluster/CTA-in-cluster ids, distributed shared memory | No | absent; idiom doc line 717 | DSLarch, DSLcp (CopyDsmemStoreOp) | `ClusterBarrier`, `ClusterId`, `Dsmem` space, remote-address op |
| Sync | mbarrier: init, init_fence, arrive (also on peer CTA), arrive_and_expect_tx, expect_tx, wait(phase), try_wait, test_wait, conditional_try_wait | No | absent: mbarrier = 0 | DSLarch | `MBarrier` object in smem + ops with phase bit and tx-byte count |
| Sync | Memory fences: acq_rel at cta/cluster/gpu/sys; proxy fences (async/tensormap/generic); tmem view fences | No | absent | DSLarch | `Fence{scope,proxy}` |
| Sync | cp.async / bulk-async group commit and wait_group(N) | No | absent | DSLarch, GEMM | see copy atoms |
| Pipeline | PipelineState (stage index + phase bit, circular buffer) | No | absent | PIPE, DSLpipe | `PipelineState{stages, index, phase}` |
| Pipeline | PipelineAsync/CpAsync/TmaAsync/TmaUmma/UmmaAsync/AsyncUmma/TmaStore/TmaMultiConsumers; producer_acquire/commit, consumer_wait/release; CooperativeGroup(Agent: Thread/Warp/ThreadBlock/Cluster) | No | absent | PIPE (PipelineAsync, TmaAsync only), DSLpipe (rest) | `Pipeline{stages, producer_kind, consumer_kind, groups}` + 4 verbs as IR statements with barrier-lowering left to the emitter |
| Pipeline | Cluster Launch Control dynamic scheduling (PipelineClcFetchAsync); persistent / stream-K tile schedulers | No | absent | DSLpipe | `TileScheduler` node (static persistent, CLC) |
| Specialisation | Warp specialisation: role per warp, `setmaxregister_increase/decrease`, warpgroup_reg_alloc/dealloc | No | idiom doc line 720 lists as non-transferable; absent | DSLarch | `WarpRole{warp_set, body}` regions + register budget attributes |
| Atomics | Atomics: add/and/or/xor/max/min/exch/cas/fmax/fmin, `red` | Partial | only WriteCombine Assign/AtomicAdd/Max/Min on ONE scatter output; integer max/min only; no CAS/exch/and/or/xor, no returned old value, no smem atomics (ir.rs:2202-2211) | DSLarch | General `Atomic{op,space,returns_old}` incl. CAS/exch/bitwise |
| Control | Per-thread control flow: if/for/while with loop-carried values; tile loops (k-tile mainloop); `range_constexpr` unrolled loops | No | OpDef is a single expression + one Access; no statements, loops, locals or mutation | DSLintro (preprocessor turns Python if/loops into IR ops); range_constexpr UNVERIFIED | A statement-level IR: blocks, `For/While/If`, mutable locals / SSA with block args |
| Math | Fast-math intrinsics: exp2, rcp_approx, packed exp2, fma packed, bfe/bfi/brev/clz/popc/mul_hi, cvt_* | Partial | UnaryOp/BinaryOp set large (ir.rs:185-330); integer bit-intrinsics and packed forms not verified | DSLarch | Check UnaryOp for popc/clz/bfe; add packed-vector forms |

### 1e. Launch / JIT-level features (CuTe DSL)

| Category | Capability | IR | Evidence | Source | IR addition needed |
|---|---|---|---|---|---|
| Launch | grid/block/cluster, fallback_cluster, dynamic smem, PDL (`use_pdl`), cooperative launch, min_blocks_per_mp | No | schedules are emitter-chosen (plan::Schedule); no launch config in IR | DSLintro | `LaunchConfig` attributes on kernel (cluster dims, smem bytes, PDL, cooperative, occupancy hint) |
| Host/device | Host function that launches multiple kernels (@jit calling @kernel); dynamic vs compile-time call conventions | No | single-op granularity | DSLintro | Multi-kernel host graph in the IR (or explicitly out of scope; owner to rule) |
| Specialisation | Compile-time (Constexpr) vs runtime args; preprocessor evaluating static control flow | Partial | Const vs Param distinction exists (ir.rs:65-73) | DSLintro (static/dynamic only indirectly stated; Constexpr UNVERIFIED) | Const-evaluated control flow once loops exist |
| JIT | JIT caching, AOT export, TVM-FFI, GPU-arch option, ptxas options, assertions, keep-ptx/cubin/sass, line info | N/A | build-tooling, not IR semantics; arch is TargetId (layout.rs:34-56) | DSLopts, DSLintro | none for IR; compile options are backend metadata |
| Interop | DLPack tensors (PyTorch/JAX) with mark_layout_dynamic / dynamic shape | Partial | runtime shape/stride arrays are launch args; key abstracts extents (ir.rs:2005) | DSLintro (DLPack: yes; mark_layout_dynamic UNVERIFIED) | none beyond layout-with-dynamic-modes above |

## 2. Ten highest-impact gaps (ranked)

1. No statement-level / block-structured kernel body: OpDef = one scalar expression + one Access. Every cooperative or pipelined CuTe kernel needs loops, locals, mutation, tile mainloops, if. Nothing else below is expressible without it.
2. No Layout value (hierarchical Shape:Stride IntTuple, static + dynamic modes) and no algebra (composition, complement, logical divide/product, coalesce, inverses, crd2idx). The key stores stride classes, extents abstracted away.
3. No memory-space-typed tensors (gmem/smem/rmem/tmem/dsmem) and no shared-memory allocation or register fragment objects.
4. No Swizzle<B,M,S>/ComposedLayout and no smem layout atoms (needed for ldmatrix, WGMMA, UMMA, TMA).
5. No tensor partitioning: local_tile, local_partition, TV layouts, thr_mma/thr_copy partition_*, plus no thread/block/cluster/warp identity values.
6. No MMA atom model: Contraction is rank-2/uniform-dtype SIMT with one AccumSpec; missing mma.sync, WGMMA (async group), tcgen05/UMMA with TMEM accumulators, block-scaled, sparse, mixed A/B dtype, runtime MMA fields.
7. No Copy atom model: cp.async, ldmatrix/stmatrix, TMA (tile/multicast/gather4/scatter4/reduce/im2col, descriptors), bulk copies, TMEM Ld/St/S2T, `copy`/`copy_if` statements.
8. No synchronization vocabulary: CTA/named/cluster barriers, mbarrier with phase + tx-count, fences (scope/proxy), async groups, warp shuffle/vote/match/elect.
9. No pipeline abstraction: PipelineState and the producer/consumer variants (Async, CpAsync, TmaAsync, TmaUmma, UmmaAsync), CooperativeGroup, warp specialisation with register reallocation, CLC/persistent schedulers.
10. No first-class predication (identity tensor, predicate tensor, copy_if, round-up residue); only OobPolicy on gather/scatter. Plus general atomics (CAS/exch/bitwise/return-old, smem atomics) and launch config (cluster dims, PDL, cooperative).

## 3. Where the existing layout/view model is wrong or too weak relative to CuTe

1. **View is a closed 4-variant enum, not a layout.** Permute/Broadcast/Reshape/Identity (ir.rs:1973-2000) are three special cases of "a layout with strides"; composition of two Views (permute then broadcast then reshape-of-non-contiguous) has no representation, and `Reshape` is explicitly "recognition + keying only" with no emit for non-contiguous producers (ir.rs:1968-1971). In CuTe, reshape of a strided tensor is composition and works whenever divisibility holds.
2. **The model is flat and rank-capped at 8** (MAX_RANK, structure_key.rs:48; `is_permutation` guard ir.rs:2033). CuTe's hierarchy (e.g. a GEMM operand `((tile_m, rest_m), (tile_k, rest_k))`, TV layouts, `(V,M,K)` fragments) has depth-nested modes; flattening to 8 loses the mode grouping that partitioning algorithms key off, and a depth-3 tile hierarchy exceeds 8 leaves quickly.
3. **Strides are classes, not values, in the key** (Contig/InnerContig/Strided/Broadcast + alignment; structure_key.rs:98-107, 961-970). Swizzled, padded (`bM+Int<1>` in GEMM tutorial), or interleaved smem layouts are all `Strided` and indistinguishable; two kernels with different bank-conflict behaviour hash to one cell. The key cannot carry a swizzle at all. Consequence for "full union": anything whose correctness or performance contract depends on the exact layout (smem atoms, TMA boxes) cannot be keyed.
4. **Negative-stride/flip is a mask bit** (OperandKey flip) while CuTe strides are arbitrary integers (incl. 0 and ScaledBasis basis elements, TMAT). Stride-0 handled as Broadcast only; a non-trivial stride-0 mode inside a hierarchical mode (e.g. predicate tensors with stride 0 over K, PRED) has no representation.
5. **`ReadIndex::Indexed` and `WriteIndex::ScatterIndexed` rewrite exactly one axis with one index operand** (ir.rs:2119, 2274) and are mutually exclusive with View on the same input (ir.rs:2093). CuTe-style gather (TMA gather4, gathered-row GEMM operands, MoE grouped GEMM) composes an indirection with a layout; that composition is unrepresentable.
6. **Coord leaf is float-only and Elementwise-only** (ir.rs:96-131): CuTe identity tensors / predicates are integer coordinates used in every access (including reductions and contractions). The IR cannot compute an integer-valued predicate from a coordinate.
7. **ShapeExpr has one node, `SameAs`** (shape_expr.rs:105-108) and DimExpr has no ceil-div (CuTe's round-up divide, PRED) or mod; padded/rounded tile counts cannot be expressed.
8. **AccumSpec/ReductionAccum are single-variant enums reserved for "tensor-core policies"** (ir.rs:1928, 1949): encoding the MMA instruction choice as an accumulator policy conflates numerics (what precision) with structure (which atom, which thread layout, async or sync, TMEM vs RMEM accumulator). CuTe separates them (MMA_Atom vs math precision). Recommend modelling the atom separately and keeping AccumSpec numerics-only.
9. **Docs contradict the new ruling**: idiom-lifting-design.md section 1a states the IR never models cooperation and the lifter refuses `__shared__/__shfl/__syncthreads` unless an idiom recognizer claims them. Under the owner's ruling the IR must carry these; lifting to Reduction/Contraction can remain an optimisation, not the only path. Lines 714-720 of that doc ("not transferable": cp.async, TMA, wgmma, clusters, warp specialisation) also describe things the IR must now express (with a `transferable:false` capability tag), not exclude.

## Unverified / limits

- Did not read the full DSL `cute.html` (262k chars; only first 100k) nor `cute_arch.html` tail (ldmatrix/stmatrix/TMEM alloc/griddepcontrol not seen there; ldmatrix/stmatrix confirmed on the warp page instead). TMEM alloc/dealloc op location: UNVERIFIED (not on cute_nvgpu_tcgen05 page; `TmemAllocator` listed in pipeline module name list but not described).
- `range_constexpr`, `cute.compile` semantics, `mark_layout_dynamic`, `retile`, `make_tiled_copy_A/B/C`, `smem_ptr_swizzle`, `recast_tensor`, `make_identity_tensor` (the last is in PRED doc): not on the fetched pages; partial UNVERIFIED as noted. SM90 TMA multicast/mbarrier pipeline semantics beyond names: from DSL API pages only, not from the C++ pipeline.md (which covers only PipelineState/Async/TmaAsync).
- UnaryOp/BinaryOp full member lists not enumerated (ir.rs:185-330), so the integer bit-intrinsic status is Partial/unknown.
- Did not check `ops.rs` for already-shipped op constructors beyond grep for smem/barrier words; claims of "No" concern the IR vocabulary (ir.rs), not emitter-internal schedules (plan.rs mentions cooperative smem bitonic sort and blockscan as emitter schedules).
