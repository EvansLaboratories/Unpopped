> **Raw survey output, 2026-10-24/25. Read `docs/ir-union-design.md` first; where it and this file disagree, the design doc wins.** Written by a read-only Sonnet agent from the official documentation (every row carries its URL; rows it could not confirm from a primary page are marked UNVERIFIED) and from `ir.rs` at `origin/main` `a575e66`. The design doc re-checked the load-bearing IR claims against the source; this file as a whole has not been reviewed line by line.

# CUTLASS capability survey vs Unpopped IR

Ref: Unpopped `origin/main` as locally fetched (head a575e66, NOT 1b90fa6: `git fetch` failed with a ref-lock error; local `origin/main` may be stale by a few commits). Files: `crates/unpopped/src/ir.rs` (4300 lines, read: ScalarExpr, Access, AxisRole/ContractionAxes, AccumSpec, ReductionAccum, View, ReadIndex, WriteIndex, OpDef), `crates/unpopped/src/plan.rs` (Schedule enum), `docs/idiom-lifting-design.md`, `crates/unpopped-vocab/src/{ops,element,plan}.rs` (grepped, not read whole).
Abbrev: ir = ir.rs line; idiom = docs/idiom-lifting-design.md line.
CUTLASS sources (primary, fetched): F=github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/functionality.md; B=.../blackwell_functionality.md; G=.../gemm_api_3x.md; E=.../efficient_gemm.md; CL=.../blackwell_cluster_launch_control.md; PDL=.../dependent_kernel_launch.md; X=github.com/NVIDIA/cutlass/blob/main/CHANGELOG.md (version tags); M=github.com/NVIDIA/cutlass/tree/main/examples/55_hopper_mixed_dtype_gemm; D=docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/overview.html; C=research.colfax-intl.com/epilogue_visitor_tree/ (third-party tutorial on EVT; NVIDIA's own EVT doc URL 404'd -> EVT node list is UNVERIFIED against primary source, headers `include/cutlass/epilogue/fusion/sm90_visitor_*.hpp` not read).
Status key: Yes / Partial / No. Grep control: `git grep` for `shared|barrier|warp|mma|cluster|TMA|stream.?K|grouped|sparse` over ir.rs finds only doc-comment mentions (ir.rs has zero structural hits); same grep finds `AxisRole`, `ReadIndex`, `Im2Col`, which exist, so the query works.

## (1) Capability table

| Category | Capability | IR | Evidence | Source | IR addition sketch |
|---|---|---|---|---|---|
| GEMM shape | Dense GEMM D=epi(AB), rank-2 | Yes | `Access::Contraction` ir:1675 | G | - |
| GEMM shape | Batched GEMM (L mode, strided) | Yes | `ContractionAxes::batched_matmul` (Batch role, rank-3) ir:~1900 | G (ProblemShape (M,N,K,L)) | - |
| GEMM shape | Multiple batch modes / hierarchical modes (CuTe nested shapes) | No | Contraction doc: "v1 rank-2 single-K"; one Batch axis only | G | multi-axis Batch roles; nested-mode shapes |
| GEMM shape | Multiple contracted K axes / tensor contraction (GETT) | No | doc: "one contracted axis group (K)"; roles vector allows only one K practically | X 3.1 (GETT example) | allow N free/K roles w/ mode lists (einsum roles) |
| GEMM shape | Operand layouts N/T for A,B,C,D (transposes) | Partial | doc: "transposes ... are the node's growth axes"; vocab layout.rs has layout enum (NT/TN/..) but Contraction doc is row-major only; `View::Permute` is per-operand but contraction reading through it unverified | F | confirm View::Permute on Contraction inputs, add stride/ld facts |
| GEMM shape | Interleaved layouts (NCxHWx, ColumnMajorInterleaved) | No | no layout kind in ir.rs; View has Identity/Permute/Broadcast/Reshape only | F | `View::Interleave{k}` |
| GEMM shape | Arbitrary strides / alignment contract (lda, 16B alignment, TMA 16B stride rules) | No | views carry no strides/alignment (extents "caller precondition") | G, M | alignment/stride facts on operand key |
| GEMM shape | Split-K (parallel reduction, workspace + reduce kernel) | Partial | only a `"splitk"` variant tag at backend.rs:274 (idiom:621); not an IR concept | E | `Access::Contraction.k_split{n, mode: serial/parallel/stream-K}` + workspace op (two-kernel reduce) |
| GEMM shape | Stream-K / data-parallel hybrid | No | grep stream.?K = 0 hits in ir.rs | X 3.2, 4.3 | schedule-level (Schedule::Contraction) variant + fixup/workspace semantic; numerics: reassociation of K |
| GEMM shape | Sliced-K (warps split CTA-K, reduce at end) | No | no warp-level concept | E | schedule hint; numerics = reassociable |
| GEMM shape | Grouped GEMM (heterogeneous M,N,K per group, ptr-array) | No | `ops.rs MoeKind` exists in vocab (op category) but IR has no ragged/grouped Access; grep grouped = 0 in ir.rs | X 3.4, 3.6, 3.8, 4.3 (MoE ragged contiguous), 4.8 (IndexPtrGrouped) | `Access::GroupedContraction{group_dims: Runtime table}`; problem-visitor |
| GEMM shape | Pointer-array batched GEMM (per-batch pointers) | No | BaseOffset::Runtime is per-operand single offset, not per-batch | X 3.4, 3.5.1 | `BaseOffset::PerBatch(ptr_array)` |
| GEMM shape | Gather/scatter GEMM (indexed A rows / D rows) | Partial | ReadIndex::Indexed / WriteIndex::ScatterIndexed exist (ir:2119, 2274) but Contraction compatibility with them not shown (v1 "mutually exclusive" with views; gate for contraction unverified) | X 3.3 (scatter-gather), 3.5 (gather/scatter conv) | allow Indexed read on FreeM/FreeN/K of Contraction operands |
| GEMM shape | GEMV / small-M specialisations (FP4 GEMV) | Partial | contraction rank-2 covers math; no GEMV Schedule ("decode/flat-GEMM long tail", plan.rs:67 comment) | X 4.2 | schedule only |
| GEMM shape | Rank-2k, SYRK/HERK/TRMM/SYMM (Blas3 structured: symmetric, triangular, Hermitian) | No | grep syrk/trmm/symmetric in ir.rs = 0; `vocab LinalgKind` is op list only | F (CUTLASS 2.x Blas3) UNVERIFIED in page | structured-operand (triangular/sym mask) read role; Coord-based predicate possible via Select+Coord (Coord Elementwise-only today) |
| GEMM shape | Mixed-input GEMM (fp16/bf16 x int8/int4/int2, fp8 x int4) with in-register upcast | No | Contraction "uniform dtype" (user-verified); `ElementKind` lists I4/U4 but a mixed key is impossible; sk4 packed weight+scale is a sibling, not general | M, X 3.3, 3.4, 4.2 | per-operand dtype in key; operand `Dequant{scale, zero_point, group_size}` read role |
| GEMM shape | Mixed-input scale modes: scale-only, scale+zero-point, group size = K or multiple of tile-K, fp16/bf16/fp32 scales | Partial | sk4 packed-weight+scale sibling exists (stated in task); group size / zero-point support unverified | M | generalise sk4 dequant role |
| GEMM shape | LUT-based int4->fp8 conversion with weight/scale re-encoding | No | no operand-prepack concept | M | prepacked-layout descriptor (`reorder_tensor` equivalent) |
| GEMM shape | Narrow-operand register-shuffle/reordering layout (`compute_memory_reordering_atom`) | No | no layout-transform-at-pack op | M | named packing layout on operand |
| GEMM shape | Block-scaled MX GEMM (mxf8, mxf6, mxf4, UE8M0 scale, vec size 32) | Partial | `ElementKind::F8E8M0` scale exists (element.rs:1154); MX element f6e2m3/f6e3m2/f4 deliberately NOT ElementKind rows (element.rs:1149 ruling); no scale-operand role on Contraction | B, X 3.8, 4.5 | scale operands SFA/SFB with vec size along K + scale layout; element dtypes E2M3/E3M2/E2M1 |
| GEMM shape | NVFP4 (nv_float4_t, UE4M3 scale, vec 16) | No | no UE4M3 unsigned scale dtype (E4M3FN is signed; UE4M3 equivalence UNVERIFIED); no FP4 element | B, X 4.8 (UE5M3 scale) | add f4 elem + ue4m3/ue5m3 scales |
| GEMM shape | Scale-factor layout (512B basic block 128x4, `Sm1xxBlockScaledConfig`) | No | no scale-layout concept | B | opaque swizzled-layout descriptor on scale operand |
| GEMM shape | Block-scale output generation (epilogue computes SFD, `LinCombBlockScaleFactor`) | No | epilogue is scalar-expr over Reduced(0), no cross-element amax/scale-out | B | epilogue block-reduce + second output (scale) w/ vec-size; dependent on Contraction epilogue growth |
| GEMM shape | Blockwise / groupwise-scaled FP8 GEMM (DeepSeek-style 1x128 / 128x128 scales applied in mainloop) | No | no per-K-block scale multiply inside K fold | X 3.7, 3.9 | K-fold with per-block scale (Reduced accumulates `scale_k*partial`) |
| GEMM shape | Structured sparse GEMM 2:4 (Ampere), SM90 sparse, SM100 `tcgen05.mma.sp` | No | grep sparse in ir.rs only "BlockSparseAttention" in vocab ops (different); no metadata operand | F, B, X 3.6, 3.9 | `Access::Contraction` operand role `Sparse24{metadata operand}`; compressor op |
| GEMM shape | Complex GEMM (cf32/cf64, 3xTF32, Gaussian 3m) | Partial | `Complex64/128` ElementKinds exist (element.rs:1194/1199) but plan gate: Add/Sub/Mul ok? Complex ops use function lowering (ArithOp); Contraction on complex unverified; Gaussian-3m/3xTF32 emulation not expressible | F | complex Contraction + `AccumSpec::EmulatedMultiPass` |
| GEMM shape | Emulated FP32 via bf16x3/bf16x9, 3xTF32 | No | AccumSpec only `WideFloat` (ir:1933) | X 3.8 (BF16x9), F | AccumSpec variants |
| GEMM shape | Binary (b1 xor.popc) GEMM | Partial | `ElementKind::B1` (element.rs:1188); xor-popc combine impossible: Contraction fixed sum of products, no combine op field (ir doc pins this) | F | `Contraction.combine: {MulAdd, XorPopc, AndPopc, MinPlus..}` |
| GEMM shape | Distributed GEMM (multi-GPU TP overlap) | No | single-device IR | X 3.7, 3.9 | out of scope for IR; note |
| Epilogue | alpha*AB + beta*C linear combination | Partial | epilogue is `ScalarExpr` over `Reduced(0)` only; Param exists (f32-only per ir table); no C source operand: ir:1675 "epilogue over Reduced(0) only; fused bias inputs are growth axes" | G | allow `Input(i)` (C, bias) in Contraction epilogue w/ MN-shaped read |
| Epilogue | Source C of different dtype/layout than D; ElementCompute != ElementAccumulator != ElementD | No | `out_dtype` limited to U8 / I64 (ir OpDef.out_dtype doc); acc dtype only WideFloat | G, F | per-operand out dtype and compute dtype in epilogue |
| Epilogue | EVT: AccFetch/SrcFetch | Partial | `Reduced(0)` = acc; no C fetch | C, X 3.2 | see above |
| Epilogue | EVT: Compute nodes (relu, gelu, silu, etc.) | Yes | ScalarExpr Unary/Binary/Select (ir:~140-420) cover activations incl. Gelu/Silu/Sigmoid/Tanh | C | - (need to also validate dReLU/dGELU: Select+Cmp expresses) |
| Epilogue | EVT: AuxLoad (extra MxN matrix, TMA-loaded) | No | no Input in contraction epilogue | C | as above |
| Epilogue | EVT: Row/Col/Scalar broadcast (bias, per-channel scale, per-token scale) | No | Broadcast View exists but epilogue can't read Inputs | C, X 3.3 | epilogue Input with `View::Broadcast` (bcast mask already exists) |
| Epilogue | EVT: AuxStore (store intermediate, e.g. pre-activation for backward) | No | multi-output exists only for Elementwise (`elementwise_multi`, ir OpDef.extra_out_bodies; "Access::Elementwise only in v1") | C, X 3.2 | multi-output Contraction |
| Epilogue | EVT: Row/Col/Scalar reduction (dbias, amax) fused in epilogue | No | no reduction after a Contraction; Access variants not composable | C, X 3.3 (dbias) | `Access::Contraction.epilogue_reduce: [{axis, op, atomic/workspace}]`; a two-level fold |
| Epilogue | EVT: TopK / softmax over N in epilogue | No | RowReduce is separate Access; no composition with Contraction | X 3.6 | Contraction -> RowReduce fusion node (attention-style) |
| Epilogue | EVT DAG (shared intermediates, topological visitor) | Partial | ScalarExpr is a tree but `ExprDag`/`DagNode` interns shared subexprs (ir:991, 1060) | C | - |
| Epilogue | Fused amax / output scaling for FP8 (D scale, aux scale, amax out) | No | no cross-tile reduction in epilogue | X 3.2 | epilogue reduction + atomic max output |
| Epilogue | Epilogue tile / subtile, epilogue schedule (TMA store, no-smem) | No | scheduling belongs in `Schedule` (plan.rs) but no variant carries tile/subtile | B | `Schedule::Contraction{...}` fields |
| Epilogue | Stream-K / split-K reduction epilogue (fixup) | No | see split-K | X 3.2 | |
| Epilogue | Epilogue with RNG/dropout | Partial | dropout noted in ir doc (Select keep_prob); RNG state not modeled | ir:2547 | - |
| Convolution | Implicit GEMM conv2d fprop | Partial | only `Im2Col` (explicit expand) + Contraction; "im2col->GEMM->reshape Conv2D FUSION ... deferred" (ir:1850); no implicit-GEMM node | F, X 3.5 | `Access::Conv{kind, ...}` or Contraction operand `Im2ColRead` view (implicit) |
| Convolution | dgrad / wgrad | No | "col2im/backward deferred" (ir:1850); vocab `ConvKind` lists them as op categories only | F, X 3.5, 3.8 | Conv kinds; col2im = `WriteIndex::ScatterIndexed{AtomicAdd}` already expressible for 1D? |
| Convolution | 1D/3D, grouped/depthwise, dilation, asymmetric padding, strided | Partial | Im2Col: rank-4 NCHW, groups=1, symmetric pad (u8 params) | X 3.5 | rank-N window spec, `pad_lo/pad_hi`, groups |
| Convolution | NHWC (channels-last) tensors, NCxHWx | No | Im2Col validated for NCHW contiguous only | F | layout-generic window access |
| Convolution | Transposed conv / output-padding | No | not in IR (vocab ConvKind only) | F | |
| Convolution | TMA im2col descriptors | No | no TMA | X 3.5 | |
| Convolution | Conv with stream-K / split-K (SM100) | No | | X 4.3 | |
| Convolution | Conv fused epilogue (bias+act, per-channel scale) | No | contraction epilogue limits | F | |
| Scheduling | Tile shape (CTA M,N,K), warp shape, instruction shape | No | `Schedule` has no tile fields for Contraction: `Contraction,` unit variant (plan.rs:69) | F, B | `Schedule::Contraction{tile_mnk, warp, instr, stages}` |
| Scheduling | Pipeline stages / StageCount auto / smem carveout | No | | G, B | `stages: u8` |
| Scheduling | Kernel schedule: CpAsync / Tma / Warp-specialized / Pingpong / Cooperative | No | | G, E, B | schedule enum `{Multistage, Tma, WarpSpecialized{Pingpong,Cooperative}}` |
| Scheduling | Persistent kernel with work-tile scheduler | No | grep persistent = 0 in ir/plan; "persistent" only in vocab KV-cache comments | G, E | `persistent: bool`, `TileScheduler` enum |
| Scheduling | CTA rasterization / swizzle (L2 locality) | No | | X 3.2 | `raster: {AlongM, AlongN, swizzle:n}` |
| Scheduling | Dynamic persistent scheduling via CLC (`clusterlaunchcontrol.try_cancel`) | No | | CL, X 3.8 | scheduler variant |
| Scheduling | Thread block clusters, cluster shape, preferred+fallback cluster, dynamic cluster | No | idiom:717 lists clusters as "not transferable"; no IR field | G, B, X 3.7, 3.8 | `cluster: [u8;3]`, fallback shape |
| Scheduling | 2-SM UMMA (cta_group::2), M-halving per SM | No | | B | `cta_group: 1|2` |
| Scheduling | Warp specialization roles (producer/MMA/epilogue/scheduler warps, register reallocation `setmaxnreg`) | No | idiom:720 "not transferable" | G, E, B | role assignment in schedule |
| Scheduling | Programmatic dependent launch (PDL / griddepcontrol) | No | | PDL, X 3.6 | launch attribute on op |
| Data movement | TMA load/store (tensor-map descriptors, box, swizzle) | No | idiom:716 lists as not transferable; IR has no async copy | G, E, X 3.0 | `AsyncCopy{tensor_map, box, swizzle}` op |
| Data movement | TMA multicast across cluster | No | | G | cluster-scoped copy |
| Data movement | TMA reduce-add store, L2 prefetch | No | | X 3.5 | `StoreReduce` |
| Data movement | cp.async (LDGSTS) multistage, sub-16B alignment via cp.async | No | idiom:715 | G, X 3.3 | `AsyncCopy{Ampere}` |
| Data movement | Shared-memory tiles with swizzled layouts (Crosswise/Congruous, CuTe Swizzle<B,M,S>) | No | idiom §1a: "IR never models cooperation"; grep shared in ir.rs = doc only | F, X 3.0 | `SharedBuf{shape, layout/swizzle, dtype}` |
| Data movement | Tensor memory (TMEM) alloc and tcgen05.ld/st/cp, S2T copies | No | | B, X 3.8, 4.5 | `TmemAlloc` + copy atoms |
| Data movement | Distributed shared memory (DSMEM) | No | idiom:717 | X 3.x | cluster-scope mem space |
| Data movement | Register-level layout transforms (ldmatrix, stmatrix, shuffle atoms) | No | | G | tile ops |
| Compute | Tensor-core MMA instructions (mma.sync 16x8x{8,16,32,64,128,256}, wgmma, tcgen05.mma, WMMA) | No | `AccumSpec::WideFloat` only variant (ir:1933); doc: "Tensor-core/TF32 policies join as variants" | F, B | `TileMma{shape, atom, a/b/acc dtype, kind}` tile-level op |
| Compute | TF32 compute with f32 storage | Partial | vocab element.rs:227/283 document "f32 GEMM routes through TF32", but `AccumSpec` cannot select TF32 vs strict | F | AccumSpec::Tf32 |
| Compute | FP8 fast-accumulate vs high-precision accumulation (every N K-steps promotion) | No | | X 3.2, 3.7 | `AccumSpec::PromoteEvery{n}` |
| Compute | Accumulator dtype choice (f16 acc, s32 acc, f64 DMMA) | No | WideFloat = f32 (f64 for f64/strict) only | F | AccumSpec{acc: ElementKind} |
| Compute | Saturating (satfinite) int output | No | no saturate op in Unary/Binary lists | F | `Unary::SatCast{dtype}` / `Convert` node |
| Compute | Dtypes: f16 bf16 f32 f64 | Yes | ElementKind F16,Bf16,F32,F64 (element.rs:1052-1071) | F | - |
| Compute | tf32 as storage/MMA operand | Partial | not in ElementKind (Tf32 appears at element.rs:1314 in another enum; which enum UNVERIFIED) | F | |
| Compute | fp8 e4m3 / e5m2 | Yes | Fp8E4M3FN, Fp8E5M2 | B | - |
| Compute | fp6 (e2m3, e3m2), fp4 (e2m1) | No | element.rs:1149 ruling: "block-scoped, not ElementKind rows" | B | add as sub-byte packed ElementKind with MX-only context |
| Compute | ue4m3, ue5m3 scale types | No | | B, X 4.8 | |
| Compute | int8 / uint8 / int4 / uint4 / b1 | Yes | I8,U8,I4,U4,B1 (element.rs:1076-1188) | F | (dtype exists; Contraction with s32 acc: acc dtype gap above) |
| Compute | int2 | No | | M | |
| Compute | complex f32/f64 | Partial | exist as ElementKind | F | see above |
| Compute | Half2/bfloat162/vectorized packed math (HFMA2, FP32x2 packed) | No | idiom/vectorized schedule only | X 4.1 | packed vec types |
| Compute | Fast-math/approx transcendentals (`__expf`) as explicit "Approx" | No | idiom §2: `__expf/__logf` currently refused | n/a | `Approx` qualifier (idiom §4.3 planned) |
| Compute | Warp/block collectives (shuffle, reduce) | Partial | exist only as emitter lowering of Reduction/RowReduce/Scan (idiom:56-58); not user-expressible | G | cooperation IR |
| CuTe/DSL | CuTe layouts algebra (composition, complement, logical divide/product, hierarchical shapes/strides) | No | View = {Identity,Permute,Broadcast,Reshape} | X 3.0, D | `Layout{shape: IntTuple, stride: IntTuple}` + algebra |
| CuTe/DSL | Tiled copy/MMA abstraction (TiledMma, TiledCopy, thread-value layouts) | No | | D | tile-level IR |
| CuTe/DSL | Pipelines (PipelineTmaAsync, mbarrier, producer/consumer) | No | | D | barrier/pipeline objects |
| CuTe/DSL | Control flow with dynamic loops/ifs inside kernel | No | IR is pure expression + Access; no loop/branch statements (Select only) | D | statement IR |
| CuTe/DSL | JIT/AOT, PyTorch/JAX integration | n/a | host-side concern | D, X 4.4 | |
| CuTe/DSL | Block API for TMA / S2T copies, SIMT-driven tensor-core primitives, task-scheduling framework, in-kernel tracing | No | | X 4.5, 4.7, 4.6 | |
| Other | FMHA/MLA/Flash attention (fprop, bwd), decode/GQA | No | NO `Access::Attention` (user-verified); only RowReduce softmax; vocab `AttentionKind` is op-name only | X 3.8, 3.9, 4.0, 4.4 | Contraction-RowReduce-Contraction fusion w/ online softmax state (m,l,o) |
| Other | Mamba2 SSD / linear-attention scans | Partial | Scan exists (monoid only; no matrix-state scan); docs/qwen3_5-linear-attention-assessment.md | X 4.1, 4.4 | generic associative scan with struct state |
| Other | GEMM + reduction fusion kernels (GEMM+softmax, GEMM+layernorm) | No | see Epilogue | X 3.6 | |
| Other | Autotuning heuristics (nvMatmulHeuristics) | n/a | tooling | X 4.2 | |
| Other | Rubin (SM107), SM103/SM120/SM121 target specifics, arch-family targets (100f/120f) | No | `target.rs` exists in vocab (not read) | X 4.x | target caps |

## (2) Ten highest-impact gaps (ranked)

1. Tensor-core MMA as a first-class tile op: `AccumSpec` has one variant (`WideFloat`, ir:1933). No MMA atom (shape, a/b/acc dtype, kind), no accumulator dtype choice (s32, f16, f64 DMMA), no TF32/3xTF32/bf16x9 emulation, no FP8 fast-accumulate/promotion. Every other CUTLASS GEMM capability sits on this.
2. Contraction epilogue cannot read any operand: only `Reduced(0)`, `Const`, `Param`. So no `beta*C`, bias/row/col broadcast, AuxLoad, per-channel scale. Plus no AuxStore/multi-output and no fused reductions (dbias, amax) = all of EVT.
3. Mixed-input and block-scaled GEMM: key is uniform dtype; no per-operand dtype, no dequant role (scale/zero-point/group size), no scale-operand with vec size and swizzled SF layout, no FP4/FP6 elements, no UE4M3/UE5M3 scales, no block-scale output (SFD).
4. Cooperation substrate (owner-ruled in scope): no shared-memory buffer type with swizzle, no barrier/mbarrier/pipeline object, no warp/block collective node, no async copy (cp.async/TMA, multicast, TMA-reduce), no TMEM, no DSMEM, no cluster scope, no warp-role specialization. idiom §1a still says "IR never models cooperation" (see section 3).
5. Schedule/mapping vocabulary: `Schedule::Contraction` is a unit variant (plan.rs:69): no CTA/warp/instr tile, stages, kernel schedule (cpasync/TMA/pingpong/cooperative), persistent + tile scheduler (static, stream-K, CLC), rasterization swizzle, cluster shape/fallback, 2-SM cta_group.
6. Grouped / ptr-array GEMM (MoE, ragged): no per-group problem table, no per-batch pointer array; `BaseOffset::Runtime` is a single scalar per operand.
7. Split-K / stream-K / sliced-K with workspace and fixup (reassociation numerics): only a backend variant tag, no IR concept or numerics axis.
8. Implicit-GEMM convolution (fprop/dgrad/wgrad, NHWC, 1D-3D, grouped, TMA-im2col): only explicit NCHW `Im2Col`, groups=1, symmetric pad; fusion and backward deferred (ir:1850).
9. Structured sparse GEMM (2:4, metadata operand, compressor, `mma.sp`/`tcgen05.mma.sp`): absent.
10. CuTe layout algebra and general strides/alignment: `View` has four variants; no hierarchical (shape,stride) layouts, no interleaved/swizzled layouts, no alignment/stride contract (TMA's 16B rule), no structured Blas3 (triangular/symmetric) operands. Near-tie: attention (FMHA/MLA) fusion with no `Access::Attention`.

## (3) Things in the IR that look WRONG or MISLEADING relative to CUTLASS

a. `docs/idiom-lifting-design.md` §1a (idiom:71-80): "the IR never models cooperation". Contradicts the owner ruling reported in the task; the table at idiom:716-720 also files cp.async/TMA/wgmma/clusters/warp-specialization as "not transferable" (a registry fact about portability, not an IR-scope statement; keep that distinction when rewriting).
b. `AccumSpec` doc (ir:1920s) says "Accumulate in `float` (`double` for f64/f32-strict) ... Tensor-core/TF32 policies join as variants", while `element.rs:227/283` documents that `f32` GEMM "routes through TF32 tensor cores". An f32 Contraction is therefore under-specified: the vocab claims one numerics (TF32, 10-bit mantissa), the IR's only accumulator says full-float SIMT. CUTLASS makes this an explicit choice (tf32 vs 3xTF32 vs f32 SIMT, F). The two must be reconciled by an explicit AccumSpec axis; today a key can silently mean either.
c. Contraction doc line "a contraction cannot express a non-sum fold" is accurate for CUTLASS dense GEMM, but CUTLASS also exposes `b1 xor.popc` GEMM (F: b1 TensorOp) where the combine is popcount-xor: `ElementKind::B1` exists in the vocab (element.rs:1188) while the IR cannot express its fold. Either B1 is dead in IR or a combine field is needed; the pinned test `a_contraction_cannot_express_a_non_sum_fold` will have to be revisited, and the `is_bit_move` ruling (always computed) is still right for it.
d. `OpDef.out_dtype` legal set is only `U8` and `I64` (ir OpDef doc): CUTLASS GEMM routinely has ElementD != ElementAccumulator != ElementC != ElementCompute (e.g. s8xs8->s32 acc, f32 alpha/beta, s8 or fp8 D with satfinite, F). The IR's "uniform dtype v1" is stated, but the vocab docs for int8/int4 GEMM (element.rs:470-520: "int8 inputs, int32 accumulator, float alpha/beta") read as if they were IR-expressible. They are vocab-level (Element trait) claims, not IR ones.
e. `element.rs:1149` ruling that MX element formats (f6e2m3, f6e3m2, f4) are "not a thing" outside block context: in CUTLASS they are plain first-class C++ types (`float_e2m3_t`, `float_e3m2_t`, `float_e2m1_t`) usable in `cute::Tensor`s and with non-block-scaled `f8f6f4` tcgen05 kinds (B: "kinds: f8f6f4, mxf8f6f4.block_scale"); `tcgen05.mma kind::f8f6f4` takes FP4/FP6 WITHOUT scales. The ruling is therefore wrong for the CUTLASS union. UNVERIFIED that the PTX `f8f6f4` kind is scale-free beyond the B page's kind list.
f. `Const`/`Param`: "Param f32-only", Const spelled as f64 literal: a CUTLASS alpha/beta at f16/bf16/int accumulators or f64 GEMM cannot be a runtime `Param` outside f32 (table in BinaryOp doc). Also Coord is Elementwise-only (ir Coord doc), so triangular/causal masks in epilogues (a common CUTLASS FMHA/Blas3 construct) cannot be expressed in contraction epilogues.
g. `Access::Window`/`Im2Col` docs call Im2Col "the conv-lowering workhorse ... Conv2d = im2col then GEMM then reshape": this is the cuDNN-style explicit lowering CUTLASS deliberately avoids (implicit GEMM, F/X 3.5). Not wrong, but an Unpopped lifter given CUTLASS conv will see no matching node.
h. `View::Reshape` is "for recognition + keying only; genuine rank-change emit deferred" (ir:~1990): CUTLASS CuTe reshapes/coalesces are everywhere; a CuTe-derived kernel can't round-trip.

## Unverified / gaps in this survey
- Fetch failure: `git fetch` errored (ref lock); IR read at local origin/main a575e66.
- NVIDIA's own EVT doc URL (media/docs/cpp/epilogue_visitor_tree.md) 404'd; EVT node list (AuxStore, TopKSoftmax, RowReduction, ColReduction) rests on the Colfax tutorial (C) and the 3.x changelog (X), not NVIDIA's headers.
- Multicast, DSMEM and Ampere-TMA were not covered in the pages fetched (X returned no hits); their entries rest on G (cluster shape "over which TMA multicast operates") and idiom:716-717.
- Not read: `target.rs`, `capability.rs`, `backend.rs`, `lift.rs`; Contraction-with-View/Indexed and Complex-contraction status marked Partial from docs, not code.
- sk4 packed-weight sibling: I found no `packed|scale` structural hit in ir.rs (grep returned only comments); I relied on the task statement that it exists, perhaps outside ir.rs or on a newer commit.
- CuTe DSL cluster support is not covered by page D.
