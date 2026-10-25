> **Raw survey output, 2026-10-24/25. Read `docs/ir-union-design.md` first; where it and this file disagree, the design doc wins.** Written by a read-only Sonnet agent from the official documentation (every row carries its URL; rows it could not confirm from a primary page are marked UNVERIFIED) and from `ir.rs` at `origin/main` `a575e66`. The design doc re-checked the load-bearing IR claims against the source; this file as a whole has not been reviewed line by line.

# Triton capability survey vs Unpopped IR

Triton version read: `triton.__version__ = '3.9.0'` (main, python/triton/__init__.py, fetched 2026-10-24); docs are triton-lang.org/main (no version shown on pages). Unpopped read at fetched origin/main a575e66 (`crates/unpopped/src/ir.rs` 4300 lines, `plan.rs`, `unpopped-vocab/src/{ops,plan,element}.rs`, `docs/idiom-lifting-design.md`). Local fetch of `origin/main` ref failed (ref lock) so I read the FETCH_HEAD SHA.

Abbrev: DOCS=https://triton-lang.org/main/python-api/ ; GEN=DOCS+generated/ ; DIAL=https://triton-lang.org/main/dialects/ ; CORE=https://github.com/triton-lang/triton/blob/main/python/triton/language/core.py ; ir=ir.rs.

## How Triton differs structurally (block-programming, not thread-level)

- A Triton program instance = one CTA (or cluster with num_ctas) operating on whole BLOCKS (tensors with constexpr shapes). The user never names threads, shared memory, or barriers: the compiler (TritonGPU layouts, pipeliner, warp-spec pass) chooses them. Unpopped's IR is a per-OUTPUT-COORDINATE scalar DAG plus Access "loop-nest shape" nodes; it has no notion of a tile as a first-class value, no program grid, no program_id.
- Consequence: Triton kernels are mostly arbitrary tile-level dataflow graphs (load tile -> math -> dot -> reduce -> store tile, with loops carrying tiles). Unpopped can only express one fixed Access shape per op (single reduce, single contraction, ...); a Triton kernel with two dots + softmax (flash-attn), a loop with data-dependent bounds, or a persistent tile loop is not one OpDef. This is the biggest structural gap, not a list of missing ops.
- Triton has TWO layers beyond the DSL that also fall under "cooperation": Gluon (explicit layouts, shared-mem descriptors, mbarriers, tcgen05, warp_specialize) and the ttg/ttng/amdg dialects. Both enumerated below.

## (1) Capability table

Status key: Yes / Partial / No. Evidence cites ir.rs structure (names, not line numbers, because line numbers shift; grep counts are on a575e66).

### Program model

| Category | Capability | IR | Evidence | Source | IR addition |
|---|---|---|---|---|---|
| Program model | `tl.program_id(axis)`, `tl.num_programs(axis)` (grid of SPMD block programs) | No | absent: grep program_id/ProgramId/num_programs = 0 hits in ir.rs, element.rs, idiom doc. Only `ScalarExpr::Coord(axis)` = output element coordinate (Elementwise-only, f32/f64 only) | GEN triton.language.program_id.html (listed in DOCS triton.language.html "Programming Model") | Add a launch-grid model: `Grid{dims}` on OpDef and `ScalarExpr::ProgramId(axis)/NumPrograms(axis)` leaves; tile-level coordinates `Coord` should be integer-typed |
| Program model | Block-level tensors with constexpr shapes, `tl.arange`, `tl.full/zeros`, `expand_dims/broadcast/reshape/permute/trans/join/split/interleave/cat/ravel/flip` | Partial | View covers Identity/Permute/Broadcast/Reshape(recognition only, "carried for recognition + keying only"); no tile value type, no join/split/interleave/cat/flip, no arange/full as ops (Const only) | DOCS triton.language.html (Shape Manipulation, Creation Ops) | A first-class `Tile{shape,dtype}` SSA value with shape ops (reshape real emit, join/split/interleave/cat/flip/arange) |
| Program model | `constexpr` / `@triton.jit` specialization (`do_not_specialize`, divisibility-by-16 / alignment auto-specialization, `noinline`, `debug`) | No | Param(i) is runtime scalar, Const folded; no specialization-on-value-alignment, no constexpr meta-params, no call/noinline function | GEN triton.jit.html | `Constexpr` meta-params on OpDef + `Specialize{divisible_by,..}` facts on operands |
| Program model | Device function calls (`tt.call`, `@triton.jit` helper fns, `combine_fn` lambdas) | No | no function/call node; ScalarExpr is a closed tree/DAG with no call | DIAL TritonOps.html (tt.call, tt.func) | Subroutine/closure node for user combine_fns |
| Program model | Compiler hints `tl.assume`, `max_contiguous`, `max_constancy`, `multiple_of`, `tt.annotate` | No | absent; Contiguity exists in StructureKey (per operand) but no assume/range facts | DOCS triton.language.html (Compiler Hint Ops); DIAL TritonOps (tt.annotate) | `Hint`/`Assume(cond)` facts attached to values |

### Loads / stores

| Category | Capability | IR | Evidence | Source | IR addition |
|---|---|---|---|---|---|
| Memory | Masked load `tl.load(ptr, mask, other)` (arbitrary boolean tile mask + fill value) | Partial | Only structural OOB: `OobPolicy::{Skip,Clamp,ZeroFill}` on indexed gather, `Window` pad, `Select` in bodies. No arbitrary per-element mask operand with `other` fill on Direct reads | GEN triton.language.load.html | `ReadMask{operand, other}` per input operand |
| Memory | Masked store `tl.store(ptr, val, mask)` | Partial | `OobPolicy::Skip` store predicate on ScatterIndexed only; Direct writes unconditional on output domain | GEN triton.language.store.html | `WriteMask{operand}` on WriteIndex::Direct |
| Memory | Pointer tensors: arbitrary per-element pointer arithmetic (`ptr + offs_m[:,None]*stride + offs_n[None,:]`, `tl.addptr`, ptr<->int) | Partial | Only affine strides (View/stride) + one-axis `ReadIndex::Indexed` (index tensor replaces ONE iteration axis, `index_operand`/`axis`/`index_dtype` I32/I64). No general multi-axis computed address, no pointer dtype | DIAL TritonOps (tt.addptr, tt.int_to_ptr, tt.ptr_to_int) | `ReadIndex::Affine/Computed(ScalarExpr)` address expression over Coord/ProgramId |
| Memory | `cache_modifier` (".ca/.cg/.cv"), `eviction_policy`, `volatile` | No | absent: grep cache_modifier/eviction/volatile = 0 | GEN triton.language.load.html (signature) | Per-access `CacheHint{modifier,evict,volatile}` hints (hint, not intent) |
| Memory | Block pointers `tl.make_block_ptr(base,shape,strides,offsets,block_shape,order)` + `tl.advance` (boundary_check/padding_option is not in the 3.9 tl.load signature, only in make_tensor_descriptor `padding_option`) | No | absent: BaseOffset exists (Zero/Runtime) but no block-shape / order / advance-in-loop | CORE (make_block_ptr, advance at ~L2524-2536) | Tile-view operand `BlockPtr{shape,strides,order,offsets}` + loop-carried advance |
| Memory | Tensor descriptors `tl.make_tensor_descriptor(base,shape,strides,block_shape,padding_option)`, `desc.load/store([offs])`, 2-5D, 16B alignment, last dim contiguous | No | absent: grep descriptor/TMA/tensormap = 0 in ir.rs (design doc mentions no TMA either, only hardware table) | GEN triton.language.make_tensor_descriptor.html | Descriptor operand kind with tile shape/padding/alignment; reads and writes by tile offsets |
| Memory | Descriptor extras: `desc.gather(x_offsets,y)`, `desc.scatter`, `desc.atomic_add/min/max/and/or/xor`, `tt.descriptor_reduce` (8 reduce kinds), `round_f32_to_tf32` descriptor flag | No | absent | CORE tensor_descriptor_base methods (~L1506-1540); DIAL TritonOps (tt.descriptor_gather/scatter/reduce) | Descriptor ops with row-gather/scatter and reducing store; `RoundF32ToTf32` flag |
| Memory | `tl.gather(src,index,axis)` (in-register tile gather along axis) | Partial | `ReadIndex::Indexed` gathers from a MEMORY operand along one axis (same semantics shape as torch.gather); no gather over a register tile produced inside the same kernel (needs tile value type) | GEN triton.language.gather.html | Becomes expressible once an intermediate Tile value exists: `Gather{src_tile,index_tile,axis}` |
| Memory | Scratch: `ttg.global_scratch_alloc` (program-private global buffer), `triton.set_allocator` | No | absent | DIAL TritonGPUOps (ttg.global_scratch_alloc) | `Scratch{bytes,scope}` buffer resource |

### Dot / scaled dot

| Category | Capability | IR | Evidence | Source | IR addition |
|---|---|---|---|---|---|
| Matmul | `tl.dot(a,b,acc)` 2D/3D (3D = batched), int8/fp8e5/fp16/bf16/fp32 operands, acc fp16/fp32/int32, `out_dtype` | Partial | `Access::Contraction{axes: ContractionAxes, accum: AccumSpec, epilogue}`; v1 rank-2 single-K uniform dtype (verified: AxisRole::Batch exists "v1: unused (rank-2)"); `AccumSpec` has ONE variant `WideFloat`; no acc operand input (tl.dot(a,b,acc) chains) | GEN triton.language.dot.html | Mixed dtypes (a,b,acc,out), real batch, accumulate-into-acc operand, multi-K |
| Matmul | `input_precision` = tf32 / tf32x3 / ieee / bf16x3 / bf16x6 (AMD: ieee,bf16x3,bf16x6, tf32 on CDNA3) | No | `AccumSpec` is `#[non_exhaustive]` with only `WideFloat`; doc says "tensor-core/TF32 policies join as variants". `MathPrecision` exists in vocab element.rs (F16,Bf16,Tf32,F32,F64,Int8,Fp8E4M3FN,Fp8E5M2,Int4) but has no x3/x6 split-precision modes and is not on the IR | GEN triton.language.dot.html; CORE dot() L2253-2290 | Add `AccumSpec::{Tf32,Tf32x3,Ieee,Bf16x3,Bf16x6,...}` with honest contract flips |
| Matmul | `max_num_imprecise_acc` (Hopper fp8 accumulator promotion interval) | No | absent | GEN triton.language.dot.html (signature; undocumented semantics) | `AccumSpec::PromoteEvery(n)` |
| Matmul | `tl.dot_scaled(lhs,lhs_scale,lhs_fmt,rhs,rhs_scale,rhs_fmt,acc,fast_math,lhs_k_pack,rhs_k_pack,out_dtype)`; formats e2m1,e4m3,e5m2,bf16,fp16; e8m0 scales group 32 along K | No | ElementKind has F8E8M0 and Fp8 variants but NO fp4 e2m1 and no packed-fp4 storage; no scale operand concept on Contraction (the sk4 seam in `ReductionAccum` is a placeholder "reserves the MX element+scale identity coordinate") | GEN triton.language.dot_scaled.html | Block-scaled operand kind `Scaled{elem, scale: F8E8M0, group:32, k_pack}`; `Fp4E2M1` (packed x2) element |
| Matmul | `warp_group_dot` / `tc_gen5_mma(_scaled)` / tmem (Hopper wgmma, Blackwell tcgen05, tensor memory) | No | absent: the IR has no tensor-core tile or accumulator-location concept (Schedule::Contraction = "skinny SIMT kernel", MMA "bench-gated variants" future) | DIAL TritonNvidiaGPUOps (ttng.warp_group_dot, ttng.tc_gen5_mma, ttng.tmem_*) | MMA tile shape/accumulator space (register/tmem) in the contraction Schedule/plan |

### Reductions / scans / sort / topk / histogram

| Category | Capability | IR | Evidence | Source | IR addition |
|---|---|---|---|---|---|
| Reduce | `tl.sum/max/min/prod? (no prod; reduce combine_fn)/argmax/argmin/any/all/reduce_or/xor_sum` over any axis, `keep_dims`, `axis=None` (all) | Partial | `Access::Reduction{op: ReduceOp::{Sum,Mean,Max,Min,Prod}, axes: AxisMask, keepdim, post, accum}`; no Any/All/Or/Xor/BitAnd ops; `ReduceOp` is closed; argmax/argmin: no (index-carrying reduce absent except RowSort) | GEN triton.language.reduce.html; DOCS triton.language.html (Reduction Ops) | Add `ReduceOp::{And,Or,Xor,ArgMax,ArgMin}`; Reduction needs a multi-output / index-carrying variant |
| Reduce | `tl.reduce(input, axis, combine_fn)` with tuple inputs (multi-operand welford/argmax) and user `@triton.jit` combine_fn | No | `ReduceOp` is a closed enum of 5; `RowReduce` has multi-stage but each stage is a single-tensor associative op | GEN triton.language.reduce.html | `ReduceOp::Custom{combine: ExprDag over (acc tuple, elem tuple), identity}` |
| Reduce | Softmax / online softmax / row stats fused | Yes (3-pass) | `Access::RowReduce{stages,epilogue}` with `Reduced(j)` bridging; design doc says online/flash needs new `Access::Attention` (not present - verified) | DOCS (tl.softmax) | n/a (see attention below) |
| Scan | `tl.associative_scan(input, axis, combine_fn, reverse)` tuple inputs, `cumsum`, `cumprod` | Partial | `Access::Scan{op: ReduceOp, axis, reverse, exclusive, pre, post}` covers cumsum/cumprod/cummax; v1 axis = rank-1 only; no custom combine_fn or tuple-carry (e.g. segmented scans, linear recurrences a*x+b) | GEN triton.language.associative_scan.html | `Scan` over `Custom` monoid with tuple state |
| Sort | `tl.sort(x, dim, descending)` | Partial | `Access::RowSort{order, stable, out: SortOut::{Values,Indices,Both}, limit}`; row-wise last axis only. Triton sort is on a block, any dim? (docs page for sort has no content in fetch: UNVERIFIED semantics) | DOCS triton.language.html (Scan/Sort Ops) | Sort axis other than last (Triton sort `dim`, UNVERIFIED) |
| Topk | `tl.topk(x, k: constexpr power-of-2, dim=None=last, descending=True)` | Yes | `SortLimit::TopK` + `OpDef::row_topk/row_bottomk`; k runtime in IR (broader than Triton's pow2 constexpr) | GEN triton.language.topk.html | n/a |
| Histogram | `tl.histogram(input, num_bins, mask)` unit bins from 0, integer input | Partial | `OpDef::bincount` = ScatterIndexed{AtomicAdd} with Input(0) as its own index; no per-element mask on the count (the mask would need a Select to 0, which is expressible), output is global buffer not a tile (tile result needed for a per-block histogram) | GEN triton.language.histogram.html | Tile-result histogram once Tile values exist; mask operand |
| Window | (no Triton primitive; Triton does pooling via loads) | Yes (extra) | `Access::Window`, `Im2Col` have no Triton analogue | - | n/a |
| Scan/select | `tl.where`, `tl.maximum/minimum` with `propagate_nan` (PropagateNan), `tl.clamp`, `tl.swizzle2d` | Partial | `ScalarExpr::Select` (float dtypes only), `BinaryOp::{Max,Min,FmaxIeee,FminIeee}` cover NaN modes; `Select` rejects INT dtypes in v1 (Triton where is any dtype); no swizzle2d (program-id remap for L2 locality) | CORE minimum L2923; GEN tl.where | Allow integer Select; `Swizzle2d` over ProgramId |

### Atomics

| Category | Capability | IR | Evidence | Source | IR addition |
|---|---|---|---|---|---|
| Atomics | `atomic_add/max/min` (returns OLD value) with `sem` (acquire/release/acq_rel/relaxed) and `scope` (gpu/cta/sys) and mask | Partial | `WriteCombine::{Assign,AtomicAdd,AtomicMax,AtomicMin}` only on `WriteIndex::ScatterIndexed`; FP AtomicMax/Min unsupported "integer cells in v1"; no returned old value; no sem/scope | GEN triton.language.atomic_add.html | `Atomic{op,sem,scope}` write combine + optional old-value output; float max/min |
| Atomics | `atomic_cas`, `atomic_xchg`, `atomic_and`, `atomic_or`, `atomic_xor` | No | absent: grep Cas/Exchange/AtomicXchg = 0 hits ir.rs | CORE tensor.atomic_* L1199-1229; DIAL TritonOps (tt.atomic_cas, tt.atomic_rmw: 10 rmw ops) | Extend WriteCombine; CAS needs compare operand and old-value output |
| Atomics | `atomic_load`, `atomic_store`, `atomic_poll(expected, sem, scope, timeout_ns)` (inter-program spin sync, persistent/stream-K) | No | absent | CORE L1199-1226; DIAL TritonOps (tt.atomic_poll) | `AtomicLoad/Store/Poll` + loop/wait control (needs control flow) |
| Atomics | Descriptor atomics / `tt.descriptor_reduce` / `ttng.async_tma_reduce` (reducing store) | No | absent | CORE L1506-1526; DIAL TritonNvidiaGPUOps | covered under descriptor row |
| Grid sync | `tt.grid_dependency_wait/launch_dependents` (Programmatic Dependent Launch) | No | absent | DIAL TritonOps | Launch-graph edges/PDL flag |

### Dtypes / casts / numerics

| Category | Capability | IR | Evidence | Source | IR addition |
|---|---|---|---|---|---|
| Dtypes | int1/int8/16/32/64, uint8/16/32/64 | Partial | ElementKind lacks U32 and U64 (has I8,I16,U8,U16,I32,I64,Bool,I4,U4,B1). `IndexOutputKind::U32` exists in vocab but is an index-ABI marker, not an element dtype | CORE L833-841 | Add `U32`,`U64` |
| Dtypes | fp16, bf16, fp32, fp64 | Yes | F16,Bf16,F32,F64 (+F32Strict) | CORE L849-852 | n/a |
| Dtypes | tf32 as a storage dtype | n/a | Triton has no tf32 dtype; tf32 is `input_precision` only | GEN dot.html | see dot |
| Dtypes | fp8 family: float8e4nv (E4M3FN), float8e5 (E5M2), float8e4b15, float8e4b8 (E4M3FNUZ), float8e5b16 (E5M2FNUZ), float8e8m0fnu | Partial | Present: Fp8E4M3FN, Fp8E5M2, Fp8E4M3FNUZ, Fp8E5M2FNUZ, F8E8M0. Missing: **float8e4b15** (E4M3 bias 15) | CORE `triton/language/__init__.py` exports float8e4b15 | Add `Fp8E4M3B15` |
| Dtypes | fp4: `float4e2m1x2` (two e2m1 packed in a byte) | No | absent: grep E2M1/Fp4/F4 = 0 in ir.rs and element.rs | CORE (language/__init__.py exports float4e2m1x2); GEN dot_scaled.html | Add `Fp4E2M1` (packed) + `ttg.fp4_to_fp` decode |
| Casts | `tl.cast(x, dtype, fp_downcast_rounding={"rtne","rtz"}, bitcast=False)` | No | absent: grep Cast = 0; conversions are implicit in dtype of the op (`OpDef.dtypes`, `out_dtype`); no explicit Cast node, no rounding-mode, no bitcast (the IR has `is_bit_move` predicates over existing ops, not a Bitcast node) | GEN triton.language.cast.html | `ScalarExpr::Cast{to, round: Rtne/Rtz/Rtp/Rtn/Stochastic?, saturate?}` + `Bitcast{to}` |
| Casts | `tt.fp_to_fp` rounding attr, `tl.convert saturation / overflow_mode` (UNVERIFIED: overflow_mode not in 3.9 cast docs) | No | absent | DIAL TritonOps (tt.fp_to_fp rounding, 2 cases) | same Cast node |
| Numerics | Math: `exp/exp2/log/log2/sqrt/rsqrt/sin/cos/erf/sigmoid/floor/ceil/abs/fma/clamp`, `div_rn`, `sqrt_rn`, `fdiv` (fast), `umulhi`, `cdiv` | Mostly Yes | UnaryOp has Exp,Exp2,Log,Log2,Sqrt,Rsqrt,Sin,Cos,Erf,Sigmoid,Floor,Ceil,Abs...; BinaryOp has Max,Min,Pow,Rem,Atan2... Missing: **fma** (3-ary fused), **div_rn / sqrt_rn / fdiv fast vs precise variants** (only ArithOp::Div), **umulhi** (mulhi), **clamp** (as one op; can be Min(Max)), **cdiv** (integer ceil-div) | DOCS (Math Ops list); DIAL TritonOps (tt.mulhiui, tt.precise_divf, tt.precise_sqrt, tt.approx_divf) | `Fma` ternary node, `Umulhi`, precise/approx Div/Sqrt mode flags |
| RNG | `tl.rand/randn/randint/rand4x/randint4x/randn4x`, `uint_to_uniform_float`, Philox counter-based RNG (seed, offset, n_rounds=10) | No | absent: grep Philox = 0; all "rand" hits in ir.rs are substrings (operand, brand etc.) | GEN triton.language.rand.html; DOCS (Random Number Generation) | `Rng{kind: Philox, rounds}` ScalarExpr leaf with seed Param and offset expr |
| Packed/vector | `tl.map_elementwise(scalar_fn, *args, pack)`, `ttng.packed_arith` | No | absent | CORE L3195; DIAL TritonNvidiaGPUOps | Higher-order elementwise `Map{fn, pack}` |

### Control flow

| Category | Capability | IR | Evidence | Source | IR addition |
|---|---|---|---|---|---|
| Control | `for i in tl.range(...)` / `tl.static_range` loops with loop-carried tiles (the heart of every GEMM/attention kernel), runtime bounds, negative constexpr step | No | The only loops are implicit inside Access nodes (reduce/scan/window/contraction K loop). No `Loop{carried}` node | GEN triton.language.range.html | `Loop{lo,hi,step,carried[],body}` region |
| Control | `if`/`while` on scalar (program-uniform) predicates, early return, data-dependent loop bounds (e.g. varlen via `seq_lens[pid]`, CSR rows) | No | `Select` is data movement only, no branch; no While | (core language semantics) CORE; DIAL TritonOps (scf is used via MLIR) | `If`, `While`, `Return` regions |
| Control | Loop attributes: `num_stages` (loop-level pipelining), `loop_unroll_factor`, `disallow_acc_multi_buffer`, `flatten`, `warp_specialize`, `disable_licm` | No | absent; AccumSpec/Schedule have no per-loop attribute channel | GEN triton.language.range.html | Per-loop `LoopHints{...}` (hints, per the idiom doc governing test) |

### Scheduling knobs / launch

| Category | Capability | IR | Evidence | Source | IR addition |
|---|---|---|---|---|---|
| Schedule | `num_warps` (default 4) | No | `Schedule` = Vectorized/Scalar/Strided/Reduction/RowReduce/Contraction/Scan/Window/... none carries warps; design doc §9 says lifted hints are "starting points" for autotune | GEN triton.Config.html | `ScheduleHints{num_warps,..}` |
| Schedule | `num_stages` (default 3, software-pipelining depth) | No | absent | GEN triton.Config.html | `num_stages` hint |
| Schedule | `num_ctas` (cluster size, SM90+) | No | absent: grep cluster = 0 in ir.rs | GEN triton.Config.html | `ClusterShape` on launch |
| Schedule | `maxnreg` (PTX .maxnreg), `ir_override` (.ttgir/.llir/.ptx/.amdgcn), `pre_hook` | No | absent | GEN triton.Config.html | `RegBudget` hint; ir_override is a non-goal (opaque escape) |
| Schedule | `@triton.autotune(configs, key, prune_configs_by{perf_model,top_k,early_config_prune}, reset_to_zero, restore_value, cache_results, do_bench)`, `@triton.heuristics` | No | design doc §9 Autotuning exists as prose; no IR representation of a config space or key | GEN triton.autotune.html | Out of IR proper; `TuneSpace{params, key}` metadata |
| Schedule | Meta-params as constexpr tile sizes (BLOCK_M/N/K, GROUP_M) swept by autotune | No | tile sizes are not an IR concept (contraction is skinny SIMT) | GEN triton.Config.html (kwargs) | Tile shape on Contraction schedule |
| Schedule | `tl.swizzle2d` / grouped program-id ordering for L2 reuse | No | absent | DOCS (Indexing Ops) | requires ProgramId |
| Schedule | Persistent kernels (grid = #SMs, loop over tiles via `tl.num_programs`), stream-K via atomics, CLC dynamic tile scheduler (`ttng.clc_*`) | No | absent | DIAL TritonNvidiaGPUOps (clc_try_cancel, clc_get_program_id); GEN range.html | Persistent tile loop + program-id/num-programs + grid sync |

### Cooperation (owner ruling: in scope)

| Category | Capability | IR | Evidence | Source | IR addition |
|---|---|---|---|---|---|
| Coop | Shared-memory buffers (`ttg.local_alloc/load/store/dealloc`, memdesc_index/subslice/trans/reshape/reinterpret views, Gluon `allocate_shared_memory`) | No | design doc §1a: "groupshared scope call"; ir.rs has no shared-memory resource (collectives are Schedule-internal: `Schedule::RowReduce` "warp-shuffle + shared-memory tree") | DIAL TritonGPUOps.html | `Shared{shape,dtype,layout}` resource + indexed views |
| Coop | Shared-mem gather/scatter/atomic scatter (`ttg.local_gather/local_scatter/local_atomic_scatter_rmw`) | No | absent | DIAL TritonGPUOps | indexed access to Shared |
| Coop | CTA barriers `tl.debug_barrier`, `ttg.barrier` (CTA or warp scope), `__syncthreads` | No | grep Barrier in ir.rs = 0 (design doc table lists it as a hardware feature only) | DOCS (Compiler Hint Ops: debug_barrier); DIAL TritonGPUOps (ttg.barrier) | `Barrier{scope: Warp/Cta/Cluster}` op + phase ordering |
| Coop | Async global->shared copy (`ttg.async_copy_global_to_local`, `async_commit_group`, `async_wait N`) = cp.async | No | absent | DIAL TritonGPUOps | `AsyncCopy{src,dst,group}` + commit/wait |
| Coop | TMA: `async_tma_copy_global_to_local`/`local_to_global`, `async_tma_gather/scatter/reduce`, `async_tma_store_wait`, `tensormap_create`, `tensormap_fenceproxy_acquire`, `fence_async_shared` | No | absent | DIAL TritonNvidiaGPUOps | TMA descriptor type + async tile copy + proxy fence |
| Coop | mbarrier: `init_barrier/inval_barrier/arrive_barrier/wait_barrier/barrier_expect/async_copy_mbarrier_arrive`, cluster barrier, `fence_mbarrier_init_release_cluster` | No | absent | DIAL TritonNvidiaGPUOps | `MBarrier{count, phase}` resource + arrive/expect/wait |
| Coop | Warp specialization: `ttg.warp_specialize` (default partition + N worker partitions with own warp counts/regs), auto-WS via `tl.range(warp_specialize=True)` (Blackwell, simple matmul loops), Gluon `warp_specialize` | No | absent | DIAL TritonGPUOps (warp_specialize, warp_yield, warp_return, warp_id); GEN range.html | Partitioned-region node with per-partition warps/regs |
| Coop | Tensor memory (TMEM) alloc/load/store/copy/subslice; tcgen05 MMA/commit | No | absent | DIAL TritonNvidiaGPUOps | Accumulator-in-TMEM resource, async MMA with barrier completion |
| Coop | Clusters / multi-CTA (`num_ctas`, cluster_barrier, multicast TMA, DSMEM, CLC launch control) | No | absent: grep cluster in ir.rs = 0 | DIAL TritonNvidiaGPUOps; AMD amdg.cluster_barrier_*; Gluon tutorials https://triton-lang.org/main/getting-started/tutorials/gluon/ | `Cluster{shape}` + remote-shared addressing |
| Coop | Layouts: blocked / mma / dot_operand / shared (swizzle, linear layouts), `ttg.convert_layout`, Gluon explicit layouts | No | absent (layouts are the compiler's business in Triton; Gluon exposes them) | DIAL TritonGPUOps (ttg.convert_layout); Gluon tutorial index | Layout is a SCHEDULE concept, not intent: model only as hints plus an explicit `ConvertLayout` for Gluon |
| Coop | Warp/block collectives (shuffle, ballot, vote) | Partial | Reductions/scans lower to warp shuffle + smem trees as Schedule variants (`scan_blockscan_variant`); no user-visible shuffle/vote primitives. Triton has none either (tile-level) except via `inline_asm` | n/a | `WarpShuffle/Ballot` only for CUDA/Slang lifting, not Triton |
| Coop | AMD: buffer_load/store/atomic (scalar base + offsets), `async_copy_local_to_global`, TDM (tensor data mover) copy/gather/scatter/prefetch/wait, `scaled_upcast/downcast_fp4/fp8`, `in_thread_transpose`, `cond_barrier` | No | absent | DIAL TritonAMDGPUOps.html (33 ops) | AMD TDM descriptor + buffer-addressing modes (backend-specific; model as lowerings of the generic async-copy/descriptor ops) |

### Inline asm / extern / debug

| Category | Capability | IR | Evidence | Source | IR addition |
|---|---|---|---|---|---|
| Escape | `tl.inline_asm_elementwise(asm, constraints, args, dtype (tuple allowed), is_pure, pack)` | No | absent (ir.rs "extern" hit is a doc-comment about "externally sole-consumer"); idiom doc lists refusal of such constructs | GEN triton.language.inline_asm_elementwise.html | Opaque `InlineAsm{target, asm, constraints, pack, pure, n_out}` node (ThunderKittens/CUTLASS/CUDA need it too) |
| Escape | `ttg.inline_asm` (once per thread with explicit layouts), Gluon `inline_asm_elementwise` taking smem/tmem descriptors (needs is_pure=False) | No | absent | GEN inline_asm_elementwise.html (Gluon only paragraph); DIAL TritonGPUOps | per-thread asm with memdesc operands |
| Escape | Extern libdevice / math library calls: `tl.extra.libdevice.*` (CUDA libdevice, AMD ocml), `tt.extern_elementwise(libname, libpath, symbol, pure)` | No | absent; UnaryOp is a closed set (≈40); tail like `Lgamma`, `Erfc` exist but no open extern symbol | DIAL TritonOps (tt.extern_elementwise); package `python/triton/language/extra/libdevice.py` (GitHub contents API) | `ExternCall{lib,symbol,pure,sig}` |
| Debug | `tl.device_print(prefix,*args,hex)`, `tl.device_assert(cond,msg,mask)` (TRITON_DEBUG), `static_print`, `static_assert`, `expect_zero`, `tt.print`, `tt.assert` | No | absent; the IR is pure/functional with no side-effect ops | CORE L3405, L3455; DIAL TritonOps | `Print`, `Assert` side-effect ops (+ effect ordering) |

### Backends (UNVERIFIED beyond repo layout)

| Category | Capability | IR | Evidence | Source | IR addition |
|---|---|---|---|---|---|
| Backends | NVIDIA (sm80/90/100/Rubin tcgen05), AMD (CDNA3/4, gfx12 TDM/cluster), Intel, CPU | Partial | upstream third_party has only `amd`, `nvidia`, `proton`, `f2reduce` (GitHub contents API, 2026-10-24). Intel (intel-xpu-backend-for-triton) and CPU (triton-cpu) are separate downstream repos: UNVERIFIED here. Unpopped has backend-neutral Lowering with CUDA and Slang emitters | https://github.com/triton-lang/triton/tree/main/third_party | Backend capability mapping in the technique registry (idiom doc §8), not new IR |
| Unique | Proton profiler ops (`proton.record`, ProtonGPU) | No | absent | DIAL (proton, proton_gpu dialect list) | out of scope (profiling) |
| Unique | `triton.language.extra` per-backend libdevice, `tl.target_info`/`tt.get_num_programs` | No | absent | GitHub language/extra listing | covered by ExternCall |

## (2) Top-10 highest-impact gaps (ordered)

1. **No tile/block value model and no multi-op tile-level dataflow.** IR has one Access per OpDef; Triton kernels are arbitrary tile graphs with several dots, reductions, elementwise stages and loop-carried tiles (flash-attention, fused norm+GEMM, MoE). Needs an SSA tile-value kernel body (or at least a fused-Access composition).
2. **No program model**: `program_id`/`num_programs`/grid/constexpr meta-params/swizzle2d. Every Triton kernel starts here.
3. **No control flow**: loops with carried state, `if`/`while`, data-dependent bounds (varlen, CSR, paged attention).
4. **No arbitrary masked loads/stores nor computed multi-axis pointer arithmetic** (only one-axis `ReadIndex::Indexed` and ScatterIndexed OobPolicy). `tl.load(ptr+offs, mask, other)` is the core Triton memory idiom.
5. **No tensor descriptors / TMA / block pointers** (and descriptor gather/scatter/reduce).
6. **Contraction is rank-2 single-K uniform dtype with one AccumSpec**: no `input_precision` (tf32/tf32x3/bf16x3/bf16x6), no mixed dtypes, no `acc` input chaining, no batched, no `dot_scaled` / MX block scaling, no `max_num_imprecise_acc`, no MMA tile/Hopper/Blackwell shapes.
7. **Cooperation resources absent**: shared memory buffers, barriers/mbarriers, async copy (cp.async/TMA), warp specialization, clusters/multi-CTA, TMEM, tcgen05/wgmma (all in ttg/ttng dialects and Gluon).
8. **No explicit Cast/Bitcast with rounding modes (rtne/rtz)**, no fp4 e2m1 (packed), no float8e4b15, no U32/U64 element kinds.
9. **Atomics are scatter-write combine only**: no CAS/xchg/and/or/xor, no returned old value, no sem/scope, no atomic_load/store/poll (stream-K, persistent sync), no FP max/min.
10. **No escape hatches/side effects**: no `inline_asm_elementwise`, no extern libdevice calls, no custom `reduce`/`associative_scan` combine_fn (tuple state), no RNG (Philox), no print/assert, no `fma`/`umulhi`/precise-vs-fast div/sqrt.

Next just below the top 10: scheduling hints (`num_warps/num_stages/num_ctas/maxnreg`, per-loop `num_stages/warp_specialize/flatten`), `Select` on integer dtypes, argmax/argmin/any/all/xor reductions, sort on non-last axes (UNVERIFIED), cache/eviction modifiers.

## (3) Things in the IR that look WRONG or narrower than Triton

1. `ScalarExpr::Select` is restricted to float compute dtypes ("int select ... rejected outright"). `tl.where` accepts any dtype (https://triton-lang.org/main/python-api/generated/triton.language.where.html, UNVERIFIED page; CORE L2826). Any Triton integer/index `where` (e.g. index masking) cannot be expressed; the rationale in the doc comment (U8/I8 cond observer) is a design choice, not a Triton constraint.
2. `ScalarExpr::Coord(axis)` is float-only (f32/f64), documented as exact only up to 2^24 elements for f32. Triton's `tl.arange` is int32 and program ids are int32; using a float Coord as an index (the main Triton use) would silently lose exactness. Evidence: ir.rs Coord doc ("dtype: F32/F32Strict/F64 ONLY ... Int dtypes reject").
3. `ReduceOp::Max/Min` hard-code NaN-propagating semantics ("NaN-propagating (torch.amax)"). Triton's `tl.max/min` reduce has no NaN guarantee documented on the reduce page; `tl.maximum/minimum` take a `propagate_nan` option defaulting to `PropagateNan.NONE` (CORE L2923). The IR has `FmaxIeee/FminIeee` elementwise but no reduce-op variant for the non-propagating mode. Not wrong per se, but a Triton `tl.max` reduction lowers to a different NaN behavior than Unpopped's Max reduce.
4. `Contraction` `AxisRole::Batch` is "v1: unused" and `AccumSpec` has a single variant, while the vocabulary already has `MathPrecision::Tf32`/`PrecisionGuarantee`; the IR is therefore less expressive than the vocab crate describing the same kernels (inconsistency inside Unpopped, not vs Triton).
5. `ReadIndex::Indexed` doc says full-shape index = torch.gather, 1-D index = index_select/embedding; Triton's `tl.gather(src,index,axis)` is in-register, so the mapping holds only if the source is a memory operand. A Triton `tl.gather` over a value computed earlier in the same kernel cannot lift to this node without materializing a buffer (changes semantics for fusion).
6. `topk`: IR allows runtime `k_out <= k_in`; Triton requires constexpr power-of-two k and last-dim only (GEN topk.html). IR is a superset, fine; but Triton's tie/stability order for `descending=False` is not stated in docs (UNVERIFIED), while the IR pins NaN-greatest + stable index tie-break, so a lift of `tl.topk`/`tl.sort` cannot be claimed bit-equal without an oracle comparison.
7. Doc-level: `Access::Reduction` doc says arbitrary multi-axis and keepdim are "represented" while the emitter only generalizes later ("follow-up (item 03 step 3)") - representable != emittable; the Triton `axis=None` full reduce relies on it.

## Version/verification notes

- Verified from primary pages in this session: dot, dot_scaled, load, atomic_add, cast, make_tensor_descriptor, range, Config, autotune, jit, inline_asm_elementwise, gather, topk, associative_scan, reduce, histogram, rand, triton.language index, dialect op pages TritonOps / TritonGPUOps / TritonNvidiaGPUOps / TritonAMDGPUOps, and core.py + `__init__.py` from main (version 3.9.0).
- UNVERIFIED (page content empty or not fetched): `tl.sort` signature/dim semantics, `tl.where` page, `randint/randn` pages, libdevice function inventory, Gluon API pages (only tutorial titles and `warp_specialize` / `tcgen05_commit` page summaries seen via search: https://triton-lang.org/main/getting-started/tutorials/gluon/ , https://triton-lang.org/main/gluon/api/generated/triton.experimental.gluon.language.warp_specialize.html), Intel and CPU backends, `cast` overflow_mode, `tl.atomic_cas` sem/scope values, atomic_rmw enum member names, descriptor `padding_option` values beyond 'zero', and the TritonGPU encodings (blocked/mma/dot_op/shared layouts, ttg attribute pages not fetched).
- `tl.load` in 3.9.0 has NO `boundary_check`/`padding_option` params (CORE L2425 signature: pointer, mask, other, cache_modifier, eviction_policy, volatile). Block-pointer boundary checks are therefore not a current Triton feature (tl.make_block_ptr still exists, L2524).
