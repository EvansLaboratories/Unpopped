# The IR as the full union of kernel languages: gap matrix and design

Status 2026-10-25. Docs only; **no code, no builds.** Author: Unpopped lane. Gate: the PM, who allocates
every version and rules on every decision in section 7.

## 0. The directive, and what it overrules

CireSnave, verbatim, 2026-10-24: *"I want the IR to cover the full union of the capabilities of every
kernel language in common use today including, but not limited to, CUDA, CUTLASS, Slang, CuTe, Mojo,
Triton, and ThunderKittens.  If Unpopped's IR were a union of all of those, there is nothing that could
be expressed in any of those that Unpopped couldn't express in it's IR."*

Scope ruling, verbatim, relayed by the PM 2026-10-25: *"for now, CUDA, CUTLASS, and Slang parsers and
emitters should be sufficient as long as Unpopped builds their IR based on the full union of the
capabilities of all popular kernel languages."* So the IR is built from all seven languages; parser and
emitter **modules** exist only for CUDA, CUTLASS (both baracuda) and Slang (Fuel) for now. Triton, Mojo,
CuTe and ThunderKittens are inputs to this design, not modules.

PM ruling 2026-10-24: **the IR models cooperation** (shared memory, barriers, warp/block collectives,
tensor-core tiles, async copy/TMA, clusters). That reverses `docs/idiom-lifting-design.md` section 1a
(*"the IR never models cooperation"*) and its "not transferable" list (lines ~714-720: cp.async, TMA,
wgmma, clusters, named barriers, warp specialization). Both are to be **replaced, not caveated**, in the
first increment (I0 below). `docs/groupshared-sizing.md` carries the same premise and is replaced with it.

"Cannot express" in this document means *not built yet*, never *out of scope*. Every row is a backlog item.

## 1. What the IR is today (verified at `origin/main` `a575e66`)

- `OpDef` = one scalar expression `body` + one `Access` + per-input `View`, `ReadIndex`, `BaseOffset`
  + one single-output `WriteIndex`. `ScalarExpr` has 12 variants: `Input, Const(f64), Param, Reduced,
  Coord, Add, Sub, Mul, Div, Unary, Binary, Select`. No `Cast`, no `Fma`, no local, no loop, no vector.
- `Access` = `Elementwise, Reduction, RowReduce, Contraction, Scan, Window, RowSort, Im2Col`.
- `AccumSpec` has exactly one variant, `WideFloat` (`ir.rs:1933`); the doc comment says tensor-core and TF32
  "join as variants". `Schedule::Contraction` is a unit variant.
- Data-dependent indexing **exists**: `ReadIndex::Indexed` (gather / index_select) and
  `WriteIndex::ScatterIndexed` with `WriteCombine::{Assign, AtomicAdd, AtomicMax, AtomicMin}`.
  Packed-weight + scale-sibling contraction (sk4) exists in the vocabulary. `ContractionAxes::batched_matmul()`
  exists, although the `Contraction` doc comment still says "v1 rank-2".
- The vocabulary is **wider than the IR**: `OpCategory::Attention` is a reserved discriminant with no IR
  node behind it; `MathPrecision` (TF32 etc.) exists in `unpopped-vocab` but is not an `OpDef` field.
- **Cooperation exists today only as emitter-chosen schedules** (block-parallel reduction with warp shuffle
  + shared-memory tree, Kogge-Stone block-scan, bitonic row-sort): `Schedule` variants in `plan.rs`. The
  IR cannot say it, an emitter may choose it. This is why the grep results below are zero in `ir.rs` and
  non-zero in comments and schedules.

Absence evidence, positive-controlled (concatenation of `ir.rs`, `plan.rs`, `ops.rs`, `element.rs`,
vocab `plan.rs` at `a575e66`): `Let` 0, `Loop` 0, `ThreadIdx` 0, `swizzle` 0, `mbarrier` 0, `tma` 0,
`cluster` 0, `ballot` 0; `barrier` 1, `warp` 5, `shuffle` 9, `shared` 49, `Cast` 2, `Fma` 1,
`compare_exchange|CAS` 8 are **all comments or unrelated op discriminants, none an IR construct** (read
individually). Controls that the greps can see things: `Indexed` 50, `Reduced` 133, `AtomicAdd` 16,
`Select` 44, `Contraction` 96.

## 2. How the gap list was made, and its limits

Seven read-only Sonnet surveys, one per language, each comparing the language's documented capabilities
(official docs, URL per row) to the IR source. Raw reports: `docs/ir-union-surveys/<language>.md`. Limits:

- Each report marks rows it could not confirm from a primary page **UNVERIFIED** (74 such lines in total);
  this document treats those as unconfirmed, not as gaps. Mojo's docs moved (docs.modular.com to
  mojolang.org) and several package pages were reachable only as indexes; Triton Intel/CPU backends are
  downstream repos; the Slang atomics-per-target table and several CUDA intrinsics were not reachable.
- The agents contended on `git fetch` and read `a575e66` (which is current `main`), so the IR side is not stale.
- IR claims **re-verified by me in the source** for this document: no `Access::Attention`; `Coord` float-only
  (`F32/F32Strict/F64`; the doc names the integer spelling "the queued follow-up"); `Select` float-only
  (f32/f32s/f64/f16/bf16); `AccumSpec` single-variant; the MX element formats absent from `ElementKind` by
  a *maintainer ruling of 2026-08-08* (`element.rs` near `F8E8M0`); the absence counts in section 1.
  Everything else in the matrix is **as reported by the surveys** and must be re-read before an increment
  starts on it. That re-read is the first task of each increment.
- Features that are not kernel-body capabilities (generics, comptime, autotune configs, autodiff) are
  classified in 3.20 rather than counted as gaps.

## 3. The gap matrix

Language columns say whether that language's survey lists the capability as a gap (X) or does not
(blank). The marks are **approximate**: I read them from each survey's top-10 list and table, not from an
exhaustive count, and a blank means "not listed", not "checked and absent". Increment is the section-5
increment that closes it.

| # | Gap | CUDA | CUTL | CuTe | Slang | Mojo | Trit | TK | Inc |
|---|---|---|---|---|---|---|---|---|---|
| 3.1 | **Statement layer**: local variables, mutation, `for`/`while`/`if`/`break`/`return`, device function calls, predicated regions | X | X | X | X | X | X | X | I2 |
| 3.2 | **Execution model**: thread/lane/warp/block/cluster ids and dims, scope (thread, warp, warpgroup, block, cluster, grid), launch configuration, `program_id`/`num_programs` (block-SPMD), persistent kernels | X | X | X | X | X | X | X | I3 |
| 3.3 | **Memory spaces and allocation**: shared (static/dynamic), local/register fragments, constant, tensor memory (TMEM), distributed shared memory; typed load/store; pointers, byte-offset access, `Bitcast` | X | X | X | X | X | X | X | I3 |
| 3.4 | **Layout algebra and swizzle**: hierarchical shape:stride, composition, complement, logical divide/product, coalesce, `Swizzle<B,M,S>`, partitioning, stride-0 inside hierarchy | X | X | X |  | X | X | X | I6 |
| 3.5 | **Synchronization**: block/named/cluster barriers, `mbarrier` (phase, expect_tx, split arrive/wait), scoped and proxy fences, async commit/wait groups | X | X | X | X | X | X | X | I4 |
| 3.6 | **Collectives**: warp shuffle/vote/match/elect/redux, block reduce/scan/broadcast, cooperative groups, wave/quad ops, clustered reductions (as nodes inside a larger kernel, not whole-op schedules) | X |  | X | X | X |  | X | I5 |
| 3.7 | **Tensor-core tile ops**: `mma.sync`, WGMMA, `tcgen05`/UMMA with TMEM accumulators, 2:4 sparse `.sp`, block-scaled MX/NVFP4, TF32/3xTF32/bf16x9, accumulate-into-D, fragment type; `AccumSpec` is one variant | X | X | X | X | X | X | X | I7 |
| 3.8 | **Async data movement**: `cp.async`, TMA (tile, bulk, multicast, reduce, gather4/scatter4, im2col, descriptors, block pointers), `ldmatrix`/`stmatrix`, TMEM load/store | X | X | X |  | X | X | X | I8 |
| 3.9 | **Pipelines and roles**: stages, producer/consumer, warp specialization, `setmaxnreg`, persistent tile schedulers (static, stream-K, cluster launch control), rasterization, cluster shape | X | X | X |  | X | X | X | I8 |
| 3.10 | **Atomics and the memory model**: CAS, exchange, and/or/xor, inc/dec, float min/max, returned old value, memory order, scope, atomics on shared memory, 64-bit and unsigned | X |  | X | X | X | X |  | I4 |
| 3.11 | **Dtypes**: fp4 e2m1, fp6 e2m3/e3m2, fp8 e3m4/e4m3b15, UE4M3/UE5M3 scales, packed/vector types (`half2`, `float4`, 128-bit), `Cast`/`Bitcast` with rounding and saturation; the 2026-08-08 ruling keeping MX elements out of `ElementKind` | X | X |  | X | X | X | X | I1 |
| 3.12 | **Numerics controls**: `Fma` node, rounding-mode suffixes, ftz/denormal, approx vs exact transcendental, fast vs precise div/sqrt, NaN/Inf preservation, `dp4a` wrap/saturate | X | X |  | X |  | X |  | I1 |
| 3.13 | **Contraction generalization**: mixed A/B/accumulator/output dtypes, accumulate into C, epilogue reading `C`, bias, row/col broadcast, aux load/store, multi-output, fused reductions (dbias, amax) i.e. EVT, grouped and pointer-array GEMM, split-K / stream-K / sliced-K, structured sparsity, block-scaled and mixed-input (dequant role), batched, `out_dtype` beyond U8/I64, xor-popc fold for b1 | X | X | X |  | X | X | X | I7, I9 |
| 3.14 | **Attention** as an intent node: score contraction + softmax + value contraction without materializing scores; online-softmax state `(max, sum)`; causal/ALiBi masks; paged KV via block-table addressing; MLA; backward | X | X | X | X | X | X | X | I10 |
| 3.15 | **Implicit-GEMM convolution**: fprop/dgrad/wgrad, NHWC, 1D/3D, grouped, TMA-im2col (today `Im2Col` is explicit, NCHW, groups=1, symmetric pad) | X | X |  |  |  |  |  | I9 |
| 3.16 | **Reductions/scans/sort with richer state**: tuple state and custom combine functions, `argmax/argmin/any/all/xor_sum`, non-propagating NaN max/min reduce variant, histogram, `topk` variants | X |  |  |  |  | X | X | I5 |
| 3.17 | **Indexing**: masked load/store with `other` fill, computed multi-axis pointer arithmetic, gather combined with a `View`, in-register gather, integer index dtypes beyond I32/I64, integer `Coord`/`Select` | X |  | X | X | X | X |  | I1, I2 |
| 3.18 | **Resources**: textures, samplers, images, bindless handles, push-constant/`ParameterBlock` structs, struct/array/vector/matrix values, non-scalar `Param` | | |  | X |  |  |  | I3 |
| 3.19 | **Escape hatches**: inline PTX / `inline_asm_elementwise` / `spirv_asm`, extern device libraries (libdevice) | X |  |  | X |  | X |  | I11 |
| 3.20 | **Front-end-only (not IR)**: see 3.20 below | | |  | X | X | X |  | n/a |
| 3.21 | **Multi-GPU**: peer pointers, multicast, `multimem` ld_reduce/red/st, peer TMA (ThunderKittens PGL, NVLink) | | |  |  |  |  | X | I12 |
| 3.22 | **Existing IR defects the surveys found** (each a fix in its increment): `Coord` Elementwise-only and float-only; `Select` float-only; `Param` f32-only; `ReadIndex` index dtype I32/I64; `u8` Window/Im2Col parameters (windows > 255 inexpressible); `ReduceOp` single-state; `Reduction` NaN-propagation hardcoded; `B1` contraction pinned to a sum fold (test `a_contraction_cannot_express_a_non_sum_fold` must be revisited); the `Contraction` doc says "rank-2" although batched exists; `element.rs` says f32 GEMM "routes through TF32 tensor cores" though the only accumulator is SIMT | | |  |  |  |  |  | per row |

### 3.20 Classified as front-end, not IR (decision D3 asks the PM to confirm)

A front end resolves these before the IR exists; the IR carries the concrete result. **Generics /
interfaces / specialization constants** (Slang, CuTe templates, Mojo `comptime`), **autotune configs**
(Triton `num_warps/num_stages`, CUTLASS tile configs) as *schedule hints on the lowered kernel*, **Mojo
parameter metaprogramming**. Two do need an IR home: **automatic differentiation** (Slang `fwd_diff` /
`bwd_diff`) is a *transformation over the IR* that needs 3.1; **RNG (Philox), `device_assert`,
`device_print`** are side-effecting ops that need the statement layer's effect model.

## 4. Architecture: two layers, one union

The surveys' common finding is structural. Seven independent reports reach the same list because the IR is
a single per-output-coordinate scalar expression with one `Access`, while the languages are written as
statement programs over memory spaces, threads and tiles. Adding nodes one at a time to `ScalarExpr` cannot
close that; it would make `ScalarExpr` a statement language by accretion.

**Proposal (decision D1).** Keep today's `OpDef` as the **intent layer** and add a **kernel layer** beneath it.

- **Intent layer** (exists): "what is computed" per output coordinate. This is what the optimizer, e-graph,
  `contract`, `pattern` and the conformance oracle understand, and what idiom lifting targets. It gains new
  intent nodes where the semantics are real and compact: `Access::Attention` (3.14), generalized
  `Contraction` (3.13), implicit-GEMM `Conv` (3.15), and richer `Scan`/`Reduction` state (3.16).
- **Kernel layer** (new): "how, on this machine model". A structured, typed statement IR: locals, loops, `if`,
  memory-space-typed tensors with layouts (3.3, 3.4), execution scopes and ids (3.2), barriers/fences (3.5),
  atomics with order and scope (3.10), collectives (3.6), MMA/copy atoms (3.7, 3.8), pipelines and roles
  (3.9), escape hatches (3.19). A kernel-layer function can be **total over the union**: anything any of the
  seven languages writes has a kernel-layer representation, so "nothing expressible that Unpopped cannot
  express" is a property of this layer.
- **Lifting** connects them: a parser tries to lift a construct to an intent node (optimizable, portable) and
  otherwise keeps it as a kernel-layer body. Nothing is refused for being cooperative. Residue lists
  (`CUDA_RESIDUE`, `SLANG_RESIDUE`) shrink to "kernel-layer-only" markers, not refusals.
- **Emitters declare a capability set** (`IrFeature`: which kernel-layer features and intent nodes they lower)
  and return the existing typed `Decline` for the rest. This is what keeps a total IR honest: an emitter for a
  target without shared memory declines, it does not mis-emit.
- **Two program models, one IR.** A kernel records `ProgramModel::ThreadSpmd` (CUDA, Slang, Mojo thread
  code, CuTe) or `BlockSpmd` (Triton: tile-valued, `program_id`). Lowering `BlockSpmd` to `ThreadSpmd` is a
  pass inside Unpopped, not each emitter's problem (decision D6).

Why not a single layer: the intent layer's value (rewrite rules with bit-exactness proofs, the oracle) comes
from being small and pure. A union of seven statement languages in the same node set would void those proofs.
Why not only a kernel layer: it would discard the part of Unpopped that already works.

## 5. Increments

Each increment is a **breaking** change, one PM-allocated version (pre-1.0: the second number), docs and
tests first, one oracle each. Nothing here is started; the order is by dependency, then by how many surveys
list the gap. **Every increment begins by re-reading its rows against the source** (section 2).

| Inc | Closes | Delivers | Depends on | Oracle |
|---|---|---|---|---|
| I0 | decisions, docs | Replace idiom-lifting 1a, its "not transferable" table, and `groupshared-sizing`'s premise with the cooperation ruling; fix the stale claims in 3.22 docs; record the ownership split. Docs only | D1-D8 ruled | doc review |
| I1 | 3.11, 3.12, 3.17 (part) | `Cast`/`Bitcast` with rounding+saturation; `Fma`; rounding/ftz/approx qualifiers; new `ElementKind` rows for fp4/fp6/e3m4 and MX scales (reverses the 2026-08-08 ruling, **D5**); packed/vector values; integer `Coord`/`Select`; wider index dtypes | D5 | exhaustive small-dtype tables vs cpu-c, as the f32 harness does |
| I2 | 3.1, 3.17 | Kernel layer v0: `Let`, assignment, `For`/`While`/`If`/`Break`/`Return`, calls, effects model; lifting of today's `OpDef` into it (so the layers agree on every existing op) | I1 | **differential**: every existing `OpDef` lowered through the kernel layer equals its intent-layer emission on cpu-c, bit-exact |
| I3 | 3.2, 3.3, 3.18 | Execution scopes and ids; memory spaces and allocation; typed load/store; pointers; resources (textures/samplers/structs); `ProgramModel` | I2 | cpu-c emits a serialized "thread loop" reference for `ThreadSpmd`, so semantics are testable without a GPU |
| I4 | 3.5, 3.10 | Barriers, fences, mbarrier, async groups; full atomics with order/scope/old-value, on shared memory too | I3 | litmus tests on cpu-c (sequential model) + device legs on the self-hosted runner |
| I5 | 3.6, 3.16 | Collectives as nodes; tuple-state reductions/scans with custom combine; arg-reductions; non-propagating NaN reduce | I3, I4 | cpu-c reference collectives; existing exhaustive harness |
| I6 | 3.4 | Layout value + algebra (hierarchical shape:stride, composition/complement/divide/product/coalesce/swizzle) **as pure vocabulary with a property-test suite**; `View` re-expressed over it | I1 | algebra laws (CuTe's documented identities) as property tests; `View` equivalence on all existing cells |
| I7 | 3.7, 3.13 (part) | MMA atoms (`mma.sync`, WGMMA, tcgen05/UMMA, sparse, block-scaled) and fragments; `AccumSpec` variants (TF32, 3xTF32, int32, f16, fp8 fast-accumulate); mixed-dtype contraction; accumulate-into-C | I3, I6 | exact integer MMA vs cpu-c; float MMA by accumulation-order contract |
| I8 | 3.8, 3.9 | Copy atoms (cp.async, TMA family, ldmatrix); pipelines, roles, persistent schedulers | I4, I6, I7 | device legs; cpu-c reference as a sequential copy |
| I9 | 3.13 (rest), 3.15 | EVT-style epilogue trees, grouped/ptr-array, split/stream-K, sparse, block-scaled GEMM; implicit-GEMM conv | I7, I8 | reference GEMM/conv in cpu-c; CUTLASS emitter output compared |
| I10 | 3.14 | `Access::Attention` incl. online-softmax state, masks, paged KV (block-table addressing over `ReadIndex`), backward | I5, I7, I9 | naive attention in cpu-c; flash vs naive within a stated ULP contract |
| I11 | 3.19 | Typed `Extern`/inline-asm with declared effects and a declared per-target body | I3 | cpu-c cannot run asm: legs per target; the declared effects are what is tested |
| I12 | 3.21 | Peer/multicast scopes and multimem ops | I4, D4 | multi-GPU device legs |

Each increment closes its rows in `ROADMAP.md` section 6 and updates the survey rows it consumed.

## 6. Module interface for the parser and emitter owners

The existing seams are the interface; this section says what changes.

- **Parser = `Frontend` descriptor** (`convert.rs`; for CUDA it moves to `baracuda-cuda-parse` once baracuda's
  condition (b) holds). It gains a *kernel-layer path*: where lifting to intent fails today it returns
  `Residue`; after I2 it returns a kernel-layer body. The `Frontend` type becomes `#[non_exhaustive]`
  (breaking, in I2). A CUTLASS front end is the CUDA/C++ front end plus a **recognizer set** for CUTLASS
  template idioms (collective mainloop, epilogue visitor tree, tile scheduler); CuTe and ThunderKittens
  recognizers are the same shape, deferred.
- **Emitter = `Backend`** with a declared `IrFeature` set (new) and the existing typed `Decline`.
  - **CUDA emitter** (`baracuda-cuda-emit`): lowers the kernel layer to CUDA C++; first consumer of I3-I5.
  - **CUTLASS emitter** (baracuda), per the PM ruling 2026-10-24: **emit CUTLASS C++ source (template
    instantiations and EVT) for intent nodes that map to a CUTLASS collective** (`Contraction`, grouped,
    `Conv`, epilogue trees, later `Attention`), **raw CUDA otherwise**. `baracuda-cutlass` stays a runtime
    wrapper, not the emitter. The ruling is revisited only if a node class fits neither.
  - **Slang emitter** (`unpopped-slang`, handed to Fuel): lowers the kernel layer to Slang; the in-tree
    crate is the reference until Fuel accepts ownership (ROADMAP P3).
- **Conformance stays here.** `unpopped-conformance` and the exactness ledger define what "agrees with the
  oracle" means per target; owners run it, they do not redefine it.
- Triton, Mojo, CuTe and ThunderKittens modules do not exist; their surveys are the acceptance test that
  the IR is wide enough for a future front end.

## 7. Decisions for the PM (each also sent as an `[ASK]`)

| ID | Decision | Recommendation | Ruling (2026-10-24, PM) |
|---|---|---|---|
| D1 | Two layers (intent + kernel) vs growing `ScalarExpr` into a statement language | Two layers (section 4) | **Yes** |
| D2 | Is the typed escape hatch (inline asm / extern, 3.19) in the IR? "Nothing unexpressible" is not achievable without it | Yes, with declared effects and per-target bodies; it is opaque to the optimizer by construction | **Yes** |
| D3 | Generics, specialization constants, comptime, autotune configs are front-end, not IR (3.20); autodiff is a transformation over I2; RNG/assert/print are effects | Confirm | **Confirmed** |
| D4 | Multi-GPU (PGL, multimem) in scope (3.21) | In scope, last (I12) | **Last** |
| D5 | Reverse the 2026-08-08 maintainer ruling keeping MX element formats (fp4/fp6) out of `ElementKind` | Reverse: they are first-class types in CUTLASS, Mojo, Triton, ThunderKittens; keep the `structure_key` guard that a bare dtype position cannot name them without a scale | **PENDING: CireSnave's call (board 290). Nothing touching D5 proceeds until he rules** |
| D6 | `ThreadSpmd` and `BlockSpmd` as one IR with a lowering pass inside Unpopped | Yes | **Yes** |
| D7 | Emitters declare an `IrFeature` set and decline the rest, rather than every emitter implementing every feature | Yes | **Yes** |
| D8 | Versioning: each increment is breaking; batch I1-I2 into one pre-1.0 minor, others one each | PM allocates | **Each breaking increment is one pre-1.0 minor; I1+I2 together** |

## 8. Corrections to existing docs (done in I0, 0.22.19, except the last item)

`idiom-lifting-design.md` 1a and lines ~714-720; `groupshared-sizing.md` premise (and its header saying the
question is "answered"); the `Contraction` doc ("rank-2"; `AxisRole::Batch` "unused"); `element.rs` TF32 claim
for f32 GEMM; `joint-gpu-milestone-plan.md:28`; `normative-seams.md:292` (`supports_dtype` returns bool);
`slang-second-blocker-sizing.md:77-79` (`Select` is float-only in Slang); the Slang `lower()` message about
i16/u16 and "vocabulary v5" (**not done in I0**: it is panic/error text that adopters
match, a cross-repo API; fix it in a code increment with a version note).

## 8b. The layout/shape frontier (baracuda SA-9), folded in and decided

CireSnave, 2026-10-25, via the PM: *"SA-9: Fold them into the Unpopped union design and decide there with the evidence in hand."* Baracuda's list (its `ROADMAP.md` SA-9, with `docs/superpowers/specs/2026-08-01-ir-layout-shape-parent-design.md` section 5-8 and `...-subspec-a-roles-layout-design.md` A.6): output views, rank-changing `Reshape`, general strides, multi-K-group einsum, TF32/tensor-core `AccumSpec`, a faster transposed-RHS schedule. Evidence read: those specs and the cited `plan.rs` View checks (`assert_valid_views`); **not** run on a device, and baracuda's emitter code was not re-read here.

All of it lands under increments already in section 5: layout is **U6** (the `Layout` value and algebra, with `View` re-expressed over it), contraction structure is **U7**/**U9**. None needs a new increment. Decisions:

| Item | Decision | Why / where |
|---|---|---|
| **B. Output views** (`write_slice`) | A store-side `Layout`, the dual of the input view, using the **same** type as the input side. Mutually exclusive with a `WriteIndex` scatter in v1 (declined as a pair, typed). | Symmetry avoids a second mechanism; scatter is already a data-dependent output remap, and composing the two needs a rule nobody has asked for. **U6.** |
| **C. Rank-change `Reshape`** | A layout operation, not a re-key of `StructureKey.rank`: a per-operand read-through (unravel at the producer rank, ravel at the consumer's). **Contiguous producers only**; a non-contiguous reshape is a typed decline (it needs materialization). | Keeps rank in the key meaning "the iteration space". It is the missing piece of im2col-GEMM-reshape Conv2D fusion, so it unblocks U9's implicit-GEMM conv. **U6.** |
| **D. Arbitrary, negative, stride-2 strides** | Strides are part of the `Layout` value. A stride is **static** (folded into the address math) or **runtime** (a launch argument, keyed by a runtime-strided contiguity class); negative strides are a signed stride (baracuda folds them through `OperandKey.flipped` today). Agreement between a view and the runtime strides stays a caller precondition, stated, not checked. | One mechanism for all three cases; baracuda's own spec calls D the natural extension of A's binding. **U6.** |
| **Multi-K-group einsum** | `ContractionAxes` generalizes from one contracted group to a list; nested K folds, one accumulator per group order stated in the node. v1 of U7 keeps a single group; the list form is the same increment's second step. | An accumulator-structure change, not a layout one, so it belongs with the MMA/contraction work. **U7.** |
| **TF32 / tensor-core `AccumSpec`** | Already scheduled: U7 lists TF32, 3xTF32, int32, f16 and fp8 fast-accumulate variants. Confirmed as a compute-precision axis with its own contract flips (KISS-Contract 6.8), **not** a layout node. | Baracuda pulled it out of its layout charter for the same reason. **U7.** |
| **Coalesced smem-staged transposed-RHS schedule** | **No IR change.** A schedule variant in the emitter, bench-gated, like `contraction_splitk_variant`. The IR's only duty is that the layout is visible enough (U6) for the emitter to recognize a transposed RHS. | Correct-first, optimize-as-variant is baracuda's ratified scoping; the IR must not encode a speed trick. Emitter side: baracuda. |

**Order.** U6 first (B, C, D together, with a property-test suite as the U6 row requires), then U7 (multi-K, TF32). Nothing here blocks or is blocked by the integer-index work already landed.

**What is open, not decided.** Whether the runtime-strided contiguity class is a `StructureKey` field or a plan-level attribute: it changes the cache key's meaning, so it is the U6 design's first question and goes to the PM as an `[ASK]` when U6 starts, with a recommendation then. Baracuda owns the emitter side of every row and follows once the IR nodes land.

## 9. What this document does not establish

It does not prove the two-layer design is right; it argues for it from seven surveys that agree. The
agreement is partly shared method (the same brief, the same IR reading), so it is weaker evidence than seven
independent studies. It does not size any increment. It has not been reviewed by baracuda, Fuel or KISS.
UNVERIFIED rows in the surveys are not counted as gaps here.
