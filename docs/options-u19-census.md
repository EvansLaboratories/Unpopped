# U19 decline census over the real kernels (2026-11-22)

pm-kernels `[TASK]` 2026-11-22: run `lift_kernel` read-only and parse-only over Baracuda's CUDA kernels, Fuel's Slang kernels and FlashInfer
`prefill.cuh`; per kernel lifted or declined, with the typed decline class; histogram, top ten with an example; every class a walker slice
(`NotYetLifted`) or an IR row (`OutsideIr`); counts with their ref, the command and a control; order the next slices by how many real kernels each
unlocks. **No GPU, no emitter, no run: only the lift.**

## Headline

| Corpus | Ref | Kernels | Lifted as is | Lifted with heuristic template instantiation |
|---|---|---|---|---|
| Baracuda `crates/baracuda-kernels-sys` (all of `.cu` and `.cuh`) | `ce2911956` | 535 definitions (408 templates) | **8** | **56** |
| the same, `kernels/` only (Baracuda's own) | `ce2911956` | 431 (319 templates) | 7 | 37 |
| fishstringer `fishstringer-kernels-source` (Slang, the current home of Fuel's) | `8c203f2` | 153 entry points in 149 files | **0** | n/a |
| Fuel `fuel-kernels-source` (Slang), the 2026-09-25 corpus | `ded9bed` | 151 entry points in 147 files | **0** | n/a |
| FlashInfer `attention/prefill.cuh` (second goal kernel) | `ce2911956` (vendored) | 3 (all templates) | **0** | not measurable (see below) |
| **Control**: the decode fixture | Baracuda `ce2911956` | 1 | **1 of 1** | n/a |

Walker: this lane's `lane/u19-structs` at `dd67e9c5` (unpushed; the stack on #385). The control lifts under Baracuda's **real** CUDA frontend
(`tree-sitter-cuda` 0.21), not only under the tree-sitter-cpp the unit tests use. Fuel's `origin/main` (`f747717f4`) holds **no `.slang` file**: its 147 were
deleted on 2026-11-07 (#638) and live in fishstringer's `fishstringer-kernels-source`; both are measured, the old Fuel ref for continuity (147 files, the
2026-09-25 positive control). The old walker (`lift_elementwise`/`reduction`/`scan`) accepted 0 of 147 files; the kernel-body walker accepts **0 of 147**
too, for a different and now named reason.

Command (the tool is stored in `docs/u19-census-tool/`, scratch, not a workspace member): `census cuda baracuda-all corpus/baracuda`,
`... --instantiate`, `census slang fishstringer-kernels corpus/fishstringer`; each corpus exported with `git archive <ref> <path>` from the stated ref,
never read from a working tree. Counts are **first decline per kernel**.

## Method, and what it cannot see

- **A kernel is a textual definition.** CUDA: a `__global__` followed by a name; Slang: `[numthreads]` then `void name(`. **370 of the 388 `.cu` under
  `kernels/` contain no `__global__` at all**: they are per-dtype instantiation files (`BARACUDA_KERNELS_AFFINE_INSTANTIATE(affine_f32, float)`) over
  template kernels in the `.cuh` headers. So the 535 are **definitions, each template counted once**, not the instantiated population, which is larger
  and which only Baracuda's own list of instantiation sites knows. Counts are of definitions.
- **The first decline hides the rest.** A kernel stopped on a template parameter is not known to lift once it is given. `--instantiate` is a
  **heuristic** (type parameters to `float`, non-type to `128`, file-wide, header blanked) to see one layer deeper; it is **not** an instantiation, a real
  one is the caller's. Numbers from it are labelled. **It has a known artifact: substituting `float` for a *class-typed* template parameter (a functor `F`, a policy `POLICY`, a traits class `KTraits`) writes invalid C++ (`float::init()`), which then fails to parse. Those "grammar errors" are mine, not the parser's (checked: the unsubstituted `T acc = F::init();` and `__launch_bounds__(KTraits::NUM_THREADS)` both parse); they are excluded below.**
- **One generic context** for every kernel (block 128, dynamic shared 49152 B, `__CUDA_ARCH__` 800, `__CUDACC_VER_MAJOR__` 13): a kernel needing another launch
  shape is not penalised by a wrong guess, because the walker takes it as given.
- Noise: 11 prototype-only matches and 9 names taken from comments (`points`, `problem`) are counted as `NoSuchFunction`. 9 of 535 is the error bar on "found".
- `baracuda-cuda-parse@ce2911956` pins `unpopped-convert ~0.47`, nine minor versions behind this walker, so its `CUDA` constant cannot be handed to it. The
  census builds the same `Frontend` (its exported `parse_cuda`, its private residue list copied). **Baracuda has to bump to consume the walker at all.**
- **Static construct presence** (below) is a text match over the files that hold a kernel, per kernel in those files; it is not a decline and a kernel can need
  several. It is the only handle on what is behind Slang's universal first blocker.

## Bugs the census found in the walker, fixed on the stack (each with a regression test)

All three were invisible to the unit tests, which use tree-sitter-cpp with an **empty residue list** and no real grammar:

1. **The frontend's residue list gated the walker.** `lift_kernel*` went through the legacy `prepared()`, which refuses any source containing a word of the
   frontend's residue (`__shared__`, `atomicAdd`, `__syncthreads`, `asm`, `cp.async`, `__shfl`) as `NotAKernel`. Under Baracuda's real frontend **the decode
   kernel itself was refused**. Now only the kernel marker is checked. (`a_frontends_residue_list_does_not_stop_the_walker`)
2. **A macro the kernel never used refused it.** Any file with a function-like (or non-integer) `#define` declined every kernel in it: **456 of 535 (85%)**.
   Now such a macro is recorded and the decline happens at its **use**, naming it (`the function-like macro `F` (expansion is a later slice)`). A grammar error
   in another function of the file no longer refuses this kernel either (it counted the whole file before).
3. **A comment between parameters was a parameter**: 94 kernels stopped on "a `comment` parameter". (`comments_between_parameters_are_not_parameters`)

Effect on Baracuda, as is: 3 lifted, then 5, then 8 of 535; heuristic-instantiated: 43 then 56.

## Baracuda: where the 535 stop

**As is** (first decline; classes normalised, 21 distinct): 355 a **template type parameter** (`T`, `TIn`, `TOut`, `DType`, `BiasT` ...), 36 a **template
non-type parameter** used as a name (`BLOCK_SIZE` 20, `MAX_N` 8, `TILE_M`, `W`), 81 a **non-template type** (76 of them `void`: a `void*` parameter),
9 a call, 8 a struct parameter with a non-scalar field, 10 grammar errors, 9 noise names, 8 lifted. **391 kernels (73%) stop on a template parameter before
anything else is visible.**

**Heuristically instantiated** (56 lift, 48 of them templates; 35 distinct classes), top ten by exact construct, with an example each:

| Stopped | Construct | Class | Example |
|---|---|---|---|
| 76 | `the type `void`` (a `void*` kernel parameter) | walker slice | `kernels/gguf/dequantize.cu:29 baracuda_gguf_dequantize_q4_0_f32_kernel` |
| ~158 | a call to a **template helper function with explicit template arguments** (`load_as_acc<float>(x)` 47, `load_as_f32<float>` 28, `qptg_load_as_f32` 8, `store_from_acc` 6, `seg_zero`, `to_float` ...) | walker slice | `include/baracuda_flce.cuh:355 flce_inplace_scale_kernel`; `conv/causal_conv1d_backward_fp.cu:148 causal_conv1d_bwd_kernel` |
| ~67 | a **struct parameter with an array field** (`shape.v`, `input_shape.v`, `output_shape.v`, `shape.d`) | **IR row** (`ParamKind::Struct` takes scalar and nested-struct fields only; an array field is an IR addition, pm-kernels' call) | `include/baracuda_affine.cuh:155 affine_strided_kernel` |
| 20 | a `using X = ...;` **alias declaration** statement | walker slice | `elementwise/trace_fp.cu:61 trace_kernel` |
| 14 | a struct parameter type declared **in another header** (`baracuda::elementwise::DimsI32`) | walker slice: the caller supplies the header text, or `#include` resolution | `include/baracuda_norm.cuh:207 rms_norm_fp_kernel` |
| 11 | the typedef/macro type `floatX` | walker slice (type aliases) | `vendor/mhc/include/utils.cuh:22 float_to_bf16_kernel` |
| 8 | `cg::thread_block` (cooperative groups) | walker slice, or IR (`Collective` exists) | `vendor/mhc/src/sinkhorn_knopp.cuh:31 sinkhorn_knopp_warp_optimized_kernel` |
| 8 | `vec_t<float, 128>` (FlashInfer's vector class template) | walker slice (class templates) | `vendor/flashinfer/include/flashinfer/attention/cascade.cuh:184 AttentionSumKernel` |
| 6 | `__shared__` with more than one dimension | walker slice | `include/baracuda_bin_rcr_sm89.cuh:83` |
| 10 as is (6 not templates, 4 templates; the 12 more that appear under `--instantiate` are the artifact above) | the **one real `tree-sitter-cuda` 0.21 gap**: `extern __shared__ __align__(alignof(TempStorage)) uint8_t smem[];` | **parser** (Baracuda's lane), not the walker | `vendor/flashinfer/include/flashinfer/sampling.cuh:303 OnlineSoftmaxFusedKernel` (also 430, 509, 739, 788) |

Then, each under 10: intrinsics missing from the frontend's math table (`__double2int_rn`, `__frcp_rn`, `__powf`, `fast_sigmoid`, `INFINITY`, 4 to 6 each), a loop
bound naming an assigned variable (5), `static_assert` (3), an optional-parameter default (3), `a call to op` (functor, 11), statements (24, unclassified `X statement`).

**No `OutsideIr` appeared in any run.** Every decline is a walker slice, a parser gap, or (for the struct array field) an IR addition that is a decision, not a
refusal the IR already makes.

## Slang: one gate, then the layer behind it

All 153 (and Fuel's 151) entry points stop on the **same first construct**: 140 (138) on `the type `uint3`` at the entry parameter
`uint3 gid : SV_DispatchThreadID`, and 13 on a **grammar error** in the parser (the first ERROR node starts at `float var = ...` in 8 `layer_norm*` kernels, `uint is = ...` in `dequant_q4_km`, and `out[...] = ...` in the 4 `rope*` kernels;
minimal sources reproduce each, and a renamed control parses; the cause and any report are fishstringer's to establish). The walker's kernel model is a function whose **parameters are the
buffers**; a Slang kernel declares its resources **at file scope** and gets its thread id from a **semantic**. What is present (static, per kernel, `fishstringer`
ref; Fuel's is the same shape):

| Kernels needing | Construct |
|---|---|
| 153 / 153 | `[[vk::binding]]` + `StructuredBuffer`/`RWStructuredBuffer` globals + `ConstantBuffer<Params>` + an `SV_` semantic parameter + `struct` + vector types |
| 60 | `half`/`bf16` and a helper function returning a value |
| 38 | `groupshared` and barriers (`GroupMemoryBarrierWithGroupSync`) |
| 29 | `Wave*` intrinsics |
| 19 | `Interlocked*` atomics |
| 12 | `while`/`do` loops |

## FlashInfer `prefill.cuh` (second goal kernel): not near

Three kernels (`SinglePrefillWithKVCacheKernel`, `BatchPrefillWithRaggedKVCacheKernel`, `BatchPrefillWithPagedKVCacheKernel`), all templates over a `KTraits` class. As is, all
three stop on `the type `Params``, a template parameter. **What is behind that is not measured:** `KTraits` is a traits class, so the `float` substitution cannot instantiate it
(it writes invalid C++), and a real instantiation needs the caller's class definitions. What is present, statically, in all three: `cp.async`, `mma`/tensor cores, `asm volatile`,
`__shared__`, barriers, vector and half types, `struct`/`class` types and `cutlass::`/`std::` library types. The decode kernel needed none of the class-template machinery; prefill is
built on it. **The verdict "not near" rests on those constructs and on `KTraits`, not on a parser error**, and I would not promise it from slices of this size.

## The next slices, by how many real kernels each is stopping

A count is the kernels **stopped** by the class (an upper bound on what the slice unlocks: they then meet the next layer, so every slice is re-measured).

1. **Template instantiation from the caller (Baracuda, 391 kernels stopped; 8 -> 56 lift even under the crude heuristic).** `LiftContext::with_type("T", ..)` and
   `with_const("BLOCK_SIZE", 256)`, bound before the walk. The caller has the list (its `_INSTANTIATE` macro sites); the census cannot. Smallest, biggest, and
   it makes every later number honest.
2. **Template helper calls with explicit template arguments (~158 after 1).** Inline `load_as_acc<float>(x)` binding the helper's own parameters.
3. **`void*` parameters as byte buffers (76).** A `U8` buffer with `Place::Raw` loads through the casts the walker already lowers.
4. **Slang entry-point model (140 of 153, and Fuel's 138 of 151).** `SV_DispatchThreadID`/`SV_GroupThreadID`/`SV_GroupID` to the `ExecId` leaves; global
   `StructuredBuffer`/`RWStructuredBuffer` as buffer parameters in binding order; `ConstantBuffer<Params>` as a `ParamKind::Struct`. Largest single gate in the
   corpus, and 0 Slang kernels lift without it. fishstringer owns the Slang front end: ask whether the model lives in the walker or in `SLANG`.
5. **Struct parameters with array fields (~67): an IR decision first** (ask pm-kernels), then the walker.
6. **Type aliases and cross-header types** (`using`, `floatX`, `DimsI32`: ~45) and **small tables** (intrinsics, `INFINITY`, multi-dimensional `__shared__`: ~30).
7. **Parser gaps (CUDA: one construct, 10 kernels; Slang: 13):** `tree-sitter-cuda` rejects `extern __shared__ __align__(alignof(T)) uint8_t smem[];`; `tree-sitter-slang`
   stops on the 13 Slang kernels above ("stopped on grammar"; the cause is fishstringer's to establish). Not the walker's. The CUDA one is Baracuda issue #420.

**Not in this note, and needed before anyone plans past slice 3:** the instantiated population (Baracuda's list), the second layer under Slang's gate (needs
slice 4 built), and a macro-expansion slice (`#define F(x)` appears in 456 of the 535 kernels' files; the walker now only declines at a use, which this run found
rare).


## Re-measured after slices 1, 2, 3 and 5 (2026-11-22, branch `lane/u19-templates`, same corpus and tool)

Same Baracuda corpus, same method (`--instantiate` heuristic: type parameters `float`, non-type `128`), same generic context. **Lifted: 80 of 535 kernel definitions
(was 56; 72 of them template kernels, was 48)**; with no instantiation 8 of 535 (unchanged: a template kernel with no arguments now declines as "the template parameter `T` has no
value", not as "template"). The 24 gained are not "391 + 158 + 76 + 67 unlocked" as the census order hoped: **each slice removed the first wall and met the next one**.
What the first decline is now, by count (first decline per kernel):

- **160 + 40 + 13: a call to a helper defined in another file** (`load_as_acc<float>`, `load_as_f32<float>`, `fast_sigmoid`, `op`). The extraction is per file and `#include` is
  dropped, so a helper in `baracuda_dtype_promote.cuh` is invisible to a kernel in `causal_conv1d_backward_fp.cu`. **This is now the largest wall and it is not a walker
  construct: it needs the included text.** Slice 2 works when the helper is in the same source (tested), and finds none here. Candidate next slice: `LiftContext::with_include(name, text)`
  (the caller supplies the file's includes; the walker splices them in front, no filesystem read). Not built; needs pm-kernels' ruling (a caller-supplied prelude widens the preprocessor).
- **49 + 20 + 11: types and names from other headers** (`DimsI32`, `floatX`, `INFINITY`): the same include wall plus a small intrinsic and constant table.
- **26 + 18: `using T = ...;`** inside the kernel (an `alias_declaration` statement): not modelled.
- **13: a local array with an initialiser** (`int a[3] = {..}`): the "later slice" the decline names.
- 10 FlashInfer kernels decline "the template parameter `X` of `X` has no value": the heuristic does not name their parameters; a caller who gives them lifts as the decode goal test does.

So slice 1 to 3 and 5 are built and correct (each tested on the oracle, mutants caught), and they move 56 to 80; the next 100-odd kernels wait on includes, not on the IR.
**The `for` loop that counts down** (`for (d = rank - 1; d >= 0; --d)`, which every Baracuda strided kernel runs over its `shape.v[d]`) was a hidden gate behind slice 5: it was found by
slice 5's own test and is lifted now (`tests/kernel_body_descending_for.rs`).

— lane: unpopped
