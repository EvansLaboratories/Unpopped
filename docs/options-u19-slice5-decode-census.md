# U19 slice 5: what Baracuda's flattened decode kernel needs that the walker and the IR do not have yet

Status: **census and options, no code** (unpopped lane, 2026-11-21, after slices 1 to 4b). Input read: `flashinfer_paged_decode_f16_d128_g1_flat.cu`,
572 lines, Baracuda `docs/planning/u19/` (branch `r6-reviews-2`, `cc6183d86`, unpushed when read), one config (half, `HEAD_DIM=128`, `kNone`, group 1). Counts below
are `grep -c` over that file at that commit, so they describe the file, not the walker.

## The file, by the numbers

22 `__device__` functions, 2 `__global__` mentions, 4 structs, 11 `asm volatile`, 25 `#pragma unroll`, 6 `alignas(16)` local arrays, 9 `#if`/`#ifdef`,
7 pointer casts, 11 by-reference parameters. The kernel takes one `const __grid_constant__ DecodeParamsHalf params`: **a struct holding 8 pointers**
(`q`, `o`, `lse`, `request_indices`, ...), one nested struct (`PagedKvHalf`), a `bool*`, and 13 scalars. Its dynamic shared memory is
`extern __shared__ uint8_t smem[]`, carved by pointer arithmetic and cast to `half*`, `float*` and `size_t*`.

## What each need maps to, and who owns the gap

| need in the file | status after slice 4b | what closes it | kind |
|---|---|---|---|
| `#pragma unroll`, `__syncthreads`, thread ids, `__ldg(&a[i])`, math calls | built (slices 2 to 4a) | - | done |
| 22 `__device__` callees, by-value and by-reference (`State&`, `float*`) parameters | declined (`a call to ...`) | inline the callee at the call site in the walker; a by-reference scalar or struct becomes the caller's own locals; or lift each callee to a `KernelFn` and use `Stmt::Call` (U3) | walker, M to L |
| `struct` locals with array members (`State { alignas(16) float o[8]; float m, d; }`) | declined | scalarize: one local per field and per array lane, as slice 4b does for `float4` | walker, M |
| local arrays `alignas(16) float v[8]`, indexed by a loop counter | declined | `Private` `AllocDecl` + `Place::Alloc` (exists; needs a launch) | walker, S |
| pointer values: `const half* q = params.q;`, `p + i`, `(half*)smem`, a pointer passed to a callee | declined | **no pointer value in the IR by ruling** (U9c, `docs/options-u9c-pointer-values.md`, level (b): a base named by a parameter plus an integer offset). The walker can carry a **symbolic pointer** (base + byte offset expression) while it inlines, and lower every dereference to `Place::Raw` or `Place::AllocRaw`. A pointer that is stored to memory or compared stays a decline. | walker, L; needs a ruling (Q6) |
| `extern __shared__ uint8_t smem[]` carved into `half*`, `float*`, `size_t*` regions | declined | one dynamic `Shared` allocation (`Extent::Param`) whose size the caller names (the source does not carry it), read through `Place::AllocRaw` with the region's byte offset; U9c built `AllocRaw` | walker + caller input, M; Q7 |
| `__grid_constant__ DecodeParamsHalf params` holding **pointers** | **IR gap**: `ParamKind::Struct` fields are scalars (U23). A pointer field has no home: it is neither a `Buffer` parameter nor a scalar | a struct parameter whose fields may be buffers (each field a buffer with its own element type and role). `layout_only_guard` and the patent read apply: layout only, no thread-set or memory-kind annotation | IR (additive: `ParamKind` and `StructField` are `non_exhaustive`), M; Q8 |
| `#if (__CUDACC_VER_MAJOR__ >= 11)`, `#ifdef FLASHINFER_CP_ASYNC_ENABLED`, `__CUDA_ARCH__` guards | declined (a preprocessor conditional is not parsed) | the caller supplies the macro definitions; the walker evaluates `#if` over integer macros and keeps one arm | walker + caller input, S; Q7 |
| `asm volatile("ex2.approx.ftz.f32 ...")`, `lg2`, `rcp` | declined | a recognised template table: `ex2.approx` to `Unary(Exp2)` qualified approximate, `lg2` to `Log2`, `rcp` to `Recip`; **each carries an approximation the accurate op does not** (`Numerics.approx`), so the lift must say so | walker, S each; the IR has `Qualified(_, Numerics { approx })` |
| `asm volatile("shfl.sync.bfly.b32 ...")` | declined | `Stmt::Collective` (exists) with the XOR kind | walker, S |
| `asm volatile("cp.async.cg.shared.global ...")`, `commit_group`, `wait_group` | declined | `Stmt::CopyAsync` and the async-group statements (U8, built); the predicated, zero-fill forms are U13 (built) | walker, M |
| `asm volatile("griddepcontrol.wait;")` | declined | `Stmt::GridDepWait` (U17, built) | walker, S |
| `half`, `uint8_t`, `size_t`, `bool*`, `int32_t*` types; `constexpr`; `min`; `static_cast` | partly | add type names; `constexpr uint32_t x = ...;` as an immutable local; `min`/`max` into `Binary(Min/Max)` | walker, S |
| `if (...) return;` early exit, ternaries over loads (`partition_kv ? ... : ...`) | built; a load inside `?:` declines (only one arm runs) | a `MaskedLoad` form for a load in a conditional arm | walker, S |

## Recommendation

**Do not start slice 5 as one piece.** The pointer model (Q6) is the long pole: 8 of the 22 functions take or return pointers, and the params struct holds
eight of them. The order I would build, each an oracle differential with seeded mutants as before:

1. **4c-1** struct locals and local arrays (scalarize; `Private` allocations). No ruling needed.
2. **4c-2** callee inlining for by-value and by-reference scalar and struct parameters. No ruling needed.
3. **4c-3** the `asm volatile` template table for `ex2`, `lg2`, `rcp`, `shfl`, `griddepcontrol`; `cp.async` after it. No ruling needed, but each approximate op's `Numerics` is a numerics statement the PM should see in the options note for it.
4. **After Q6 to Q8 are ruled:** symbolic pointers, the carved dynamic shared memory, `#if` evaluation, and the pointer-holding struct parameter.

## Questions

- **Q6 (PM): symbolic pointers in the walker.** The walker carries `(base, byte offset)` for a pointer while it inlines and emits `Place::Raw`/`AllocRaw` at each dereference, so the IR stays pointer-free. Alternative: ask for pointer values in the IR (U9c option C, which the U9c ruling declined). Recommend the walker model.
- **Q7 (Baracuda): inputs only the caller has.** The dynamic shared size, the macro definitions for `#if`, and the launch shape (block (16,1,8), grid (batch, kv_heads), 36864 B per Baracuda's report). Recommend one `LiftContext` argument carrying all three, replacing today's separate `LaunchShape` argument of `lift_kernel_launched`.
- **Q8 (PM): a struct parameter with buffer-typed fields.** Additive on `ParamKind`/`StructField`; needs its own patent read of the U23 guard (layout only). Alternative: Baracuda flattens the params struct into scalar and buffer parameters, which is lossy for an emitter that wants to pass the struct by address.

— lane: unpopped
