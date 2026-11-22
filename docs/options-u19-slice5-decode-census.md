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

## Rulings and Baracuda's answers (2026-11-21)

**Q6 ruled (pm-kernels): walker-symbolic pointers**, lowered to `Place::Raw`/`Logical`; a pointer that must survive as a value (stored, compared, returned across a call that was not inlined) is `OutsideIr` and files an IR row. **Q8 ruled: not additive IR**; a pointer field in `ParamKind::Struct` is a pointer value by another door, so Baracuda flattens.

**Built since (2026-11-22):** slices 4c-1 (local arrays), 4c-2 (inlining), 4c-3 (the asm table) and **4c-4 (symbolic pointers, Q6)**; **4d (structs) built 2026-11-22**, so the kernel's local `PagedKvHalf` and the scalar parameter struct lift; what remains for slice 5 is member functions (`uint_fastdiv::divmod`, `operator unsigned int`), the `LiftContext` (Q7: dynamic shared size, macro definitions for `#if`), a masked load for a load inside `?:`, `cp.async` copies, and walking the real file to file each decline as a row.

**Baracuda's flatten** (`flashinfer_paged_decode_f16_d128_g1_flat.cu`, baracuda branch `r6-reviews-2`, `e8e06fee7`, pushed, PR pending; bitwise harness re-run on sm89, 30 cases, 0 diffs in `o` and `lse`; mechanical, Apache notice kept). Kernel parameters, in order: 0 `q` `half*`, 1 `q_rope_offset` `i32*`, 2 `o` `half*`, 3 `lse` `f32*`, 4 `maybe_alibi_slopes` `f32*`, 5 `request_indices`, 6 `kv_tile_indices`, 7 `o_indptr`, 8 `kv_chunk_size_ptr` (all `i32*`), 9 `block_valid_mask` `bool*`, 10 `k_data` `half*`, 11 `v_data` `half*`, 12 `indices`, 13 `indptr`, 14 `last_page_len`, 15 `rope_pos_offset` (`i32*`); then **16, one `__grid_constant__ DecodeScalars` struct** of scalars: `padded_batch_size`, `num_qo_heads` (`u32`), `q_stride_n`, `q_stride_h`, `window_left` (`i32`), `logits_soft_cap`, `sm_scale`, `rope_rcp_scale`, `rope_rcp_theta` (`f32`), `partition_kv` (`u32`, was `bool`), `page_size` (`d_`, `m_`, `l_`, three `u32`), then `num_heads`, `head_dim`, `batch_size`, `stride_page`, `stride_n`, `stride_h` (`u32`). Read by the kernel: `sm_scale`, `q_stride_n`, `q_stride_h`, `num_qo_heads`, `partition_kv`, `page_size`, `batch_size`, `stride_page`, `stride_n`, `stride_h`; the rest are layout-only. The body still rebuilds a local `PagedKvHalf` from the parameters. **Slice 5 reads this mapping** (the lift's field-to-buffer record is the parameter order above).

**Q7 values from Baracuda** (a `LiftContext`, to be built): launch block (16,1,8), grid (batch, kv_heads); dynamic shared memory 36864 B (32768 for the kv tiles plus 4096 for offsets); `__CUDACC_VER_MAJOR__` 13; **`__CUDA_ARCH__` 800** (the `cp.async` path; `griddepcontrol` is off at 800 and at the validated 890; 900 turns the `griddepcontrol` asm on and Baracuda did not validate it); `FLASHINFER_CP_ASYNC_ENABLED` is **not** predefined, the file defines it itself when the version is at least 11 and the arch at least 800.

## Census of the real decode kernel (walked 2026-11-22; `LiftContext` built, committed `ff03d210`/`92822c63` on `lane/u19-structs`)

Walked with `LiftContext::new(LaunchShape::new([16,1,8],[4,2,1]))`, dynamic shared 36864, `__CUDACC_VER_MAJOR__` 13, `__CUDA_ARCH__` 800. Declines cleared, in the order met (each has a test in `kernel_body_context.rs`, `kernel_body_structs.rs` or `kernel_body_methods.rs`): `extern __shared__`; pointer fields and brace-list init of `PagedKvHalf`; compute type (an F16-only kernel computes in F32); `bool` initialiser (normalised to a byte via `Cast(CmpNe)`); a buffer pointer as a truth value (a null check, true unless `with_null_buffer`); the short-circuit hoist guard (a constant side decides whether the other is evaluated); a struct pointer member as a base; `min`/`max`; `sizeof(half)`; constant array sizes; a vector load through a cast pointer.

**The decline now (line 284): `((float2*)dst)[i] = __half22float2(((half2*)&tmp)[i]);`.** Three rows, none a one-liner:

1. **`half2`/`float2` vector types and `__half22float2`/`__float22half2_rn`.** Lane-wise, the same shape as `float2`; needs F16 as a vector lane.
2. **Address-of a vector local (`&tmp`, then `(half2*)` of it).** A bit-pun: the `int4` local's four 32-bit lanes read back as eight halves. The scalarized per-lane local cannot express it, so a vector local whose address is taken has to be backed by a Private alloc and addressed with `Place::AllocRaw` (the pointer machinery of 4c-4 already does `AllocRaw` for arrays). Design: decide at declaration, by scanning the enclosing function for `&name`.
3. **`float4` memory copy loops** (`((float4*)ptr)[i] = ((const float4*)src)[i]`) are the vector load and store already built (`vecmem.rs`), to be re-run once rows 1 and 2 clear.

Still ahead after those (not yet reached by the walk): `cp.async` asm whose operands are pointers (`__cvta_generic_to_shared`), a masked load inside `?:`, `size_t` loop counters.

## Result (2026-11-22): the whole kernel lifts and runs on the oracle, **leniently**

`tests/kernel_body_decode_goal.rs` lifts `BatchDecodeWithPagedKVCacheKernel_f16_d128_g1_none` from the verbatim fixture (Baracuda `ce29119567c7`, Baracuda #419) with `LiftContext` block (16,1,8), dynamic shared 36864, `__CUDACC_VER_MAJOR__` 13, `__CUDA_ARCH__` 800, and runs it with `run_launch_lenient` against an independent host reference (softmax(sm_scale q . K^T) V in `f64`, gathered by hand from the same page table over the f16-rounded inputs). **9 tests, 75 s in debug.** Worst error over the cases below: `o` 2.4e-4 (the f16 output rounding), `lse` 2.3e-7; a control with a 1% wrong scale moves `lse` by 7e-3 and is seen.

Cases: 1 request of 5 tokens; 2 requests x 2 heads with shuffled physical pages; 130 tokens (several tiles, a partial last page); page size 1; page edges 16/17/32; **600 tokens, which runs the `kv_offset_smem` refresh**; an **empty request** (`o` exactly 0; its `lse` is upstream's -INF sentinel, asserted to be that, not compared); `lse` null (`o` right, `lse` untouched). **This is a result on those cases against an `f64` reference, not a proof about the kernel.**

**It is a lenient run, never "oracle-clean".** The strict `run_launch` refuses the kernel three ways, each a property of FlashInfer's own code that hardware shrugs off:
1. **`UninitializedShared`**: the predicated no-fill `cp.async` leaves the tokens past a request's length unwritten; `compute_qk` reads them and masks the logit to `-INF` afterwards. Allowed by `Leniency::zero_shared`, and **proven benign here, not assumed**: the same launches with those slots zero and with two different non-zero patterns (seeds 3 and 41, every byte 1 to 58 so no NaN) give bit-identical `o` and `lse`.
2. **`DataRace` in shared**: in `sync_state` all 16 `tx` threads of a row store the same `st.m`/`st.d` to `smem_md`. 3. **`DataRace` in global**: under `if (tz == 0)` all `tx` threads store the same value to `lse[...]`. Both allowed by `Leniency::same_value_writes`, which accepts only a store of the **same bits** the element already holds (typed and raw places, shared and global); a different value, `+0.0` against `-0.0`, and any read racing a store still race (`tests/kernel_lenient.rs`, 11 mutants caught).

The lift also needed the checker to call a load from a `Read` buffer at a uniform index uniform (the early exit `if (block_valid_mask && !block_valid_mask[bx]) return;` and the bounds loaded from `indptr` sit above `__syncthreads`), so the walker **derives buffer roles from use** and the checker applies the refinement to an **entry function only** (`docs/ir-kernel-layer.md`, "Buffer roles and the barrier-uniformity check").

**Found by running it** (neither was visible to a per-construct differential), each now its own regression test: a local alias of a null buffer was not null (`const bool* m = m_ptr;`: the guard read a dummy buffer and the kernel returned early, writing nothing), and an integer `Select` with a comparison condition (the total `%`) failed on the oracle in an `f32` kernel because the comparison is typed as the compute type.

— lane: unpopped
