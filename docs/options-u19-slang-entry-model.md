# Options: the Slang entry model in the kernel-body walker (census slice 4)

Status: **options note, no code** (pm-kernels `[TASK]` 2026-11-22). To go to fishstringer, who owns the Slang front end and E17 slice 3
(`docs/options-e17-slice3-abi.md` in its repo), so the two stay on one convention. Source: `docs/options-u19-census.md`: **0 of 153** Slang entry points lift
(fishstringer `8c203f2`; 0 of 151 at Fuel `ded9bed`), **140 of them stopped on one construct**: the entry parameter `uint3 gid : SV_DispatchThreadID`.

## The gap

The walker's kernel model is a function whose **parameters are the buffers** and whose thread ids are **built-ins** (`threadIdx.x`). A Slang compute kernel is the other
shape: **resources are file-scope globals** and the thread ids are **semantics on the entry parameters**:

    [[vk::binding(0, 0)]] StructuredBuffer<float>   input;
    [[vk::binding(1, 0)]] RWStructuredBuffer<float> output;
    [[vk::binding(2, 0)]] ConstantBuffer<Params>    p;
    [shader("compute")] [numthreads(256, 1, 1)]
    void main(uint3 gid : SV_DispatchThreadID) { uint idx = gid.x; if (idx >= p.out_size) return; ... }

Measured over the 153 entry points (text occurrences, not kernels): `numthreads(256,1,1)` 138, `(64,1,1)` 9, `(128,1,1)` 3, `(32,1,1)` 1, `(16,16,1)` 1, and one
`(WG_SIZE,1,1)` naming a constant. Entry signatures: `uint3 gid : SV_DispatchThreadID` (most), `uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID`,
`uint3 lid : SV_GroupThreadID`. Resources: `RWStructuredBuffer<T>` and `StructuredBuffer<T>` under many names (`input`, `output`, `src`, `dst`, `shape_buf`), and one
`ConstantBuffer<Params> p`. **Binding slots are dense from 0 in declaration order in all but one file**: `autodiff_poc.slang` holds two entry points (`silu_forward`, `silu_backward`), each with its own globals (`fwd_*`, `bwd_*`) numbered from 0.

## What the walker would need (all of it maps to IR that exists)

1. **`[numthreads(x,y,z)]` is the block shape.** It gives `LaunchShape.block`; the caller still supplies the grid. A literal or a file-scope constant (`static const uint WG_SIZE = 128;` in `matmul_q4_0_tiled.slang`; the walker's constant folding reads `constexpr` today, `static const` is part of this slice).
2. **Semantic parameters, not buffers.** `SV_DispatchThreadID` is `blockIdx * blockDim + threadIdx` per axis (`ExecId` arithmetic, `U32`), `SV_GroupID` is `blockIdx`,
   `SV_GroupThreadID` is `threadIdx`, `SV_GroupIndex` is the linear thread index in the block. The parameter is a `uint3`, so **3-lane vectors** (the walker scalarizes
   2- and 4-lane ones today) with `.x .y .z`.
3. **The file-scope resources the entry references become `ParamKind::Buffer`**, in **binding-slot order** (references, not every global of the file: `autodiff_poc.slang`'s two entries each
   use one group of globals; unreferenced ones are ignored): `StructuredBuffer<T>` is `Read`, `RWStructuredBuffer<T>` is `ReadWrite` (the walker's
   derived roles then narrow a never-written one to `Read`), `rank: Some(1)`, `name[i]` is `Place::Logical`. Slot order is the parameter order, so `Backend::lower_kernel`'s
   **dense bindings** reproduce the source's. Among the referenced ones, a **gap, a repeat or a non-zero start declines typed**, never guessed.
4. **`ConstantBuffer<Params> p` becomes one `ParamKind::Struct`** named `p` after the buffers, its fields the struct's scalars (`p.out_size`), through the
   struct-parameter machinery that already exists. A non-scalar field is the census's slice 5 (`docs/options-u23b-struct-array-fields.md`).
5. **What sits behind the gate** (static counts in `docs/options-u19-census.md`, all in IR already): `groupshared T a[N]` to a `Shared` allocation (38 kernels' files),
   `GroupMemoryBarrierWithGroupSync()` to `Barrier`, `InterlockedCompareExchange/Or/Xor(dst, ..., orig)` to `AtomicCas`/`AtomicRmw` with the old value assigned to the
   out argument, `WaveActiveSum/Max/Min` to `Collective`, `WaveGetLaneCount()` to `ExecId::SubgroupSize`, `WaveIsFirstLane()` to `LaneId == 0`, `while`/`do` loops
   (12; the walker lifts only counted `for` today, the IR has `While`), a helper function returning a value (60; the walker inlines these).

## Where it lives: the decision for fishstringer

- **A. In the walker, keyed on the CST.** The walker learns the Slang node kinds (attribute, semantics, global resource declaration). Simple to call, but it puts a dialect
  into the neutral walker and makes Unpopped track `tree-sitter-slang`'s grammar.
- **B. In the `Frontend` (recommended).** `Frontend` gains one optional hook, `entry_model(tree, name) -> EntryModel`, implemented by `SLANG` in
  `fishstringer-slang-parse`: `{ block: Option<[u32;3]>, semantics: [(param, Semantic)], resources: [Resource { name, kind, slot }] }`. The walker consumes the model
  (3 to 4 above) and stays neutral; fishstringer keeps the grammar knowledge it already owns. **A new optional field on `Frontend` with a `None` default is additive;
  CUDA's `CUDA` const is unaffected.**

## Consistency with E17 slice 3 (the plan path)

fishstringer's slice 3 proposes, for a plan kernel that needs them: explicit `[[vk::binding(k, 0)]]` on every resource, **buffers first** (inputs, then outputs), **one
`ConstantBuffer<Params_> pb_` last**, `n` first in `Params_`, and a `KernelInterface` with `bindings` and `params_block` (offsets checked against `slangc` reflection).
The lifted corpus already has that order (buffers `0..k`, the constant buffer at the next slot). To keep the two one convention:

- **Slot order = parameter order, buffers before the struct**, both directions (lift reads it, `lower_kernel` writes it).
- **Names differ** (a lifted struct is `p`/`Params`, the emitter's is `pb_`/`Params_`): the round trip renames, **the layout must not move**. `ParamsBlock::of_struct` is the
  natural C layout, equal to Slang's for 4-byte scalars and **not authoritative for the size** (std140 rounding, vec types): `slangc` reflection stays the authority, as already agreed.
- **The `CallerCaps` hook** (`docs/options-caller-caps.md`) is what lets a plan lowering refuse an interface-needing shape for a caller that cannot bind it; a lifted kernel
  re-emitted through `lower_kernel` already carries one.

## Not decided here

Whether `numthreads` overrides or must equal the context's block (recommend: the attribute is the authority, a context block that disagrees declines); entry points that
share a file (`autodiff_poc.slang`: two kernels, each lifted with the resources it references); `ByteAddressBuffer`, `Texture*` and
`Ptr<T>` (absent from the corpus). **The 13 grammar errors are not this walker's**: the parser stops on them (first ERROR node at `float var = ...` in 8 `layer_norm*` kernels, `uint is = ...` in
`dequant_q4_km.slang:110`, `out[...] = ...` in the 4 `rope*` kernels); the cause, and whether to report it, is fishstringer's to establish.

— lane: unpopped
