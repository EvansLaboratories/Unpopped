<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
# U8c emitter migration note: what an emitter (Baracuda, Vulkane, Fuel, cpu-c) must handle

For: `baracuda-cuda-emit`/`-parse`/`-seam`, `vulkane-*`, Fuel's consumers, and our own `unpopped-convert` and `unpopped-cpu-c`.
Scope: the U8c kernel-layer statements. **Released in 0.57.0** (#402) is marked `0.57.0`; **0.58.0** (#403, merged 2026-11-26, not yet published) is marked `0.58.0`. Authority for every behaviour is the doc comment on the variant in `crates/unpopped/src/kernel.rs` and `docs/design-wgmma-descriptors.md`; this note only lists what must be handled. A target that cannot lower a statement **declines it with a typed error** (as `unpopped-cpu-c` does); it must not silently drop it.

## 1. Exhaustive matches that break

| Item | Where | Kind | Status |
|---|---|---|---|
| `Stmt::ProxyFence { space }` | `Stmt` | new variant (only `MemorySpace::Shared` is admitted) | 0.57.0 |
| `KernelTrap::MissingProxyFence { alloc }` | `KernelTrap` | new variant | 0.57.0 |
| `Stmt::Tcgen05Alloc`, `Tcgen05Dealloc`, `Tcgen05RelinquishPermit` | `Stmt` | new variants | 0.58.0 |
| `KernelTrap::TmemMisuse { what }` | `KernelTrap` | new variant | 0.58.0 |
| `Stmt::CopyUnpack { dst, src, groups, bar }` | `Stmt` | new variant | 0.58.0 |
| `KernelError::Tcgen05Unsupported`, `BulkCopyUnsupported`, `WgmmaUnsupported` | `KernelError` | cases the checker gives for the new rules (not new variants in this batch) | n/a |

Fields added to existing statements (a struct literal `Stmt::X { .. }` stops compiling; a `..` pattern does not; stored kernel text without the field still parses, each has `#[serde(default)]`):

| Statement | New field | Default | Status |
|---|---|---|---|
| `Wgmma`, `Tcgen05Mma` | `a_layout`, `b_layout: OperandLayout` | `RowMajor` | 0.57.0 |
| `Wgmma` | `a_regs` (A from registers) | off | 0.58.0 |
| `Tcgen05Mma` | `m` (128 or 64), `d_lane` (0 or 16 for m = 64) | 128, 0 | 0.58.0 |
| `Tcgen05Mma` | `b_ty: Option<ElementKind>` (B's type when it differs from A's) | `None` | 0.58.0 |
| `Tcgen05Mma` | `negate_a`, `negate_b` | false | 0.58.0 |

`acc` of a `Tcgen05Mma` may now be `F16` (0.58.0; see section 3).

## 2. What each statement means for lowering

* **`ProxyFence`** = `fence.proxy.async.shared::cta`. The kernel places it **before the barrier that publishes a `Shared` tile** to the async proxy (`wgmma`, `tcgen05.mma`); a generic store followed by such a read with no fence by the writing thread is the oracle trap `MissingProxyFence`. `wgmma.fence` is **not** a statement: the emitter derives it. A one-proxy target lowers `ProxyFence` as nothing (cpu-c does).
* **`OperandLayout`.** `RowMajor` is **oracle-only**: a plain row-major tile is not a layout the matrix descriptor can read, so an emitter for hardware **refuses** it. `Canonical { major: K | MN, swizzle: None | B32 | B64 | B128, lbo: Option<u32>, sbo: u32 }` is the PTX canonical shared-memory layout; `lbo` and `sbo` are in **bytes** as the descriptor carries them (the field encodes `bytes >> 4`). A swizzled K-major operand has `lbo: None` ("not used, assumed to be 1"). The checker guarantees: offsets nonzero multiples of 16 within the 14-bit (`wgmma`) or 15-bit leading (`tcgen05`) field, a **constant**, swizzle-pattern-aligned tile start (so the descriptor base offset is 0), an injective in-allocation layout. The swizzle is `Swizzle<B,4,3>` on the byte address. An emitter builds the descriptor words from these; it needs no other layout fact. Element size matters: `T = 16 bytes / element size` (8 for 16-bit, 16 for 8-bit; 0.58.0: 8-bit tiles are admitted for `Tcgen05Mma`).
* **`Tcgen05Alloc { d, ncols }` / `Tcgen05Dealloc` / `Tcgen05RelinquishPermit`** (0.58.0) = `tcgen05.alloc.cta_group::1` / `.dealloc` / `.relinquish_alloc_permit`. `d` is a `Tensor` allocation (128 lanes x `ncols` columns of 32-bit cells); the oracle acts on the warp's first lane (`.sync.aligned`: the whole warp issues it). `ncols` is a power of two from 32 to 512 and equals the allocation's declared columns. The address `tcgen05.alloc` writes to its `[dst]` shared word is **not** in the IR: the emitter allocates that shared word and names the tensor allocation by index. Rules the oracle traps as `TmemMisuse`: use before alloc or after dealloc; allocating twice; allocating after `RelinquishPermit`; a column count that increases between two allocations of the CTA; tensor memory never deallocated when the block ends. `cta_group::2` and `.exclusive` are not modelled.
* **`Tcgen05Mma`**: `ab`/`b_ty` are `F16`/`Bf16` (A and B alike, K = 16), or the 8-bit floats `Fp8E4M3FN`/`Fp8E5M2` (A and B independent, K = 32, `.kind::f8f6f4`), or (0.58.0) `F4E2M1` (K = 32, `.kind::f8f6f4`). `m = 64` is Layout F (D row r in tensor-memory lane `32*(r/16) + d_lane + r%16`); `negate_a`/`negate_b` are instruction-descriptor bits 13 and 14. **Instruction-descriptor fields are the emitter's to build:** the `dtype` (D F16 = 0, F32 = 1), atype/btype codes (`E4M3 = 0, E5M2 = 1, E2M3 = 3, E3M2 = 4, E2M1 = 5`), `M`, `N`, negate and major bits are derived from the statement. D `F16` (0.58.0): the tensor allocation is `U32` words and the f16 is in the **low 16 bits** of each; the upper 16 are unspecified (the oracle writes zero) and a kernel must not read them. Rounding of an `F16` accumulate inside the instruction is not specified by PTX; the oracle rounds once from the exact sum.
* **`F4E2M1` descriptor code is kind-dependent.** On `Tcgen05Mma` an `F4E2M1` operand means `.kind::f8f6f4` and the atype/btype code is **5**; the `.kind::mxf4`/`mxf4nvf4` kinds use **1** (PTX Tables 52-53; CUTLASS `MXF8F6F4Format` vs `MXF4Format`) and are not built. An emitter must pick the code from the kind, not from the `ElementKind`.
* **`CopyUnpack`** (0.58.0; patent note as `CopyBulk`: emitters for other manufacturers decline it): copies `groups` x 16 `F4E2M1` elements from a packed global buffer (two per byte, low nibble first) into a `U8` `Shared` allocation as 16-byte slots (8 data bytes + 8 padding), completing `groups * 16` bytes on an `mbarrier`. On NVIDIA this is the tensor-copy type `.b4x16_p64` (PTX 5.5.1.1). The oracle writes zero padding; PTX leaves it uninitialised.
* **`Wgmma` with `a_regs`** (0.58.0): A from registers in the PTX Figure 151 fragment layout (each thread holds the values the checker/oracle `a_position` gives); the descriptor-B form of `wgmma` is the lowering.

## 3. Behaviour changes to know about

* **0.57.0**: a kernel that stores to `Shared` and then runs a `Wgmma`/`Tcgen05Mma` over it without its own `ProxyFence` now traps in the oracle (a conforming kernel fences). Swizzled layouts that slice 1 refused are admitted.
* **0.58.0**: `types::elem_bits(F4E2M1)` is 4 (it panicked), so a kernel may name an `F4E2M1` buffer; `Tcgen05Mma`'s tile checks accept a `U8` allocation for an `F4E2M1` operand (row-major only, oracle-only like all `RowMajor` tiles).

## 4. Not built (an emitter must decline, not guess)

FP6 operands (`F6E2M3`/`F6E3M2`; no KISS buffer packing pin), canonical/swizzled layouts for 4-bit tiles (no source for a 16-byte slot inside a core matrix), block-scaled `mxf8f6f4`/`mxf4`/`nvf4` kinds and `tcgen05.cp`/`st` (scales; board 501 for the NVFP4 scale tokens), `cta_group::2` (design only, note 5f), `.ws`/M = 32 layouts, sparse, `tf32`/`i8` kinds, transposed operands, `tcgen05.mma` with A in tensor memory.

## 5. Per target

* **Baracuda (`baracuda-cuda-emit`/`-parse`/`-seam`)**: rebuild against 0.57.0 (struct literals and the two new enum variants); then lower `ProxyFence`, the layouts (section 2), and, from 0.58.0, the `tcgen05` allocation statements, `m = 64`, 8-bit/4-bit inputs, negate and F16 D. The parse side must read the new PTX forms if it round-trips them.
* **Vulkane / Fuel**: no tensor-core `wgmma`/`tcgen05` on those targets: decline the statements with a typed error naming the statement (as cpu-c does: tcgen05 works on tensor memory, which the serial C does not spell; `CopyBulk`/`CopyUnpack` complete on an `mbarrier`, which a serial target cannot block a thread on). `ProxyFence` may lower to nothing on a single-proxy target.
* **`unpopped-cpu-c`**: already updated in this batch (refuses the tcgen05 and `CopyUnpack` statements; `ProxyFence` is nothing).

— lane: unpopped
