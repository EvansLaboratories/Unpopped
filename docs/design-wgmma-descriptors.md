# Design note: shared-memory layout fields for `Stmt::Wgmma` and `Stmt::Tcgen05Mma` (ROADMAP U8c, Baracuda's ask)

Status: **design, not built.** Written 2026-11-25 by the Unpopped lane on pm-kernels' relay of Baracuda's `[ASK]`. The build waits for Baracuda's review of this note. The IR change is **BREAKING** for `unpopped` (new fields on two `Stmt` variants): batch it with the next breaking release (Q14 A). Nothing here is legal advice.

## 1. The ask

`Stmt::Wgmma` (and `Tcgen05Mma`) say "option A: no swizzle and no descriptor bits are modelled, and there was no ISA text to check them against". An emitter cannot build the 64-bit matrix descriptor from them. Per operand it needs a swizzle mode, the leading and stride byte offsets (or a core-matrix layout), the base-offset and alignment contract, and MN-major (transposed) support, with the **oracle reading through that layout**. A plain row-major K-major 16-bit tile is not a layout the no-swizzle descriptor reads. Baracuda will not guess bits: until these fields exist it refuses both statements by name.

## 2. Sources (what was read, and how far to trust it)

PTX ISA **version 9.5**, <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html>, read 2026-11-25 from a downloaded copy of the page (the fetch tool's summariser truncates the page before the `wgmma` sections, so the page was fetched whole with `curl`, its tags stripped by a script, and the sections read as text). Every quotation below was checked against that text unless marked **(figure)**: the `D` fragment layout is a figure image, which was viewed, not read as text. Sections: 9.7.17.5.1.2 (shared memory matrix layout, tables 47 and the canonical-layout table), 9.7.17.5.1.2.1.1/.2/.3 (leading offset, stride offset, canonical layouts), 9.7.17.5.1.2.2 (matrix descriptor), 9.7.17.5.2 (`wgmma.mma_async`, transpose), 9.7.17.5.1.1.1 (D fragments, Figure 152), 9.7.18.1 (tensor memory), 9.7.18.3 and 9.7.18.4.1/.2 (tcgen05 strides, shared-memory descriptor, instruction descriptor), 9.7.18.5 (CTA pairs), 9.7.18.7.1 (`tcgen05.alloc`).

**Not found in the text, so not claimed:** a per-swizzle-mode start-address alignment for `wgmma` beyond "their starting addresses must be aligned to 16 bytes"; a K-major 16-bit worked LBO/SBO example; an MN-major 128B worked example; an explicit "core matrix = rows x bytes" sentence (the text says "swizzle atom", in units of 128-bit elements); the tcgen05 accumulator lane-to-row map for `cta_group::2`; the `mxf8f6f4` instruction descriptors.

## 3. What the ISA says (the facts the fields come from)

**`wgmma` matrix descriptor** (64 bits, 9.7.17.5.1.2.2): bits 13-0 `matrix-descriptor-encode(start address)`; 29-16 `encode(leading dimension byte offset)`; 45-32 `encode(stride dimension byte offset)`; 51-49 "Matrix base offset. This is valid for all swizzling modes except the no-swizzle mode."; 63-62 swizzle ("0: No swizzle 1: 128-Byte swizzle 2: 64-Byte swizzle 3: 32-Byte swizzle"). `matrix-descriptor-encode(x) = (x & 0x3FFFF) >> 4`. Base offset is 0 when the repeating pattern starts on a 1024 (128B), 512 (64B) or 256 (32B) byte boundary; otherwise `(pattern start addr >> 0x7) & 0x7`. "The contents of a matrix descriptor must be same across all the warps in the warpgroup."

**Layouts** (canonical, CuTe notation, `T = 128 / sizeof_bits(element)`, 8 for 16-bit): K-major none `((8,m),(T,2)):((1T,SBO),(1,LBO))`; K-major 32/64/128B `((8,m),(T,2)):((2T|4T|8T,SBO),(1,T))` (LBO "not used, assumed to be 1"); MN-major none `((T,1,m),(8,k)):((1,T,SBO),(1T,LBO))`; MN-major 32/64/128B `((T,2|4|8,m),(8,k)):((1,T,LBO),(2T|4T|8T,SBO))`; each under `Swizzle<0|1|2|3,4,3>`. Stride offset: K-major "the offset from the first 8 rows to the next 8 rows"; MN-major swizzled "offset from the first 8 columns to the next 8 columns". Leading offset: K-major none "the offset from the first column to the second columns of the 8x2 tile"; MN-major swizzled "offset from the first (swizzle-byte-size/16) rows to the next (swizzle-byte-size/16) rows".

**Transpose:** "The transpose operation is only supported for the wgmma.mma_async variants with .f16/ .bf16 types on matrices accessed from shared memory using matrix descriptors"; `imm-trans-a/b` 0 means K-major, 1 means M-major (A) or N-major (B). With `A` in registers there is no transpose operand for `A`.

**`D` fragment** `m64nNk16`, `.f32`: N/2 registers `d0..`; **(figure)** warp `w` holds rows `16w..16w+15`, register `i` of lane `l` is row `16w + l/4 + 8*((i/2)%2)`, column `8*(i/4) + 2*(l%4) + i%2`. This is the formula `oracle/kernel_wgmma.rs::position` already uses, now cross-checked against the figure.

**tcgen05** shared-memory descriptor (9.7.18.4.1) is **not** the same word: 15-bit start address (bits 0-14), 15-bit leading offset (16-30), stride offset (32-45), bits 46-48 the fixed constant `0b001`, base offset 49-51, a leading-stride-mode bit 52 (absolute byte address mode is sm_103a only, 128B swizzle, K-major only, base offset 0), bits 61-63 swizzle with **different codes** ("0 No swizzling, 1 128-Byte with 32B atomic swizzling, 2 128-Byte, 4 64-Byte, 6 32-Byte; 3, 5 and 7 are invalid"). The layouts and the LBO/SBO meanings are the same as `wgmma`'s. Instruction descriptor (32 bits, 9.7.18.4.2, table cells flattened so partly inferred): bits 4-5 D type, 7-9 / 10-12 A / B type, 13/14 negate, 15/16 transpose A/B, 17-22 `N >> 3`, 24-28 `M >> 4`. Tensor memory: 128 lanes x 512 columns of 32 bits per CTA; address = lane in bits 31:16, column in 15:0; allocation unit 32 columns, a power of two, `tcgen05.alloc ... nCols` in [32, 512]; every allocation must be deallocated before exit; `tcgen05.alloc` is illegal after `relinquish_alloc_permit`. A CTA pair is two CTAs whose `%cluster_ctarank` differ in the last bit; all `tcgen05` instructions in a kernel use one `cta_group`.

## 4. The design

**One new value type, shared by both statements** (`crates/unpopped/src/kernel.rs`):

```text
OperandLayout =
    RowMajor                                  // today's meaning: a plain row-major tile, pitch = K (16) elements
  | Canonical { major: Major, swizzle: Swizzle, lbo: Option<u32>, sbo: u32 }
Major   = K | MN                              // imm-trans 0 | 1
Swizzle = None | B32 | B64 | B128             // the PTX layout family; tcgen05's wire codes are the emitter's
```

`Stmt::Wgmma` and `Stmt::Tcgen05Mma` gain `a_layout` and `b_layout: OperandLayout` (`#[serde(default)]` = `RowMajor`, so stored text from before parses). `RowMajor` keeps every current test meaning and is **oracle-only**: an emitter for hardware refuses it with the reason "a plain row-major tile is not a layout the descriptor reads" (this is Baracuda's by-name refusal, now stated in the IR). `Canonical` is what an emitter lowers.

- **Offsets are bytes** (`lbo`, `sbo`), multiples of 16, below `2^18` (the 14-bit field times 16; tcgen05's leading field is 15 bits, so it may exceed `2^18`: the checker takes the stricter bound for `Wgmma` and the wider one for `Tcgen05Mma`). `lbo` is `None` for a swizzled K-major operand (the ISA: "not used, assumed to be 1") and the checker rejects `Some` there; it is required elsewhere.
- **Base offset is not a field.** The checker requires the tile's start (allocation base plus a constant offset) to be a compile-time multiple of the swizzle's pattern boundary (1024 / 512 / 256 bytes; 16 for no swizzle), so the descriptor's base offset is 0. A start that is not provably aligned is `Unsupported`. *Question 1 to Baracuda: do you need a non-zero base offset (a tile that starts mid-pattern)? If so the field is `base_offset: u8`, the checker derives it with the ISA formula, and the oracle reads from the unaligned start.*
- **MN-major** (`Major::MN`) is admitted for `F16`/`Bf16` only, which is all the statements carry today (the ISA restricts transpose to those types). `A` from registers stays out of scope.
- **Extents.** For a 64 x 16 `A` and an `n x 16` `B` of 16-bit elements, K-major canonical needs `m = rows / 8` groups of 8 rows and `(T,2)` = two 16-byte core columns along K; MN-major needs `m = rows / (T * s)` patterns (`s` = 1, 2, 4, 8 for none, 32B, 64B, 128B) and `k = 2`. The checker computes them from the tile and refuses a layout whose offsets would make two elements share a byte or leave the allocation.
- **The oracle reads through the layout.** `kernel_wgmma.rs` replaces "row `t` of `A` is 16 contiguous elements" with `element(row, k) -> byte offset` from the canonical layout above, then the swizzle (`Swizzle<B,4,3>`: XOR of address bits `[4,4+B)` with bits `[7,7+B)` of the byte address; **to confirm against Figures 159-168, which were not read**, because the text gives the notation only). Every read is still an ordinary logged `Shared` load, so the race detector is unchanged.
- **`D` fragment:** unchanged (section 3, figure-checked).

**`Tcgen05Mma` additionally:** `m: 64 | 128` (the ISA lists 64 or 128 for `cta_group::1`; today it is fixed at 128); `cta_group` stays 1 (a pair is declined by name until asked); a new `Tcgen05Alloc { d, ncols }`, `Tcgen05Dealloc { d, ncols }` and `Tcgen05RelinquishPermit` replace "the allocation is static": `ncols` a power of two in 32..=512 equal to the allocation's columns, issued by one warp, every alloc deallocated before the kernel ends, no alloc after the permit is relinquished (checker rules, oracle traps). The descriptor words themselves (15-bit start, the `0b001` constant, the swizzle code table, the instruction descriptor) are **not IR**: they are the emitter's, computed from the fields above and the statement's types; the facts are recorded in section 3 so the emitter has a checked source.

## 5. Patent line

Unchanged and stated here: the descriptor, swizzle, TMEM and CTA-pair limbs were read by `unpopped-patents-2`, claims only, as **outside as read**: US12204897B2 and US12710961B2 (the claims recite a wait-on-MMA limb; `WgmmaWait` stays under the board-453 ruling and the existing patent note, untouched), US20230289398A1 (descriptor appears only in dependent claim 16, claim 1 needs a work-distributor circuit distributing warps; **re-read at grant**), US12645455B2, and US12669996B2 as `CopyBulk` is shipped. Never "clear". These fields describe the published PTX instruction operands; they add no new asynchronous-completion statement and no new warp-role statement, so the standing holds (HOLD 2 to 4 and the warp-role hold) are not touched. If a read later covers a field named here, that is a hold on the field, not a decision.

## 5b. The two fences (Baracuda's review, 2026-11-25; text checked in PTX 9.5)

PTX 9.7.17.7.1: "wgmma.fence instruction establishes an ordering between prior accesses to any warpgroup registers and subsequent accesses to the same registers by a wgmma.mma_async instruction. Only the accumulator register and the input registers containing the fragments of matrix A require this ordering." It "must be issued by all warps of the warpgroup ... Before the first wgmma.mma_async operation in a warpgroup" and "Between a register access by a thread in the warpgroup and any wgmma.mma_async instruction that accesses the same registers ... Otherwise, the behavior is undefined." And: "An async proxy fence must be used to establish an ordering between prior writes to shared memory matrices and subsequent reads of the same matrices in a wgmma.mma_async instruction." (9.7.10.28.2: "Accessing the same memory location across multiple proxies needs a cross-proxy fence. For the async proxy, fence.proxy.async should be used".)

**Decision.** (1) **`wgmma.fence` is not an IR statement**: it orders register accesses only, the oracle lands `D` at the wait and has no register hazard to observe, so it is a hardware synchronisation the emitter derives (before every `Wgmma`, as Baracuda does). (2) **`fence.proxy.async` is an IR statement, `Stmt::ProxyFence { space: Shared }`** (the `.shared::cta` form; `.global` and `.shared::cluster` only when a statement needs them): it is a fact of the source (CUTLASS and CuTe kernels write it), it has oracle-visible meaning because the oracle models shared bytes, and a derived-after-every-store rule costs fences the author did not need. (3) **A missing proxy fence is an oracle trap, `MissingProxyFence`**: the oracle keeps, per `Shared` byte, whether the last write was by a generic-proxy store not yet followed by the writing thread's `ProxyFence`; a `Wgmma`, `Tcgen05Mma` or `CopyBulk` whose source reads such a byte traps. `CopyBulk` completion already carries the ISA's implicit generic-async fence ("The completion of a cp{.reduce}.async.bulk operation is followed by an implicit generic-async proxy fence"), so bytes it wrote are fenced. Cross-thread ordering is still the existing barrier and race rules; the proxy fence adds only the proxy dimension.

## 6. Slices (each red-first, gate alone, U38d absent)

0. **`Stmt::ProxyFence` and the `MissingProxyFence` trap** (small, BREAKING: a new `Stmt`). Tests: a generic `Shared` store then `Wgmma` (or `CopyBulk` source) without the fence traps; with the fence it runs; a store by another thread is not fenced by this thread's fence; bytes written by a completed `CopyBulk` need none.
1. **`OperandLayout`, `Wgmma`, K-major and MN-major, no swizzle.** Tests: the same logical `A`/`B` placed under `RowMajor` and under `Canonical` give byte-identical `D` (a differential, written by laying the tile out by hand); swapping `lbo` and `sbo`, or flipping `Major`, turns it red; checker declines (zero-sized offsets, misaligned offsets, `lbo: Some` where unused, an out-of-allocation extent).
2. **Swizzle 32/64/128B**, K- and MN-major, the XOR confirmed against the figures first (a person or a figure read; this is the one open reading).
3. **`Tcgen05Mma`** takes the same fields, plus `m`.
4. **`Tcgen05Alloc`/`Dealloc`/`RelinquishPermit`** and their oracle traps.
5. **`cta_group::2`**: only on request, with the pair rules (issue from one thread of the pair, the peer must be active).

## 7. Baracuda's answers (2026-11-25, `approved-with-answers`)

Q1 no, base offset stays 0. Q2 fields, not descriptor words (the descriptor words stay the emitter's). Q3 `A` from registers: later, soon after slice 1 (attention P.V needs it). Q4 `m = 64` yes later; `cta_group::2` only on request; both need sm_100. Q5 f16/bf16 first, then `f8f6f4`, `tf32`/`i8` on demand. Swizzled layouts are not lowered until the swizzle unit is confirmed.

## 7a. The questions as asked

1. Non-zero base offset needed (section 4)?
2. Is `Canonical { major, swizzle, lbo, sbo }` the right grain, or do you want the descriptor words (start, lbo, sbo, base offset, swizzle) built by Unpopped and handed over?
3. `A` from registers (`wgmma` descriptor-B form): needed soon, or later?
4. `Tcgen05Mma` with `m = 64`, `cta_group::2`: needed, and from which sm?
5. Which `.kind` values after `f16` (`tf32`, `f8f6f4`, `i8`) does the descriptor work have to carry first, since the instruction descriptor's dtype field is per kind?

— lane: unpopped
