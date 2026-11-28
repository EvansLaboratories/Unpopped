# Design note: oracle storage for `F6E2M3` and `F6E3M2` (no code)

Status: **draft for `unpopped` (owner of the files it touches).** The build follows in its own PR. Sources: PTX ISA 9.5 section 5.2.3 (p. 43) and the `mma.m16n8k32` fragment tables (Figures 91, 93, 94, 95, pp. 573-577), text read, figures cited by number; single source. KISS-CLASSIFY section 6.1 through `crates/unpopped-vocab/kiss/dtype_manifest.json` (`f6e2m3`, `f6e3m2`: `storage_bits` 6). Patent column: applies 519 B (`docs/PATENT-SWEEP.md`); this note enables atoms that carry the US12314717B2 wording (wrapper read 2026-11-27; not a clearance; no attorney; findings not advice).

## 1. The gap

`check_kernel` declines a kernel that declares an F6 dtype (`DtypeNotComputed`, `kernel_check.rs::check_declared_dtypes`, #501) because the oracle panicked on a `Private` allocation of one: `types.rs::elem_size` has no arm for them (`oracle: unsupported dtype F6E2M3 (v1)`). The codec already exists (`cast_ref::decode_mx(bits, 2, 3)` / `(3, 2)`, `encode_mx`; `oracle/value.rs::raw_to_f64` decodes them; a `Cast` into one saturates with `ClampFinite`). What is missing is only the **container**: how many bytes an F6 element occupies in an oracle buffer or `Private` array. That is why the `e3m2` and `e2m3` `mma.sync` atoms (`m16n8k32`) are placement-only.

## 2. The choice

**A. An 8-bit container, one element per byte, the six value bits in the low bits and the two high bits zero (recommended).** PTX holds an F6 element exactly so: a register of two `e2m3` or `e3m2` values is a `.b16` where "each `.b8` element has [a] 6-bit floating point value and 2 MSB bits padded with zeros" (s5.2.3, p. 43), and the `mma.m16n8k32` tables give `e3m2` and `e2m3` the same fragment as `.s8` and `.e4m3` (Figures 91, 93, 94, 95), so a fragment of 16 elements per lane is 16 bytes. The oracle's `Private` fragment of an F6 element is therefore the PTX register content, byte for byte.

**B. Packed 6-bit (four elements in three bytes).** This is the transfer format of the bulk copies (`.b6x16_p32`, `.b6p2x16`, Figures 6 and 7, pp. 56-58), where the padding is added or discarded in flight: a property of a copy, not of a register or an `mma` fragment. It needs an unaligned 6-bit read-modify-write, the same objection ruled in ROADMAP U9b for 4-bit elements outside `Private`, and nothing the atoms need. Out of scope here; a bulk copy that packs 6-bit data is its own row.

KISS keeps `storage_bits` 6 as the **dtype's identity** (the structure key does not change); the oracle's container is an execution detail, as `elem_size` is already "the CONTAINER width for a sub-byte dtype" (`types.rs`).

## 3. What changes under A

1. `types.rs::elem_size`: `F6E2M3 | F6E3M2 => 1`. `elem_bits` (`elem_size * 8`, so 8) is left alone: F6 is not packed, and `packed_bytes` is `n`. `is_sub_byte` stays false for F6, so none of the sub-byte packing or the sub-byte `Store` rule (`is_subbyte_int`: `I4`, `U4`) applies.
2. Read: the oracle already reads the raw byte and `decode_mx` masks to the six value bits; a set bit 6 or 7 is ignored on read and written as zero by every store (tested both ways).
3. `kernel_check.rs::check_declared_dtypes`: the two F6 dtypes leave the declined set, which becomes exactly `F128`, `F8E3M4`, `F8UE4M3` again; `kernel.rs` and `types.rs` doc text that names them follows.
4. Tests: `tests/union_batch1_dtypes.rs` pin test returns to the three-dtype set and the #501 probe becomes the positive test: a `Private` F6 allocation, a `ClampFinite` store, a load, which must **run and round-trip** over every one of the 64 codes of each format against `decode_mx`/`encode_mx`, with `F4E2M1` as the control. Red first: the probe fails (panics) on current `main`.
5. The atoms: `nvidia_m16n8k32_{e3m2,e2m3}_f32` run in the oracle through the shared harness (a host encoder for the 64-code lattice, written from the OCP MX definition in the test, not from `cast_ref`), the block-scaled `mxf8f6f4` atom gains its F6 pairings, and `tests/mma_atoms_8bit.rs` stops saying they cannot run.

## 4. Not changed

No new `Stmt`, `ScalarExpr` or `ElementKind`; no packed or vector popc statement; the plan gate and every emitter keep refusing F6 as a compute dtype (`plan/dtype_float.rs`, `contract.rs`); the 519 B and US12314717B2 wording stays on the atoms this enables, none is added here. The cpu-c emitter's `F6` refusal is untouched.

## 5. Open for `unpopped`

- Accept A (8-bit container) as the oracle's F6 storage?
- Should a set padding bit on read be a trap rather than ignored? I propose ignore (the bits are padding by the PTX text) and zero on every write.

— lane: unpopped-mma
