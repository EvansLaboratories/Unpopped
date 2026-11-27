<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
# MMA atom coverage (ROADMAP U7 theme 3): which PTX `mma` shapes are atoms

State of the dense NVIDIA `mma.sync.aligned` shapes as data (`MmaAtom` constructors). Source: public PTX ISA 9.5, section 9.7.16.1 "Matrix Shape", Table 44, pp. 510-512 (text of the PDF, outside every repo; no text copied). Fragment layouts are read from sections 9.7.16.5.1-.13 and cross-checked against CUTLASS `mma_traits_sm80/sm89/sm90.hpp` (`main` 5c179ede). Page numbers are given where this lane read the section; "section only" means the section was identified but its pages were not recorded. Written 2026-11-27, at `origin/main` a0a406db plus #491. Not legal advice; patent wording for the 8-bit rows is in each PR body (519 B; US12314717B2: applies 519 B; wrapper read 2026-11-27; not a clearance; no attorney; findings not advice).

| Shape | Types (A, B -> C/D) | Status | Constructor / source |
|---|---|---|---|
| m16n8k8 | f16, bf16 -> f32 | done (#473) | `mma_sm80.rs`; 9.7.16.5.7 pp. 555-561 |
| m16n8k16 | f16, bf16 -> f32 | done (earlier) | `mma.rs` |
| m16n8k4 | tf32 -> f32 | done (#473) | `mma_sm80.rs`; 9.7.16.5.6 pp. 549-553 |
| m16n8k8 | tf32 -> f32 | done (earlier) | `mma.rs` |
| m8n8k4 | f64 -> f64 | done (#473) | `mma_sm80.rs`; 9.7.16.5.2 pp. 537-538 |
| m16n8k4, k8, k16 | f64 -> f64 | done (#485) | `mma_f64.rs`; 9.7.16.5.6, .7, .8 (k16: Figures 83, 85, 86 pp. 564-567) |
| m16n8k32 | s8 -> s32 | done (earlier) | `mma.rs` |
| m16n8k32 | u8, s8/u8 mixed -> s32; e4m3, e5m2 incl. A != B -> f32 | done (#484) | `mma_sm80_8bit.rs`; 9.7.16.5.10 pp. 572-576 |
| m16n8k16 | s8, u8, mixed -> s32 | done (#491) | `mma_sm80_8bit.rs`; 9.7.16.5.9 pp. 568-571 |
| m8n8k16 | u8/s8 -> s32 | TODO, unblocked | 9.7.16.5.3 (section only) |
| m8n8k4 | f16 -> f16/f32 (`.row`/`.col` variants) | TODO, unblocked | 9.7.16.5.1 (section only) |
| m16n8k16 | e4m3, e5m2 -> f16/f32 | TODO, single source (PTX only; CUTLASS not read) | 9.7.16.5.9 |
| m8n8k32, m16n8k32, m16n8k64 | u4/s4 -> s32 | TODO, unblocked (`S4`/`U4` exist in the vocab; element kind to confirm) | 9.7.16.5.4, .11 |
| m8n8k128, m16n8k128, m16n8k256 | b1 -> s32 (`.and` / `.xor` + popc) | HELD: a `bit_op: Option<BitOp{And,Xor}>` field on the `non_exhaustive` `MmaAtom` is additive (one atom type); the atoms wait on the bit-op hold (no packed/vector popc without a re-read, US10452398B2) until `patents-2` reads it; design only | 9.7.16.5.5, .12, .13 |
| m16n8k32 | e3m2, e2m3, e2m1 -> f32 | TODO, unblocked: `ElementKind` already has `F6E2M3`, `F6E3M2`, `F4E2M1` (`element.rs`); check each dtype against the atom's accumulate rule | 9.7.16.5.10 |
| m16n8k32, m16n8k64 | block scale, `.ue8m0` / `.ue4m3` | TODO: `ElementKind::F8E8M0` exists and the oracle `MmaAtom` carries block + `ScaleA`/`ScaleB` extras (`kernel_mma.rs`); the F8E8M0 decode (`unpopped-leaves`, commit 26b7800c) is vocab, not a blocker for the atom data; check main before the oracle run | 9.7.16.3 |
| all `mma.sp` shapes | sparse A | HELD (no sparse atoms) | 9.7.16.6 |
| wmma m16n16k16, m8n32k16, m32n8k16 | f16, bf16 | NOT an atom: PTX 9.7.16.4.1 leaves the fragment layout unspecified; modelled as an opaque tile (T2 `TileMma`), not an `MmaAtom` | 9.7.16.4 (p. 518 on) |
| wmma | u8/s8, u4/s4, b1, tf32 | not expressible as a T2 tile today: a tile's element must be F16, BF16, F32 or F64 (`kernel_check/tile.rs`) | 9.7.16.4 |
| Apple `simdgroup_matrix` 8x8 | half, bfloat, float | T2 opaque tile (no lane-layout source: MSL leaves the mapping unspecified) | `docs/design-opaque-tile-mma.md` |
