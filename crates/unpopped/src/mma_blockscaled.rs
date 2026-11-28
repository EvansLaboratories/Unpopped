// SPDX-License-Identifier: MIT OR Apache-2.0
//! The NVIDIA block-scaled `mma.sync.aligned` atoms as data (ROADMAP U7 theme 3): `m16n8k32` with `.kind::mxf8f6f4` (any
//! pairing of e4m3, e5m2, e3m2, e2m3, e2m1, scale `ue8m0`, one scale per row and per 32 `k`) and `m16n8k64` with
//! `.kind::mxf4nvf4` (e2m1, scale `ue8m0` per 16 or 32 `k`, or `ue4m3` per 16 `k`). F32 accumulator. Additive: constructors
//! only, no new statement, flag or element kind (`MmaAtom::with_block_scale` carries the scale operands).
//!
//! **Source: single (CUTLASS), the PTX figures are unread.** The scale fragments are given in PTX ISA 9.5 section 9.7.16.3
//! (pp. 514-517) only through the selectors `byte-id`/`thread-id` (Table 47) and Figures 46-48, which are images with no
//! text in the extracted PDF, so the lane layout of `SF_A` and `SF_B` is NOT confirmed against the PTX figures (a person
//! should read printed pages 516-517 of the PDF). The text does say that `thread-id-a = 0` makes the lanes with `%laneid % 4`
//! of 0 or 1 supply `SF_A`, and that the lanes with `%laneid % 4 == thread-id-b` supply `SF_B`. The layouts below are CUTLASS
//! `include/cute/atom/mma_traits_sm120.hpp` (BSD-3-Clause, Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES,
//! notice retained at the file's head; read through the GitHub API at `NVIDIA/cutlass` commit
//! `5c179edeb3e2abd0dd1da18bd1aa88ee98e4f8ee`, file sha256 `3901ad2ee516a3da1cfe731e6ecc6be04ba3bc1c7f112dd57145667ecf5c835c`,
//! nothing cloned or run):
//! * `SM120_16x8x32_TN_VS` (lines 171-200): `ALayout`/`BLayout`/`CLayout` are those of `SM120_16x8x32_TN` (the 8-bit
//!   `m16n8k32` tables, lines 121-131, here `m16n8k32_with` in `mma_sm80_8bit.rs`); `SFALayout` is
//!   `((2,2,8),32):((8,0,1),16)` and `SFBLayout` is `((4,8),32):((0,1),8)`, in `(M, K)` and `(N, K)` coordinates, with the
//!   comment "Effectively 16 threads due to the 2:0 mode" for `SFA` and "Effectively 8 threads due to the 4:0 mode" for `SFB`.
//! * `SM120_16x8x64_TN_VS` (lines 136-166): `ALayout` `((4,8),(8,2,2)):((128,1),(16,8,512))`, `BLayout`
//!   `((4,8),(8,2)):((64,1),(8,256))`, `SFALayout` `((2,2,8),64):((8,0,1),16)`, `SFBLayout` `((4,8),64):((0,1),8)`.
//! * `cute/arch/mma_sm120.hpp` (sha256 `4848512808aa9c1c461445ff4b35f5a02f25ce734f81a5994176e0a46e57c37a`): the k32 form is
//!   `scale_vec::1X`, `VS == 32`, one-byte `SFA`/`SFB` registers, selectors all 0; the k64 `ue8m0` forms are `scale_vec::2X`
//!   (`VS` 32) and `4X` (`VS` 16), the `ue4m3` form is `4X`, selectors all 0.
//!
//! The CUTLASS scale layouts are over `k`, one entry per `k`; the atom's scale operands are `M x K/block` and `K/block x N`,
//! so the scale of row `m` and block `kb` is held by the lanes `gid = m % 8`, `lane % 2 = m / 8` (both lanes of the pair
//! `lane / 2 % 2`), and the scale of column `n` by the lanes `gid = n`, every `lane % 4`. Whether the hardware's
//! `byte-id`/`thread-id` selectors other than 0 move these is not modelled (selectors fixed at 0, as CUTLASS uses them).
//!
//! **Patent column** (`docs/PATENT-SWEEP.md`): US12314717B2, the element-size method claim (fp8/fp6/fp4 inputs): applies 519
//! B; wrapper read 2026-11-27; not a clearance; no attorney; findings not advice. Re-read triggers: a new continuation of
//! 17/516,023; any feature that decodes an instruction word with operand-register fields; any change in 519.

use unpopped_vocab::ElementKind::{
    self, F4E2M1, F6E2M3, F6E3M2, F8E8M0, F8UE4M3, F32, Fp8E4M3FN, Fp8E5M2,
};
use unpopped_vocab::layout_algebra::Layout;

use crate::exec::Scope;
use crate::mma::{AtomSpec, MmaAtom, c_m16n8, flat};

fn tag(t: ElementKind) -> &'static str {
    match t {
        Fp8E4M3FN => "e4m3",
        Fp8E5M2 => "e5m2",
        F6E3M2 => "e3m2",
        F6E2M3 => "e2m3",
        _ => "e2m1",
    }
}

fn base(
    name: &str,
    mnk: (u32, u32, u32),
    ab: (ElementKind, ElementKind),
    layouts: [&Layout; 3],
) -> MmaAtom {
    MmaAtom::new(
        AtomSpec {
            name,
            mnk,
            dtypes: (ab.0, ab.1, F32),
            scope: Scope::Subgroup,
            lanes: 32,
        },
        layouts,
    )
    .expect("the fragment layouts are bijections")
}

impl MmaAtom {
    /// `mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32` with `ue8m0` scales (one per row and per 32
    /// `k`), f32 accumulator; `a` and `b` are each e4m3, e5m2, e3m2, e2m3 or e2m1.
    ///
    /// # Panics
    ///
    /// If `a` or `b` is not one of those five element kinds.
    #[must_use]
    pub fn nvidia_m16n8k32_mxf8f6f4_f32(a: ElementKind, b: ElementKind) -> Self {
        let ok = |t| matches!(t, Fp8E4M3FN | Fp8E5M2 | F6E3M2 | F6E2M3 | F4E2M1);
        assert!(
            ok(a) && ok(b),
            "mxf8f6f4 inputs are e4m3, e5m2, e3m2, e2m3 or e2m1"
        );
        // the 8-bit m16n8k32 data tables (one 8-bit container per element)
        let la = flat(&[(4, 64), (8, 1), (4, 16), (2, 8), (2, 256)]);
        let lb = flat(&[(4, 4), (8, 32), (4, 1), (2, 16)]);
        let name = format!(
            "nvidia.mma.sync.m16n8k32.mxf8f6f4.{}{}.ue8m0.f32",
            tag(a),
            tag(b)
        );
        // SFA is 16 x 1: row `gid + 8 * (lane % 2)`; SFB is 1 x 8: column `gid`
        let sfa = flat(&[(2, 8), (2, 0), (8, 1)]);
        let sfb = flat(&[(4, 0), (8, 1)]);
        base(&name, (16, 8, 32), (a, b), [&la, &lb, &c_m16n8()])
            .with_block_scale(32, F8E8M0, (&sfa, &sfb))
            .expect("the scale layouts cover their matrices")
    }

    /// `mma.sync.aligned.kind::mxf4nvf4.block_scale.m16n8k64` with e2m1 inputs, f32 accumulator, and the scale element
    /// `sf` per `block` `k`: `(F8E8M0, 32)` is `scale_vec::2X`, `(F8E8M0, 16)` and `(F8UE4M3, 16)` are `scale_vec::4X`.
    ///
    /// # Panics
    ///
    /// If `(sf, block)` is not one of those three pairs.
    #[must_use]
    pub fn nvidia_m16n8k64_mxf4nvf4_f32(sf: ElementKind, block: u32) -> Self {
        assert!(
            matches!((sf, block), (F8E8M0, 16 | 32) | (F8UE4M3, 16)),
            "mxf4nvf4: ue8m0 per 16 or 32, or ue4m3 per 16"
        );
        let la = flat(&[(4, 128), (8, 1), (8, 16), (2, 8), (2, 512)]);
        let lb = flat(&[(4, 8), (8, 64), (8, 1), (2, 32)]);
        let kb = 64 / block; // scale blocks along k
        // SFA is 16 x kb (id = m + 16 * kb_index), SFB is kb x 8 (id = kb_index + kb * n)
        let sfa = flat(&[(2, 8), (2, 0), (8, 1), (kb.into(), 16)]);
        let sfb = flat(&[(4, 0), (8, i64::from(kb)), (kb.into(), 1)]);
        let name = format!(
            "nvidia.mma.sync.m16n8k64.mxf4nvf4.e2m1.{}x{block}.f32",
            if sf == F8E8M0 { "ue8m0" } else { "ue4m3" }
        );
        base(&name, (16, 8, 64), (F4E2M1, F4E2M1), [&la, &lb, &c_m16n8()])
            .with_block_scale(block, sf, (&sfa, &sfb))
            .expect("the scale layouts cover their matrices")
    }
}
