// SPDX-License-Identifier: MIT OR Apache-2.0
//! The 4x element-size NVIDIA `mma.sync.aligned.m16n8k32` atoms as data (ROADMAP U7 theme 3): u8, mixed s8/u8 and the fp8
//! atoms, including pairings with `A != B` in type. Additive: constructors only, no new statement, flag or element kind.
//!
//! **Source.** PTX ISA 9.5 section 9.7.16.5.10 (text of the public PDF, outside every repo): A pp. 572-573 (Figure 91), B
//! pp. 574-576 (Figures 93, 94), C p. 575 (Figure 95); one table serves `.s8`, `.u8`, `.e4m3`, `.e5m2` (the page also
//! lists `.e3m2`, `.e2m3`, `.e2m1`, not built here). `A` and `B` are typed separately. Second source: CUTLASS
//! `mma_traits_sm80.hpp` (`SM80_16x8x32_S32{S8U8,U8S8,U8U8}S32_TN`) and `mma_traits_sm89.hpp`
//! (`SM89_16x8x32_F32E4M3E5M2F32_TN` and the other pairings), `main` 5c179ede, 2026-11-25.
//!
//! **Patent column** (`docs/PATENT-SWEEP.md`): US12314717B2, the 4x element-size method claim: applies 519 B; wrapper read
//! 2026-11-27; not a clearance; no attorney; findings not advice. Re-read triggers: a new continuation of 17/516,023; any
//! feature that decodes an instruction word with operand-register fields; any change in 519.

use unpopped_vocab::ElementKind;

use crate::exec::Scope;
use crate::mma::{AtomSpec, MmaAtom, c_m16n8, flat};
use unpopped_vocab::layout_algebra::Layout;

fn build(
    name: &str,
    mnk: (u32, u32, u32),
    dtypes: (ElementKind, ElementKind, ElementKind),
    layouts: [&Layout; 3],
) -> MmaAtom {
    MmaAtom::new(
        AtomSpec {
            name,
            mnk,
            dtypes,
            scope: Scope::Subgroup,
            lanes: 32,
        },
        layouts,
    )
    .expect("the CuTe fragment layouts are bijections")
}

impl MmaAtom {
    /// `mma.sync.aligned.m16n8k32` with u8 inputs and an s32 accumulator (CuTe `SM80_16x8x32_S32U8U8S32_TN`).
    #[must_use]
    pub fn nvidia_m16n8k32_u8_s32() -> Self {
        Self::m16n8k32(
            ElementKind::U8,
            ElementKind::U8,
            "nvidia.mma.sync.m16n8k32.u8.s32",
        )
    }

    /// `m16n8k32` with signed `A` and unsigned `B` (CuTe `SM80_16x8x32_S32S8U8S32_TN`): `A != B` in type.
    #[must_use]
    pub fn nvidia_m16n8k32_s8_u8_s32() -> Self {
        Self::m16n8k32(
            ElementKind::I8,
            ElementKind::U8,
            "nvidia.mma.sync.m16n8k32.s8u8.s32",
        )
    }

    /// `m16n8k32` with unsigned `A` and signed `B` (CuTe `SM80_16x8x32_S32U8S8S32_TN`): `A != B` in type.
    #[must_use]
    pub fn nvidia_m16n8k32_u8_s8_s32() -> Self {
        Self::m16n8k32(
            ElementKind::U8,
            ElementKind::I8,
            "nvidia.mma.sync.m16n8k32.u8s8.s32",
        )
    }

    /// `mma.sync.aligned.m16n8k32` with e4m3 inputs and an f32 accumulator (CuTe `SM89_16x8x32_F32E4M3E4M3F32_TN`).
    #[must_use]
    pub fn nvidia_m16n8k32_e4m3_f32() -> Self {
        Self::m16n8k32_fp8(ElementKind::Fp8E4M3FN, ElementKind::Fp8E4M3FN, "e4m3")
    }

    /// `m16n8k32` with e5m2 inputs and an f32 accumulator (CuTe `SM89_16x8x32_F32E5M2E5M2F32_TN`).
    #[must_use]
    pub fn nvidia_m16n8k32_e5m2_f32() -> Self {
        Self::m16n8k32_fp8(ElementKind::Fp8E5M2, ElementKind::Fp8E5M2, "e5m2")
    }

    /// `m16n8k32` with e4m3 `A` and e5m2 `B` (CuTe `SM89_16x8x32_F32E4M3E5M2F32_TN`): `A != B` in type.
    #[must_use]
    pub fn nvidia_m16n8k32_e4m3_e5m2_f32() -> Self {
        Self::m16n8k32_fp8(ElementKind::Fp8E4M3FN, ElementKind::Fp8E5M2, "e4m3e5m2")
    }

    /// `m16n8k32` with e5m2 `A` and e4m3 `B` (CuTe `SM89_16x8x32_F32E5M2E4M3F32_TN`): `A != B` in type.
    #[must_use]
    pub fn nvidia_m16n8k32_e5m2_e4m3_f32() -> Self {
        Self::m16n8k32_fp8(ElementKind::Fp8E5M2, ElementKind::Fp8E4M3FN, "e5m2e4m3")
    }

    /// `mma.sync.aligned.m16n8k32` with e3m2 inputs and an f32 accumulator (PTX ISA 8.7; one fragment table serves every
    /// 8-bit-container type, PTX 9.7.16.5.10 Figures 91, 93, 94, 95).
    #[must_use]
    pub fn nvidia_m16n8k32_e3m2_f32() -> Self {
        Self::m16n8k32_fp8(ElementKind::F6E3M2, ElementKind::F6E3M2, "e3m2")
    }

    /// `m16n8k32` with e2m3 inputs and an f32 accumulator.
    #[must_use]
    pub fn nvidia_m16n8k32_e2m3_f32() -> Self {
        Self::m16n8k32_fp8(ElementKind::F6E2M3, ElementKind::F6E2M3, "e2m3")
    }

    /// `m16n8k32` with e2m1 inputs and an f32 accumulator.
    #[must_use]
    pub fn nvidia_m16n8k32_e2m1_f32() -> Self {
        Self::m16n8k32_fp8(ElementKind::F4E2M1, ElementKind::F4E2M1, "e2m1")
    }

    fn m16n8k32_fp8(a: ElementKind, b: ElementKind, tag: &str) -> Self {
        let name = format!("nvidia.mma.sync.m16n8k32.{tag}.f32");
        Self::m16n8k32_with(a, b, ElementKind::F32, &name)
    }

    fn m16n8k32(a: ElementKind, b: ElementKind, name: &str) -> Self {
        Self::m16n8k32_with(a, b, ElementKind::I32, name)
    }

    /// CuTe `ALayout`/`BLayout` of the `S32S8S8S32` atom, shared by every 8-bit `m16n8k32` atom.
    fn m16n8k32_with(a: ElementKind, b: ElementKind, acc: ElementKind, name: &str) -> Self {
        let la = flat(&[(4, 64), (8, 1), (4, 16), (2, 8), (2, 256)]);
        let lb = flat(&[(4, 4), (8, 32), (4, 1), (2, 16)]);
        build(name, (16, 8, 32), (a, b, acc), [&la, &lb, &c_m16n8()])
    }
}
