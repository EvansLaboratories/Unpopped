// SPDX-License-Identifier: MIT OR Apache-2.0
//! More NVIDIA `mma.sync.aligned` atoms as data (ROADMAP U7 theme 3). Additive: constructors only, no new
//! statement, flag or element kind.
//!
//! **Source of every layout:** CUTLASS `include/cute/atom/mma_traits_sm80.hpp` and `mma_traits_sm89.hpp` as fetched
//! from `main` (head `5c179edeb3e2abd0dd1da18bd1aa88ee98e4f8ee`, 2026-11-25), `MMA_Traits<...>::{ALayout, BLayout,
//! CLayout}`. A CuTe `BLayout` indexes `(N, K)`; this crate's `B` is `K x N` column-major, so each B stride below is
//! recomputed from the `(n, k)` the CuTe layout names, not copied. **The PTX ISA fragment-table text for these shapes
//! was NOT READ** (the 9.5 page is cut off before section 9.7.16 by the fetch tool): the layouts are UNVERIFIED against
//! the ISA, and the tests state them as `(row, col)` formulas derived from the CuTe layouts.
//!
//! Patent column (`docs/PATENT-SWEEP.md`): the 8-bit atoms are the 4x element-size family of US12314717B2 (method
//! claim): applies 519 B; wrapper read 2026-11-27; not a clearance; no attorney; findings not advice.

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
    /// `mma.sync.aligned.m16n8k8` with f16 inputs and an f32 accumulator (CuTe `SM80_16x8x8_F32F16F16F32_TN`).
    #[must_use]
    pub fn nvidia_m16n8k8_f16_f32() -> Self {
        Self::m16n8k8(ElementKind::F16, "nvidia.mma.sync.m16n8k8.f16.f32")
    }

    /// `mma.sync.aligned.m16n8k8` with bf16 inputs and an f32 accumulator (CuTe `SM80_16x8x8_F32BF16BF16F32_TN`).
    #[must_use]
    pub fn nvidia_m16n8k8_bf16_f32() -> Self {
        Self::m16n8k8(ElementKind::Bf16, "nvidia.mma.sync.m16n8k8.bf16.f32")
    }

    /// CuTe: `ALayout = SM80_16x8_Row`, `BLayout = SM80_8x8_Row`, `CLayout = SM80_16x8_Row`.
    fn m16n8k8(input: ElementKind, name: &str) -> Self {
        let a = c_m16n8();
        let b = flat(&[(4, 2), (8, 8), (2, 1)]);
        build(
            name,
            (16, 8, 8),
            (input, input, ElementKind::F32),
            [&a, &b, &c_m16n8()],
        )
    }

    /// `mma.sync.aligned.m16n8k4` with tf32 inputs and an f32 accumulator (CuTe `SM80_16x8x4_F32TF32TF32F32_TN`).
    /// The operands are `F32` fragments and the atom carries the tf32 rounding flag, as [`Self::nvidia_m16n8k8_tf32_f32`].
    #[must_use]
    pub fn nvidia_m16n8k4_tf32_f32() -> Self {
        let a = flat(&[(4, 16), (8, 1), (2, 8)]);
        let b = flat(&[(4, 1), (8, 4)]);
        build(
            "nvidia.mma.sync.m16n8k4.tf32.f32",
            (16, 8, 4),
            (ElementKind::F32, ElementKind::F32, ElementKind::F32),
            [&a, &b, &c_m16n8()],
        )
        .with_tf32_inputs()
        .expect("an F32 atom takes the tf32 flag")
    }

    /// `mma.sync.aligned.m8n8k4` with f64 inputs and an f64 accumulator (CuTe `SM80_8x8x4_F64F64F64F64_TN`).
    #[must_use]
    pub fn nvidia_m8n8k4_f64_f64() -> Self {
        let ab = flat(&[(4, 1), (8, 4)]);
        let a = flat(&[(4, 8), (8, 1)]);
        let c = flat(&[(4, 16), (8, 1), (2, 8)]);
        build(
            "nvidia.mma.sync.m8n8k4.f64.f64",
            (8, 8, 4),
            (ElementKind::F64, ElementKind::F64, ElementKind::F64),
            [&a, &ab, &c],
        )
    }

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
