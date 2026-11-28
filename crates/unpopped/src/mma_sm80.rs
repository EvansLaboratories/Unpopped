// SPDX-License-Identifier: MIT OR Apache-2.0
//! More NVIDIA `mma.sync.aligned` atoms as data (ROADMAP U7 theme 3), the 2x and 1x element-size ones: `m16n8k8` f16 and
//! bf16, `m16n8k4` tf32, `m8n8k4` f64. Additive: constructors only, no new statement, flag or element kind.
//!
//! **Sources.** Each fragment layout is read from the public PTX ISA 9.5 (text of the PDF, outside every repo): `m16n8k8`
//! f16/bf16 section 9.7.16.5.7 (pp. 555-556 A, 558-559 B, 560-561 C; Figures 74, 77, 80), `m16n8k4` tf32 section
//! 9.7.16.5.6 (pp. 549-553; Figures 68, 70, 72), `m8n8k4` f64 section 9.7.16.5.2 (pp. 537-538; Figures 56-58). A second,
//! independent source is CUTLASS `include/cute/atom/mma_traits_sm80.hpp` (`main` head 5c179ede, 2026-11-25): a CuTe
//! `BLayout` indexes `(N, K)` and this crate's `B` is `K x N` column-major, so B strides are recomputed. The tests state
//! the layouts as `(row, col)` formulas from the PTX text. No patent column applies beyond the standing note on the
//! 4x element-size atoms, which are not in this file.

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
}
