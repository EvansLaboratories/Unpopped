// SPDX-License-Identifier: MIT OR Apache-2.0
//! The NVIDIA `mma.sync.aligned.m8n8k16` integer atoms (s8, u8, mixed) as data (ROADMAP U7 theme 3): 8-bit inputs and an s32
//! accumulator. Additive: constructors only, no new statement, flag or element kind.
//!
//! **Source.** PTX ISA 9.5 section 9.7.16.5.3 (text of the public PDF, outside every repo): A Figure 59 pp. 539-540, B
//! Figure 60 pp. 540-541, C Figure 61 p. 542. Shape `.m8n8k16` for `.u8`/`.s8` is in PTX ISA 6.5 (section 9.7.16.1 Table
//! 44, p. 510). Single source: CUTLASS was not read for this shape.
//!
//! **Patent column** (`docs/PATENT-SWEEP.md`): US12314717B2, the 4x element-size method claim: applies 519 B; wrapper read
//! 2026-11-27; not a clearance; no attorney; findings not advice. Re-read triggers: a new continuation of 17/516,023; any
//! feature that decodes an instruction word with operand-register fields; any change in 519.

use unpopped_vocab::ElementKind;
use unpopped_vocab::layout_algebra::Layout;

use crate::exec::Scope;
use crate::mma::{AtomSpec, MmaAtom, flat};

impl MmaAtom {
    /// `mma.sync.aligned.m8n8k16` with s8 inputs and an s32 accumulator.
    #[must_use]
    pub fn nvidia_m8n8k16_s8_s32() -> Self {
        Self::m8n8k16(ElementKind::I8, ElementKind::I8, "s8")
    }

    /// `m8n8k16` with u8 inputs.
    #[must_use]
    pub fn nvidia_m8n8k16_u8_s32() -> Self {
        Self::m8n8k16(ElementKind::U8, ElementKind::U8, "u8")
    }

    /// `m8n8k16` with signed `A` and unsigned `B`: `A != B` in type.
    #[must_use]
    pub fn nvidia_m8n8k16_s8_u8_s32() -> Self {
        Self::m8n8k16(ElementKind::I8, ElementKind::U8, "s8u8")
    }

    /// `m8n8k16` with unsigned `A` and signed `B`: `A != B` in type.
    #[must_use]
    pub fn nvidia_m8n8k16_u8_s8_s32() -> Self {
        Self::m8n8k16(ElementKind::U8, ElementKind::I8, "u8s8")
    }

    fn m8n8k16(a: ElementKind, b: ElementKind, tag: &str) -> Self {
        // A is 8 x 16 (row `gid`, four columns from `tig * 4`), B is 16 x 8, C is 8 x 8 (two columns from `tig * 2`).
        let la = flat(&[(4, 32), (8, 1), (4, 8)]);
        let lb = flat(&[(4, 4), (8, 16), (4, 1)]);
        let lc = flat(&[(4, 16), (8, 1), (2, 8)]);
        build(
            &format!("nvidia.mma.sync.m8n8k16.{tag}.s32"),
            (8, 8, 16),
            (a, b, ElementKind::I32),
            [&la, &lb, &lc],
        )
    }
}

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
    .expect("the PTX fragment layouts are bijections")
}
