// SPDX-License-Identifier: MIT OR Apache-2.0
//! The NVIDIA `mma.sync.aligned.m16n8k16` fp8 atoms (e4m3, e5m2, including `A != B` in type) as data (ROADMAP U7 theme 3):
//! fp8 inputs and an f32 accumulator. Additive: constructors only, no new statement, flag or element kind. The f16
//! accumulator form (PTX ISA 8.7) is not built.
//!
//! **Source.** PTX ISA 9.5 section 9.7.16.5.9 (text of the public PDF, outside every repo): the `.e4m3`/`.e5m2` rows share the
//! `.u8`/`.s8` tables, A Figure 87 p. 568 (its column formula continues on p. 569), B Figure 88 p. 570, C Figure 89 p. 571; the `f32.e5m2.e4m3.f32` form
//! is an instruction example in section 9.7.16.5.14. Single source: CUTLASS was not read for this shape.
//!
//! **Patent column** (`docs/PATENT-SWEEP.md`): US12314717B2, the 4x element-size method claim: applies 519 B; wrapper read
//! 2026-11-27; not a clearance; no attorney; findings not advice. Re-read triggers: a new continuation of 17/516,023; any
//! feature that decodes an instruction word with operand-register fields; any change in 519.

use unpopped_vocab::ElementKind::{self, F32, Fp8E4M3FN as E4, Fp8E5M2 as E5};

use crate::exec::Scope;
use crate::mma::{AtomSpec, MmaAtom, c_m16n8, flat};

impl MmaAtom {
    /// `mma.sync.aligned.m16n8k16` with e4m3 inputs and an f32 accumulator.
    #[must_use]
    pub fn nvidia_m16n8k16_e4m3_f32() -> Self {
        Self::m16n8k16_fp8(E4, E4, "e4m3")
    }

    /// `m16n8k16` with e5m2 inputs.
    #[must_use]
    pub fn nvidia_m16n8k16_e5m2_f32() -> Self {
        Self::m16n8k16_fp8(E5, E5, "e5m2")
    }

    /// `m16n8k16` with e4m3 `A` and e5m2 `B`: `A != B` in type.
    #[must_use]
    pub fn nvidia_m16n8k16_e4m3_e5m2_f32() -> Self {
        Self::m16n8k16_fp8(E4, E5, "e4m3e5m2")
    }

    /// `m16n8k16` with e5m2 `A` and e4m3 `B`: `A != B` in type.
    #[must_use]
    pub fn nvidia_m16n8k16_e5m2_e4m3_f32() -> Self {
        Self::m16n8k16_fp8(E5, E4, "e5m2e4m3")
    }

    fn m16n8k16_fp8(a: ElementKind, b: ElementKind, tag: &str) -> Self {
        // the integer m16n8k16 tables: A 16 x 16 (row `gid`, +8 from value 4; four columns from `tig * 4`), B 16 x 8.
        let la = flat(&[(4, 64), (8, 1), (4, 16), (2, 8)]);
        let lb = flat(&[(4, 4), (8, 16), (4, 1)]);
        MmaAtom::new(
            AtomSpec {
                name: &format!("nvidia.mma.sync.m16n8k16.{tag}.f32"),
                mnk: (16, 8, 16),
                dtypes: (a, b, F32),
                scope: Scope::Subgroup,
                lanes: 32,
            },
            [&la, &lb, &c_m16n8()],
        )
        .expect("the PTX fragment layouts are bijections")
    }
}
