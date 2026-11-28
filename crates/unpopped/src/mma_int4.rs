// SPDX-License-Identifier: MIT OR Apache-2.0
//! The NVIDIA `mma.sync.aligned` 4-bit integer atoms (s4, u4, mixed) `m8n8k32`, `m16n8k32` and `m16n8k64` as data (ROADMAP U7
//! theme 3): 4-bit inputs and an s32 accumulator. Additive: constructors only, no new statement, flag or element kind.
//!
//! **Source.** PTX ISA 9.5 (text of the public PDF, outside every repo): `m8n8k32` section 9.7.16.5.4 (A Figure 62 p. 543, B
//! Figure 63 pp. 543-544, C Figure 64 p. 545); `m16n8k32` section 9.7.16.5.10 (A Figure 90 pp. 571-572, B Figure 92 pp.
//! 573-574, C Figure 95 p. 575); `m16n8k64` section 9.7.16.5.11 (A Figure 96 pp. 578-579, B Figures 97, 98 pp. 579-581, C
//! Figure 99 p. 582). Shapes in section 9.7.16.1 Table 44 pp. 510-511. Single source: CUTLASS was not read for these shapes.
//! Each of `A` and `B` is `.s4` or `.u4` independently, so every shape has the four signedness pairings.
//!
//! **Patent column** (`docs/PATENT-SWEEP.md`): US12314717B2, the element-size method claim, applied to the narrow integer
//! atoms as to the 8-bit ones: applies 519 B; wrapper read 2026-11-27; not a clearance; no attorney; findings not advice.
//! Re-read triggers: a new continuation of 17/516,023; any feature that decodes an instruction word with operand-register
//! fields; any change in 519.

use unpopped_vocab::ElementKind::{self, I4, I32, U4};
use unpopped_vocab::layout_algebra::Layout;

use crate::exec::Scope;
use crate::mma::{AtomSpec, MmaAtom, c_m16n8, flat};

/// The signedness tag of a 4-bit element in an atom name.
fn tag(t: ElementKind) -> &'static str {
    if t == I4 { "s4" } else { "u4" }
}

fn build(
    shape: &str,
    mnk: (u32, u32, u32),
    ab: (ElementKind, ElementKind),
    layouts: [&Layout; 3],
) -> MmaAtom {
    let name = format!("nvidia.mma.sync.{shape}.{}{}.s32", tag(ab.0), tag(ab.1));
    MmaAtom::new(
        AtomSpec {
            name: &name,
            mnk,
            dtypes: (ab.0, ab.1, I32),
            scope: Scope::Subgroup,
            lanes: 32,
        },
        layouts,
    )
    .expect("the PTX fragment layouts are bijections")
}

impl MmaAtom {
    /// `mma.sync.aligned.m8n8k32` with 4-bit inputs (`a`, `b` each `I4` or `U4`) and an s32 accumulator.
    ///
    /// # Panics
    ///
    /// If `a` or `b` is neither `I4` nor `U4`.
    #[must_use]
    pub fn nvidia_m8n8k32_int4_s32(a: ElementKind, b: ElementKind) -> Self {
        assert!(
            matches!(a, I4 | U4) && matches!(b, I4 | U4),
            "4-bit integers only"
        );
        // A 8 x 32: row `gid`, eight columns from `tig * 8`; B 32 x 8; C 8 x 8.
        let la = flat(&[(4, 64), (8, 1), (8, 8)]);
        let lb = flat(&[(4, 8), (8, 32), (8, 1)]);
        let lc = flat(&[(4, 16), (8, 1), (2, 8)]);
        build("m8n8k32", (8, 8, 32), (a, b), [&la, &lb, &lc])
    }

    /// `mma.sync.aligned.m16n8k32` with 4-bit inputs and an s32 accumulator.
    ///
    /// # Panics
    ///
    /// If `a` or `b` is neither `I4` nor `U4`.
    #[must_use]
    pub fn nvidia_m16n8k32_int4_s32(a: ElementKind, b: ElementKind) -> Self {
        assert!(
            matches!(a, I4 | U4) && matches!(b, I4 | U4),
            "4-bit integers only"
        );
        // A 16 x 32: row `gid` (+8 from value 8), eight columns from `tig * 8`.
        let la = flat(&[(4, 128), (8, 1), (8, 16), (2, 8)]);
        let lb = flat(&[(4, 8), (8, 32), (8, 1)]);
        build("m16n8k32", (16, 8, 32), (a, b), [&la, &lb, &c_m16n8()])
    }

    /// `mma.sync.aligned.m16n8k64` with 4-bit inputs and an s32 accumulator.
    ///
    /// # Panics
    ///
    /// If `a` or `b` is neither `I4` nor `U4`.
    #[must_use]
    pub fn nvidia_m16n8k64_int4_s32(a: ElementKind, b: ElementKind) -> Self {
        assert!(
            matches!(a, I4 | U4) && matches!(b, I4 | U4),
            "4-bit integers only"
        );
        // A 16 x 64: 32 values; bit 3 of the value index is row + 8, bit 4 is column + 32.
        let la = flat(&[(4, 128), (8, 1), (8, 16), (2, 8), (2, 512)]);
        // B 64 x 16: 16 values; bit 3 of the value index is row + 32.
        let lb = flat(&[(4, 8), (8, 64), (8, 1), (2, 32)]);
        build("m16n8k64", (16, 8, 64), (a, b), [&la, &lb, &c_m16n8()])
    }
}
