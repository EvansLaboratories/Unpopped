// SPDX-License-Identifier: MIT OR Apache-2.0
//! The Volta `mma.sync.aligned.m8n8k4` atoms with `.f16` inputs as data (ROADMAP U70, theme 3): **one warp computes four
//! independent `8x8x4` MMAs**, so each atom is an 8-lane MMA with [`groups`](MmaAtom::groups) `= 4`, the lane map
//! sending local lane `lo + 4 * hi` of group `g` to subgroup lane `lo + 4 * g + 16 * hi` (a group is two quads of
//! lanes: `0-3` with `16-19`, `4-7` with `20-23`, and so on). `.row`/`.col` per operand and an `.f16` or `.f32`
//! accumulator. Additive: constructors only.
//!
//! **Source.** PTX ISA 9.5 section 9.7.16.5.1 (text of the public PDF, outside every repo), single source: A Figures 49
//! (row) and 50 (col), B Figures 51 (row) and 52 (col), C Figure 53 (`.f16`) and Figures 54-55 (`.f32`), pp. 531-536.
//! Formulas read from the page text; the figures are cited by number (Figures 54 and 55 only split the drawing of the four
//! computations; the formula does not depend on the group). `docs/design-mma-groups.md` has the design.
//!
//! **Patent column** (`docs/PATENT-SWEEP.md`): US12314717B2: applies 519 B; wrapper read 2026-11-27; not a clearance; no
//! attorney; findings not advice. 519 B: lifted by CireSnave 2026-11-26 (board 519 B) as his risk call: the PM did not
//! support it with a reading; not a clearance; no attorney; the Intel/NVIDIA tile-instruction method and medium claims
//! (US11086623B2 = 16/487,787, US11288069B2 = 16/487,755, US10884734B2 = 16/459,191) remain NEEDS MORE as read.

use unpopped_vocab::ElementKind::{self, F16, F32};

use crate::exec::Scope;
use crate::mma::{AtomSpec, Major, MmaAtom, flat};

impl MmaAtom {
    /// `mma.sync.aligned.m8n8k4.{a}.{b}.{acc}.f16.f16.{acc}` (PTX ISA 9.5 s9.7.16.5.1): four independent `8x8x4` MMAs per
    /// warp (`groups = 4`). `acc` is `F16` or `F32`.
    ///
    /// # Panics
    /// If `acc` is neither `F16` nor `F32`.
    #[must_use]
    pub fn nvidia_m8n8k4_f16(a: Major, b: Major, acc: ElementKind) -> Self {
        assert!(matches!(acc, F16 | F32), "the accumulator is f16 or f32");
        // local lane l = lo + 4 * hi (lo = laneid % 4, hi = laneid >= 16); A is 8 x 4 (id = row + 8 col), B 4 x 8 (id = row + 4 col)
        let la = match a {
            // row = l, col = i
            Major::Row => flat(&[(8, 1), (4, 8)]),
            // row = i + 4 hi, col = lo
            Major::Col => flat(&[(4, 8), (2, 4), (4, 1)]),
        };
        let lb = match b {
            // row = lo, col = i + 4 hi
            Major::Row => flat(&[(4, 1), (2, 16), (4, 4)]),
            // row = i, col = lo + 4 hi
            Major::Col => flat(&[(4, 4), (2, 16), (4, 1)]),
        };
        let lc = if acc == F16 {
            // row = l, col = i
            flat(&[(8, 1), (8, 8)])
        } else {
            // row = (lane & 1) + (i & 2) + 4 hi, col = (i & 4) + (lane & 2) + (i & 1)
            flat(&[(2, 1), (2, 16), (2, 4), (2, 8), (2, 2), (2, 32)])
        };
        let name = format!(
            "nvidia.mma.sync.m8n8k4.{}.{}.f16.{}",
            major(a),
            major(b),
            if acc == F16 { "f16" } else { "f32" }
        );
        MmaAtom::new(
            AtomSpec {
                name: &name,
                mnk: (8, 8, 4),
                dtypes: (F16, F16, acc),
                scope: Scope::Subgroup,
                lanes: 8,
            },
            [&la, &lb, &lc],
        )
        .and_then(|atom| atom.with_groups(4, &flat(&[(4, 1), (2, 16), (4, 4)])))
        .expect("the PTX m8n8k4 fragment layouts are bijections and the lane map is one")
    }
}

const fn major(m: Major) -> &'static str {
    match m {
        Major::Row => "row",
        Major::Col => "col",
    }
}
