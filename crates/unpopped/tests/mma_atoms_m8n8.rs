// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The NVIDIA `mma.sync.aligned.m8n8k16` integer atoms (s8, u8, mixed) of ROADMAP U7 theme 3 put every fragment element
//! where the PTX ISA says** and run through the oracle. Layout: `(row, col)` formulas from PTX ISA 9.5 section 9.7.16.5.3
//! (A Figure 59 pp. 539-540, B Figure 60 pp. 540-541, C Figure 61 p. 542). Patent column: US12314717B2, applies 519 B;
//! wrapper read 2026-11-27; not a clearance; no attorney; findings not advice.

use unpopped::mma::MmaAtom;
use unpopped_vocab::ElementKind::{I8, I32, U8};

#[path = "support/mma_run.rs"]
#[allow(dead_code, reason = "shared helpers: each test file uses a subset")]
mod support;
use support::{agrees, places};

/// m8n8k16: A `(gid, tig * 4 + i)`, B `(tig * 4 + i, gid)`, C `(gid, tig * 2 + i)`.
fn k16(which: usize, lane: u32, i: u32) -> (u32, u32) {
    let (gid, tig) = (lane >> 2, lane % 4);
    match which {
        0 => (gid, tig * 4 + i),
        1 => (tig * 4 + i, gid),
        _ => (gid, tig * 2 + i),
    }
}

#[test]
fn the_m8n8k16_integer_atoms_place_and_run() {
    for (atom, a, b) in [
        (MmaAtom::nvidia_m8n8k16_s8_s32(), I8, I8),
        (MmaAtom::nvidia_m8n8k16_u8_s32(), U8, U8),
        (MmaAtom::nvidia_m8n8k16_s8_u8_s32(), I8, U8),
        (MmaAtom::nvidia_m8n8k16_u8_s8_s32(), U8, I8),
    ] {
        places(&atom, k16, (8, 8, 16), (a, b, I32));
        agrees(&atom, k16);
    }
}
