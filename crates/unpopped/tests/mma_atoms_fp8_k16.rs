// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The NVIDIA `mma.sync.aligned.m16n8k16` fp8 atoms (e4m3, e5m2, mixed) of ROADMAP U7 theme 3 put every fragment element
//! where the PTX ISA says** and run through the oracle. Layout: `(row, col)` formulas from PTX ISA 9.5 section 9.7.16.5.9
//! (A Figure 87 p. 568, B Figure 88 p. 570, C Figure 89 p. 571). Patent column: US12314717B2, applies 519 B; wrapper read
//! 2026-11-27; not a clearance; no attorney; findings not advice.

use unpopped::mma::MmaAtom;
use unpopped_vocab::ElementKind::{F32, Fp8E4M3FN as E4, Fp8E5M2 as E5};

#[path = "support/mma_run.rs"]
#[allow(dead_code, reason = "shared helpers: each test file uses a subset")]
mod support;
use support::{agrees, places};

/// m16n8k16: A `(gid + 8 * (i / 4), tig * 4 + i % 4)`, B `(tig * 4 + i, gid)`, C `(gid + 8 * (i / 2), tig * 2 + i % 2)`.
fn k16(which: usize, lane: u32, i: u32) -> (u32, u32) {
    let (gid, tig) = (lane >> 2, lane % 4);
    match which {
        0 => (gid + 8 * (i / 4), tig * 4 + i % 4),
        1 => (tig * 4 + i, gid),
        _ => (gid + 8 * (i / 2), tig * 2 + i % 2),
    }
}

#[test]
fn the_fp8_k16_atoms_place_and_run() {
    for (atom, a, b) in [
        (MmaAtom::nvidia_m16n8k16_e4m3_f32(), E4, E4),
        (MmaAtom::nvidia_m16n8k16_e5m2_f32(), E5, E5),
        (MmaAtom::nvidia_m16n8k16_e4m3_e5m2_f32(), E4, E5),
        (MmaAtom::nvidia_m16n8k16_e5m2_e4m3_f32(), E5, E4),
    ] {
        places(&atom, k16, (16, 8, 16), (a, b, F32));
        agrees(&atom, k16);
    }
}
