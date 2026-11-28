// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The NVIDIA `mma.sync.aligned` 4-bit integer atoms (`m8n8k32`, `m16n8k32`, `m16n8k64`; s4, u4, mixed) of ROADMAP U7
//! theme 3 put every fragment element where the PTX ISA says** (placed, and run in the oracle against a plain triple loop). Layouts: `(row, col)` formulas
//! from PTX ISA 9.5 sections 9.7.16.5.4 (Figures 62-64, pp. 543-545), 9.7.16.5.10 (Figures 90, 92, 95, pp. 571-575) and
//! 9.7.16.5.11 (Figures 96-99, pp. 578-582). Patent column: US12314717B2, applies 519 B; wrapper read 2026-11-27; not a
//! clearance; no attorney; findings not advice.

use unpopped::kernel::check_kernel;
use unpopped::mma::MmaAtom;
use unpopped_vocab::ElementKind::{I4, I32, U4};

#[path = "support/mma_run.rs"]
#[allow(dead_code, reason = "shared helpers: each test file uses a subset")]
mod support;
use support::{agrees, mma_kernel, places};

/// m8n8k32: A `(gid, tig * 8 + i)`, B `(tig * 8 + i, gid)`, C `(gid, tig * 2 + i)`.
fn k32_8(which: usize, lane: u32, i: u32) -> (u32, u32) {
    let (gid, tig) = (lane >> 2, lane % 4);
    match which {
        0 => (gid, tig * 8 + i),
        1 => (tig * 8 + i, gid),
        _ => (gid, tig * 2 + i),
    }
}

/// m16n8k32: A `(gid + 8 * (i / 8), tig * 8 + i % 8)`, B `(tig * 8 + i, gid)`, C `(gid + 8 * (i / 2), tig * 2 + i % 2)`.
fn k32_16(which: usize, lane: u32, i: u32) -> (u32, u32) {
    let (gid, tig) = (lane >> 2, lane % 4);
    match which {
        0 => (gid + 8 * (i / 8), tig * 8 + i % 8),
        1 => (tig * 8 + i, gid),
        _ => (gid + 8 * (i / 2), tig * 2 + i % 2),
    }
}

/// m16n8k64: A row `gid` (+8 where bit 3 of `i` is set), column `tig * 8 + i % 8` (+32 where `i >= 16`); B row
/// `tig * 8 + i % 8` (+32 where `i >= 8`), column `gid`.
fn k64(which: usize, lane: u32, i: u32) -> (u32, u32) {
    let (gid, tig) = (lane >> 2, lane % 4);
    match which {
        0 => (gid + 8 * ((i / 8) % 2), tig * 8 + i % 8 + 32 * (i / 16)),
        1 => (tig * 8 + i % 8 + 32 * (i / 8), gid),
        _ => (gid + 8 * (i / 2), tig * 2 + i % 2),
    }
}

const PAIRS: [(unpopped_vocab::ElementKind, unpopped_vocab::ElementKind); 4] =
    [(I4, I4), (U4, U4), (I4, U4), (U4, I4)];

#[test]
fn the_m8n8k32_int4_atoms_place() {
    for (a, b) in PAIRS {
        let atom = MmaAtom::nvidia_m8n8k32_int4_s32(a, b);
        places(&atom, k32_8, (8, 8, 32), (a, b, I32));
    }
}

#[test]
fn the_m16n8k32_int4_atoms_place() {
    for (a, b) in PAIRS {
        let atom = MmaAtom::nvidia_m16n8k32_int4_s32(a, b);
        places(&atom, k32_16, (16, 8, 32), (a, b, I32));
    }
}

#[test]
fn the_m16n8k64_int4_atoms_place() {
    for (a, b) in PAIRS {
        let atom = MmaAtom::nvidia_m16n8k64_int4_s32(a, b);
        places(&atom, k64, (16, 8, 64), (a, b, I32));
    }
}

#[test]
fn the_int4_atoms_run_in_the_oracle() {
    // A fragment is a `Private` allocation of 4-bit elements; a `Store` into one is allowed since #506, so the oracle fills it,
    // runs `Stmt::Mma` and the D fragments equal the plain triple loop placed through the same table.
    for (a, b) in PAIRS {
        agrees(&MmaAtom::nvidia_m8n8k32_int4_s32(a, b), k32_8);
        agrees(&MmaAtom::nvidia_m16n8k32_int4_s32(a, b), k32_16);
        agrees(&MmaAtom::nvidia_m16n8k64_int4_s32(a, b), k64);
    }
}

#[test]
fn the_harness_kernels_are_well_formed() {
    // the harness kernels check, 8-bit control and 4-bit.
    assert_eq!(
        check_kernel(&mma_kernel(&MmaAtom::nvidia_m16n8k32_u8_s32())),
        Ok(())
    );
    assert_eq!(
        check_kernel(&mma_kernel(&MmaAtom::nvidia_m8n8k32_int4_s32(I4, U4))),
        Ok(())
    );
}
