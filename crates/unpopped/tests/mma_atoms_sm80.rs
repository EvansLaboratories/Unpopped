// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The 2x and 1x element-size NVIDIA `mma.sync` atoms of ROADMAP U7 theme 3 put every fragment element where the PTX ISA
//! says** and run through the oracle. Layouts are `(row, col)` formulas from PTX ISA 9.5: `m16n8k8` f16/bf16 section
//! 9.7.16.5.7 (pp. 555-561), `m16n8k4` tf32 section 9.7.16.5.6 (pp. 549-553), `m8n8k4` f64 section 9.7.16.5.2 (pp. 537-538);
//! second source CUTLASS `mma_traits_sm80.hpp` (`main` 5c179ede). `wmma` is absent on purpose: PTX ISA 9.5 section
//! 9.7.16.4.1 says its fragment layout is unspecified and architecture dependent.

use unpopped::mma::MmaAtom;
use unpopped_vocab::ElementKind;

#[path = "support/mma_run.rs"]
#[allow(dead_code, reason = "shared helpers: each test file uses a subset")]
mod support;
use support::{agrees, places};

/// m16n8k8 (f16, bf16): A is `16 x 8`, B `8 x 8`, C `16 x 8` (CuTe `SM80_16x8_Row`, `SM80_8x8_Row`).
fn k8(which: usize, lane: u32, i: u32) -> (u32, u32) {
    let (gid, tig) = (lane >> 2, lane % 4);
    match which {
        1 => (tig * 2 + i, gid),
        _ => (gid + 8 * (i / 2), tig * 2 + (i % 2)),
    }
}

/// m16n8k4 tf32: A value `i` is row `gid + 8 i`, column `tig`; B is row `tig`, column `gid` (CuTe `SM80_8x4`).
fn k4(which: usize, lane: u32, i: u32) -> (u32, u32) {
    let (gid, tig) = (lane >> 2, lane % 4);
    match which {
        0 => (gid + 8 * i, tig),
        1 => (tig, gid),
        _ => (gid + 8 * (i / 2), tig * 2 + (i % 2)),
    }
}

/// m8n8k4 f64: A row `gid`, column `tig`; B row `tig`, column `gid`; C row `gid`, column `2 tig + i`.
fn d884(which: usize, lane: u32, i: u32) -> (u32, u32) {
    let (gid, tig) = (lane >> 2, lane % 4);
    match which {
        0 => (gid, tig),
        1 => (tig, gid),
        _ => (gid, tig * 2 + i),
    }
}

use ElementKind::{Bf16, F16, F32, F64};

#[test]
fn the_k8_half_atoms_place_and_run() {
    for (atom, ty) in [
        (MmaAtom::nvidia_m16n8k8_f16_f32(), F16),
        (MmaAtom::nvidia_m16n8k8_bf16_f32(), Bf16),
    ] {
        places(&atom, k8, (16, 8, 8), (ty, ty, F32));
        agrees(&atom, k8);
    }
}

#[test]
fn the_k4_tf32_atom_places_and_runs_and_carries_the_flag() {
    let atom = MmaAtom::nvidia_m16n8k4_tf32_f32();
    places(&atom, k4, (16, 8, 4), (F32, F32, F32));
    assert!(atom.tf32_inputs());
    agrees(&atom, k4);
}

#[test]
fn the_f64_atom_places_and_runs() {
    let atom = MmaAtom::nvidia_m8n8k4_f64_f64();
    places(&atom, d884, (8, 8, 4), (F64, F64, F64));
    agrees(&atom, d884);
}
