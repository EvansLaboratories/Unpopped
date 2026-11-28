// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The NVIDIA block-scaled `mma.sync.aligned` atoms (`m16n8k32` mxf8f6f4, `m16n8k64` mxf4nvf4) of ROADMAP U7 theme 3 place
//! their data and scale fragments as the formulas below say.** Single source: CUTLASS `mma_traits_sm120.hpp` at
//! `NVIDIA/cutlass` commit 5c179ede (`SM120_16x8x32_TN_VS`, `SM120_16x8x64_TN_VS`); the PTX figures for the scale fragments
//! (PTX ISA 9.5 section 9.7.16.3, Figures 46-48) are images and were not read. The scale formulas below are written from
//! the CUTLASS layouts `((2,2,8),K):((8,0,1),16)` (SFA) and `((4,8),K):((0,1),8)` (SFB): the row of `SF_A` is `gid + 8 *
//! (lane % 2)`, the column of `SF_B` is `gid`. Placement only: not run in the oracle (F8UE4M3 is declined as a fragment
//! element by name, and the FP6 types too). Patent column: US12314717B2, applies 519 B; wrapper read 2026-11-27; not a
//! clearance; no attorney; findings not advice.

use unpopped::mma::{ExtraRole, MmaAtom};
use unpopped_vocab::ElementKind::{
    self, F4E2M1, F6E2M3, F6E3M2, F8E8M0, F8UE4M3, F32, Fp8E4M3FN as E4, Fp8E5M2 as E5,
};

#[path = "support/mma_run.rs"]
#[allow(dead_code, reason = "shared helpers: each test file uses a subset")]
mod support;
use support::places;

/// The 8-bit `m16n8k32` data tables.
fn k32(which: usize, lane: u32, i: u32) -> (u32, u32) {
    let (gid, tig) = (lane >> 2, lane % 4);
    match which {
        0 => (gid + 8 * ((i / 4) % 2), tig * 4 + (i % 4) + 16 * (i / 8)),
        1 => (tig * 4 + (i % 4) + 16 * (i / 4), gid),
        _ => (gid + 8 * (i / 2), tig * 2 + (i % 2)),
    }
}

/// The 4-bit `m16n8k64` data tables.
fn k64(which: usize, lane: u32, i: u32) -> (u32, u32) {
    let (gid, tig) = (lane >> 2, lane % 4);
    match which {
        0 => (gid + 8 * ((i / 8) % 2), tig * 8 + i % 8 + 32 * (i / 16)),
        1 => (tig * 8 + i % 8 + 32 * (i / 8), gid),
        _ => (gid + 8 * (i / 2), tig * 2 + i % 2),
    }
}

/// `SF_A` (`16 x kb`, column-major): value `v` of `lane` is row `gid + 8 * (lane % 2)`, block `v`.
/// `SF_B` (`kb x 8`): value `v` is block `v`, column `gid`.
fn scales(atom: &MmaAtom, kb: u32) {
    let by_role = |r| atom.extra_which(r).expect("the scale operand exists");
    let (sa, sb) = (by_role(ExtraRole::ScaleA), by_role(ExtraRole::ScaleB));
    assert_eq!(
        (atom.values(sa), atom.values(sb)),
        (kb, kb),
        "{}",
        atom.name()
    );
    let (mut seen_a, mut seen_b) = (vec![0u32; 16 * kb as usize], vec![0u32; 8 * kb as usize]);
    for lane in 0..32u32 {
        let gid = lane >> 2;
        for v in 0..kb {
            let a = gid + 8 * (lane % 2) + 16 * v;
            let b = v + kb * gid;
            assert_eq!(
                atom.id(sa, lane, v),
                Some(a),
                "{} SFA lane {lane} v {v}",
                atom.name()
            );
            assert_eq!(
                atom.id(sb, lane, v),
                Some(b),
                "{} SFB lane {lane} v {v}",
                atom.name()
            );
            seen_a[a as usize] += 1;
            seen_b[b as usize] += 1;
        }
    }
    // every scale element is held, and by the same number of lanes (the broadcast: 2 for SFA, 4 for SFB)
    assert!(seen_a.iter().all(|&c| c == 2), "{} SFA cover", atom.name());
    assert!(seen_b.iter().all(|&c| c == 4), "{} SFB cover", atom.name());
}

const KINDS: [ElementKind; 5] = [E4, E5, F6E3M2, F6E2M3, F4E2M1];

#[test]
fn the_mxf8f6f4_k32_atoms_place_data_and_scales() {
    for a in KINDS {
        for b in KINDS {
            let atom = MmaAtom::nvidia_m16n8k32_mxf8f6f4_f32(a, b);
            places(&atom, k32, (16, 8, 32), (a, b, F32));
            scales(&atom, 1);
        }
    }
}

#[test]
fn the_mxf4nvf4_k64_atoms_place_data_and_scales() {
    for (sf, block) in [(F8E8M0, 32), (F8E8M0, 16), (F8UE4M3, 16)] {
        let atom = MmaAtom::nvidia_m16n8k64_mxf4nvf4_f32(sf, block);
        places(&atom, k64, (16, 8, 64), (F4E2M1, F4E2M1, F32));
        scales(&atom, 64 / block);
    }
}

#[test]
#[should_panic(expected = "mxf4nvf4")]
fn an_unlisted_scale_pairing_is_refused() {
    // PTX Table 46: ue4m3 is only valid with scale_vec::4X (block 16)
    let _ = MmaAtom::nvidia_m16n8k64_mxf4nvf4_f32(F8UE4M3, 32);
}
