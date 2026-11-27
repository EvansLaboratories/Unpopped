// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The NVIDIA `mma.sync` atoms of ROADMAP U7 theme 3 put every fragment element where CuTe says** and run through the
//! oracle. Each atom's thread-value layouts are checked against `(row, col)` formulas written here, derived from CUTLASS
//! `mma_traits_sm80.hpp` / `mma_traits_sm89.hpp` (`main` head 5c179ede, 2026-11-25; a CuTe `BLayout` indexes `(N, K)`).
//! The PTX ISA fragment tables for these shapes were NOT READ (the 9.5 page is cut off before 9.7.16): UNVERIFIED
//! against the ISA. Patent column of the 8-bit atoms: applies 519 B; wrapper read 2026-11-27; not a clearance; no
//! attorney; findings not advice.

use unpopped::mma::MmaAtom;
use unpopped_vocab::ElementKind;

#[path = "support/mma_run.rs"]
#[allow(dead_code, reason = "shared helpers: each test file uses a subset")]
mod support;
use support::{Table, agrees, fragments, logical, reference, run};

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

/// m16n8k32 8-bit (s8, u8, fp8): the `S32S8S8S32` layouts, shared by every signedness and fp8 pairing.
fn k32(which: usize, lane: u32, i: u32) -> (u32, u32) {
    let (gid, tig) = (lane >> 2, lane % 4);
    match which {
        0 => (gid + 8 * ((i / 4) % 2), tig * 4 + (i % 4) + 16 * (i / 8)),
        1 => (tig * 4 + (i % 4) + 16 * (i / 4), gid),
        _ => (gid + 8 * (i / 2), tig * 2 + (i % 2)),
    }
}

/// Every atom places its fragment elements exactly where the table says, the table covers each matrix once, and the
/// atom's name, shape and types are stated.
fn places(
    atom: &MmaAtom,
    table: Table,
    mnk: (u32, u32, u32),
    tys: (ElementKind, ElementKind, ElementKind),
) {
    assert_eq!((atom.mnk(), atom.dtypes()), (mnk, tys), "{}", atom.name());
    let (m, n, k) = mnk;
    for (which, rows, cols) in [(0, m, k), (1, k, n), (2, m, n)] {
        let mut seen = vec![0u32; (rows * cols) as usize];
        for lane in 0..atom.lanes() {
            for v in 0..atom.values(which) {
                let (r, c) = table(which, lane, v);
                assert_eq!(
                    atom.id(which, lane, v),
                    Some(r + rows * c),
                    "{} op {which} lane {lane} v {v}",
                    atom.name()
                );
                seen[(r + rows * c) as usize] += 1;
            }
        }
        assert!(
            seen.iter().all(|&c| c == 1),
            "{} table {which} is not a bijection",
            atom.name()
        );
    }
}

use ElementKind::{Bf16, F16, F32, F64, Fp8E4M3FN as E4, Fp8E5M2 as E5, I8, I32, U8};

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

#[test]
fn the_unsigned_and_mixed_8_bit_atoms_place_and_run() {
    for (atom, a, b) in [
        (MmaAtom::nvidia_m16n8k32_u8_s32(), U8, U8),
        (MmaAtom::nvidia_m16n8k32_s8_u8_s32(), I8, U8),
        (MmaAtom::nvidia_m16n8k32_u8_s8_s32(), U8, I8),
    ] {
        places(&atom, k32, (16, 8, 32), (a, b, I32));
        agrees(&atom, k32);
    }
}

#[test]
fn the_fp8_atoms_place_and_run_including_a_different_type_for_b() {
    for (atom, a, b) in [
        (MmaAtom::nvidia_m16n8k32_e4m3_f32(), E4, E4),
        (MmaAtom::nvidia_m16n8k32_e5m2_f32(), E5, E5),
        (MmaAtom::nvidia_m16n8k32_e4m3_e5m2_f32(), E4, E5),
        (MmaAtom::nvidia_m16n8k32_e5m2_e4m3_f32(), E5, E4),
    ] {
        places(&atom, k32, (16, 8, 32), (a, b, F32));
        agrees(&atom, k32);
    }
}

#[test]
fn the_mixed_signedness_atoms_are_not_the_same_atom_as_their_swap() {
    // control: A != B in type means the swap of the two atoms computes a different D on the same data, so a test that
    // swapped the types would not pass silently.
    let (ab, ba) = (
        MmaAtom::nvidia_m16n8k32_s8_u8_s32(),
        MmaAtom::nvidia_m16n8k32_u8_s8_s32(),
    );
    assert_ne!(ab.dtypes(), ba.dtypes());
    let l = logical(&ab);
    let right = fragments(k32, 2, 16, 4, &reference(&ab, &l, |x| x));
    assert_eq!(run(&ab, k32, &l).unwrap(), right);
    assert_ne!(
        run(&ba, k32, &l).unwrap(),
        right,
        "the swapped atom reads B as signed and A as unsigned"
    );
}
