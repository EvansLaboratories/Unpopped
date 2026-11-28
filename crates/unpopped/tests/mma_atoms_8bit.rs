// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The 4x element-size NVIDIA `mma.sync.aligned.m16n8k32` atoms (u8, mixed s8/u8, fp8) of ROADMAP U7 theme 3 put every
//! fragment element where the PTX ISA says** and run through the oracle. Layout: `(row, col)` formulas from PTX ISA 9.5
//! section 9.7.16.5.10 (pp. 572-576, Figures 91, 93, 95); second source CUTLASS `mma_traits_sm80.hpp` and
//! `mma_traits_sm89.hpp` (`main` 5c179ede). Patent column: US12314717B2, applies 519 B; wrapper read 2026-11-27; not a
//! clearance; no attorney; findings not advice.

use unpopped::mma::MmaAtom;
use unpopped_vocab::ElementKind;

#[path = "support/mma_run.rs"]
#[allow(dead_code, reason = "shared helpers: each test file uses a subset")]
mod support;
use support::{agrees, fragments, logical, places, reference, run};

/// m16n8k32 8-bit (s8, u8, fp8): the `S32S8S8S32` layouts, shared by every signedness and fp8 pairing.
fn k32(which: usize, lane: u32, i: u32) -> (u32, u32) {
    let (gid, tig) = (lane >> 2, lane % 4);
    match which {
        0 => (gid + 8 * ((i / 4) % 2), tig * 4 + (i % 4) + 16 * (i / 8)),
        1 => (tig * 4 + (i % 4) + 16 * (i / 4), gid),
        _ => (gid + 8 * (i / 2), tig * 2 + (i % 2)),
    }
}

/// m16n8k16 integer (PTX 9.7.16.5.9, Figures 87-89): A `(gid + 8 * (i / 4), tig * 4 + i % 4)`, B `(tig * 4 + i, gid)`.
fn k16(which: usize, lane: u32, i: u32) -> (u32, u32) {
    let (gid, tig) = (lane >> 2, lane % 4);
    match which {
        0 => (gid + 8 * (i / 4), tig * 4 + (i % 4)),
        1 => (tig * 4 + i, gid),
        _ => (gid + 8 * (i / 2), tig * 2 + (i % 2)),
    }
}

use ElementKind::{F32, Fp8E4M3FN as E4, Fp8E5M2 as E5, I8, I32, U8};

#[test]
fn the_unsigned_and_mixed_8_bit_atoms_place_and_run() {
    for (atom, a, b) in [
        (MmaAtom::nvidia_m16n8k32_u8_s32(), U8, U8),
        (MmaAtom::nvidia_m16n8k32_s8_u8_s32(), I8, U8),
        (MmaAtom::nvidia_m16n8k32_u8_s8_s32(), U8, I8),
    ] {
        places(&atom, k32, (16, 8, 32), (a, b, I32));
    }
}

#[test]
fn the_integer_k16_atoms_place_and_run() {
    for (atom, a, b) in [
        (MmaAtom::nvidia_m16n8k16_s8_s32(), I8, I8),
        (MmaAtom::nvidia_m16n8k16_u8_s32(), U8, U8),
        (MmaAtom::nvidia_m16n8k16_s8_u8_s32(), I8, U8),
        (MmaAtom::nvidia_m16n8k16_u8_s8_s32(), U8, I8),
    ] {
        places(&atom, k16, (16, 8, 16), (a, b, I32));
        agrees(&atom, k16);
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

#[test]
fn the_fp6_and_fp4_k32_atoms_share_the_8_bit_placement() {
    // PTX 9.7.16.5.10 gives one A, B and C table to s8, u8, e4m3, e5m2, e3m2, e2m3 and e2m1 (Figures 91, 93, 94, 95), so the
    // FP6 and FP4 atoms are placed by `k32`.
    use ElementKind::{F4E2M1, F6E2M3, F6E3M2};
    for (atom, t) in [
        (MmaAtom::nvidia_m16n8k32_e3m2_f32(), F6E3M2),
        (MmaAtom::nvidia_m16n8k32_e2m3_f32(), F6E2M3),
        (MmaAtom::nvidia_m16n8k32_e2m1_f32(), F4E2M1),
    ] {
        places(&atom, k32, (16, 8, 32), (t, t, F32));
    }
}

#[test]
fn the_fp6_and_fp4_k32_atoms_run() {
    // E2M1 fragments run in the oracle (two to a byte, low nibble first) and so do E3M2 and E2M3 since f6-oracle-storage: one
    // element per byte, the two high bits zero (`docs/design-f6-storage.md`). The values are lattice points of each format,
    // encoded by the test from the OCP MX definition.
    for atom in [
        MmaAtom::nvidia_m16n8k32_e3m2_f32(),
        MmaAtom::nvidia_m16n8k32_e2m3_f32(),
        MmaAtom::nvidia_m16n8k32_e2m1_f32(),
    ] {
        agrees(&atom, k32);
    }
}
