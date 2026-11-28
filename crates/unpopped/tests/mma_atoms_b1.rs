// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The 1-bit NVIDIA `mma.sync.aligned` atoms (`m8n8k128`, `m16n8k128`, `m16n8k256`; `.and.popc`, `.xor.popc`) of ROADMAP
//! U7 theme 3 put every fragment element where the PTX ISA says and compute `D = C + popc(A op B)`.** Layouts: `(row, col)`
//! formulas from PTX ISA 9.5 sections 9.7.16.5.5 (Figures 65-67, pp. 546-548), 9.7.16.5.12 (Figures 100-102, pp. 583-585) and
//! 9.7.16.5.13 (Figures 103-106, pp. 586-589). Patent column: outside as read for the b1 mma atom (US10452398B2 cl 1; findings
//! not advice; the packed/vector popc statement stays held); applies 519 B; US12314717B2 wrapper read 2026-11-27; not a
//! clearance; no attorney.

use unpopped::mma::{BitOp, MmaAtom};
use unpopped_vocab::ElementKind::{B1, I32};

#[path = "support/mma_run.rs"]
#[allow(dead_code, reason = "shared helpers: each test file uses a subset")]
mod support;
use support::{Table, fragments, logical, places, run};

/// m8n8k128: A `(gid, tig * 32 + i)`, B `(tig * 32 + i, gid)`, C `(gid, tig * 2 + i)`.
fn k128_8(which: usize, lane: u32, i: u32) -> (u32, u32) {
    let (gid, tig) = (lane >> 2, lane % 4);
    match which {
        0 => (gid, tig * 32 + i),
        1 => (tig * 32 + i, gid),
        _ => (gid, tig * 2 + i),
    }
}

/// m16n8k128: A row `gid` (+8 for `i >= 32`), col `tig * 32 + i % 32`; B as above; C row `gid` (+8 for `i >= 2`), col
/// `tig * 2 + i % 2`.
fn k128_16(which: usize, lane: u32, i: u32) -> (u32, u32) {
    let (gid, tig) = (lane >> 2, lane % 4);
    match which {
        0 => (gid + 8 * (i / 32), tig * 32 + i % 32),
        1 => (tig * 32 + i, gid),
        _ => (gid + 8 * (i / 2), tig * 2 + i % 2),
    }
}

/// m16n8k256: A row `gid` (+8 where `i` is in 32..64 or 96..128), col `tig * 32 + i % 32` (+128 for `i >= 64`); B row
/// `tig * 32 + i % 32` (+128 for `i >= 32`), col `gid`; C as m16n8k128.
fn k256(which: usize, lane: u32, i: u32) -> (u32, u32) {
    let (gid, tig) = (lane >> 2, lane % 4);
    match which {
        0 => (gid + 8 * ((i / 32) % 2), tig * 32 + i % 32 + 128 * (i / 64)),
        1 => (tig * 32 + i % 32 + 128 * (i / 32), gid),
        _ => (gid + 8 * (i / 2), tig * 2 + i % 2),
    }
}

const OPS: [BitOp; 2] = [BitOp::And, BitOp::Xor];

/// `D = C + popc(A op B)` over the logical matrices, placed through `table`, versus what the oracle writes.
fn runs(atom: &MmaAtom, table: Table, op: BitOp) {
    let l = logical(atom);
    let (m, n, k) = atom.mnk();
    let (m, n, k) = (m as usize, n as usize, k as usize);
    let mut d = vec![0.0; m * n];
    for col in 0..n {
        for row in 0..m {
            d[row + m * col] = (0..k).fold(l.c[row + m * col], |s, i| {
                let (x, y) = (l.a[row + m * i] as u8, l.b[i + k * col] as u8);
                s + f64::from(match op {
                    BitOp::And => x & y,
                    _ => x ^ y,
                })
            });
        }
    }
    let want = fragments(table, 2, m as u32, atom.values(2), &d);
    let got = run(atom, table, &l).unwrap_or_else(|e| panic!("{}: {e:?}", atom.name()));
    assert_eq!(got, want, "{}", atom.name());
}

#[test]
fn the_m8n8k128_b1_atoms_place_and_run() {
    for op in OPS {
        let atom = MmaAtom::nvidia_m8n8k128_b1_s32(op);
        assert_eq!(atom.bit_op(), Some(op));
        places(&atom, k128_8, (8, 8, 128), (B1, B1, I32));
        runs(&atom, k128_8, op);
    }
}

#[test]
fn the_m16n8k128_b1_atoms_place_and_run() {
    for op in OPS {
        let atom = MmaAtom::nvidia_m16n8k128_b1_s32(op);
        places(&atom, k128_16, (16, 8, 128), (B1, B1, I32));
        runs(&atom, k128_16, op);
    }
}

#[test]
fn the_m16n8k256_b1_atoms_place_and_run() {
    for op in OPS {
        let atom = MmaAtom::nvidia_m16n8k256_b1_s32(op);
        places(&atom, k256, (16, 8, 256), (B1, B1, I32));
        runs(&atom, k256, op);
    }
}

#[test]
fn the_two_bit_operations_differ_and_the_flag_is_checked() {
    // control: the same data gives different D under `and` and `xor`, so the runs above cannot pass with the flag ignored
    let l = logical(&MmaAtom::nvidia_m8n8k128_b1_s32(BitOp::And));
    let and = run(&MmaAtom::nvidia_m8n8k128_b1_s32(BitOp::And), k128_8, &l).unwrap();
    let xor = run(&MmaAtom::nvidia_m8n8k128_b1_s32(BitOp::Xor), k128_8, &l).unwrap();
    assert_ne!(and, xor);
    // the flag needs B1 operands
    let s8 = MmaAtom::nvidia_m16n8k32_s8_s32();
    assert!(s8.with_bit_op(BitOp::And).is_err());
}

#[test]
fn the_harness_packs_b1_lowest_bit_first_like_the_oracle() {
    // the byte 0b0000_0101 holds elements 0 and 2 set (KISS packing: LSB = lowest logical index, pinned in
    // `sub_byte_packing.rs`); the harness must pack the same bits for the same elements
    let xs = [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    assert_eq!(support::buffer(B1, &xs).raw_bytes(), &[0b0000_0101]);
}
