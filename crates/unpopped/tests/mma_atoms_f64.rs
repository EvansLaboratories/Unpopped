// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The f64 `mma.sync.aligned.m16n8k{4,8,16}` atoms of ROADMAP U7 theme 3 put every fragment element where the PTX ISA
//! says** and run through the oracle. Layouts are `(row, col)` formulas from PTX ISA 9.5 sections 9.7.16.5.6 (`k4`),
//! 9.7.16.5.7 (`k8`) and 9.7.16.5.8 (`k16`: A Figure 83 pp. 564-565, B Figure 85 p. 566, C Figure 86 p. 567; the A column formula is read across a page break, see
//! `mma_f64.rs`). Second source of all three: CUTLASS `mma_traits_sm90.hpp`, `main` 5c179ede.

use unpopped::mma::MmaAtom;
use unpopped_vocab::ElementKind::F64;

#[path = "support/mma_run.rs"]
#[allow(dead_code, reason = "shared helpers: each test file uses a subset")]
mod support;
use support::{agrees, places};

/// `A` value `i` is row `gid + 8 * (i % 2)`, column `tig + 4 * (i / 2)` (`k4` has two values, `k8` four, `k16` eight);
/// `B` value `i` is row `tig + 4 * i`, column `gid`; `C` as every `m16n8` atom.
fn f64_table(which: usize, lane: u32, i: u32) -> (u32, u32) {
    let (gid, tig) = (lane >> 2, lane % 4);
    match which {
        0 => (gid + 8 * (i % 2), tig + 4 * (i / 2)),
        1 => (tig + 4 * i, gid),
        _ => (gid + 8 * (i / 2), tig * 2 + (i % 2)),
    }
}

#[test]
fn the_f64_m16n8_atoms_place_and_run() {
    for (atom, k) in [
        (MmaAtom::nvidia_m16n8k4_f64_f64(), 4),
        (MmaAtom::nvidia_m16n8k8_f64_f64(), 8),
        (MmaAtom::nvidia_m16n8k16_f64_f64(), 16),
    ] {
        places(&atom, f64_table, (16, 8, k), (F64, F64, F64));
        agrees(&atom, f64_table);
    }
}

#[test]
fn the_fragment_lengths_follow_the_ptx_register_counts() {
    // PTX: A holds two, four and eight .f64 registers; B one, two and four; C and D four.
    for (atom, a, b) in [
        (MmaAtom::nvidia_m16n8k4_f64_f64(), 2, 1),
        (MmaAtom::nvidia_m16n8k8_f64_f64(), 4, 2),
        (MmaAtom::nvidia_m16n8k16_f64_f64(), 8, 4),
    ] {
        assert_eq!(
            (atom.values(0), atom.values(1), atom.values(2)),
            (a, b, 4),
            "{}",
            atom.name()
        );
    }
}
