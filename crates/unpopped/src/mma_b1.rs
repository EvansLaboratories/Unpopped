// SPDX-License-Identifier: MIT OR Apache-2.0
//! The 1-bit NVIDIA `mma.sync.aligned` atoms `m8n8k128`, `m16n8k128` and `m16n8k256` as data (ROADMAP U7 theme 3): the
//! operands are `B1`, the accumulator `I32`, and the atom's [`BitOp`] says whether the bits are combined with `and` or
//! `xor` before they are counted (PTX `.and.popc`, `.xor.popc`). Additive: constructors only, no new statement and no
//! new element kind; the one new datum is the `bit_op` flag on [`MmaAtom`].
//!
//! **Source.** PTX ISA 9.5 (text of the public PDF, outside every repo), single source: section 9.7.16.5.5 `m8n8k128`
//! (A Figure 65 p. 546, B Figure 66 p. 547, C Figure 67 p. 548); 9.7.16.5.12 `m16n8k128` (A Figure 100 p. 583, B Figure
//! 101 p. 585, C Figure 102 p. 585); 9.7.16.5.13 `m16n8k256` (A Figure 103 p. 586, B Figures 104 and 105 pp. 588-589, C
//! Figure 106). Formulas read from the page text; the figures are cited by number.
//!
//! **Patent column** (`docs/PATENT-SWEEP.md`): outside as read for the b1 mma atom (US10452398B2 cl 1: per-field vector
//! popcount decoder, packed destination register: the atom accumulates popc into one s32 per C element); the packed/vector
//! popc statement stays held; findings not advice. 519 B: lifted by CireSnave 2026-11-26 (board 519 B) as his risk call:
//! the PM did not support it with a reading; not a clearance; no attorney; the Intel/NVIDIA tile-instruction method and
//! medium claims (US11086623B2 = 16/487,787, US11288069B2 = 16/487,755, US10884734B2 = 16/459,191) remain NEEDS MORE as
//! read. US12314717B2: applies 519 B; wrapper read 2026-11-27; not a clearance; no attorney; findings not advice.

use unpopped_vocab::ElementKind::{B1, I32};

use crate::exec::Scope;
use crate::mma::{AtomSpec, BitOp, MmaAtom, c_m16n8, flat};
use unpopped_vocab::layout_algebra::Layout;

fn build(name: &str, mnk: (u32, u32, u32), layouts: [&Layout; 3], op: BitOp) -> MmaAtom {
    MmaAtom::new(
        AtomSpec {
            name,
            mnk,
            dtypes: (B1, B1, I32),
            scope: Scope::Subgroup,
            lanes: 32,
        },
        layouts,
    )
    .and_then(|a| a.with_bit_op(op))
    .expect("the PTX 1-bit fragment layouts are bijections")
}

fn suffix(op: BitOp) -> &'static str {
    match op {
        BitOp::And => "and_popc",
        BitOp::Xor => "xor_popc",
    }
}

impl MmaAtom {
    /// `mma.sync.aligned.m8n8k128.row.col.s32.b1.b1.s32` with `.and.popc` or `.xor.popc` (PTX ISA 9.5 s9.7.16.5.5).
    #[must_use]
    pub fn nvidia_m8n8k128_b1_s32(op: BitOp) -> Self {
        // A: row = lane / 4, col = (lane % 4) * 32 + i; B: row = (lane % 4) * 32 + i, col = lane / 4; C: row = lane / 4,
        // col = (lane % 4) * 2 + i
        let name = format!("nvidia.mma.sync.m8n8k128.b1.{}.s32", suffix(op));
        build(
            &name,
            (8, 8, 128),
            [
                &flat(&[(4, 256), (8, 1), (32, 8)]),
                &flat(&[(4, 32), (8, 128), (32, 1)]),
                &flat(&[(4, 16), (8, 1), (2, 8)]),
            ],
            op,
        )
    }

    /// `mma.sync.aligned.m16n8k128.row.col.s32.b1.b1.s32` with `.and.popc` or `.xor.popc` (PTX ISA 9.5 s9.7.16.5.12).
    #[must_use]
    pub fn nvidia_m16n8k128_b1_s32(op: BitOp) -> Self {
        // A: row = lane / 4 (+8 for i >= 32), col = (lane % 4) * 32 + i % 32; B as m8n8k128 with K = 128
        let name = format!("nvidia.mma.sync.m16n8k128.b1.{}.s32", suffix(op));
        build(
            &name,
            (16, 8, 128),
            [
                &flat(&[(4, 512), (8, 1), (32, 16), (2, 8)]),
                &flat(&[(4, 32), (8, 128), (32, 1)]),
                &c_m16n8(),
            ],
            op,
        )
    }

    /// `mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32` with `.and.popc` or `.xor.popc` (PTX ISA 9.5 s9.7.16.5.13).
    #[must_use]
    pub fn nvidia_m16n8k256_b1_s32(op: BitOp) -> Self {
        // A: row = lane / 4 (+8 where bit 5 of i is set), col = (lane % 4) * 32 + i % 32 (+128 for i >= 64); B: row =
        // (lane % 4) * 32 + i % 32 (+128 for i >= 32), col = lane / 4
        let name = format!("nvidia.mma.sync.m16n8k256.b1.{}.s32", suffix(op));
        build(
            &name,
            (16, 8, 256),
            [
                &flat(&[(4, 512), (8, 1), (32, 16), (2, 8), (2, 2048)]),
                &flat(&[(4, 32), (8, 256), (32, 1), (2, 128)]),
                &c_m16n8(),
            ],
            op,
        )
    }
}
