//! "Innermost" means axis `rank − 1`, and the work class counts the
//! right-aligned frame: KISS-CLASSIFY as of KISS#517 (`KISS@bc16715`), tracked
//! in issue #31.
//!
//! Two rules, each a divergence `unpopped-vocab ≤ 0.13` had:
//!
//! 1. **§6.3-0011 innermost axis = `rank − 1`**, *"including when that axis has
//!    extent `1`"*, for vector width, divisibility and the stride test. Only the
//!    layout tag keeps the "innermost non-unit axis" notion (§6.5-0002).
//! 2. **§6.6-0013 / §6.5-0010 right alignment for the work class.** The frame
//!    extent at each axis is the max over operands *right-aligned*. Before this,
//!    the work class aligned them left.
//!
//! **Held, not tested here:** §6.5-0014's frame-padded layout/mask. It treats
//! every rank deficiency as a broadcast, which collides dense and strided im2col
//! outputs under one token. KISS confirmed the gap is in the clause, and the
//! ruling is pending (issue #31).
//!
//! # Cross-implementation goldens
//!
//! Values marked `FUEL` were produced by Fuel's independent deriver
//! (`derive_structure_key_token`, Fuel `origin/main` on 2026-10-02) from the same
//! inputs. They were run, not read off a doc comment. Agreement between two
//! implementations is evidence only because the methods differ.

use unpopped_vocab::{ArchSku, ElementKind, OpCategory, OperandDesc, structure_key_token};

fn od(shape: &[i64], strides: &[i64]) -> OperandDesc {
    OperandDesc::new(shape.len(), shape, strides, ElementKind::F32, 256)
}

fn tok(op: OpCategory, ops: &[OperandDesc]) -> String {
    structure_key_token(op, ops, ArchSku::Sm89)
}

/// FUEL golden, whole token. `[4, 1]`: the innermost axis is axis 1 (extent
/// 1), so the bucket is `da` and the width `v1`. It was `d4`, read off axis 0.
#[test]
fn a_trailing_unit_axis_is_still_the_innermost_axis() {
    let x = od(&[4, 1], &[1, 1]);
    assert_eq!(
        tok(OpCategory::UnaryElementwise, &[x]),
        "sk4|une|f32|cuda:sm89|ix32|warp|r2|co/00/v1/da/f|-"
    );
}

/// The `[8, 0]` case 0.13.0's CHANGELOG left open: the innermost axis is axis 1
/// with extent 0, so `da`/`v1`. It was `d8`. The layout tag ignores extent-0
/// axes (§6.5-0002), so it stays `co`.
#[test]
fn a_zero_extent_innermost_axis_is_da_v1() {
    let x = od(&[8, 0], &[1, 1]);
    assert_eq!(
        tok(OpCategory::UnaryElementwise, &[x]),
        "sk4|une|f32|cuda:sm89|ix32|warp|r2|co/00/v1/da/f|-"
    );
}

/// FUEL golden, work-class field. §6.6-0013 right alignment: `[2, 64]` with
/// `[64]` is a frame of `2 × 64 = 128` elements, which is `block`. Left alignment
/// counted `64 × 64 = 4096`, which is `grid`. Fuel's deriver maps operand axes to
/// the frame's trailing indices (`off = rank − len`).
///
/// Only the work-class field is compared. Fuel's whole token also carries the
/// §6.5-0014 padding of operand 1 (`br/01/…`), and that part is held.
#[test]
fn the_work_class_counts_the_right_aligned_frame() {
    let full = od(&[2, 64], &[64, 1]);
    let row = od(&[64], &[1]);
    let t = tok(OpCategory::BinaryElementwise, &[full, row, full]);
    assert_eq!(t.split('|').nth(5), Some("block"), "token was {t}");
}

/// Control for the work class: same rank, same frame either way. A fix that
/// changed same-rank counting would fail here.
#[test]
fn same_rank_work_class_is_unchanged() {
    let a = od(&[2, 64], &[64, 1]);
    let t = tok(OpCategory::BinaryElementwise, &[a, a, a]);
    assert_eq!(t.split('|').nth(5), Some("block"), "token was {t}");
}

/// Control: same-rank operands are untouched by both rules. This is the crate's
/// own doc example.
#[test]
fn same_rank_operands_are_unchanged() {
    let a = od(&[128, 256], &[256, 1]);
    assert_eq!(
        tok(OpCategory::BinaryElementwise, &[a, a, a]),
        "sk4|bin|f32|cuda:sm89|ix32|grid|r2|co/00/v4/d16/f;co/00/v4/d16/f;co/00/v4/d16/f|-"
    );
}

/// Control: a leading unit axis is not the innermost axis. `[1, 8]` reads axis 1
/// (extent 8), so `d8`, the clause's own contrasting example.
#[test]
fn a_leading_unit_axis_does_not_change_the_innermost() {
    let x = od(&[1, 8], &[8, 1]);
    assert_eq!(
        tok(OpCategory::UnaryElementwise, &[x]),
        "sk4|une|f32|cuda:sm89|ix32|warp|r2|co/00/v4/d8/f|-"
    );
}
