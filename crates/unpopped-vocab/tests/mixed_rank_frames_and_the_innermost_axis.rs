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
//! 3. **§6.5-0014 as amended by KISS#519 (option C): the broadcast MASK is
//!    computed over the frame**. Bit `i` is set iff frame axis `i` has extent
//!    `> 1` and the operand's stride along it is `0`, where an axis the operand
//!    lacks counts as stride `0`. The layout tag, vector width, divisibility and
//!    the §6.5-0013 stride test still read the operand's *own* axes. The first
//!    draft of the clause padded the layout too. That collided dense and strided
//!    im2col outputs under one token, and KISS amended it after Unpopped's im2col
//!    gate went red. The examples below are the amendment's own.
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

/// KISS#519 golden. `[256]` in frame `[128, 256]`: frame axis 0 is absent
/// from the operand (stride 0, frame extent 128), so mask `01`. The layout
/// reads its own axes (`co`), and the width is not forced to `v1`.
#[test]
fn the_mask_is_over_the_frame_but_the_layout_is_own_axes() {
    let row = od(&[256], &[1]);
    let full = od(&[128, 256], &[256, 1]);
    let t = tok(OpCategory::BinaryElementwise, &[row, full, full]);
    assert_eq!(sub(&t, 0), "co/01/v4/d16/f", "token was {t}");
    assert_eq!(sub(&t, 1), "co/00/v4/d16/f", "token was {t}");
}

/// KISS#519 golden. `[1, 256]` with stride `[0, 1]`: its own extent-1 axis 0 is
/// stride 0 and the frame extent there is 128, so the bit is set (§6.6-0008),
/// although the operand's own axis is unit.
#[test]
fn an_own_unit_axis_of_stride_zero_sets_its_bit_when_the_frame_is_wider() {
    let row = od(&[1, 256], &[0, 1]);
    let full = od(&[128, 256], &[256, 1]);
    let t = tok(OpCategory::BinaryElementwise, &[row, full, full]);
    assert_eq!(sub(&t, 0), "co/01/v4/d16/f", "token was {t}");
}

/// KISS#519 goldens, the case that forced the amendment: a rank-3 operand in a
/// rank-4 frame. Dense and strided must now be DISTINGUISHABLE (`co` vs `st`),
/// though both carry mask `01` for the absent frame axis.
#[test]
fn dense_and_strided_rank_deficient_operands_no_longer_collide() {
    let frame = od(&[8, 4, 16, 64], &[4096, 1024, 64, 1]);
    let dense = od(&[4, 16, 64], &[1024, 64, 1]);
    let strided = od(&[4, 16, 64], &[2048, 128, 2]);
    let td = tok(OpCategory::BinaryElementwise, &[frame, dense, frame]);
    let ts = tok(OpCategory::BinaryElementwise, &[frame, strided, frame]);
    assert_eq!(sub(&td, 1), "co/01/v4/d16/f", "token was {td}");
    assert_eq!(sub(&ts, 1), "st/01/v1/d16/f", "token was {ts}");
}

/// The operand sub-key at position `i` of a token.
fn sub(token: &str, i: usize) -> &str {
    token.split('|').nth(7).unwrap().split(';').nth(i).unwrap()
}
