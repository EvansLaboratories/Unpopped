//! The frame broadcast mask cannot tell whether an output aliases. The
//! operand's own layout tag can.
//!
//! **Which field answers which question, for a consumer writing a gate:**
//! - *Does this output write two positions to one address?* Read
//!   `OperandKey::contig`. `Contiguity::Broadcast` means an own axis with extent
//!   `> 1` and stride `0`. Use this for reduction, im2col, select and any other
//!   output that is narrower than the frame on purpose.
//! - *Does an elementwise output cover the whole frame?* Read `OperandKey::bcast`.
//!   In an elementwise op an output that misses a frame axis overwrites itself
//!   across that axis, and only the frame mask can see it, because `contig`
//!   ignores axes the operand doesn't have.
//!
//! Neither field answers both questions.
//!
//! Since 0.14.0 `OperandKey::bcast` is over the **frame**, with operands
//! right-aligned (KISS-CLASSIFY §6.5-0014 as amended by KISS#519, §6.6-0008,
//! §6.6-0013). Bit `i` means "frame axis `i` has extent `> 1` and this operand
//! has stride `0` there, or no axis there at all". That is the address-math
//! question: in both cases the axis drops out of the offset.
//!
//! It is not the aliasing question. A reduction, im2col or select output that
//! is narrower than the frame on purpose sets the same bit as an elementwise
//! output with a real stride-0 broadcast. Ask the aliasing question of the
//! operand's **own** axes instead: `contig == Contiguity::Broadcast` iff
//! the operand has an own axis with extent `> 1` and stride `0`
//! (`derive_operand_key`). `Contiguity::Contig` implies no such axis.
//!
//! Baracuda's 0.11-era gates read `bcast` in own-axis coordinates. Moving to
//! 0.14 turned 19 of its tests red (board #106).

use unpopped_vocab::{ArchSku, Contiguity, ElementKind, OpCategory, OperandDesc, structure_key};

fn od(shape: &[i64], strides: &[i64]) -> OperandDesc {
    OperandDesc::new(shape.len(), shape, strides, ElementKind::F32, 256)
}

fn key(op: OpCategory, ops: &[OperandDesc]) -> unpopped_vocab::StructureKey {
    structure_key(op, ops, ArchSku::Sm89)
}

const ROW_SUM_IN: [i64; 2] = [256, 128];

/// A dense collapsed row-reduction output `[256,128] → [256]`. It sets frame
/// bit 0 because it has no frame axis 0. Right alignment puts its one own
/// axis (the kept row axis) on frame axis 1, so the bit position does not name
/// the reduced axis either. The layout tag is dense.
#[test]
fn a_dense_collapsed_output_sets_a_frame_bit_and_is_contig() {
    let k = key(
        OpCategory::Reduction,
        &[od(&ROW_SUM_IN, &[128, 1]), od(&[256], &[1])],
    );
    let out = k.operands[1];
    assert_eq!(out.bcast.0, 0b01, "absent from frame axis 0");
    assert_eq!(out.contig, Contiguity::Contig, "own axes: dense, no alias");
}

/// The control: a collapsed output whose one own axis really has stride 0. It
/// writes all 256 rows to one element. The frame mask gains bit 1. The layout
/// tag flips to `Broadcast`, and that change is what an aliasing gate must read.
#[test]
fn an_aliasing_collapsed_output_is_broadcast_on_its_own_axes() {
    let k = key(
        OpCategory::Reduction,
        &[od(&ROW_SUM_IN, &[128, 1]), od(&[256], &[0])],
    );
    let out = k.operands[1];
    assert_eq!(out.bcast.0, 0b11);
    assert_eq!(out.contig, Contiguity::Broadcast);
}

/// Keepdim `[256,1]`: the reduced axis has extent 1, so it is never an own
/// broadcast, and the layout is dense either way. Its frame bit follows the
/// unit axis's *stride* (§6.6-0008: an own extent-1, stride-0 axis sets its bit
/// where the frame is wider), so two allocations of the same dense output key
/// differently. A gate that reads `bcast` here is reading a stride on a
/// unit axis.
#[test]
fn a_keepdim_outputs_frame_bit_follows_its_unit_axis_stride() {
    for (strides, mask) in [([1, 0], 0b10), ([1, 1], 0b00)] {
        let k = key(
            OpCategory::Reduction,
            &[od(&ROW_SUM_IN, &[128, 1]), od(&[256, 1], &strides)],
        );
        let out = k.operands[1];
        assert_eq!(out.bcast.0, mask, "strides {strides:?}");
        assert_eq!(out.contig, Contiguity::Contig, "strides {strides:?}");
    }
}

/// A bare `[128]` beside a `[256,128]` input is absent from frame axis 0. Since
/// 0.14.0 the key separates it from a full `[256,128]` operand, which 0.11
/// could not do.
#[test]
fn a_bare_row_vector_is_distinguishable_from_a_full_operand() {
    let x = od(&ROW_SUM_IN, &[128, 1]);
    let bare = key(OpCategory::Normalization, &[x, od(&[128], &[1]), x]);
    let full = key(OpCategory::Normalization, &[x, x, x]);
    assert_eq!(bare.operands[1].bcast.0, 0b01);
    assert_eq!(full.operands[1].bcast.0, 0b00);
}
