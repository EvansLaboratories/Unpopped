//! The `seam` feature's entry point, which nothing else in this workspace runs.
//!
//! # Why this file exists
//!
//! `unpopped::jit::seam` is gated behind `--features seam`, off by default. Until
//! 2026-09-02 that meant it was never COMPILED by CI, never linted, and had zero
//! tests — while being the function Fuel and Baracuda actually call. A vacuous-pass
//! sweep found it by a count that did not add up: 734 `#[test]` functions exist in
//! this workspace and 712 ran.
//!
//! That is the worst place in a repository for a hole. A breaking change to
//! `fuel_kernel_seam_types`' frozen grammar, or a refactor of `region_to_op`, would
//! have landed green here and failed in a consumer's tree — where it costs them a
//! debugging session to discover it was ours.
//!
//! # What it covers
//!
//! The seam's own logic, which is a CONVERSION and not a passthrough: it walks
//! Fuel's `PatternNode` into our internal node form, derives the op, and hands off
//! to the same `synthesize_op` the native path uses. So the happy path proves the
//! walk reaches core synthesis, a nested region proves the walk RECURSES, and the
//! declines prove the guards are the seam's own rather than something downstream.
//!
//! Deliberately uses `StubBackend`/`StubCompiler`: what is under test is the
//! grammar conversion, not any target's spelling. A real emitter here would test
//! two things and blame the wrong one.

#![cfg(feature = "seam")]

mod common;

use common::StubBackend;
use fuel_kernel_seam_types::{OpAttrs, OpTag, PatternNode};
use unpopped::jit::{JitError, StubCompiler, seam};
use unpopped_vocab::{ArchSku, ElementKind, OpCategory, OperandDesc};

/// `Add(bind0, bind1)` — the smallest region Fuel can hand us.
fn add_region() -> PatternNode {
    PatternNode::Op {
        op: OpTag::Add,
        operands: vec![
            PatternNode::Bind { index: 0 },
            PatternNode::Bind { index: 1 },
        ],
        attrs: OpAttrs::default(),
    }
}

fn f32_operand() -> OperandDesc {
    OperandDesc::new(1, &[256], &[1], ElementKind::F32, 4)
}

/// Inputs-then-output, per the seam's documented projection.
fn operands(n: usize) -> Vec<OperandDesc> {
    vec![f32_operand(); n]
}

fn call(
    region: &PatternNode,
    ops: &[OperandDesc],
    budget: u32,
) -> Result<unpopped::jit::JitResponse, JitError> {
    seam::synthesize(
        region,
        ops,
        OpCategory::BinaryElementwise,
        ArchSku::Sm89,
        "fused_add",
        1_000 * u32::from(budget > 0) + budget,
        &StubBackend,
        &StubCompiler,
    )
}

#[test]
fn a_fuel_region_reaches_core_synthesis_and_comes_back_with_a_kernel() {
    // 2 inputs + 1 output.
    let out = seam::synthesize(
        &add_region(),
        &operands(3),
        OpCategory::BinaryElementwise,
        ArchSku::Sm89,
        "fused_add",
        1_000,
        &StubBackend,
        &StubCompiler,
    )
    .expect("an Add over two binds is the simplest region the seam accepts");

    // Not just "it returned Ok" — the four halves of the response must all be
    // populated, because a consumer that cannot link the kernel has not been
    // given one. `link` in particular is what makes an adopted kernel bindable.
    assert!(
        !out.kernel.entry_point.is_empty(),
        "a kernel with no entry-point symbol cannot be bound"
    );
    assert!(
        !out.contract.is_empty(),
        "the FKC contract is the half a consumer validates against"
    );
    assert!(
        !out.link.entry_point.is_empty(),
        "the link_registry row resolves the entry point at load; empty is unbindable"
    );
}

#[test]
fn the_walk_recurses_rather_than_reading_only_the_root() {
    // Mul(Add(b0, b1), b2). A conversion that handled only the root node would
    // return Ok on the flat case above and fail here — which is the whole
    // reason the flat case is not sufficient on its own.
    let nested = PatternNode::Op {
        op: OpTag::Mul,
        operands: vec![add_region(), PatternNode::Bind { index: 2 }],
        attrs: OpAttrs::default(),
    };

    let out = seam::synthesize(
        &nested,
        &operands(4), // 3 inputs + 1 output
        OpCategory::BinaryElementwise,
        ArchSku::Sm89,
        "fused_mul_add",
        1_000,
        &StubBackend,
        &StubCompiler,
    )
    .expect("a two-level region must convert; the seam's job is the whole tree");

    assert!(
        !out.kernel.source.is_empty(),
        "a nested region must still produce source, not an empty shell"
    );
}

#[test]
fn an_empty_operand_list_declines_as_arity_rather_than_panicking() {
    match call(&add_region(), &[], 1_000) {
        Err(JitError::OperandArity { .. }) => {}
        other => panic!(
            "the seam must reject an empty projection as an ARITY error the caller \
             can act on, not by indexing operands[0]; got {other:?}"
        ),
    }
}

#[test]
fn a_zero_budget_declines_as_budget_rather_than_compiling_forever() {
    match call(&add_region(), &operands(3), 0) {
        Err(JitError::Budget(_)) => {}
        other => panic!("a zero compile budget must be a typed Budget decline; got {other:?}"),
    }
}

#[test]
fn a_mixed_dtype_projection_declines_at_the_seam() {
    // The seam takes dtype from operands[0] and requires the rest to agree, so a
    // disagreement must be caught HERE rather than silently adopting the first.
    let mut ops = operands(3);
    ops[1] = OperandDesc::new(1, &[256], &[1], ElementKind::I32, 4);

    match call(&add_region(), &ops, 1_000) {
        Err(JitError::MixedDtype) => {}
        other => panic!(
            "a mixed-dtype projection must decline rather than adopt operands[0]'s \
             dtype and emit a kernel that reads the others wrong; got {other:?}"
        ),
    }
}
