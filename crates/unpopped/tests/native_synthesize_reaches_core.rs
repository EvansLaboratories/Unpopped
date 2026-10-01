//! The native (`JitRequest`) entry point into synthesis — happy path,
//! recursion, and the trust-boundary declines.
//!
//! # Why this file exists
//!
//! Before the `fuel-kernel-seam-types` decoupling, `unpopped::jit::seam`
//! (Fuel's `PatternNode`, `--features seam`) was the ONLY tested way into
//! `region_to_op` + `synthesize_op` — `tests/seam_reaches_core_synthesis.rs`
//! covered the happy path, recursion, and the arity/budget/mixed-dtype
//! declines, and its own header explains why that mattered: this exact path
//! is what Fuel and Baracoda actually call, and a vacuous-pass sweep once
//! found it uncompiled and untested (712 of 734 tests ran).
//!
//! Removing the `seam` module (the decoupling: unpopped's public API no
//! longer names `fuel_kernel_seam_types::PatternNode`, and the crate drops
//! that dependency entirely) removes that file's premise along with it — but
//! `synthesize(&JitRequest, ..)`, the ONLY entry point left, had never been
//! called directly by ANY test in this crate. Every test of this path was
//! reached only through the seam wrapper. Deleting the seam tests without
//! this file would silently recreate the exact hole they were written to
//! close, just against the native API instead of the Fuel one.
//!
//! Mirrors `seam_reaches_core_synthesis.rs`'s five cases against the native
//! [`JitRequest`] shape, so the coverage transfers rather than disappearing.

mod common;

use common::StubBackend;
use unpopped::jit::{JitBudget, JitError, JitRequest, StubCompiler, synthesize};
use unpopped::pattern::PatternNode;
use unpopped_vocab::{ElementKind, OpCategory, OperandDesc, TargetId};

/// `Add(bind0, bind1)` — the smallest region synthesis can accept.
fn add_region() -> PatternNode {
    PatternNode::Op {
        op: "Add".to_string(),
        operands: vec![PatternNode::Bind(0), PatternNode::Bind(1)],
        consumers: None,
        extract: Vec::new(),
    }
}

fn f32_operand() -> OperandDesc {
    OperandDesc::new(1, &[256], &[1], ElementKind::F32, 4)
}

/// Inputs-then-output, matching [`JitRequest::operands`]'s documented shape.
fn operands(n: usize) -> Vec<OperandDesc> {
    vec![f32_operand(); n]
}

fn request(region: PatternNode, ops: Vec<OperandDesc>, n_inputs: u8, budget: u32) -> JitRequest {
    JitRequest {
        region,
        n_inputs,
        op_category: OpCategory::BinaryElementwise,
        operands: ops,
        target: TargetId::parse("cuda:sm89").expect("a hardcoded token is well-formed"),
        fused_op_id: "fused_add".to_string(),
        budget: JitBudget {
            max_compile_ms: budget,
        },
    }
}

#[test]
fn a_region_reaches_core_synthesis_and_comes_back_with_a_kernel() {
    // 2 inputs + 1 output.
    let req = request(add_region(), operands(3), 2, 1_000);
    let out = synthesize(&req, &StubBackend, &StubCompiler)
        .expect("an Add over two binds is the simplest region synthesize accepts");

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
    // Mul(Add(b0, b1), b2). A conversion/walk that handled only the root node
    // would return Ok on the flat case above and fail here.
    let nested = PatternNode::Op {
        op: "Mul".to_string(),
        operands: vec![add_region(), PatternNode::Bind(2)],
        consumers: None,
        extract: Vec::new(),
    };

    let req = request(nested, operands(4), 3, 1_000); // 3 inputs + 1 output
    let out = synthesize(&req, &StubBackend, &StubCompiler)
        .expect("a two-level region must synthesize; the walk covers the whole tree");

    assert!(
        !out.kernel.source.is_empty(),
        "a nested region must still produce source, not an empty shell"
    );
}

#[test]
fn an_empty_operand_list_declines_as_arity_rather_than_panicking() {
    let req = request(add_region(), vec![], 0, 1_000);
    match synthesize(&req, &StubBackend, &StubCompiler) {
        Err(JitError::OperandArity { .. }) => {}
        other => panic!(
            "an empty projection must be an ARITY error the caller can act on, \
             not a panic on operands[0]; got {other:?}"
        ),
    }
}

#[test]
fn a_zero_budget_declines_as_budget_rather_than_compiling_forever() {
    let req = request(add_region(), operands(3), 2, 0);
    match synthesize(&req, &StubBackend, &StubCompiler) {
        Err(JitError::Budget(_)) => {}
        other => panic!("a zero compile budget must be a typed Budget decline; got {other:?}"),
    }
}

#[test]
fn a_mixed_dtype_projection_declines_at_synthesis() {
    // synthesize takes dtype from operands[0] and requires the rest to agree,
    // so a disagreement must be caught HERE rather than silently adopting the
    // first operand's dtype.
    let mut ops = operands(3);
    ops[1] = OperandDesc::new(1, &[256], &[1], ElementKind::I32, 4);
    let req = request(add_region(), ops, 2, 1_000);

    match synthesize(&req, &StubBackend, &StubCompiler) {
        Err(JitError::MixedDtype) => {}
        other => panic!(
            "a mixed-dtype projection must decline rather than adopt operands[0]'s \
             dtype and emit a kernel that reads the others wrong; got {other:?}"
        ),
    }
}
