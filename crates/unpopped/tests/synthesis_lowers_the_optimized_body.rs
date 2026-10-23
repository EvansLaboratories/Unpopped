//! `synthesize` lowers the OPTIMIZED body, not the one the region spelled.
//!
//! # Why this file exists
//!
//! A mutation audit of `jit.rs` (cargo-mutants, `delete field body from struct
//! OpDef expression in synthesize_op`) survived: with `body: optimize(..)` removed
//! the `..op.clone()` spread silently supplies the UNoptimized body, and every
//! existing test still passed, because they assert a kernel exists and what it is
//! called, never what body it was lowered from. The doc on `synthesize_op` promises
//! the §5.1 inward optimizer runs on the kernel body; nothing held it to that.
//!
//! The probe is a backend that records the body in the plan it is handed, since the
//! plan is the only place the lowered body is observable from outside the crate.

use std::sync::Mutex;

use unpopped::backend::{Backend, DtypeSupport, GeneratedKernel, LowerError};
use unpopped::jit::{JitBudget, JitRequest, StubCompiler, synthesize};
use unpopped::pattern::PatternNode;
use unpopped::plan::KernelPlan;
use unpopped_vocab::{ElementKind, OpCategory, OperandDesc, TargetId};

/// Records `format!("{:?}", plan.body)` for each plan it lowers.
#[derive(Default)]
struct Recorder(Mutex<Vec<String>>);

impl Backend for Recorder {
    fn name(&self) -> &str {
        "recorder"
    }
    fn provider(&self) -> &str {
        "unpopped-test"
    }
    fn supports_dtype(&self, _dtype: ElementKind, _target: TargetId) -> DtypeSupport {
        DtypeSupport::Supported
    }
    fn lower(&self, plan: &KernelPlan<'_>) -> Result<GeneratedKernel, LowerError> {
        self.0
            .lock()
            .expect("single-threaded test")
            .push(format!("{:?}", plan.body));
        Ok(GeneratedKernel::new(
            "recorded".to_string(),
            "/* r */".to_string(),
        ))
    }
}

fn neg(x: PatternNode) -> PatternNode {
    PatternNode::Op {
        op: "Neg".to_string(),
        operands: vec![x],
        consumers: None,
        extract: Vec::new(),
    }
}

fn lowered_body(region: PatternNode) -> String {
    let operand = OperandDesc::new(1, &[256], &[1], ElementKind::F32, 4);
    let req = JitRequest {
        region,
        n_inputs: 1,
        op_category: OpCategory::UnaryElementwise,
        operands: vec![operand, operand],
        target: TargetId::parse("cuda:sm89").expect("a hardcoded token is well-formed"),
        fused_op_id: "fused_neg".to_string(),
        budget: JitBudget {
            max_compile_ms: 1_000,
        },
    };
    let backend = Recorder::default();
    synthesize(&req, &backend, &StubCompiler).expect("a Neg chain over one bind synthesizes");
    let seen = backend.0.lock().expect("single-threaded test").clone();
    assert_eq!(
        seen.len(),
        1,
        "exactly one lowering per synthesis: {seen:?}"
    );
    seen.into_iter().next().expect("length asserted above")
}

#[test]
fn a_double_negation_is_lowered_as_the_bare_input() {
    // Control: a region with nothing to rewrite lowers as spelled, so the probe
    // does see the body (a recorder that always saw `Input(0)` would pass below).
    assert_eq!(
        lowered_body(neg(PatternNode::Bind(0))),
        "Unary(Neg, Input(0))"
    );
    // neg(neg x) -> x is optimize's involution; skipping the optimizer lowers the
    // two-deep spelling instead.
    assert_eq!(lowered_body(neg(neg(PatternNode::Bind(0)))), "Input(0)");
}
