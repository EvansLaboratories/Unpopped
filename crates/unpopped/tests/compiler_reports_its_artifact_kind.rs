// SPDX-License-Identifier: MIT OR Apache-2.0
//! A `Compiler` for another target reports its own `ArtifactKind` (0.38.0).
//!
//! Rockfish's hipRTC compiler (R7d) returns code-object bytes and overrides
//! `artifact_kind()`; the kind has to reach the `SynthKernel` unchanged, because a loader decides
//! what to do with the bytes from it. `ArtifactKind` and `SynthKernel` are `#[non_exhaustive]`,
//! so the match below needs its wildcard arm: that is the migration the CHANGELOG states.

mod common;

use common::StubBackend;
use unpopped::jit::{
    ArtifactKind, Compiler, JitBudget, JitRequest, StubCompiler, SynthKernel, synthesize,
};
use unpopped::pattern::PatternNode;
use unpopped_vocab::{ElementKind, OpCategory, OperandDesc, TargetId};

struct Reports(ArtifactKind);

impl Compiler for Reports {
    fn compile(&self, _: &str, entry: &str, _: u32) -> Result<Vec<u8>, String> {
        Ok(entry.as_bytes().to_vec())
    }
    fn artifact_kind(&self) -> ArtifactKind {
        self.0
    }
}

fn request() -> JitRequest {
    let d = OperandDesc::new(1, &[256], &[1], ElementKind::F32, 4);
    JitRequest {
        region: PatternNode::Op {
            op: "Add".to_string(),
            operands: vec![PatternNode::Bind(0), PatternNode::Bind(1)],
            consumers: None,
            extract: Vec::new(),
        },
        n_inputs: 2,
        op_category: OpCategory::BinaryElementwise,
        operands: vec![d; 3],
        target: TargetId::parse("cuda:sm89").expect("a hardcoded token is well-formed"),
        fused_op_id: "fused_add".to_string(),
        budget: JitBudget {
            max_compile_ms: 1_000,
        },
    }
}

/// What a loader does: accept only what it can load, refuse the rest, including a kind added later.
fn loadable_on_amd(k: ArtifactKind) -> bool {
    match k {
        ArtifactKind::HipCodeObject => true,
        ArtifactKind::Ptx | ArtifactKind::Cubin | ArtifactKind::SpirV | ArtifactKind::Stub => false,
        _ => false,
    }
}

#[test]
fn the_compilers_kind_reaches_the_synth_kernel() {
    for kind in [
        ArtifactKind::Ptx,
        ArtifactKind::Cubin,
        ArtifactKind::HipCodeObject,
        ArtifactKind::SpirV,
    ] {
        let out = synthesize(&request(), &StubBackend, &Reports(kind)).expect("synthesizes");
        assert_eq!(out.kernel.kind, kind);
        assert_eq!(
            loadable_on_amd(out.kernel.kind),
            kind == ArtifactKind::HipCodeObject
        );
    }
}

#[test]
fn the_default_kind_stays_ptx_and_the_stub_stays_a_stub() {
    struct Plain;
    impl Compiler for Plain {
        fn compile(&self, _: &str, _: &str, _: u32) -> Result<Vec<u8>, String> {
            Ok(vec![])
        }
    }
    let out = synthesize(&request(), &StubBackend, &Plain).expect("synthesizes");
    assert_eq!(out.kernel.kind, ArtifactKind::Ptx);
    let stub = synthesize(&request(), &StubBackend, &StubCompiler).expect("synthesizes");
    assert_eq!(stub.kernel.kind, ArtifactKind::Stub);
}

#[test]
fn a_synth_kernel_is_built_by_new() {
    let k = SynthKernel::new("e".into(), "s".into(), vec![1, 2], ArtifactKind::SpirV);
    assert_eq!((k.entry_point.as_str(), k.artifact.len()), ("e", 2));
}
