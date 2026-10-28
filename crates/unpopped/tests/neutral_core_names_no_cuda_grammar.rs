// SPDX-License-Identifier: MIT OR Apache-2.0
//! The neutral core does not name the CUDA grammar.
//!
//! The CUDA front end (`parse_cuda`, the CUDA residue list, the `CUDA` `Frontend`
//! descriptor and the `tree-sitter-cuda` dependency) is owned by Baracuda as
//! `baracuda-cuda-parse`, which depends on this crate: an edge back would be a
//! cycle, so the absence is guarded here by reading the sources. Slang stays, so
//! the same checks must find it: that is the control, without which an empty or
//! misplaced include would pass every "absent" assertion.

const MANIFEST: &str = include_str!("../Cargo.toml");
const CONVERT: &str = include_str!("../src/convert.rs");

#[test]
fn the_controls_find_what_does_stay() {
    assert!(MANIFEST.contains("tree-sitter-slang"), "manifest control");
    assert!(CONVERT.contains("pub fn parse_slang"), "convert.rs control");
    assert!(CONVERT.contains("pub const SLANG"), "convert.rs control");
    assert!(CONVERT.contains("pub fn lift_elementwise"), "walkers stay");
}

#[test]
fn the_manifest_does_not_depend_on_the_cuda_grammar() {
    assert!(!MANIFEST.contains("tree-sitter-cuda"));
}

#[test]
fn convert_rs_does_not_bind_the_cuda_grammar_or_define_the_cuda_frontend() {
    assert!(!CONVERT.contains("tree_sitter_cuda"));
    assert!(!CONVERT.contains("pub fn parse_cuda"));
    assert!(!CONVERT.contains("pub const CUDA"));
    assert!(!CONVERT.contains("const CUDA_RESIDUE"));
}
