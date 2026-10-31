// SPDX-License-Identifier: MIT OR Apache-2.0
//! The neutral core names no grammar.
//!
//! The CUDA front end (`parse_cuda`, the CUDA residue list, the `CUDA` `Frontend`
//! descriptor and the `tree-sitter-cuda` dependency) is owned by Baracuda as
//! `baracuda-cuda-parse`, and the Slang one (`parse_slang`, the Slang residue list, the `SLANG`
//! descriptor and the `tree-sitter-slang` dependency) by Vulkane as `vulkane-slang-parse`. Both
//! depend on this crate: an edge back would be a cycle, so the absence is guarded here by reading
//! the sources.
//!
//! The Slang grammar is still a **dev**-dependency (the tests build their own `Frontend` from it),
//! so the Slang checks read only the shipped part of each file, and their controls prove the split
//! lands where it should: the dev part must contain what the shipped part must not.

const MANIFEST: &str = include_str!("../Cargo.toml");
const CONVERT: &str = include_str!("../src/convert.rs");

/// The manifest before `[dev-dependencies]`, and the dev part.
fn manifest_parts() -> (&'static str, &'static str) {
    MANIFEST
        .split_once("[dev-dependencies]")
        .expect("the manifest has a dev-dependencies table")
}

/// `convert.rs` before its test module, and the test module.
fn convert_parts() -> (&'static str, &'static str) {
    CONVERT
        .split_once("#[cfg(test)]\nmod tests")
        .or_else(|| CONVERT.split_once("#[cfg(test)]\r\nmod tests"))
        .expect("convert.rs has a test module")
}

#[test]
fn the_controls_find_what_does_stay() {
    let (shipped_manifest, dev_manifest) = manifest_parts();
    let (shipped, tests) = convert_parts();
    assert!(
        dev_manifest.contains("tree-sitter-slang"),
        "the Slang grammar is a dev-dependency"
    );
    assert!(
        shipped_manifest.contains("tree-sitter ="),
        "the walker's parser stays"
    );
    assert!(
        tests.contains("tree_sitter_slang"),
        "the tests bind the grammar"
    );
    assert!(
        tests.contains("fn parse_slang"),
        "the tests define the Slang frontend"
    );
    assert!(shipped.contains("pub fn lift_elementwise"), "walkers stay");
    assert!(shipped.contains("pub const fn new"), "Frontend::new stays");
}

#[test]
fn the_manifest_does_not_depend_on_the_cuda_grammar() {
    assert!(!MANIFEST.contains("tree-sitter-cuda"));
}

#[test]
fn the_shipped_manifest_does_not_depend_on_the_slang_grammar() {
    let (shipped, _) = manifest_parts();
    assert!(!shipped.contains("dep:tree-sitter-slang"));
    assert!(
        !shipped
            .lines()
            .any(|l| l.trim_start().starts_with("tree-sitter-slang")),
        "tree-sitter-slang is not a normal dependency"
    );
}

#[test]
fn convert_rs_does_not_bind_a_grammar_or_define_a_frontend() {
    assert!(!CONVERT.contains("tree_sitter_cuda"));
    assert!(!CONVERT.contains("pub fn parse_cuda"));
    assert!(!CONVERT.contains("pub const CUDA"));
    assert!(!CONVERT.contains("const CUDA_RESIDUE"));
    let (shipped, _) = convert_parts();
    assert!(!shipped.contains("tree_sitter_slang"));
    assert!(!shipped.contains("pub fn parse_slang"));
    assert!(!shipped.contains("pub const SLANG"));
    assert!(!shipped.contains("const SLANG_RESIDUE"));
}
