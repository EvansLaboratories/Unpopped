// SPDX-License-Identifier: MIT OR Apache-2.0
//! The stable core carries no parser technology (ROADMAP U31, ARC-003/ARC-004).
//!
//! The tree-sitter walkers that lift C-family source into the IR live in `unpopped-convert`, which depends on this crate; this
//! crate's manifest must not depend on `tree-sitter` or a grammar again (as it did behind the former `convert` feature), nor
//! name a `convert` feature. The check reads the manifest's non-comment lines, with a control that the same predicate sees a
//! dependency line when one is present.

const MANIFEST: &str = include_str!("../Cargo.toml");

fn names_parser(manifest: &str) -> bool {
    manifest.lines().any(|l| {
        let l = l.trim_start();
        !l.starts_with('#')
            && (l.starts_with("tree-sitter")
                || l.starts_with("convert =")
                || l.contains("dep:tree-sitter"))
    })
}

#[test]
fn the_control_sees_a_parser_dependency() {
    assert!(names_parser(
        "[dependencies]\ntree-sitter = { version = \"0.27\", optional = true }\n"
    ));
    assert!(names_parser(
        "[features]\nconvert = [\"dep:tree-sitter\"]\n"
    ));
    assert!(!names_parser(
        "# tree-sitter was here\n[dependencies]\nserde = \"1\"\n"
    ));
}

#[test]
fn the_core_manifest_names_no_parser_dependency_and_no_convert_feature() {
    assert!(
        !names_parser(MANIFEST),
        "unpopped's manifest names a parser dependency or the convert feature"
    );
}
