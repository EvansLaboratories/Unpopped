// SPDX-License-Identifier: MIT OR Apache-2.0
//! **No source, test, doc or manifest file in this repository contains a NUL byte.**
//!
//! A NUL in a text file makes git and every diff tool call it binary, and in a Rust string literal it
//! is a raw control byte standing where the author meant the two-character escape for zero. This
//! happened: a backslash-zero typed through a tool that interprets it shipped as a raw NUL in
//! `contract.rs` and `CHANGELOG.md` in 0.37.1, and compiled fine, which is why nothing caught it.
//!
//! The scan is deliberately crude (any 0 byte, in every file of the directories below) so it is hard
//! to fool and cheap to read. Two controls keep it honest: it must find a real population of files,
//! and its matcher must fire on a buffer that has a NUL in it.

use std::path::{Path, PathBuf};

/// Directories under the workspace root whose text files are scanned, and the file extensions.
const DIRS: &[&str] = &["crates", "docs", ".github"];
const EXTENSIONS: &[&str] = &["rs", "md", "toml", "yml", "yaml", "json", "txt"];

fn has_nul(bytes: &[u8]) -> bool {
    bytes.contains(&0u8)
}

fn files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name();
        if path.is_dir() {
            // Build output and VCS internals are not the repository's text.
            if name != "target" && name != ".git" && name != "target-msrv" {
                files(&path, out);
            }
        } else if path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| EXTENSIONS.contains(&e))
        {
            out.push(path);
        }
    }
}

#[test]
fn the_matcher_fires_on_a_nul() {
    assert!(has_nul(&[b'a', 0u8, b'b']), "the control: a NUL is seen");
    assert!(!has_nul(b"plain text"));
}

#[test]
fn no_text_file_contains_a_nul_byte() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("the workspace root");
    let mut all = Vec::new();
    for d in DIRS {
        files(&root.join(d), &mut all);
    }
    for top in std::fs::read_dir(root).expect("read the root").flatten() {
        let p = top.path();
        if p.is_file()
            && p.extension()
                .is_some_and(|e| EXTENSIONS.contains(&e.to_str().unwrap_or("")))
        {
            all.push(p);
        }
    }
    // The population control: a moved root would otherwise scan nothing.
    assert!(
        all.len() > 100,
        "found only {} files; the scan is not looking at the repository",
        all.len()
    );
    let offenders: Vec<String> = all
        .iter()
        .filter(|f| std::fs::read(f).is_ok_and(|b| has_nul(&b)))
        .map(|f| f.display().to_string())
        .collect();
    assert!(
        offenders.is_empty(),
        "NUL bytes in text files (a mistyped backslash-zero escape?): {offenders:#?}"
    );
}
