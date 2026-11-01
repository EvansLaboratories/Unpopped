// SPDX-License-Identifier: MIT OR Apache-2.0
//! ROADMAP C5 (PM ruling A + D + C, 2026-11-01): a ratchet on hand-written tolerant bands.
//!
//! `oracle::compare_for_plan` derives the band a cell is entitled to, so a test that uses it cannot
//! widen one. This test counts the textual uses of the tolerant `Fidelity` constructor in every `.rs`
//! file of this workspace and fails when a file has more than the table allows, so a NEW hand-written
//! band cannot appear without an entry that states why. It fails the other way too (a file with fewer
//! than the table says), so the table is lowered as the sites migrate and the ratchet only ever
//! tightens. The PM's rule: revisit the breaking `Entitled` type if this ratchet shows callers still
//! loosening (`docs/options-compelled-fidelity.md`, option B).
//!
//! The count is of TEXT, patterns included (a `match` arm on the constructor counts), at the refs the
//! numbers were taken: Unpopped `c030e46` plus this change.

use std::path::{Path, PathBuf};
use std::process::Command;

/// `(file, allowed count, why a hand-written band is legitimate there)`.
const ALLOWED: &[(&str, usize, &str)] = &[
    (
        "crates/unpopped/src/oracle.rs",
        12,
        "the comparator and `required_fidelity` themselves, and their unit tests",
    ),
    (
        "crates/unpopped/tests/compare_for_plan.rs",
        3,
        "the tests of the compelled-band API (a deliberately loose band is the subject)",
    ),
    (
        "crates/unpopped/tests/wide_integer_comparison.rs",
        6,
        "tests of `compare` itself on integers wider than f64's mantissa",
    ),
    (
        "crates/unpopped/tests/a_flushing_device_declines_exact.rs",
        1,
        "a test of the derivation's refusal for a flushing device",
    ),
    (
        "crates/unpopped-cpu-c/tests/required_fidelity.rs",
        10,
        "tests of the derivation (they read the band out and assert on it)",
    ),
    (
        "crates/unpopped-cpu-c/tests/complex_domain.rs",
        5,
        "complex-domain comparisons; to migrate to `compare_for_plan`",
    ),
];

/// The tolerant constructor, assembled so this file does not count itself.
fn needle() -> String {
    ["Fidelity::", "Tolerant {"].concat()
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Every tracked `.rs` file under `crates/`, from the git index; if git is not available (a source
/// tarball), a walk of `crates/` that skips build output. Returns which one ran, so a reader of a
/// failure can tell (a disk walk would see other lanes' worktrees; `crates/` of THIS tree does not).
fn rust_files(root: &Path) -> (Vec<String>, &'static str) {
    if let Ok(out) = Command::new("git")
        .args(["ls-files", "--", "crates"])
        .current_dir(root)
        .output()
    {
        if out.status.success() {
            let files = String::from_utf8_lossy(&out.stdout)
                .lines()
                .filter(|l| l.ends_with(".rs"))
                .map(str::to_string)
                .collect();
            return (files, "git ls-files");
        }
    }
    let mut files = Vec::new();
    walk(&root.join("crates"), root, &mut files);
    (files, "directory walk of crates/")
}

fn walk(dir: &Path, root: &Path, out: &mut Vec<String>) {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    for e in rd.flatten() {
        let p = e.path();
        if p.is_dir() {
            if p.file_name().is_some_and(|n| n == "target") {
                continue;
            }
            walk(&p, root, out);
        } else if p.extension().is_some_and(|x| x == "rs") {
            if let Ok(rel) = p.strip_prefix(root) {
                out.push(rel.to_string_lossy().replace('\\', "/"));
            }
        }
    }
}

#[test]
fn hand_written_tolerant_bands_do_not_grow() {
    let root = workspace_root();
    let (files, how) = rust_files(&root);
    assert!(
        files.iter().any(|f| f == "crates/unpopped/src/oracle.rs"),
        "CONTROL: the enumeration ({how}) must find oracle.rs, or an empty list passes the ratchet"
    );
    let needle = needle();
    let mut problems = Vec::new();
    for f in &files {
        let text = std::fs::read_to_string(root.join(f)).unwrap_or_default();
        let n = text.matches(&needle).count();
        let allowed = ALLOWED
            .iter()
            .find(|(p, _, _)| p == f)
            .map_or(0, |(_, c, _)| *c);
        if n != allowed {
            problems.push(format!(
                "{f}: {n} tolerant constructors, the table allows {allowed}"
            ));
        }
    }
    assert!(
        problems.is_empty(),
        "enumerated by {how}. A NEW hand-written tolerant band needs an `ALLOWED` entry with a reason (or, \
         better, `oracle::compare_for_plan`); fewer than the table says means lower the table:\n{}",
        problems.join("\n")
    );
}
