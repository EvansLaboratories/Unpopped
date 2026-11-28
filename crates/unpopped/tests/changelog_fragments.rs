// SPDX-License-Identifier: MIT OR Apache-2.0
//! Every `changelog.d/*.md` fragment is well formed for `tools/stamp-version.py` (Pip's review of the atom series: four fragments
//! lacked the `---` separator and the stamp died on each). The same rule as the stamp's `parse_fragment`: a first line
//! `kind: <text>`, a line `---`, then a non-empty entry. This runs on the PR that adds the fragment, so a malformed one fails
//! there, not at the stamp. The rule is a function of the text, so its own broken fixtures are the control: each is flagged,
//! and a good one is not.

use std::fs;
use std::path::PathBuf;

/// What is wrong with the fragment `text`, or `None` when the stamp accepts it.
fn problem(text: &str) -> Option<&'static str> {
    let text = text.replace("\r\n", "\n");
    let (head, sep, body) = match text.split_once("\n---\n") {
        Some((h, b)) => (h, true, b),
        None => (text.as_str(), false, ""),
    };
    if !sep {
        return Some("no `---` line after the first line");
    }
    let Some(kind) = head.strip_prefix("kind:") else {
        return Some("the first line does not start with `kind:`");
    };
    if kind.trim().is_empty() {
        return Some("an empty `kind:`");
    }
    if kind.contains('\n') {
        return Some(
            "the `kind:` runs over more than one line before the `---` (the heading is one line)",
        );
    }
    if body.trim().is_empty() {
        return Some("an empty entry after the `---`");
    }
    None
}

#[test]
fn the_rule_flags_each_malformed_fragment_and_passes_a_good_one() {
    assert_eq!(problem("kind: a thing\n---\nWhat changed.\n"), None);
    assert_eq!(
        problem("kind: a thing\r\n---\r\nWhat changed.\r\n"),
        None,
        "CRLF"
    );
    for (bad, why) in [
        ("kind: a thing\nWhat changed.\n", "no `---`"),
        ("a thing\n---\nWhat changed.\n", "no `kind:`"),
        ("kind:\n---\nWhat changed.\n", "empty kind"),
        (
            "kind: a thing\nsecond line\n---\nWhat changed.\n",
            "two-line kind",
        ),
        ("kind: a thing\n---\n\n", "empty body"),
    ] {
        assert!(problem(bad).is_some(), "{why}: {bad:?} must be flagged");
    }
}

/// The malformed fragments among `files` (`(file name, text)`), and how many of them are fragments (a `README.md` is not).
fn scan(files: &[(String, String)]) -> (Vec<(String, &'static str)>, usize) {
    let mut bad = Vec::new();
    let mut seen = 0;
    for (name, text) in files {
        if !name.ends_with(".md") || name.eq_ignore_ascii_case("readme.md") {
            continue;
        }
        seen += 1;
        if let Some(why) = problem(text) {
            bad.push((name.clone(), why));
        }
    }
    (bad, seen)
}

/// Every file of `dir` as `(name, text)`; `None` when there is no such directory.
fn read_dir_files(dir: &std::path::Path) -> Option<Vec<(String, String)>> {
    let entries = fs::read_dir(dir).ok()?;
    Some(
        entries
            .flatten()
            .filter_map(|e| {
                let name = e.file_name().to_str()?.to_string();
                let text = fs::read_to_string(e.path()).ok()?;
                Some((name, text))
            })
            .collect(),
    )
}

#[test]
fn the_scan_finds_a_malformed_fragment_and_skips_the_readme() {
    // the control for the real scan below: one good, one bad fragment and a README (any case)
    let file = |n: &str, t: &str| (n.to_string(), t.to_string());
    let (bad, seen) = scan(&[
        file(
            "good.md",
            "kind: ok
---
Body.
",
        ),
        file(
            "bad.md",
            "kind: no separator
Body.
",
        ),
        file(
            "README.md",
            "# not a fragment
",
        ),
        file(
            "readme.md",
            "# also not
",
        ),
        file("notes.txt", "not markdown"),
    ]);
    assert_eq!(
        seen, 2,
        "the READMEs and the non-markdown file are not fragments"
    );
    assert_eq!(bad.len(), 1);
    assert_eq!(bad[0].0, "bad.md");
}

#[test]
fn every_fragment_in_changelog_d_is_well_formed() {
    let dir: PathBuf = [env!("CARGO_MANIFEST_DIR"), "..", "..", "changelog.d"]
        .iter()
        .collect();
    // a packaged copy of the crate has no workspace root: nothing to scan there (the rule's own tests above still ran)
    let Some(files) = read_dir_files(&dir) else {
        eprintln!(
            "changelog.d not found at {}: not a workspace checkout",
            dir.display()
        );
        return;
    };
    let (bad, seen) = scan(&files);
    eprintln!("checked {seen} fragment(s)");
    assert!(
        bad.is_empty(),
        "malformed changelog fragment(s), see changelog.d/README.md (tools/stamp-version.py would die on them): {bad:?}"
    );
}
