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

/// The malformed fragments of `dir` (name and what is wrong) and how many fragments it holds.
fn scan(dir: &std::path::Path) -> Option<(Vec<(String, &'static str)>, usize)> {
    let entries = fs::read_dir(dir).ok()?;
    let (mut bad, mut seen) = (Vec::new(), 0);
    for e in entries.flatten() {
        let path = e.path();
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_string();
        if !name.ends_with(".md") || name.eq_ignore_ascii_case("readme.md") {
            continue;
        }
        seen += 1;
        let text = fs::read_to_string(&path).unwrap_or_else(|e| panic!("{name}: {e}"));
        if let Some(why) = problem(&text) {
            bad.push((name, why));
        }
    }
    Some((bad, seen))
}

#[test]
fn the_scan_finds_a_malformed_fragment_in_a_directory_and_skips_the_readme() {
    // the control for the real scan below: a directory with one good, one bad fragment and a README
    let dir = std::env::temp_dir().join(format!("unpopped-fragments-{}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    fs::write(
        dir.join("good.md"),
        "kind: ok
---
Body.
",
    )
    .unwrap();
    fs::write(
        dir.join("bad.md"),
        "kind: no separator
Body.
",
    )
    .unwrap();
    fs::write(
        dir.join("README.md"),
        "# not a fragment
",
    )
    .unwrap();
    let (bad, seen) = scan(&dir).expect("the directory exists");
    fs::remove_dir_all(&dir).ok();
    assert_eq!(seen, 2, "the README is not a fragment");
    assert_eq!(bad.len(), 1);
    assert_eq!(bad[0].0, "bad.md");
}

#[test]
fn every_fragment_in_changelog_d_is_well_formed() {
    let dir: PathBuf = [env!("CARGO_MANIFEST_DIR"), "..", "..", "changelog.d"]
        .iter()
        .collect();
    // a packaged copy of the crate has no workspace root: nothing to scan there (the rule's own tests above still ran)
    let Some((bad, seen)) = scan(&dir) else {
        eprintln!(
            "changelog.d not found at {}: not a workspace checkout",
            dir.display()
        );
        return;
    };
    eprintln!("checked {seen} fragment(s)");
    assert!(
        bad.is_empty(),
        "malformed changelog fragment(s), see changelog.d/README.md (tools/stamp-version.py would die on them): {bad:?}"
    );
}
