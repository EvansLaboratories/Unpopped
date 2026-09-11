//! **Every crate in this workspace must carry the SAME version number.**
//!
//! # The rule, quoted rather than paraphrased
//!
//! CireSnave, standing: *"I always want all crates within a project to use the
//! same version number so developers consuming them know which go with which
//! with one exception where a crate in one project should keep a version that
//! matches another project such as an emitter crate in Baracuda designed to work
//! with Unpopped keeping its version matching the version of Unpopped it works
//! with."*
//!
//! ⚠️ **The exception is CROSS-PROJECT and cannot reach anything here.** The
//! clause after *"so"* is the rule's test: a number earns the exception by
//! answering *"which Unpopped does this work with"* from **another** project.
//! `unpopped-cpu-c` and `unpopped-slang` are inside this one, so their numbers
//! answer nothing. The crate the exception was written for is
//! `baracuda-cuda-emit`, and it lives in Baracuda.
//!
//! # Why a test and not a convention
//!
//! ⚠️ **PR #5 originally carried `0.11.0` / `0.10.0` / `0.8.0` / `0.4.0` — four
//! different numbers — and every gate was green.** It bumped three crates by one
//! minor each, faithfully preserving offsets that already violated the rule. **A
//! release that preserves a rule violation still violates it**, and nothing in
//! the repository could say so: the rule lived in a message.
//!
//! **`unpopped-conformance` is `publish = false` and moves anyway.** Its version
//! is inert — no external consumer can read it — so the whole value is that a
//! reader scanning five manifests sees one number. **Leaving it out would make it
//! the only crate requiring an explanation, which is the question the rule
//! closes.**

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

fn crates_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates/<pkg> has a parent")
        .to_path_buf()
}

/// `(crate name, version)` for every member of `crates/`, read from the manifest
/// rather than from `cargo metadata` — the file is what a human edits and what a
/// reviewer sees in a diff.
fn declared_versions() -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    for entry in fs::read_dir(crates_dir()).expect("crates/ is readable") {
        let dir = entry.expect("a readable entry").path();
        let manifest = dir.join("Cargo.toml");
        if !manifest.is_file() {
            continue;
        }
        let text = fs::read_to_string(&manifest).expect("manifest is readable");
        let field = |key: &str| -> Option<String> {
            text.lines()
                .take_while(|l| !l.starts_with("[dependencies"))
                .find(|l| l.trim_start().starts_with(key))
                .and_then(|l| l.split('"').nth(1))
                .map(str::to_string)
        };
        if let (Some(name), Some(version)) = (field("name"), field("version")) {
            out.insert(name, version);
        }
    }
    out
}

#[test]
fn every_crate_in_the_workspace_carries_the_same_version() {
    let versions = declared_versions();

    // Non-vacuity: an empty or one-element map satisfies "all equal" having
    // examined nothing, which is how this kind of test passes for years while
    // its subject drifts.
    assert!(
        versions.len() >= 4,
        "only {} crate manifests parsed, so the extractor is broken rather than \
         the workspace being tiny: {versions:?}",
        versions.len()
    );

    let distinct: std::collections::BTreeSet<&String> = versions.values().collect();
    assert_eq!(
        distinct.len(),
        1,
        "the workspace carries {} distinct version strings, and the standing rule \
         is that it carries ONE so a consumer knows which crates go together.\n  \
         {versions:?}\n\nThe rule's exception is CROSS-PROJECT (an emitter crate \
         in another repo pinned to the Unpopped it works with) and cannot reach \
         anything in here. If a release is in flight, bump all of them in the \
         same commit -- PR #5 shipped three of four and the fourth would have \
         frozen on the old line while every sibling moved.",
        distinct.len()
    );
}

/// ⚠️ The control: this test must be able to SEE a difference.
///
/// Without it, a broken extractor returning one entry — or the same entry five
/// times — satisfies the assertion above having measured nothing. **A test whose
/// subject is "all values are equal" is exactly the shape that passes when the
/// values are missing.**
#[test]
fn the_extractor_reads_distinct_crates_and_a_real_version_each() {
    let versions = declared_versions();

    assert!(
        versions.contains_key("unpopped")
            && versions.contains_key("unpopped-vocab")
            && versions.contains_key("unpopped-cpu-c")
            && versions.contains_key("unpopped-slang"),
        "the four published crates must all be found by name, or the map is not \
         what the assertion above is ranging over: {versions:?}"
    );

    for (name, v) in &versions {
        assert!(
            v.split('.').count() == 3 && v.split('.').all(|p| p.parse::<u32>().is_ok()),
            "{name} parsed as version {v:?}, which is not `x.y.z` -- the \
             extractor picked up the wrong line"
        );
    }
}
