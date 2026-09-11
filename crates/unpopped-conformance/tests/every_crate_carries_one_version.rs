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

/// The value of `key` in the manifest's **`[package]`** table, or `None`.
///
/// # ⚠️ Two things this does that the obvious version does not
///
/// **It isolates `[package]`.** A manifest has several tables carrying a
/// `version` — `[dependencies]`, `[dev-dependencies]`, and in the workspace root
/// `[workspace.dependencies]`. ⚠️ **A parser that cannot tell which table it is
/// in cannot tell a crate's version from a PIN on that crate, which is this
/// file's entire subject.** The reachable mistake this guard exists to catch —
/// bump the crate version *and* the workspace pin together — is exactly the case
/// where reading the wrong table returns the reassuring answer.
///
/// **It matches the key EXACTLY.** A `starts_with` match lets `version_suffix`
/// or `name_override` satisfy a search for `version`/`name`. ⚠️ **Both failure
/// directions are bad — a false PASS if the wrong key happens to agree, a false
/// FAIL if it does not — and neither names the real cause.**
///
/// Raised as a HIGH finding on PR #7. **Measured: zero such keys exist in this
/// workspace today, so the defect was latent** — which is why the guard was green
/// while being wrong.
fn package_field(text: &str, key: &str) -> Option<String> {
    text.lines()
        .skip_while(|l| l.trim() != "[package]")
        .skip(1)
        .take_while(|l| !l.trim_start().starts_with('['))
        .find(|l| l.split_once('=').is_some_and(|(k, _)| k.trim() == key))
        .and_then(|l| l.split('"').nth(1))
        .map(str::to_string)
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
        if let (Some(name), Some(version)) = (
            package_field(&text, "name"),
            package_field(&text, "version"),
        ) {
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

/// ⚠️ **The parser's two hazards, on SYNTHETIC manifests — because on the real
/// ones the naive parser and the correct one agree, so the defect is LATENT.**
///
/// # How this test came to be written twice
///
/// The first version pointed at the real workspace root: no `[package]` table,
/// a `version = "..."` inside `[workspace.dependencies]`, expect `None`.
/// **Mutation-proving it against the original naive parser: still GREEN.**
///
/// ⚠️ **The naive parser looked for a line STARTING with `version`, and the
/// root's bait sits inside an inline table (`unpopped-vocab = { path = "...",
/// version = "..." }`), so no line starts with it.** The control asserted a
/// property the defective parser also had. **It was a control that could not
/// fire, and only running the mutation said so — the test output was `ok` in
/// both arms, which is exactly what a working control looks like.**
///
/// A latent defect cannot be demonstrated on data that does not exhibit it.
/// **These fixtures exhibit it.**
#[test]
fn the_parser_isolates_the_package_table_and_matches_keys_exactly() {
    // HAZARD 1 -- a `version` line OUTSIDE `[package]`, at the start of a line,
    // which is the ordinary `[workspace.package]` shape.
    let wrong_table = "[workspace.package]
version = \"9.9.9\"
name = \"not-this-one\"

[package]
name = \"right-crate\"
version = \"1.2.3\"
";
    assert_eq!(
        package_field(wrong_table, "version").as_deref(),
        Some("1.2.3"),
        "a `version` in `[workspace.package]` precedes `[package]` in the file;          a parser that does not isolate the table reads 9.9.9 -- a PIN read as          the crate's own version, which is this file's entire subject"
    );
    assert_eq!(
        package_field(wrong_table, "name").as_deref(),
        Some("right-crate")
    );

    // HAZARD 2 -- a key that PREFIX-matches the one being sought.
    let prefix_key = "[package]
version_suffix = \"-alpha\"
name_override = \"wrong\"
version = \"4.5.6\"
name = \"real\"
";
    assert_eq!(
        package_field(prefix_key, "version").as_deref(),
        Some("4.5.6"),
        "`version_suffix` must not satisfy a search for `version`; a prefix          match returns \"-alpha\""
    );
    assert_eq!(package_field(prefix_key, "name").as_deref(), Some("real"));

    // HAZARD 3 -- a `[dependencies]` version must not leak in when `[package]`
    // genuinely lacks the key.
    let no_version = "[package]
name = \"only-a-name\"

[dependencies]
serde = { version = \"1.0\" }
version = \"8.8.8\"
";
    assert_eq!(
        package_field(no_version, "version"),
        None,
        "`[package]` has no version here; anything returned came from a later          table"
    );

    // CONTROL -- the same parser on a REAL manifest still finds real fields, so
    // the three `None`/exact answers above are isolation rather than a parser
    // that never finds anything.
    let real = fs::read_to_string(crates_dir().join("unpopped").join("Cargo.toml"))
        .expect("the unpopped manifest is readable");
    assert_eq!(
        package_field(&real, "name").as_deref(),
        Some("unpopped"),
        "control: the parser must still work on the files it is actually used on"
    );
}
