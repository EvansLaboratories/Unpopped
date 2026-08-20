//! A test file must exercise the crate it lives in — unless it exercises no
//! crate at all.
//!
//! # The defect this closes, which was four files deep
//!
//! `neutral_spelling.rs` pinned that `unpopped::cfamily` — the module documented
//! as "deliberately backend-neutral" — spells no vendor type outside its two
//! known CUDA arms. It lived in `unpopped-cpu-c/tests/` and imported **nothing**
//! from that crate. So `cargo test -p unpopped` did not run the guard on
//! `unpopped`'s own neutrality. Three more sat beside it (`fp8_codecs`,
//! `sub_byte_packing`, `wide_integer_comparison`), all testing
//! `unpopped::oracle`.
//!
//! Measured: `cargo test -p unpopped` was **7 suites / 433 tests** and became
//! **11 / 455** on moving them. Workspace totals did not change — nothing was
//! gained, 22 tests simply started running where their property lives.
//!
//! It is a quiet failure in the way that matters: every suite is green, the
//! workspace count is right, and the only symptom is that a contributor
//! iterating on one crate — or a consumer who depends on it alone — is measuring
//! less than they think.
//!
//! # Why this is a test and not a review habit
//!
//! Nothing about moving code from one crate to another forces its test to
//! follow. The four here were all left behind by the emitter split, which moved
//! *emitters* out of core and had no reason to look at what the remaining tests
//! were testing. The next split will do the same thing.
//!
//! # The two exemptions are measured, not listed
//!
//! An earlier cut of this file exempted `unpopped-conformance` **by name**. Fuel
//! pointed out why that is the wrong axis: a crate-keyed allowlist is correct
//! only until a repo-policy test lands in an ordinary crate, and it says nothing
//! about *why* the file is exempt. Their formulation — **does the file walk the
//! tree, or does it test its host crate?** — keys on what the file does, so it
//! cannot go stale when a file moves. Both exemptions below are properties this
//! test measures:
//!
//! - **Names no workspace crate at all.** Then it is not misplaced, it is a
//!   *repo-policy gate*: its subject is the tree. This very file is one.
//! - **Host crate has no library surface.** Then the crate exists to hold
//!   evidence spanning several crates, and naming another crate is the point.
//!   Measured as zero `pub` items in `src/lib.rs`, not as a name on a list — a
//!   test-host crate still carries a `lib.rs` (this one carries twenty lines of
//!   prose explaining why the crate exists), so testing for the *file* finds
//!   nothing. That is the trap that made a portfolio-wide sweep report zero
//!   exempt crates.
//!
//! Each exemption is asserted to still **fire** somewhere. A rule that matches
//! nothing is dead code claiming to do work, and it would leave this guard green
//! while asserting less than it says.
//!
//! # Why the no-lib crate hosts the guard
//!
//! A rule cannot be enforced from inside the one place that legitimately breaks
//! it.

use std::fs;
use std::path::{Path, PathBuf};

fn crates_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates/<pkg> has a parent")
        .to_path_buf()
}

fn crate_dirs() -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = fs::read_dir(crates_dir())
        .expect("crates/ is readable")
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.join("Cargo.toml").is_file())
        .collect();
    v.sort();
    v
}

fn crate_name(krate: &Path) -> String {
    krate
        .file_name()
        .expect("named dir")
        .to_string_lossy()
        .to_string()
}

/// Test *targets* only: `tests/*.rs`. A `tests/common/` module is a helper
/// compiled into other targets, not a target itself, so it is not expected to
/// name anything.
fn test_targets(krate: &Path) -> Vec<PathBuf> {
    let Ok(entries) = fs::read_dir(krate.join("tests")) else {
        return Vec::new();
    };
    let mut out: Vec<PathBuf> = entries
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.is_file() && p.extension().is_some_and(|x| x == "rs"))
        .collect();
    out.sort();
    out
}

/// A crate with no `pub` item in `src/lib.rs` exists to hold tests, not API.
fn has_lib_surface(krate: &Path) -> bool {
    match fs::read_to_string(krate.join("src/lib.rs")) {
        Ok(src) => code_only(&src)
            .lines()
            .any(|l| l.trim_start().starts_with("pub ")),
        Err(_) => krate.join("src/main.rs").is_file(),
    }
}

/// Strip `//`-comments so a crate named only in prose does not count as use.
///
/// Load-bearing: `neutral_spelling.rs` mentioned `CpuC` twice, both times in a
/// doc comment. Without this the file would have looked like it used the crate
/// it sat in, and the defect would have been invisible to exactly this check.
fn code_only(src: &str) -> String {
    src.lines()
        .map(|l| match l.find("//") {
            Some(i) => &l[..i],
            None => l,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn mentions(code: &str, lib: &str) -> bool {
    code.match_indices(lib).any(|(i, _)| {
        let before = code[..i].chars().next_back();
        let after = code[i + lib.len()..].chars().next();
        let ident = |c: Option<char>| c.is_some_and(|c| c.is_alphanumeric() || c == '_');
        !ident(before) && !ident(after)
    })
}

/// Why a given test target is, or is not, required to name its own crate.
#[derive(Debug, PartialEq, Eq)]
enum Verdict {
    NamesItsOwnCrate,
    /// Exempt: its subject is the tree, not a crate.
    RepoPolicyGate,
    /// Exempt: the host crate has no API to test.
    HostHasNoLibSurface,
    Misplaced,
}

fn classify(krate: &Path, target: &Path, all_libs: &[String]) -> Verdict {
    let code = code_only(&fs::read_to_string(target).expect("test file is readable"));
    let own = crate_name(krate).replace('-', "_");
    if mentions(&code, &own) {
        Verdict::NamesItsOwnCrate
    } else if !all_libs.iter().any(|l| mentions(&code, l)) {
        Verdict::RepoPolicyGate
    } else if has_lib_surface(krate) {
        Verdict::Misplaced
    } else {
        Verdict::HostHasNoLibSurface
    }
}

fn survey() -> Vec<(String, String, Verdict)> {
    let dirs = crate_dirs();
    let libs: Vec<String> = dirs
        .iter()
        .map(|d| crate_name(d).replace('-', "_"))
        .collect();
    let mut out = Vec::new();
    for krate in &dirs {
        for t in test_targets(krate) {
            let file = t
                .file_name()
                .expect("named file")
                .to_string_lossy()
                .to_string();
            out.push((crate_name(krate), file, classify(krate, &t, &libs)));
        }
    }
    out
}

#[test]
fn every_test_target_names_the_crate_it_lives_in() {
    let survey = survey();
    assert!(
        survey.len() > 10,
        "only {} test targets scanned — the walk found nothing, so a green here \
         would certify nothing",
        survey.len()
    );

    let misplaced: Vec<String> = survey
        .iter()
        .filter(|(.., v)| *v == Verdict::Misplaced)
        .map(|(k, f, _)| format!("{k}/tests/{f} names another crate but not `{k}`"))
        .collect();

    assert!(
        misplaced.is_empty(),
        "a test target must exercise the crate it lives in, or `cargo test -p \
         <that crate>` silently skips it:\n  {}",
        misplaced.join("\n  ")
    );
}

/// Both exemptions still match something.
///
/// An exemption that matches nothing is dead code claiming to do work: the guard
/// above stays green while quietly checking a narrower rule than it states.
/// Same argument as preferring `#[expect]` to `#[allow]` — a suppression that
/// stops being needed should go red rather than sit there.
#[test]
fn each_exemption_is_still_exercised() {
    let survey = survey();
    for want in [Verdict::RepoPolicyGate, Verdict::HostHasNoLibSurface] {
        assert!(
            survey.iter().any(|(.., v)| *v == want),
            "no test target is exempt as {want:?} — that branch matches nothing, \
             so the guard is asserting a narrower rule than it claims"
        );
    }
}
