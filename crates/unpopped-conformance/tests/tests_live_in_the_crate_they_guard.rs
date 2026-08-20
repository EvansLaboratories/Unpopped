//! A test file must exercise the crate it lives in.
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
//! # `unpopped-conformance` is the exception, and it is the reason this file
//!
//! This crate has no library surface. Its whole purpose is holding evidence that
//! must see several crates at once — a dtype matrix spanning both reference
//! emitters is a test of neither. So it is exempt by construction, and the
//! exemption is what makes it the right host: a rule cannot be enforced from
//! inside the one place that legitimately breaks it.

use std::fs;
use std::path::{Path, PathBuf};

/// Crates whose `tests/` legitimately exercise *other* crates. Keep this list
/// short and argued: every entry is a place the guard cannot see.
const NO_LIB_SURFACE: &[&str] = &["unpopped-conformance"];

fn crates_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates/<pkg> has a parent")
        .to_path_buf()
}

/// Test *targets* only: `tests/*.rs`. A `tests/common/` module is a helper
/// compiled into other targets, not a target itself, so it is not expected to
/// name anything.
fn test_targets(krate: &Path) -> Vec<PathBuf> {
    let dir = krate.join("tests");
    let Ok(entries) = fs::read_dir(&dir) else {
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

#[test]
fn every_test_target_names_the_crate_it_lives_in() {
    let mut checked = 0;
    let mut offenders = Vec::new();

    for entry in fs::read_dir(crates_dir()).expect("crates/ is readable") {
        let krate = entry.expect("readable entry").path();
        if !krate.is_dir() {
            continue;
        }
        let name = krate
            .file_name()
            .expect("named dir")
            .to_string_lossy()
            .to_string();
        if NO_LIB_SURFACE.contains(&name.as_str()) {
            continue;
        }
        let lib = name.replace('-', "_");
        for t in test_targets(&krate) {
            let src = fs::read_to_string(&t).expect("test file is readable");
            checked += 1;
            if !mentions(&code_only(&src), &lib) {
                offenders.push(format!(
                    "{}/tests/{} does not use `{lib}`",
                    name,
                    t.file_name().expect("named file").to_string_lossy()
                ));
            }
        }
    }

    assert!(
        checked > 10,
        "only {checked} test targets scanned — the walk found nothing, so a \
         green here would certify nothing"
    );
    assert!(
        offenders.is_empty(),
        "a test target must exercise the crate it lives in, or `cargo test -p \
         <that crate>` silently skips it:\n  {}",
        offenders.join("\n  ")
    );
}

/// The detector discriminates — it is not returning "no offenders" for every
/// input.
///
/// `unpopped-conformance` is exempt precisely because its tests use *other*
/// crates. So its own files must trip the check when it is pointed at them. If
/// this ever goes green the exemption has become vacuous and the guard above is
/// asserting nothing.
#[test]
fn the_detector_fires_on_the_exempt_crate() {
    let me = Path::new(env!("CARGO_MANIFEST_DIR"));
    let targets = test_targets(me);
    assert!(!targets.is_empty(), "this crate has test targets");

    let tripped = targets
        .iter()
        .filter(|t| {
            let src = fs::read_to_string(t).expect("readable");
            !mentions(&code_only(&src), "unpopped_conformance")
        })
        .count();

    assert!(
        tripped > 0,
        "no file in the exempt crate trips the check — the detector cannot \
         distinguish a test that uses its own crate from one that does not"
    );
}
