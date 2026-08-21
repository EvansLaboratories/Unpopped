//! The workflow must not restate a version that already lives somewhere else.
//!
//! Two versions govern this workspace and they are different claims:
//!
//! ```text
//! rust-toolchain.toml  channel = "1.98.0"   what this workspace is VERIFIED with
//! Cargo.toml           rust-version = "…"   what a CONSUMER may build with
//! ```
//!
//! Once CI exists, each could be restated in the workflow — and a duplicated
//! fact with no gate on the duplication is a scheduled divergence. MLMF hit this
//! and gated the agreement (`toolchain_pin_matches_ci`), which is the right move
//! when the duplication is unavoidable.
//!
//! **Here it was avoidable, so it was removed rather than gated**, and this test
//! is what keeps it removed:
//!
//! - the `verify` job names **no** toolchain at all — `rustup` reads
//!   `rust-toolchain.toml`, so the pin has exactly one home;
//! - the `msrv` job **reads** `rust-version` out of `Cargo.toml` at run time
//!   rather than restating it, so the thing under test and the thing declared
//!   cannot drift.
//!
//! A gate on an agreement proves two copies still match. This proves there is
//! only one copy — which is the stronger property and the cheaper one, and it
//! stops being true the moment somebody pastes a version into the workflow "to
//! be explicit".
//!
//! # Why a test rather than a review habit
//!
//! Pasting a literal version into a CI file is the single most natural edit
//! anyone will ever make to it. Nothing about doing so fails, and the divergence
//! surfaces months later as "CI passes and my machine doesn't".

use std::fs;
use std::path::{Path, PathBuf};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crates/<pkg> sits two below the root")
        .to_path_buf()
}

fn workflows() -> Vec<(String, String)> {
    let dir = repo_root().join(".github/workflows");
    let mut out: Vec<(String, String)> = fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("no workflows at {}: {e}", dir.display()))
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "yml" || x == "yaml"))
        .map(|p| {
            let name = p.file_name().expect("named").to_string_lossy().to_string();
            (name, fs::read_to_string(&p).expect("readable"))
        })
        .collect();
    out.sort();
    out
}

/// A literal Rust release, e.g. `1.98.0` or `1.85` — not a date, not a version
/// of something else. Deliberately narrow: an action pin like `checkout@v4` and
/// an edition like `2024` must not trip it.
fn literal_rust_versions(src: &str) -> Vec<String> {
    let mut found = Vec::new();
    for line in src.lines() {
        // Comments explain the pin and are allowed to name it; code must not.
        let code = match line.find('#') {
            Some(i) => &line[..i],
            None => line,
        };
        let bytes: Vec<char> = code.chars().collect();
        let mut i = 0;
        while i < bytes.len() {
            if bytes[i] == '1' && bytes.get(i + 1) == Some(&'.') {
                let rest: String = bytes[i..].iter().collect();
                let tok: String = rest
                    .chars()
                    .take_while(|c| c.is_ascii_digit() || *c == '.')
                    .collect();
                let parts: Vec<&str> = tok.split('.').filter(|p| !p.is_empty()).collect();
                // `1.85` or `1.98.0`, with a plausible minor — enough to be a
                // Rust release rather than a stray decimal.
                if (2..=3).contains(&parts.len())
                    && parts[1].parse::<u32>().is_ok_and(|m| m >= 50)
                    && !tok.ends_with('.')
                {
                    found.push(tok);
                }
            }
            i += 1;
        }
    }
    found
}

#[test]
fn no_workflow_restates_a_rust_version_that_lives_elsewhere() {
    let wfs = workflows();
    assert!(
        !wfs.is_empty(),
        "no workflows found — this test scans nothing"
    );

    let mut offenders = Vec::new();
    for (name, src) in &wfs {
        for v in literal_rust_versions(src) {
            offenders.push(format!(
                ".github/workflows/{name} names Rust {v} in an executable line"
            ));
        }
    }

    assert!(
        offenders.is_empty(),
        "a workflow restated a version that already has a home. The pin lives in \
         rust-toolchain.toml (rustup reads it — name no toolchain), and the MSRV \
         lives in Cargo.toml `rust-version` (read it at run time). Two copies of \
         one fact diverge silently:\n  {}",
        offenders.join("\n  ")
    );
}

/// The detector fires. Without this, the test above passes just as happily
/// against a `literal_rust_versions` that returns nothing for every input — and
/// it is a hand-rolled scanner, which is exactly the kind that quietly stops
/// matching.
#[test]
fn the_version_detector_fires_on_a_pasted_pin() {
    let sabotage = "      - uses: dtolnay/rust-toolchain@1.98.0\n";
    assert_eq!(
        literal_rust_versions(sabotage),
        vec!["1.98.0".to_string()],
        "the detector missed a pasted toolchain pin"
    );

    let msrv_paste = "      - run: cargo +1.85 test --workspace\n";
    assert_eq!(
        literal_rust_versions(msrv_paste),
        vec!["1.85".to_string()],
        "the detector missed a pasted MSRV"
    );

    // And it must NOT fire on the things that legitimately look like versions.
    for benign in [
        "      - uses: actions/checkout@v4\n",
        "        edition = \"2024\"\n",
        "        # pinned to 1.98.0 in rust-toolchain.toml\n",
        "        threshold: 0.85\n",
    ] {
        assert!(
            literal_rust_versions(benign).is_empty(),
            "false positive on {benign:?} — a detector that flags comments and \
             action pins gets switched off"
        );
    }
}
