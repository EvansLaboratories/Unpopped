// SPDX-License-Identifier: MIT OR Apache-2.0
//! **No source file in this workspace writes a kernel launch directive.**
//!
//! A standing constraint, not a style rule (`docs/e1d-parallel-mapping.md`, "Patent re-read",
//! rule 3; `docs/PATENT-SWEEP.md` hit 10, US10949182B2). Unpopped emits kernel source and
//! *declares* a `LaunchShape` that a manifest reports; the host launches. An emitter that wrote
//! `kernel<<<blocks, threads>>>(...)` or a dispatch call carrying block and thread counts would be
//! doing the one thing the re-read said to stop and re-read the patent before doing.
//!
//! The scan is deliberately crude (substrings, in every `crates/*/src/**/*.rs`) because it must be
//! hard to fool by accident and cheap to read. Two controls keep it honest: it must find a real
//! population of files, and its matcher must fire on a launch directive written the way CUDA
//! writes one.

use std::path::{Path, PathBuf};

/// Spellings of a launch directive in the languages this workspace emits or targets: the CUDA
/// chevrons, the CUDA/HIP driver and runtime launch calls, Vulkan/OpenGL dispatch, and SYCL's
/// `parallel_for`.
const LAUNCH_SPELLINGS: &[&str] = &[
    "<<<",
    "cudaLaunchKernel",
    "cuLaunchKernel",
    "hipLaunchKernel",
    "vkCmdDispatch",
    "dispatchCompute",
    "DispatchThreadGroups",
    "parallel_for",
];

fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).unwrap_or_else(|e| panic!("read {}: {e}", dir.display())) {
        let path = entry.expect("dir entry").path();
        if path.is_dir() {
            rust_files(&path, out);
        } else if path.extension().is_some_and(|x| x == "rs") {
            out.push(path);
        }
    }
}

fn launch_spellings_in(src: &str) -> Vec<&'static str> {
    LAUNCH_SPELLINGS
        .iter()
        .copied()
        .filter(|s| src.contains(s))
        .collect()
}

#[test]
fn the_matcher_fires_on_a_launch_directive() {
    assert_eq!(
        launch_spellings_in("k<<<grid, block>>>(a, b);"),
        vec!["<<<"],
        "the control: a CUDA launch must be recognized, or the scan below proves nothing"
    );
    assert!(launch_spellings_in("let launch = LaunchShape::new(b, g);").is_empty());
}

#[test]
fn no_crate_source_writes_a_launch_directive() {
    let crates = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates/");
    let mut files = Vec::new();
    for krate in std::fs::read_dir(crates).expect("read crates/") {
        let src = krate.expect("crate dir").path().join("src");
        if src.is_dir() {
            rust_files(&src, &mut files);
        }
    }
    // The population control: a moved `crates/` or a renamed `src` would otherwise scan nothing.
    assert!(
        files.len() > 20,
        "found only {} source files; the scan is not looking at the workspace",
        files.len()
    );
    let offenders: Vec<String> = files
        .iter()
        .filter_map(|f| {
            let src = std::fs::read_to_string(f).ok()?;
            let hits = launch_spellings_in(&src);
            (!hits.is_empty()).then(|| format!("{}: {hits:?}", f.display()))
        })
        .collect();
    assert!(
        offenders.is_empty(),
        "a launch directive in source: {offenders:#?}\n\nSTOP and re-read US10949182B2 before \
         keeping it (docs/e1d-parallel-mapping.md, rule 3)."
    );
}
