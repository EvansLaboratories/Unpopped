//! Sizing harness for the Slang parser-split job (PM task, 2026-09-25):
//! runs `convert.rs`'s Slang [`Frontend`] over every source file in a
//! directory and reports accepted-vs-refused, bucketed by [`LiftError`]
//! variant — NOT `convert::lift`'s combined result, because `lift` chains
//! `elementwise.or_else(reduction).or_else(scan)` and returns only the LAST
//! attempt's error, which can misattribute the real reason a file was
//! refused. This harness calls all three lifters directly so the histogram
//! reflects what actually happened, not an artifact of the combinator.
//!
//! Usage: `cargo run --example slang_corpus_report --features convert -- <dir>`
//! `<dir>` holds `.slang` source files (any extension is accepted; only the
//! directory's file count is reported as the population).

use std::env;
use std::fs;
use std::path::PathBuf;

use unpopped::LiftError;
use unpopped::convert::{SLANG, lift_elementwise, lift_reduction, lift_scan};
use unpopped_vocab::ElementKind;

const F32: &[ElementKind] = &[ElementKind::F32];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Bucket {
    AcceptedElementwise,
    AcceptedReduction,
    AcceptedScan,
    NotAKernel,
    Inexpressible,
    Unrecognized,
    DtypeNotAdmissible,
}

fn classify(src: &str, name: &str) -> (Bucket, Option<String>) {
    let ew = lift_elementwise(&SLANG, src, name, F32);
    if ew.is_ok() {
        return (Bucket::AcceptedElementwise, None);
    }
    let rd = lift_reduction(&SLANG, src, name, F32);
    if rd.is_ok() {
        return (Bucket::AcceptedReduction, None);
    }
    let sc = lift_scan(&SLANG, src, name, F32);
    if sc.is_ok() {
        return (Bucket::AcceptedScan, None);
    }

    // All three failed. prepared()'s first three steps (kernel-marker check,
    // residue check, parse) are IDENTICAL across all three calls, so
    // NotAKernel / Inexpressible show up on all three errors together and
    // any one of them is representative. Only op-class recognition
    // (Unrecognized / DtypeNotAdmissible) can differ between the three, so
    // when we get there we report each attempt's variant separately isn't
    // useful for a single bucket -- take elementwise's as representative and
    // note the raw strings from all three for the appendix.
    let errs = [ew.unwrap_err(), rd.unwrap_err(), sc.unwrap_err()];
    let detail = errs
        .iter()
        .map(|e| format!("{e:?}"))
        .collect::<Vec<_>>()
        .join(" | ");

    for e in &errs {
        if matches!(e, LiftError::NotAKernel) {
            return (Bucket::NotAKernel, Some(detail));
        }
    }
    for e in &errs {
        if matches!(e, LiftError::Inexpressible(_)) {
            return (Bucket::Inexpressible, Some(detail));
        }
    }
    for e in &errs {
        if matches!(e, LiftError::DtypeNotAdmissible { .. }) {
            return (Bucket::DtypeNotAdmissible, Some(detail));
        }
    }
    (Bucket::Unrecognized, Some(detail))
}

fn main() {
    let dir = env::args()
        .nth(1)
        .expect("usage: slang_corpus_report <dir-of-slang-files>");
    let dir = PathBuf::from(dir);

    let mut entries: Vec<PathBuf> = fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", dir.display()))
        .map(|e| e.unwrap().path())
        .filter(|p| p.is_file())
        .collect();
    entries.sort();

    let population = entries.len();
    let mut counts: std::collections::BTreeMap<&'static str, usize> = Default::default();
    let mut per_file: Vec<(String, Bucket, Option<String>)> = Vec::new();

    for path in &entries {
        let name = path.file_stem().unwrap().to_string_lossy().to_string();
        let src =
            fs::read_to_string(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        let (bucket, detail) = classify(&src, &name);
        let label = match bucket {
            Bucket::AcceptedElementwise => "accepted: elementwise",
            Bucket::AcceptedReduction => "accepted: reduction",
            Bucket::AcceptedScan => "accepted: scan",
            Bucket::NotAKernel => "refused: NotAKernel",
            Bucket::Inexpressible => "refused: Inexpressible (residue)",
            Bucket::Unrecognized => "refused: Unrecognized",
            Bucket::DtypeNotAdmissible => "refused: DtypeNotAdmissible",
        };
        *counts.entry(label).or_insert(0) += 1;
        per_file.push((name, bucket, detail));
    }

    println!("population: {population} files in {}", dir.display());
    println!();
    println!("=== histogram ===");
    for (label, n) in &counts {
        println!("{n:4}  {label}");
    }

    println!();
    println!("=== refused files (name, bucket, detail) ===");
    for (name, bucket, detail) in &per_file {
        if !matches!(
            bucket,
            Bucket::AcceptedElementwise | Bucket::AcceptedReduction | Bucket::AcceptedScan
        ) {
            println!("{name}\t{bucket:?}\t{}", detail.as_deref().unwrap_or(""));
        }
    }
}
