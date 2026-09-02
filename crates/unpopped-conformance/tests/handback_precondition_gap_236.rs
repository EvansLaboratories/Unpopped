//! Shipping a kernel-handback provider requires Fuel's GAP-236 increment 2.
//!
//! # The hole this guards, stated as the measurement rather than the worry
//!
//! Fuel's architect measured it: a kernel that lifts `fmaxf` (**NaN-suppressing**)
//! while honestly claiming `Max` (**NaN-propagating**) is **bit-identical to a
//! correct one across 256 seeds** on Fuel's current admission probe, and would be
//! **admitted**.
//!
//! Their gate is stronger than it first looked — *structural* infidelity is
//! caught, and a candidate claiming `ROPE` implemented as interleaved is
//! genuinely rejected against Fuel's rotate-half recipe, before any GPU work.
//! What survives is **numerical infidelity inside a matching structure**: the
//! mis-lifted kernel claims `Max`, lowers to `Max`, agrees with Fuel's recipe on
//! everything a seeded `[-0.5, 0.5)` fill can reach, and passes.
//!
//! **GAP-236 increment 1 is complete** — their probe set now reaches the
//! divergence, 0 → 3,584 divergences, admission policy verified live.
//! **Increment 2, wiring it into the production call site, is deliberately not
//! landed**: the call site is `#[cfg(feature = "cuda")]` and verifying it needs a
//! ~56-minute forge on a contended machine-wide slot. The portfolio PM ruled that
//! sequencing acceptable **on the condition that the precondition be enforceable
//! rather than remembered.**
//!
//! # Why the guard has to be HERE
//!
//! Fuel built a tripwire that reds the moment a provider appears — and then said
//! what it cannot see:
//!
//! > The tripwire scans **Fuel's** workspace. If Unpopped wires a provider in
//! > **your** repo, consuming Fuel as a dependency, the scan cannot see it — the
//! > trigger is not in their tree.
//!
//! **That is the likely shape of the handback.** Their guard is the backstop;
//! this repo is the path the risk actually takes. A guard belongs where its
//! trigger is, which is the same reason this workspace pins an adopter's panic
//! strings on its own side and they pin our `generate`/`try_generate` pairing on
//! theirs.
//!
//! # What "shipping a provider" means, concretely
//!
//! Reaching Fuel's admission entry — `verify_candidate` / `ingest_one`, which
//! take a `&CudaDevice` — or depending on a Fuel crate that offers it.
//!
//! **`fuel-kernel-seam-types` is NOT that** and is explicitly allowed: it is the
//! frozen `-types` grammar dep, optional, behind the off-by-default `seam`
//! feature, carrying `PatternNode` and no live-call envelope. The envelope
//! (`fuel_kernel_seam::Synthesizer`) lives in `baracuda-cuda-emit`, not here.
//!
//! # It cannot fire today, which is exactly why the control matters
//!
//! No provider exists, so the main assertion will sit green for as long as that
//! stays true — possibly months. **A guard nobody has seen fail is
//! indistinguishable from one that cannot**, so the detector is exercised
//! directly against seeded triggers rather than trusted.

use std::fs;
use std::path::{Path, PathBuf};

/// **Flip to `true` only on written confirmation from Fuel's architect that
/// GAP-236 increment 2 is wired at the production call site, naming the ref it
/// landed at.** Not on a roadmap entry, not on "increment 1 is done", and not on
/// this repo's belief about their tree — this workspace cannot measure Fuel's
/// state, so the discharge is testimony and must be recorded as such.
///
/// When it flips, put the ref in the commit message. A boolean that changed for
/// a reason nobody wrote down is the next reader's dead end.
const GAP_236_INCREMENT_2_LANDED: bool = false;

/// Fuel crates that do NOT reach the admission gate, with the reason each is
/// safe. Anything else counts as a trigger.
const ALLOWED_FUEL_DEPS: &[(&str, &str)] = &[(
    "fuel-kernel-seam-types",
    "frozen -types grammar only (PatternNode); the live-call envelope \
     fuel_kernel_seam::Synthesizer lives in baracuda-cuda-emit, not here",
)];

/// Names that mean this workspace is submitting candidates for admission.
///
/// # ⚠️ THIS AXIS IS EXTERNAL — AND, MEASURED, IT IS DOMINATED RATHER THAN FRAGILE
///
/// **Corrected 2026-09-02 by fuel's own measurement, after this note first said
/// the axis was fragile.** It is external, but the guard has **two** axes and one
/// strictly dominates the other:
///
/// ```text
/// unallowed_fuel_deps(manifest)   scans Cargo.toml — OWNED ENTIRELY HERE.
///                                 No rename in fuel touches it.
/// ADMISSION_API name scan of .rs  keyed on fuel's symbol names — the external one
/// ```
///
/// **Fuel measured that these are not public entry points at all**:
/// `mod jit_ingest` is **not** `pub mod`, `adopt_verified` is `fn` rather than
/// `pub fn`, and there are **zero** re-exports. *(Control: all six symbols exist
/// at their `origin/main` with one definition site each.)*
///
/// **So compiled code cannot name any of them without first declaring a
/// dependency on `fuel-dispatch` — and a private module cannot be re-exported, so
/// there is no second route.** Every trigger the name scan could catch in
/// compiled code **must pass the manifest check first.**
///
/// ⚠️ **The name list therefore adds coverage only for NON-COMPILED mentions** —
/// a stub, a plan, a name written before the dependency exists. **Real, but small,
/// and it is exactly the half that rots.** Treat it as a **best-effort tripwire,
/// not a coverage claim.**
///
/// **A guard with two axes where one dominates reads as two checks and is one.**
/// The dominated axis costs maintenance and **its rot is invisible, because the
/// surviving axis keeps the test green.** That is a different finding from *"this
/// guard will silently stop checking"* — nothing here stops checking; one half
/// was never load-bearing for the case that matters.
///
/// # What was originally written here, kept because the reasoning was sound
///
/// Every other corpus in this workspace is either derived from a
/// guaranteed-complete source (`ElementKind::ALL`, complete by mechanism) or is
/// honestly a corpus over a space with no authoritative list. **This one is
/// neither: it is a hand-copy of six entry points in *fuel's* public API, and
/// fuel can rename any of them without this repo observing it.**
///
/// **The failure mode is not a break — it is a stop.** A renamed entry point
/// makes this guard match nothing, and **a guard that has stopped checking looks
/// exactly like a guard that keeps passing.** Nothing here goes red; the trigger
/// simply becomes unreachable, which is the same shape as
/// `KNOWN_PANICKING = 0` measured over a population that excluded its own
/// subject.
///
/// **No local fix exists and none should be attempted.** The real remedies all
/// live on fuel's side — an exported name list, a stability declaration, or a
/// contract test in their repo that fails when an entry point is renamed.
/// Routed to their architect 2026-09-02.
///
/// **What is achievable from here is exactly this note**: converting a silent
/// decay into a documented limitation. That is the honest ceiling, and writing
/// it down is not a substitute for the fix — it is a statement of which half of
/// the guard is load-bearing and which half is a copy that will rot.
///
/// Sibling shape, recorded because the pair is instructive: this workspace pins
/// an adopter's panic strings on *its own* side because the adopter cannot run
/// the tests. **Here the guard runs fine and cannot see its subject.** Same
/// boundary, opposite failure.
const ADMISSION_API: &[&str] = &[
    "verify_candidate",
    "ingest_one",
    "CandidateKernel",
    "IngestOutcome",
    "adopt_verified",
    "RejectionReport",
];

fn crates_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates/<pkg> has a parent")
        .to_path_buf()
}

/// Fuel dependencies declared anywhere in the workspace, minus the allowlist.
fn unallowed_fuel_deps(manifest: &str) -> Vec<String> {
    let mut found = Vec::new();
    for line in manifest.lines() {
        let code = match line.find('#') {
            Some(i) => &line[..i],
            None => line,
        };
        let Some(rest) = code.trim().strip_prefix("fuel-") else {
            continue;
        };
        let name = format!(
            "fuel-{}",
            rest.split(|c: char| !c.is_alphanumeric() && c != '-')
                .next()
                .unwrap_or("")
        );
        if !ALLOWED_FUEL_DEPS.iter().any(|(a, _)| *a == name) {
            found.push(name);
        }
    }
    found
}

/// Admission-API names used in production code — comments and strings stripped,
/// because a name in prose is not a call. That distinction has cost this
/// workspace three separate false readings already.
fn admission_api_uses(src: &str) -> Vec<&'static str> {
    let mut code = String::with_capacity(src.len());
    for line in src.lines() {
        let stripped = match line.find("//") {
            Some(i) => &line[..i],
            None => line,
        };
        let mut in_str = false;
        let mut escaped = false;
        for c in stripped.chars() {
            match c {
                _ if escaped => escaped = false,
                '\\' if in_str => escaped = true,
                '"' => {
                    in_str = !in_str;
                    code.push(' ');
                }
                _ if in_str => code.push(' '),
                _ => code.push(c),
            }
        }
        code.push('\n');
    }
    ADMISSION_API
        .iter()
        .copied()
        .filter(|n| code.contains(n))
        .collect()
}

fn workspace_files(suffix: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let root = crates_dir()
        .parent()
        .expect("crates/ has a parent")
        .to_path_buf();
    let mut stack = vec![root.clone()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = fs::read_dir(&dir) else {
            continue;
        };
        for e in entries.flatten() {
            let p = e.path();
            let name = p.file_name().unwrap_or_default().to_string_lossy();
            if p.is_dir() {
                if name != "target" && !name.starts_with('.') {
                    stack.push(p);
                }
            } else if p.to_string_lossy().ends_with(suffix) {
                let rel = p
                    .strip_prefix(&root)
                    .unwrap_or(&p)
                    .to_string_lossy()
                    .replace('\\', "/");
                if let Ok(s) = fs::read_to_string(&p) {
                    out.push((rel, s));
                }
            }
        }
    }
    out.sort();
    out
}

#[test]
fn no_handback_provider_ships_before_gap_236_increment_2() {
    if GAP_236_INCREMENT_2_LANDED {
        return;
    }

    let manifests = workspace_files("Cargo.toml");
    let sources: Vec<(String, String)> = workspace_files(".rs")
        .into_iter()
        .filter(|(p, _)| !p.contains("/tests/") && !p.starts_with("crates/unpopped-conformance/"))
        .collect();

    // THE WALK REACHES THE PLACES A TRIGGER WOULD LIVE.
    //
    // "It scanned something" is not enough here: the two triggers have specific
    // homes, and a walk that found forty files but not `crates/unpopped` would
    // be green for the wrong reason. Named explicitly rather than counted,
    // because a count cannot say WHICH files.
    //
    // This is the assertion an end-to-end mutation would have covered. Adding a
    // Fuel dependency to prove the guard fires turned out to break the manifest
    // PARSE, so cargo never ran the test and the "mutation" proved nothing —
    // twice, in two shapes. The detector functions are exercised directly in
    // `the_trigger_detector_fires_and_discriminates`; this covers the half that
    // exercise cannot reach.
    assert!(
        manifests
            .iter()
            .any(|(p, _)| p == "crates/unpopped/Cargo.toml"),
        "the manifest walk did not reach crates/unpopped/Cargo.toml, which is          where a Fuel dependency would be declared. Found: {:?}",
        manifests.iter().map(|(p, _)| p).collect::<Vec<_>>()
    );
    assert!(
        sources
            .iter()
            .any(|(p, _)| p.starts_with("crates/unpopped/src/")),
        "the source walk did not reach crates/unpopped/src/, which is where a          provider would be written"
    );

    let mut triggers = Vec::new();
    for (path, src) in &manifests {
        for dep in unallowed_fuel_deps(src) {
            triggers.push(format!("{path} depends on {dep}"));
        }
    }
    for (path, src) in &sources {
        for name in admission_api_uses(src) {
            triggers.push(format!("{path} names Fuel's admission API `{name}`"));
        }
    }

    assert!(
        triggers.is_empty(),
        "this workspace looks like it ships a kernel-handback provider, and \
         Fuel's GAP-236 increment 2 is not recorded as landed.\n\n\
         Until it is, a kernel that lifts `fmaxf` while honestly claiming `Max` \
         is bit-identical to a correct one across 256 seeds on Fuel's admission \
         probe, and would be ADMITTED — structural infidelity is caught, \
         numerical infidelity inside a matching structure is not. Fuel's own \
         tripwire cannot see this repo.\n\n\
         Either wait for increment 2, or flip GAP_236_INCREMENT_2_LANDED with \
         the ref it landed at in the commit message.\n\n  {}",
        triggers.join("\n  ")
    );
}

/// The detector fires on each trigger shape, and on nothing else.
///
/// This is the only assertion in the file that can fail today. The one above
/// guards a thing that does not exist yet and will sit green until it does —
/// **a guard nobody has seen fail is indistinguishable from one that cannot**,
/// and this file would otherwise be a comment with a `#[test]` on it.
#[test]
fn the_trigger_detector_fires_and_discriminates() {
    assert_eq!(
        unallowed_fuel_deps("fuel-dispatch = { version = \"0.1\" }\n"),
        vec!["fuel-dispatch".to_string()],
        "missed a Fuel dependency that reaches the admission gate"
    );
    assert!(
        unallowed_fuel_deps("fuel-kernel-seam-types = { version = \"0.10.3\" }\n").is_empty(),
        "fired on the allowlisted -types grammar dep, which carries no admission path"
    );
    assert!(
        unallowed_fuel_deps("# fuel-dispatch = { version = \"0.1\" }  # considered\n").is_empty(),
        "fired on a COMMENTED-OUT dependency"
    );

    assert_eq!(
        admission_api_uses("    let out = ingest_one(&cand, &device);\n"),
        vec!["ingest_one"],
        "missed a call into Fuel's admission entry"
    );
    assert!(
        admission_api_uses("    // ingest_one is Fuel's, not ours\n").is_empty(),
        "fired on a name in a comment"
    );
    assert!(
        admission_api_uses("    Err(E::Detail(\"verify_candidate is Fuel's\".into()))\n")
            .is_empty(),
        "fired on a name inside a string"
    );
}
