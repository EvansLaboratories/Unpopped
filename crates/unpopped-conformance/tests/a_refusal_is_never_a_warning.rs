//! Production code refuses. It does not warn, and it does not print.
//!
//! # Why this is a gate rather than a habit
//!
//! Fuel's architect, on what a generator owes a consumer when a fidelity check
//! fails:
//!
//! > If Fuel's gate cannot see lift infidelity, **Unpopped must refuse to emit
//! > the variant**, not emit it with a warning. **A warning is a fallback nobody
//! > reads.**
//!
//! Measured before building this: **zero** warn or print sites exist in any
//! `crates/*/src` today. So this is not a repair — it is a property the
//! workspace already has, written down where it can fail, because the moment a
//! fidelity check exists somebody will want to soften it. `eprintln!("warning:
//! this lift may be unfaithful")` is the single most natural line anyone will
//! ever add to a checker, it compiles, it looks responsible, and it converts a
//! refusal into a fallback nobody reads.
//!
//! The type system already makes the honest path the easy one — `try_generate`
//! and `Backend::lower` return `Result`, `Lowering`'s seams return
//! `Result<Spelling, LowerError>`, and a lift returns `Result<Lifted,
//! LiftError>`. **Every refusal in this crate has somewhere typed to go.** This
//! stops the untyped channel from being opened beside them.
//!
//! # Scope, stated because a gate that overreaches gets deleted
//!
//! `crates/*/src` only. **Tests may print freely and several must** — the
//! `cpu_end_to_end` skip path prints why it skipped, which is exactly right for
//! a developer's terminal and exactly the thing that must not be a library's
//! error channel. `#[cfg(test)]` modules inside `src` are excluded for the same
//! reason.
//!
//! It also does not forbid `Display`/`Debug` impls, `format!`, or a `detail`
//! string on a typed error — those are how a refusal explains itself, and this
//! workspace's declines carry their reasons in exactly that form.

use std::fs;
use std::path::{Path, PathBuf};

/// The untyped output channels. A refusal must never leave through one of these.
const CHANNELS: &[&str] = &[
    "eprintln!",
    "println!",
    "eprint!",
    "print!",
    "dbg!",
    "log::warn",
    "log::error",
    "tracing::warn",
    "tracing::error",
];

fn crates_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates/<pkg> has a parent")
        .to_path_buf()
}

fn production_sources() -> Vec<(String, String)> {
    let mut out = Vec::new();
    for entry in fs::read_dir(crates_dir()).expect("crates/ is readable") {
        let krate = entry.expect("entry").path();
        let src = krate.join("src");
        if !src.is_dir() {
            continue;
        }
        let mut stack = vec![src];
        while let Some(dir) = stack.pop() {
            for e in fs::read_dir(&dir).expect("readable dir").flatten() {
                let p = e.path();
                if p.is_dir() {
                    stack.push(p);
                } else if p.extension().is_some_and(|x| x == "rs") {
                    let rel = p
                        .strip_prefix(crates_dir())
                        .unwrap_or(&p)
                        .to_string_lossy()
                        .replace('\\', "/");
                    out.push((rel, fs::read_to_string(&p).expect("readable")));
                }
            }
        }
    }
    out.sort();
    out
}

/// Strip `//` comments, string literals, and `#[cfg(test)]` modules.
///
/// All three matter and each was learned the hard way in this workspace: a name
/// in a doc comment is not a use, a name in a message is not a use, and a test
/// module inside `src` is not production.
fn production_code(src: &str) -> String {
    let lines: Vec<&str> = src.lines().collect();
    let mut out = String::with_capacity(src.len());
    let mut i = 0;
    while i < lines.len() {
        if lines[i].trim_start().starts_with("#[cfg(test)]") {
            // Skip to the end of the following item by brace balance.
            let mut depth = 0i32;
            let mut started = false;
            while i < lines.len() {
                depth += lines[i].matches('{').count() as i32;
                depth -= lines[i].matches('}').count() as i32;
                if lines[i].contains('{') {
                    started = true;
                }
                i += 1;
                if started && depth <= 0 {
                    break;
                }
            }
            continue;
        }
        let code = match lines[i].find("//") {
            Some(c) => &lines[i][..c],
            None => lines[i],
        };
        let mut in_str = false;
        let mut escaped = false;
        for c in code.chars() {
            match c {
                _ if escaped => escaped = false,
                '\\' if in_str => escaped = true,
                '"' => {
                    in_str = !in_str;
                    out.push(' ');
                }
                _ if in_str => out.push(' '),
                _ => out.push(c),
            }
        }
        out.push('\n');
        i += 1;
    }
    out
}

/// Match at a token boundary, not as a substring.
///
/// **`"eprintln!"` CONTAINS `"println!"`.** A plain `contains` reports both for
/// every `eprintln!` in the tree — harmless for the verdict, since both are
/// offences, but it makes the message name a call that is not there, and a gate
/// whose output is wrong in the details is one people stop reading.
///
/// Caught by this file's own positive control before it shipped, which is the
/// argument for having one: the main gate was green and would have stayed green.
fn channels_in(code: &str) -> Vec<&'static str> {
    let boundary = |c: Option<char>| !c.is_some_and(|c| c.is_alphanumeric() || c == '_');
    CHANNELS
        .iter()
        .copied()
        .filter(|needle| {
            code.match_indices(needle)
                .any(|(i, _)| boundary(code[..i].chars().next_back()))
        })
        .collect()
}

#[test]
fn no_production_source_writes_to_an_untyped_channel() {
    let sources = production_sources();
    assert!(
        sources.len() > 20,
        "only {} production sources scanned — the walk found almost nothing, so \
         a green here would certify nothing",
        sources.len()
    );

    let mut offenders = Vec::new();
    for (path, src) in &sources {
        for channel in channels_in(&production_code(src)) {
            offenders.push(format!("{path} uses {channel}"));
        }
    }

    assert!(
        offenders.is_empty(),
        "a production source writes to an untyped channel. A refusal must return \
         a typed error — `LowerError`, `PlanError`, `LiftError`, `JitError` — not \
         print. A warning is a fallback nobody reads:\n  {}",
        offenders.join("\n  ")
    );
}

/// The detector fires, and does not fire on the things that merely look like it.
///
/// Without this, the test above passes just as happily against a `channels_in`
/// that returns nothing for every input — and it is a hand-rolled scanner, which
/// is the kind that quietly stops matching.
#[test]
fn the_channel_detector_fires_and_discriminates() {
    assert_eq!(
        channels_in(&production_code("    eprintln!(\"oops\");\n")),
        vec!["eprintln!"],
        "missed a bare eprintln!"
    );
    assert!(
        channels_in(&production_code(
            "    // eprintln!(\"explained, not called\");\n"
        ))
        .is_empty(),
        "fired on a COMMENTED-OUT call — a detector that flags comments gets \
         switched off within a week"
    );
    assert_eq!(
        channels_in(&production_code("    log::warn!(\"x\");\n")),
        vec!["log::warn"],
        "missed a path-qualified channel — the boundary rule must not reject `::`"
    );
    assert!(
        channels_in(&production_code(
            "    return Err(LowerError::UnsupportedOp { detail: \"eprintln! is not called here\".into() });\n"
        ))
        .is_empty(),
        "fired on a name inside a STRING — the same false positive, one level over"
    );
    assert!(
        channels_in(&production_code(
            "#[cfg(test)]\nmod tests {\n    fn t() { eprintln!(\"tests may print\"); }\n}\n"
        ))
        .is_empty(),
        "fired inside a #[cfg(test)] module — tests may print, and several must"
    );
}
