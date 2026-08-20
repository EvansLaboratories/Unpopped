//! The KISS `structure_key` byte-match leg — KISS-CLASSIFY-6.7.
//!
//! This crate is one of three independent token-tier derivers (KISS's reference,
//! Fuel's seam, and this codec). What that independence is worth is decided
//! here: every vector below is KISS's, vendored verbatim, and compared as
//! **decode → re-encode → compare bytes**. Nothing is transcribed, and nothing
//! is asserted field-by-field — a test checking `key.dtype == F32` would pass
//! against a codec that spells the token differently, which is the only failure
//! a byte-match exists to find.
//!
//! # Skips have to be earned
//!
//! A consumer implementing only some target namespaces scopes the rest as
//! capability **exclusions** (§6.8), not mismatches. The risk is obvious: a skip
//! list is where a real divergence goes to hide, because "we don't support that
//! target" is indistinguishable from "we get that one wrong" if nobody checks.
//!
//! So a skip is never taken on the artifact's say-so. For each vector this codec
//! declines, the harness substitutes a target it *does* implement, holding every
//! other byte identical. Only if the substituted token round-trips **byte-exact**
//! is the exclusion accepted — which demonstrates that every field under test is
//! correct and the target alone is unimplemented. A vector that still fails
//! after substitution is a real divergence and fails the leg.
//!
//! That control is not hypothetical: on the provisional run it turned six
//! failures into one unimplemented namespace plus one missing arch variant, and
//! ruled out contraction handling, which a bare count would have left everyone
//! suspecting.

use unpopped_vocab::{STRUCTURE_KEY_VERSION, StructureKey, TokenDecline};

const VECTORS: &str = include_str!("../kiss/structure_key_vectors.json");

/// A target this codec implements, used to prove a decline is target-only.
/// Must be one this codec parses, or the control proves nothing — asserted by
/// `the_skip_control_can_fail`, not left to this sentence. (It named
/// `arch_from_code` until that function was deleted with the closed `ArchSku`
/// codec, at which point the precondition pointed at nothing.)
const SUBSTITUTE_TARGET: &str = "cuda:sm89";

// ---------------------------------------------------------------------------
// Provenance
// ---------------------------------------------------------------------------

/// The vendored artifact is the one this leg claims to have run against.
///
/// Two commits with two meanings: the file was copied from a KISS `main` commit
/// (recorded in `kiss/README.md`), while its own `source_commit` is the *spec*
/// provenance it was generated against.
///
/// **This is not a currency check and must not be read as one.** `source_commit`
/// has never moved — `19c3ad7` across every revision of the artifact — because
/// it names the spec commit and the changes have all been generator changes. It
/// therefore cannot fail when the artifact is regenerated. What identifies the
/// revision is the content hash in
/// `the_vendored_artifacts_are_the_revisions_this_leg_was_written_against`.
#[test]
fn the_artifact_is_the_one_this_leg_claims() {
    for (key, want) in [
        ("\"schema\"", "kiss-structure-key-vectors-v1"),
        ("\"source_commit\"", "19c3ad7"),
        ("\"token_prefix\"", "sk4"),
        ("\"clause\"", "KISS-CLASSIFY-6.7"),
    ] {
        assert_eq!(
            scalar(VECTORS, key).as_deref(),
            Some(want),
            "vendored artifact's {key} is not what this leg was written against"
        );
    }
    assert_eq!(
        scalar(VECTORS, "\"structure_key_schema_version\"").and_then(|v| v.parse::<u16>().ok()),
        Some(STRUCTURE_KEY_VERSION),
        "artifact is from a different schema version than this build implements"
    );
}

/// **Which REVISION of the artifact this leg holds**, pinned as a content hash.
///
/// # `source_commit` cannot do this job, and I was treating it as though it did
///
/// `the_artifact_is_the_one_this_leg_claims` asserts `source_commit == 19c3ad7`.
/// The KISS maintainer reports that value has **never moved** — the same across
/// all six revisions of this artifact, through decline counts of 10 → 15 → 17
/// and the `vulkan:` respell — because it records the **spec** commit and every
/// change has been a *generator* change. It even names a commit at which the
/// current artifact does not exist.
///
/// So that assertion cannot fail on a re-vendor, which means it was never a
/// currency check. Fuel's framing: *the stamp tells you which thing you bound
/// to, not whether that thing moved.*
///
/// What actually identified the revision lived in `kiss/README.md` as prose — a
/// blob sha and a sha256 that no test read. A stale vendored copy would have
/// passed this entire leg. That is the same can't-fire shape as the
/// absence-assertion this file already carries a post-mortem for, and it went
/// unnoticed for the same reason: a pin that is *present* reads like a pin that
/// *works*.
///
/// # Why FNV-1a and not sha256
///
/// This detects "the file changed", not tampering by an adversary. FNV-1a-64 is
/// dependency-free — adding a crypto crate to a driver-free vocabulary crate to
/// hash a test fixture would be a real cost for no security property this needs.
/// The sha256 in `kiss/README.md` remains the figure to cite when comparing
/// against KISS; this is the figure that FAILS A BUILD when the two drift.
#[test]
fn the_vendored_artifacts_are_the_revisions_this_leg_was_written_against() {
    // FNV-1a-64, the same construction `unpopped`'s kernel revision hash uses.
    fn fnv1a64(bytes: &[u8]) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for &b in bytes {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x100_0000_01b3);
        }
        h
    }
    const MANIFEST: &str = include_str!("../kiss/dtype_manifest.json");

    for (name, text, want_len, want_hash) in [
        (
            "structure_key_vectors.json",
            VECTORS,
            16_458_usize,
            0x0279_76ca_a2ee_ca73_u64,
        ),
        (
            "dtype_manifest.json",
            MANIFEST,
            3_183_usize,
            0x2401_3977_364a_8de2_u64,
        ),
    ] {
        assert_eq!(
            text.len(),
            want_len,
            "{name}: vendored length changed — re-verify against KISS and update              BOTH this pin and kiss/README.md, naming the commit you took it from"
        );
        assert_eq!(
            fnv1a64(text.as_bytes()),
            want_hash,
            "{name}: vendored CONTENT changed. This is the check `source_commit`              cannot perform, because that field records the spec commit and does              not move when the artifact is regenerated."
        );
        // CR-free, asserted rather than assumed. Two sibling KISS artifacts were
        // being written LF to file and CRLF to stdout by the same generator, so
        // "which output path did this come through" was a real question. Taking
        // the vendor with `git cat-file blob > dest` avoids the translation; this
        // proves it stayed avoided.
        assert!(
            !text.as_bytes().contains(&13u8), // 13 = CR, written numerically
            //     because the escape keeps collapsing through tooling layers
            "{name}: carries CR bytes — it was checked out rather than blob-copied,              and a translated copy cannot be diffed against its source"
        );
    }
}

/// **The per-namespace vocabulary versions are ASSERTED**, not merely read.
///
/// KISS-CLASSIFY §6.8-0009 requires a consumer to assert a vocabulary version.
/// This crate carries the target token opaquely and compares it whole
/// (§6.8-0002), so a vocabulary bump changes bytes it reproduces faithfully and
/// never interprets — but *immune is not conformant*, and a consumer that
/// asserts nothing satisfies the clause by accident.
///
/// # This replaces an absence-assertion that FAILED TO FIRE
///
/// Before the field existed, this test asserted it was **absent** under four
/// guessed names — `vocabulary_version`, `namespace_vocabulary_version`,
/// `vulkan_vocabulary_version`, `target_vocabulary_version` — so that its
/// arrival would be loud. The field arrived as **`namespace_vocabulary_versions`**
/// (plural, and an *object* rather than a scalar) and the assertion **passed**,
/// silently, exactly as it had every day before.
///
/// The lesson is general and cost nothing to learn only because a peer told me
/// the field had landed: **an absence-assertion over enumerated names is worth
/// no more than the enumeration.** The robust form does not guess. It pins the
/// artifact's whole top-level key set, so *any* added or removed field breaks
/// this test regardless of what it is called — which is what
/// `the_artifact_shape_is_pinned_so_any_new_field_is_loud` below does.
#[test]
fn the_namespace_vocabulary_versions_are_asserted() {
    // Object-valued, so the scalar reader does not apply — match the whole
    // block byte-exactly, which is also the strictest thing available.
    for (ns, ver) in [("cuda", 1), ("vulkan", 4)] {
        let needle = format!("\"{ns}\": {ver}");
        assert!(
            VECTORS.contains(&needle),
            "artifact must declare {ns} vocabulary version {ver}. If the version \
             moved, RE-VERIFY this leg against the new vocabulary before bumping \
             the number here — the point of asserting is that a bump is a \
             decision, not a diff."
        );
    }
}

/// **Any change to the artifact's top-level shape is loud**, whatever it is
/// called.
///
/// The predecessor of this test guessed four names for a field that had not
/// landed yet and missed the one that did. Enumerating the keys that ARE present
/// has no such failure mode: a field added under any name breaks this, and so
/// does one removed.
#[test]
fn the_artifact_shape_is_pinned_so_any_new_field_is_loud() {
    // Every top-level key, in the artifact's own order.
    const EXPECTED: &[&str] = &[
        "schema",
        "generated_from",
        "source_commit",
        "clause",
        "namespace_vocabulary_note",
        "namespace_vocabulary_versions",
        "structure_key_schema_version",
        "token_prefix",
        "dtype_axis_note",
        "recognition_count",
        "usable_count",
        "dtype_recognition_set",
        "dtype_usable_set",
        "reserved_dtypes",
        "target_axis_note",
        "target_namespaces",
        "mapping_guard_note",
        "coverage_note",
        "positive_vectors",
        "decline_vectors",
    ];
    let found: Vec<&str> = EXPECTED
        .iter()
        .copied()
        .filter(|k| VECTORS.contains(&format!("\"{k}\":")))
        .collect();
    assert_eq!(
        found, EXPECTED,
        "a pinned top-level key is missing from the artifact"
    );

    // And nothing BEYOND them: count the top-level keys by depth so an addition
    // under an unguessed name cannot slip through the way one already did.
    let mut depth = 0i32;
    let mut top_level = 0usize;
    let mut in_str = false;
    let mut esc = false;
    let b: Vec<char> = VECTORS.chars().collect();
    for i in 0..b.len() {
        let c = b[i];
        if esc {
            esc = false;
            continue;
        }
        match c {
            '\\' if in_str => esc = true,
            '"' => {
                in_str = !in_str;
                // A key at depth 1 is one whose closing quote is followed by ':'.
                if !in_str && depth == 1 {
                    let rest = b[i + 1..].iter().take_while(|c| c.is_whitespace()).count();
                    if b.get(i + 1 + rest) == Some(&':') {
                        top_level += 1;
                    }
                }
            }
            '{' | '[' if !in_str => depth += 1,
            '}' | ']' if !in_str => depth -= 1,
            _ => {}
        }
    }
    assert_eq!(
        top_level,
        EXPECTED.len(),
        "the artifact has {top_level} top-level keys but this leg pins          {}. A field was added or removed — read it, decide whether this leg          must assert it, and update EXPECTED deliberately.",
        EXPECTED.len()
    );
}

// ---------------------------------------------------------------------------
// The leg
// ---------------------------------------------------------------------------

#[test]
fn positive_vectors_round_trip_byte_exact() {
    let vectors = positives();
    assert_eq!(vectors.len(), 20, "artifact positive-vector count changed");

    let mut matched = Vec::new();
    let mut skipped = Vec::new();
    let mut failed = Vec::new();

    for v in &vectors {
        match StructureKey::parse_token(&v.token) {
            Ok(k) if k.to_token() == v.token => matched.push(v),
            Ok(k) => failed.push(format!(
                "{}: re-encoded to different bytes\n    want {}\n    got  {}",
                v.name,
                v.token,
                k.to_token()
            )),
            Err(e) => match earn_skip(v) {
                // Every other field is correct; the target alone is unimplemented.
                Ok(()) => skipped.push(v),
                Err(why) => failed.push(format!(
                    "{}: declined ({e:?}) and NOT attributable to the target — {why}\n    {}",
                    v.name, v.token
                )),
            },
        }
    }

    println!("\n=== KISS byte-match leg — positives ===");
    println!(
        "claimed and byte-exact : {}/{}",
        matched.len(),
        vectors.len()
    );
    println!("capability exclusions  : {}", skipped.len());
    for v in &skipped {
        println!("   SKIP {} — target {} ({})", v.name, v.target, v.namespace);
    }
    for f in &failed {
        println!("   FAIL {f}");
    }

    assert!(failed.is_empty(), "{} real divergence(s)", failed.len());

    // Non-vacuity: a leg that skipped everything would report zero failures.
    assert!(
        matched.len() >= vectors.len() - skipped.len(),
        "accounting error"
    );
    assert!(
        matched.len() > vectors.len() / 2,
        "most vectors must be CLAIMED, not skipped — {} skipped of {}",
        skipped.len(),
        vectors.len()
    );
}

/// Every skipped vector names a target outside what this codec implements, and
/// every claimed one names a target inside it.
///
/// The skip list is the soft spot in any conformance leg, so it gets its own
/// assertion rather than riding along inside the pass above: a skip must
/// correlate with the target axis, not with which vectors happen to be hard.
#[test]
fn skips_fall_only_on_unimplemented_targets() {
    let mut claimed_targets = Vec::new();
    let mut skipped_targets = Vec::new();
    for v in &positives() {
        match StructureKey::parse_token(&v.token) {
            Ok(_) => claimed_targets.push(v.target.clone()),
            Err(_) => skipped_targets.push(v.target.clone()),
        }
    }
    for t in &skipped_targets {
        assert!(
            !claimed_targets.contains(t),
            "target `{t}` is both claimed and skipped — the skip is hiding something \
             that is not a capability exclusion"
        );
    }
    println!("\nclaimed targets: {claimed_targets:?}");
    println!("skipped targets: {skipped_targets:?}");
}

#[test]
fn decline_vectors_produce_the_same_verdict() {
    let vectors = declines();
    assert_eq!(vectors.len(), 17, "artifact decline-vector count changed");

    let mut exact = 0;
    let mut failed = Vec::new();
    for v in &vectors {
        match StructureKey::parse_token(&v.token) {
            Ok(_) => failed.push(format!(
                "{}: ACCEPTED a token that must decline as {}\n    {}",
                v.name, v.decline, v.token
            )),
            Err(e) => {
                // `wire_name` is an exhaustive match inside the vocab crate, so a
                // decline variant added later is a build failure there rather
                // than a silent misreport here. That guard cannot live in this
                // file: `TokenDecline` is `#[non_exhaustive]`, so a match written
                // out here would need a catch-all and would rot invisibly.
                let name = e.wire_name();
                let payload_ok = match (&e, v.got) {
                    (TokenDecline::UnsupportedSchemaVersion { version }, Some(g)) => {
                        i64::from(*version) == g
                    }
                    (TokenDecline::UnsupportedSchemaVersion { .. }, None) => false,
                    (_, Some(_)) => false,
                    (_, None) => true,
                };
                if name == v.decline && payload_ok {
                    exact += 1;
                } else {
                    failed.push(format!(
                        "{}: want {}{}, got {}{}\n    {}",
                        v.name,
                        v.decline,
                        v.got.map(|g| format!(" got={g}")).unwrap_or_default(),
                        name,
                        match &e {
                            TokenDecline::UnsupportedSchemaVersion { version } =>
                                format!(" got={version}"),
                            _ => String::new(),
                        },
                        v.token
                    ));
                }
            }
        }
    }

    println!("\n=== KISS byte-match leg — declines ===");
    println!("exact verdict match: {exact}/{}", vectors.len());
    for f in &failed {
        println!("   FAIL {f}");
    }
    assert!(failed.is_empty(), "{} verdict mismatch(es)", failed.len());
    assert_eq!(exact, vectors.len(), "every decline must match exactly");
}

/// The artifact's `mapping_guard_note` is heeded, not merely present.
///
/// KISS's note says its `E0004` guard protects only the KISS side and a consumer
/// must guard its own mapping. Ours is
/// [`TokenDecline::wire_name`](unpopped_vocab::TokenDecline::wire_name) — an
/// exhaustive match in the crate that owns the enum, with distinctness pinned by
/// its own unit test. This asserts the note still says what we responded to: if
/// KISS changes the contract, this fails rather than silently leaving us
/// compliant with a superseded one.
#[test]
fn the_mapping_guard_note_is_still_the_one_we_answered() {
    assert!(
        VECTORS.contains("\"mapping_guard_note\""),
        "artifact no longer carries the mapping-guard contract"
    );
    assert!(
        VECTORS.contains("MUST guard the mapping itself"),
        "the mapping-guard note's requirement changed — re-check `wire_name`"
    );
}

/// **This leg does not cover the dtype axis, and saying so is the point.**
///
/// Found by mutation: misspelling `c128` as `c127` in the dtype codec leaves
/// every assertion in this file green. The 20 positive vectors exercise exactly
/// three dtypes in the dtype position (`f32`, `f16`, `f8e4m3fn`) and five
/// anywhere at all, against a usable set of 22 — so a spelling divergence on any
/// of the other 17 is invisible here.
///
/// That is not a defect in the vectors. A byte-match is a *token-grammar*
/// instrument: it proves two implementations agree on field order, optional-field
/// presence, sentinel spellings and decline verdicts, and it would take 22×
/// the vectors to also make it a vocabulary instrument. The vocabulary is covered
/// by `kiss_dtype_manifest.rs`, which compares the full token image against
/// KISS's generated manifest in both directions — and which *does* fail on that
/// same mutation, naming `c128` exactly.
///
/// # But "covered by the manifest test" was too broad, and the gap was total
///
/// The sentence above is true for a **misspelling** (`c127` is not in KISS's set,
/// so a set comparison catches it) and false for a **swap**. Permuting two arms —
/// `dtype_token(I32) -> "u32"` and `dtype_token(U32) -> "i32"` — leaves the token
/// *set* identical, and `the_dtype_set_matches_kiss_exactly_in_both_directions`
/// sorts and dedups, so it compares sets and cannot see it.
///
/// Measured, by seeding exactly that swap: the manifest set test passed, **and
/// every one of the 11 tests in THIS file passed too** — `i32`/`u32` are not in
/// the vector dtype positions. So a same-width dtype swap was invisible to the
/// entire vocabulary suite, not merely to this leg. Two tests, each sound for its
/// own purpose, composing to leave a hole neither's documentation admitted.
///
/// Closed by `every_dtype_arm_is_pinned_by_identity_not_by_set_membership`, which
/// pins the mapping rather than the image and fails on that swap. The complement
/// is now three-way and the division is: **this leg = grammar, the set test =
/// membership, the identity pin = mapping.** Rule owed to MLMF.
///
/// Recorded as an executable note because "the byte-match passed" is the kind of
/// sentence that gets quoted as though it meant more than it does. The two tests
/// are complementary, and neither is sufficient alone.
#[test]
fn this_leg_is_not_dtype_coverage() {
    let usable = VECTORS
        .split_once("\"usable_count\": ")
        .and_then(|(_, r)| {
            r.split(|c: char| !c.is_ascii_digit())
                .next()?
                .parse::<usize>()
                .ok()
        })
        .expect("usable_count");

    let mut in_dtype_position: Vec<String> = positives()
        .iter()
        .filter_map(|p| p.token.split('|').nth(2).map(str::to_string))
        .collect();
    in_dtype_position.sort();
    in_dtype_position.dedup();

    println!("\ndtype position coverage: {in_dtype_position:?} of {usable} usable");
    assert!(
        in_dtype_position.len() < usable,
        "the vector set now exercises every usable dtype in the dtype position — \
         good news, and this note plus its caveat in the module docs can be relaxed"
    );
}

// ---------------------------------------------------------------------------
// Earning a skip
// ---------------------------------------------------------------------------

/// `Ok(())` if this vector's decline is attributable to its target ALONE.
///
/// Substitutes a target this codec implements, holding every other byte
/// identical. Round-tripping byte-exact after substitution proves the remaining
/// fields are all correct — so the exclusion is a capability boundary and not a
/// divergence wearing one as a disguise.
fn earn_skip(v: &Positive) -> Result<(), String> {
    let mut parts: Vec<&str> = v.token.split('|').collect();
    if parts.len() < 4 {
        return Err("token has no target field".to_string());
    }
    if parts[3] == SUBSTITUTE_TARGET {
        return Err(format!(
            "target is already `{SUBSTITUTE_TARGET}`, which this codec implements — \
             the decline is not a target exclusion"
        ));
    }
    parts[3] = SUBSTITUTE_TARGET;
    let substituted = parts.join("|");
    match StructureKey::parse_token(&substituted) {
        Ok(k) if k.to_token() == substituted => Ok(()),
        Ok(k) => Err(format!(
            "with the target substituted it still re-encodes differently: got {}",
            k.to_token()
        )),
        Err(e) => Err(format!(
            "with the target substituted it still declines: {e:?}"
        )),
    }
}

/// The substitution control is only meaningful if the substituted target is one
/// this codec actually implements, and if substitution is capable of *failing*.
#[test]
fn the_skip_control_can_fail() {
    // The substitute must itself be implemented.
    let real = "sk4|bin|f32|cuda:sm89|ix32|grid|r2|co/00/v4/d16/f;co/00/v4/d16/f|-";
    assert!(
        StructureKey::parse_token(real).is_ok(),
        "control target `{SUBSTITUTE_TARGET}` must be implemented, or every skip is free"
    );

    // A vector broken in a NON-target field must NOT earn a skip, even though
    // its target is unimplemented. This is the case the control exists for.
    let poisoned = Positive {
        name: "poisoned".into(),
        target: "vulkan:whatever".into(),
        namespace: "vulkan".into(),
        // `zz` is not a valid `<mp>` code — broken independently of the target.
        token: "sk4|gem|f32|vulkan:whatever|ix32|grid|r2|\
                co/00/v4/d16/f;co/00/v4/d16/f;co/00/v4/d16/f|-|ctll/d16/f32/f32/f32/zz"
            .into(),
    };
    assert!(
        earn_skip(&poisoned).is_err(),
        "a vector broken outside the target field must not be skippable — \
         that is exactly how a divergence would hide in the skip list"
    );
}

// ---------------------------------------------------------------------------
// Artifact scanning. Deliberately dependency-free; positive-controlled below.
// ---------------------------------------------------------------------------

struct Positive {
    name: String,
    target: String,
    namespace: String,
    token: String,
}

struct Decline {
    name: String,
    token: String,
    decline: String,
    got: Option<i64>,
}

fn scalar(src: &str, quoted_key: &str) -> Option<String> {
    let (_, rest) = src.split_once(&format!("{quoted_key}: "))?;
    let rest = rest.trim_start();
    if let Some(s) = rest.strip_prefix('"') {
        return Some(s.split('"').next()?.to_string());
    }
    let end = rest.find(|c: char| !c.is_ascii_digit())?;
    Some(rest[..end].to_string())
}

fn field(line: &str, key: &str) -> Option<String> {
    let v = line.trim().strip_prefix(&format!("\"{key}\": "))?;
    Some(v.trim_end_matches(',').trim_matches('"').to_string())
}

fn section(start_key: &str, end_key: Option<&str>) -> &'static str {
    let s = VECTORS.find(start_key).expect("section present");
    let e = end_key
        .and_then(|k| VECTORS.find(k))
        .filter(|e| *e > s)
        .unwrap_or(VECTORS.len());
    &VECTORS[s..e]
}

fn positives() -> Vec<Positive> {
    let sec = section("\"positive_vectors\"", Some("\"decline_vectors\""));
    let mut out = Vec::new();
    let (mut name, mut target, mut ns, mut token) = (None, None, None, None);
    for line in sec.lines() {
        name = field(line, "name").or(name);
        target = field(line, "target").or(target);
        ns = field(line, "target_namespace").or(ns);
        token = field(line, "token").or(token);
        if line.trim().starts_with('}') {
            if let (Some(n), Some(t), Some(s), Some(k)) =
                (name.take(), target.take(), ns.take(), token.take())
            {
                out.push(Positive {
                    name: n,
                    target: t,
                    namespace: s,
                    token: k,
                });
            }
            (name, target, ns, token) = (None, None, None, None);
        }
    }
    out
}

fn declines() -> Vec<Decline> {
    let sec = section("\"decline_vectors\"", None);
    let mut out = Vec::new();
    let (mut name, mut token, mut dec, mut got) = (None, None, None, None);
    for line in sec.lines() {
        name = field(line, "name").or(name);
        token = field(line, "token").or(token);
        dec = field(line, "decline").or(dec);
        got = field(line, "got").and_then(|g| g.parse().ok()).or(got);
        if line.trim().starts_with('}') {
            if let (Some(n), Some(k), Some(d)) = (name.take(), token.take(), dec.take()) {
                out.push(Decline {
                    name: n,
                    token: k,
                    decline: d,
                    got: got.take(),
                });
            }
            (name, token, dec, got) = (None, None, None, None);
        }
    }
    out
}

/// Positive control on the scanner.
///
/// Every count and comparison above comes from these two functions. A scanner
/// that silently returned nothing would make the whole leg pass while testing
/// nothing at all — the exact vacuous-green this file exists to prevent
/// elsewhere.
#[test]
fn the_artifact_scanner_actually_reads_the_vectors() {
    let pos = positives();
    let dec = declines();
    assert_eq!(pos.len(), 20);
    assert_eq!(dec.len(), 17);

    // Namespaces are tagged and split the way the artifact says.
    let vulkan = pos.iter().filter(|p| p.namespace == "vulkan").count();
    let cuda = pos.iter().filter(|p| p.namespace == "cuda").count();
    assert_eq!((cuda, vulkan), (19, 1), "namespace split changed");

    // Tokens and targets are non-empty and internally consistent: the tag must
    // agree with the token's own field 3, or the skip axis is being read from a
    // label rather than from the data.
    for p in &pos {
        assert!(p.token.starts_with("sk4|"), "{}: token malformed", p.name);
        let field3 = p.token.split('|').nth(3).expect("target field");
        assert_eq!(field3, p.target, "{}: tag disagrees with token", p.name);
        assert!(
            p.target.starts_with(&format!("{}:", p.namespace)),
            "{}: namespace tag does not prefix the target",
            p.name
        );
    }

    // The payload-carrying declines are present and parsed as numbers.
    let with_payload: Vec<_> = dec.iter().filter(|d| d.got.is_some()).collect();
    assert_eq!(with_payload.len(), 2, "expected two versioned declines");
    let mut versions: Vec<i64> = with_payload.iter().filter_map(|d| d.got).collect();
    versions.sort_unstable();
    assert_eq!(versions, [3, 9]);
}
