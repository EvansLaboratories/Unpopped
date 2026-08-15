//! This crate's §6.1 dtype vocabulary == KISS's, checked against the vendored
//! machine-readable manifest rather than against a reading of the prose table.
//!
//! # Why a test and not a review
//!
//! Before sk4 this crate carried a 22-member §6.1 set against KISS's 24, and
//! nothing detected it. Both sides were prose, and prose is compared by a human
//! who is already fairly sure it matches. The failure was silent in the worst
//! way available: a token naming a dtype this build had never heard of decoded
//! as *unknown* rather than as *reserved*, which is precisely the distinction
//! §6.1-0001 exists to require.
//!
//! `kiss/dtype_manifest.json` is generated from `spec/classify.md`, so the
//! comparison can be exact. Equality is asserted **in both directions** — a
//! missing token and an extra one are different bugs with different blast
//! radii, and a set-difference in one direction hides the other.
//!
//! # Why the JSON is scanned by hand
//!
//! `unpopped-vocab`'s dependency graph is `half` + `float8` and nothing else,
//! deliberately — every consumer of the shared vocabulary inherits it. A
//! dev-dependency would not reach consumers, but the extractor needed for one
//! flat array of strings is twenty lines, and twenty lines is cheaper than a
//! dependency this crate would have to explain. The extractor is
//! positive-controlled below, since a scanner that silently returns nothing
//! would make every assertion here vacuous.

use unpopped_vocab::{ElementKind, dtype_token};

const MANIFEST: &str = include_str!("../kiss/dtype_manifest.json");

/// The `all_dtypes` array — KISS's closed §6.1 set, lexicographically sorted.
fn kiss_all_dtypes() -> Vec<String> {
    json_string_array(MANIFEST, "all_dtypes")
}

/// The set this crate actually implements, taken from the **codec** rather than
/// from the enum: `ElementKind` has 25 variants but only 24 distinct tokens,
/// because `F32Strict` folds to the canonical `f32` (sk3 D4 retired `f32s`; the
/// strict axis rides the `<mp>` coordinate). Comparing variant counts across
/// implementations manufactures phantom divergence — the token image is the only
/// thing that goes on the wire, so it is the only thing worth comparing.
fn our_all_dtypes() -> Vec<String> {
    let mut v: Vec<String> = ElementKind::ALL
        .iter()
        .map(|&k| dtype_token(k).to_string())
        .collect();
    v.sort();
    v.dedup();
    v
}

#[test]
fn the_dtype_set_matches_kiss_exactly_in_both_directions() {
    let kiss = kiss_all_dtypes();
    let ours = our_all_dtypes();

    let missing: Vec<_> = kiss.iter().filter(|t| !ours.contains(t)).collect();
    let extra: Vec<_> = ours.iter().filter(|t| !kiss.contains(t)).collect();

    assert!(
        missing.is_empty(),
        "tokens KISS defines that this crate cannot spell: {missing:?}\n\
         a peer using one of these gets `Unrecognized` where §6.1-0001 requires a \
         precise verdict"
    );
    assert!(
        extra.is_empty(),
        "tokens this crate spells that KISS does not define: {extra:?}\n\
         these would be emitted onto a shared wire and refused by every other deriver"
    );
    // Sorted and deduped on both sides, so this also pins the count.
    assert_eq!(ours, kiss, "the §6.1 token image must be identical");
}

/// The vendored manifest is the schema version this build implements.
///
/// A copy taken from a newer KISS fails here rather than quietly widening the
/// set — which is the whole reason to re-copy at a schema event instead of
/// trusting that someone would have mentioned it.
#[test]
fn the_vendored_manifest_is_the_schema_version_we_implement() {
    assert_eq!(
        json_number(MANIFEST, "structure_key_schema_version"),
        u64::from(unpopped_vocab::STRUCTURE_KEY_VERSION),
        "vendored manifest is from a different schema version than this build"
    );
    assert!(
        MANIFEST.contains("\"token_prefix\": \"sk4\""),
        "manifest token prefix must be the one this codec emits"
    );
    assert!(
        MANIFEST.contains("\"clause\": \"KISS-CLASSIFY-6.1-0001\""),
        "manifest must be the §6.1 closed-set artifact, not another corpus file"
    );
}

/// The reserved members match KISS's, and reserved-ness is not confused with
/// absence.
///
/// §6.1-0001 requires a reader to tell a reserved member of the shared
/// vocabulary apart from a token it has never heard of. That obligation is only
/// meaningful if both implementations agree on *which* members are reserved.
#[test]
fn the_reserved_members_match_kiss() {
    let kiss_reserved = reserved_tokens(MANIFEST);
    let ours_reserved: Vec<String> = {
        let mut v: Vec<String> = ElementKind::ALL
            .iter()
            .filter(|k| k.is_reserved())
            .map(|&k| dtype_token(k).to_string())
            .collect();
        v.sort();
        v.dedup();
        v
    };
    assert_eq!(
        ours_reserved, kiss_reserved,
        "the reserved set must match KISS exactly"
    );

    // And every reserved token is still a RECOGNIZED member of the closed set —
    // the distinction §6.1-0001 is about. Reserved must not mean absent.
    for t in &kiss_reserved {
        assert!(
            our_all_dtypes().contains(t),
            "reserved token `{t}` must still be spellable — reserved is not unknown"
        );
    }
}

/// Positive control on the extractor.
///
/// Every assertion above compares two lists this file produced. If
/// `json_string_array` returned an empty vector — a missing key, a renamed
/// field, a format change upstream — the comparisons would still pass against an
/// equally-empty other side, and this file would be decoration. So: the scanner
/// must find the real array, must find the right number of entries, and must
/// return nothing for a key that is not there.
#[test]
fn the_json_scanner_actually_reads_the_manifest() {
    let all = kiss_all_dtypes();
    assert_eq!(all.len(), 24, "KISS's closed §6.1 set is 24 members");
    assert!(all.contains(&"f8e4m3fn".to_string()), "a known member");
    assert!(
        all.contains(&"c128".to_string()),
        "the sk4 complex spelling"
    );
    assert!(
        !all.contains(&"c32".to_string()),
        "the sk3 complex spelling must be gone"
    );
    assert!(
        json_string_array(MANIFEST, "no_such_key_here").is_empty(),
        "an absent key must yield nothing, not something"
    );
    assert!(
        !reserved_tokens(MANIFEST).is_empty(),
        "reserved scan finds rows"
    );
}

// ---------------------------------------------------------------------------
// Minimal JSON scanning. Sufficient for this manifest's shape and no more.
// ---------------------------------------------------------------------------

/// The string array at top-level `key`, in file order.
fn json_string_array(src: &str, key: &str) -> Vec<String> {
    let Some(rest) = src.split_once(&format!("\"{key}\"")).map(|(_, r)| r) else {
        return Vec::new();
    };
    let Some(open) = rest.find('[') else {
        return Vec::new();
    };
    let Some(close) = rest[open..].find(']') else {
        return Vec::new();
    };
    quoted_strings(&rest[open..open + close])
}

/// The integer at top-level `key`.
fn json_number(src: &str, key: &str) -> u64 {
    src.split_once(&format!("\"{key}\": "))
        .and_then(|(_, r)| {
            let end = r.find(|c: char| !c.is_ascii_digit()).unwrap_or(r.len());
            r[..end].parse().ok()
        })
        .unwrap_or_else(|| panic!("manifest has no numeric `{key}`"))
}

/// Tokens of the `dtypes` rows whose `reserved` is `true`, sorted.
///
/// Each row is `{"token": ..., "kind": ..., "storage_bits": ..., "reserved": ...}`
/// in that fixed order, so a row's token is the last one seen before its
/// `reserved` flag.
fn reserved_tokens(src: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut token: Option<String> = None;
    for line in src.lines() {
        let t = line.trim();
        if let Some(v) = t.strip_prefix("\"token\": ") {
            token = quoted_strings(v).into_iter().next();
        } else if t.starts_with("\"reserved\": true") {
            if let Some(tok) = token.take() {
                out.push(tok);
            }
        }
    }
    out.sort();
    out
}

/// Every `"…"`-quoted run in `s`. The manifest contains no escapes, and a token
/// set that grew one would fail the equality tests loudly rather than silently.
fn quoted_strings(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut it = s.chars();
    while let Some(c) = it.next() {
        if c == '"' {
            let mut cur = String::new();
            for c2 in it.by_ref() {
                if c2 == '"' {
                    break;
                }
                cur.push(c2);
            }
            out.push(cur);
        }
    }
    out
}

/// **Every arm pinned by identity — because the set comparison above cannot see a
/// swap.**
///
/// # The hole this closes
///
/// [`our_all_dtypes`] collects `dtype_token` over `ElementKind::ALL`, then
/// **sorts and dedups**. That discards the mapping and keeps only the token
/// *set*. So if `dtype_token(I32)` returned `"f32"` and `dtype_token(F32)`
/// returned `"i32"` — a clean swap — the set is unchanged and
/// `the_dtype_set_matches_kiss_exactly_in_both_directions` still passes. The
/// same holds for any permutation of the table.
///
/// That is not a flaw in how that test was written; it is a property of
/// comparing sets. Its own doc says *"the token image is the only thing that
/// goes on the wire, so it is the only thing worth comparing"* — right about
/// **what** to compare, wrong about **how**, because the wire carries a
/// per-operand token derived from a specific `ElementKind`, not the set.
///
/// # Why the byte-match leg does not cover it either
///
/// `kiss_byte_match.rs` compares real tokens and would catch a swap — but only
/// for the dtypes its vectors exercise, recorded there as **3 in the dtype
/// position and 5 anywhere, against 22 usable**. That note says the remainder is
/// "caught by `kiss_dtype_manifest.rs` instead". For *membership* it is; for
/// *mapping* it was not, until this test. Two tests, each sound for its own
/// purpose, composing to leave ~19 dtypes where a swap was invisible to both.
///
/// # Why it matters more here than in most tables
///
/// A dtype token on the wire is an **identity**, not a classification: a swap
/// produces a well-formed, byte-stable token naming the wrong type, and a
/// consumer reinterprets the bytes with nothing raised at any layer. Width does
/// not help — this set clusters hard by width (`f16`/`bf16`/`i16`/`u16` at two
/// bytes; `f32`/`i32`/`u32` at four), so a same-width mis-route changes no size
/// anywhere.
///
/// And `baracuda-kernels-types` is `pub use unpopped_vocab::*` — it derives
/// nothing and inherits this table entire. **There is no second opinion
/// available downstream**, so a mis-route here cannot be caught anywhere else.
///
/// Rule and remedy owed to MLMF, who hit the same shape in a ggml type table
/// where `{BF16, F16, I16}` share two bytes.
#[test]
fn every_dtype_arm_is_pinned_by_identity_not_by_set_membership() {
    use ElementKind::*;

    // The complete table, one line per variant. `F32` and `F32Strict`
    // deliberately share `f32` — the strict axis rides the `<mp>` coordinate,
    // not the dtype token — which is why the set comparison must dedup and why
    // this pin, not that one, is the mapping's guard.
    let pinned: &[(ElementKind, &str)] = &[
        (F16, "f16"),
        (Bf16, "bf16"),
        (F32, "f32"),
        (F32Strict, "f32"),
        (F64, "f64"),
        (I8, "i8"),
        (U8, "u8"),
        (I16, "i16"),
        (U16, "u16"),
        (I32, "i32"),
        (U32, "u32"),
        (I64, "i64"),
        (U64, "u64"),
        (Bool, "bool"),
        (I4, "i4"),
        (U4, "u4"),
        (B1, "b1"),
        (Fp8E4M3FN, "f8e4m3fn"),
        (Fp8E5M2, "f8e5m2"),
        (Fp8E4M3FNUZ, "f8e4m3fnuz"),
        (Fp8E5M2FNUZ, "f8e5m2fnuz"),
        (F8E8M0, "f8e8m0"),
        (F8E6M2, "f8e6m2"),
        (Complex64, "c64"),
        (Complex128, "c128"),
    ];

    for (kind, token) in pinned {
        assert_eq!(
            dtype_token(*kind),
            *token,
            "{kind:?} must spell `{token}`. A same-width neighbour's token here \
             is a well-formed wire token naming the wrong type, and every \
             set-based and size-based check in this crate would stay green."
        );
    }

    // Exhaustiveness: a new variant must be pinned here deliberately rather than
    // inherit a spelling and slip past on set membership.
    assert_eq!(
        pinned.len(),
        ElementKind::ALL.len(),
        "ElementKind has {} variants but {} are pinned — add the new one above, \
         with its token, rather than relying on the set comparison",
        ElementKind::ALL.len(),
        pinned.len()
    );
    for k in ElementKind::ALL {
        assert!(
            pinned.iter().any(|(p, _)| *p == k),
            "{k:?} is in ElementKind::ALL but not pinned by identity here"
        );
    }
}
