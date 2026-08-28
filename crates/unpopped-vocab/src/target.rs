//! Open target identity — `<namespace>:<capability-set>`, interned.
//!
//! # What this replaces, and why the closed enum had to go
//!
//! [`crate::ArchSku`] is a CUDA-only enum: four `sm*` variants and no way to
//! spell `vulkan:`, `rocm:` or `metal:` at all. That is a **vocabulary owned by
//! one vendor sitting inside a crate whose entire claim is neutrality**, and it
//! had two measured costs beyond the aesthetic one:
//!
//! * A conforming `vulkan:` reference vector could not be represented, so it was
//!   *excluded* from the cross-project byte-match rather than matched.
//! * Nothing could express a target-conditional capability, because there was no
//!   target to condition on.
//!
//! # Why interned, rather than a `String` or a `Cow`
//!
//! [`crate::StructureKey`] documents itself as `Copy` and heap-free "so it can
//! be hashed into a dispatch table or an autotuner cache directly", and a
//! downstream consumer caches on exactly that. A `String` field would take both
//! properties away to buy openness. A [`TargetId`] is a `u16` handle into a
//! process-wide table, so the key stays 80 bytes and keeps every property it
//! advertises, while the *set* of targets becomes open.
//!
//! # The invariant that makes interning safe
//!
//! **A [`TargetId`]'s numeric value MUST NOT be serialized, persisted, or sent
//! anywhere.** Ids are assigned in registration order, which differs between
//! processes: the same token can be `7` here and `4` there. The wire form is
//! always the token string ([`TargetId::as_str`]), which is what
//! [`crate::StructureKey::to_token`] emits and what §6.8-0002 matches
//! byte-exactly. `Serialize`/`Deserialize` are therefore deliberately NOT
//! implemented on this type, and `to_token`/`from_token` are the only crossing
//! points. See `id_values_are_process_local` for the test that pins it.
//!
//! # What KISS pins, and what it leaves to namespace owners
//!
//! Read from the merged KISS-Classify text rather than inferred:
//!
//! * **§6.8-0001** — the token is `<namespace>:<capability-set>`: non-empty
//!   namespace, **exactly one** `:`, non-empty capability-set. Zero or more than
//!   one `:`, or either side empty, is a typed decline.
//! * **§6.8-0002** — matching is **byte-exact on the full string**. No ordering,
//!   subset, prefix, or feature-implication logic. This crate gets that for free
//!   by interning: equal tokens share an id, so `==` on `TargetId` *is* the
//!   byte-exact comparison.
//! * **§6.8-0005** — case-sensitive ASCII; must not contain the `structure_key`
//!   field separators `|`, `;`, `/`, nor whitespace or control bytes
//!   (`0x00`–`0x20`, `0x7f`), so the token embeds in a key as one unambiguous
//!   field.
//! * **§6.8-0003 / §6.8-0004** — the namespace must be registered with the
//!   steward, and each namespace's capability-set vocabulary is **its
//!   maintainer's**, never pinned by KISS. This crate therefore validates the
//!   *grammar* and never the *vocabulary*: it will accept `rocm:gfx942` without
//!   knowing what a gfx942 is, which is the correct behaviour for a module that
//!   does not own that namespace.
//!
//! Enforcement of §6.8-0003's registry is explicitly **not** done here. The
//! clause scopes registration to the moment a party first *produces* a kernel
//! under a namespace, and says byte-exact matching MUST NOT consult the
//! registry. A generator that refused to *decode* an unregistered namespace
//! would be reading a producer-side rule as a reader-side one.

use core::fmt;
use std::sync::{OnceLock, RwLock};

use crate::layout::ArchSku;

/// The four CUDA tokens, interned at fixed ids so they are const-nameable and
/// their byte spelling cannot drift.
///
/// Order is load-bearing: it defines the reserved id block, so a token may be
/// **appended** but never reordered or removed. (Ids are process-local by
/// contract — see the module docs — so this is about keeping [`ArchSku`]'s
/// mapping total and cheap, not about wire stability, which the strings carry.)
///
/// # SCHEDULED FOR REMOVAL — decided 2026-08-15 with the `cuda` maintainer
///
/// **These four strings are the last CUDA vocabulary in this neutral crate**, and
/// they are here purely as an optimization: they let `From<ArchSku>` be a const
/// index instead of a lookup. That is a poor trade for a crate whose claim is
/// neutrality, and Baracuda (who owns the `cuda:` vocabulary under §6.8-0004)
/// agreed to **drop the block** — the conversion goes through
/// [`TargetId::parse`] instead, at registration time rather than on any hot
/// path.
///
/// It has not happened yet because the eviction must **lock-step with a registry
/// repoint**: KISS's `conformance/registry/namespaces.json` names
/// `unpopped-vocab` as the `cuda` namespace's `reference_implementation`, so
/// removing the tokens before that pointer moves to `baracuda-cuda-vocab` breaks
/// a PR-gated file. Sequencing is Baracuda's; this crate's side is four sites
/// and no codec work — the `structure_key` codec stopped baking `ArchSku`
/// entirely when `TargetId` landed.
///
/// The one thing that changes here when it goes: `From<ArchSku>` stops being a
/// const index, and `id_values_are_process_local`'s
/// `TargetId::from(ArchSku::Sm80).0 == 0` assertion goes with it — which is
/// fine, since that test's own doc says ids are registration handles rather than
/// stable names.
const RESERVED: &[&str] = &["cuda:sm80", "cuda:sm89", "cuda:sm90", "cuda:sm90a"];

/// An interned `target_capability` token (KISS-CLASSIFY §6.8).
///
/// `Copy`, 2 bytes, and comparable with `==` — which by construction is the
/// byte-exact match §6.8-0002 requires, since equal tokens intern to equal ids.
///
/// Obtain one with [`TargetId::parse`] (validating, for an arbitrary token) or
/// from an [`ArchSku`] via `From` (infallible, for the four CUDA targets).
/// Recover the token with [`TargetId::as_str`].
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct TargetId(u16);

/// Why a `target_capability` token was refused.
///
/// Each variant names the clause it enforces, because a decline whose reason a
/// reader cannot map back to the standard is a decline they will argue with.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum TargetError {
    /// Not exactly one `:` (§6.8-0001). Carries how many were found.
    Separators(usize),
    /// The namespace side is empty (§6.8-0001).
    EmptyNamespace,
    /// The capability-set side is empty (§6.8-0001).
    EmptyCapabilitySet,
    /// A byte §6.8-0005 forbids: a `structure_key` field separator (`|`, `;`,
    /// `/`), whitespace, a control byte, or non-ASCII. Carries the offending
    /// byte and its offset.
    ForbiddenByte {
        /// The byte that was rejected.
        byte: u8,
        /// Its zero-based offset in the token.
        at: usize,
    },
    /// The intern table is full.
    ///
    /// Not a KISS rule — an implementation limit of this crate, reported
    /// distinctly so it is never mistaken for a malformed token. Reaching it
    /// takes 65 535 *distinct* target tokens in one process, which is a bug in
    /// the caller (registering per-request rather than per-target) far sooner
    /// than it is a real workload.
    TableFull,
}

impl fmt::Display for TargetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Separators(n) => write!(
                f,
                "target token must have exactly one ':' (KISS-CLASSIFY §6.8-0001), found {n}"
            ),
            Self::EmptyNamespace => {
                write!(f, "target token has an empty namespace (§6.8-0001)")
            }
            Self::EmptyCapabilitySet => {
                write!(f, "target token has an empty capability-set (§6.8-0001)")
            }
            Self::ForbiddenByte { byte, at } => write!(
                f,
                "target token byte {byte:#04x} at offset {at} is forbidden by §6.8-0005 \
                 (structure_key separators '|', ';', '/', whitespace, control, non-ASCII)"
            ),
            Self::TableFull => write!(f, "target intern table is full (65535 distinct targets)"),
        }
    }
}

impl std::error::Error for TargetError {}

/// The process-wide intern table.
///
/// `RwLock<Vec<String>>` rather than anything cleverer: registration happens
/// once per distinct target (a handful of times in a process), reads are
/// frequent but short, and correctness here is worth more than contention that
/// no realistic workload generates.
fn table() -> &'static RwLock<Vec<String>> {
    static TABLE: OnceLock<RwLock<Vec<String>>> = OnceLock::new();
    TABLE.get_or_init(|| RwLock::new(RESERVED.iter().map(|s| (*s).to_string()).collect()))
}

impl TargetId {
    /// Validate a `target_capability` token and intern it.
    ///
    /// Idempotent: the same token always returns the same id within a process.
    /// Validation is the §6.8 **grammar** only — a namespace's capability-set
    /// vocabulary belongs to its maintainer (§6.8-0004), so `rocm:gfx942` is
    /// accepted here without this crate knowing what a gfx942 is.
    ///
    /// # Errors
    ///
    /// [`TargetError`] naming the clause the token violates.
    pub fn parse(token: &str) -> Result<Self, TargetError> {
        validate(token)?;
        // Read-lock first: the overwhelmingly common case is a target already
        // interned, and taking the write lock to discover that would serialize
        // every key construction in the process behind one mutex.
        if let Some(id) = table()
            .read()
            .expect("target table poisoned")
            .iter()
            .position(|s| s == token)
        {
            return Ok(Self(id as u16));
        }
        let mut t = table().write().expect("target table poisoned");
        // Re-check under the write lock: another thread may have interned this
        // token between the two acquisitions, and a duplicate entry would break
        // the "equal tokens are equal ids" invariant that makes `==` a
        // byte-exact match.
        if let Some(id) = t.iter().position(|s| s == token) {
            return Ok(Self(id as u16));
        }
        if t.len() > u16::MAX as usize {
            return Err(TargetError::TableFull);
        }
        t.push(token.to_string());
        Ok(Self((t.len() - 1) as u16))
    }

    /// The token this id stands for, e.g. `"cuda:sm89"`.
    ///
    /// This is the **only** form that may cross a process boundary; the numeric
    /// id is process-local (module docs).
    #[must_use]
    pub fn as_str(&self) -> String {
        table().read().expect("target table poisoned")[self.0 as usize].clone()
    }

    /// The namespace component — `"cuda"` for `"cuda:sm89"`.
    ///
    /// Whose vocabulary the capability side belongs to (§6.8-0004).
    #[must_use]
    pub fn namespace(&self) -> String {
        let s = self.as_str();
        s[..s.find(':').expect("interned tokens are validated")].to_string()
    }

    /// The capability-set component — `"sm89"` for `"cuda:sm89"`.
    #[must_use]
    pub fn capability_set(&self) -> String {
        let s = self.as_str();
        s[s.find(':').expect("interned tokens are validated") + 1..].to_string()
    }

    /// The tuples carried by one named capability field, if the token has it.
    ///
    /// `vulkan:sg64.ops-abr.arith-f16-i8.cm-none` with `field = "arith"` gives
    /// `["f16", "i8"]`; `arith-none` gives `[]`; a token without the field gives
    /// `None`. **Absent and empty are different answers** — "this token does not
    /// speak about arithmetic" is not "this target does no arithmetic".
    ///
    /// # Grammar, which is not this crate's to invent
    ///
    /// Fields separate on `.`, tuples within a field on `-`, and an empty set is
    /// spelled `<field>-none`. **Juxtaposition (`arith-f16i8`) is malformed**, not
    /// merely unusual: tuple names are variable-length, so juxtaposition is only
    /// decodable while the name set happens to stay uniquely decodable as it
    /// grows, which nothing checks. Owned by the namespace's vocabulary — for
    /// `vulkan:` that is Vulkane, V-6.
    ///
    /// # This returns tuples in the order written, and that matters
    ///
    /// §6.8-0002 matching is **byte-exact**, and a set is *spelled* in
    /// lexicographic name order. So a caller BUILDING a token must sort before
    /// joining — `arith-i8-f16` is well-formed and matches nothing. This reader
    /// does not sort for you, because silently accepting an unsorted token would
    /// hide exactly the bug that a byte-exact match exists to catch.
    #[must_use]
    pub fn capability_field(&self, field: &str) -> Option<Vec<String>> {
        capability_field_of(&self.capability_set(), field)
    }
}

/// [`TargetId::capability_field`] over a bare capability set.
///
/// Split out so the grammar is testable without interning a token, and so a
/// caller holding a set from elsewhere (a manifest, a probe) can use it.
#[must_use]
pub fn capability_field_of(capability_set: &str, field: &str) -> Option<Vec<String>> {
    for part in capability_set.split('.') {
        // Exact field match, never a prefix: a bare `contains` would let a
        // future `arith2-...` answer a query for `arith`, and that class of
        // substring collision has cost this workspace real time before.
        let Some(rest) = part.strip_prefix(field) else {
            continue;
        };
        let Some(rest) = rest.strip_prefix('-') else {
            // `field` matched a longer name (`arith` vs `arithmetic`), or the
            // field carries no `-` at all. Neither is this field.
            continue;
        };
        if rest == "none" {
            return Some(Vec::new());
        }
        return Some(rest.split('-').map(str::to_string).collect());
    }
    None
}

/// The §6.8-0001 and §6.8-0005 grammar, split out so it is testable without
/// touching the intern table.
fn validate(token: &str) -> Result<(), TargetError> {
    // §6.8-0005 first: a control byte or a `|` would otherwise be reported as a
    // shape problem, and the charset violation is the more specific truth.
    for (at, b) in token.bytes().enumerate() {
        let forbidden =
            !b.is_ascii() || b <= 0x20 || b == 0x7f || b == b'|' || b == b';' || b == b'/';
        if forbidden {
            return Err(TargetError::ForbiddenByte { byte: b, at });
        }
    }
    let colons = token.bytes().filter(|&b| b == b':').count();
    if colons != 1 {
        return Err(TargetError::Separators(colons));
    }
    let (ns, cap) = token.split_once(':').expect("exactly one colon");
    if ns.is_empty() {
        return Err(TargetError::EmptyNamespace);
    }
    if cap.is_empty() {
        return Err(TargetError::EmptyCapabilitySet);
    }
    Ok(())
}

impl From<ArchSku> for TargetId {
    /// Infallible: the four CUDA tokens occupy the reserved id block, so this is
    /// an index rather than a lookup and cannot fail or allocate.
    ///
    /// This conversion is what keeps every existing CUDA call site compiling
    /// unchanged through the target-model opening — `structure_key(.., Sm89)`
    /// still works, because the parameter takes `impl Into<TargetId>`.
    fn from(a: ArchSku) -> Self {
        Self(match a {
            ArchSku::Sm80 => 0,
            ArchSku::Sm89 => 1,
            ArchSku::Sm90 => 2,
            ArchSku::Sm90a => 3,
        })
    }
}

impl fmt::Display for TargetId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The four CUDA tokens keep their exact bytes through the opening.
    ///
    /// This is the regression that would silently break every cross-project
    /// byte-match vector at once, and it would break them by producing a
    /// *well-formed* key with a different spelling — which no grammar check
    /// catches.
    #[test]
    fn the_cuda_tokens_are_byte_identical_to_the_closed_enums() {
        for (sku, want) in [
            (ArchSku::Sm80, "cuda:sm80"),
            (ArchSku::Sm89, "cuda:sm89"),
            (ArchSku::Sm90, "cuda:sm90"),
            (ArchSku::Sm90a, "cuda:sm90a"),
        ] {
            assert_eq!(TargetId::from(sku).as_str(), want);
            // And the round trip through the open path lands on the same id, so
            // a key built from a parsed token equals one built from the enum.
            assert_eq!(TargetId::parse(want).unwrap(), TargetId::from(sku));
        }
    }

    /// A namespace this crate knows nothing about is accepted.
    ///
    /// The point of the whole change: §6.8-0004 puts the capability-set
    /// vocabulary in its maintainer's hands, so validating it here would be
    /// this crate claiming ownership of a namespace it does not own.
    #[test]
    fn foreign_namespaces_are_accepted_without_being_understood() {
        for token in [
            "vulkan:spirv1.6",
            "rocm:gfx942",
            "rocm:gfx1100",
            "metal:apple9",
            "cuda:sm100a",
            // Dots, dashes and digits are all legal capability-set bytes; this
            // is the shape a real Vulkan capability token takes.
            "vulkan:sg64.ops-abr.arith-f16.cm-none",
        ] {
            let id = TargetId::parse(token).unwrap_or_else(|e| panic!("{token}: {e}"));
            assert_eq!(id.as_str(), token);
        }
    }

    /// Interning is idempotent, which is what makes `==` a byte-exact match.
    #[test]
    fn equal_tokens_intern_to_equal_ids_and_different_ones_do_not() {
        let a = TargetId::parse("rocm:gfx90a").unwrap();
        let b = TargetId::parse("rocm:gfx90a").unwrap();
        let c = TargetId::parse("rocm:gfx90A").unwrap();
        assert_eq!(a, b, "the same token must intern once");
        // §6.8-0005: case-sensitive. `gfx90a` and `gfx90A` are different
        // targets, and a case-insensitive compare would silently merge them.
        assert_ne!(a, c, "matching is case-SENSITIVE (§6.8-0002/-0005)");
    }

    /// §6.8-0001: the separator count and both non-empty sides.
    #[test]
    fn the_token_grammar_declines_with_the_clause_it_enforces() {
        assert_eq!(TargetId::parse("sm89"), Err(TargetError::Separators(0)));
        assert_eq!(
            TargetId::parse("cuda:sm89:extra"),
            Err(TargetError::Separators(2))
        );
        assert_eq!(TargetId::parse(":sm89"), Err(TargetError::EmptyNamespace));
        assert_eq!(
            TargetId::parse("cuda:"),
            Err(TargetError::EmptyCapabilitySet)
        );
        assert_eq!(TargetId::parse(""), Err(TargetError::Separators(0)));
    }

    /// §6.8-0005: the charset, including the three `structure_key` separators.
    ///
    /// `|`, `;` and `/` matter beyond tidiness — each is a field separator in
    /// the key's own token grammar, so a target carrying one would not merely
    /// look odd, it would **re-split the key** into a different set of fields
    /// that still parses.
    #[test]
    fn the_charset_rejects_bytes_that_would_re_split_a_key_token() {
        for (token, at) in [
            ("cuda:sm89|x", 9),
            ("cuda:sm89;x", 9),
            ("cuda:sm/89", 7),
            ("cuda:sm 89", 7),
            ("cuda:sm\t89", 7),
            ("cuda:sm\n89", 7),
            ("cuda:sm\u{7f}89", 7),
        ] {
            match TargetId::parse(token) {
                Err(TargetError::ForbiddenByte { at: got, .. }) => {
                    assert_eq!(got, at, "{token:?}: wrong offset reported");
                }
                other => panic!("{token:?} must decline on charset, got {other:?}"),
            }
        }
        // Non-ASCII is out too — §6.8-0005 says ASCII, and a multi-byte
        // character would also make byte offsets and char offsets disagree.
        assert!(matches!(
            TargetId::parse("cuda:smé"),
            Err(TargetError::ForbiddenByte { .. })
        ));
    }

    /// The namespace / capability split is the one §6.8-0001 describes.
    #[test]
    fn the_token_splits_where_the_clause_says_it_does() {
        let id = TargetId::parse("vulkan:sg64.ops-abr").unwrap();
        assert_eq!(id.namespace(), "vulkan");
        assert_eq!(id.capability_set(), "sg64.ops-abr");
    }

    /// **The interning contract**: an id is a process-local handle, never data.
    ///
    /// A single process cannot observe another process's numbering, so what this
    /// pins is the property that *implies* the invariant: an id is a position in
    /// a registration-ordered table, and a token's position depends on what else
    /// was registered first. Two processes that register in different orders
    /// give the same token different ids — which is precisely why
    /// `StructureKey::to_token` must emit [`TargetId::as_str`] and never `.0`.
    ///
    /// # Why this asserts a range and not `first + 1`
    ///
    /// The table is process-global and Rust runs tests **concurrently**, so
    /// another test interning a token between these two calls is legal and
    /// expected. An `assert_eq!(second.0, first.0 + 1)` passes on a quiet run
    /// and fails under load — a flake that would look like a bug in the intern
    /// table rather than in the test. Ordering is all that is actually
    /// guaranteed, so ordering is all that is asserted.
    #[test]
    fn id_values_are_process_local_and_must_not_be_serialized() {
        let first = TargetId::parse("proclocal:aaa").unwrap();
        let second = TargetId::parse("proclocal:bbb").unwrap();
        assert_ne!(first, second);
        assert!(
            second.0 > first.0,
            "ids are registration-ordered handles, not stable names"
        );
        // Both sit past the reserved block, which is the part that IS stable —
        // and is what lets `From<ArchSku>` be a const index rather than a lookup.
        assert!(first.0 >= RESERVED.len() as u16);
        assert_eq!(TargetId::from(ArchSku::Sm80).0, 0);
    }

    /// Validation happens before interning, so a bad token cannot enter the table.
    ///
    /// Without this a malformed-token loop is an unbounded memory leak that
    /// eventually reports `TableFull` — a confusing second-order symptom for
    /// what is really a rejected input.
    ///
    /// Asserts **membership**, not the table's length: the table is
    /// process-global and tests run concurrently, so its length moves under this
    /// test for reasons that have nothing to do with it. (Measured — the
    /// length form failed with `left: 6, right: 4` while every token under test
    /// was correctly declined.)
    #[test]
    fn a_rejected_token_is_never_interned() {
        let bad = ["nocolon", "two::colons", ":empty", "empty:", "bad|byte:x"];
        for b in bad {
            assert!(TargetId::parse(b).is_err(), "{b:?} must decline");
        }
        let t = table().read().unwrap();
        for b in bad {
            assert!(
                !t.iter().any(|s| s == b),
                "declined token {b:?} reached the intern table"
            );
        }
    }

    /// The multi-value form, which is the one nothing in the wild exercises.
    ///
    /// Vulkane's own normative vector set carries `arith-none` in all eight
    /// vectors, so a parser written against the shipped artifact alone would
    /// never see more than one tuple. This is that case.
    #[test]
    fn a_field_yields_its_tuples_in_written_order() {
        let set = "sg64.ops-abr.arith-f16-i8.cm-none";
        assert_eq!(
            capability_field_of(set, "arith"),
            Some(vec!["f16".to_string(), "i8".to_string()])
        );
        assert_eq!(
            capability_field_of(set, "ops"),
            Some(vec!["abr".to_string()])
        );
    }

    /// **Order is preserved, never sorted.**
    ///
    /// §6.8-0002 matching is byte-exact and a set is *spelled* in lexicographic
    /// order, so a caller BUILDING a token must sort before joining. If this
    /// reader silently sorted, `arith-i8-f16` would round-trip looking correct
    /// and still match nothing at the consumer — hiding the exact defect the
    /// byte-exact rule exists to expose.
    #[test]
    fn the_reader_does_not_silently_sort_a_misordered_set() {
        assert_eq!(
            capability_field_of("arith-i8-f16", "arith"),
            Some(vec!["i8".to_string(), "f16".to_string()]),
            "the reader reordered a malformed token into a valid-looking one"
        );
    }

    /// Absent and empty are different answers.
    ///
    /// `None` is "this token does not speak about arithmetic"; `Some([])` is
    /// "this target advertises no 8-bit or 16-bit arithmetic". Collapsing them
    /// would make a silent token indistinguishable from a denying one — the
    /// same absence-is-silence-not-denial distinction the vocabulary itself
    /// draws.
    #[test]
    fn absent_and_empty_are_distinguished() {
        assert_eq!(
            capability_field_of("sg64.arith-none", "arith"),
            Some(vec![])
        );
        assert_eq!(capability_field_of("sg64.ops-abr", "arith"), None);
    }

    /// A longer field name never answers a shorter query.
    ///
    /// Without the `-` check, `strip_prefix("arith")` accepts `arithmetic-x` and
    /// `arith2-x`. A `contains`-based reader would too. That substring class has
    /// cost this workspace real time before.
    #[test]
    fn a_field_query_does_not_match_a_longer_field_name() {
        assert_eq!(capability_field_of("arith2-f16", "arith"), None);
        assert_eq!(capability_field_of("arithmetic-f16", "arith"), None);
        assert_eq!(
            capability_field_of("sg64", "sg"),
            None,
            "no separator, no field"
        );
    }

    /// A juxtaposed set is returned as written — one tuple, not two.
    ///
    /// `arith-f16i8` is **malformed** per V-6, and the honest reading of
    /// malformed input is what it literally says. Splitting it into `f16` and
    /// `i8` would be this crate inventing a grammar the namespace owner
    /// explicitly forbade, and would make a malformed token work by accident.
    #[test]
    fn a_juxtaposed_set_is_not_helpfully_split() {
        assert_eq!(
            capability_field_of("arith-f16i8", "arith"),
            Some(vec!["f16i8".to_string()]),
            "the reader split a juxtaposed set and made malformed input work"
        );
    }

    #[test]
    fn the_field_reader_works_through_an_interned_token() {
        let t = TargetId::parse("vulkan:sg32.arith-f16-i8").expect("valid token");
        assert_eq!(
            t.capability_field("arith"),
            Some(vec!["f16".to_string(), "i8".to_string()])
        );
        assert_eq!(t.capability_field("cm"), None);
    }
}
