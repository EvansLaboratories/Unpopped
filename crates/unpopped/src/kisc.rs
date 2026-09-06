//! KISC self-delimiting contract framing (KISS-Contract §2.8 / §6.11).
//!
//! A KISC document is a **structured/text** frame — not a binary envelope: a
//! single header line carrying the magic, kind, version, inner body byte-length,
//! and a CRC-32 over the body, followed by the body itself. A reader hard-rejects
//! (never repairs) a document whose magic is wrong, whose header is malformed,
//! whose kind/version is unrecognized, or whose declared length/CRC don't match —
//! which is what kills the silent-adopt-empty and truncation hazards the older
//! `## `-heading framing was prone to.
//!
//! # Header-line format — PINNED BY §6.11, AND THIS FILE SAID OTHERWISE
//!
//! ⚠️ **This section used to call the spelling below a "strawman" and claim
//! §6.11 pinned the header's FIELDS but not the exact literal bytes. That was
//! FALSE, and it was false on the day it was written — not stale.** Corrected
//! 2026-09-06 after baracuda refuted it and the KISS architect confirmed;
//! re-measured here rather than relayed.
//!
//! **`spec/contract.md` at KISS `origin/main`, read directly:**
//!
//! - **§6.11-0002 (contract.md:1348)** pins the magic **as bytes** —
//!   `0x4B 0x49 0x53 0x43` — then one space, `kiss-contract`, one space, the
//!   decimal version, one space, `len=<N>`, one space, `crc32=<HHHHHHHH>`, then a
//!   **single LF (`0x0A`)**, with the literal example line spelled out.
//! - **§6.11-0003 (contract.md:1355)** pins the CRC fully: **IEEE 802.3
//!   polynomial, reflected, initial `0xFFFFFFFF`, final XOR `0xFFFFFFFF`, as 8
//!   LOWERCASE hex digits**, over exactly the `N` body bytes.
//!
//! **No free parameter remains** — including reflection and final XOR, the two
//! nobody could have guessed.
//!
//! **Both entered `contract.md` on 2026-07-13; the strawman note was dated
//! 2026-08-08.** ⚠️ **Pinned twenty-six days before this file asserted they were
//! not.**
//!
//! # ⚠️ AND THE IMPLEMENTATION CONFORMS TO THE SPEC ITS OWN DOC DENIED
//!
//! Measured here:
//!
//! ```text
//! kisc_frame("body")     ->  KISC kiss-contract 1 len=4 crc32=dba80bb2

//! crc32(b"123456789")    ->  0xCBF43926   the published CRC-32 check value
//! ```
//!
//! **The emitted form matches §6.11-0002 byte for byte, `{:08x}` gives the
//! lowercase §6.11-0003 requires, and the check value proves the CRC variant.**
//! Pinned by `the_kisc_framing_conforms_to_the_pinned_spelling`.
//!
//! # How it happened, because the mechanism is the transferable part
//!
//! ⚠️ **The note MEASURED FUEL AND INFERRED KISS.** Its Fuel findings are careful,
//! dated and — as far as anyone knows — still true: their KISC reply of
//! 2026-07-14 predates the 2026-07-15 ask for the exact spelling, it adopts the
//! framing and says nothing about field order, hex case or CRLF tolerance, and
//! Fuel has **no KISC implementation at all** (zero hits for `KISC`/`crc32`).
//!
//! **All of that is a fact about the party holding no code. It was used to
//! support a claim about the party that owns the document, and the document was
//! never opened.**
//!
//! ⚠️ **The note even states the rule it breaks** — *"reading Baracuda's side
//! alone would have recorded 'Fuel confirms' about a party holding no opinion,
//! because it holds no code."* **It read the parties holding no code and did not
//! read the spec.**
//!
//! **What the Fuel observation actually supports, and is worth keeping:** no
//! importer has ever exercised this framing. That is *unexercised*, not
//! *unpinned* — the same distinction as an empty affected set meaning "nobody is
//! affected" versus "nobody has implemented it yet".
//!
//! **The isolation in [`kisc_frame`]/[`kisc_unframe`] stays** — as an ordinary
//! single-point-of-change, not as a hedge against a format that was never open.
//!
//! **Where it gets pinned: KISS, not bilaterally.** `KISC` is KISS-Contract
//! §2.8/§6.11 vocabulary, and KISS-owned vocabulary is imported from KISS rather
//! than re-invented downstream. A spelling agreed between Unpopped and Fuel would
//! be a *third* spelling of a frame neither owns. The proposal to take there is
//! the field list plus the `0xCBF43926` ("123456789") CRC-32 validation vector.
//!
//! ```text
//! KISC kiss-contract 1 len=<N> crc32=<8 lowercase hex>\n<body of N bytes>
//! ```

/// The pinned contract kind (KISS-Contract §6.11).
pub const KISC_KIND: &str = "kiss-contract";
/// The KISC document version this seed emits and accepts.
pub const KISC_VERSION: u32 = 1;

/// Why a KISC document is rejected. Hard-reject discipline (§6.11): the reader
/// never repairs and never adopts a malformed document as an empty/no-op contract.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KiscError {
    /// The document does not begin with the `KISC` magic.
    BadMagic,
    /// The header line is absent or structurally malformed.
    BadHeader,
    /// The `contract_kind` is not `kiss-contract`.
    UnknownKind,
    /// The `contract_version` is not one this seed understands.
    UnknownVersion,
    /// The declared `len=<N>` does not equal the actual body byte length.
    LenMismatch,
    /// The declared `crc32=<…>` does not match the body's CRC-32.
    CrcMismatch,
}

/// CRC-32 (IEEE 802.3, reflected, `0xEDB8_8320`), the checksum KISS §6.11 names.
#[must_use]
pub fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &b in bytes {
        crc ^= u32::from(b);
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
        }
    }
    !crc
}

/// Frame `body` as a single KISC document (see the module header for the format).
#[must_use]
pub fn kisc_frame(body: &str) -> String {
    format!(
        "KISC {KISC_KIND} {KISC_VERSION} len={} crc32={:08x}\n{body}",
        body.len(),
        crc32(body.as_bytes()),
    )
}

/// Parse + validate a single KISC document, returning its body on success or a
/// typed hard-reject. Never panics.
pub fn kisc_unframe(doc: &str) -> Result<&str, KiscError> {
    let (header, body) = doc.split_once('\n').ok_or(KiscError::BadHeader)?;
    let mut f = header.split(' ');
    match f.next() {
        Some("KISC") => {}
        _ => return Err(KiscError::BadMagic),
    }
    if f.next() != Some(KISC_KIND) {
        return Err(KiscError::UnknownKind);
    }
    match f.next().and_then(|v| v.parse::<u32>().ok()) {
        Some(KISC_VERSION) => {}
        _ => return Err(KiscError::UnknownVersion),
    }
    let len = f
        .next()
        .and_then(|s| s.strip_prefix("len="))
        .and_then(|s| s.parse::<usize>().ok())
        .ok_or(KiscError::BadHeader)?;
    let crc = f
        .next()
        .and_then(|s| s.strip_prefix("crc32="))
        .and_then(|s| u32::from_str_radix(s, 16).ok())
        .ok_or(KiscError::BadHeader)?;
    if f.next().is_some() {
        return Err(KiscError::BadHeader);
    }
    if body.len() != len {
        return Err(KiscError::LenMismatch);
    }
    if crc32(body.as_bytes()) != crc {
        return Err(KiscError::CrcMismatch);
    }
    Ok(body)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc32_matches_the_standard_check_value() {
        // The canonical CRC-32/ISO-HDLC check value for "123456789" — which is
        // the variant KISS-CONTRACT §6.11-0003 pins BY PARAMETER (IEEE 802.3
        // polynomial, reflected, initial 0xFFFFFFFF, final XOR 0xFFFFFFFF). One
        // value determines all four, which is why a check VALUE beats a
        // description of an ALGORITHM.
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
    }

    #[test]
    fn frame_carries_magic_kind_version_len_and_body() {
        let f = kisc_frame("hello");
        assert!(
            f.starts_with("KISC kiss-contract 1 len=5 crc32="),
            "header line: {f:?}"
        );
        assert!(
            f.ends_with("\nhello"),
            "body follows the header line: {f:?}"
        );
    }

    #[test]
    fn frame_unframe_round_trips_a_multiline_body() {
        // A realistic body has interior newlines (the seven contract sections).
        let body = "kernel: relu_add\nop_kind: ReluAddElementwise\naccept: sk4|...\n";
        let doc = kisc_frame(body);
        assert_eq!(kisc_unframe(&doc), Ok(body));
    }

    #[test]
    fn unframe_hard_rejects_a_magic_less_document() {
        // A `## `-heading document (the OLD framing) is magic-less → rejected, not
        // silently adopted as an empty contract.
        assert_eq!(
            kisc_unframe("## heading\n\nkernel: x\n"),
            Err(KiscError::BadMagic)
        );
        // No header line at all.
        assert_eq!(kisc_unframe(""), Err(KiscError::BadHeader));
    }

    #[test]
    fn unframe_hard_rejects_a_corrupted_body() {
        let mut doc = kisc_frame("kernel: relu_add\n");
        // Flip a byte in the body; the CRC (or length) must catch it.
        doc.push('X');
        assert!(matches!(
            kisc_unframe(&doc),
            Err(KiscError::LenMismatch) | Err(KiscError::CrcMismatch)
        ));
    }

    /// Tiny deterministic PRNG for fuzz coverage — no external dependency.
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0
        }
    }

    #[test]
    fn unframe_never_panics_on_arbitrary_input() {
        let mut rng = Lcg(0x0000_B16C);
        // (1) Random bytes INCLUDING '\n' (~1/97) — without a newline every input
        // short-circuits at the first `split_once('\n')`, so the header field
        // parsers (magic/kind/version/len/crc) would never see random fuzz.
        for _ in 0..5000 {
            let len = (rng.next() % 200) as usize;
            let s: String = (0..len)
                .map(|_| {
                    let r = rng.next() % 97;
                    if r == 0 {
                        '\n'
                    } else {
                        char::from((r - 1 + 32) as u8) // 32..=127
                    }
                })
                .collect();
            let _ = kisc_unframe(&s);
        }
        // (2) HEADER-SHAPED fuzz — a well-formed `KISC …\n<body>` frame with each
        // field independently valid-or-corrupted, so the kind/version/len=/crc32=
        // parsers AND the LenMismatch/CrcMismatch/Ok paths are all exercised.
        for _ in 0..5000 {
            let blen = (rng.next() % 48) as usize;
            let body: String = (0..blen)
                .map(|_| char::from((rng.next() % 95 + 32) as u8))
                .collect();
            let kind = if rng.next() % 4 == 0 {
                "kiss-contract"
            } else {
                "wrong-kind"
            };
            let ver = rng.next() % 4; // 1 is valid; others exercise UnknownVersion
            let len = if rng.next() % 2 == 0 {
                body.len() // correct — enables the Ok / CrcMismatch paths
            } else {
                (rng.next() % 100) as usize // wrong — LenMismatch
            };
            let crc = if rng.next() % 2 == 0 {
                crc32(body.as_bytes()) // correct
            } else {
                rng.next() as u32 // wrong — CrcMismatch
            };
            let doc = format!("KISC {kind} {ver} len={len} crc32={crc:08x}\n{body}");
            let _ = kisc_unframe(&doc);
        }
        // (3) Adversarial headers that must decline (never panic): overflowing len,
        // non-numeric len, bad hex crc, magic only.
        for s in [
            "",
            "KISC",
            "KISC kiss-contract 1 len=999999999999999999999 crc32=00000000\nx",
            "KISC kiss-contract 1 len=abc crc32=00000000\nx",
            "KISC kiss-contract 1 len=0 crc32=zzzzzzzz\n",
        ] {
            let _ = kisc_unframe(s);
        }
    }

    #[test]
    fn unframe_round_trips_random_bodies() {
        let mut rng = Lcg(0x1234);
        for _ in 0..2000 {
            let len = (rng.next() % 120) as usize;
            let body: String = (0..len)
                .map(|_| {
                    let r = rng.next() % 96;
                    if r == 0 {
                        '\n' // bodies may be multi-line
                    } else {
                        char::from((r + 31) as u8)
                    }
                })
                .collect();
            assert_eq!(kisc_unframe(&kisc_frame(&body)), Ok(body.as_str()));
        }
    }
}

#[cfg(test)]
mod conformance_to_the_pinned_spelling {
    use super::*;

    /// The framing matches KISS-CONTRACT §6.11-0002/-0003 exactly.
    ///
    /// ⚠️ This file's module doc claimed for a month that the spelling was
    /// unpinned. It was pinned on 2026-07-13, and the implementation happened to
    /// conform anyway — so nothing was broken and nothing would have failed.
    /// **A test is what turns "happens to conform" into "conforms".**
    #[test]
    fn the_kisc_framing_conforms_to_the_pinned_spelling() {
        // ⚠️ NO CRC ASSERTION HERE. `crc32_matches_the_standard_check_value`
        // above already pins 0xCBF43926, and this test was drafted with a SECOND
        // COPY of it — the exact duplication this workspace spent the day removing,
        // written while fixing a doc defect about the same clause. The CRC variant
        // was already pinned; the header LINE was not, and that is the whole of
        // what this adds.

        // §6.11-0002: magic bytes, single spaces, decimal version, len=, crc32=,
        // one LF. Asserted as the literal line rather than field-by-field,
        // because the clause pins the LINE.
        let doc = kisc_frame("body");
        let header = doc.lines().next().expect("a header line");
        assert_eq!(
            header, "KISC kiss-contract 1 len=4 crc32=dba80bb2",
            "§6.11-0002 pins this line, including LOWERCASE hex (§6.11-0003)"
        );
        assert!(
            doc.as_bytes().starts_with(&[0x4B, 0x49, 0x53, 0x43]),
            "§6.11-0002 pins the magic AS BYTES 0x4B 0x49 0x53 0x43"
        );
        assert_eq!(
            doc.as_bytes()[header.len()],
            0x0A,
            "§6.11-0002 pins a single LF after the header line, never CRLF"
        );
    }
}
