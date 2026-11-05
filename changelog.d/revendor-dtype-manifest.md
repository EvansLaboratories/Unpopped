kind: re-vendor KISS `dtype_manifest.json` (KISS#772; vendored data and its pin only)
---
`crates/unpopped-vocab/kiss/dtype_manifest.json` re-vendored verbatim from KISS `e786097`: only the `generated_from` member
changed (it names the emitting tool now); no entry, count or schema moved. Blob `129690fe` read back equal to KISS, 0 CR bytes;
the content pin in `tests/kiss_byte_match.rs` (length 3,650, FNV-1a-64) and the `kiss/README.md` table moved with it.
