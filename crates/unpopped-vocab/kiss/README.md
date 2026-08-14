# Vendored KISS artifacts

Files here are **copied verbatim** from the KISS repository. They are not edited,
reformatted, or trimmed — a vendored copy that has been touched cannot be
compared against its source with `diff`, which is the only thing that makes
vendoring better than retyping.

| File | Source path in KISS | Vendored from commit |
|---|---|---|
| `dtype_manifest.json` | `conformance/corpus/dtype_manifest.json` | `19c3ad7f6924161e7b0fd8c7a5b88d9e194b5db7` |
| `structure_key_vectors.json` | `conformance/corpus/structure_key_vectors.json` | `bea9416` (blob `c83b5b7faeca9638a396ba998de363e36b8bac98`) |

### How this copy was taken

Written with `git cat-file blob <commit>:<path> > <dest>`, **not** a checkout.
A checkout on Windows can CRLF-translate, and the point of vendoring is that
`diff` against the source is meaningful. Verified after writing:
`sha256 = 619c834e563fb5bce565915b2d3f225cdf2e71ea803d04a6404ed1cedd29656e`,
CRLF count 0 — both matching what the KISS maintainer cited.

The artifact is LF-clean and `.gitattributes`-enforced upstream, so a raw hash is
stable across platforms. Two sibling artifacts (`dtype_manifest.json`,
`op_manifest.json`) were **not** — their generators wrote files LF and stdout
CRLF-translated on Windows, i.e. the same generator producing different bytes by
output path. That is being fixed upstream; do not re-vendor those until it lands.

### Two commits, two meanings

`structure_key_vectors.json` carries its own `source_commit: 19c3ad7`, which is
**not** the commit above and is **not** staleness. The table records where the
*artifact* was copied from (KISS `main`); `source_commit` records the *spec*
provenance the artifact was generated against. They differ because the artifact
landed after the spec anchor and `spec/` has not moved since.

`tests/kiss_byte_match.rs` asserts the `source_commit` value, so citing one
without the other is caught rather than merely discouraged — a report naming
only one of the two is not falsifiable.

## Why these are here

`unpopped-vocab` is one of four independent implementations of the KISS
`structure_key` codec. Its §6.1 dtype vocabulary has to be *exactly* the closed
set KISS defines — same members, same spellings, same reserved flags — and the
failure mode when it isn't is silent: a token decodes under the wrong vocabulary
and means something else. Before sk4 this crate had drifted to a 22-member set
against KISS's 24, and nothing detected it, because both sides were prose that
a human had transcribed.

The manifest is machine-readable and generated from `spec/classify.md`, so the
comparison can be exact instead of careful. `tests/kiss_dtype_manifest.rs`
asserts equality **in both directions** against the vendored copy.

## Updating

1. Copy the file again, verbatim:

   ```sh
   git -C ../KISS show origin/main:conformance/corpus/dtype_manifest.json \
     > crates/unpopped-vocab/kiss/dtype_manifest.json
   ```

2. Update the commit in the table above.
3. Run `cargo test -p unpopped-vocab`. A vocabulary change surfaces as a **test
   failure naming the exact tokens that differ**, which is the point — the
   manifest is what tells you a schema event happened, not a message from
   someone who noticed.

## What the vendored copy does and does not prove

It proves this crate agrees with KISS **as of the recorded commit**. It cannot
notice that KISS has since moved: nothing here reaches the network, and a
vendored file is a snapshot by definition. Staleness is caught by re-copying at
each schema event — the manifest carries `structure_key_schema_version`, so a
copy taken from a newer schema fails the version assertion immediately rather
than quietly widening the set.
