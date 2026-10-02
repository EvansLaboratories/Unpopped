# Vendored KISS artifacts

Files here are **copied verbatim** from the KISS repository. They are not edited,
reformatted, or trimmed — a vendored copy that has been touched cannot be
compared against its source with `diff`, which is the only thing that makes
vendoring better than retyping.

| File | Source path in KISS | Vendored from commit |
|---|---|---|
| `dtype_manifest.json` | `conformance/corpus/dtype_manifest.json` | `bc16715eb04852e35baa59c86bb7fda0949c6cb7` (blob `af851db548f7cb6e05f63e0a6a03dee9c46b2e62`) |
| `structure_key_vectors.json` | `conformance/corpus/structure_key_vectors.json` | `bc16715eb04852e35baa59c86bb7fda0949c6cb7` (blob `86b43dfcced3d0c7809f3872ba3445bce0e00b79`) |

### How this copy was taken

Written with `git cat-file blob <commit>:<path> > <dest>`, **not** a checkout.
A checkout on Windows can CRLF-translate, and the point of vendoring is that
`diff` against the source is meaningful.

**Re-vendored 2026-10-02 from KISS `bc16715`**, after KISS#517 merged. That PR
reconciled KISS-Classify with Fuel's deriver (S1–S6, S15). Verified after writing:

| File | sha256 | CR bytes | `git hash-object` = KISS blob |
|---|---|---|---|
| `dtype_manifest.json` | `ddf12ec4a4f5d873d4bee6d6bb48532f00320b6c45dee42fe09244c2d735cb94` | 0 | yes |
| `structure_key_vectors.json` | `80fb85b3730df3f83b10506ae00473641ba8c9e42590555ae8cd91cd2353a36a` | 0 | yes |

What the refresh carried, and how each part was handled:
- **`f8e6m2` became reserved** (§6.1-0013). `ElementKind::is_reserved` now
  includes it, so a token naming it is a typed decline. This is breaking, and is
  why the refresh ships in 0.14.0, not a patch.
- **Vulkan vocabulary 4 → 5.** This was re-verified, not just bumped. V-15 adds
  three `<arith>` names (`i16`, `i64`, `f64`). This crate validates `vulkan:`
  tokens by grammar only, and the one Vulkan positive vector names none of the
  three. Every vector shared with the previous copy is byte-identical.
- **`target_match_vectors`** (§6.8-0002 discriminating pairs) are now asserted
  by `target_match_vectors_are_byte_exact`, as bare `TargetId`s and as whole keys.
- **New positive vector `gem_weight_role_discriminator`** (§6.6-0019)
  round-trips through the codec. Its *derivation* (weight-role `<wdt>`) is a
  separate gap, tracked on #31, because `structure_key` takes no role hints.

The previous copy (`dtype_manifest` from `19c3ad7`, vectors from `bea9416`) had
`sha256 = 619c834e…656e` for the vectors file.

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
