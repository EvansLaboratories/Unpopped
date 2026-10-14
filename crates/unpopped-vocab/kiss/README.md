# Vendored KISS artifacts

Files here are **copied verbatim** from the KISS repository. They are not edited,
reformatted, or trimmed — a vendored copy that has been touched cannot be
compared against its source with `diff`, which is the only thing that makes
vendoring better than retyping.

| File | Source path in KISS | Vendored from commit |
|---|---|---|
| `dtype_manifest.json` | `conformance/corpus/dtype_manifest.json` | `ff12265` (blob `24ff38f068aa18be723c07d307c86ae572c1333f`) |
| `structure_key_vectors.json` | `conformance/corpus/structure_key_vectors.json` | `ff12265` (blob `e5eec85c82754f030f1ee2f0ab4bf7829fa7e5d2`) |
| `structure_key_derivation_vectors.json` | `conformance/corpus/structure_key_derivation_vectors.json` | `ff12265` (blob `697aa54370b51df0e26fe5fbd23ceca8419290c5`) |

`ff12265` is KISS `main` as read on 2026-10-14 (KISS 0.2.39). It is a single
commit for all three files, so the table cannot name two different spec states.

### Re-vendored 2026-10-14 from KISS `ff12265` (sk5)

The schema moved `sk4` → `sk5` (KISS#535, KISS#536): the 24 op-family codes
became the derived seven-class `cell_class`, `dequant_form` is keyed, and an own
extent-1 axis counts as stride 0. Verified after writing:

| File | sha256 | CR bytes | `git hash-object` = KISS blob |
|---|---|---|---|
| `dtype_manifest.json` | `d12a6a9dc7893f743a48be036b375d3865a4462832eca9e740ff5719f6ccd5f8` | 0 | yes |
| `structure_key_vectors.json` | `8368d9911a2a4f0895157dda2950ecd23d633d9781ce5ce116b670fad25b634e` | 0 | yes |
| `structure_key_derivation_vectors.json` | `5b75de430d1b7a7e749c6e2964ea1a1deba2ff40160c05074fe321280b8ade9d` | 0 | yes |

- **`structure_key_derivation_vectors.json` is new.** `kiss_byte_match.rs` proves
  the codec; `kiss_derivation_vectors.rs` proves the *derivation*, which a codec
  can pass while deriving the wrong cell. Only the vectors carrying
  `spec_golden` are spec-anchored; the rest are reference-deriver output.
- **The CRLF/LF hold below is lifted.** KISS measured both generators at
  `5b28bf5` as emitting 0 CR bytes, and the three blobs above are LF with the
  hashes shown. `dtype_manifest.json` is re-vendored here for that reason and
  because it carries `token_prefix: sk5`.

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
output path. That was fixed upstream and the hold was lifted on 2026-10-14 (see above).

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
