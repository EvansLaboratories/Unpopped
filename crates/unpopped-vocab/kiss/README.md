# Vendored KISS artifacts

Files here are **copied verbatim** from the KISS repository. They are not edited,
reformatted, or trimmed — a vendored copy that has been touched cannot be
compared against its source with `diff`, which is the only thing that makes
vendoring better than retyping.

| File | Source path in KISS | Vendored from commit |
|---|---|---|
| `dtype_manifest.json` | `conformance/corpus/dtype_manifest.json` | `19c3ad7f6924161e7b0fd8c7a5b88d9e194b5db7` |

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
