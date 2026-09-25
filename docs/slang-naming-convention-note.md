# The `output`/`inputK` naming convention: where it lives, and what fixing it actually buys

Design note only, per the PM's brief — **no code changes to `convert.rs` in
this PR.** Answers the three questions on the 44-file naming-convention
hypothesis from `docs/slang-corpus-sizing.md`.

## 1. Where the convention is enforced

**One declaration site, several consumption sites, all parameterized —
not scattered special-casing.** `crates/unpopped/src/convert.rs:162-163`,
the `SLANG` `Frontend` const:

```rust
pub const SLANG: Frontend = Frontend {
    ...
    out_name: "output",
    in_prefix: "input",
    ...
};
```

`out_name`/`in_prefix` are threaded as plain `&str` parameters into five
shared walker functions that all name-match against them: `find_out_store`
(elementwise store), `find_running_out_store` (scan store),
`find_scalar_out_store` (reduction store), `first_input_index` (locates the
loop index via `{in_prefix}{K}[idx]`), and `Walk::expr` (reads `{in_prefix}
{K}[idx]` inside the body). All five take `out_name`/`in_prefix` as
arguments rather than hardcoding the literal string themselves, so the
convention is genuinely a single configuration value per `Frontend` — CUDA's
`out`/`in` and Slang's `output`/`input` are two instances of the same
parameter, not two separate implementations.

## 2. Smallest change compatible with the existing IR vocabulary

**A literal-string change doesn't fix this — Fuel's kernels don't use one
fixed name.** `binary.slang` uses `out_buf`/`a_buf`/`b_buf`;
`add_assign_scaled.slang` uses `dst`/`src`; several others (`affine.slang`,
`clamp.slang`, `cast_bf16_to_f32.slang`) already use exactly `output`/`input`
literally. Changing the constant to any single string only ever matches the
kernels that happen to share that one name.

**Proposed direction: discover the buffer's role from its declared TYPE, not
its name.** Slang/HLSL already carries this distinction in its own type
system — `RWStructuredBuffer<T>` (writable) vs. `StructuredBuffer<T>`
(read-only) — and CUDA has the analogous discriminator in a plain pointer
parameter's `const`-qualification (`const float*` vs. `float*`). Scan the
kernel's buffer declarations once, classify each by writability, and assign
ordinals to the read-only ones by declaration order — then feed that
derived `(output_name, [input_names_in_order])` mapping into the *same* five
walker functions above, unchanged, instead of the two fixed strings on the
`Frontend` const.

This is compatible with the constraint the task named: it touches only
`convert.rs`'s Slang-frontend-internal declaration scanning. It does not
touch `OpDef`, `Backend`, or `Lowering` — the IR's `Input(K)` addressing by
ordinal already exists exactly as-is; this only changes *where the ordinal
comes from* (declared position instead of a parsed numeric name suffix).

## 3. How many of the 44 would actually convert — measured, not assumed

**Prototyped rather than guessed**, per the instruction not to report 44
without a method: wrote a throwaway script (not committed — this is a
sizing prototype, not the design) that scans each of the 147 `.slang`
sources for `RWStructuredBuffer`/`RWByteAddressBuffer` vs.
`StructuredBuffer`/`ByteAddressBuffer` declarations, and where there is
**exactly one** writable declaration (the unambiguous case), textually
renames that identifier to `output` and each read-only declaration to
`input0`, `input1`, … in declaration order — then re-ran the **unmodified,
already-merged** `slang_corpus_report` harness against the rewritten
corpus, so the result comes from the real recognizer, not a simulation of
it.

- **146 of 147 files had exactly one writable declaration** and were
  renamed. One (`autodiff_poc.slang`) declares two `RWStructuredBuffer`s and
  was left untouched as ambiguous — correctly out of scope for a
  single-output elementwise/reduction/scan recognizer regardless of naming.
- **Result: 0/147 accepted before → 4/147 accepted after.** Not 44, not a
  double-digit fraction of it. The four: `triangular_b8` (elementwise) and
  `cumsum_f16`/`cumsum_f32`/`cumsum_f64` (scan).

⚠️ **The honest number is 4, not "at least N, at most 44" — this was
measured, not estimated, so there is no remaining uncertainty band to
report for this specific fix.** Cross-checked against the original 44
`NotElementwise` files by name: only **one** of the four rescued files
(`triangular_b8`) was in that original 44; the `cumsum_*` three were
originally bucketed under a *different* first error
(`Unrecognized("identifier 'running'")`, an elementwise-attempt failure) and
were rescued on their **scan** attempt, where the blocker really was the
input-naming check (`"scan body reads no inK[idx]"`).

**Why the other 43 of the "44" don't convert on a naming fix alone**:
re-reading them after the rename, the dominant remaining blocker is a
**different** limitation — the walker expects the store's RHS to be a
single inline expression over `{in_prefix}K[idx]` reads, and most of Fuel's
kernels bind an intermediate local first (`float x = input[idx]; output[idx]
= affine(x, ...);`), which the expression walker doesn't recognize
(`Unrecognized("identifier 'x'")`, `identifier 'b0'`, `identifier
'best_i'`, …) regardless of what the buffers are named. That is a second,
separate recognizer gap — local-variable binding before the store — not
sized or designed here, since sizing it needs its own prototype the same
way this one did.

## What this means for sequencing

The naming-role fix is real, cheap, and worth doing — but it is **not** the
lever that gets Slang portability out of single digits. **44 files being
"one rename away" was the wrong read of the bucket**; the real naming-only
opportunity, measured, is 4/147. Whatever comes next (local-variable
binding support is the next-largest visible blocker, per the bucket
breakdown in `docs/slang-corpus-sizing.md`) should be sized the same way —
prototyped and re-run — before it is quoted as a number to anyone.
