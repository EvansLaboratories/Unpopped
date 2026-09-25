# Slang parser job: sized by running it

Answers the PM's "size, don't design" deliverable (2026-09-25) for the Slang
half of the Parser -> Unpopped -> Emitter split. **Report only — no design
proposed.** Measured by running `crates/unpopped/src/convert.rs`'s Slang
[`Frontend`] over Fuel's full kernel corpus, not by reading the recognizer and
guessing.

## Method

- **Population**: every `.slang` file in Fuel's `fuel-kernels-source/kernels/`
  at `origin/main` (`ded9bed`, fetched 2026-09-25) — enumerated via
  `git ls-tree -r --name-only origin/main -- fuel-kernels-source/kernels`,
  cross-checked against `git ls-files` on Fuel's local index (both = 147, the
  positive control the PM asked for). Content read via `git show
  origin/main:<path>` into a scratch directory (**only** the 147 `.slang`
  blobs were exported into it — no `.glsl`/`.metal` files were placed
  alongside them), never Fuel's working tree — the fuel lane is busy and its
  checkout must not be touched or trusted as current.
- **Extension-filtered, and the harness says so.** The scratch directory this
  run used held only `.slang` files, so the distinction didn't bite here —
  but the harness itself now filters on the `.slang` extension explicitly and
  prints `"N .slang files (of M files present)"`, not a bare file count. A
  first version of the harness counted every file in the given directory
  regardless of extension; pointed at a directory that also holds Fuel's
  GLSL/Metal kernels (167 files total: 147 `.slang` + 20 `.glsl`), it would
  have silently reported 171 (the actual total including Fuel's compiled
  `.spv` outputs) as if it were the Slang count — the exact wrong number
  already given out once elsewhere. Fixed before merge; the numbers in this
  doc were measured against the clean, `.slang`-only scratch directory either
  way, so they are unaffected.
- **Harness**: `crates/unpopped/examples/slang_corpus_report.rs` (added in
  this PR), run as `cargo run --example slang_corpus_report --features
  convert -p unpopped -- <dir>`. For each file it calls `convert::SLANG`'s
  `lift_elementwise`, `lift_reduction`, and `lift_scan` directly, declaring
  `dtypes = &[ElementKind::F32]` for every file regardless of the kernel's own
  name (`_bf16`, `_f16`, …) — this measures **op-class recognizability**, not
  dtype admissibility, and holding the declared dtype constant makes the 147
  results comparable to each other.
- ⚠️ **Deliberately does NOT call `convert::lift`, the combined function.**
  `lift` is `lift_elementwise(..).or_else(reduction).or_else(scan)` — on
  failure it returns only the **last** attempt's error, which would silently
  misattribute *why* a file was refused (this is the same failure shape as
  "true of the instrument, false of the result": `lift`'s return value is a
  true fact about the scan attempt, not a true fact about the file). The
  harness calls all three and classifies from all three errors together.

## Reading this result: two different questions, two different owners

⚠️ **0/147 does not mean "Slang is 0% portable" and should not be read that
way.** The 94-file "Unrecognized" bucket splits into two categories with
different costs and different owners, and they must be kept separate rather
than collapsed into the single headline number:

- **(a) The recognizer correctly refusing constructs it does not model** — the
  53-file residue bucket (`groupshared`/`InterlockedCompareExchange`/
  `RWByteAddressBuffer`). This is the recognizer working as designed, the same
  way it refuses shared-memory/atomic CUDA. Not a gap; not actionable by
  changing the recognizer.
- **(b) The recognizer failing to recognise things it could** — the largest
  single cause is mundane, not deep: **44 of the 94 unrecognized files are
  `NotElementwise` purely because Fuel names its buffers `out_buf`/`a_buf`/
  `b_buf` while the `Frontend` hardcodes `output`/`inputK`.** A naming
  convention mismatch is not an expressiveness limit. The remaining
  unrecognized files add real structure on top of that (branching,
  strided/broadcast rank, narrow-intrinsic calls) — see the breakdown below.

## Result: 0 of 147 accepted

```
population: 147 files
  53  refused: Inexpressible (residue)
  94  refused: Unrecognized
   0  accepted (elementwise + reduction + scan)
```

53 + 94 = 147 — every file accounted for, no third bucket.

### Bucket: Inexpressible (residue) — 53 files

The recognizer's residue check runs before any op-class attempt and is
identical across all three, so this bucket is unambiguous:

| residue marker | files |
|---|---|
| `groupshared` | 36 |
| `InterlockedCompareExchange` | 16 |
| `RWByteAddressBuffer` | 1 |

These are the recognizer's own residue list working as designed (shared
memory, atomics — the same categories `CUDA_RESIDUE`/`SLANG_RESIDUE` refuse
for CUDA), not a gap.

### Bucket: Unrecognized — 94 files

This bucket is NOT one cause. Breaking down the first (elementwise-attempt)
error, since that is where the population splits:

| shape | count | example |
|---|---|---|
| `NotElementwise` — no assignment to `output[...]` found anywhere in the file | 44 | — |
| `Unrecognized("read '...[..]' (not inputK)")` | 14 | `input[..]`, `src[..]` |
| `Unrecognized("identifier '...'")` | 24 | `b0`, `best_i`, `y`, `running`, `out_lo`, `x`, `lo`, `lo_bits`, `bits_lo` |
| `Unrecognized("call '...'")` | 9 | `float16_t(_)`, `f16tof32(_)`, `asfloat(_)`, `double(_)`, `clamp` (3-arg), `f8e4m3_to_f32(_)`, `f32_to_bf16_bits(_)`, `apply(_,_)` |
| `Unrecognized("node '...'")` | 3 | `conditional_expression` |

**44 of the 94 (30% of the whole corpus) are `NotElementwise` for one
structural reason**: `lift_store` looks for an assignment to a buffer
literally named `output` (`Frontend::out_name`), and Fuel's kernels use their
own buffer names (`out_buf`, `a_buf`, `b_buf`, per-op names) — confirmed by
reading `binary.slang` directly: `RWStructuredBuffer<float> out_buf` next to
`StructuredBuffer<float> a_buf, b_buf`, with strided/broadcast indexing and an
`if (a_contig && b_contig) { ... } else { ... }` fast/slow-path branch, not a
single `output[i] = <expr>;` statement. That branching, and the general-rank
strided/broadcast decomposition, would each independently fail the current
recognizer even under a naming fix — the naming mismatch is necessary but not
sufficient to explain the whole `NotElementwise` count, and this report does
not attempt to separate "naming-only" from "naming-plus-structure" further,
since doing so is design work, not sizing.

The remaining 50 (`read`, `identifier`, `call`, `node` shapes) are each a
recognizer gap against one specific Slang construct — narrow intrinsics
(`f16tof32`, `asfloat`, `float16_t(_)`, bit-manipulation casts), a
conditional-expression node the walker doesn't traverse, and non-`inputK`
buffer reads.

## What this sizes

**0 kernels port today as-is.** The recognizer's bounded idiom (`out[i] =
<expr>` under a fixed naming convention, no branches, no bit tricks) does not
match Fuel's actual kernel shapes, which use production concerns (strided/
broadcast general-rank access, fast/slow-path branching, explicit narrow-dtype
bit manipulation) the pilot recognizer was never built to handle. This is not
"nearly there" — it is two different shapes of program.

**No estimate of "M need K new idioms" is given here**, per the brief's own
request to report buckets and propose nothing: the buckets above are the
input to that sizing, not the sizing itself, and deciding which idioms to add
(support other buffer names? branching? which intrinsics?) is a design
decision for the gate, not a count this report can produce without also
designing the fix.

## Reproducing this

```
cargo run --example slang_corpus_report --features convert -p unpopped -- <dir-of-slang-files>
```

The harness is committed at `crates/unpopped/examples/slang_corpus_report.rs`
so this is re-runnable against a later Fuel snapshot without rebuilding it from
scratch.
