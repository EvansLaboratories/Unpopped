# E1 design note: reference emitters beyond `Schedule::Scalar`

Status: **design only, no code, no trait edit.** Closes nothing yet; it frames ROADMAP 6.1 row E1 (and unblocks I6) for a PM ruling.
Written 2026-10-30 against `origin/main` a428dc6 (0.34.3). The PM asked for this note first (task of 2026-10-30); row E1 had been
parked on 2026-10-24 with the question *"how far should the reference emitters grow"*.

## 1. The facts, each checkable at a428dc6

- `Schedule` has ten variants (`plan.rs:27`): `Vectorized`, `Scalar`, `Strided`, `Reduction`, `RowReduce`, `Contraction`, `Scan`, `Window`,
  `RowSort`, `Im2Col`.
- `unpopped-cpu-c` and `unpopped-slang` emit `Schedule::Scalar` only (cpu-c `lib.rs` `match plan.schedule`, slang `lib.rs:350`). Every other
  schedule returns `Err(LowerError::UnsupportedSchedule)`, a typed decline, **not a panic**: the module docs of cpu-c (`lib.rs:37-45`) still
  say "panics clearly" and are stale. Fixing that comment is part of the first E1 PR.
- Scalar is only ever chosen for a contiguous, single-output, view/gather/scatter/offset/coord-free Elementwise cell, so both emitters decline
  every complex cell today (cpu-c's own module docs say so).
- The kernel layer (`kernel.rs`, `docs/ir-kernel-layer.md`) already has a lift for **every access pattern the emitters decline**:
  `lift_elementwise_opdef` (including views, gather and scatter, base offsets), `lift_reduction_opdef`, `lift_scan_opdef`,
  `lift_contraction_opdef`, and `kernel_lift/{window,row_reduce,row_sort,im2col,gather,scatter}.rs`. Each agrees with the oracle on the existing
  harnesses by an interpreter differential (`run_kernel`, no device).
- `ir-kernel-layer.md` section 1 says U2 "does not deliver a path from a `KernelFn` to any emitter". That path is what E1 is.
- Row 4 added ids, launch shape, memory spaces and barriers to the kernel layer and `run_launch` (0.34.0-0.34.2), so a `KernelFn` can now say
  "one thread per output element, shared tile, barrier", which the intent layer's `Reduction`/`RowReduce` schedules express only as a tag.

## 2. The options

**A. Grow `Backend::lower(plan)` one schedule at a time** in each reference emitter (ten schedules x two targets).
- Reuses the existing seam, no trait change. Matches what the CUDA emitter in Baracuda does.
- Cost: each schedule is a second, hand-written implementation of loop semantics that the kernel-layer lift already holds, per target. Every one
  needs its own differential against the oracle, and two reference emitters would each re-derive reduction order, NaN rule and index arithmetic.
  `memory: one-rule-one-author` applies: the rule lives in the lift, a mirror cannot take its refinements.

**B. Emit source from a `KernelFn`.** One statement-level emitter per target language over the already-lifted kernel, for every schedule at once.
- One implementation of loop semantics (the lift), one of spelling (the emitter). Differential test: compile the emitted C with a real compiler,
  run it, compare bytes with `run_kernel` on the same arguments; no new oracle needed.
- Matches the direction of the IR union (`ir-union-design.md` I2) and the "IR is the full union" ruling: kernel languages are emitters of the
  statement IR, not of a tag.
- Cost: a new `Backend` item (section 4) and an argument ABI for a kernel that has none (section 5).

**C. Leave the emitters scalar-only** and let adopters (Baracuda, Fuel) own every other schedule. Zero work; leaves the first cross-backend
experiment (`idiom-lifting-design.md` section 10: RowReduce on the 4070 through CUDA vs Slang) impossible, and I6 behind E1 forever.

**Recommendation: B, in stages (section 6).** A gives a second copy of semantics the lift already owns; C is a decision to stop, which is the
PM's, not the lane's.

## 3. What B does and does not deliver

It delivers **correctness coverage**: cpu-c compiles and runs every lifted op, bit-exact against the interpreter where the lift is exact and within
the lift's declared tolerance elsewhere (spec 7.1: F64 byte-exact, F32 <= 2 ulp for tolerance-mode lifts; the same gate applies to the emitted
code, and a tolerance PR must carry the test-proven F32-accumulator mutant).

It does **not** deliver performance or the thesis experiment. A lifted `KernelFn` is a single logical thread (`ir-kernel-layer.md` section 2), so the
first emission is a **serial loop nest**: right for a CPU reference, a correctness baseline for Slang, and meaningless as a 4070 timing. The
RowReduce A/B of I6 needs a parallel mapping (stage E1c), which is a design in its own right and not assumed here.

## 4. The trait shape (needs a PM ruling; no trait edit before it)

Proposed: one defaulted method on `Backend`:

```rust
fn lower_kernel(&self, k: &KernelFn, bindings: &ResourceMap)
    -> Result<GeneratedKernel, LowerError> { Err(<typed decline>) }
```

- Defaulted, so every existing `Backend` (including Baracuda's and Fuel's) compiles unchanged and keeps `lower(plan)`.
- Reuses `GeneratedKernel` with its artifact/manifest from U3 rows 1-3 (0.32.x), so a Slang emitter can return a manifest.
- `LowerError` is `#[non_exhaustive]`, so a new decline variant ("kernel construct not emitted", e.g. an unsupported `Stmt`) is additive. Whether to reuse
  `UnsupportedOp` instead is the lane's call at PR time and not a ruling.
- Alternative: a separate trait `KernelBackend`. Cleaner types, but it splits `name()`/`provider()`/capability checks across two traits. Not recommended.

## 5. The argument ABI the kernel layer lacks

`ir-kernel-layer.md` section 7 records the open question (a `KernelFn` has no parameter carrying a runtime base offset; U3 spec "no argument ABI").
Row 3's `ResourceMap` is the caller-supplied answer: buffers by name and slot, a `Params` block layout with its layout rule. For E1:

- **cpu-c**: positional pointers in `Param` order, scalars by value, a runtime base offset passed as an extra `int64_t` immediately after its buffer.
  The generated C and the harness that calls it are both ours, so this is documented in the emitted file's header comment and in the manifest.
- **slang**: bindings and `Params` come from the `ResourceMap` (declared or inferred, with the provenance flag row 3 added). A kernel with a buffer that
  the map does not name is a typed decline, never a guess.

## 6. Stages, one PR each (E1 lane = this one; PM gates, allocates versions)

| Stage | Delivers | Test |
|---|---|---|
| E1a | The `lower_kernel` item (defaulted), `LowerError` decline, stale cpu-c doc fix, **cpu-c emitter for Elementwise** (the path with the most existing differentials) | compile with gcc/clang, run, bytes vs `run_kernel`, on the exhaustive Elementwise harness cells; a mutant per emitted construct |
| E1b | cpu-c for Reduction, Scan, Contraction; then Window, RowReduce, RowSort, Im2Col, Gather, Scatter, one PR per two families | per family: the family's existing lift differential, re-run through the compiled C; F32-accumulator mutant where tolerance applies |
| E1c | Slang emitter, serial, from a `KernelFn` + `ResourceMap` | `slangc` text compile on the CI box; GPU run only on the self-hosted runner (CireSnave 2026-10-16: tests that cannot run on GitHub run on a self-hosted runner, never `#[ignore]`) |
| E1d | Parallel mapping (grid-stride over the outer loop; block-tree RowReduce with `Barrier`) | needs its own note and PM ruling; unblocks I6 |

Each stage declines (typed) any `Stmt` or `ScalarExpr` it has not been taught; nothing is spelled by default.

## 7. Risks I could name now

1. **Narrow-float and MX dtypes**: cpu-c today declines `Select` for every narrow float (ROADMAP section 4). The kernel layer's `Cast` rows for them exist; whether
   the emitter spells them or declines is per-dtype work, to be listed in E1a rather than assumed.
2. **`Vectorized` and `Strided` are performance tags**, not different semantics: a `KernelFn` carries neither. Under B the emitter ignores the tag and
   the cell's key still names it; whether a consumer may treat the key as promising that schedule's speed is an adopter question (Fuel's dispatch table).
3. **Integer `Div`/`Const` dtype-blindness**: the existing backstop `assert_no_int_div_or_const` is written for plans; the kernel-layer path needs the same guard
   on `ScalarExpr`, tested by a mutant that removes it.
4. **Patents**: statement-level emission from a kernel IR is the closest area to US10949182B2 (nested loops from an IR for parallel units). The 2026-10-28/10-30 reads are
   "outside as read" for the current crates; E1d (deriving launch geometry from loop bounds) is the stage that triggers the re-read named in `PATENT-SWEEP.md`.
   Not a verdict.

## 8. Decisions for the PM

1. **Direction**: B (emit from `KernelFn`), A, or C? Recommended: B.
2. **Trait**: defaulted `Backend::lower_kernel` as in section 4, or a separate trait? Recommended: the defaulted method.
3. **Order**: cpu-c complete (E1a-b) before any Slang (E1c)? Recommended: yes; cpu-c is the reference and the one that can be differentially tested on GitHub.
4. **E1d** (parallel mapping) is out of scope here and is a separate note and ruling.
