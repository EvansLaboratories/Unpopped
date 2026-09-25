# Sizing the second-layer blockers: local bindings, branching, strided access

Design note only, per the brief — **no committed code changes.** The one
Rust patch used to test the branching hypothesis was written locally,
verified to compile and to measure, then reverted; it is described below,
not shipped. Answers "which one of the three unlocks the most files,
measured by prototyping" from `docs/slang-corpus-sizing.md`'s three
remaining blocker categories: an intermediate local-variable binding before
the store, branching, and strided/broadcast general-rank access.

**Baseline for all of this**: the 146/147-renamed corpus from
`docs/slang-naming-convention-note.md` (buffer names already fixed by
declared-type role), which alone converts 4/147. Everything below measures
the MARGINAL effect of one more fix on top of that baseline, not from zero.

## Result up front

| candidate | measured | verdict |
|---|---|---|
| Local-variable binding (naive global inlining) | **net negative**: 2 accepted vs. the 4-file baseline, and one of those 2 is a **silent misclassification**, not a genuine win | unsafe as prototyped; a real fix needs to be scoped inside the walker, not a blind text pass |
| Branching → `Select` | **0 files rescued in isolation**; IR support already exists and costs nothing to add, but the branches that use it also need local-binding support, which the naive approach can't safely provide | real, cheap, but not separable from the first category |
| Strided/broadcast general-rank access | **not prototyped — structurally out of reach** for the current flat-index model; rough population (not a "would convert" claim) is up to 70/147 | biggest population, and the one this design note cannot safely estimate further |

None of the three is "the cheapest lever." **Branching support is
essentially free once local-binding support exists safely, but nothing
here safely delivers local-binding support** — that is the actual finding.

## 1. Local-variable binding — prototyped, and it broke things

Wrote a textual prototype (not committed): fold a simple two-armed
`if (cond) { x = a; } else { x = b; }` (single statement per arm, same
identifier) into `x = (cond) ? (a) : (b);`, then repeatedly inline any
identifier assigned **exactly once** anywhere in the file at its use
site(s), iterating to convergence (so chains like `c3 -> a_off -> x`
resolve). Ran the **unmodified, already-merged** harness against the
result.

- **If/else → ternary merges applied: 0 across all 147 files.** Every
  branch actually present in the corpus is multi-statement (stride/offset
  decomposition, like `binary.slang`'s fast/slow path), not a simple
  single-line value select. The category I set out to test — "simple
  branch, one value" — measured **zero occurrences**, not a small number.
- **Single-assignment locals inlined: 1,661**, across all 147 files (~11
  per file on average) — this part of the transform did fire, broadly.
- **Net result: 0/147 → 2/147 accepted, down from the 4/147 baseline.**
  ⚠️ **And one of the 2 is worse than a regression — it's a silent
  misclassification.** `cumsum_f32`/`cumsum_f64` were correctly accepted as
  **scan** in the baseline; after blanket inlining they are accepted as
  **elementwise** instead — the inliner destroyed the exact syntactic shape
  `find_running_out_store`/`find_accumulation` depend on (the scan
  accumulator's own recognized variable got inlined away), and the
  now-mangled body happened to ALSO satisfy the elementwise store pattern.
  That is not "fails safely" — it is a wrong `OpDef` that would look
  accepted. `triangular_b8` and `cumsum_f16` simply stopped being accepted
  at all (`NotElementwise` / a different `Unrecognized`), the ordinary
  regression shape.

**Why this matters more than the number**: a blind, corpus-wide textual
inlining pass is not a safe approximation of "add local-variable support to
the walker" — it can corrupt a recognizer's existing successful matches,
including by producing a *wrong accept* rather than a clean *refuse*. A
real implementation needs to inline **only** the identifiers that feed the
one store/scan/reduction shape the walker is currently trying to match,
done inside `Walk`/`find_*_store` with context, not as a preprocessing
pass blind to which variable is "the" accumulator. That is real design
work this note does not attempt, and the honest floor from what was
prototyped is: **not usable as tested — 0 safely-attributable new files,
with a demonstrated correctness hazard.**

## 2. Branching → `Select` — the IR already supports it, tested in isolation

**Checked before prototyping, not assumed**: the IR already has
`ScalarExpr::Select(cond, a, b)` (`crates/unpopped/src/ir.rs:174`) and
comparison ops that produce a 1.0/0.0 mask suitable as `cond`
(`BinaryOp::CmpEq/CmpNe/CmpLt/CmpLe/CmpGt/CmpGe`, `ir.rs:368-383`). Both
reference emitters already lower `Select` — `unpopped-slang` directly
(`src/lib.rs:367`), `unpopped-cpu-c` for `F32`/`F32Strict`/`F64`
(`src/lib.rs:580-591`, declining other dtypes as a named v1 limitation, not
silently). **This means recognizing a simple branch and adding
comparison-operator support requires zero `Backend`/`Lowering` changes** —
the constraint the task named — because the target vocabulary already
exists and is already reachable by at least two of the three targets.

**Prototyped the Rust side of this alone**: a 17-line patch to
`crates/unpopped/src/convert.rs`'s `Walk::expr` — six new `binary_expression`
operator arms (`==`,`!=`,`<`,`<=`,`>`,`>=`) plus one new match arm for
tree-sitter's `conditional_expression` node (`cond ? a : b` →
`ScalarExpr::Select`). Compiled clean, reverted after measuring (not
shipped).

- **Run against the renamed-only baseline (no textual inlining at all):
  0/147 → 0 additional files.** Identical histogram to the 4-file baseline.
- **3 files use Slang's native `?:` directly** (`masked_fill_b4`,
  `masked_fill_b8`, `triangular_b4` — originally bucketed as
  `Unrecognized("node 'conditional_expression'")`). None of the 3 were
  rescued by the ternary patch alone: their conditions/branches themselves
  read intermediate locals, so they need category-1 support too, and the
  only category-1 mechanism tried here is the one just shown to be unsafe.

**Branching and local-binding support are not separable in this corpus.**
Adding `Select` support costs nothing (it's already in the IR, already
lowered, and the walker change is small and self-contained) but by itself
unlocks 0 files here, because every branch that would benefit from it is
entangled with a local variable it also needs resolved.

## 3. Strided/broadcast general-rank access — not prototyped, and here is why

This is not "one more construct." The walker's addressing model is a
single flat `{in_prefix}K[idx]` read per operand; Fuel's strided kernels
compute per-dimension offsets from a `ConstantBuffer<Params>` struct's
fields (`shape0..shape3`, `a_s0..a_s3`, `b_s0..b_s3` in `binary.slang`) —
the walker has **no** representation for a struct-field read, a
`ConstantBuffer`, integer division/modulo chains, or a multi-dimensional
offset at all. This is a different addressing STRATEGY, not a gap in the
current one, and prototyping it honestly would mean writing a materially
different recognizer, not patching the existing one — out of scope for a
sizing note. Per the task's own allowance, reporting this rather than
estimating it.

**Rough context only, not a "would convert" claim**: grepped the renamed
corpus for `shape0`, a trailing `_s0`, or a bare `rank` identifier (markers
of this pattern) and found **70 of 147 files** exhibiting it, all 70
already inside the original 94-file "Unrecognized" bucket. This is the
largest population of the three candidates by a wide margin — larger than
the naming mismatch (44, of which 4 converted) — but it is a population
count of a textual marker, not a measured acceptance rate, and this note
does not claim otherwise.

## What this means for sequencing

**No single blocker is the cheap next step.** The naming fix (previous
note) is real but small (4 files). Branching support is architecturally
free but empirically inert alone (0 files) because it's entangled with
local-variable resolution. Local-variable resolution is the connective
tissue both naming's residual 43 files and branching's 3 files need, but
the only way tested here to provide it is unsafe (silent misclassification
risk). Strided/broadcast access is the largest population (up to ~70) and
the least tractable with the current recognizer design.

**If there is a next sizing question worth answering the same way**, it is
whether a *scoped* (walker-internal, store-shape-aware) local-variable
resolution mechanism can be prototyped without the blind-substitution
hazard found here — that would need to touch `find_out_store`/
`find_running_out_store`/`find_scalar_out_store`'s own matching logic
rather than the source text before it reaches them, which is a real
prototype, not a text substitution, and was not attempted in this pass.
