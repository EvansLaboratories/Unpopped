# Parser/Emitter inventory and ownership map

Deliverable 1 for the Parser -> Unpopped -> Emitter separation task (PM task,
2026-09-24). **This is an inventory for the PM's gate. No crate has been moved.**
Everything below was measured against `origin/main` at `15f38e3` (Unpopped),
`6609d15` (Fuel), `677caea` (Vulkane), `0796251` (Baracuda) on 2026-09-24.

CireSnave's architectural correction, which shapes every call below: *"Unpopped
uses external parsers and emitters. Parsers read in a kernel language and
convert it to Unpopped's IR. Emitters write out a kernel language from
Unpopped's IR. Unpopped is supposed to only be the piece in the middle."*

## 1. Inventory: every parser and emitter that exists today

### Emit side — already split (measured, not re-litigated)

| target | crate | owner | in/out of tree |
|---|---|---|---|
| CUDA | `baracuda-cuda-emit` | Baracuda | out-of-tree, owned |
| CPU (C99) | `unpopped-cpu-c` | Unpopped | in-tree |
| Slang | `unpopped-slang` | Unpopped | in-tree |

`unpopped-cpu-c` and `unpopped-slang` are both **published** on crates.io since
0.1.0 (2026-08-19), now 0.3.0 — verified against `docs/deferred.md`'s own
registry-index check. `crates/unpopped/README.md:135` still says *"Both are
in-tree and unpublished"* — that line is stale and should be corrected in the
same PR that fixes anything else here; leaving it risks someone repeating the
PM's own near-miss of telling CireSnave the emitters were unavailable.

### Parse side — NOT split; both parsers live in neutral core

| file | size | language(s) | feature-gated? | exported? | status |
|---|---|---|---|---|---|
| `crates/unpopped/src/lift.rs` | 21,914 B | CUDA only | **no** — always compiled | yes, at crate root: `pub use lift::{ConsumeRefusal, LiftError, Lifted, lift_elementwise}` | **live**, not a dead pilot |
| `crates/unpopped/src/convert.rs` | 41,732 B | CUDA + Slang | yes — `#[cfg(feature = "convert")]`, feature off by default | no (module itself is feature-gated, not re-exported at crate root) | **live**, opt-in |

⚠️ **`lift.rs` is not a superseded pilot that can simply be deleted.** Three
independent facts, all measured:

1. Its `lift_elementwise` is re-exported at the crate root and is default-on
   (not behind the `convert` feature), so it ships in every default build.
2. `crates/unpopped/tests/lift_validates_declared_dtypes.rs:24` calls
   `unpopped::lift::{LiftError, lift_elementwise}` directly — a live test
   exercises it.
3. `convert.rs:38` does `use crate::lift::{LiftError, Lifted, binary_fn,
   unary_fn};` — **`convert.rs` depends on `lift.rs`'s types**. `lift.rs` is a
   dependency of the newer module, not a module the newer one replaced.

So the honest description is: `lift.rs` is a hand-written, CUDA-only,
elementwise-only recognizer that also happens to define shared vocabulary
(`LiftError`, `ConsumeRefusal`, `Lifted`, `binary_fn`, `unary_fn` — none of
these types name a language) that `convert.rs`'s tree-sitter-based, multi-op-class
(elementwise/reduction/scan), multi-language (CUDA+Slang, via a `Frontend`
descriptor) recognizer reuses. Both are live. Neither supersedes the other —
`convert.rs` supersedes `lift.rs`'s *recognizer* but not its *types*.

**Consequence for the move:** the CUDA-naming half of `lift.rs` (the
`lift_elementwise` recognizer, which only ever handles CUDA's
`out[i] = <expr>;` idiom) can move to a CUDA parser crate. The language-neutral
half (`LiftError`, `ConsumeRefusal`, `Lifted`, `binary_fn`, `unary_fn`) cannot
move with it without breaking `convert.rs`'s Slang half, which also depends on
those same types. That neutral half is the parse-side analogue of the
`Backend`/`Lowering` traits that stay in core for emitters — **this is a design
call for the gate, not decided here.**

`convert.rs` itself splits cleanly along its own `Frontend` abstraction: the
`Frontend` struct and the three `Frontend`-taking functions (`lift_elementwise`,
`lift_reduction`, `lift_scan`, `lift`) are language-neutral; `parse_cuda`, the
`CUDA` const and `CUDA_RESIDUE` are CUDA-only; `parse_slang`, the `SLANG` const
and `SLANG_RESIDUE` are Slang-only. This mirrors the emit side's `Backend`
trait / per-target implementation split exactly.

## 2. Proposed owner per language, with the stake tested (not assumed)

CireSnave's Slang reasoning was stated as a hypothesis ("I suspect") and is
tested here against measured file counts at `origin/main`, not inferred:

| language | candidate owner | evidence | verdict |
|---|---|---|---|
| CUDA (parse) | Baracuda | 495 `.cu` + 119 `.cuh` files in-tree; already owns the CUDA emit side (`baracuda-cuda-emit`) | confirmed — matches existing `docs/deferred.md` row 34 plan (see §4 below) |
| Slang | Fuel | `fuel-kernels-source/kernels/*.slang`: **147 files**, not 171 (see correction below) | confirmed — Fuel is the *only* project with a Slang stake |
| GLSL (compute) | Fuel | `fuel-kernels-source/kernels/*.glsl`: 20 files, in the **same directory** as the Slang kernels (matmul, flash_attn, conv2d — numeric compute, not rendering) | Fuel is the only candidate; **not currently in scope** — Unpopped has no GLSL parser or emitter today, so this is a future-scope note, not part of this split |
| Metal (compute) | Fuel | `fuel-metal-kernels/src/metal_src/*.metal`: 16 files (Apple GPU compute) | Fuel is the only candidate; **not currently in scope** — same caveat as GLSL |

⚠️ **Correction to the PM's figure:** the task message cites *"171 `.slang`
files"* in Fuel. Measured directly against `origin/main` (6609d15): Fuel has
**147** `.slang` files. 171 is the count of `.spv` files (compiled SPIR-V
binaries) under `fuel-vulkan-kernels/spv/` — a different, compiled artifact,
not source, and not evidence about Slang. The undercount doesn't change the
conclusion (Fuel is still the only Slang stakeholder), but the number itself
was wrong and is corrected here per the portfolio's "verify peer dep claims"
discipline.

**Vulkane tested and confirmed NOT a stakeholder**, closing the "only other
candidate" question the task raised. Vulkane has 0 `.slang`, 0 `.glsl`
compute files. It does have `.wgsl` (5), `.comp` (2) and `.spv` (9) — all
under `vulkane/examples/shaders/` (`deferred_shading`, `depth_prepass`,
`shadow_map`, `instanced_mesh`, `textured_quad`, plus a `triangle.frag`/`.vert`
pair). These are **graphics rendering shaders for Vulkane's own examples**, a
different domain from Unpopped's elementwise/reduction/scan numeric compute
kernels — not a stake in a kernel *language* Unpopped would parse or emit.
This is the positive control for the negative: Vulkane's Slang/GLSL count is a
true zero, not an artifact of a bad grep, because the same query finds real
files in the adjacent-but-different graphics-shader category.

## 3. Languages with NO candidate owner

Checked for `.cl` (OpenCL), `.hip` (ROCm/HIP), `.ispc` (ISPC) across Fuel,
Vulkane, and Baracuda at `origin/main`: **zero hits in all three repos, for
all three extensions.** No project in the portfolio has a stake in OpenCL,
HIP, or ISPC today. These are named explicitly rather than assigned a default
owner, per the task's instruction — "every common kernel language owned by
some project with a stake" has a failure mode (a language nobody wants), and
this is it. If Unpopped ever needs one of these, the owner question is open
and should go back to CireSnave rather than being decided by proximity.

C99 (the `unpopped-cpu-c` target) is deliberately excluded from this table: it
is not a *vendor's* kernel language with an external stake, it is the
project's own portable reference target, which is why it stays Unpopped-owned
and in-tree — that is not a gap in the ownership rule, it is the rule's
intended exception (a neutral core needs one dependency-free reference).

## 4. `docs/deferred.md` row 34 — restating the broken trigger

Current text (row 34, `unpopped-cuda` sub-crate): trigger was *"the 0.2 trait
freeze"*, which the row's own text says **passed unmet** — 0.2 shipped
2026-08-15 and `Backend`/`Lowering` took two more breaking changes since
(0.4.0's `Result<Spelling, LowerError>`, 0.5.0's `DeclinedOp` split). A version
number was the trigger last time, and a version number is exactly what moved
out from under it — restating against another version number would repeat the
same failure mode.

**Proposed restatement (artifact-shaped, for the gate to accept or reject):**
the `unpopped-cuda` sub-crate move triggers when **both** of the following
exist as committed artifacts, checkable by diff rather than by promise:

1. A **parser registry design** is committed (§5 below) — so the CUDA parser
   crate has a documented seam to plug into, symmetric with the emitter
   registry it will also need to satisfy on the emit side.
2. `Backend`/`Lowering` has gone **one full release with no breaking change**
   — i.e., the next version bump after this one is checked against the prior
   trait shape and is not "major" in the pre-1.0 sense CireSnave uses
   (0.n -> 0.(n+1) breaking). This is checkable from the changelog/manifest
   history at gate time, not from a stated intention.

Per this crate's own register's lesson (quoted in the PM's task): *"an
artifact-shaped trigger is checkable, but it still only fires for the future
you imagined."* Flagging that the same limitation applies to this
restatement too — it is a proposal, not a ruling.

The row's existing note that the crate has **two** inbound streams (Baracuda's
IR->`.cu` donation, and `lift.rs`+`convert.rs`'s `.cu`->IR parser) and "must not
be designed emit-only" still holds and is unaffected by this restatement.

## 5. Parser registry proposal (mirrors `docs/catalog.md` §5's emitter registry)

`docs/catalog.md` §5 designs an emitter registry: compiled in, selected by
name, no dynamic loading. Nothing in the codebase or docs proposes the
parse-side equivalent — confirmed by reading `docs/catalog.md` in full (its
Phase 1 `unpopped-catalog` names "op registry, emitter registry" only) and by
grepping the tree directly for "parser" (GitHub code search returned 0 for
this repo's `docs/`, which is **not** reported as an absence per the task's own
warning about unindexed-repo false negatives — the direct grep is the real
answer, and it also found nothing naming a parser registry or a parser
ownership table).

Proposed shape, symmetric with §5's `EmitterRegistry`:

```rust
let mut reg = FrontendRegistry::new();
reg.register("cuda",  Box::new(baracuda_cuda_parse::Cuda));   // owned by Baracuda
reg.register("slang", Box::new(fuel_slang_parse::Slang));     // owned by Fuel
```

A `Frontend`-shaped trait (or the existing `convert.rs::Frontend` struct,
generalized) is the seam: a parser crate depends on `unpopped` (for the IR
types and whatever subset of `lift.rs`'s neutral vocabulary the gate decides
stays in core, per §1), never the reverse — same direction as the emit side.
Selection happens per request, same as `EmitterRegistry::register`/lookup.
This is a proposal for the gate; no registry code has been written.

## 6. Coordination gaps, named rather than assumed

- **Baracuda has no lane.** The CUDA parser donation (moving `convert.rs`'s
  CUDA half + `lift.rs`'s CUDA recognizer into a Baracuda-owned crate)
  requires Baracuda's assent, which cannot be obtained from here. Recorded as
  a blocker for the PM to raise with CireSnave, not assumed.
- **Fuel has a lane (`gf5jcpe8`) but is mid-dissolution and busy.** Not
  contacted yet pending the PM's gate on this inventory, per the task's
  "ask, do not block on it" instruction — will reach out once the map is
  gated, to confirm Fuel's Slang (and, if in scope later, GLSL/Metal)
  ownership and get their read on what a Fuel-owned `unpopped`-parser crate
  should depend on.
- **What this inventory could NOT determine:** whether Fuel's GLSL/Metal
  kernels are meant to ever go through Unpopped's IR at all, or are permanently
  out of scope (Fuel may intend them as hand-written, backend-specific code
  with no shared neutral representation). That's a scope question for Fuel
  and CireSnave, not inferable from file counts.

## What changed vs. the PM's brief, and why

- Corrected the Fuel `.slang` count (171 -> 147; 171 is `.spv`, a different,
  compiled artifact).
- Found two additional Fuel kernel-language stakes not mentioned in the brief
  (GLSL, Metal) — named as future-scope, not folded into this split.
- Positively confirmed Vulkane's non-stake with a control (its WGSL/GLSL/SPIR-V
  files are graphics-shader examples, not compute kernels), rather than taking
  the "Vulkane is pure Vulkan" reasoning on faith.
- Corrected `crates/unpopped/README.md:135`'s stale "in-tree and unpublished"
  claim about the two reference emitters (not yet fixed in this doc's commit —
  flagged for the same PR that acts on this gate, since the PM asked for it
  fixed "while you are in there").
