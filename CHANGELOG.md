# Changelog

**Why this file exists, and it is not bookkeeping.**

Measured 2026-09-06 by diffing each published `.crate` tarball against the tree —
a practice relayed by baracuda as *"keep the last release's consumer code and run
it against the new artifact; it answers what did this fix break, which no gate
written for the fix can ask."*

⚠️ **A public-API diff of `unpopped` 0.9.0 vs the tree reported PURELY ADDITIVE:
292 items → 293, nothing removed. Underneath were TWO behaviour changes in
`unpopped-vocab`, one of them NUMERIC — and both sat at a version number
IDENTICAL to the published one.** A signature diff cannot see a same-signature
behaviour change, and a version check cannot see a tree that never bumped.

---

## `0.20.18` — additive: `unpopped::exactness`, the record of what a device measured (PM-allocated)

New `DeviceStamp` (an `HwStamp` plus driver version and compiler arguments), `ExactnessRecord`, `ExactnessVerdict { Exact, Flushes }`, `ExactnessLedger::lookup` and `body_key` (dtype + the IR's serde-JSON, pinned by a test). Nothing reads a ledger yet: the next change wires it into `AccuracyKey` so a `Flushes` record declines Exact. Records apply to exactly their stamp, body and dtype; a changed driver or argument set matches nothing; disagreeing records resolve to `Flushes`. Exhaustive 2^32 exists only for unary f32 bodies, so a binary or f64 body has a record only if a measured run exists, and is declined otherwise (no extrapolation).

## `0.20.17` — docs only: the harness family moved out of scratch

`ROADMAP.md` row 3 and a status line in `docs/harness-family.md`: the harness lives in `ThinkersJournal/unpopped-harness` (cpu-c, CUDA, Vulkan). No code, no key bytes, no API change.

## `0.20.16` — slang: name the subnormal mode (`required_slangc_args`, header line)

Exhaustive 2^32 f32 relu, cpu-c reference vs Slang through `slangc` -> SPIR-V -> Vulkan (`unpopped-harness` 0.0.2/0.0.3, AMD Radeon 610M and RTX 4070 Laptop): the default Slang kernel differs on exactly 8,388,607 inputs, the negative subnormals (the device flushes them in `x < 0.0`, then returns `x`; cpu-c and CUDA return 0). With `slangc -denorm-mode-fp32 preserve` (SPIR-V `DenormPreserve 32`): zero mismatches on both. The mode is a compile argument, not source text.

New `unpopped_slang::required_slangc_args(dtype)` (`-denorm-mode-fp32 preserve` for F32/F32Strict, `-denorm-mode-fp64 preserve` for F64, none for integers). Every float kernel's header gains one comment line, `// slangc: <args>` (comment only: the lifter ignores it; the kernel body, name and cell are unchanged). Module docs: a device must NOT be gated on `VkPhysicalDeviceFloatControlsProperties` alone (the RTX 4070 reports `shaderDenormPreserveFloat32 = false` yet preserves when asked and flushes by default); how a plan declines waits on a `denorm` field in the `vulkan:` token, which is vulkane's vocabulary, not KISS's (KISS lane ruling 2026-10-21). Until then an `Exact` float Slang kernel is only as exact as the flag the consumer passes. Not checked here: the `f64` flag on a device (no f64 run), and the `slangc` self-hosted test still `#[ignore]`.

## `0.20.15` — docs only: `max(x, 0)` does not scrub NaN

`ir.rs` (`UnaryOp::Relu`, `Expr::relu`) said `max(x, 0)` scrubs NaN. That is IEEE `fmax` (`BinaryOp::FmaxIeee`); `BinaryOp::Max` propagates NaN and keeps A on ties, so `Max(x, 0)` is bit-identical to `Relu` (exhaustive 2^32 f32 comparison through cpu-c, Exact tier; other emitters not run). Comments only, no code.

## `0.20.14` — docs only: roadmap row and design note for the harness family

`ROADMAP.md` row 3 and `docs/harness-family.md`: external test/profile harnesses that wrap Unpopped, kept in scratch until a second target runs (CireSnave's request, PM ruling 2026-10-19). No code, no key bytes, no API change.

## `0.20.13` — tests only: mutation audit of `oracle.rs`, cluster 9 (walkers' index arithmetic)

`cargo-mutants` 27.1.0 over the evaluators' index and layout arithmetic. `oracle::walk_pins` pins a top-k row sort with both outputs (each output is sized from its own
operand), a skipping gather (indices just outside the axis write nothing), a scatter whose index operand is longer than its value operand (the walk is over the value
operand), a middle-axis reduction placed by the output strides (collapsed and keepdim), and `required_fidelity` for a u8 mask beside an f32 value (tolerant, not bit-exact).
The three survivors left in these functions are equivalent and are listed with their reasons in the PR. No code, no key bytes, no API change.

## `0.20.12` — tests only: mutation audit of `oracle.rs`, cluster 8 (value plumbing)

`cargo-mutants` 27.1.0 over the plumbing around the evaluators. `oracle::plumbing_pins` pins `raw_sign_masks` (narrow floats only), `round_to_compute`
(through the compute dtype, f64 untouched), `unravel` (last axis fastest), `store_val` (a same-dtype raw value moves verbatim, a different one converts,
a real or integer value into a complex slot gets a zero imaginary part), `alloc_output` (each output of a multi-output op takes its own descriptor), and the
`Neg`, `Abs` and `Select` arms of `eval` on narrow floats (payload-preserving sign edits; the select condition is tested in the compute dtype). No code,
no key bytes, no API change.

## `0.20.11` — tests only: mutation audit of `oracle.rs`, cluster 7 (storage encoders and codecs)

`cargo-mutants` 27.1.0 over the storage encoders and bit codecs. `oracle::encode_pins` pins `encode_int` and `encode_float` for the four-bit and one-bit
dtypes, `encode_complex` (real half low, imaginary half high), `encode_int_as_complex`, the `Val::complex` projection for float, integer and complex
raw values, and `unit_roundoff`. The 28 survivors left in this group are each provably equivalent (a `|` over disjoint bit sets, or a comparison whose equal case returns
earlier) and are listed with their reasons in the PR. No code, no key bytes, no API change.

## `0.20.10` — tests only: mutation audit of `oracle.rs`, cluster 6 (`arith_steps`)

`cargo-mutants` 27.1.0 over `arith_steps`, the count that sizes the derived tolerance band. `oracle::arith_pins` builds expressions whose operands
cost 1, 2, 3 and 5 steps, so a combining `+` that became `-` or `*` cannot land on the right total: leaves cost nothing, infix and named binary nodes cost
one plus both operands, a unary costs one plus its operand, and a `Select` costs only its three subexpressions. No code, no key bytes, no API change.

## `0.20.9` — tests only: mutation audit of `oracle.rs`, cluster 5 (`eval_binary`)

`cargo-mutants` 27.1.0 over `eval_binary`. `oracle::binary_pins` reads back the storage bits of `Copysign`, `Max` and `Min` on `e5m2`, `e4m3fn`,
`f16` and `bf16`: the sign mask, either operand winning, a NaN operand returned with its payload in either position, and a tie between `+0` and `-0`
keeping the first operand's bits (the `f64` detour agrees on every ordinary value and loses exactly those). No code, no key bytes, no API change.

## `0.20.8` — tests only: mutation audit of `oracle.rs`, cluster 4 (`compare`)

`cargo-mutants` 27.1.0 over `compare`. `oracle::compare_pins` checks which rejection each branch of the comparator returns (not just that it
returns `Err`): the first differing element in bit-exact mode, the byte-length case, infinities and NaN in the complex and float paths, the
tolerance band `abs + rel * max(|e|, |a|)` per component for complex, exactly for integers and in `f64` for floats, and that a mixed
complex/real or int/float pair takes the real projection rather than the component-wise or exact-integer path. No code, no key bytes, no API change.

## `0.20.7` — tests only: mutation audit of `oracle.rs`, cluster 3 (`eval_im2col`)

`cargo-mutants` 27.1.0 over `eval_im2col`. `oracle::im2col_pins` checks the evaluator against a direct six-deep loop over
(batch, channel, tap, output position) that never divides or takes a remainder, across non-square, strided, padded and
dilated geometries, a non-zero buffer base offset, and size-1 batch and channel axes (admitted by the plan gate since 0.20.6).
No code, no key bytes, no API change.

## `0.20.6` — fix: the Im2Col, Window and row-sort gates no longer read a size-1 axis as a broadcast

`build_plan` panicked ("output must be forward-dense contiguous" / "input 0 must be a full dense tensor") for ordinary shapes:
an Im2Col with batch 1 beside two or more channels, a single-channel batch (N ≥ 2, C = 1), a Window pooling a row to one
element (a global pool), and a top-1 / arg-max row sort. The key's broadcast mask is over the iteration frame, and an own
axis of extent 1 beside a wider frame axis counts as stride 0 there, so the gates' `bcast.is_empty()` clauses rejected dense
tensors. The clauses are removed: `contig` already reads `Broadcast` for a genuine stride-0 axis of extent > 1, so the
dense-layout clause alone rejects every operand that is not dense. Found while pinning `eval_im2col` (mutation audit of
`oracle.rs`). Gates only loosen: every plan that built before still builds, and no key bytes change. The emitters have not been
exercised on the newly admitted shapes.

## `0.20.5` — tests only: mutation audit of `structure_key.rs` and `oracle.rs`, clusters 1 and 2

`cargo-mutants` 27.1.0. Two earlier test-only PRs merged to `main` **without a version bump**, so `main` stayed at `0.20.4`; their
tests ship in this release: #71 (`structure_key.rs` codec and helper pins, and a `synthesis_lowers_the_optimized_body`
integration test) and #72 (`oracle.rs` cluster 1: the fp8/f16 encoders and the scalar evaluators). This release adds cluster 2: the
fold, row-reduce, scan and window evaluators (`oracle::fold_pins`). The survivors that are provably equivalent are listed in the
PRs. No code, no key bytes, no API change.

## `0.20.4` — tests only: mutation audit of the contraction and plan checks

`cargo-mutants` 27.1.0 over `check_contraction`, `validate_contraction_roles`, `contraction_epilogue_admissible`,
`ContractionDecline`/`PlanError`, `ContractionRoles`, the contraction keyer and the legacy bridge. Survivors became pin tests
(`plan::contraction_check_pins`, `structure_key::contraction_mutation_pins`); the proven-equivalent ones are listed in the
PR. No code, no key bytes, no API change.

## `0.20.3` — tests and docs only: two `allow(dead_code)` and one ignored doctest removed

CireSnave 5.1g (`#[ignore]` and `#[allow(dead_code)]` are last resorts). `oracle.rs` `coverage`: spurious allow
removed. `tests/common`: `OtherStub` moved into `provenance.rs`, its only user, so the shared module no longer
needs a crate-wide `allow(dead_code)`. `sku.rs` `ozaki_slices`: the ```` ```ignore ```` example (it named a crate
we do not depend on) is a real doctest. No code, no key bytes, no API change. One `#[ignore]` remains
(`emitted_slang_compiles_with_slangc`), a documented last resort: see `ROADMAP.md` row 1.

## `0.20.2` — comments only: `rm` is a permission to round, never a requirement

Two comments (the `from_op` bridge in `structure_key.rs`, and `tests/ctr_roles.rs`) said an `f32`
contraction "rounds its inputs". KISS-OPS 6.17-0003 makes `rm` a permission (KISS#593); the comments now
say so. No code, no key bytes, no API change. `ROADMAP.md` refreshed to 0.20.1.

## `0.20.1` — a contraction cell the plan cannot serve is a typed refusal, not a panic

`try_build_plan` gains `check_contraction`: the `Access::Contraction` arm of the plan gate no longer
`assert!`s. A key derived from declared roles (`ContractionRoles`, 0.19.0) can name a cell no `OpDef` pairs
with, so the refusal is reachable by *requesting* a cell and owes a typed decline (KISS-EMIT §6.8-0004).

- **Added** `PlanError::UnservableContraction(ContractionDecline)` and `ContractionDecline`
  (`RolesInvalid`, `KeyHasNoContractionFacts`, `EpilogueInadmissible`, `BatchDisagrees`,
  `BatchNotOutermost`), both `#[non_exhaustive]`, so this is additive. The first three are the old
  asserts, now typed; the last two are new: the op and the key disagree on whether the cell is batched
  (e.g. a rank-3 key whose caller declared no batch axis), and a real batch axis stored inside the matrix
  axes (the v1 emitters bind batch as the outermost stride; a broadcast batch axis is exempt). Rank 4 and
  above is `RolesInvalid` (v1 lowers one batch coordinate).
- **Behaviour:** `try_build_plan` / `try_generate` return `Err` where `build_plan` panicked; `build_plan` is
  the same gate's panicking wrapper, with a similar message. `try_generate` maps it into
  `LowerError::InadmissiblePlan`, so backends inherit the decline without code.
- **Not covered, deliberately:** each backend's own limits (a tile schedule, an accumulator spelling, a
  dtype list). Those remain the backend's decline.

## `0.20.0` — `ContractionAxes` (unpopped-vocab) is now `ContractionRoles` (BREAKING)

`unpopped::ir::ContractionAxes` (the *op's* per-input axis roles) and `unpopped_vocab::ContractionAxes` (0.19.0's
declared M/N/K and batch axes of the *key*) were two public types with one name. The vocab type is renamed, with
what hangs off it, while the surface is new. No behaviour and no token byte changes.

| 0.19.x | 0.20.0 |
|---|---|
| `unpopped_vocab::ContractionAxes` | `unpopped_vocab::ContractionRoles` (same API: `new`, `gemm`, `with_batch_axes`, `batch_axes`) |
| `CellFacts::contraction_axes` | `CellFacts::contraction_roles` |
| `DeriveDecline::ContractionAxesNotDeclared` (wire `ContractionAxesNotDeclared`) | `DeriveDecline::ContractionRolesNotDeclared` (wire `ContractionRolesNotDeclared`) |

`unpopped::ir::ContractionAxes` is unchanged. The 0.19.0 entry below names the old spellings, as shipped.

## `0.19.1` — the derivation vectors are re-vendored from KISS `026feba` (63 vectors)

Tests and docs only; no behaviour, API or token byte changes. KISS#592 corrected its `<acc>` warning
(the undeclared default is operand 0's dtype, not the weight's): one vector is renamed
(`ctr_weight_first_undeclared_accumulator_defaults_to_operand0_dtype`) and one added
(`ctr_weight_second_undeclared_accumulator_defaults_to_operand0_dtype`). Measured: PASS 63 | FAIL 0 |
DECLINE-DIFFERS 0 (of 63).

## `0.19.0` — the ctr stage of KISS 0.4.3: declared M/N/K and batch roles (BREAKING)

KISS 0.4.3's 62 derivation vectors now byte-match (PASS 62, from 35 at 0.18.1; `KNOWN_GAP` is empty).
No `STRUCTURE_KEY_VERSION` change: the token grammar is `sk6`'s.

**One derivation, role-driven.** A contraction's M/N/K and batch axes are *declared*
(KISS-CLASSIFY-6.6-0016, -0023), never read from operand positions; the old positional rank-2/3
deriver, which also hardcoded `<acc>`/`<mp>` from the lhs dtype and differed from the spec on 26 of the
27 new vectors, is gone from the conformant path.

- **Added** `AxisRef`, `ContractionAxes` (+ `ContractionAxes::gemm(rank, batch)` for the plain
  `[B..,M,K].[B..,K,N]` declaration), `BatchRoleDecline`, `CellFacts::contraction_axes`.
- **Derivation:** M/N/K size classes from the declared axes; the batch class is that of the **product**
  of the declared batch extents (a declared extent-1 axis is batched; no batch axis, no `b<class>`, at any
  rank); **rank above 3 derives**; `<acc>` is the declared accumulator, else operand 0's dtype (§6.7-0006;
  the weight's only when the weight is operand 0, so a weight-first caller must declare it); `<mp>` is the
  declared `math_fidelity`.
- **BREAKING — a declared contraction with no `contraction_axes` now declines**
  (`DeriveDecline::ContractionAxesNotDeclared`, this crate's name, as `WeightRoleNotDeclared`). Callers of
  `derive_structure_key{,_token}` that set `contraction_declared` must set the axes
  (`ContractionAxes::gemm(rank, batch)` is the old positional meaning) and, **to keep their 0.18 bytes,
  declare `accumulator` (`f16`/`bf16` -> `f32`, `i8` -> `i32`) and `math_fidelity` (`rm` for `f32`)**:
  the spec default is operand 0's dtype and `st`, so an f16 cell undeclared now keys `.../f16/f16/f16/st`
  where 0.18 keyed `.../f16/f32/f16/st`. Consumers that pin ctr goldens must regenerate them if they move
  to the spec default.
- **The legacy `op` bridge (`structure_key`, `try_structure_key*`) is unchanged in bytes**: it now
  *declares* the positional GEMM roles, the `f32` accumulator and the `f32`->`rm` policy that it always
  implied, and derives through the same role path (pinned by `tests/ctr_roles.rs` and the existing suite).
- **New typed declines:** `ContractionAxesNotDeclared`, `ContractionAxisInvalid`, `WeightRoleOutOfRange`,
  `ContractionNoOutput`, `Batch(BatchRoleDecline)` (`BatchAxisOutOfRange`, `BatchAxisDuplicate`,
  `BatchAxisOverlapsContractionRole`, `BatchAxisNotInOutput`, `BatchAxisNonPositiveExtent`),
  `ReservedContractionDtype`. `ScaleRoleOutOfRange`'s wire name is now `RoleSlotOutOfRange` (KISS's
  spelling, shared with the weight slot); the variant is unchanged.
- **Not decided here:** the key says nothing about whether an emitter can bind these axes. The v1 CUDA
  emitter binds one outermost batch axis; an emitter must decline what it cannot bind (KISS keys any
  declared batch axes, e.g. `ctr_rank4_one_batch_axis_declared`). The old "batch must be storage-outermost"
  decline is no longer a key-time decline.

## `0.18.1` — KISS 0.4.3's 62 derivation vectors are vendored; 27 are pinned as the known gap

Tests and docs only; no behaviour, API or token byte changes. `structure_key_derivation_vectors.json`
is re-vendored from KISS `f88d389` (36 -> 62 vectors: 15 `ctr` token vectors, 12 role-hint declines;
`decline_contraction_out_of_scope` removed) and the harness reads each vector's `contraction`
object where `CellFacts` can express it. Measured: PASS 35 | FAIL 7 | DECLINE-DIFFERS 20. The 27 are
named in `KNOWN_GAP` and asserted exactly, so the follow-up that implements the ctr stage (0.19.0,
Breaking) has to empty the list.

## Unreleased — `0.18.0`: `sk6`, the scale dtype is keyed (BREAKING)

**`STRUCTURE_KEY_VERSION` 5 → 6** (KISS#588, conformance 0.4.0,
KISS-CLASSIFY-6.6-0022). A quantized `ctr` cell keys the dtype of its designated
scale operand as the optional last contraction coordinate `q<dtype>`
(`…/st/qf8e8m0`, `…/st/qf16`). At `sk5` an `f8e8m0` and an `f16` scale derived
one key although they decode the same stored values to different numbers.

- **Every token changes its prefix** (`sk5|` → `sk6|`); an `sk5` token now parses
  to `UnsupportedSchemaVersion { version: 5 }`. Re-derive, do not rewrite: stored
  or announced tokens must be regenerated.
- **`ContractionKey::scale: Option<ElementKind>`** (+ `with_scale`) and
  **`CellFacts::scale_operand: Option<usize>`** (canonical slot, a declared role,
  never inferred). `None` omits the coordinate; every existing caller derives the
  same body as at `sk5`.
- **New declines** in `derive_structure_key`: `ScaleRoleOutOfRange` (a hint naming
  no operand is not silently dropped) and `ReservedScaleDtype`. Both names are
  this crate's. The reader declines a `q` coordinate with an unknown dtype
  (`UnknownDtype`), a reserved one (`ReservedDtype`), a second one, or one outside
  a contraction field.
- **Re-vendored from KISS `86196f7`** (conformance 0.4.0): vectors file 28
  positive / 25 decline, derivation vectors unchanged in outcome (35 PASS, 1 SKIP).

Not covered by KISS's derivation vectors (they exclude `ctr`, KISS#543): the
derivation of `q<dtype>` is pinned only by `tests/sk6_scale_dtype.rs`.

---

## `0.17.3`: the weight role is declared (additive)

**Fixed: `<wdt>` read operand 1 positionally**, which KISS-CLASSIFY-6.6-0019
forbids. Measured at `0.17.0`: `Gemm [W:f8e4m3fn, x:f16, y:f16]` derived
`<wdt>` = `f16` (the activation) while `[x:f16, W:f8e4m3fn, y:f16]` derived
`f8e4m3fn`, so a weight-on-the-left cell got a well-formed, byte-stable token
naming the wrong dtype.

- **`CellFacts::weight_operand: Option<usize>`** names the weight operand (canonical
  slot). `<wdt>` is that operand's dtype.
- **`derive_structure_key` declines a declared contraction without it**:
  `DeriveDecline::WeightRoleNotDeclared`. This name is this crate's, **not in
  KISS's decline vocabulary**. `0.17.0`'s `derive_structure_key` silently read
  operand 1 there; that API is one release old.
- **The `op = Gemm` bridge declares operand 1 and says so**, so no token from
  `structure_key` / `try_structure_key` changes. A caller whose weight is not the
  rhs (`W @ x`, a sibling-scale operand order) moves to `derive_structure_key`.
- **Out-of-range weight slot** is `ContractionOutOfScope`, not an index panic.

Still positional: the M/N/K axis roles (§6.6-0016), and the batch role is
undefined upstream (KISS#543).

---

## Unreleased — `0.17.2`

CI and docs only: `.github/CODEOWNERS` (`* @ciresnave`), a top-level
`permissions: contents: read` in `verify.yml`, and `docs/glossary.md`. No code,
token or API change.

---

## Unreleased — `0.17.1`

Docs only. Adds `ROADMAP.md` at the repository root: the queue, what each item
waits on, and the KISS items this lane consumes with their blobs. No code,
token or API change.

---

## `0.17.0` (breaking; shipped in `0.17.1`, never published on its own)

Moves `STRUCTURE_KEY_VERSION` 4 → 5 and aligns the keyer with KISS-CLASSIFY at
`KISS@ff12265` (KISS#535, KISS#536). **Every non-trivial token changes.** If you
cache on a token, re-derive; an `sk4` token is `UnsupportedSchemaVersion`.

### What changes in the token

| Field | `sk4` | `sk5` | KISS-CLASSIFY |
|---|---|---|---|
| Field 1 | 24 op-family codes (`bin`, `une`, `red`, `gem`, …) | the **derived** cell class: `ctr` `ndx` `rdc` `ew1` `ew2` `ew3` `oth` | §6.5-0006, §6.5-0015 |
| Operand sub-key | five parts | five parts, plus `/dl` or `/dc` iff the operand carries a `quant` record | §6.3-0009a, §6.7-0004 |
| Own extent-1 axis, non-zero stride, frame wider | mask bit clear | bit **set** (an own extent-1 axis counts as stride 0) | §6.6-0008, KISS#520 |
| Innermost axis of a reduction | `v4` / `v8` when aligned | `v1` | §6.5-0009(b) |

### ⚠️ BREAKING (API)

- **`StructureKey::op: OpCategory` is replaced by `StructureKey::cell_class:
  CellClass`.** The token no longer names an op family, so none can be recovered
  from it. `seed_winner` reads `CellClass::Ctr`.
- **`OperandKey` gains `dequant: Option<DequantForm>`; `OperandDesc` gains
  `role: OperandRole` and `dequant_form: Option<DequantForm>`.** All three types
  are `#[non_exhaustive]`; construct through `new` / `with_role` /
  `with_dequant_form`. `OperandRole::Unspecified` keeps the pre-sk5 convention
  (inputs, then the output), so existing callers key as before.
- **`TokenDecline::UnknownOpFamily` is `UnknownCellClass`** (and its wire name),
  and every retired sk4 op code lands there. New variants: `BadOperandSubKey`,
  `BadDequantForm`, `ReduceNotGatedToRdc`, `RdcWithoutReduce`,
  `ContractionPresenceMismatch` (§6.6-0017, §6.6-0010, §6.7-0004).
- **`DeriveDecline` gains** `ReservedDtypeAtOperand0`, `CollapsedReduction`,
  `ReducedAxisOutOfRange` and `ContractionOutOfScope`, with a `wire_name()` in
  the KISS vocabulary.

### New

- **`derive_structure_key` / `derive_structure_key_token`** take `CellFacts`
  (declared contraction roles, index role, reduced axes, accumulator,
  math-fidelity) beside the operands, which is KISS's derivation input
  (§6.6-0012), and put operands with roles in canonical order (§6.6-0014).
  `structure_key` and `try_structure_key` keep their `op` argument and build
  the facts from it through a bridge: `Gemm` declares contraction roles,
  `Indexing`/`Embedding` an index role, a keepdim `Reduction` the axes it folded.
  **That bridge is not an op-to-class table**; the class is still derived from
  the operands.
- **`CellClass`, `DequantForm`, `OperandRole`, `CellFacts`.**

### Fixed: four keyer defects against KISS (measured by baracuda at `0.16.0`)

Each is pinned through both the new and the legacy entry point in
`tests/sk5_keyer_defects.rs`, and by KISS's vectors.

| | Clause | `sk4` | `sk5` |
|---|---|---|---|
| (a) | §6.5-0009(b) | a reduced innermost axis derived `v4`/`v8` | `v1` |
| (b) | §6.6-0009 | a collapsed reduction output was keyed as a non-reduction | typed decline `CollapsedReduction` |
| (c) | §6.6-0009 | a reduced axis at or past the rank was keyed | typed decline `ReducedAxisOutOfRange` |
| (d) | §6.1-0001 | a reserved dtype (`f8e4m3fnuz`, `f8e5m2fnuz`) at operand 0 was keyed | typed decline `ReservedDtypeAtOperand0` |

**Not a defect, and left alone:** a reserved dtype at a *non-primary* operand
still derives. Only operand 0 is a key position (§6.6-0015); KISS ruled this on
2026-10-14 and the sk4 behaviour was right.

### Conformance

Against `kiss/structure_key_derivation_vectors.json` (36 vectors, blob
`697aa543`): **35 PASS, 0 FAIL, 1 skipped**, run as
`cargo test -p unpopped-vocab --test kiss_derivation_vectors -- --nocapture`.
Only vectors with `spec_golden` are spec-anchored; the rest are
reference-deriver output, so agreement there is two derivers agreeing. The one
skip is `decline_contraction_out_of_scope`, which waits on a batch role that is
undefined (KISS#543). The codec leg (`kiss_byte_match.rs`) passes 26 positive
and 21 decline vectors. `[8,0]` derives `da` / `v1`, as KISS reads §6.3-0011.

**Not covered:** a declared contraction. `ctr` cells are keyed as before from
plain `[lhs, rhs, out]` and bias operand lists; the weight role (§6.6-0019) is
still not an input (#31), and a declared contraction whose field cannot be
derived is now a `ContractionOutOfScope` decline instead of a `gem` token with
no contraction field.

### Other

- **Vendored from KISS `ff12265`:** `structure_key_vectors.json`,
  `dtype_manifest.json` and the new `structure_key_derivation_vectors.json`. The
  CRLF/LF re-vendor hold in `kiss/README.md` is lifted (KISS measured 0 CR bytes
  from both generators).
- **`unpopped-vocab` gains a dev-dependency, `serde_json`**, for the derivation
  harness. Its normal dependencies are unchanged.
- **Docs corrected:** the `[8,0]` "deliberately not changed" note below and the
  `docs/deferred.md` "ask KISS for a target-match vector" remedy were both stale
  (the vector has been vendored and asserted since the `bc16715` copy).

---

## `0.16.0` (breaking; published)

### f16/bf16 take FP8's shape: a portable carrier and an emitted codec

The last vendor spelling in the neutral `cfamily` module is gone. baracuda#154's local
shadow keeps every CUDA byte `baracuda-cuda-emit` emits unchanged.

- **`cfamily::scalar_ctype(F16 | Bf16)` returns `"unsigned short"`**, the
  storage carrier, instead of `"__half"` / `"__nv_bfloat16"`.
- **New `cfamily::half_helpers(ElementKind)`**: portable-C99 f16/bf16 codecs
  (`unpopped_f16_load`/`_store`, `unpopped_bf16_load`/`_store`), emitted with
  the kernel. Round to nearest, ties to even; NaN encodes as the canonical
  quiet NaN. Checked bit for bit through a real C compiler against the oracle's
  codec on all 65536 patterns and on 524288 products that need rounding; the
  oracle's codec is itself checked against the `half` crate.
- **`narrow_load_fn` / `narrow_store_fn` name the codec for the halves**, so
  `promote_load_f32`, `demote_store_f32` and `cast_scalar` emit
  `unpopped_f16_load(x)` where they emitted `__half2float(x)`.
- **Removed: `cfamily::half_load_intrinsic` and `half_store_intrinsic`.**
- **`store_expr_of` encodes a uniform half store once, at the store**, as for
  FP8: a half body root is now a plain f32 expression.
- **`unpopped-cpu-c` computes f16 and bf16.**
- **Fix: a narrow-float predicate stored a wrong mask.** The U8 store of a
  `Cmp*` body over FP8 decoded the f32 `0.0`/`1.0` root as if it were a storage
  pattern, so every true lane stored `0`. Live in `unpopped-cpu-c` since FP8
  support; `store_expr_of` now converts from what the root is.
- **kiss-ref `0.3.4` -> `0.4.0`** (`kiss-ref-core`, `kiss-ops-vocab`,
  `kiss-classify-vocab`; KISS contract v3). Its enums are now
  `#[non_exhaustive]`; the in-tree `kiss_ref_diff` test reference panics on a
  `Monoid` it does not know rather than folding it as another. No published API
  of ours changes. `unpopped-cpu-c` takes `kiss-ref-core 0.4.0` as well.

---

## 2026-10-11 — everything at `0.15.0`

### ⚠️ BREAKING — `__expf` / `__logf` no longer lift (they are `Unrecognized`)

`unpopped::lift` and `unpopped::convert` (CUDA frontend) used to map CUDA's
fast-math `__expf` and `__logf` to the exact `UnaryOp::Exp` / `Log`, which the
shared lowering spells `expf` / `logf`. A lift-and-re-emit silently turned an
approximation into the accurate function: a numerics change with no record
(idiom-lifting rule 2, `docs/idiom-lifting-design.md` §2).

- **What used to lift and now is `LiftError::Unrecognized`:** any kernel body
  calling `__expf(x)` or `__logf(x)`. `expf` / `logf` are unchanged.
- **No signature changed:** `unary_fn` is `pub(crate)`. Callers that matched
  `Ok` from `lift_elementwise` on such source now get `Err`.
- **Adopter action:** tests that pin the old mapping must expect
  `Unrecognized` (baracuda's `baracuda-cuda-parse/tests/convert.rs:84` does).
  Spell `expf` / `logf` if the exact function is what the kernel means.
- **Interim shape:** an `Approx` qualifier (§4.3) is still the full fix and is
  not in this release.

## 2026-10-11 — everything at `0.14.6` (CI only)

ci: `cargo deny` job and `deny.toml`; every workflow `uses:` pinned by full
commit SHA. No code, emitted byte or golden moves; the number changes because a
pushed change changes the version.

## 2026-10-11 — everything at `0.14.5` (docs only)

docs: 0.14.4 published, stale claims corrected. No code, emitted byte or golden
moves; the number changes because a pushed change changes the version.

- `README.md` no longer says the emitters are unpublished (false since 2026-08-19).
- `crates/unpopped/README.md` says the emitters are at 0.14.4, not 0.3.0.

## 2026-10-08 — everything at `0.14.4`

### `HalfArith`: the first plan decision that differs by sm (additive)

U1 of the joint P40 (sm_61) / RTX 4070 (sm_89) plan
(`docs/joint-gpu-milestone-plan.md`). No emitted byte or golden moves.

- **`TargetCapabilities` gains `fp16_results_per_clk_per_sm` and
  `fp32_results_per_clk_per_sm`** (`Option<u32>`), transcribed from NVIDIA's
  12.9.1 "Throughput of Native Arithmetic Instructions" table (page sha256
  `06499a0c…`). sm_61 does 2 fp16 results per clock per SM against 128 for fp32;
  sm_89 does 128 for both. The table has no column for 8.7, 10.3, 11.0 or 12.1,
  so those are `None`. `from_queried` leaves both `None`, and
  `with_arith_throughput` attaches sourced figures.
- **`KernelPlan::half_arith()` and `half_arith_for(TargetId)` return `HalfArith`
  (`Native` / `ViaF32`).** `ViaF32` is returned only when both rates are sourced
  and fp16 < fp32, which today means sm_61 only. Everything else is `Native`,
  the behaviour every emitter already has.
- It is a method, not a `KernelPlan` field, because `KernelPlan` is not
  `#[non_exhaustive]` and emitters build it by struct literal.
- Consumer: baracuda-cuda-emit gates its packed native `__h*2` path on
  `Native`. That path is bit-identical to the scalar float path by design, so on
  the P40 this changes speed, not numerics.

### Also in this change set (no API or emitted-byte change)

These merged after `0.14.3` with no version change of their own, and ship under
`0.14.4` (the PM's ruling, 2026-10-07: an unpublished number covers everything
merged under it). The change set is #37–#51, except #49.

- **Docs in the crate:** `telemetry::cuda_target_of` documents that its
  `sm{major}{minor}` format is not injective in principle (#37).
- **Tests:** `unpopped-vocab` pins what the frame broadcast mask says about
  collapsed outputs (`a_collapsed_output_is_absent_not_broadcast.rs`, #38).
  `unpopped` checks that every Qwen3 dense op plans and generates for
  `cuda:sm61` (`qwen3_dense_ops_plan_for_sm61.rs`, U2, #42).
  `unpopped-conformance`'s `golden_regen_checkpoint.rs` follows the
  `docs/deferred.md` split and the 2026-11-05 checkpoint (#43, #45).
- **Repository docs (not in any crate):** the joint P40 (sm_61) / RTX 4070
  (sm_89) milestone plan, `docs/joint-gpu-milestone-plan.md` (#39, kept current
  in #41, #46, #47, #48, #50, #51); `docs/deferred.md` Section B split, with the
  f16/bf16 shadow surface counted as eleven functions (#43, #45); and the
  qwen3_5 linear-attention assessment (#44).

## Released

⚠️ **What reached crates.io, read from its API on 2026-10-03:** `0.11.0`
(2026-09-11), `0.12.0` (2026-10-01), `0.13.0` (2026-10-01), `0.14.0` (2026-10-02), `0.14.1`
(2026-10-03), for all four published crates. `0.14.3` (2026-10-03) was confirmed
from the same API on 2026-10-07; `0.14.2` was never published. `0.14.4` was
published 2026-10-08 (`unpopped-vocab` 21:32:11Z, `unpopped` 21:33:14Z,
`unpopped-cpu-c` 21:33:43Z, `unpopped-slang` 21:33:59Z, all `yanked=false`,
read back from the crates.io API); `unpopped-conformance` is `publish = false`
and stays absent. **`0.11.1`–`0.11.6` existed only in the workspace and were
never published.** A consumer went straight from `0.11.0` to `0.12.0`, so
everything those intermediate numbers carried is listed under `0.12.0`.


## 2026-10-03 — everything at `0.14.3`

### Fixed: telemetry from any compute capability now merges, and every drop is counted

**Behaviour change: H100 telemetry now counts.** `telemetry::merge_reports` used to
stamp a record through `arch_sku_of`, which has answers only for 8.x and 9.x, and
dropped every other capability before the merge saw it (sm_61, sm_70, sm_75,
Blackwell). Its 9.x answer was `cuda:sm90a`, but Fuel keys a Hopper record
`cuda:sm90`, so `merge`'s arch gate (stamp target == key target) rejected every
H100 report as well. Both drops were silent.

- **New `cuda_target_of(major, minor)`:** `cuda:sm{major}{minor}`, digits
  concatenated, no `a` suffix. This is baracuda's `cuda:` rule (it owns the
  namespace, §6.8-0004), and it is the same rule Fuel keys records with.
  `merge_reports` stamps with it.
- **Changed rows, on purpose:** 9.0 now stamps `cuda:sm90` (was `cuda:sm90a`), so
  H100 reports keyed `cuda:sm90` now merge into the `cuda:sm90` cells, where
  before they changed nothing. 8.6/8.7 now stamp `cuda:sm86`/`cuda:sm87` (was
  `cuda:sm80`); Fuel builds no key for those today, so no live row changes.
  6.x, 7.x, 10.x and 12.x now merge into their own targets. 8.0 and 8.9 are
  byte-identical.
- **New `merge_reports_counted`** returns a `MergeReport`: merged records per
  full target token, plus counts for no stamp, no capability, an unformable
  target, an unknown backend, a declined entry, and an arch-gate reject.
  `merge_reports` is the same fold with the report discarded.
- `arch_sku_of` is unchanged. It still answers "which cutlass dispatch cell".

## `0.14.2` (docs only, never published; its content shipped in `0.14.3`)

### Docs only: `TargetId` is the architecture identity; `ArchSku` is a cutlass dispatch SKU

No code, signature or behaviour change. This release records board #106 option C
(2026-10-03), which every affected lane approved with CireSnave's prior authorisation:

- **`TargetId` names every target, CUDA architectures included.** Any
  architecture is spelled as a token (`cuda:sm61`, `cuda:sm100`, …) whether or not
  `ArchSku` has a variant for it. The `unpopped-vocab::target` module docs and
  `TargetId`'s doc now say so.
- **`ArchSku` is re-scoped to a baracuda-cutlass dispatch SKU.** It stays closed
  and exhaustive, with no new variants and no freeze. A missing variant is not a
  missing architecture. Whether to add one (Blackwell `Sm100a`, say) is baracuda's
  call, made for its cutlass dispatch. Its doc also lost a pointer to a ROADMAP that
  this repo doesn't have.
- **`docs/idiom-lifting-design.md` §6:** the planned `ArchSku::Sm61` breaking
  release is marked superseded. No variant is added. The capability row shipped in `0.14.1`.

The `telemetry::merge_reports` drop this entry listed as a follow-up is fixed in `0.14.3`.

## 2026-10-03 — everything at `0.14.1`

### `cuda_capabilities` / `capabilities_for` cover sm_61 through sm_121 (additive)

These rows were missing, so they returned `None`. Each is now transcribed from
NVIDIA's own tables, parsed from the HTML rather than copied by hand. The
sources and the verification state are recorded on `cuda_capabilities`:

| New rows | Source | Toolkit accepts the arch | Verified on hardware |
|---|---|---|---|
| 6.1 | CUDA C++ Programming Guide **12.9.1** | CUDA 12.9 (NVRTC); not 13.x | no (awaits a P40) |
| 10.0, 10.3, 11.0, 12.0, 12.1 | CUDA Programming Guide (13.x edition), Sep 10, 2026 | CUDA 13.3 | no, compile-level only |

**Every existing row was re-checked against the same sources and is
unchanged.**

**Deliberately absent:**
- **10.7**: NVIDIA tabulates it, but CUDA 13.3 doesn't accept it.
- **8.8**: CUDA 13.3 accepts it, but neither table has it.
- Anything below 6.1.

**The two NVIDIA editions disagree on 12.x resident blocks per SM.** The CUDA
12.9.1 edition says 32 and the current edition says 24. The row uses **24**,
the later figure. A launcher must query the device regardless.

This is part of CireSnave's ruling that baracuda supports sm_61 through the
latest architecture via Unpopped's CUDA parse/emit. Nothing in Unpopped's IR is
arch-specific. These rows only inform which schedule variants are worth
offering.

## 2026-10-02 — everything at `0.14.0`

One breaking release that conforms `unpopped-vocab`'s structure-key derivation
to KISS-Classify as of **KISS#517 + KISS#519** (`KISS@bc16715` plus the #519
amendment). Tracking: #31.

⚠️ **Wire-visible. Tokens change for the inputs below.** If you cache on the
token, re-derive the affected entries rather than migrating them.

**Every same-rank cell with ordinary strides keys exactly as in 0.13.0.** The
KISS byte-match vectors and every pre-existing token test pass unchanged.

### Which tokens change

| Input | `0.13.0` | `0.14.0` | KISS-CLASSIFY |
|---|---|---|---|
| A lower-rank operand in a higher-rank cell, e.g. `[256]` with `[128,256]` | mask `00` | mask `01` (in frame coordinates) | §6.5-0014 (as amended by #519), §6.6-0008, §6.6-0013 |
| An own extent-1, stride-0 axis where the frame is wider, e.g. `[1,256]` strides `[0,1]` | mask `00` | mask `01` | §6.6-0008 |
| A trailing unit axis, e.g. `[4,1]` | `d4` (and possibly a packed width) | `da` / `v1` | §6.3-0011 |
| A zero-extent innermost axis, e.g. `[8,0]` | `d8` | `da` / `v1` | §6.3-0011, §6.5-0012 |
| A mixed-rank work class, e.g. `[2,64]` with `[64]` | `grid` (left-aligned frame: 4096) | `block` (right-aligned: 128) | §6.6-0013, §6.5-0010 |
| A token naming `f8e6m2` | parsed and keyed | **typed decline** (`ReservedDtype`) | §6.1-0013 |

**Unchanged on purpose:** a rank-deficient operand is still laid out from its
**own** axes. It is not `br` and not forced to `v1` merely for lacking a frame
axis. KISS#517's first text padded the layout too. That gave im2col's dense and
strided rank-3 outputs the same token, which Unpopped's im2col gate caught.
KISS#519 amended the clause.

### New API: `try_structure_key` / `try_structure_key_token`

§6.6-0021: a scale-type dtype (`f8e8m0`, `f8e6m2`) at operand 0 has no defined
primary dtype, and derivation MUST decline. The new functions return
`Result<_, DeriveDecline>` with `DeriveDecline::ScaleDtypeAtOperand0 { dtype }`.
On every other input they return exactly what the infallible pair does.

**`structure_key` / `structure_key_token` stay, undeprecated for now.** They key
a scale-first list with the scale's dtype, which is non-conformant for that one
input, and their docs say so. `#[deprecated]` arrives in the release that
migrates this workspace's own ~200 call sites.

A scale in any *other* operand slot (the sk4 sibling model) is unaffected.

### Consumers in this workspace

- **The im2col plan gate** now admits mask bit 0, the frame axis a rank-3 output
  cannot span, and still requires the output's own layout to be dense.
- **`contract::index_is_1d`:** a rank-deficient index tensor gathered along the
  **last** axis now reads as a 1-D (broadcast) index. Its frame mask has the
  leading bits set, where its empty own mask used to read as full-shape. This
  is the right reading, but it changes the emitted contract for that input. No
  in-tree test exercises a rank-deficient index.

### Vendored KISS artifacts

- Re-vendored from `KISS@bc16715`, verbatim; the files hash to KISS's blob IDs.
- The Vulkan vocabulary moved 4 → 5. It was re-verified, not just re-pinned: v5
  only adds `i16`/`i64`/`f64` arith names, and no vendored vector names them.
- KISS's §6.8-0002 discriminating target-match pairs are now asserted.

### Not in this release

- The §6.6-0019 weight-role `<wdt>` (the `gem_weight_role_discriminator`
  vector). It needs role hints in the derivation API, and has a separate ticket.
- The stride-0 vs nonzero-stride unit-axis mask question (two tokens for one
  broadcast meaning). The derivation follows the text literally, and KISS has
  raised the question with the PM.

**Cross-checked against an independent implementation:** the `[4,1]` token and
the right-aligned work class match Fuel's own deriver, run on the same inputs.
The #519 examples are pinned as goldens.

> **About the merge commit.** `9574050`, the squash of #32, is titled *"(DRAFT) …
> awaiting baracuda + Fuel sign-off"*. The squash took the PR's title as it stood
> before the sign-offs arrived. Both had landed before the merge, and both are
> recorded on #32: Fuel's by its own re-run (fuel#285), baracuda's as relayed by
> the portfolio PM. Published 2026-10-02 07:43–07:44Z from that commit, in the order
> `unpopped-vocab` → `unpopped` → `unpopped-cpu-c` → `unpopped-slang`.

## 2026-10-01 — everything at `0.13.0`

### ⚠️ Structure keys change for two zero-valued inputs (wire-visible)

Two derivations read zero as "maximally divisible", because zero passes every
`x % n == 0` test. Both were live in `0.12.0` and both are fixed (#29):

| Input | `0.12.0` derived | `0.13.0` derives | KISS-CLASSIFY |
|---|---|---|---|
| A contraction with `K = 0` (rank 2 or 3) | `k_div` `d16` | `da` | §6.5-0012, §6.6-0016 |
| An operand with `align_bytes = 0` (unspecified alignment) | a packed width, e.g. `v4` | `v1` | §6.5-0009 |

**Who is affected:** only callers who build those two inputs. Every other key is
byte-identical; the full workspace and its KISS byte-match vectors pass
unchanged. **If you cache on the token, a cached entry for either input was
keyed under the wrong cell.** Re-derive it rather than migrating it.

**Why it went unnoticed:** the per-operand sub-key reaches the divisibility
ladder only through `inner_axis`, which picks an axis of extent `> 1`, so that
path never passes `0`. The contraction path calls it on the raw K extent. An
outside report of the K case was retracted on the strength of the per-operand
path, and measuring through the public `structure_key` reversed the
retraction. Pinned by `unpopped-vocab/tests/zero_is_not_maximally_divisible.rs`,
whose three tests fail on `0.12.0`.

**Not changed here, changed in `0.14.0`:** an operand whose innermost axis has
extent `0` (e.g. `[8, 0]`) derives `da` / `v1` from axis `rank − 1`, not `d8`
from the outer axis. KISS closed the "active axis" question for §6.3-0011
(innermost active axis is axis `rank − 1`, no non-unit exclusion; §6.5-0012
buckets `E = 0` as `da`). Measured at `0.16.0` and at sk5: `[8,0]` strides
`[1,1]` keys `co/00/v1/da/f`, as KISS's
`innermost_axis_extent_zero_is_axis_rank_minus_1` vector says.

## 2026-10-01 — everything at `0.12.0`

Covers `0.11.1`–`0.12.0`, none of which was published before this.

### ⚠️ BREAKING — the `seam` feature and `unpopped::jit::seam` are removed

`0.11.0` shipped an optional `seam` feature (`seam =
["dep:fuel-kernel-seam-types"]`) and `#[cfg(feature = "seam")] pub mod seam` in
`jit.rs`, which converted Fuel's `PatternNode` into this crate's. Both are gone,
along with the `fuel-kernel-seam-types` dependency (#18).

**Why:** this crate's public API named another project's type, so every major
release of `fuel-kernel-seam-types` forced a major release of `unpopped`. It had
already happened once. A `0.10.3` → `0.11.2` bump put two incompatible
`PatternNode` types into `baracuda-cuda-emit`'s dependency graph.

**Migrate:**
- To synthesize from this crate's own `unpopped::pattern::PatternNode`, call the
  native entry point `unpopped::jit::synthesize(&JitRequest, ..)` directly.
- If you start from Fuel's `PatternNode`, the conversion now lives in
  `baracuda-cuda-emit` (`src/seam.rs`, relocated near-verbatim). That crate
  already depends on `fuel-kernel-seam-types` directly.
- `--all-features` no longer turns on anything but `convert`.

The five tests that reached synthesis only through the seam (happy path,
recursion, three typed declines) now call the native entry point, in
`tests/native_synthesize_reaches_core.rs`. Before this change, no test in the
crate called that entry point directly.

### `convert` (the CUDA/Slang lifter, off by default) lifts more

- **Local variables resolve (#22).** A local that is assigned exactly once is
  substituted into the lifted body. Compound assignments (`+=`) and `++`/`--` count
  as further assignments, so a running accumulator is never mistaken for a
  constant. The crate's own scan-vs-elementwise test caught a first draft that
  missed them.
- **Comparisons and the ternary operator lift (#25)** to `Cmp*` and `Select`.

Both target constructs the lifter previously *refused*: an unresolved
identifier, or an unrecognized comparison token. They are meant to widen what is
accepted, not to change what an already-accepted kernel lifts to. That is the
design intent. No before/after comparison over a corpus has been run to measure
it.

### `unpopped-vocab`: the `cuda:` reserved id block is gone (#15)

The four `cuda:sm*` target tokens were interned at fixed ids `0`–`3`, the last
CUDA-specific vocabulary in this neutral crate. They now intern like every
other namespace's tokens, via `TargetId::parse`. **Only code that relied on a
`TargetId`'s numeric value could notice, and that was always forbidden:** ids
are process-local and never serialized. Tokens and `==` are unchanged.

### Also in this range, with no behaviour change

- **Doc comment correction (#8):** the `VariantFidelity::DeterministicallyDivergent`
  doc comment no longer cites the retracted measurement (see the 0.11.0
  correction below). The fix landed after `0.11.0` was packaged, so `0.12.0` is
  the first published version to carry it.
- **New tests that pin existing behaviour:**
  - the one-version rule (#7);
  - dtype `storage_bits`/`kind` checked against KISS's manifest (#16);
  - `Complex64`'s width (#19).
- **Dependencies:**
  - lockfile refreshed to the latest in-range versions (#20);
  - `kiss-ref-core`/`kiss-ops-vocab`/`kiss-classify-vocab` `0.3.4` (#24), a
    dev-dependency only, never reaching consumers.
- **Design and sizing docs:** #10–#12, #14, #26, #27. The
  `golden_regen_checkpoint` date moved to 2026-10-22 (#28). CI's checkout
  action moved to v7 (#9).

## 2026-09-09 — everything at `0.11.0`

⚠️ **`unpopped-vocab 0.4.0 → 0.11.0`, `unpopped-cpu-c 0.9.0 → 0.11.0`,
`unpopped-slang 0.7.0 → 0.11.0`, `unpopped 0.10.0 → 0.11.0` — the skipped version
numbers are a DELIBERATE UNIFICATION, not a mistake, and this line exists so
nobody re-derives it later.**

**CireSnave's standing rule, quoted:** *"I always want all crates within a project
to use the same version number so developers consuming them know which go with
which."* The exception in that rule is for a **cross-project** pair — an emitter
crate in another project whose number answers *"which Unpopped does this work
with"*. `unpopped-cpu-c` and `unpopped-slang` are **inside** this project, so
their numbers answered nothing and the exception does not reach them.

**`unpopped-conformance` moves too, although it is `publish = false` and has no
external consumer.** Leaving one crate off the shared number recreates the
question the rule exists to close.

### ⚠️ For consumers: one number, written four times, and all four must move together

Before this release a consumer wrote **three different numbers** for four crates.
Now they write `0.11.0` four times.

**Every current pin excludes `0.11.0`, so nothing arrives on a `cargo update`:**

    ^0.10.0  = >=0.10.0, <0.11.0     excludes 0.11.0
    ^0.4.0   = >=0.4.0,  <0.5.0      excludes 0.11.0
    ^0.9.0   = >=0.9.0,  <0.10.0     excludes 0.11.0
    ^0.7.0   = >=0.7.0,  <0.8.0      excludes 0.11.0

⚠️ **A PARTIAL edit resolves two versions of a crate into one graph** — e.g.
moving `unpopped` to `0.11.0` while leaving `unpopped-vocab` at `^0.4.0` gives
`0.4.0` (the direct pin) *and* `0.11.0` (what `unpopped 0.11.0` requires), which
are different types with the same name. **Unification makes that mistake more
visible, not less: four identical numbers make a missed line obvious, where three
different numbers made it look plausible.**


**Remedies issue #4 for TODAY'S INSTANCE ONLY.** ⚠️ Publishing makes published ==
workspace **at one moment**; the member keeps evolving and the gap reopens on the
next unpublished commit. **A green publish is not the class being closed** —
baracuda's argument, and the portfolio gate vulkane is building is the durable
half.

### ⚠️ BREAKING — `VariantFidelity::ReassociatedDeterministic` → `DeterministicallyDivergent`

The old name asserted **reassociation** for every member of the class. The new one
names the class extensionally — deterministic on fixed hardware, different bits
from the default, undirected — and **asserts no mechanism**.

**One selection policy, so one variant** — a fifth with identical semantics is a
distinction no consumer can act on. **The FKC determinism spelling is unchanged
(`same_hardware_bitwise`)**, asserted by `the_fidelity_rename_is_wire_invisible`.

> 🔴 **CORRECTION, 2026-09-11 — this entry originally cited a RETRACTED
> measurement, and the retracted text is immutable in the published `0.11.0`.**
>
> It read: *"baracuda measured a member whose reduction tree was provably
> unchanged and whose bits differed anyway — cache-vs-recompute of an equal
> `expf`, 12,283,172 of 16,777,216 elements, worst 11 ULP, reproducible."*
>
> **That measurement was withdrawn on 2026-09-09 (baracuda#99, closed
> `not_planned`), two days before `0.11.0` shipped.** It had been taken against
> `variants[1]` — which is `prec`, declared `MorePrecise`, whose entire purpose is
> to differ from the base. Measured against `smemrow`'s own kernel: **0 ULP over
> 16,777,216 elements.** `smemrow` is `BitIdentical` and always was.
>
> ⚠️ **So the second mechanism this rename was argued from has NO measured
> instance.** It is recorded in `backend.rs` as a hypothesis rather than an
> observation.
>
> **The name is kept.** The old name asserted a mechanism for every member; this
> one asserts none, and **removing an unsupported claim is not the same act as
> adding one.** Reverting would re-assert *"every member is a reassociation"* on
> evidence no better than what was withdrawn, and would cost the one adopter a
> second 19-site rename. **If the class turns out to be reassociation-only, this
> name is less specific than it could be, which is not wrong.**
>
> **How it happened:** the retraction reached baracuda's issue and two of their
> code comments, and never reached here. ⚠️ **A RETRACTION MUST TRAVEL AS FAR AS
> THE CLAIM DID — and a correction is *more* discoverable than the original while
> still not travelling, because discoverability is pull and propagation is push.
> Nobody goes looking at a number they have already written down.**

### `ulp_bound` no longer declines an expression whose exactness is by construction

An expression built entirely from `BinaryOp::is_int_only` operators has no
rounding step on any hardware, so a **non-CUDA** target now gets `0.0` where it got
`INFINITY`, and `precision_of` rates it `("correctly_rounded", Some(0))` instead
of `("approximate", None)`.

⚠️ **Keyed on `is_int_only`, NOT on `ulp_sum(e) == 0`.** `unary_ulp` rates
`Sqrt`/`Recip`/`Floor` at 0.0 and `binary_ulp` rates
`Copysign`/`Nextafter`/`FmaxIeee` at 0.0 — **those zeros are IEEE claims about a
target's float unit**, exactly the borrowed assertion the namespace gate exists to
refuse. **A CUDA target is untouched**, asserted across four rating tiers.

Found by baracuda while adopting 0.10.0, applying this repo's own rule: *a
prescription that errs conservatively has no complainant.*

### `Access::Contraction` is documented ALWAYS COMPUTED, and its premise is guarded

"No predicate applies" is an answer; its absence read as an oversight. The K-fold
is a sum over products and the variant carries **no operator field at all**.
Guarded by an exhaustive match on `AccumSpec` — **0 exhaustive matches existed
before, so a variant that falsifies the ruling used to compile with 0 errors, 0
clippy warnings and 0 test failures.**

### ⚠️ Why the emitters move at all

`unpopped-cpu-c 0.9.0` and `unpopped-slang 0.7.0` **both require `unpopped =
"0.10.0"`** — measured from their served manifests. For a 0.x crate `^0.10.0`
**excludes 0.11.0**, so publishing `unpopped` alone would strand both emitters on
the old line while every sibling moved. **`baracuda-cuda-emit` declares all four**,
so it would then resolve two `unpopped` versions into one graph — the two-artifacts
defect this release exists to close.

**Source of both emitters is byte-identical to their published versions. They move
because a dependency did, and then again because of the unification above.**

⚠️ **This section was headed *"Why three crates and not one"* until the
unification, and the count was correct when written.** The unification made it
five, and **a heading carrying a stale number is the artifact a reader trusts
most** — it was found by a reviewer's nitpick about an unrelated count two
sections above, which is the only reason anyone re-read this one.

## 2026-09-06 — `unpopped-vocab 0.4.0` · `unpopped 0.10.0` · `unpopped-cpu-c 0.9.0` · `unpopped-slang 0.7.0`

⚠️ **Everything below shipped. It sat under "Unreleased" for forty minutes after
being published** — caught while adding the next entry, which is the only reason
it was caught at all. **A changelog whose "Unreleased" section describes released
work is worse than no changelog: it is the same object as a version number that
no longer matches its code, one level up.**

**Published and verified against the served tarballs; `unpopped-vocab 0.4.0` and
`unpopped 0.10.0` src trees are byte-identical to the merge commit.**

**Version cascade applied 2026-09-06 on the PM's ruling: `unpopped-vocab 0.4.0`,
never `0.3.3`.**

| crate | was | now | why it moved |
|---|---|---|---|
| `unpopped-vocab` | 0.3.2 | **0.4.0** | two behaviour changes, one numeric |
| `unpopped` | 0.9.0 | **0.10.0** | new public fn, a deprecation, and its vocab requirement moved |
| `unpopped-cpu-c` | 0.8.0 | **0.9.0** | ⚠️ **source BYTE-IDENTICAL to the published 0.8.0** — bumped ONLY because a dependency did |
| `unpopped-slang` | 0.6.0 | **0.7.0** | ⚠️ **source BYTE-IDENTICAL to the published 0.6.0** — bumped ONLY because a dependency did |
| `unpopped-conformance` | 0.1.0 | 0.1.0 | `publish = false`; not on the registry |

⚠️ **The two "byte-identical" rows are stated because a reader who sees a version
move with no behaviour change will otherwise assume there was one they cannot
find.** Both were verified by diffing the served `.crate` tarball against the
tree, not by inspection.

### Downstream, measured rather than reasoned

**Registry reverse-dependencies of `unpopped-vocab`: 6, all in this portfolio**
(control: `serde` returns 120,294, so the query works).

    baracuda-cuda-emit / -kernels-types / -types   require ^0.1.0  (an OLD line;
                                                    0.3.x never reached them)
    unpopped / unpopped-cpu-c / unpopped-slang     require ^0.3.2

⚠️ **And the hop that mattered was the one nobody had measured.** baracuda's
**working tree** declares `unpopped-vocab = "0.3.2"` **directly** — measured at
their `origin/main` `2b3292ca`, after fetching, because the local ref was stale.
**So the reassurance "baracuda is caret-pinned `^0.8.7` on `unpopped`, therefore
cannot reach the new vocab" was FALSE**: their direct vocab dependency would have
taken `0.3.3` on the next `cargo update`, bypassing the `unpopped` pin entirely.

**Their exposure to the numeric change is nonetheless zero, measured on their
side:** `Fp8E5M2::from_f32` has **0** call sites in baracuda. **`Fp8E4M3FN` is
unchanged by this release** — 0 diff lines between published 0.3.2 and the tree
mention E4M3, against a control of 5 mentioning E5M2 — so their two E4M3 call
sites are unaffected.


### `unpopped-vocab` 0.3.2 → 0.4.0 — the divergence that forced this release

**Two behaviour changes. Neither could ship as `0.3.3`.** The then-published
`unpopped 0.9.0` requires `unpopped-vocab = "0.3.2"`, a caret range that
**accepts 0.3.3** — so a patch bump reaches every existing consumer on their next
`cargo update`, with no compile error and no version signal.

- **`ElementKind` E5M2 `from_f32` overflow now produces INFINITY, not a saturated
  max-finite.** Overflow at `|x| >= 61440.0` (the midpoint between max-finite
  `57344` and `2^16`; the midpoint itself rounds to infinity, because
  ties-to-even picks the significand ending `00`) returns `sign | 0x7C`.
  Previously it delegated to `float8` 0.7.0, whose E5M2 encoder saturates **every**
  overflow to `0x7B` — including a literal `f32::INFINITY` — while **its own
  decoder maps `0x7C` to infinity**. ⚠️ **This is a NUMERIC change to emitted fp8
  bytes**: a value that encoded as `57344.0` now encodes as `inf`.
- **`winner_of` resolves an exact timing tie deterministically.** Previously
  `sort_by` stability made the winner, its `entry_point`, and the `margin` depend
  on the order the caller pushed candidates. The tiebreak is
  median → implementor code → entry point and carries **no** performance or
  preference meaning. ⚠️ **Emitted dispatch tables can differ for tied cells**,
  which is generated code a consumer has committed.

### `unpopped` 0.9.0 → 0.10.0

- **Added `ir::is_bit_move_row_reduce_output(stages, epilogue)`** — the
  §6.16-0011 classification for the `Access::RowReduce` shape, which had no
  helper at all. Purely additive.
- **Deprecated `ir::is_bit_move_reduce`** — it answers a fold question for shapes
  that may have several folds. Non-breaking; the two replacements have signatures
  that cannot be satisfied without the whole input→output path.
- **Corrected the `RowReduce` gating prescription in `ir.rs`'s field table.** The
  old advice classified an all-move multi-stage fold as *computed*. ⚠️ **It erred
  CONSERVATIVELY, so a consumer following it emitted pessimised-but-conforming
  code and had no symptom to report.**

### `unpopped-cpu-c` 0.8.0 → 0.9.0 · `unpopped-slang` 0.6.0 → 0.7.0

- **Published 0.8.0 source is IDENTICAL to the tree.** No bump needed on its own
  account; it moves only if its `unpopped` requirement does.

---


Versions at or before `unpopped 0.9.0` / `unpopped-cpu-c 0.8.0` /
`unpopped-vocab 0.3.2` predate this file. **Their contents were verified against
the served tarballs rather than reconstructed from memory**, which is the only
claim this file makes about them.
