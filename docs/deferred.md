# Deferred work

Everything knowingly left undone, and — more importantly — **why**, and **what
unblocks it**.

This file exists because the reasoning is the perishable part. A deferral whose
justification is lost becomes indistinguishable from an oversight, and gets
either re-litigated or silently done wrong. Several entries below are deferred
*because doing them naively is worse than not doing them*, and that is not
recoverable from the code.

Grouped by what unblocks each, since that is the actionable axis.

> **Keeping this honest.** `docs/conformance.md`'s scope list rotted once — it
> claimed the oracle could not evaluate contractions long after it could, and a
> reader would have skipped writing a test that already worked. An over-cautious
> register is not harmless. When an item lands, delete it here in the same
> commit.

---

## A. Blocked on an external event

Nothing to do but be ready. Each names its trigger.

| Item | Trigger |
|---|---|
| **sk4 regen** of `unpopped-vocab` — renames, MX scale dtypes, `(acc+mp)` coordinate, `STRUCTURE_KEY_VERSION` 3→4 | KISS Phase-0 PR-1 landing. Execution basis is `docs/dtype-spelling-delta.md`. Then a four-way byte-match, then a coordinated semver-major. **Unpopped is on the critical path** — Baracuda re-points at `unpopped-vocab@sk4`, so publish and then name the exact version to them. |
| **Vendor the KISS dtype manifest** + assert token-set equality both ways in CI | KISS PR #129 (adds `structure_key_schema_version` / `token_prefix`). Do **not** vendor before it: a version-less vendored manifest is itself the clause-D violation — persisted, indexed dtype tokens detached from their schema version — and `c64` changes meaning at sk4, so a stale pin is silently wrong rather than loudly wrong. |
| **`unpopped-cuda` sub-crate**: CUDA emitter donated by Baracuda, plus the `convert.rs` CUDA parser moving out of neutral core | The 0.2 trait freeze. The crate has **two** inbound streams — IR→`.cu` (Baracuda's donation) and `.cu`→IR (our parser) — so it must not be designed emit-only. |
| **Make the target namespace pluggable**; move the `cuda:` vocabulary out of `unpopped-vocab` | Same carve. `ArchSku` stays Baracuda-owned content sourced from KISS's SSOT, generated-and-committed, never `build.rs`. |

**Publication is held** on `unpopped-vocab` generally: per sk4 §6 (Eric-ratified)
that clause binds the *token-deriving* crate, so its breaking changes ride the
coordinated cut. The generator (`unpopped`) is explicitly **not** gated and ships
on its own schedule.

---

## B. Gated on a coordinated golden regen

These rewrite emitted text. Doing any of them quietly breaks byte-identity
goldens *including Baracuda's physical CUDA corpus*, so they ride a regen event
rather than a cleanup commit.

- **The f16/bf16 spelling seam.** `cfamily::scalar_ctype` spells `__half` /
  `__nv_bfloat16` and `cast_scalar` emits `__half2float`-class intrinsics from
  the *neutral* module. Tripwired in `tests/neutral_spelling.rs`, which fails in
  both directions.
  **Trap for the implementer, found by running the seam as a mutation:** making
  `scalar_ctype`/`half_load_intrinsic` return `None` does **not** make
  `cast_scalar` decline — it drops into the arithmetic arm and emits `(float)v`
  on a `__half`, trading a visible vendor leak for a silent numerical bug. The
  neutral core must *refuse*, never fall through.
- **The temp-binding pass.** `Sqr`/`Relu` reference their operand twice,
  `Gelu`/`Silu`/`Sign` three times, `Max`/`Min` four.
  **It is only safe because of the op set, not because hoisting is safe.** All of
  those are float-only (the plan gate rejects every `UnaryOp`, the float binary
  fns and `Cmp*` at integer dtypes), and a float temp round-trips exactly. At
  sub-`int` widths hoisting *changes results* — C promotes `char`/`short` to
  `int`, so an inlined compound operand is un-truncated while a hoisted one is
  truncated by its temp: `(in0+in1)>>in2` at u8 is `150` inlined, `22` hoisted.
  Anyone extending these ops to 8/16-bit must settle truncation **first**.
- **`const_lit` dtype-correctness**, welded to threading dtype into the
  optimizer. These must land together: `exact_pow2_recip` reasons in f64
  normality and never sees the kernel dtype, so the `x / 2^k → x * 2^-k` rule is
  sound *only because* `const_lit` emits a bare double and C promotes. Measured:
  of 2045 constants the f64 predicate accepts, **44 are wrong under f32**; an
  f32-aware guard admits 253 with 0 wrong. Pinned by
  `optimize::pow2_rule_soundness_is_coupled_to_double_promotion`.

---

## C. Needs a ruling — not ours to decide alone

- **`required_fidelity(plan)`** — nothing derives which fidelity a cell is
  *entitled* to; `Fidelity` is picked by whoever calls `compare`, so the standard
  can only say "passed at a tolerance someone chose". Note for whoever takes it:
  **existing usage cannot seed the table** — there are six construction sites and
  five are the oracle testing its own comparator, so it must be derived from the
  numerics. Plausibly KISS-Conform's rather than ours. (`conformance.md` OPEN-1.)
- **What `VariantFidelity::BitIdentical` means across backends** — bit-identical
  to *what*? Local-to-this-backend is real but says nothing cross-backend; to a
  normative reference makes accumulation order normative, contradicting the
  incidental list. (OPEN-2.)
- **Whether NaN propagation (rule N2) is universally required** — on Vulkan it
  may be unachievable *by any lowering*:
  `shaderSignedZeroInfNanPreserveFloat32` is a `returnedonly` **property**, not a
  feature you enable. Either gate on the capability and declare non-advertising
  devices out of conformance, or weaken the rule. (OPEN-4.)
- **The KISS cost model** (#125) — vector-authoritative vs optional sibling. Our
  position is recorded: we already emit a two-axis vector with provenance, and a
  generator can only ever say `declared`.

---

## D. Ordinary engineering — unblocked, just not done

- **The catalog / server mode** — designed in [`catalog.md`](catalog.md), not
  built. Phase 1 is the library (op registry, emitter registry, `resolve`);
  phase 2 is a server over it, deliberately sequenced after the PROVISIONAL wire
  formats are co-pinned. The design names one gap that must be closed *in* the
  build rather than after it: a catalog entry is baked against caller-supplied op
  logic, which no current validity field names (§7).

- **Dtype lowering coverage.** 22/22 named, 12 with a scalar type, CpuC lowers 9,
  Slang 5. Roughly by cost: `Bool` (u8 storage, cheap) · `u64` (needs an
  unsigned-wrap audit — `wrap_bits` is two's-complement *signed* and f64 cannot
  represent every u64, which is why it is named but deliberately not lowered) ·
  `S4`/`U4`/`Bin` (sub-byte pack/unpack) · both FP8s (software codec — and the
  oracle needs an *independent* one or it stops being a differential) ·
  `Complex32`/`64` (struct ABI + complex arithmetic in the IR).
  The reserved `fnuz` pair must **never** be lowered at this schema version.
- **Oracle coverage**: `RowSort` (NaN-greatest `key_lt`, stable ties, TopK), and
  gather/scatter — the latter needs a *different notion of correct*, since under
  nondeterministic FP `atomicAdd` only an order-independent invariant is
  checkable at all. Pinned by the exhaustive `Coverage` classifier in
  `oracle.rs`'s tests.
- **Quant: adopt the scale-sibling-operand model.** `QuantFacts` is the wrong
  *shape*, not the right shape un-keyed — removing it is breaking, so it rides
  the sk4 cut. Residual it does **not** close: blk32 vs blk128 both contribute
  one same-rank scale sibling and still collide. That part is a genuine sk5 item.
- **Constructors for `ContractionKey`/`OperandKey`/`QuantFacts`/`SymExtent`,
  then `#[non_exhaustive]`.** Ordering matters: marking them reserved *first*
  breaks cross-crate struct literals with no constructor to fall back on — that
  is why the earlier attempt was backed out and only `StructureKey`/`OperandDesc`
  kept it.
- **Per-target N2 verification.** NaN-propagation surviving the toolchain is a
  property of *one optimizer*, re-measured per target, never inherited. Verified
  on portable C (`/O2`) and CUDA (nvrtc→PTX→driver JIT, RTX 4070). Any new target
  needs its own; a source-text golden is structurally blind to it.

---

## Not deferred, just worth knowing

The PR-gating box **has GPU hardware**. This crate's tests are CPU-only because
it holds no device backend, not because the hardware is absent — so the device
legs become runnable in-repo the moment `unpopped-cuda` lands.
