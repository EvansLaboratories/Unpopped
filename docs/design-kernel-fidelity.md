<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
# Design: measured exactness for kernel-layer kernels (ROADMAP U59)

Status: **design only, docs only** (2026-11-26), **rewritten the same day: the first version proposed a new `KernelFidelity` type on `GeneratedKernel` and was wrong about what exists** (section 1). Asked by pm-kernels from a gap vulkane found reading Unpopped 0.57.0. Nothing here is built. This is a row, not a skip (CIRESNAVE-EXPECTATIONS 6.21: a gap is a row).

## 1. What already exists (the first version of this note missed it)

`unpopped::exactness` (`crates/unpopped/src/exactness.rs`, design `docs/design-per-device-decline.md`, tests `a_flushing_device_declines_exact.rs`, `exactness_admission.rs`) is the mechanism this note first reinvented:

* An **`ExactnessRecord`** is a *measurement*: a `DeviceStamp` (target, device, runtime, driver, compiler version, compiler arguments), a **key** naming what was measured, a `kernel_hash` of what actually ran, a verdict (`Exact` | `Flushes`) and a `Coverage` (`Exhaustive` | `Sampled`).
* An **`ExactnessLedger`** answers `find` / `lookup` / `admit` for one stamp, kernel hash, body and dtype. **`Flushes` dominates**; only an exhaustive `Exact` admits; a sampled `Exact` is evidence that admits nothing; a record for another stamp, hash, body or dtype is not consulted; no record means no claim. `admit` returns a typed `Admission` (`Admitted`, `ClaimWithheld(Withheld)`, `Declined(Withheld)`, `UseArgs(..)`) the adopter acts on. JSONL persistence and an `AdmissionTally` for the safe failures exist.
* Unpopped **never probes a device**; a harness that ran the kernel hands the record in (the `unpopped-harness` run of 2026-10-21 over all 2^32 f32 inputs is the motivating measurement). The Vulkan property bits are explicitly rejected as a gate: only the measurement is.

So "the declaration is the outcome of a measurement with provenance, not a capability table, and a stale record declines" is **already the rule**, and the KISS side's per-kernel `Guarantees.math_fidelity` (contract 6.8-0004; no dtype dimension, no value admitting flush, 6.17-0002) is read through it.

## 2. The real gap

The ledger is keyed by a **plan-layer body**: `ExactnessRecord::new(stamp, body: &ScalarExpr, dtype, ..)`, `find(.., body: &ScalarExpr, dtype)`, `admit(.., body: &ScalarExpr, ..)`, with `body_key` = `"{dtype}:{serde-JSON of the ScalarExpr}"`. A kernel-layer kernel (`Backend::lower_kernel(&KernelFn)` -> `GeneratedKernel`, e.g. the MSL emitter) has **no `ScalarExpr` body**, so a measurement of it **cannot be recorded or looked up**: vulkane's finding (Metal f32/bf16 flush, f16 preserves) has nowhere to be stored that a consumer can ask, and a consumer holding a `GeneratedKernel` from `lower_kernel` has no ledger entry point. `GeneratedKernel` itself needs no new field: the `kernel_hash` the ledger compares is the adopter's hash of what was built (`GeneratedKernel::revision_hash` covers source and words), which is how the op layer already works.

## 3. Proposal (additive)

1. **A kernel key.** `kernel_key(kernel: &KernelFn, dtype: ElementKind) -> String`, `"kernel:{dtype}:{serde-JSON of the KernelFn}"` (the `kernel:` prefix cannot collide with a `body_key`, which starts with a dtype token), pinned by a test exactly as `body_key` is (a silent change of the key drops every record).
2. **Record and look-up entry points for it**: `ExactnessRecord::for_kernel(stamp, &KernelFn, dtype, verdict, kernel_hash, emitter_version, coverage)` and `ExactnessLedger::{find_kernel, lookup_kernel, admit_kernel}` with the same logic and the same `Admission`, refactored over a key string so the two layers share one implementation (no behaviour change for the op layer: its tests stay as they are).
3. **The kernel-layer reading of coverage.** A kernel-layer kernel's input space is usually not enumerable, so its `Exact` will mostly be `Sampled` and admit nothing: the practically recordable and **useful verdict is `Flushes`**, which declines on one mismatch, sampled or not. That is the case that started this (a device flushes; the consumer must decline `Exact`). An exhaustive `Exact` stays possible for a probe-sized kernel (the shared probe recipes below).
4. **Probe recipes** (shared, in IR so the oracle is the reference): small kernels with subnormal, infinity and NaN operands through add, mul and fma at each dtype. The emitter lane's harness runs them on its device, compares bit for bit with the oracle and hands a record in; the emitter (or its adopter) keeps the ledger. The recipes are the only new content that is not plumbing.
5. **Who sets what.** The harness and the emitter lane produce records (vulkane for Metal/Vulkan/GLSL, Baracuda for CUDA, bonefish for SYCL); Unpopped owns the key, the entry points and the recipes, and never probes.

**Dropped from the first version:** a `KernelFidelity` / `Exactness` / `Reason` type carried on `GeneratedKernel`. It would duplicate the ledger's verdicts and provenance, and a field on the artifact would be an unrecorded claim of the kind the ledger exists to refuse (a record names the stamp and the hash; a field would not). The consumer reads the ledger.

## 4. KISS and the wire

KISS answered (2026-11-26, `kiss`, `origin/main@153c09a5`): `Guarantees.math_fidelity` (`bit-stable` | `reduced-mantissa-permitted:<n>`) is per kernel with no dtype dimension, and **no value admits flush** (6.17-0002), so a flushing kernel has no admissible value and must decline `Exact`. **Ruling (pm-kernels, 2026-11-26): A now: a local extension, additive; no request for a KISS value; if CireSnave later wants one, it is a wire change, filed then.** The local mechanism (the ledger) maps to the wire thus: `Admission::Admitted` <-> may carry `bit-stable`; `ClaimWithheld`/`Declined(Flushes)` <-> no admissible value; any other `Withheld` <-> no claim. A later KISS clause changes that mapping function, not the ledger. Not addressed by KISS (not read): an FMA SIMT cell serving an `st` key; route separately if wanted.

## 5. Rules and scope

* **IR / oracle:** no new statement or expression; the oracle is the reference and is `Exact` by definition. No change to the op-layer entry points or their tests.
* **Patent:** nothing describes a hardware mechanism or claims an implementation; it records what a measurement found. Nothing to add to `docs/PATENT-SWEEP.md`.
* **Provenance:** the Metal measurement is vulkane's, cited to vulkane; KISS-OPS-6.17-0002 and contract 6.8 are KISS's, quoted verbatim from the vendored corpus when the row is built (this note quotes them as `kiss` and pm-kernels reported them).
* **Not in scope:** run-to-run determinism (`VariantFidelity`, plan layer); accuracy bands for `approx` ops (U42, U37); per-operation rounding evaluation (U45).

## 6. Slice 1 (the next code)

`kernel_key`, `ExactnessRecord::for_kernel`, `ExactnessLedger::{find_kernel, lookup_kernel, admit_kernel}` over the shared implementation, with tests beside `a_flushing_device_declines_exact.rs` for a kernel-layer kernel: a flushing device is declined, a preserving exhaustive one admitted, unmeasured declined, and a record for another stamp, hash, kernel, dtype or an op body changes nothing (each with its control); a pin of the key text. The MSL `f32`/`bf16` `Flushes` records are vulkane's harness output, stamped on its device; slice 1 ends where that vulkane-side step begins and is reported as such.

— lane: unpopped
