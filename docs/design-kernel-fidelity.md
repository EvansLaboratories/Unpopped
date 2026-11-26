<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
# Design: a fidelity declaration on a lowered kernel (ROADMAP U59)

Status: **design only, docs only** (2026-11-26). Asked by pm-kernels from a gap vulkane found reading Unpopped 0.57.0. Nothing here is built. The KISS side was asked and has answered (section 5, question 1): recommendation A, Unpopped records the measured result in its own field as a local extension; a KISS value for it is a wire change that needs CireSnave's verbatim word through pm-kernels. This is a row, not a skip (CIRESNAVE-EXPECTATIONS 6.21: a gap is a row).

## 1. The gap

`Backend::lower_kernel` returns a `GeneratedKernel` (`crates/unpopped/src/backend.rs`): name, source, optional words, optional provenance and a `KernelInterface` (what a consumer needs to **launch** it). It carries nothing about what the lowered kernel **computes**: whether the artifact reproduces the oracle's result bit for bit on every input the IR defines, or deviates (flushes subnormals, contracts a multiply-add, uses a reduced-mantissa path, substitutes an approximate intrinsic).

What exists, and why none of it is that declaration:

| Existing | Layer | What it says | Why it does not cover the gap |
|---|---|---|---|
| `Numerics { approx, ftz, rounding }` / `Expr::Qualified` (`ir.rs`) | IR, per operation | a **request** the source makes (this operation may be approximate, flush, round this way) | a property of the program, not of what the target did with it; an emitter that cannot flush can still lower an `ftz` request wrongly, and nothing records the outcome |
| `contract.rs` `required_fidelity` / `AccuracyKey`, oracle `fidelity.rs` | plan layer | the accuracy band a plan needs | consumed at the plan gate; the kernel layer has no equivalent artifact field |
| `VariantFidelity` (`backend.rs`, returned by `lower_variants`) | plan layer, per **variant** | `BitIdentical` / `DeterministicallyDivergent` / `Nondeterministic` / `MorePrecise`: determinism and divergence **between variants of one op** | an axis about run-to-run and variant-to-variant bits, not "does this artifact flush subnormals"; and `lower_variants` is a plan-layer entry that a kernel-layer emitter (MSL) does not implement |
| KISS `MathFidelity` (KISS-OPS-6.17; `docs/accum-policies.md`) and the contract's `Guarantees.math_fidelity` (KISS contract 6.8-0001, 6.8-0004/-0004a: `bit-stable` or `reduced-mantissa-permitted:<n>`; KISS answer 2026-11-26) | a cell fact in the `structure_key` **and**, per kernel, a guarantee in its contract | what a request may be served with, and what a kernel's contract promises | KISS does have a per-kernel carrier, but it has **no dtype dimension** (the contract's dtype comes from its identity) and **no value for flushing**: 6.17-0002 says an FTZ kernel cannot honestly declare `bit-stable`, and `reduced-mantissa-permitted:n` does not admit flushing either, so such a kernel has **no admissible `math_fidelity`**: it must decline `Exact`, with nowhere to say why |
| ROADMAP "Known gaps": an FMA SIMT cell can never serve an `st` key (6.17-0002) | KISS / plan | told to the KISS lane; a caller wanting bit-exact F32 gets a decline | the same missing third state, one level up |

The measured instance (vulkane): on Metal the f32 and bf16 paths flush subnormals and f16 preserves them. An MSL emitter therefore must not let a kernel be read as `Exact` for f32, but today it has nowhere to say so and a consumer has nowhere to look.

## 2. Proposal: `KernelFidelity` on `GeneratedKernel`

A per-(kernel, dtype) declaration, **set by the emitter from a measured result**, read by consumers.

```rust
/// Per element type the kernel computes in: did the artifact reproduce the oracle exactly?
pub struct KernelFidelity { /* ordered (ElementKind, Exactness) pairs, no duplicates */ }

pub enum Exactness {
    /// Bit-identical to the oracle on every input the IR defines for this dtype, including subnormals,
    /// infinities and NaNs, *as measured* (see section 3).
    Exact,
    /// Not claimed exact; the reason is typed so a consumer can decide.
    Declined(Reason),
}

#[non_exhaustive]
pub enum Reason {
    FlushesSubnormals,        // measured: a subnormal operand or result reads as zero
    ReducedMantissa { bits: u8 }, // a tf32-like or fast path narrower than the dtype
    ContractsFma,             // a*b+c fused where the IR rounds twice (U45 per-op rounding)
    ApproximateIntrinsic,     // an `approx` Qualified op lowered to a hardware approximation
    Reassociates,             // a reduction whose order the target chooses
    NotMeasured,              // nothing has probed this (target, dtype): the honest default
}
```

Carried as `GeneratedKernel::with_fidelity(KernelFidelity)` / `fidelity() -> Option<&KernelFidelity>`, exactly as `with_interface`/`interface` (an additive, `#[non_exhaustive]` struct: no break). `None` means the backend states nothing, which a consumer must read as **not claimed exact**, never as exact. It is **not** part of `revision_hash` for the same reason the interface is not (the emitter-set fields follow from the source the emitter wrote) *unless* the probe result changes the generated source, in which case the source hash already moves.

Consumer rule (Fuel/Baracuda dispatch): a request keyed `st` (KISS `math_fidelity = bit-stable`, the `Exact` demand) is served only by an artifact whose declaration for the request's dtype is `Exact`; `Declined(reason)` or `None` is a typed decline naming the reason (the same "decline with the fix named" convention as `LowerError`). A request that permits reduced mantissa may accept `Declined(_)` it understands.

## 3. Who sets it: a measured result, not a capability

The rule that matters: **the declaration is the outcome of a probe on the target, never an assertion from a feature table.** A capability ("Metal supports f16") says nothing about subnormal behaviour.

1. **Probe.** A small conformance kernel per (target, dtype) written in the IR (so the oracle supplies the expectation): feed subnormal, near-subnormal, infinity and NaN operands through add, mul, fma, and compare the device result bit for bit with the oracle. It is an ordinary conformance test of the kind `crates/unpopped/tests/*_emit_agrees.rs` already are, with the oracle as the independent reference.
2. **Record.** The emitter crate keeps a table `(target id, driver/OS, dtype) -> Exactness` **with provenance**: the probe kernel's revision hash, the device, the date. A declaration without provenance is `NotMeasured`.
3. **Stamp.** `lower_kernel` copies the row for the kernel's dtypes into `with_fidelity`. A kernel with a dtype that has no row stamps `Declined(NotMeasured)` for it.
4. **Stale.** A table row is invalid when the probe kernel, the target's compiler/driver or the emitter's lowering of the probed operations changes; the emitter's own test that re-runs the probe is what keeps it honest (so a driver update that stops flushing is noticed as a *probe disagreement*, not silently).

Who: the **emitter lane** owns its table and probe (vulkane for Metal/Vulkan/GLSL, Baracuda for CUDA, bonefish for SYCL, cpu-c for the serial C, which is `Exact` for f32/f64 by construction only if the C compiler is invoked without `-ffast-math`/FMA contraction, itself a measured fact). Unpopped owns the type, the probe-kernel *recipe* (a shared set of IR probe kernels so the lanes do not each invent one) and the consumer rule.

## 4. Scope and rules

* **IR rules.** No new statement or expression; no change to the oracle (it is the reference, `Exact` by definition). The declaration is data beside the artifact, like `KernelInterface`.
* **Patent.** Nothing here describes a hardware mechanism or claims an implementation; it records what a measurement found. Nothing to add to `docs/PATENT-SWEEP.md`.
* **Provenance.** The Metal measurements (f32/bf16 flush, f16 preserve) are vulkane's measurements, cited to vulkane; the KISS-OPS-6.17-0002 text is KISS's, to be quoted verbatim from the vendored corpus when the row is built (this note quotes it only as pm-kernels reported it).
* **Not in scope.** Determinism across runs (that is `VariantFidelity`'s axis, plan layer); accuracy bands for `approx` ops (U42, U37); per-op rounding evaluation (U45). The declaration may name `ContractsFma` / `ApproximateIntrinsic` as reasons but does not replace those rows.

## 5. Open questions (answers folded in before code)

1. **KISS: answered 2026-11-26 (`kiss`, `origin/main@153c09a5`, spec read of contract 6.8 and ops 6.17).** KISS already has a per-kernel carrier, `Guarantees.math_fidelity` (`bit-stable` | `reduced-mantissa-permitted:<n>`), set by the kernel's contract, with no dtype dimension. The real gap is that **no admissible value advertises flushing**: closing it (a third value, or an artifact-level Exact/Declined+reason field) is a wire change to the Contract/Ops text that needs CireSnave's verbatim word via pm-kernels; KISS will not draft clause text and will write 6.17 once if it is wanted so emitters spell it alike. **KISS recommends A now:** keep `MathFidelity` as it is; Unpopped records the measured Exact/Declined+reason in its own field, spelled as a **local extension**, not as a KISS value. Mapping to the wire value until then: a kernel whose declaration is `Exact` for every dtype it computes in may carry `bit-stable`; a kernel with any `Declined(_)` carries **no `bit-stable`** (and, for `FlushesSubnormals`, no admissible value at all), so it declines `Exact` in the request. Not addressed by KISS (not read): the FMA SIMT cell versus an `st` key; route separately if wanted. **Open for the PM:** whether to ask for option B (a KISS value); an `[ASK]` to pm-kernels with this note if so.
2. **Granularity**: per (kernel, dtype) as above, or per (kernel, operation class)? A kernel mixing an f32 add (flushes) and an f16 fma (preserves) is one declaration per dtype in this design; a per-op class would be finer than the probe set can honestly measure. Recommended: per dtype.
3. **Where the probe recipes live**: a `unpopped::conformance` module (shared by the emitter lanes) or each lane's tests. Recommended: shared, because the oracle is the reference and the probes are IR.
4. **Breaking?** Additive (`#[non_exhaustive]` struct, new accessor); an emitter that does not stamp yields `None`. Not breaking.

— lane: unpopped
