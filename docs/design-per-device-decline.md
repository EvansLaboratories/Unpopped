# Design note: the per-device decline mechanism (no code)

Status: **accepted 2026-10-22** with the PM's rulings (section 8). Written against Unpopped
`origin/main` `ed8ccbd` (0.20.19) and unpopped-harness `origin/main` `4a48c4d` (0.0.7; f64 screens in
#8). Roadmap row 3's tail. Nothing here is implemented; section 7 is the order.

## 1. What exists, and the gap

Exists (0.20.19 + harness 0.0.5): the harness runs a kernel on a device and emits an
`ExactnessRecord { stamp, key, verdict }`. `ExactnessLedger::lookup` answers for exactly one stamp
(device, API version, vendor+driver, `slangc` args), body and dtype; `Flushes` dominates; no record
says nothing. `contract_with_exactness`, `precision_of_measured` and `required_fidelity_with_exactness`
read it, so a `Flushes` device stops claiming `correctly_rounded` and an unmeasured device never
claims it. Tested in `a_flushing_device_declines_exact.rs`.

The gap, as four separate things (they have been discussed as one):

1. **Persistence.** A ledger is an in-memory value the caller builds. Nothing writes a record to
   disk or reads one back, so a measurement dies with the process that took it.
2. **Plan level.** `build_plan(op, key)` is keyed by `TargetId` (a device *class*, e.g.
   `vulkan:sg64.ops-abr...`). A record is per device *instance* and compiler flags. The decline
   today is a decline of a **claim** (precision block, fidelity band); the kernel is still emitted
   and still runs. Nobody can ask "this device flushes; give me the other variant".
3. **Variants.** The measured fix for the flushing devices was a compiler argument
   (`-denorm-mode-fp32 preserve`). The stamp already keeps args in its identity, so the same body
   has a `Flushes` record for the default args and an `Exact` record for the preserve args. Nothing
   chooses between them.
4. **Staleness.** The record's validity is narrower than its key (section 4).

## 2. Decision: what "decline" means

Three distinct outcomes, kept distinct in the type that comes back:

| Outcome | Meaning | Who acts |
|---|---|---|
| **Withhold the claim** | Kernel stays; the contract does not say Exact. Exists today. | Unpopped (pure function of ledger + stamp) |
| **Refuse the plan for a requirement** | The caller required Exact (or a ULP band); this stamp cannot meet it; return a typed decline carrying the verdict, not a string. | Unpopped gate, driven by the caller's stated requirement |
| **Offer a variant** | A different `compiler_args` set has an `Exact` record for this device; say which. | Unpopped names it; the adopter's runtime applies it |

Unpopped never launches a kernel or probes a device (module docs of `exactness.rs`), so it can
*decide and name*, never *apply*. Recommended: add the second and third as a typed enum
(`Admission { Admitted, ClaimWithheld{verdict|Unmeasured}, Declined(..), UseArgs(Vec<String>) }`) returned by one
function that takes `(op, key, stamp, ledger, requirement)`. `Unmeasured` and `Flushes` stay separate
variants: collapsing them is the safe-failing trap (a conservative decline has no reporter, so no
adopter notices their device was never measured).

## 3. Owner and where the record lives

| Piece | Owner | Why |
|---|---|---|
| Producing records (running kernels, verdict, coverage) | **unpopped-harness** | The only producer; Unpopped must not probe (standing design). |
| Record schema, `body_key`, lookup/dominance rules, `Admission` | **Unpopped core** | Pure data and logic, no I/O; the rules already live there. |
| Reading/writing the file | **Unpopped core** (serde, text only, no paths) | A schema owned in one place; the harness and adopters both link it. Same-author rule: no mirrored copy in the harness. |
| Where the file sits on a machine; when to measure; applying `UseArgs` | **The adopter runtime** (Fuel, Baracuda/Vulkane integration) | It is the only party that knows the real device at run time. |
| Reference ledger for our two dev devices (Radeon 610M, RTX 4070 Laptop) | **unpopped-harness repo**, as evidence | Shows the format and the 2026-10-21 finding; not a source of truth for other machines. |

Format: JSON Lines, one record per line, `schema` field first, unknown fields preserved, an unknown
`schema` or malformed line is a typed decline of that line (never a panic, never silently dropped
without a count). Fields beyond today's `ExactnessRecord`: see section 4. No crates.io-external
location is assumed (portfolio rule: published versions, no machine-relative paths).

A record is **not** committed to a consumer's repo as a claim about hardware it was not measured on.
A ledger travels with a machine (cache dir), or ships for named devices with the stamp intact; a
stamp mismatch means no record, which means no Exact.

## 4. The record must name what it was baked against

An `Exact` record is a statement about a **kernel as compiled**, but the key is body + stamp only.
Three things can change the compiled kernel without changing either:

- **The emitter**: a new `unpopped-slang` release can emit different source for the same `OpDef`.
- **The compiler**: the stamp carries `slangc` *arguments*, not the `slangc` *version*. Confirmed in
  harness `vulkan::device_stamp` (device, API version, vendor+driver, args). A `slangc` upgrade
  changes the SPIR-V and so possibly the denorm behaviour.
- **The coverage**: `Exact` from an exhaustive unary-f32 run and `Flushes` from a 256-pair sample
  are both one `ExactnessVerdict`; the record does not say which.

Recommended additions (additive; `ExactnessRecord` is not `#[non_exhaustive]` today, so this is a
0.21 decision for the PM, or a sibling `ExactnessRecordV2` first):

- `kernel_hash`: hash of the emitted source/SPIR-V the run actually executed;
- `compiler_version` in the stamp identity (harness fills it from `slangc -v`);
- `coverage`: `{ kind: exhaustive | sampled, tuples, seed }` (the f64/binary/ternary reports already
  carry these);
- `emitter_version`.

Lookup applies a record only when `kernel_hash` equals the hash of the kernel being admitted. That
makes "I re-emitted and the old Exact still applies" impossible by construction, and makes the
`UseArgs` offer checkable: the variant's record has its own hash.

## 5. Reporter for the safe failure

Declining on "no record" is correct and invisible. Add a counter the adopter can read, per
(device stamp, reason in { Unmeasured, Flushes, StaleHash, CoverageSampled }): the existing
telemetry module already counts drops per capability. Without it a machine that was never measured
looks identical to one that was measured and failed.

## 6. What this does not do

- No extrapolation: a unary-f32 record says nothing about binary, f16 or f64 (unchanged).
- No gating on Vulkan property bits (NVIDIA reports `shaderDenormPreserveFloat32 = false` and
  honours the mode).
- No CUDA/slang `Max(x,0) == Relu` claim: still unmeasured; needs the runner.
- No auto-measure inside Unpopped.

## 7. Suggested order (each a separate PR, none started)

1. Harness: record `compiler_version`, `kernel_hash`, coverage in `device_stamp`/records; write
   JSONL. (Harness-only; no Unpopped change needed to start.)
2. Unpopped: serde for ledger + stamp + record (additive), `Admission` + the one function, tests
   with known-bad controls (stale hash, other compiler version, sampled-vs-exhaustive, malformed
   line counted).
3. Adopter integration: cache location, measure-on-first-run policy, applying `UseArgs`.

## 8. Rulings (PM, 2026-10-22)

1. **One record type, new fields.** `kernel_hash`, `compiler_version` (in the stamp identity),
   `coverage` and `emitter_version` are added to `ExactnessRecord` as a **0.21 Breaking minor**
   (pre-1.0: the second number is the major). No V2 sibling.
2. **`Declined` is a typed value.** The adopter chooses what to do with it; Unpopped does not turn
   it into an error.
3. **The adopter owner is Fuel** (it places kernels and sees the device): the ledger location,
   measure-on-first-run policy and applying `UseArgs` are Fuel's.
