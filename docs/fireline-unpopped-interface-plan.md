# FireLine plan: Unpopped's IR and lowering contract to the emitters

PM task 2026-11-07 (board 295; CireSnave: "create a plan for each"). Plan only, no code moves. Measured at Unpopped `origin/main` (0.44.20); emitter
imports counted by `git grep` at each repo's `origin/main` on 2026-11-07 (`unpopped::<module>` paths, a census of `use` lines, not of calls). Not yet
reviewed by vulkane, fishstringer or baracuda: section 7 lists what each must confirm.

## 1. The interface today

Two layers, both inside the one crate `unpopped`. **Intent layer:** `ir` (`ScalarExpr`, ops, `Access`, `View`), `plan` (`KernelPlan`, `Schedule`),
`backend` (`Backend::lower(&KernelPlan) -> Result<GeneratedKernel, LowerError>`, `Decline`, `Spelling`, `DtypeSupport`, `lower_expr` and
`lower_dag*` helpers), `cfamily`. **Kernel layer:** `kernel` (`KernelFn`, `Stmt`, `Module`, `ExternDecl`), `exec` (`LaunchShape`, `ExecId`,
`MemorySpace`), `mma` (`MmaAtom`), `kernel_check`, and the oracle `run_launch`. Plus `capset`/`capability` (`TargetCapabilities`, `CapabilitySet`)
and `unpopped-vocab` (element kinds, `TargetId`, ops, layout, `DeviceRepr`).

Measured dependency facts that decide the cut: `ir.rs` and `exec.rs` import nothing else in the crate; `kernel.rs` imports only `exec`, `ir`, `mma`,
`types`; `backend.rs` imports `ir` (and names `plan::KernelPlan` in `lower`); `capset.rs` imports `artifact` and `backend`. So the data types are
already separable; the planner is not what they depend on.

Who uses what (import census): `baracuda-cuda-emit` 92 `unpopped::ir` plus `build_plan` 28, `derive_pattern` 21, `plan` 27, `Backend` 23;
`vulkane-slang-emit` / `-spirv-emit` / `-hlsl-emit` use `ir`, `backend`, `try_generate`/`generate`, `plan::KernelPlan`, `cfamily`; hlsl-emit also
enables Unpopped's `convert` feature. **Nobody outside Unpopped consumes the kernel layer yet** (E1 is design-only; no emitter lowers a `KernelFn`).
Fishstringer: no `unpopped` import found at `origin/main`; its root is a workspace manifest only, so I cannot say what it will need (7).

## 2. What moves into FireLine, one crate per interface

| FireLine crate | Takes from Unpopped | Why it is the interface |
|---|---|---|
| `fireline-ir` | `ir`, `types`, `exec`, `mma`, the data types of `kernel` (`Stmt`, `Module`, `ExternDecl`, `Effects`), `numerics` | The wire types every parser and emitter reads or writes; `ir` has no crate deps. Serde text form (`text.rs`) comes with it |
| `fireline-lowering` | `backend` (`Backend`, `GeneratedKernel`, `Decline`, `Spelling`, `LowerError`, `DtypeSupport`, `lower_*`), `cfamily`, `KernelPlan`/`Schedule` types from `plan` | The one contract emitters implement; the plan data types, not `build_plan` |
| `fireline-caps` | `capset`, `capability`, `TargetId` | Capability bits both sides read; decoupled from `artifact` (see 4) |
| `fireline-vocab` | `unpopped-vocab` as is (already its own crate) | Shared element/op vocabulary; KISS and Baracuda already consume it |
| `fireline-conformance` | corpus plus runner trait, replacing `unpopped-conformance` (today `publish = false`) | The test that fails on drift (6) |

## 3. What stays in Unpopped

The planner (`build_plan`, `derive_pattern`, `optimize*`, `jit`, `recipe`, `resource_map`), the lifts (`lift`, `convert`, `kernel_lift/*`, `kisc`,
`kiss_ref_diff`), the oracle (`oracle/*`, `run_launch`), `check_kernel`/`kernel_check`, `link`, `dispatch_artifact`, `telemetry`, `unpopped-cpu-c`.
Anything that names planner internals or a backend stays. The oracle stays: it is Unpopped's own meaning of the IR; FireLine freezes its outputs as data (6).

**Open: `build_plan` and `derive_pattern`.** Baracuda calls them ~49 times. They are planner, not interface, yet its emitter cannot lower without
a plan. Options: (a) Baracuda keeps depending on `unpopped` for them (recommended; thin dependency, version-pinned) or (b) `KernelPlan` construction
becomes an interface crate. I recommend (a) until a second emitter wants the planner.

## 4. Thin adapters and what breaks on the first release

- **Unpopped:** depends on `fireline-ir/-lowering/-caps/-vocab`; re-exports them under the old paths (`pub use fireline_ir::ir`, etc.) so downstream
  source compiles unchanged. `capset` loses its `artifact` import: `CapSource` carries a plain id instead of an artifact type (**breaking**, one field type).
  `GeneratedKernel` stays `#[non_exhaustive]`.
- **Emitters (vulkane x3, baracuda, fishstringer):** no source change if the re-export holds; they move `unpopped = "0.43.2"` to `fireline-lowering = "~0.1"`
  plus `unpopped = "~..."` only where they call the planner. Cargo.toml churn only.
- **Breaks:** the **type identity** of every moved type changes crate (two crates naming `ScalarExpr` must agree: all must move in one release window, or
  a `use` through the old path resolves to a different type and fails to compile, which is the loud failure we want). Feature `convert` stays on `unpopped`.
- **Versioning (per the agreed rules):** each crate's version follows its interface; pre-1.0 a breaking change moves that crate's second number alone;
  consumers require `~`/exact. Today's one-version-per-project rule (Unpopped 0.44.20 across all crates) is replaced for the moved crates; **CireSnave
  said the common number applies at major releases only, so `fireline-*` start at their own 0.1.0 and Unpopped keeps its number.** A decision to confirm.

## 5. Consumers, KISS, and the IR union

- **Consumers:** Baracuda emit, Vulkane emits (Slang, SPIR-V, HLSL), Fishstringer, Fuel (Slang parser/emitter, `SLANG-SEAM-REQUIREMENTS.md`), KISS
  (`kiss-ref-diff` and the verdict column in `docs/union-coverage-master.md`), and any future parser crate.
- **KISS:** KISS states what is required; FireLine holds the Rust types that say it. `fireline-vocab` is the crate KISS-Emit and KISS-Consume already
  point at through `unpopped-vocab`; the rename needs KISS's agreement on citations. KISS's own crates stay KISS's.
- **Growth by the master list (`docs/ir-union-master-capabilities.md`, #314):** a row that needs a new IR construct adds a **variant or field** to
  `fireline-ir`. Rules: every public enum and struct is `#[non_exhaustive]`; new variants default-decline in `Backend` (`Decline`, never a panic); a new
  `CapKey` is added with the construct, so an emitter reads "not offered" rather than failing. Additive change = patch release of the crate; removing or
  re-meaning anything = second-number bump. Today `Stmt` and `ScalarExpr` are **not** uniformly `#[non_exhaustive]` (I have not counted which are);
  making them so is itself breaking and belongs in the first release.

## 6. The test that fails on drift

1. **Frozen corpus:** `fireline-conformance` holds kernels (IR text) with inputs and the oracle's expected bytes, checked in. Unpopped CI regenerates
   from `run_launch` and fails if any byte differs without a corpus update (oracle drift). Each emitter runs the corpus through its own backend
   (`cpu-c` compiles C; Vulkane and Baracuda run on self-hosted GPU runners, as CireSnave agreed on 2026-10-16) and compares at the tolerance in the
   row's numerics (emitter drift).
2. **Wire pin:** a snapshot test of the serde text of every `fireline-ir` type fails on any shape change not paired with a version bump.
3. **Decline pin:** every corpus kernel an emitter does not handle must return a typed `Decline`; an unlisted decline fails.
Seed three mutants when built (drop a variant's handling in one emitter; change one oracle rounding rule; reorder a struct field) and show each caught.

## 7. Needs from others, and the unknowns

- **Baracuda:** confirm the `build_plan`/`derive_pattern` choice (3) and that it can take `~` pins. **Vulkane:** confirm HLSL's `convert` use is the only
  non-`ir`/`backend` dependency. **Fishstringer:** say what it will import (I found nothing to measure). **KISS:** rename and citation impact (5).
- **PM / CireSnave:** project name is FireLine (not SeamAnt); crate names above are placeholders; the version-start decision in 4.
- Unknown to me: how many public types are `#[non_exhaustive]` (5); whether `artifact` can be cut from `capset` without touching Baracuda's cache key.

## 8. The Unpopped-to-Fuel boundary (CireSnave 2026-11-07: Fuel talks to Unpopped via FireLine, no direct wiring)

**Measured today** at Fuel `origin/main` d010189e4 (`git grep` over `*.rs` and every `Cargo.toml`; a census of use sites, not of behaviour):
- **Crates:** Fuel depends on `unpopped-vocab` only, pinned `0.43.2` in the workspace manifest. **It has no dependency on `unpopped`** (the core), `unpopped-cpu-c`, or any kernel-layer or `Backend` type; the only `unpopped::` text in Fuel sources is prose (`gaps.md`, `ROADMAP.md`, a doc comment).
- **Who:** `fuel-dispatch` (optional, behind feature `jit`, which also pulls `fuel-kernel-seam` and `kiss-ops-vocab`), `fuel-kernel-seam`, and `fuel-kernel-seam-types` (it re-states the vocabulary types so a synthesizer backend does not depend on Fuel's graph).
- **Symbols, by use-site count:** `ElementKind` (about 22), `OperandDesc` (about 21), `ArchSku` (`Sm80`, `Sm89`, `Sm90`, `Sm90a`, mapped by hand in `fuel-dispatch/src/telemetry/baracuda_provider.rs` `map_arch_sku`, GAP-179), `TargetId` (1), `structure_key` (1). Files with `unpopped_vocab` uses: `jit_ingest.rs` 17, `jit_adopt.rs` 4, `jit_carrier.rs` 2, `jit_ingest_probe.rs` 2, `jit_synth_kernel_live.rs` 1, `fuel-kernel-seam/src/lib.rs` 2.
- **Direction:** Fuel calls nothing in Unpopped at run time; it uses vocabulary types as the key and operand description for a JIT request that goes to a synthesizer backend (Baracuda). Unpopped never calls Fuel.
- **Not read:** anything in Fuel under other names (`baracuda-kernels-types` re-exports the vocabulary; I did not walk that crate), and Fuel's unmerged branches.

**What crosses the boundary, as FireLine crates:** `fireline-vocab` (the six symbols above, the whole of today's link) and, for the JIT request, `fireline-caps` if the arch/target keys move there. **What does not cross:** `fireline-ir`, `fireline-lowering` and the planner. Fuel is a consumer of identity and capability vocabulary, not of the IR. Fuel's planned Slang parser and emitter (`SLANG-SEAM-REQUIREMENTS.md`, ROADMAP row 101, not started) would be the first Fuel code on `fireline-ir`/`-lowering`, as a parser/emitter provider, not as a consumer.
**For the first release:** Fuel's change is `unpopped-vocab = "0.43.2"` to `fireline-vocab = "~x"`; `ArchSku` and `ElementKind` paths are unchanged if Unpopped re-exports; the hand-written `ArchSku` mapping is the drift risk the conformance suite should pin (a test that every `ArchSku` variant round-trips through Fuel's string form). Fuel ROADMAP row 41 ("adopt unpopped 0.22.0 `DtypeSupport`") names a symbol Fuel does not import today.
