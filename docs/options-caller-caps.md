# Options: `CallerCaps`, so a backend can refuse a plan a legacy caller would bind blindly

Status: **implemented in #386** (`CallerCaps`, `try_generate_with`, `Backend::lower_with`; `crates/unpopped/src/backend.rs`, `lib.rs`, tests `tests/caller_caps.rs` and `unpopped-cpu-c/tests/caller_caps_unchanged.rs`); it began as an options note (pm-kernels `[TASK]` 2026-11-22; code after #385 merges, never on its stack). Asked for by fishstringer's E17 slice 3
(`docs/options-e17-slice3-abi.md` in its repo, branch `cefa057`): a plan kernel that needs an `n` guard, a params block or several outputs
carries a `KernelInterface` that a consumer written before it existed would ignore and bind wrongly.

## The ruling this implements

pm-kernels, Q2: **YES, additive**, a capability argument the caller opts into. `CallerCaps::LEGACY` is the default and means *cannot bind a params block
or read an interface*. `generate()` and `try_generate()` are unchanged.

## Shape

```rust
/// What the caller of a lowering can do with the result. A backend that would emit something needing more refuses instead.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct CallerCaps {
    /// The caller reads `GeneratedKernel::interface` and honours it: the `bindings` slots and the `params_block` layout.
    pub interface: bool,
}
impl CallerCaps {
    /// Binds buffers by the kernel's declaration order and passes no params block: every caller before this existed.
    pub const LEGACY: Self = Self { interface: false };
    /// This, with `interface` on.
    pub const fn with_interface(self) -> Self { Self { interface: true, ..self } }
}
impl Default for CallerCaps { fn default() -> Self { Self::LEGACY } }

// lib.rs, beside try_generate:
pub fn try_generate_with(op: &OpDef, key: &StructureKey, backend: &dyn Backend, caps: &CallerCaps)
    -> Result<GeneratedKernel, LowerError>;           // try_generate(op, key, b) == try_generate_with(op, key, b, &CallerCaps::LEGACY)

// backend.rs, a NEW defaulted method on Backend (an existing signature is not touched):
fn lower_with(&self, plan: &KernelPlan<'_>, caps: &CallerCaps) -> Result<GeneratedKernel, LowerError> { self.lower(plan) }
```

- **One capability, not three.** The interface carries the bindings and the params block together, and a consumer that reads one reads both, so a
  separate `params_block` / `multi_output` flag would only let a caller claim half a contract. A backend decides which plans *need* the interface.
- **Where the default lives.** `CallerCaps::LEGACY` and `Default` in `unpopped` (`backend.rs`); a backend never sees "no caps given", only `LEGACY`.
- **How a backend learns it.** `try_generate_with` calls `backend.lower_with(&plan, caps)`. The default `lower_with` ignores `caps` and calls `lower`, so
  **every existing backend compiles and behaves unchanged** (Baracuda's CUDA, Vulkane, rockfish, openclownfish, CpuC). A backend that emits an
  interface-needing shape overrides `lower_with`, and its `lower` becomes `lower_with(plan, &CallerCaps::LEGACY)`.
- **The refusal reuses an existing variant**, `LowerError::UnsupportedPlanShape { detail }`, with a detail that names the capability ("needs a caller that binds
  `GeneratedKernel::interface`; call `try_generate_with` with `CallerCaps::LEGACY.with_interface()`"). `LowerError` is `#[non_exhaustive]` and is **not**
  extended, and no exhaustive enum is touched. No existing signature changes; the additions are a type, a function, and a defaulted trait method.
- **`generate_variants`** is untouched in this slice. It calls `backend.lower(&plan)` (`lib.rs`), not `lower_with`, so it is covered exactly when the backend follows the convention above (its `lower` is `lower_with(plan, &CallerCaps::LEGACY)`): then `generate_variants` panics with the refusal, as the test stub `NeedsInterface` shows (probed 2026-11-23). A backend that overrides `lower_with` to refuse LEGACY callers but leaves `lower` serving the old shape is still served through `generate_variants` (probed with a stub that breaks the convention: `try_generate` refuses, `generate_variants` serves). The convention is required, not enforced. **Fixed 2026-11-23 (the core PR of the U19 walls work):** `generate_variants` now lowers the base kernel through `lower_with(&plan, &CallerCaps::LEGACY)`, so a convention-breaking backend is refused there too (`tests/caller_caps.rs`, with a lower-only control that is unchanged). A `generate_variants_with` is the same pattern if a consumer needs it; none does today.

## Who calls `generate` today, and how each would opt in

Read with `git grep` at each repo's current checkout, 2026-11-22; test modules named separately. **Not run, and adopters outside this laptop are not covered.**

| Repo | Production callers | Opts in by |
|---|---|---|
| baracuda | `baracuda-cuda-emit/src/bin/kernelgen.rs` (AOT generation, `generate`/`generate_variants`) and three `examples/emit_*.rs` | Nothing, unless Baracuda wants a new shape: then `try_generate_with(.., &CallerCaps::LEGACY.with_interface())` in `kernelgen`, **and** its artifact contract must carry the interface to Fuel |
| fuel | none found (it consumes Baracuda's generated artifacts, not `generate`) | Through Baracuda's contract, not by calling this |
| fishstringer | none in `src` outside `#[cfg(test)]` (`slang-emit/src/lib.rs` 756, 1265-1332 are test code) | Its own slice-3 tests pass `.with_interface()`; its launcher does when it can bind the block |
| vulkane | `vulkane-hlsl-emit/src/lib.rs:141` and `vulkane-spirv-emit/src/tests.rs`, test code | Same, when it has a launcher that binds a block |
| rockfish, openclownfish | none found | Unaffected (their backends keep the default `lower_with`) |

## Baracuda's `kernelgen`: how it opts in, and why its output cannot change

`kernelgen` calls `unpopped::generate(..)` / `unpopped::generate_variants(..)` with its `Cuda` backend. **It does not change a line to stay as it is.** Three reasons, each
checkable:

1. `generate` and `generate_variants` are not touched; `try_generate` is `try_generate_with(.., &CallerCaps::LEGACY)`, and `LEGACY` is what `Default` gives.
2. `Cuda` implements `lower` only. The new `Backend::lower_with` is **defaulted to `self.lower(plan)`** and ignores `caps`, so for a backend that does not override it the
   result is **the same call with the same argument**, whatever caps are passed.
3. Nothing in `unpopped` reads `caps` except a backend's own `lower_with`; the stamp, the plan build and the provenance are the same code path.

**To opt in** (only if Baracuda wants a shape that needs an interface, which its `Cuda` backend does not emit today): replace the call with
`try_generate_with(&op, &key, &Cuda, &CallerCaps::LEGACY.with_interface())`, and carry `GeneratedKernel::interface()` into its artifact contract so Fuel can bind it.
Until then there is nothing to do, and a plan that would need the interface is refused for it with `UnsupportedPlanShape`.

**The test that pins "byte-identical"** (written with the code, in `unpopped` and in `unpopped-cpu-c`, which is the in-tree backend that overrides nothing): a Baracuda-shaped
plan, lowered through `try_generate`, `try_generate_with(LEGACY)`, **and** `try_generate_with(with_interface())`; the three `GeneratedKernel`s
compare equal field by field (source, name, interface, provenance), with a stub backend that overrides only `lower` (Baracuda's situation). **Mutant:** make the default
`lower_with` return `lower` plus any change (an interface, a different name) and the test goes red. Control: a stub that *does* override `lower_with` and refuses under `LEGACY`
shows `try_generate` refusing and `try_generate_with(with_interface())` succeeding, so the test can see a difference. **Baracuda's own side:** after the release, one test in
`baracuda-cuda-emit` that its `Cuda` output for its golden plans is unchanged, which only its lane can write (Unpopped has no Baracuda dependency).

## Honest limits

- **What the byte-identical tests cover.** Only elementwise `add` and `mul` plans (`ops()` in `tests/caller_caps.rs`; `caller_caps_unchanged.rs`). A reduction and a scatter are NOT covered: the sentence above that names them is what the test is meant to grow to, not what it does today.
- **No in-tree backend overrides `lower_with`.** `CpuC` overrides only `lower` and `lower_kernel`; the refusal fires only for the test stub `NeedsInterface` (`tests/caller_caps.rs`) until an external backend (fishstringer, Vulkane) overrides it. The hook has a test-only producer in this repository: flagged, not blocked.

- **The flag is a claim, not a check.** Unpopped cannot verify a caller really binds the interface; a caller that passes `with_interface()` and does not is wrong in its
  own code. The point is the **default**: a legacy caller is refused, not mis-bound.
- **Nothing is emitted by this hook.** It makes the refusal possible; fishstringer's slice 3 is what produces the new shapes.
- **Open for pm-kernels:** (1) is one `interface` flag enough, or do you want `params_block` kept separate for a consumer that reads bindings but cannot bind a
  block? Recommend one. (2) `generate_variants_with` now or on demand? Recommend on demand.

— lane: unpopped
