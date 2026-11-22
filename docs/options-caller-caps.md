# Options: `CallerCaps`, so a backend can refuse a plan a legacy caller would bind blindly

Status: **options note, no code** (pm-kernels `[TASK]` 2026-11-22; code after #385 merges, never on its stack). Asked for by fishstringer's E17 slice 3
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
- **`generate_variants`** is untouched in this slice. A `generate_variants_with` is the same pattern if a consumer needs it; none does today.

## Who calls `generate` today, and how each would opt in

Read with `git grep` at each repo's current checkout, 2026-11-22; test modules named separately. **Not run, and adopters outside this laptop are not covered.**

| Repo | Production callers | Opts in by |
|---|---|---|
| baracuda | `baracuda-cuda-emit/src/bin/kernelgen.rs` (AOT generation, `generate`/`generate_variants`) and three `examples/emit_*.rs` | Nothing, unless Baracuda wants a new shape: then `try_generate_with(.., &CallerCaps::LEGACY.with_interface())` in `kernelgen`, **and** its artifact contract must carry the interface to Fuel |
| fuel | none found (it consumes Baracuda's generated artifacts, not `generate`) | Through Baracuda's contract, not by calling this |
| fishstringer | none in `src` outside `#[cfg(test)]` (`slang-emit/src/lib.rs` 756, 1265-1332 are test code) | Its own slice-3 tests pass `.with_interface()`; its launcher does when it can bind the block |
| vulkane | `vulkane-hlsl-emit/src/lib.rs:141` and `vulkane-spirv-emit/src/tests.rs`, test code | Same, when it has a launcher that binds a block |
| rockfish, openclownfish | none found | Unaffected (their backends keep the default `lower_with`) |

## Honest limits

- **The flag is a claim, not a check.** Unpopped cannot verify a caller really binds the interface; a caller that passes `with_interface()` and does not is wrong in its
  own code. The point is the **default**: a legacy caller is refused, not mis-bound.
- **Nothing is emitted by this hook.** It makes the refusal possible; fishstringer's slice 3 is what produces the new shapes.
- **Open for pm-kernels:** (1) is one `interface` flag enough, or do you want `params_block` kept separate for a consumer that reads bindings but cannot bind a
  block? Recommend one. (2) `generate_variants_with` now or on demand? Recommend on demand.

— lane: unpopped
