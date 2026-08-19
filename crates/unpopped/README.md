# unpopped

A vendor-neutral kernel generator.

`unpopped` takes a neutral description of an operation and produces a kernel for
whichever target you point it at. The IR, the plan, the transform pipeline, the
contracts and the dispatch artifacts are all device-agnostic. The code that knows
about a particular device lives in that vendor's own crate and plugs in through a
trait.

The dependency only ever points one way. Vendor and device crates depend on
`unpopped`. `unpopped` depends on no backend, no driver, and no vendor crate.

## What's in it

- **The neutral IR** — `OpDef`, the scalar-expression DAG, access patterns
  (elementwise, reduction, contraction), views, gather/scatter indexing, and a
  human-readable text form that round-trips.
- **The plan and transform pipeline** — scheduling, canonicalization, and the
  optimizer, all backend-independent.
- **The `Backend` trait** — what a target implements to receive lowered plans, plus
  the `Lowering` seams for the non-universal parts of the math.
- **The `Compiler` trait** — the just-in-time compile seam, with `StubCompiler` for
  building and testing without a toolchain.
- **The CPU oracle** — an independent f64 evaluator that shares no lowering code
  with any emitter, so a bug cannot hide in both the kernel and its reference.
- **Contracts and dispatch artifacts** — KISS-Contract derivation and the
  self-delimiting KISC framing.

## Backends

Each backend is its own crate implementing `unpopped::Backend` for its target:

| target | crate |
|---|---|
| CUDA | [`baracuda-cuda-emit`](https://github.com/ciresnave/baracuda) — the `Cuda` emitter, its NVRTC compiler, and the Fuel synthesizer |
| CPU (C99) | `unpopped-cpu-c` — the portable reference emitter |
| Slang | `unpopped-slang` |

Unpopped hosts **a normative reference emitter per target**, each in its own
crate outside the core. A project that originates an emitter may keep owning it,
as Baracuda owns the CUDA one.

**The core holds no emitter.** A consumer wanting a working kernel needs
`unpopped` plus one emitter crate — the deliberate cost of not privileging any
target by accident of where it lives. `unpopped`'s own tests run against a test
double rather than a real backend, so a core property is never proven against
two things at once.

The split is not cosmetic. Moving the emitters out immediately surfaced a
`Lowering` seam with no builder setter, which made it unreachable from outside
the core — invisible for as long as the emitters lived inside it and could write
the struct literal directly.

Earlier revisions said the `unpopped-*` namespace was reserved for Unpopped's own
crates so a third-party backend never had to ask permission for a name. That is
no longer the intent — see the root README.

## Status

Pre-1.0 and moving. Pin exact versions.

The `Backend` trait in particular is expected to change. Two gaps remain, both
raised by the Vulkane review and both wanted by a SPIR-V backend rather than a
source-emitting one:

- no structured binding/ABI manifest (`Backend` review item #1);
- the artifact type is source text — `GeneratedKernel::source` is a `String` —
  rather than an arbitrary word stream such as SPIR-V `[u32]` (#2).

A **third** item on this list — "takes no target descriptor" — has since landed
and is no longer outstanding: `supports_dtype` takes a `TargetId`, and
`structure_key` takes `impl Into<TargetId>` over an open, KISS §6.8-validated
target namespace. Anything still describing the trait as target-blind is stale.

A **fourth** has since closed too: `JitRequest::arch` was typed `ArchSku`, the
closed CUDA enum, so the JIT **request path** could not name a non-CUDA target
even though the derived key already could. It is now `JitRequest::target:
TargetId` (`src/jit.rs`), and the public `Synthesizer::synthesize` takes
`impl Into<TargetId>` so `ArchSku` call sites compile unchanged — a CUDA caller's
migration is `.into()`.

Worth keeping the distinction that made it safe to fix *after* `0.2.0` rather
than in it: `jit.rs` converted before keying, so **the derived key and the
on-disk artifact identity were already target-neutral.** That made it an
API-expressiveness gap rather than a wire-format or cache-soundness one — a
re-pin and a compile fix, not a re-derivation. The expensive class of breaking
change was already correct in what shipped.

The trait is deliberately shipped pre-1.0 so all of this can land without a major
bump; expect more than one breaking `0.x`.

## Provenance

Developed in the [Baracuda](https://github.com/ciresnave/baracuda) workspace as
`baracuda-kernelgen`, alongside the CUDA backend it was originally written for,
and extracted here once the neutral generator and the device-specific emission
were separated. The commit history came with it.

## License

Dual-licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

## Deferred work

[`docs/deferred.md`](../../docs/deferred.md) is the register of everything knowingly left undone — with the reasoning and the unblocking trigger for each. Several entries are deferred *because doing them naively is worse than not doing them*; that reasoning is not recoverable from the code.
