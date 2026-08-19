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

## Migrating `0.1.0` → `0.2.0`

Eight public breaks, listed because the first adopter was told "zero code
changes" and hit two of them in `cargo check`. That characterisation was wrong:
it generalised from four *vocabulary* questions that were answered accurately to
a claim about the whole surface, which nobody had checked. **Requires
`unpopped-vocab` `0.3.0`** — not `0.2.0`, which is a different published crate.

| what changed | migration |
|---|---|
| **Emitters left the core.** `unpopped::cpu_c` / `unpopped::slang` are gone. | Depend on `unpopped-cpu-c` / `unpopped-slang`. Both are **in-tree and unpublished** — see [`docs/deferred.md`](../../docs/deferred.md) §A for why, and the trigger. |
| **`Backend::supports_dtype` takes a `target: TargetId`.** | Add the parameter. **Ignoring it is a legitimate implementation** — it is the *storage* gate ("can this target spell the scalar type"), and arch-conditional *compute* gating belongs behind a capability manifest rather than in a transcribed table. See the note below. |
| **`optimize` and `optimize_top_k` take a `dtype: ElementKind`.** | Pass the expression's compute dtype. It exists so folding rounds at the **device's** precision — folding an `f32` kernel's constants at `f64` diverges by 1 ULP on a chained fold. |
| **`convert.rs`'s per-language wrappers are replaced by a `Frontend` descriptor.** | `CUDA` and `SLANG` are now *values*, not APIs. Four generic functions replace eight per-language ones. |
| **`Lowering` gains a builder and an open constant-literal seam.** | Build via `Lowering::builder(...)`. Constant spelling is dtype-dependent and not universal, so it is a seam rather than a fixed call. |
| **`GeneratedKernel` carries `Provenance`.** | Construct via `GeneratedKernel::new(name, source)`; the core stamps provenance. A backend-built fragment carries none by design. |
| **`AccumSpec` is `#[non_exhaustive]`.** | Add a wildcard arm. |
| **One plan legality table, reachable in both shapes.** | No action unless you re-implemented the gate. |

### On `supports_dtype` ignoring its `target`

Worth stating because two independent implementations reached it separately, from
opposite directions. The predicate answers **"can this backend spell this dtype
as a scalar type on this target"** — a *storage* question. It is not "does this
target compute in this dtype".

Conflating those is a live defect class, not a pedantic distinction. This crate's
own C emitter had it: `supports_dtype` was
`!matches!(F16 | Bf16) && scalar_ctype(dtype).is_some()` — a storage fact
(`scalar_ctype` returns the *carrier*) answering a compute question. It gave
right answers for a reason it never stated, and the two-dtype exclusion list was
the patch over the gap. Vulkane's mirror is `st16` =
`storageBuffer16BitAccess`, a **storage** capability a device can advertise while
doing the arithmetic in `f32`; the CUDA mirror is fp8 living in memory on any
arch and computing only on Ada+.

So a backend whose scalar spelling does not vary by target may ignore the
parameter honestly. Arch-conditional *compute* gating wants a machine-readable
per-namespace capability manifest (KISS §6.8-0008…-0013), not a transcribed
table — transcribing what you think another vendor's capability set means is
precisely the coupling the open target model exists to remove.

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
