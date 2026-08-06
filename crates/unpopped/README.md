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
- **`CpuC`** — a portable-C99 reference backend that compiles and runs GPU-free. It
  is what lets this crate self-test with no device present.
- **The CPU oracle** — an independent f64 evaluator that shares no lowering code
  with any emitter, so a bug cannot hide in both the kernel and its reference.
- **Contracts and dispatch artifacts** — KISS-Contract derivation and the
  self-delimiting KISC framing.

## Backends

Backends are **not** `unpopped-*` crates. They live under the identity of whoever
owns the device and implement `unpopped::Backend` for their target:

| target | crate |
|---|---|
| CUDA | [`baracuda-cuda-emit`](https://github.com/ciresnave/baracuda) — the `Cuda` emitter, its NVRTC compiler, and the Fuel synthesizer |
| CPU (C99) | in-tree (`CpuC`) — the neutral reference backend |

The `unpopped-*` namespace is reserved for Unpopped's own crates, so a third-party
backend never has to ask permission for a name.

## Status

Pre-1.0 and moving. Pin exact versions.

The `Backend` trait in particular is expected to change: it does not yet carry a
structured binding/ABI manifest, its artifact type is source text rather than an
arbitrary word stream, and it takes no target descriptor — all of which a
non-CUDA backend needs. Those are known and being worked; the trait is
deliberately shipped pre-1.0 so they can land without a major bump.

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
