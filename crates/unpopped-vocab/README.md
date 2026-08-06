# unpopped-vocab

The driver-free **classifier vocabulary** for kernel generation: the pure-data
types that describe *what a kernel operates on* and *how it is keyed for
dispatch*.

This crate depends on no backend, no driver, and no vendor crate — only `half`
and `float8`. That is the point. A kernel generator, a kernel selector and a
runtime all have to agree on how a kernel is keyed, but only the runtime needs a
device; bundling the vocabulary with device views would make every consumer
inherit a driver FFI it has no use for.

So the dependency only ever points one way: vendor and device crates depend on
this vocabulary, never the reverse.

## What's in it

- **Dtype hierarchy** — the `KernelDtype` umbrella trait, the `Element` /
  `IntElement` / `FpElement` / `BinElement` / `BiasElement` traits, and the dtype
  wrapper types (`S8`, `U8`, `S4`, `U4`, `Bin`, `F32Strict`, `Fp8E4M3`,
  `Fp8E5M2`).
- **`DeviceRepr`** — the memory-layout marker the dtype hierarchy is bounded on.
  Owned here, so consumers share one definition rather than forking it.
- **Tag enums** — `ElementKind`, `MathPrecision`, `ArchSku`, `LayoutSku`,
  `EpilogueKind`, `ActivationKind`, `OpCategory`, `BackendKind`, and the
  op-family discriminants.
- **The structure key** — `StructureKey`, `OperandDesc`, and the `structure_key`
  derivation: the classifier input consumers key on. Nobody reimplements it, so
  a build matrix and a runtime lookup join on the same token by construction.
- **Dispatch tables** — `DispatchTable`, `DispatchEntry`, `Implementor`,
  `Provenance`, plus the `PlanPreference` / `PrecisionGuarantee` descriptors.

## Stability

Pre-1.0 and additive-in-progress. Op-family discriminants and the category /
backend tags are `#[non_exhaustive]`; downstream `match` arms need a `_` arm.
See the crate docs for which enums are frozen for dispatch keying and which are
still moving.

## Provenance

Developed in the [Baracuda](https://github.com/ciresnave/baracuda) workspace as
`baracuda-kernel-vocab` and extracted here so that generators and backends from
different vendors can share one vocabulary. The commit history came with it.

## License

Dual-licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.
