# Unpopped

A vendor-neutral kernel generator, and the vocabulary it generates against.

Unpopped takes a neutral description of an operation and emits a kernel for
whichever target you point it at. The generator, the intermediate
representation, and the classifier vocabulary are device-agnostic; the code that
knows about a particular device lives in that vendor's own crate and plugs in
through a trait.

The dependency only ever points one way. Vendor and device crates depend on
Unpopped. Unpopped depends on no backend, no driver, and no vendor crate.

## Crates

| Crate | Status | What it is |
|---|---|---|
| [`unpopped-vocab`](crates/unpopped-vocab) | in tree | The driver-free classifier vocabulary: dtype hierarchy, `DeviceRepr`, dispatch tags, the `StructureKey` classifier key, dispatch tables. |
| `unpopped` | not yet extracted | The generator itself: the neutral IR, the transform pipeline, the `Backend` and `Compiler` traits, the portable-C99 reference backend, and the CPU oracle. |

Backends are **not** `unpopped-*` crates. They live under the identity of
whoever owns the device — a CUDA backend is a `baracuda-*` crate, a Vulkan one a
`vulkane-*` crate — and each implements `unpopped::Backend` for its own target.
The `unpopped-*` namespace is reserved for Unpopped's own crates.

## Status

Unpopped is being extracted from the
[Baracuda](https://github.com/ciresnave/baracuda) workspace, where the generator
was developed as `baracuda-kernelgen` alongside a CUDA backend. The extraction
separates the neutral generator from the device-specific emission so that
backends from different vendors can share one generator and one vocabulary.

The vocabulary crate has been carved and is neutral. The generator carve is in
progress: it requires first lifting the shared C-family scalar emitters out of
the CUDA emitter, promoting the CPU oracle's reference semantics into an
explicit backend conformance contract, and relocating the CUDA-specific compiler
and backend to the vendor side.

Pre-1.0 and moving. Pin exact versions.

## History

The crates here carry their original commit history from Baracuda, but git
exposes it in a non-obvious way — `git log -- crates/<crate>` will show you only
the merge. See [docs/history.md](docs/history.md) for how to read it, and why
commit SHAs do not match Baracuda's.

## License

Dual-licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
