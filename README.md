# Unpopped

A vendor-neutral kernel generator, and the vocabulary it generates against.

Unpopped takes a neutral description of an operation and emits a kernel for
whichever target you point it at. The generator, the intermediate
representation, and the classifier vocabulary are device-agnostic; the code that
knows about a particular device lives behind a trait.

The dependency only ever points one way. Device and backend crates depend on
Unpopped. Unpopped depends on no backend, no driver, and no vendor crate.

## Crates

| Crate | Status | What it is |
|---|---|---|
| [`unpopped`](crates/unpopped) | published 0.1.0 | The generator: the neutral IR, the transform pipeline, the `Backend` and `Compiler` traits, a portable-C99 reference backend, and the CPU oracle. |
| [`unpopped-vocab`](crates/unpopped-vocab) | published 0.1.0 | The driver-free classifier vocabulary: dtype hierarchy, `DeviceRepr`, dispatch tags, the `StructureKey` classifier key, dispatch tables. |

## Emitters

Unpopped is becoming **a standard with a normative reference emitter per
target**. Each emitter is its own crate implementing `unpopped::Backend`; none of
them live inside the core.

That is a change of direction, and the tree does not fully reflect it yet — the
C99 and Slang emitters are still in-tree in `unpopped`, and moving them out is
in progress. Until that lands, treat the core crate's emitter modules as
scheduled to relocate rather than as stable API.

| target | crate | owner |
|---|---|---|
| CUDA | [`baracuda-cuda-emit`](https://github.com/ciresnave/baracuda) | Baracuda |
| CPU (C99) | in-tree, relocating | Unpopped |
| Slang | in-tree, relocating | Unpopped |

A project that originates an emitter may keep owning it — Baracuda's CUDA
emitter is theirs, and stays theirs. The umbrella exists so a third party
looking for an emitter has one place to look, not to take ownership away from
the people who wrote them.

Note that this reverses an earlier promise. Previous revisions of this file said
the `unpopped-*` namespace was reserved for Unpopped's own crates specifically
so that a third-party backend would never have to ask permission for a name.
That is no longer the intent, and anyone who read it that way should get in
touch rather than discover the change in a name collision.

## Status

Pre-1.0 and moving. Pin exact versions.

The `Backend` trait in particular is expected to change. It does not yet carry a
structured binding/ABI manifest, its artifact type is source text rather than an
arbitrary word stream, and it takes no target descriptor — all of which a
non-CUDA backend needs. A `0.2` batching those breaking changes is in progress;
they are deliberately being made pre-1.0 so they can land without a major bump.

One known defect worth stating plainly: `unpopped-vocab`'s `ArchSku` can only
spell NVIDIA compute capabilities, and `structure_key` hardcodes a `cuda:`
prefix. Every structure key the neutral vocabulary produces therefore claims a
CUDA target — including keys for backends that have nothing to do with CUDA. It
is schema-visible in the published `0.1.0` wire format. The fix is tracked and
routes through the target-namespace work rather than a quiet patch.

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
