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

The `Backend` trait in particular is expected to change. Two gaps remain, both
raised by the Vulkane review, and both wanted by a SPIR-V backend rather than by
a source-emitting one: it carries no structured binding/ABI manifest, and its
artifact type is source text rather than an arbitrary word stream. They are
deliberately being made pre-1.0 so they can land without a major bump — expect
more than one breaking `0.x`, not a single batch that must be complete before
anything ships.

**The `cuda:`-prefix defect is fixed in-tree and ships in `0.2.0`.** Earlier
revisions of this file described it as open; that is out of date and the
correction matters, because it was being reported upward as a live blocker.
`ArchSku` no longer determines the target: `structure_key` takes
`impl Into<TargetId>` over an open target namespace validated per KISS §6.8
(grammar and charset here; each namespace's capability vocabulary stays with its
maintainer), and `Backend::supports_dtype` takes a `TargetId`.

Two honest caveats on that:

- **It is still true of published `0.1.0`.** Every structure key `0.1.0` produces
  claims a CUDA target, and it is schema-visible in that wire format. If you are
  pinned to `0.1.0` this still affects you; the fix arrives with `0.2.0`.
- **One residual is real in-tree:** `JitRequest::arch` is still typed `ArchSku`,
  the closed CUDA enum. It converts to a `TargetId` before reaching
  `structure_key`, so the derived key and artifact identity are target-neutral —
  this is an API-expressiveness gap, not a wire-format or cache-soundness one.
  The effect is that a non-CUDA JIT request has nowhere to name its target.

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

## Deferred work

[`docs/deferred.md`](docs/deferred.md) is the register of everything knowingly left undone — with the reasoning and the unblocking trigger for each. Several entries are deferred *because doing them naively is worse than not doing them*; that reasoning is not recoverable from the code.
