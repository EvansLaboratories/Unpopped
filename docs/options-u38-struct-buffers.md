# Options: anonymous-struct typedefs read through struct pointers (U38d, census wall: 7 kernels x 5 block types = 35)

Status: **options for pm-kernels' ruling; no code. Touches struct layout: needs the U23 patent read (US20230305845A1, layout-only guard) confirmed before any option that computes layout.**

## The shape

`typedef struct { half d; uint8_t qs[QK_K/2]; ... } block_q4_K;` (`baracuda_gguf.cuh`); kernels take `const void* vx` and read `((const block_q4_K*)vx)[ib].d`, `.qs[k]`, `.scales[j]`. The kernel's declared type text is the whole `struct { ... }`, which the walker reads as the type name and declines (`the type "struct { ... }"`): an anonymous struct, then a pointer to it, then a byte buffer used as an array of blocks.

## Options

- **A. Struct-in-buffer with natural C layout (the general fix).** `(const S*)vx + i` is a pointer with a byte stride `sizeof(S)`; `p->f` / `p[i].f[k]` is a `Place::Raw` load at `i * sizeof(S) + offsetof(S, f) + k * sizeof(elem)`, with `sizeof`/`offsetof` from C's natural alignment for scalar, array and nested-struct members (packed or `aligned(n)` attributes decline). Additive, walker only. Cost L; needs: anonymous struct typedef resolution, `sizeof`/`offsetof` of a struct, pointer-to-struct arithmetic, member read through a pointer. The layout rule is what the patent guard governs: confirm it is covered by the U23 layout-only reading (`docs/` U23 patent note) before building.
- **B. The caller states the layout (EntryModel-style).** Like `CallerCaps` and `LiftContext::with_include`: `LiftContext::with_struct_layout("block_q4_K", size, [(name, offset, elem, count)])`; the walker only applies a layout it is given, never computes one. Additive. Cost M; removes the layout-derivation question (the caller, Baracuda, already knows `sizeof(block_q4_K)` from its `static_assert`s). Needs Baracuda to pass the layouts (a lane change, small: it has them).
- **C. Decline;** Baracuda rewrites these kernels over `uint8_t*` with explicit offsets (it already does for the batched variants).

## Recommendation

B (the caller supplies the layout), then A only if callers should not have to. B avoids the layout-derivation and patent question entirely and is the same shape as the other caller-stated facts. Lane change: baracuda passes the five block layouts via the context.

## Ruling and result (pm-kernels 2026-11-22: option B)

Conditions as ruled: the walker **never computes or infers a layout**; it validates the one the caller gives (offsets plus `count * elem` within `size`, no overlapping fields, named error otherwise) and declines by name on a missing or mismatching one; additive; a test with a seeded wrong offset must fail the oracle result. Baracuda derives its five layouts from a compiled C harness printing `sizeof`/`offsetof` under `nvcc` (the control), not from reading the source (a `[TASK]` to Baracuda after the PR window).

**No layout is computed here, so the U23 patent guard (US20230305845A1, layout only) is not engaged:** the walker maps a source form to an existing byte-offset place (`Place::Raw`) from numbers the caller states, and adds no IR construct (`docs/PATENT-SWEEP.md` 417-419: "still no IR construct added; each maps a source form to an existing statement"; verdict NOT CLEARED, unchanged by this work).

Step 1 built (walls-2, `da7f5f1b`): `LiftContext::with_struct_layout`, `StructLayout`, `FieldLayout`, `StructLayout::validate`, seeded inconsistencies (a field past the end, overlapping fields, size 0, a duplicate, an unknown element type, a zero count, two layouts for one name) each declined by name (`tests/kernel_body_struct_layouts.rs`, 5 mutants caught). Next: an anonymous-struct typedef resolves to a laid-out struct; `(const S*)p` is a byte pointer with stride `size`; `p[i].f` and `p[i].f[k]` read `Place::Raw` at `i * size + offset + k * width`; the oracle test with a seeded wrong offset must produce a wrong result.

