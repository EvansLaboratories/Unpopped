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
