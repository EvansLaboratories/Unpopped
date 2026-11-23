# Options: anonymous-struct typedefs read through struct pointers (U38d, census wall: 7 kernels x 5 block types = 35)

Status: **options for pm-kernels' ruling; no code. Reads struct memory through a pointer; patent status (US20230305845A1) is at the end: the U23 guard is the no-annotation rule, read as not engaged by option B (research, not counsel; not cleared).**

## The shape

`typedef struct { half d; uint8_t qs[QK_K/2]; ... } block_q4_K;` (`baracuda_gguf.cuh`); kernels take `const void* vx` and read `((const block_q4_K*)vx)[ib].d`, `.qs[k]`, `.scales[j]`. The kernel's declared type text is the whole `struct { ... }`, which the walker reads as the type name and declines (`the type "struct { ... }"`): an anonymous struct, then a pointer to it, then a byte buffer used as an array of blocks.

## Options

- **A. Struct-in-buffer with natural C layout (the general fix).** `(const S*)vx + i` is a pointer with a byte stride `sizeof(S)`; `p->f` / `p[i].f[k]` is a `Place::Raw` load at `i * sizeof(S) + offsetof(S, f) + k * sizeof(elem)`, with `sizeof`/`offsetof` from C's natural alignment for scalar, array and nested-struct members (packed or `aligned(n)` attributes decline). Additive, walker only. Cost L; needs: anonymous struct typedef resolution, `sizeof`/`offsetof` of a struct, pointer-to-struct arithmetic, member read through a pointer. Option A has the walker compute `sizeof`/`offsetof` itself, which the PM did not rule (the caller-stated B was); A's patent status is the same as B's reading below.
- **B. The caller states the layout (EntryModel-style).** Like `CallerCaps` and `LiftContext::with_include`: `LiftContext::with_struct_layout("block_q4_K", size, [(name, offset, elem, count)])`; the walker only applies a layout it is given, never computes one. Additive. Cost M; removes the layout-derivation question (the caller, Baracuda, already knows `sizeof(block_q4_K)` from its `static_assert`s). Needs Baracuda to pass the layouts (a lane change, small: it has them).
- **C. Decline;** Baracuda rewrites these kernels over `uint8_t*` with explicit offsets (it already does for the batched variants).

## Recommendation

B (the caller supplies the layout), then A only if callers should not have to. B avoids the layout-derivation and patent question entirely and is the same shape as the other caller-stated facts. Lane change: baracuda passes the five block layouts via the context.

## Ruling and result (pm-kernels 2026-11-22: option B)

Conditions as ruled: the walker **never computes or infers a layout**; it validates the one the caller gives (offsets plus `count * elem` within `size`, no overlapping fields, named error otherwise) and declines by name on a missing or mismatching one; additive; a test with a seeded wrong offset must fail the oracle result. Baracuda derives its five layouts from a compiled C harness printing `sizeof`/`offsetof` under `nvcc` (the control), not from reading the source (a `[TASK]` to Baracuda after the PR window).

**Patent status (corrected 2026-11-22 after fireline's read of the independent claims; the earlier wording here, "no layout is computed so the guard is not engaged", gave the wrong reason).** The U23 guard (US20230305845A1, `docs/PATENT-SWEEP.md` 391, 417-419 and the U38d paragraph at its end) is: no thread-set, memory-kind or copy-count annotation on a `Param`, `Place` or `ParamsBlock`, and no storage location chosen by which threads use data. Claims 1, 9, 15, 21 and 27 turn on data "selectively stored in one or more memory locations based, at least in part, on one or more threads to use the data"; none mentions layout, offset, size, alignment or struct. Option B adds no annotation and makes no store: it is **outside as read**. Research, not an opinion of counsel; the application is pending; read from a diff grep and this document, not a full code audit; **NOT CLEARED**. Re-read trigger: `docs/PATENT-SWEEP.md` (U38d paragraph).

Step 1 built (walls-2, `da7f5f1b`): `LiftContext::with_struct_layout`, `StructLayout`, `FieldLayout`, `StructLayout::validate`, seeded inconsistencies (a field past the end, overlapping fields, size 0, a duplicate, an unknown element type, a zero count, two layouts for one name) each declined by name (`tests/kernel_body_struct_layouts.rs`, 5 mutants caught). Next: an anonymous-struct typedef resolves to a laid-out struct; `(const S*)p` is a byte pointer with stride `size`; `p[i].f` and `p[i].f[k]` read `Place::Raw` at `i * size + offset + k * width`; the oracle test with a seeded wrong offset must produce a wrong result.

