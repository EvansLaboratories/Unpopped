kind: Added
---
Kernel-layer conformance kernel for grouped block rasterization (row 31 of `docs/ir-union-evidence/conformance-idiom-gaps-2026-11-27.md`: the grouped ordering of Triton's public matmul tutorial, TileLang `T.use_swizzle`): a linear tile id is mapped to `(tile_m, tile_n)` over a 5 x 4 grid with groups of two rows, so the last group is clamped to one row. Integer index arithmetic only (divide, modulo as divide-multiply-subtract, `Min`); one thread per tile id. The oracle result is compared with a reference that enumerates the order by nested loops, and the mapping is checked to be a bijection that differs from row-major order. Mutants (no clamp of the last group, a modulo where a divide belongs) fail by value. It models the mapping only: no persistent loop, no scheduling, no work stealing, no division of a tile between blocks.

— lane: unpopped-conformance
