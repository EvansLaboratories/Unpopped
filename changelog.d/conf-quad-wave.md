kind: Added
---
Kernel-layer conformance kernels for HLSL's quad reads and wave queries, built from the shuffles and votes already on main (rows 18, 45 and 46 of `docs/ir-union-evidence/conformance-idiom-gaps-2026-11-27.md`): `QuadReadAcrossX/Y/Diagonal` (XOR masks 1, 2, 3), `QuadReadLaneAt` (an index shuffle to the quad's base plus the requested lane), `WaveReadLaneFirst` and `WaveActiveAllEqual` (an index-0 shuffle and a vote). Eight lanes in one subgroup (two quads) hold distinct boundary integers; each result is compared on the oracle with a reference written from the quad's definition (base and position), not from the masks. Mutants (the wrong X mask, a dropped quad base, `Any` for `All`) fail by value. Not modelled: which lane is first among inactive lanes (the whole subgroup participates), and a float all-equal over NaN.

— lane: unpopped-conformance
