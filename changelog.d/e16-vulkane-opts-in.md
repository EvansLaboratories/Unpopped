kind: docs, ROADMAP E16 and the integer semantics note record that Vulkane opted in to the total integer ops
---
`vulkane-spirv-emit` now returns `Backend::total_integer_ops() = true` (vulkane#152): guarded `Div`, `Rem`, `Select`, comparison and `ConstInt`, with 288 device results matching the host rule. This was reported by the PM; this lane did not re-run it. `ROADMAP.md` row E16 and `docs/int-div-shift-semantics.md` ("Opted in so far") say so; Baracuda and openclownfish are still to opt in. No code changes.
