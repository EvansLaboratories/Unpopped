kind: additive: `Layout::composition_within`, `LayoutError::ExceedsExtent`
---
`Layout::composition` keeps its behaviour (the last mode of the left layout is read extended at its last stride, CuTe's reading) and its rustdoc now states that rule and that the consumer bounds the result. New `Layout::composition_within` refuses with the new non_exhaustive `LayoutError::ExceedsExtent` when `cosize(rhs) > size(lhs)`. Tests: a sweep of 60 small pairs (agree inside, refuse past, with positive controls) and a hand-worked case; two mutants caught.
