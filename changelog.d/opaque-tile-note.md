kind: Docs: the opaque-tile MMA design note (MSL simdgroup_matrix and CUDA WMMA are one class)
---
Docs only. `docs/design-opaque-tile-mma.md`: the element-to-thread mapping is unspecified in both Apple `simdgroup_matrix` and CUDA `nvcuda::wmma`, so one statement family serves both; written from primary sources. Nothing is built; Q1 (tensor/tile) is not ruled. No code changes.
