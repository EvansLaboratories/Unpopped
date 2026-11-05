kind: GEMM-family kernel K5: grouped GEMM by a problem table of offsets (ROADMAP U9; tests only)
---
`grouped_gemm` (`tests/support/grouped.rs`, copied into cpu-c): three GEMMs of different shapes `(3,2,4)`, `(4,4,2)`, `(2,4,3)`
in one launch, one block per problem, a `U32` problem table `[P, 6]` of `m, n, k` and element offsets into one flat buffer per
operand (the kernel-layer form of a grouped / pointer-array GEMM until pointer values exist, ROADMAP U9c); a thread outside
its problem's `m x n` does nothing, and `d` is exactly the packed outputs so a stray write runs off or overwrites a neighbour.
Oracle, the C compiled with a host compiler, and the host loop agree. Eleven seeded mutants (block index unused, bounds
swapped or removed, `k` end wrong, each of the three offsets ignored, row strides swapped, accumulator not accumulated) are
all caught.
