kind: GEMM-family kernel K6: a K loop of `Stmt::Mma` accumulating in the fragment (ROADMAP U9; tests only)
---
`unpopped-cpu-c/tests/kernel_gemm_mma_chain_emit_agrees.rs`: two `Stmt::Mma` run back to back over two K slices, the second reading
the first result as its `C` (`d` is `c`), for a dense atom, a 2:4 sparse atom, a block-scaled atom and both (the toy 4-lane
atom of the extras tests). For each the compiled C agrees with the launch interpreter byte for byte, and the interpreter chained
result equals the sum of the two slices run as separate single-slice MMAs on a zero `C`, which pins the accumulation itself.
Four seeded product mutants (the oracle ignoring `C` in the integer and in the float path, the C emitter starting its
accumulator at zero, its K loop one short) are all caught. The sparse WGMMA/tcgen05 atoms and a real warp-level tile loop
wait for a second lane-table source (ROADMAP U7 / U8b).
