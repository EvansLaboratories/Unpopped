kind: GEMM-family kernel K3: split-K in a block and across blocks (ROADMAP U9; tests only)
---
`split_k_in_block` (two thread layers per output sum their half of K, publish through `Shared`, a `Block` barrier, one adds
the two) and `split_k_across_blocks` (a `1x1x2` grid, each block adds its half into `out` with a device-scope
`AtomicRmw Add`), both equal to the unsplit product on mixed-sign data; oracle, C compiled with a host compiler, and host
loop agree. Nine seeded mutants (slice end short, slice start wrong, barrier removed, one partial only, wrong slot, atomic
add to max or exchange, block index unused) are all caught.
