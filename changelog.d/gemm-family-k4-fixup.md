kind: GEMM-family kernel K4: split-K with a `Device`-barrier fix-up (ROADMAP U9; tests only)
---
`split_k_with_fixup`, the stream-K shape without atomics: the two blocks of a `1x1x2` grid write half-K partials to a
global workspace, a `Device` barrier orders the phases, and block 0 adds them into `out`. Oracle, the C (the S6a phase
loop) and the host loop agree. Five seeded mutants (barrier removed or demoted to `Block`, colliding workspace slots, one
partial read, slice not chosen by block) are caught; moving the fix-up to block 1 survives and is equivalent (either block
computes the same `out`).
