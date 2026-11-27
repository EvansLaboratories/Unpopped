kind: Added
---
Kernel-layer conformance kernel for a swizzled shared-memory tile (CUTLASS survey row 66): an 8 x 8 transpose through a block-shared tile addressed with an XOR swizzle (`col ^ row`) around one block barrier, oracle-compared against a plain transpose over ordinary, NaN and all-negative corpora. Two mutants (a load that forgets the swizzle, a store with a narrowed mask) stay in range and race-free and fail by value. The swizzle is index arithmetic only; `docs/PATENT-SWEEP.md` idea 15 reads the XOR swizzle as pure data, outside as read, not a clearance. TMA-style hardware swizzle modes and bank-conflict counting are not covered, so the row stays Partial.

— lane: unpopped-conformance
