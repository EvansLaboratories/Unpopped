kind: Added
---
Kernel-layer conformance kernel for split-K convolution in the serial-reduction form (CUTLASS survey row 50): the channel reduction of a 1-D convolution is cut into four fixed slices, launch 1 writes one partial per slice (one block per slice) to a workspace, and launch 2 sums the partials in slice order. Oracle-compared against a plain convolution over ordinary, NaN and all-negative corpora; a slice that reads the wrong channels and a reduction that drops a slice are mutants that fail by value. The stream-K division of one tile between blocks, persistent scheduling and work-stealing are held and not modelled, so the row stays Partial.

— lane: unpopped-conformance
