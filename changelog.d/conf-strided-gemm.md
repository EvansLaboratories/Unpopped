kind: Added
---
Kernel-layer conformance kernel for GEMM over arbitrary strides (CUTLASS survey row 7, the `lda` and non-contiguous part): a row-major `A` padded to `lda = K + 3`, a column-major `B`, and a padded `C`, all addressed by logical coordinates with the strides on the argument views. The oracle result is compared against a reference over the logical matrices (ordinary, NaN, all-negative corpora), the poisoned storage outside the views is never read, and the output padding comes back untouched. Two mutants (swapped `B` index, shifted output row) fail by value. The misaligned-`Raw` trap half of the row is already covered by `kernel_raw_place.rs`; the TMA 16-byte stride rule is not covered here, so the row stays Partial.

— lane: unpopped-conformance
