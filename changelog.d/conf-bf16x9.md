kind: Added
---
Kernel-layer conformance kernel for nine-term `bf16` emulation of an `f32` GEMM (CUTLASS survey row 27): each operand is split into three `bf16` pieces, the nine products are accumulated in `f32`, and the oracle result is compared bit for bit against an independent reference that rounds to `bf16` by bit manipulation, over ordinary, NaN and all-negative corpora. A second test shows the emulation is within 1e-6 of the exact `f64` product where a plain `bf16` product is not. Three mutants (leading piece only, unchained residual, reversed pairing) fail by value. Finding: the oracle rounds a whole expression tree once, at the `Assign`, so the kernel emits one `Assign` per term to keep each add rounded to `f32`. Arithmetic only; no hardware atom, and no PATENT-SWEEP entry names this row.

— lane: unpopped-conformance
