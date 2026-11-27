kind: Added
---
Kernel-layer conformance kernel for complex GEMM (CUTLASS survey rows 26 and 82): interleaved `(re, im)` `F32` operands, a four-multiply and a three-multiply (Gaussian) form, oracle-compared against an independent complex reference over ordinary, NaN and all-negative corpora, with a conjugated-operand and an imaginary-sign mutant that fail by value. There is no complex element kind, so the rows stay Partial for the dtype itself; the 3xTF32 emulation is not covered here.

— lane: unpopped-conformance
