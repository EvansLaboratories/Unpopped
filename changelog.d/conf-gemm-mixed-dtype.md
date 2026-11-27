kind: test: GEMM with an I16 addend over I8 operands, I16 compute and I32 accumulator (CUTLASS 31)
---
`kernel_gemm_mixed_dtype`: `D(I32) = acc + Cast(C(I16))`, operands `I8` widened to `I16` for the product and to `I32` for the sum, built from ordinary ops only (KISS 771 Q3: no `c` role hint). Oracle-compared against a host loop; three type mutants (product narrowed to `I8`, sum kept in `I16`, addend dropped) each change the value on `-128 * -128` data.

— lane: unpopped-conformance
