kind: tests: integer `where` (Select) by U8 mask and by comparison, I32/I64/U8/U32 (Triton survey)
---
`kernel_int_where`: the index-masking idiom as an integer `Select` in the oracle, the condition a loaded `U8` mask tested `!= 0` (values 1, 2, 255) or a comparison; host-loop reference; swapped arms change the value at every width. Tests only.

— lane: unpopped-conformance
