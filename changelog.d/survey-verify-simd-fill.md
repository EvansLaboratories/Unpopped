kind: test, Metal simd_shuffle_and_fill and rotate (survey row 40) and survey batch 4
---
`kernel_simd_shuffle_fill.rs`: simd_shuffle_and_fill_up/down and simd_shuffle_rotate_up/down built from two Shuffle collectives and a Select, against hand-written arrays from the Metal 4.1 spec example; a fill that ignores its filling fails by value. Survey batch 4 (8 Metal rows) in the verification doc. Tests and docs only.
