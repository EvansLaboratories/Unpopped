kind: test, causal mask on an m16n8 accumulator subtile (survey row 35)
---
`kernel_subtile_causal_mask.rs`: one subgroup zeroes the upper triangle of a 16 x 8 tile held in the m16n8 accumulator fragment layout, checked against Rust's lower triangle; a strict-triangle and a missing-upper-half-offset mutant change values. Tests and docs only.
