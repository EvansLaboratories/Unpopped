kind: test, 64-bit integer atomics (survey row 50)
---
`kernel_atomic_wide.rs`: twelve threads apply each integer `AtomicRmw` op with an operand above 2^32 to an `I64` and a `U64` cell; expected cells come from Rust operators, and a narrowed-operand and a wrong-op mutant change the cell by value. Adds the survey batch 1 and 2 verification doc. Tests and docs only.
