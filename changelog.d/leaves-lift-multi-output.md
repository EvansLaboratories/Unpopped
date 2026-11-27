kind: added
---
`unpopped-convert`: `lift_elementwise_multi` lifts an elementwise kernel that stores several outputs (`output0`, `output1`, ..., one store each at one index) into a multi-output `OpDef`, with a per-output element type (`Some(U8)` for a mask beside a float value). A declared output with no store, a store to an undeclared output and stores at different indices are refused; `lift_elementwise` is unchanged. ROADMAP U67. The fishstringer-slang-emit spelling is UNVERIFIED; tests drive the oracle against Rust references.
