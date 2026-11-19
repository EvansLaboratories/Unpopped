kind: tests pin what the kernel interpreter does with packed values (ROADMAP U15, no API change)
---
`tests/kernel_packed.rs` shows that `Extract` of a `Pack`, of a lane-wise `Add` over two `Pack`s and of a `BitcastPacked` already evaluates inside a kernel body, lane 0 least significant, and pins the one gap: a packed local passes `check_kernel` but the interpreter traps `Unsupported("packed value")`. The ROADMAP U15 row is corrected (it listed all kernel-interpreter packed support as open). No behaviour change.
