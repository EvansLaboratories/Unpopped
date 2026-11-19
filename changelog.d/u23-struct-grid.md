kind: an integer struct-parameter field may be a launch grid extent (ROADMAP U23 slice 3, additive)
---
`GridDim::Param` and `GridDim::CeilDiv` may now name an integer field of a `ParamKind::Struct` parameter, as well as a scalar parameter. `check_kernel` accepts an integer-typed field and still returns `KernelError::BadLaunch` for a float or unbound one; `run_launch` resolves the grid from the matching `KernelArg::Struct` field and keeps its `ArgMismatch` for a missing, non-integer or non-positive value. No type changes. cpu-c refuses a grid extent that names no scalar parameter, so it still declines by name and no emitter lowers this yet.

Verified: `tests/kernel_struct_grid.rs` (the count sits in the second field); seeded mutants (scalar-only lookup, first-field read, float accepted) each applied and caught; baseline green after each restore.

Not built: `params_block` derivation, a struct `Stmt::Call` argument, emitter lowering. Layout only: no storage location is chosen per thread set.
