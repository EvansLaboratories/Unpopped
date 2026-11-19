kind: an integer struct-parameter field may size a dynamic allocation (ROADMAP U23 slice 2, additive)
---
An `Extent::Param(local)` allocation may now be sized by an integer field of a `ParamKind::Struct` parameter, as well as by a scalar parameter. `check_kernel` accepts an integer-typed field and still refuses a float or unbound one (`KernelError::AllocExtent`); `run_launch` reads the extent from the matching `KernelArg::Struct` field and keeps its `ArgMismatch` for a non-positive or non-integer value. `DynamicShared` gains `size_field: Option<usize>` (additive; the type is `#[non_exhaustive]`): `size_param` is the struct parameter's index and `size_field` the field's index, `None` for a scalar parameter. cpu-c still declines any struct parameter by name, so no emitter lowers this yet.

Verified: tests in `tests/kernel_struct_extent.rs` (the count sits in the second field so a first-field read fails); seeded mutants (first-field read, `size_field` dropped, float accepted, parameter index fixed at 0) each caught, the last by `kernel_interface`; baseline green after each restore.

Not built: a struct field as a grid dimension, `params_block` derivation, a struct `Stmt::Call` argument, emitter lowering. Layout only: no storage location is chosen per thread set, and baracuda/fishstringer emitters must not pick one either (the driver/compiler doing it is not ours).
