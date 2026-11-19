kind: a struct may be passed by value to a call (ROADMAP U23 slice 4, additive)
---
New `CallArg::Struct(Vec<ScalarExpr>)` (the enum is `#[non_exhaustive]`) passes a struct by value to a callee's `ParamKind::Struct` parameter: one expression per field, in the callee's field order, each of the field's type. `check_module` refuses a wrong field count, a field of another type, and a scalar passed for a struct (`KernelError::CallArgMismatch`); the oracle binds each field to the callee's field local, rounding a float to the field's type as a scalar argument is; a barrier-reaching callee needs every struct field uniform (`BarrierDivergent`). `unpopped-cpu-c` still refuses a struct argument by name ("a call argument and its parameter differ in kind"), so no emitter lowers this yet.

Verified: `tests/kernel_call_struct.rs` (fields are `a * 10 + b`, so a swapped order shows); seeded mutants (reversed field order, uniformity ignoring struct fields, type check dropped, count check loosened) each applied and caught; baseline green after each restore.

Not built: `params_block` derivation, emitter lowering. Layout only: no storage location is chosen per thread set.
