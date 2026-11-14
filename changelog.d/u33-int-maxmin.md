kind: the plan gate admits integer elementwise `Max`/`Min` for a backend that declares `total_integer_ops` (ROADMAP U33)
---
An Elementwise `Max` or `Min` at `I32`, `I64`, `U32` or `U64` is admitted by the plan gate (`check_int_op_admissibility_for`) when the backend declares `Backend::total_integer_ops`, as integer `Div`, `Rem`, `Select` and the comparisons already are. It is exact and total (the larger or smaller operand, no NaN, no overflow), the plan oracle evaluates it exactly since 0.40.10, and an emitter spells it as a compare-select (`unpopped-cpu-c` already spells it at the kernel layer). A backend that does not opt in is refused exactly as before, and a narrower integer dtype (`I8` and the like) stays refused even with the flag. Rockfish asked for this (PM notice, 2026-11-13).

Verified: `total_ints_flag_admits_int_max_min_at_32_and_64_bit_only` (the gate, both ops at the four dtypes with and without the flag, `I8` refused with it) and the `Max`/`Min` cases added to `integer_div_gate_is_opt_in` (a default backend refused by `try_build_plan`, `try_build_plan_for` and `try_generate`; an opted-in backend builds and generates). A mutant that disables the new arm fails both.

Not here: an emitter's compare-select spelling and its device check are the emitter's (Rockfish, Baracuda, Vulkane); this release only stops the gate refusing the op.
