kind: fix, the kernel oracle's integer `Mul` and the collective `Prod` wrap at 64 bits instead of overflowing `i128`
---
The kernel oracle multiplied two integers with a plain `i128` `*` before wrapping to the type: `u64::MAX * u64::MAX` (about 2^128) exceeds `i128::MAX` and panicked with "attempt to multiply with overflow" in a debug build (a release build wrapped to the right value). Two sites, both now `wrapping_mul` before the wrap: `Mul` in `kernel_eval::eval_int` and `ReduceOp::Prod` in `kernel_collective::combine` (a four-lane `u64::MAX` product passes `(2^64 - 1)^2`). Found by unpopped-conformance; the same-shaped sums cannot overflow (two 64-bit values add inside `i128`).

Audit of `oracle/**` for plain `+`, `-`, `*` on `i128` values: the plan paths already use `wrapping_add`/`wrapping_mul` (`plan_window.rs`, `plan_contract.rs`, `expr_eval::arith`, which is the positive control that the search finds the safe form); the MMA and accumulation-policy sums multiply at most 32-bit operands. No other site found.

Verified: `tests/kernel_int_mul_wraps.rs` panicked at both sites before the fix (red first) and passes after; the `Mul` test covers `u64::MAX * u64::MAX` (1 modulo 2^64), `u64::MAX * 2`, `2^63 * 2` and `i64::MIN * i64::MIN`.
