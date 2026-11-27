# Float `Max`/`Min`/`Sum` collectives: what the oracle says, and what each target must measure (2026-11-27)

Source of the oracle rules: `crates/unpopped/src/oracle/kernel_collective.rs` `combine` and `nan_first`, read at `origin/main` on 2026-11-27. Source of the MSL measurement: pm-kernels' relay of vulkane#189 run 45972534688 (a hosted paravirtual device, **not Apple hardware**; safe-math; 32 lanes), vulkane's measurement; Unpopped has not re-run it.

## The oracle's rule (a `Reduce` or `Scan` over lanes, ascending lane order)

* **`Max`/`Min` on a float propagate NaN.** Once any lane's value (or the running fold) is NaN, the result is NaN: the first NaN seen is kept. This is the torch convention.
* **`Max`/`Min` keep the first of equal values** (a strictly better value replaces, a tie does not), so `Max(-0.0, +0.0)` is lane order's first zero, never a normalized one.
* **`Sum` is the left fold in lane order**, rounded to `F32` at each step when the type is `F32`; `+inf` plus `-inf` is NaN (IEEE). The oracle documents that a device may reassociate, so a `Sum` is compared within a tolerance, not by bytes.
* `MaxNum`/`MinNum` (NaN-ignoring) are **not admitted** as collective folds (`CollectiveUnsupported`), so the oracle has no NaN-ignoring collective; the vector's NaN-ignoring mutant (Unpopped#464, `kernel_collective_float_nan.rs`) is Rust `f32::max`. The NaN's payload and sign are not checked, only `is_nan`.
* An exclusive scan hands lane 0 the identity: `-inf`/`+inf` for float `Max`/`Min`.

## What a target must measure before it lowers a float collective

An emitter that lowers a float `Max`/`Min` `Collective` to a vendor primitive must **measure that primitive's NaN and signed-zero behaviour on its own target** and either match the rule above or decline by name. Do not assume the vendor op propagates NaN.

| Target | Measured (by whom) | What it means for a float `Max`/`Min` lowering |
|---|---|---|
| MSL `simd_sum` | propagates NaN (any lane); `+inf` plus `-inf` is NaN; `sum(-0,+0)` is `+0`, all `-0` is `-0`; addition order not determined (vulkane#189, hosted paravirtual device, safe-math, 32 lanes) | matches the oracle's NaN rule for `Sum`; order is a tolerance matter |
| MSL `simd_max`/`simd_min` | **ignore a NaN operand** (a NaN among finite lanes returned the finite max/min); ties go to the lowest lane's zero (consistent with first-seen, not proof) (same run) | **does not match** the oracle: a lowering needs an explicit any-NaN check and select, or declines |
| Not measured | fast-math, other devices, `F16`/`Bf16` | unknown: measure |
| CUDA/Slang/HLSL/WGSL/HIP warp reduce | **not measured here.** A shuffle-based float max built on `fmaxf` is NaN-ignoring per C99; `__reduce_max_sync` is integer-only | unknown until measured: each emitter lane measures its own |

Vulkane keeps its float collectives declined by name on this evidence.

## What is open

* A conformance row: a float `Max`/`Min` `Collective` kernel with a NaN lane (and a `-0.0`/`+0.0` tie), the oracle vector, and a mutant that ignores the NaN. Assigned to unpopped-conformance.
* This note states the oracle's rule; it does not say the rule is the right one for every consumer (it is torch's and KISS's union vectors').

— lane: unpopped
