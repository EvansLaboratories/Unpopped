# Options: compare-and-swap loops (U38a, census wall: 5 Baracuda kernels)

Status: **options for pm-kernels' ruling; no code.** Measured on lane/u19-walls-2 (off 1a55b0e6), Baracuda corpus, `--args --includes`.

## The shape

`baracuda_segment.cuh` `atomic_max_f32` / `atomic_min_f32` (also in `baracuda_atomic.cuh`, `baracuda_block_atomic.cuh`, `baracuda_indexing.cuh`, `baracuda_upsample.cuh`, `baracuda_fractional_max_pool.cuh`; 27 `atomicCAS` sites):

    int* iaddr = reinterpret_cast<int*>(addr);  int old = __float_as_int(*addr);  int assumed;
    do { assumed = old; float cur = __int_as_float(assumed); if (val <= cur) return;
         old = atomicCAS(iaddr, assumed, __float_as_int(val)); } while (assumed != old);

## What the IR already has (no widening needed for B)

`Stmt::While`, `Break`, `Continue`, `Return`, `Stmt::AtomicCas { place, expected, desired, old, .. }`, `ScalarExpr::Bitcast`, `AtomicOp::{Max, Min}` (float `Max`/`Min` as an `AtomicRmw`). The walker lifts `while` and early `return` but not `do { } while`, `atomicCAS`, `__float_as_int` / `__int_as_float`, or a `reinterpret_cast<int*>(float*)` used for an atomic (declined today: "atomic through a pointer cast", `tests/kernel_body_qualified_calls.rs`).

## Options

- **A. Recognise the idiom, lift to the native op: BLOCKED as written (found 2026-11-22 before building).** The checker REFUSES a float `AtomicRmw` `Max`/`Min` (`tests/kernel_atomic_rmw.rs::a_float_add_is_admitted_for_f32_and_f64_only_and_max_min_are_not`: F32 and F64 admit `Add` only) and the oracle traps one ("an atomic outside the v1 form", `kernel_eval.rs` `atomic_apply`), so there is no native float max/min to lift to. Making one means widening what the checker, the oracle and every emitter accept (not an enum widening, but a new accepted form each emitter must spell): an `[ASK]`. NaN semantics for any such op, pinned from the C loop (corrected by pm-kernels): `if (val <= cur) return;` is false for a NaN `val`, so the loop does NOT early-out and the CAS stores the NaN; a NaN `cur` is replaced by `val` (every comparison with a NaN is false); ties return without storing. A native float `Max` that ignores NaN (`maxNum`) would differ, so it may not be used unmarked.
- **B. Lift the general form.** `do/while` to `While`, `atomicCAS(ptr, a, b)` to `AtomicCas` with `old`, `__float_as_int` and friends to `Bitcast`, and let an atomic through a same-width cast pointer address the buffer's element as the integer (a `Place::Raw` int element over a float buffer). Needs the checker's v1 atomic form to admit `Place::Raw` with an element of another type than the buffer's (an additive checker relaxation, the one IR-adjacent part; no enum widening). Cost M-L. Lifts every CAS loop in the corpus (27 sites), not only the max/min helpers; the oracle must run a CAS loop (a spin until success: single-threaded progress is fine, with many threads it is a bounded retry per the oracle's scheduler).
- **C. Decline (status quo)**, and keep the helper as it is and accept that these kernels do not lift (Baracuda could route them through an `ExternCall`-style helper the oracle models separately). Cost 0 here.

## Recommendation

A now (the two helpers cover the 5 kernels, S, no IR/checker touch), B as a later row if more CAS idioms matter (U38a-2). Lane changes: none for A; for B the emitters already spell `AtomicCas`.
