# Options note: compelling callers to use `required_fidelity` (ROADMAP C5)

Status: **options, nothing built.** Asked by the PM (2026-11-01): "C5 design note, docs-only PR first". The gap is
`docs/conformance.md` OPEN-1 item 1: `oracle::compare(expected, actual, fidelity)` takes whatever `Fidelity` it is
handed, so a test can pass at a band its author widened until it went green, and the diff looks the same as a band
that reflects the kernel's real error. `oracle::required_fidelity(plan, operands)` derives the band a cell is
*entitled* to, and nothing makes a caller ask.

## What is measured (predicate and refs)

Predicate: `git grep` over `*.rs` at each repo's `origin/main`, for `compare(` call sites in Unpopped and for
`Fidelity::(BitExact|Tolerant)` / `oracle::compare` / `oracle::Fidelity` / `required_fidelity` in the siblings.
Unpopped at `c030e46`; siblings at baracuda `c7a659ae`, bonefish `35db044`, vulkane `f7742b3`, openclownfish
`01a7425`, rockfish `9f66447`, fuel `5602a2c53`, unpopped-harness `e27a6ac`. These are counts of text, not of
reachable calls, and a repo that wraps `compare` in its own helper is counted at the helper's literals.

| where | `compare(` call sites | hand-built `Fidelity` literals |
|---|---|---|
| Unpopped `src` | `oracle.rs` 28 (production + its unit tests), `kernel_lift/row_sort.rs` 3 | `oracle.rs` 17 |
| Unpopped tests (two crates) | `complex_domain` 8, `wide_integer_comparison` 8, `required_fidelity` 3, `cpu_end_to_end` 1 | 8 + 8 + 8 + 1, plus 1 in `a_flushing_device_declines_exact` |
| baracuda | not counted by this predicate | 1 (`crates/baracuda-cuda-emit/tests/fuzz.rs:582`, a `Tolerant`) |
| bonefish | not counted by this predicate | 7, all in `bonefish-sycl-emit/tests/sycl_end_to_end.rs` (5 `BitExact`, 2 `Tolerant`, through its own `check` helper) |
| vulkane, openclownfish, rockfish, fuel, unpopped-harness | 0 | 0 |

`required_fidelity` has **zero non-test callers** (ROADMAP C5, counted at `496c192`). So the derivation is exercised by
its own tests and by nothing that gates a release.

One asymmetry decides the shape of the fix. A band **tighter** than the entitlement is harmless (the check is
stricter); a band **looser** than the entitlement is the defect. So a compelling API need not forbid a hand-picked
band, only a looser one, and a looser one can be detected: compare it against `required_fidelity`.

## Options

| | change | breaks | what it compels |
|---|---|---|---|
| **A. `compare_for_plan`** | add `compare_for_plan(expected, actual, plan, operands)` (and a `_with_exactness` twin) which derives the band itself; the band is not an argument | nothing (additive, third number) | nothing, but the compliant call is now the shortest one, and a plan-bearing caller has no reason to hand-pick |
| **B. `Entitled` newtype** | `compare` takes `Entitled`, constructible only from `required_fidelity*` or `Entitled::declared(fidelity, reason: &'static str)`; the hand-picked path stays but costs a stated reason | the 51 call sites and 43 literals above and the two sibling repos (second number) | every call states either "derived" or "declared because ..."; greppable |
| **C. Ratchet test** | a census test that counts `Fidelity::Tolerant { .. }` literals in tests and fails when the count rises without an allowlist entry that carries a reason | nothing | new loose bands, not existing ones; works across the two crates only |
| **D. `compare` refuses a looser band** | `compare_for_plan` plus a check inside `compare` when a plan is supplied: a `Fidelity` looser than `required_fidelity` is `Err`, a tighter one passes | nothing if gated behind the new entry point | exactly the defect: a band wider than entitled |

## Recommendation

**A and D together, then C as the backstop for existing sites; B only if the count of hand-picked literals outside the
comparator's own tests reaches zero.** A gives a plan-bearing caller the right call at no cost; D makes a looser band
a typed failure at the one place that can know (`compare_for_plan(..., Some(fidelity))`: derive, and refuse a caller
band looser than the derivation); C stops the 43 existing in-repo literals from growing while they are migrated. B is the
cleanest end state and the most expensive, and the sibling counts (1 and 7 literals) say it would be cheap for them,
so it can follow once the in-repo tests have moved.

What this does **not** fix, stated so it is not read as a promise: `required_fidelity` is derived from the plan and the
CUDA accuracy table (`conformance.md` OPEN-1 item 2); a target with no accuracy table gets no band at all, so on such a
target `compare_for_plan` returns "no entitlement" and a caller must still choose. That is OPEN-4's per-target seam,
not this note's.

## Needs a ruling

Whether to build A + D + C (additive, one PR, third number) or to go straight to B (breaking, second number, touches
the two sibling repos). I recommend A + D + C.
