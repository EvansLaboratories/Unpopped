# Options note: the lift declines that wait for the plan and the oracle (ROADMAP U2 remainder)

Status: **D3 built in 0.42.6; D4 dropped (PM, 2026-11-03: KISS wins, `RowSort` stays stable-only); D1 and D2 are KISS questions (filed as
ThinkersJournal/KISS#766 and #767) built on the recommended reading; D5 open; D6 waits for a consumer.** Asked by the PM (`[TASK]` 2026-11-02:
"(b) U2 remainder and (c) U1 remainder follow" U5). Read at `origin/main` `72b8e37` (0.42.3), updated at `04b0bce` (0.42.7). The ROADMAP U2 row lists five forms the kernel-layer lift of an `OpDef` declines because the plan
or the oracle does not define them yet ("the lift has nothing to agree with until the plan and the oracle define them"), and an
effects model that is "computed when a consumer asks". This note says what each should mean, in the order I would build them.

## What is declined, and where

| id | form | where it stops today | what it needs |
|---|---|---|---|
| D1 | `Mean` in an `Access::Scan` | `plan::validate_scan` asserts "Scan combine Mean is not a monoid"; `oracle` panics in `eval_scan` and `scan_identity` | a meaning for a running mean |
| D2 | `Prod` in an `Access::Window` | `plan::validate_window` rejects it ("not a pool"); `oracle` panics in `eval_window` | a padding rule for a product |
| D3 | a `Prod` stage in `Access::RowReduce` | **built in 0.42.6** (the launched lift still declines it) | - |
| D4 | an unstable `Access::RowSort` | `plan::validate_row_sort` requires `stable == true` | **dropped**: KISS-OPS-6.11-0007 makes `sort_network` stable (ties to the lower original index), so an unstable sort is not a KISS op |
| D5 | a `Scan` or a `Window` over an axis that is not the innermost | `validate_scan`/`validate_window` reject `axis != rank - 1` ("a deferred follow-up") | strided iteration |
| D6 | the effects model | `kernel.rs` header: "computed when a consumer asks" | a consumer |

Each of D1 to D5 turns an `assert!` in the gate into an accepted form, so each is **additive** (nothing downstream matches on the
gate). Every one lifts a refusal with its oracle test, an independent reference written in Rust, and seeded mutants, as the U4 and U5
slices did.

## Proposed meanings

**D3, a `Prod` stage.** The stage multiplies the per-element `pre` values along the row, starting from `1` (the multiplicative
identity), pass-through finalize, no divisor. `RowReduce` is float-only in v1, so this is `F32`/`F64` (`f32` rounds once at the store,
as the other stages do). No ruling needed.

**D2, a `Prod` window.** KISS has `avg_pool` and `max_pool` only, so a product pool is an **Unpopped extension, not KISS** (proposed as
ThinkersJournal/KISS#767; that issue first claimed the all-out-of-bounds value was unstated, which was wrong and is retracted: KISS-OPS-6.13-0011
makes a window with no in-bounds tap inexpressible, and Unpopped already declines it in `window_taps`). So no ruling is needed: `prod_pool` is the
product of the in-bounds taps, starting from the multiplicative identity `1` (a skipped tap contributes `1`, as in KISS-OPS-6.11-0002); integers
wrap at their width (as `Prod` reductions do); `Mean`-style divisors do not apply.

**D1, a `Mean` scan.** KISS is closed here: `prefix_scan` takes the monoids `{sum, prod, max, min}` (KISS-OPS-6.11-0003), and a running mean is not one,
so the question of accept, decline or decompose is KISS#766; until it is answered the reading below is Unpopped's. The running mean: inclusive lane `j` is `(pre(x_0) + ... + pre(x_j)) / (j + 1)`; exclusive lane `j` is the sum
of the lanes before it divided by `j`, so lane 0 is `0/0 = NaN` (the same as a mean over an empty run). Float only; an integer `Mean`
stays refused, as an integer `Mean` reduction is (PM ruling 2026-11-01: no rounding rule for the quotient). The scan's `post` applies
after the division. Ruling wanted: the exclusive lane 0 (recommended: `NaN`, consistent with the empty mean; the alternative `0` hides
that there is no mean).

**D4, an unstable `RowSort`: dropped.** KISS-OPS-6.11-0007 requires a stable permutation under the total order (NaN greatest, ties by the lower
original index), so the stable-only gate is KISS's own rule and there is nothing to relax.

**D5, a non-innermost axis.** Generalize the scanned or pooled axis to any `axis < rank`: the oracle iterates the outer extents,
the axis, and the inner extents with the operand's strides; the lift emits the corresponding loop nest; `shape` and the key already
carry the axis. Scan first (a strided prefix, no new numerics), then Window (new geometry: the output extent shrinks on that axis, and
the other axes pass through). Larger than D1 to D4; the emitters (`unpopped-cpu-c`) follow the lift.

**D6, the effects model.** An analysis `effects(&KernelFn) -> Effects` (buffers read and written, atomics, barriers, collectives, calls)
is a pure walk over the statements, cheap to write and nobody calls it. Recommend leaving it until a consumer (fusion, scheduling, a
race-free proof) asks, as the ROADMAP row says, and recording that the walk is mechanical so the cost of waiting is low.

## Order and cost

D3, D2, D1 (small, each ~ the size of the `MaxNum` slice: gate, oracle, lift, a C differential), then D5 scan, then D5 window. Versions are 0.42.x, one per slice.

## Needs a ruling

1. D1's exclusive lane 0: `NaN` (recommended) or `0`.
2. D6: wait for a consumer (recommended).

D3 is built. D2 needs no ruling and is built next. D1 is built on the recommended reading unless the PM rules or KISS#766 answers differently. D5 needs none.
