# Options note: a launched float Scatter add (`AtomicAdd` into a float destination)

Status: **options, nothing built.** Asked by the PM (2026-11-01) in the E1d-2 style. Today `lift_scatter_opdef_launched` declines a float destination
(`UnsupportedForm`); the serial `lift_scatter_opdef` is unchanged. Integer `AtomicAdd`/`Max`/`Min` are launched and bit-exact (0.40.0).

## The fact that forces a ruling

A float add is not associative. When `c` sources name one destination cell, the launched sum depends on the order the threads reach the cell. Two threads
never *race* on it (atomic against atomic is not a `DataRace`, `docs/ir-kernel-layer.md` section 5a); the schedule changes the **value**, not the memory
safety. So the three questions are: what order the IR promises, what order the interpreter oracle uses, and how a result is compared.

## Options

| | IR promises | Interpreter oracle | Comparison | Cost |
|---|---|---|---|---|
| **A (PM's lean)** | Nothing: float accumulation order is unspecified | The ascending-source-index order (what the serial scatter does, and `oracle::evaluate`) | A reassociation band per destination cell | A float `AtomicRmw::Add` statement (new form, the checker admits a float `Add` only) |
| **B** | A fixed order (ascending source index) | Same | Bit-exact | No atomics: sort sources by target, then a segmented scan/reduce per target. A different lift with a different kernel shape; slower, deterministic |
| **C** | Nothing, but a declared per-cell bound | Ascending index | `gamma_(c-1) * sum(abs(x_i)) + half ulp` per cell, `c` = that cell's hit count | As A, plus the bound is computed from the index data at compare time |
| **D** | Not offered | n/a | n/a | Float scatter stays serial-only |

## Does A's band agree with KISS-CONTRACT-6.8-0012?

The KISS band is `k(S,A) * N * eps_A * |max partial sum|` (KISS repo: `spec/contract.md`; `docs/options-r5-18-contract-6-8-0012-wording.md` is its wording note;
`k(S,A)` is uncalibrated, R5.13). I read the form from KISS's `conformance/UNBACKED.tsv` and ROADMAP, not the full clause. Read against a scatter:

- **Same shape.** One band per output element, proportional to the number of summands and the accumulator's epsilon. For a scatter cell the summand count `N`
  is the cell's hit count `c`, not the source size. A cell hit once is exact; a cell no source names keeps its initial bytes. A band written with the
  source size `N` would be far too loose for a sparse scatter.
- **One difference to resolve.** `|max partial sum|` is itself order-dependent (it is the largest partial sum along *some* order). The bound this repo uses for
  E1d-2, `gamma_(n-1) * sum(abs(x_i))`, does not depend on the order and dominates it, so a reader that checks the **sum of absolutes** is safe for every
  schedule, while one that uses the reference order's max partial sum is tight but could reject a valid reordered result. Recommend the band's `|max partial sum|`
  be read as an upper bound over every order, which is the sum of absolutes.
- **Initial contents.** The destination starts with caller-owned bytes (the serial kernel does not zero it); they are one more summand in the band.

So A and C are the same rule: A names the band, C says how a conformance test computes it for a scatter (per cell, from `c`). I recommend **A with C's
per-cell count as the definition of `N`**.

## What changes bytes

Nothing already published. Under A/C a float launched scatter that is *accepted* would produce bytes that differ from the serial kernel within the band; no
existing lift or test changes. If CireSnave must rule (it admits a launched kernel whose bytes are not the serial bytes), B is the only option that keeps
bit-exactness; it is the larger piece of work.

## DataRace rule under A/C

Unchanged: atomic against atomic never races; atomic against a plain access to the same element races unless a barrier orders them. The interpreter oracle's
single schedule yields one valid order, so a conformance test must compare to the reference **inside the band** and never assert equality with the serial bytes
(an equality assertion would pass today by luck of the schedule and fail on a device).

## Recommendation

A, with `N` = the cell's hit count and the band read as an upper bound over every order. Build it only after the PM (or CireSnave) rules; the lift change is
small once the float `Add` form exists (the checker admits it and the interpreter combines at the element type).
