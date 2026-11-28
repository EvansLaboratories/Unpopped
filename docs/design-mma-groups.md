# Design note: `groups` on `MmaAtom` for f16 `m8n8k4` (U70; no code)

Status: **A accepted by `unpopped` (owner of ROADMAP U70) 2026-11-28.** Docs only; the build follows in its own PR. Sources: PTX ISA 9.5 section 9.7.16.5.1 (pp. 531-536), text formulas read; Figures 49-55 are images and are cited by figure, not read. Single source. Patent column: this note describes a layout rule, applies 519 B and US12314717B2 as in `docs/mma-atom-coverage.md`; not a clearance; no attorney; findings not advice.

## 1. What the instruction is

`mma.sync.aligned.m8n8k4` with `.f16` inputs (section 9.7.16.5.1) makes **one warp compute four independent `8x8x4` MMAs**. Each computation uses two **quads** of lanes (a quad is 4 consecutive lanes):

| computation | low quad (lanes) | high quad (lanes) |
|---|---|---|
| 1 | 0-3 | 16-19 |
| 2 | 4-7 | 20-23 |
| 3 | 8-11 | 24-27 |
| 4 | 12-15 | 28-31 |

So each computation is an **8-lane atom** (same shape as every other atom here: `lanes = 8`) and the warp hosts **4 of them at once**, side by side, sharing nothing: no matrix element, no scale, no register crosses a group.

Within one group, with `l = laneid % 4 + 4 * (laneid >= 16)` (0..7), the text formulas are the ordinary 8-lane thread-value layout:

- A (row major, Figure 49): row `l`, col `i`, `i = 0..3` (4 `f16` values per lane). Column major (Figure 50): row `i`, col `l % 4`, plus 4 rows on the high quad.
- B (row major, Figure 51): row `laneid % 4`, col `i` (+4 on the high quad). Column major (Figure 52): row `i`, col as above.
- C/D `.f16` (Figure 53): row `l`, col `i`, `i = 0..7`. `.f32` (Figures 54-55): row `X` (+4 high), `X = (laneid & 1) + (i & 2)`; col `(i & 4) + (laneid & 2) + (i & 1)`.

The formulas use only `laneid % 4` and the high/low half, so **all four groups share one layout**; Figures 54 and 55 are split only to draw computations 1-2 and 3-4 (unread; the text formula is what this note relies on).

## 2. Why no atom ships today

`MmaAtom` states one group of `lanes` lanes and the launch must have a subgroup of exactly that many (`kernel_check/mma.rs:32`; the oracle refuses otherwise, `oracle/kernel_mma.rs:154`). An 8-lane atom in a 32-lane launch is refused; a 32-lane atom would have to carry four independent matrices in one flat layout (its mnk would be a lie: the bijection check `MmaAtom::new` makes would hold only by inventing a 32x8x4 matrix that does not exist). Measured (see `docs/mma-atom-coverage.md`): a hand-built 8-lane atom passes `MmaAtom::new` and `check_kernel` except the launch rule; there is no shape rule. The decline is "no constructor ships", not a rule.

## 3. Options

**A. `groups` field with an explicit lane decomposition (recommended).** `#[non_exhaustive]`, additive: `groups: u32` (1 for every existing atom) and a flat `(shape, stride)` map from `(group, local lane)` to the subgroup lane, in the same form the layouts use. For f16 m8n8k4: lane = `lo + 4 * group + 16 * hi`, so shape `(4, 4, 2)`, stride `(1, 4, 16)` read as `(lo, group, hi)`, with `local = lo + 4 * hi`. The atom keeps `lanes = 8` and all layouts stay 8-lane layouts, checked by the existing bijection. Data, not code: other multi-group instructions (any future one) are a new map, not a new rule.

**B. A rule keyed on the atom (quad-pair special case in the oracle).** Rejected: it hard-codes one instruction in `check_kernel` and the oracle, and an emitter cannot read it from data.

**C. Keep declining.** Costs only the shape; a consumer of Volta-style `m8n8k4` sees a named refusal. Acceptable if `unpopped` judges the shape too old to schedule, but "no consumer asks" is not a reason (model reason: the instruction exists in the PTX tables for sm_70+ and the IR is the full union).

## 4. What must change under A

1. `check_mma` (`kernel_check/mma.rs`): the launch subgroup must be exactly `lanes * groups`, and every lane of the subgroup must execute the `Stmt::Mma` together (the rendezvous already requires the whole subgroup).
2. Oracle (`oracle/kernel_mma.rs::compute`): accept `lanes.len() == lanes * groups`; split the contributions by the map into `groups` sets of `lanes` lanes; run the existing per-group compute on each; scatter each D back to the lanes of its group. No cross-group read: a test with different data in each group pins it.
3. `MmaAtom::new`: the map must be a bijection onto `0..lanes*groups`; `groups = 1` and the identity map for every existing constructor (no behaviour change; the tests for every shipped atom stay as they are).
4. Constructors: `nvidia_m8n8k4_f16` with `.row/.col` per operand and `.f16/.f32` accumulator (A and B each row or col, so the layout choice is a parameter, as for `m16n8k8` bf16 vs tf32) and tests per PTX formula in the `(row, col)` style of `tests/mma_atoms_*.rs`.
5. Emitters: an emitter sees `lanes`, `groups` and the map as data; no emitter change is needed to keep declining (the field is `None`-like for existing atoms).

## 5. Decided (`unpopped`, 2026-11-28)

- Option A: `groups: u32` plus a `(group, local lane) -> lane` flat map, additive, `MmaAtom` stays `non_exhaustive`.
- The map lives on the **atom** (it is per instruction); the launch states only the subgroup size, which `check_mma` requires to equal `lanes * groups`.
- Name: `groups`.
- **`MmaAtom` stays free-form** (no validated shape table, which would duplicate the constructors). The constructors carry the PTX meaning; a hand-built atom is a model, and `check_mma`'s launch rule plus `MmaAtom::new`'s bijection check are the only guards. (This answers the probe in ROADMAP U70: `MmaAtom::new` admits a hand-built 8-lane `8x8x4` f16 atom and `check_kernel` runs it over an 8-lane subgroup as one MMA, which is not what `mma.m8n8k4` does.)
- Build order for the follow-up PR: `groups = 1` plus the identity map on every shipped atom (no behaviour change, proved by the existing atom tests); the oracle split/scatter with a cross-group isolation test (different data per group); the `m8n8k4` constructors with `.row`/`.col` per operand; the 519 B / US12314717B2 wording on the atom as on the others. Full gate.

— lane: unpopped-mma
