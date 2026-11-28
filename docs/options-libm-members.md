<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
# Options and design: the libm members of the next breaking release (ROADMAP U44 and the set)

Written 2026-11-28 by `unpopped-leaves` at pm-kernels' [TASK] `hypot-ruling-and-libm-census` and `libm-list-decisions` (PM ruling 2026-11-28: add `BinaryOp::Hypot` **and the whole
set** in the **next breaking release** with U55; `BinaryOp` and `UnaryOp` stay **exhaustive** (an exhaustive match is a compile-time check: each emitter decides on a new op); a member without a reference
implementation does **not** block the release: the oracle declines it by name until one exists). **Design only: no code.**

## What was measured, and what was read

- **IR today** (origin/main 0.65.1, `crates/unpopped/src/ir.rs`): `UnaryOp` 42 members, `BinaryOp` 24. The walker maps names to ops in one place, `crates/unpopped-convert/src/kernel_body/calls.rs` (the unary table at
  lines ~571-600, the binary table at ~629-633); **`rint`/`nearbyint` map to `UnaryOp::Round` (ties to even, cfamily.rs `rintf`), and `round`/`roundf` map to nothing**, so a C `round` declines today and can never lift to the wrong op
  (`grep` of every name-to-op map finds the one line; `typed.rs:173-177` names `nearbyint`/`round` for **cast** rounding modes, which is a different table).
- **Demand** (Baracuda corpus, 613 `.cu`/`.cuh` files at origin/main; control `expf` 155 lines in 34 files): `hypotf` 5 lines in 1 file, `hypot` 6 in 2, `roundf` 4 in 1; every other name below 0. No demand is needed (CireSnave 6.31).
- **Accuracy tiers** come from NVIDIA's public CUDA C++ Programming Guide, appendix "5.6. Floating-Point Computation" (page dated "Last updated on Oct 31, 2026"), tables 53-62, **read by a delegated fetch
  agent that copied tables 54-62 verbatim and could only partly read table 53; I did not read the page myself**. Table 62 says its ULP figures are the maximum observed difference from the correctly rounded result and come from testing, **not
  guarantees**. Where no number was read, **`contract.rs` gets no number** (the tier is absent, never a guess).
- **Second read, by a different method (pm-kernels, 2026-11-28):** the page's static HTML (247,341 bytes, "Last updated on Oct 31, 2026") downloaded and stripped of tags: **every tier below matches the raw rows**, and the table-53 zero-ULP sentence for `remainder` and `fdim` is readable there. The `contract.rs` row comments cite: "CUDA Programming Guide 5.6 (v13.5 page dated 2026-10-31), Table N; observed maxima, not guarantees; read raw HTML 2026-11-28", and carry the guide's sentence verbatim: the figures are "the maximum observed ... derived from extensive, though not exhaustive, testing. Therefore, they are not guaranteed".

## The set, with kind, plan-layer name, tier and oracle

`BinaryOp` additions (7) and `UnaryOp` additions (15). "Plan" = the name the plan layer (`unpopped-vocab/src/ops.rs`) already has. Tiers are the guide's maximum ULP, single (`f`) / double.

| Member | C / CUDA names | Plan name | Tier read (single / double) | Table (guide) | Oracle |
|---|---|---|---|---|---|
| `BinaryOp::Hypot` | `hypotf`/`hypot` | `BinaryKind::Hypot` | 3 / 2 | 55 Power Functions | computes (platform `hypot`; **not independent**, a libm) |
| `BinaryOp::Rhypot` | `rhypotf`/`rhypot` (device-only) | none | 2 / 1 | 62 | computes as `1/hypot` (two roundings: stated) |
| `BinaryOp::Ldexp` | `ldexp`, `scalbn`, `scalbln` | `BinaryKind::Ldexp` | **0 (exact)**: the guide's sentence "all the following functions have a maximum ULP error of zero" | 60 Floating-Point Manipulation | computes (exact scaling) |
| `BinaryOp::RemIeee` | `remainderf`/`remainder` (IEEE, **not** the floored plan `Remainder`) | none | **0 (table-wide statement)**: Table 53's lead sentence "All the following functions have a maximum ULP error of zero"; no per-row figure; the guide computes it as `x - rint(x/y)*y` | 53 "Basic Operations" | computes (exact: `x - n*y`, `n` = x/y rounded to even) |
| `BinaryOp::Fdim` | `fdimf`/`fdim` | none | **0 (table-wide statement)**, as `remainder` | 53 | computes (`max(x - y, 0)`, one rounding) |
| `BinaryOp::Jn`, `Yn` | `jnf`/`jn`, `ynf`/`yn` | none | `jn`: absolute error only at n = 128 (`2.2e-6` float / `5e-12` double); `yn`: float `ceil(2 + 2.5n)` ULP for `|x| < n`, else absolute `2.2e-6`; double absolute `5e-12` for `|x| > 1.5n` and **no double ULP figure for `|x| <= 1.5n`: not stated** | 62 | **NotYet** |
| `UnaryOp::J0`, `J1`, `Y0`, `Y1` | `j0f`... | none | 9 / 7 ULP for `|x| < 8`, absolute error beyond: ULP tier **only** on `|x| < 8` | 62 | **NotYet** |
| `UnaryOp::Erfinv` | `erfinvf` | `UnaryKind::Erfinv` | 2 / 5 | 62 | **NotYet** |
| `UnaryOp::Erfcinv` | `erfcinvf` | none | 4 / 6 | 62 | **NotYet** |
| `UnaryOp::Erfcx` | `erfcxf` | none | 4 / 4 | 62 | **NotYet** |
| `UnaryOp::Normcdf`, `Normcdfinv` | `normcdff`, `normcdfinvf` | none | 5 / 5 and 5 / 8 | 62 | `Normcdf` computes (`erfc`-based, tier as read); `Normcdfinv` **NotYet** |
| `UnaryOp::Tgamma` | `tgammaf` | none | 5 / 10 | 58 Error and Gamma | **NotYet** |
| `UnaryOp::Sinpi`, `Cospi` | `sinpif`, `cospif` | none | 1 / 2 | 62 | computes (exact reduction mod 2, then `sin`/`cos` of `pi*r` in the wider type) |
| `UnaryOp::CylBesselI0`, `I1` | `cyl_bessel_i0f`, `...i1f` (device-only) | none | 6 / 6 | 62 | **NotYet** |
| `UnaryOp::RoundAway` | `roundf`/`round` (half **away** from zero) | none (the plan `Round` is `round_even`) | **0 (exact)**: table 59 "Nearest Integer Floating-Point Operations", zero-ULP sentence | 59 | computes (`f32::round`, exact) |

Not members (separate rows, agreed): two-result functions (`sincos`, `sincospi`, `frexp`, `modf`, `remquo`), three-or-more-operand functions (`norm3d`, `norm4d`, `norm`, `rnorm*`), the integer-result casts (`lrint`, `lround`, `llrint`, `llround`).

**One design question the table forces.** `ldexp`/`scalbn`/`scalbln`, `jn` and `yn` take an **integer** second argument, and `BinaryOp` has one element type. Options: (a) a new mixed-type binary node, a larger change than this release; (b) the second operand is the same float type and **must hold an integer value**, the walker casting the integer argument (exact for `|n| <= 2^24` in `f32`, `2^53` in `f64`), the oracle refusing a non-integer exponent with a trap. **Recommendation: (b)**, with the exact domain stated in the member's doc and a decline by name when the walker cannot prove the cast exact. **This is an IR-shape decision for the PM (pm-kernels takes it); neither is chosen here.**

## What the release touches

- **Take chain.** Files naming `UnaryOp::{Lgamma,Cbrt,Erfc}` or `BinaryOp::{Atan2,Nextafter}` at each repo's origin/main (a proxy for "files with an exhaustive match on these enums"): Unpopped 19, baracuda 5, vulkane 5, openclownfish 5, fishstringer 3, rockfish 2, bonefish 1, fuel 0 = **40 files**, one chain for both enums (they ship together). Each emitter's arm is "spell it or decline by name"; the CUDA-only functions (`erfinv`, `rhypot`, Bessel, ...) have no portable C99 spelling, so **cpu-c declines them by name** (C99 `hypot`, `ldexp`, `remainder`, `fdim`, `round` it spells).
- **IR and contract.** `BinaryOp`/`UnaryOp` get the 22 members; `contract.rs` gets an accuracy row per member **only where a tier was read** (a row without one says so); each `NotYet` member makes `check_kernel`/the oracle refuse by name (`NotYet: <fn>`, the `DtypeNotComputed` pattern) until a reference exists, never a silent value.
- **Walker.** The names map to the members above and to nothing else; a test pins `round(2.5) == 3`, `rint(2.5) == 2`, `roundf(-2.5) == -3`, and that `roundf` can only lift to `RoundAway`; `cuCabsf`/`cuCabs` lift as `Hypot` of the two parts (the walker's decline by name goes away for them; `cuCdivf` and `cuCfmaf` still decline).
- **Release.** BREAKING (new variants of exhaustive enums): the next breaking release with U55 (`Stmt::ScheduleHint`, `docs/options-u55-scheduling-hints.md`); pre-1.0 the second number; the CHANGELOG names the arm each exhaustive match needs and the 40-file census as the migration size.
- **Patent.** None read. These are standard libm and CUDA math functions; the patents batch gets a one-line glance on `Hypot` (pm-kernels) and may take the set. Findings, not advice, not a clearance.

— lane: unpopped-leaves
