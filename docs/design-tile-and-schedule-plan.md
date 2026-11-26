<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
# Plan: a tile value in the IR (board 500 Q1 A) and a `Schedule` vocabulary beside the plan (Q2 A)

Status: **plan only, docs only** (2026-11-26). Nothing here is built except what a later PR names. Written by the Unpopped lane on pm-kernels' [TASK]s `q1-A-plan` and `q2-schedule`. Every row number cites `docs/ir-union-evidence/rescore-2026-11-25/<survey>.md` (file and row), read at `origin/main`. Anything not read from a row or a source is marked **UNVERIFIED**.

## 1. What was ruled, and what it supersedes

CireSnave, board item 500 (`CIRESNAVE-DECISIONS.md`, lines 170 to 174 hit the grep; the words below are as pm-kernels relayed them): **Q1** *"500: Q1: A."* (a tile or tensor value in the IR; element type, memory space, layout; slice, partition, retile, TiledMma and TiledCopy objects, tile-granular moves; breaking for every emitter; option C is not taken); **Q2** *"500: Q2: A"* (the IR carries a `Schedule` vocabulary beside the plan: tile shape hierarchy, stage count, persistent or stream-K work division, epilogue tile; the U8 S4 recipes-only ruling **for schedules** is superseded). Q1 A reaches 24 rows (CuTe 3, 15, 17, 22 to 27, 30, 31, 40, 47, 66; ThunderKittens 1, 2, 3, 5, 6, 38, 39; Mojo 38; Slang 25, 26). Q2 A reaches 10 (CUDA 13, 26, 78, 80; CUTLASS 14, 41, 52, 53, 54; ThunderKittens 59).

Superseded by this: the lane recommendation C in `docs/ir-union-rulings.md` section 2 (additive layout-on-allocation, value type later); the "recipes, not nodes" reading for schedules in ROADMAP U8 S4 and `docs/options-u8-copy-atoms.md` (Q5); the "parked" status of ROADMAP P9. **Not superseded:** the split in `options-u8-copy-atoms.md` that a *pattern's control flow* (a ring of stages, producer and consumer warps, a persistent loop body) is ordinary statements; what changes is that a schedule's **declared parameters** get a vocabulary the emitter can honour or decline.

## 2. The tile value (Q1)

**Type.** A tile is an IR value with `TileTy { elem: ElementKind, space: MemorySpace, layout: Layout }` (the layout algebra, ROADMAP U6, is done). A tile value is held in a local slot of a new kind (a `TileDecl` beside `LocalDecl`), or is an `Alloc`-backed view; the checker rules are: shape and layout consistent, the space admits the element, a slice stays inside its parent, a partition's pieces tile the parent. **Opaque tiles** (Apple `simdgroup_matrix`, CUDA WMMA: no per-lane mapping, no addressable element) are tiles whose layout leaf is `Opaque` (`docs/design-opaque-tile-mma.md` section 2): `TileLoad`, `TileMma`, `TileFill`, `TileStore` and `MemorySpace::Tile` are the **first consumer**.

**Operations (CuTe and ThunderKittens vocabulary):** slice, `local_tile`, per-thread partition (`partition_A/B/C`, `partition_S/D`), `retile`, layout conversion, tile-granular moves, `TiledMma` and `TiledCopy` objects. **Multi-result values:** a partition or `tl.split` yields several tiles: the form is a statement that binds several tile slots (the way `CollectiveTuple` binds several locals), not a tuple type; `docs/ir-union-rulings.md` section 7 (the Q1 table) shows every existing multi-result op is already such a statement.

## 3. The `Schedule` vocabulary (Q2)

A new type, **`ScheduleSpec`**, beside the plan-layer `Schedule` enum (`plan/types.rs:14`, which is `non_exhaustive`, `Copy` and about iteration of one op; the name differs on purpose). Fields come from the rows, not from memory: `tile: { cta: [M,N,K], warp, instr }` (CUTLASS 52, CUDA 80), `stages: StageCount { Auto | N }` and smem carveout (CUTLASS 53, CUDA 78, 26), `work: WorkDivision { DataParallel | Persistent{..} | StreamK{..} | SplitK{slices} }` (CUDA 13, CUTLASS 54, Triton 51 UNVERIFIED as a gated row), `epilogue: { tile, subtile, schedule }` (CUTLASS 41), `small_m: bool`-style specialisations (CUTLASS 14, UNVERIFIED how it reduces to fields), wave and register hints (ThunderKittens 59, `RegPin`/`WaveSchedule`). **Carrier:** a field on `KernelFn` (and, at the plan layer, on `KernelPlan` once Fuel/KISS contraction scope allows, section 7).

**The rule that makes it safe:** a schedule field an emitter does not honour is **declined by name**, never silently ignored. Unpopped's own oracle ignores a `ScheduleSpec` (a schedule changes how fast, never what a deterministic kernel computes), and says so; `unpopped-cpu-c` honours none and declines any field it is asked to apply with a typed error naming the field.

## 4. Phases

| Phase | What | PRs | Releases | Breaks whom |
|---|---|---|---|---|
| **P0** | This note, the supersession edits (section 8) | 1 (docs) | docs | nobody |
| **P1 additive** | **S1** `ScheduleSpec` + `check_schedule` + a fixtures entry, no carrier, oracle unchanged (**the first code PR**); **T1** `TileTy` and a layout field on allocations (serde default), checker rules; **T1b** slice / `local_tile` / partition / retile as pure vocabulary functions in `unpopped-vocab` with property tests (CuTe 22, 23, 24, 26, 27, 31, 47; TK 6, 38; Mojo 38) | 4 | 1 additive release (or two) | nobody (new types, serde defaults) |
| **P2 breaking, ONE release B1** | **T2** opaque-tile statements and `MemorySpace::Tile` (checker, oracle, cpu-c decline); **T3** tile-value statements (slice, partition, retile, multi-result bind); **T4** `TiledMma` / `TiledCopy` objects; **S2** the `KernelFn::schedule` carrier and the decline-by-name error | 8 to 9 | **one** breaking release | every emitter, once (section 5, 6) |
| **P3 additive** | conformance kernels for the rows (CuTe 3, 15, 17, 25, 30, 40, 66; TK 1, 2, 3, 5, 39; Slang 25, 26; CUDA 13, 26, 78; CUTLASS 14, 41, 53, 54; TK 59), each running on the oracle | 6 to 8 | 1 to 2 additive releases | nobody |
| **P4 docs** | the migration sheet kept current | 1 | docs | nobody |

**Refined estimate:** the union brief said 14 to 20 PRs for Q1; from the rows it is **15 to 18 for Q1** (T1, T1b, T2, T3, T4 and their kernels) and **5 to 6 for Q2** (S1, S2, and 3 to 4 kernels), **about 20 to 24 in all**, in **three** releases that matter to an emitter: one additive (P1), **one breaking (B1)**, one additive (P3). The ThunderKittens and CuTe estimates overlap, as the brief said.

## 5. One breaking release per emitter

Everything breaking is held back and **batched into B1**; P1 and P3 are additive by construction (new types, serde defaults, new `unpopped-vocab` functions, nothing that adds a variant to an exhaustive enum). B1 holds: the new `Stmt` variants (`TileLoad`, `TileMma`, `TileFill`, `TileStore`, the tile-value statements), `MemorySpace::Tile`, and the `KernelFn::schedule` field. **Recommendation (a design call, with its cost):** make `Stmt` and `MemorySpace` `#[non_exhaustive]` **in B1**, so every later addition is non-breaking, at the cost that downstream exhaustive matches lose the compiler's help and must carry a `_` arm that declines by name (the same call as for `ElementKind` on board 515). Without it, each later tile statement is a second breaking release for every emitter. This is yours (the PM's) to take to CireSnave.

## 6. Migration sheet (per emitter, in order)

Which target each emitter lowers is what I know from this session's messages and repo; the rest is **UNVERIFIED** and each lane corrects its row.

| Emitter (target, UNVERIFIED where marked) | P1 (additive) | B1 (breaking, the only one) | P3 (additive) |
|---|---|---|---|
| **vulkane** (Metal/MSL, Vulkan, GLSL, WGSL, HLSL) | nothing | match arms for the tile statements and `MemorySpace::Tile`; **MSL honours `TileMma` etc. (`simdgroup_matrix`, the first consumer)**, every other target declines by name; `schedule` field: decline by name | MSL conformance kernels |
| **baracuda** (`baracuda-cuda-emit`/`-parse`/`-seam`, CUDA) | nothing | tile statements: lower `TileMma` to WMMA (UNVERIFIED which sm) or decline by name; `schedule`: honour `tile`, `stages` for the CUTLASS-style kernels it builds (its choice), decline the rest by name | CuTe/TK-shaped kernels |
| **bonefish** (SYCL) | nothing | decline by name (its `Sycl::lower_kernel` already scans for declined statements first: its message of 2026-11-26) | none expected |
| **rockfish** (UNVERIFIED target) | nothing | match arms, decline by name | none expected |
| **fishstringer** (UNVERIFIED target) | nothing | match arms, decline by name | none expected |
| **openclownfish** (UNVERIFIED target) | nothing | match arms, decline by name | none expected |
| **fuel's seam** (`Backend`/`GeneratedKernel`, `element_kind_to_dtype`) | nothing | rebuild; no `ElementKind` change in this plan, so `element_kind_to_dtype` is untouched (UNVERIFIED which of fuel's matches name `Stmt`) | none |
| **unpopped-cpu-c**, **unpopped-convert** (ours) | nothing | decline by name; convert lifts no tile construct until P3 | the lifter rows |

The fixtures feature (`unpopped::fixtures`, 0.58.5) gains an entry per new statement in B1, so every emitter's decline test covers them the day they ship.

## 7. Order against Fuel/KISS contraction generalisation

KISS batch 1 (K and batch products, board 515) ships first (ordered 2026-11-25). The **kernel-layer** `ScheduleSpec` (S1, S2) and the tile value do not depend on it. The **plan-layer** carrier (a schedule on `KernelPlan`, and the three rows CUDA 44, 55, 81 on `Access::Contraction` generality) waits for KISS to publish batch 1 and for Fuel to say what it consumes; until then the plan layer keeps the unit variant `Schedule::Contraction`. So the plan-layer part of Q2 is a P3-or-later slice, after KISS, and does not block P1 or B1.

## 8. Text superseded, at its reach

Grep (words `recipe`, `parked by the PM 2026-10-24`, `P9`, `recipes, not nodes`; control: the same grep on the pre-edit text finds each): ROADMAP P9 (row 145) and the U8 row S4 sentence (row 192), `docs/options-u8-copy-atoms.md` (line 24 and Q5), `docs/ir-union-rulings.md` rows 1 and 2, and `docs/design-opaque-tile-mma.md` status. Each gets a dated "SUPERSEDED 2026-11-26 by board 500" note that names this plan; none is rewritten. The 2026-11-25 evidence tables are dated records and are not edited (their "DECISION: P9 parked" cells describe the state at that scoring).

## 9. Patent read, a named gate on each row it touches

No tile, partition, schedule or epilogue construct has been charted: `docs/PATENT-SWEEP.md` and `docs/PATENT-LANES.md` hold no search for `TiledMma`, tile partition, `TiledCopy`, stream-K, persistent kernels, epilogue fusion or wave scheduling (grep, 0 hits; the same grep finds the charted US12699602 and US20230315655A1), and the brief to CireSnave said the tile and partition constructs were "not searched". So **each phase is gated, row by row, and the gate opens when a sub-lane has read:**

| Gate | Constructs | Rows touched | Already charted (do not re-read) | Sub-lane reads needed first |
|---|---|---|---|---|
| G-T2 | opaque tile load/mma/store (WMMA, `simdgroup_matrix`) | design-opaque-tile-mma | none (the design note's own section 6 states a patent line) | NVIDIA WMMA and Apple simdgroup-matrix claim families |
| G-T3 | slice, partition, retile, layout conversion | CuTe 22 to 27, 31, 47; TK 6, 38; Mojo 38 | none | CuTe/CUTLASS layout-algebra and partition claims |
| G-T4 | `TiledMma`, `TiledCopy` | CuTe 30, 40 | TMEM/descriptor limbs (US12204897B2, US12710961B2: outside as read) cover `tcgen05`/descriptors only | `TiledCopy`/copy-atom claims; TMA-copy family is charted for descriptors only |
| G-S1 | stages, software pipelining | CUDA 26, 78; CUTLASS 53 | the mbarrier family (US11803380B2 and others) charted for `mbarrier` statements, not for a stage count | async-pipeline claims (UNVERIFIED which numbers) |
| G-S2 | persistent kernels, stream-K, CLC, work division | CUDA 13; CUTLASS 54; Triton 51 | none | scheduler / stream-K / cluster-launch-control claims |
| G-S3 | epilogue tile and fusion | CUTLASS 41 | none | epilogue-fusion claims |
| G-S4 | wave scheduling, register pinning | TK 59 | none | wave-scheduling claims |

Until a gate opens, the matching phase builds only what is vocabulary with no hardware-mechanism claim (S1's types, T1's types); the statements wait. pm-kernels routes the reads.

## 10. The first PR

**S1:** `ScheduleSpec`, `StageCount`, `WorkDivision`, `EpilogueSchedule`, `TileHierarchy` and `check_schedule` in a new module `unpopped::schedule`; no carrier on `KernelFn`, no oracle change (the oracle ignores a schedule and the module says so); a `schedule_fixtures()` entry (valid and refused specs, each refusal with its typed error); additive, no gate (vocabulary only, G-S1 to G-S4 govern the *statements* that honour a field, not the type). Release slot via pm-assist.

— lane: unpopped
