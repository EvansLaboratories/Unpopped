# The IR union, re-scored: where the seven surveys stand now

## Third measurement (2026-11-25, `origin/main` `b21af35c`, Unpopped 0.56.12)

Asked by CireSnave via the PM on 2026-11-25 (is the full-union IR on schedule). **A dated measurement; docs only, no code changed.** His standard (2026-11-11): *"if Unpopped's IR does not support something one of those kernel languages is capable of then that IR is not yet complete to my standards"*. He set no deadline; his hope (2026-11-25): *"I would be happy if we can have it complete by new years day"*. The second measurement (2026-11-15, `78c7d38e`) and the first (2026-11-03, `a634b09`) are kept below, unedited.

**Union coverage index (Yes + half Partial, over applicable rows): 35% (11-03) -> 50% (11-15) -> 57% (11-25)** (Yes 149 -> 162, Partial 208 -> 251, No 150 -> 94, n/a 12; the same 519 rows, 507 applicable; 56.7%, and 56% to 57% across the judgement calls below). Yes alone: 29.4% -> 32.0%. The pace fell: 1.25 index points a day over 11-03 to 11-15, 0.68 a day over 11-15 to 11-25, because 11-22 to 11-25 went to the U19 frontend (no IR node, so no row moved).

| Survey | Rows | Yes | Partial | No | n/a | Rows changed |
|---|---|---|---|---|---|---|
| CUDA C++ | 95 | 27 -> **30** | 40 -> **51** | 28 -> **14** | 0 | 16 (15 up, 1 down) |
| CuTe / CuTe DSL | 66 | 10 -> **11** | 35 -> **41** | 20 -> **13** | 1 | 8 |
| CUTLASS | 96 | 25 -> **26** | 40 -> **44** | 29 -> **24** | 2 | 6 |
| Mojo / MAX | 73 | 26 -> **30** | 27 -> **33** | 17 -> **7** | 3 | 11 |
| Slang | 61 | 22 -> **22** | 21 -> **24** | 15 -> **12** | 3 | 6 (4 up, 2 down) |
| ThunderKittens | 58 | 15 -> **19** | 24 -> **29** | 17 -> **8** | 2 | 11 |
| Triton | 70 | 24 -> **24** | 21 -> **29** | 24 -> **16** | 1 | 8 |
| **All seven** | 519 | 149 -> **162** | 208 -> **251** | 150 -> **94** | 12 | 66 (63 up, 3 down) |

Per-row evidence for every changed row, and the blocker and PR estimate for every row that is not yet a full Yes, are in `docs/ir-union-evidence/rescore-2026-11-25/<survey>.md`. Method as in the second measurement (a code symbol plus a named test; "absent" claims carry a positive control), with two changes the PM asked for: **two status columns per row** (IR expresses it / the oracle runs it, a headline Yes needs both), and **a downward check of every row that was Yes**.

### What moved

28 commits landed since 78c7d38e. The ones that moved rows: `mbarrier` statements, `Stmt::CopyBulk`, `MatrixLoad`/`MatrixStore` (`ldmatrix`/`stmatrix`), `Wgmma` + `Scope::Warpgroup`, `MemorySpace::Tensor` + `Tcgen05Mma`/`Commit`/`Ld`, named barriers, `GridDepWait`/`GridDepLaunchDependents` (PDL), the `For` unroll hint and `CopyAsync` `valid`/`cache`/`l2_prefetch`, 8- and 16-lane packed values, launch grid from scalar arguments (U22), `Place::Raw`/`AllocRaw` (byte-offset places), `ParamKind::Struct`, the attention backward kernels. **The U19 C-family frontend (#385 to #392) adds no IR node and moved no row.** Moves are mostly No -> Partial (the statement exists; the surrounding feature, the descriptor, the second operand kind, or the conformance kernel does not).

### Three rows went DOWN, and rows the earlier passes had wrong

- **CUDA 60** (block-scaled `F8E8M0`): Yes -> Partial. `F8E8M0` is a scale type the plan gate refuses and the oracle's `raw_to_f64` panics on; only e4m3 and e5m2 run. Overstated at a634b09 and 78c7d38e.
- **Slang 31** (64-bit atomics): Yes -> Partial. The checker admits `I64`/`U64` and the oracle is generic, but no test executes a 64-bit atomic (the `I64` hits in `kernel_atomic_{rmw,cas,fetch}.rs` are refusals; the same grep finds the 18 `I32` uses in the same file). Checked by hand for this note.
- **Slang 51** (rounding mode per op): Yes -> Partial. `Qualified` runs in the oracle only at f32, and #388 made `check_kernel` refuse it at f64 and the half types by name; the old Yes overreached for `__dadd_rn`-style operations (ROADMAP U43).
- Stale evidence restated, status unchanged: CUDA 4, 50, 59, 76 (59's "no vector arithmetic beyond `Select`" was wrong: lane-wise packed operators exist since 0.27.2); CUTLASS 54, 65, 70, 83, 88, 92; CuTe 14 (`View::Layout` of 2026-10-28 predates a634b09 and was missed); Mojo and others cite `decode_mx` and some tests under a wrong crate or path (the files list the corrections). CUDA 95 is a correction that predates a634b09 (`capability.rs::cuda_capabilities` holds the shared-memory table, nothing consults it: Partial).
- **IR yes, oracle not fully:** CUTLASS 61 and Triton 31 (PDL: the oracle runs `GridDepWait` as a no-op), CUDA 60 (above), CuTe 38 (the checker reads right, no test with A != B). All four are counted Partial, not Yes.

### What this number does NOT mean (additions to the second measurement's list)

- **Nothing was built or run** (read-only). A statement-level check of the 192 `file.rs::test_name` pairs the seven files cite found 191 present at b21af35c (the 192nd is the module name `tests`, a mis-quote) and every cited test file present; presence is not proof the test asserts the claim. One downward move and one test family (`kernel_matrix_store.rs`) were opened by hand.
- **Oracle semantics for three families have no second source in the repository**: the `mbarrier` phase/parity rule (`docs/options-u8-copy-atoms.md`), the `stmatrix` row rule (ROADMAP U8b says by symmetry), and the `wgmma`/`tcgen05` fragment layouts. The oracle runs them, but a Yes on those rows means "the oracle executes our reading", not "matches hardware".
- **Judgement calls that move the headline**: CUTLASS 61 (counted Partial; Yes would be 163/250), ThunderKittens 27 and CUDA 23 (`mbarrier`; Yes on a lenient bar, no `inval`, no non-blocking test-wait, no returned token: strict gives -2 Yes), CuTe 58 and Slang 6 and 12 (generous Partials; strict gives -3 Partial, +3 No). Strict: Yes 160, Partial 250, index 56.2%. Lenient: 57% at most. The sums of the seven agents' PR estimates count a capability once per survey it appears in.
- **Patent rows** are in the "blocked" list below as PATENT. The blocker is the PATENT-SWEEP.md section named in each survey file ("outside as read", NOT CLEARED, unread claims, or a re-read trigger); no capability is called clear, and this is research, not an opinion of counsel. Only sections relevant to each row were read, not all 106 KB.
- **Equal weight per row** and **a Partial counts half**, as before. 32 rows are OUTSCOPE (host-side or frontend; listed per survey); if the PM rules them n/a rather than gaps the index rises (about 60% at today's counts), but that is a ruling, not a measurement.

### Where the work stands: the 345 rows that are not a full Yes, by what blocks them

| Blocker | Rows (summed over surveys) | Rows that can start today? |
|---|---|---|
| READY (Unpopped lane, no external wait) | 149 | yes |
| DECISION (CireSnave / PM ruling, or an unanswered options note) | 59 | no |
| TRIGGER (ROADMAP: scheduled, waits for a named consumer or second source) | 44 | no |
| PATENT (PATENT-SWEEP.md section NOT CLEARED, unread or re-read trigger) | 38 | research first |
| LANE (another lane or repo: Fuel, KISS, Baracuda, fishstringer, Vulkane) | 23 | no |
| OUTSCOPE (host-side or frontend; needs a "not IR" ruling) | 32 | n/a |

Every blocked row, by survey (numbers are the survey's row numbers; the survey file names the missing piece and the PATENT-SWEEP section or ROADMAP row for each):

- **PATENT:** CUDA 7, 40, 41, 50, 76; CuTe 33, 34, 43, 44, 45, 46, 49, 52, 64; CUTLASS 49, 59, 62, 63, 64, 67, 70; Mojo 6, 10, 27, 28, 30, 33, 34, 35, 55; Slang 6, 35; ThunderKittens 24; Triton 8, 19, 57, 58, 59.
- **DECISION:** CUDA 13, 17, 26, 44, 55, 75, 78, 80, 81, 90, 91, 93; CuTe 3, 15, 17, 22, 23, 24, 25, 26, 27, 30, 31, 40, 47, 66; CUTLASS 14, 20, 21, 41, 52, 53, 54, 72, 76, 78, 79; Mojo 17, 38; Slang 19, 25, 26, 36, 49; ThunderKittens 1, 2, 3, 5, 6, 29, 30, 38, 39, 54, 59; Triton 5, 36, 52, 65.
- **LANE:** CUDA 56, 61, 62, 64, 65; CuTe 39; CUTLASS 3, 4, 5, 15, 23, 31, 35, 38, 51; Mojo 44, 48, 71, 72; Slang 50, 52; ThunderKittens 35, 58.
- **TRIGGER:** CUDA 4, 15, 25, 38, 39, 42, 46, 47, 48, 59, 79; CuTe 20, 36, 42, 48, 62; CUTLASS 25, 61, 83; Mojo 49, 52, 53, 54; Slang 5, 8, 9, 30; ThunderKittens 9, 22, 23, 25, 26, 48, 55; Triton 2, 10, 11, 12, 13, 18, 30, 31, 56, 60.
- **OUTSCOPE:** CUDA 8, 14, 16, 27, 92; CuTe 63; Mojo 29, 58, 59, 60, 69, 73, 74; Slang 46, 47, 48, 54, 57; ThunderKittens 49; Triton 3, 44, 47, 48, 49, 61, 63, 68, 69, 70; CUTLASS 3 rows (see its file).

The single largest DECISION is one unmade ruling: **whether the IR gets a tensor/tile value type or CuTe, Triton and ThunderKittens tensors stay recipes over `Alloc` plus `Layout`.** It gates 14 CuTe rows, about 7 ThunderKittens rows and parts of CUDA, Mojo and Triton, and no ROADMAP row covers it; ROADMAP's own stance is "recipes, not nodes", so a ruling of "recipe" would turn those Partials into declined-by-ruling.

### Burn-down by implementation theme (PRs are ~150k-diff-character PRs; one theme counted once across surveys)

The seven agents' sums add to 305 to 393 PRs because a capability appears in several surveys; clustering their themes gives **about 135 to 179 PRs** for the whole union. This is my clustering and my judgement, not a measurement.

| # | Theme | Surveys | Blocker | PRs | Fits 2027-01-01? |
|---|---|---|---|---|---|
| 1 | Small additive hooks and leaves: warp masks, `activemask`/`match`/`elect`, cluster rank, `setmaxnreg`, cache hints on plain loads and stores, atomic `Inc`/`Dec` and weak orders, proxy fences, `mbarrier` drop/test-wait, id leaves, bit ops | all seven | READY | 14-18 | yes, first three weeks; one breaking batch (U36 ruling) |
| 2 | Numerics gaps: U43 (F64 and half `Qualified`), U37 exact-spelling hint, U42 F64 ratings, fast-math, `cvt.pack`, bf16x9/x6, popcount | CUDA, CUTLASS, Triton | READY (ROADMAP U37, U42, U43) | 8-10 | yes |
| 3 | MMA atom data and flags: wmma shapes, tf32 and fp8 atoms, mixed-dtype contraction, block-scale with fp8 scales, Metal 8x8, A != B tests | CUDA, CuTe, Mojo, TK, Triton | READY (U7 S3); some flags need a second lane-table source (TRIGGER) | 8-10 | mostly |
| 4 | Kernel-layer conformance kernels (grouped/ptr-array GEMM, stream-K, persistent, epilogue fusion, conv family, attention, pipeline and role recipes) | CUTLASS, CuTe, TK, Mojo | READY (U9 scope A style, U8 S4) | 32-40 | about half at best |
| 5 | Layout vocabulary completion and the first kernel use of `Swizzle` | CuTe, Mojo, CUDA | READY | 4-5 | yes |
| 6 | Tensor-map / TMA descriptor operand kind and copies, `cp.async.bulk.tensor`, multicast | CUDA, CuTe, CUTLASS, Mojo, TK, Triton | TRIGGER (U8c: Baracuda's sm_90 emitter asks) plus PATENT re-read of US12645455B2 and the US11803380B2 review | 10-14 | no, unless the trigger fires; the claim reads can start now |
| 7 | `wgmma`/`tcgen05` completion: descriptors, swizzle, TMEM alloc, CTA pairs, other kinds | CUDA, CuTe, CUTLASS, Mojo, TK, Triton | PATENT (U7 S4 and S5 re-read before building) | 10-14 | no: reads first, at best late January |
| 8 | Pointer values (U9c, trigger already met) and vector-typed access | CUDA, CuTe, CUTLASS, Mojo, Slang, Triton | PATENT (U9c, U15 claims unread) | 5-8 | only after the re-read |
| 9 | Sync extras, compile-time evaluation, vector lowering | CuTe, Mojo, Triton | PATENT | 5-8 | no |
| 10 | Tensor / tile value type, partitioning | CuTe, TK, Mojo, Triton, CUDA | DECISION (one ruling) | 14-20 if "IR value", about 0 if "recipe" | no, until ruled |
| 11 | Schedule language (P9), persistent and stream-K as a plan feature | CUTLASS, CUDA, TK | DECISION (parked) | 5-8 | no |
| 12 | D5 formats, `Fp8FastAccum`, tf32 `ElementKind`, `assume` facts | CUTLASS, TK, Triton | DECISION (CireSnave board 290) | 4-7 | no |
| 13 | Recursion, device launch, special registers, device effects | CUDA | DECISION (scope) | about 7 | no |
| 14 | Plan-layer contraction generalisation (D2) | CUTLASS, CuTe, CUDA, TK | LANE (Fuel, KISS) | 10-12 | not ours to schedule |
| 15 | KISS element rows, AMD/Metal breadth | CUDA, Mojo | LANE | about 6 | not ours to schedule |
| 16 | Host-side and frontend rows | all | OUTSCOPE | 0 | needs a "not IR" ruling |

### The honest date

**100% by 2027-01-01 is not reachable, and it should not be promised.** The reason is not the lane's speed: 164 of the 345 open rows (DECISION 59, TRIGGER 44, PATENT 38, LANE 23; themes 6 to 15) cannot start until someone other than the Unpopped lane acts (patent claim reads and a PM or CireSnave ruling on the PATENT rows; a Baracuda sm_90 emitter for the descriptor family; one tile-value ruling; the D5 board item; Fuel and KISS for D2).

Calibration: over 11-15 to 11-24 the lane merged about 15 IR-feature PRs (11-22 to 11-25 went to U19), and the second measurement moved 0.45 index points per such PR (13 Yes and 43 Partial moves over 15 PRs, 6.8 points). The READY themes (1 to 5) are 66 to 83 PRs; the rest of the union is 135 to 179 in all. 37 days remain to 2027-01-01.

| Scenario (assumption) | PRs by 01-01 | Index on 01-01 | READY themes (66-83 PRs) done | Whole union (135-179 PRs) done |
|---|---|---|---|---|
| Today's pace: one PR a day, 80% of days on union rows | about 30 | about 68% to 70% | 2027-02-16 to 2027-03-09 | 2027-05-13 to 2027-07-07 |
| Best case: catch-up releases keep giving two PRs a day, 80% on union rows | about 59 | about 78% to 83% | 2027-01-05 to 2027-01-16 | 2027-02-17 to 2027-03-17 |

The index figures use the 0.45 points per PR of the last pass and fall as the cheap rows run out, so they are ceilings. The "whole union" dates assume every blocker clears on the first day, so they are lower bounds. **To finish by 2027-01-01 the following would have to change**: (1) the PM rules on the tile-value type, P9, D5 and the 32 OUTSCOPE rows this week; (2) the patent reads for U7 S4/S5, U8c and U9c/U15 are done as research now, in parallel with building, so they do not wait behind it; (3) Baracuda states whether and when an sm_90 emitter will ask for the descriptor family; (4) two PRs a day holds for five weeks with no other work (U19 follow-ups, U39 refactor, publish prep and gate answers also draw on it, which the 80% factor only partly covers). Even then the union stops near the READY themes plus whatever the rulings unlock.

### Unblocked next rows, by order (themes 1, 2, 5, 3, 4)

Take 1, 2 and 5 first (cheap, additive, no patent area, about 26 to 33 PRs between them), then 3, then 4 (the long tail; each conformance kernel closes one to three Partials). Breaking statements batch into one release. Start the claim reads for themes 6 to 9 now as research; raise the tile-value ruling to the PM as one `[ASK]` with its options.


## Second measurement (2026-11-15, `origin/main` `78c7d38e`, Unpopped 0.47.3)

Asked by CireSnave via the PM on 2026-11-15 ("how close are we to the full union"). **A dated measurement; docs only, no code changed.** The first measurement (2026-11-03, `a634b09`, 0.42.14) is kept below, unedited, as the baseline.

**Union coverage index (Yes + half Partial, over applicable rows): 35% -> 50%** (Yes 93 -> 149, Partial 168 -> 208, No 246 -> 150, n/a 12; of the same 519 rows the first measurement counted).

| Survey | Rows | Yes | Partial | No | n/a | Rows changed |
|---|---|---|---|---|---|---|
| CUDA C++ | 95 | 18 -> **27** | 30 -> **40** | 47 -> **28** | 0 | 24 |
| CuTe / CuTe DSL | 66 | 7 -> **10** | 29 -> **35** | 29 -> **20** | 1 | 9 |
| CUTLASS | 96 | 10 -> **25** | 23 -> **40** | 61 -> **29** | 2 | 37 |
| Mojo / MAX | 73 | 19 -> **26** | 27 -> **27** | 24 -> **17** | 3 | 11 |
| Slang | 61 | 17 -> **22** | 19 -> **21** | 22 -> **15** | 3 | 11 |
| ThunderKittens | 58 | 8 -> **15** | 19 -> **24** | 29 -> **17** | 2 | 14 |
| Triton | 70 | 14 -> **24** | 21 -> **21** | 34 -> **24** | 1 | 16 |
| **All seven** | 519 | 93 -> **149** | 168 -> **208** | 246 -> **150** | 12 | 122 |

The "before" column is the first measurement's number, except where this pass found the first one wrong (below). Per-row evidence for every changed row is in `docs/ir-union-evidence/rescore-2026-11-15/<survey>.md`; each carries the code symbol and the test that runs it.

### What this number does NOT mean

- **It is not "an emitter exists".** "Yes" means the IR can express the capability and the **oracle runs it**. The only emitter in this repository is the serial C reference (`unpopped-cpu-c`). Baracuda's, fishstringer's and openclownfish's emitters spell some of it and not the rest; a `Yes` here is not a promise that any vendor's source comes out. `unpopped::launch_cases` (0.47.3) is the first corpus those emitters can all run against the oracle, and it holds four elementwise cases.
- **It is not performance and not a measurement of generated code.** No row was scored on the speed of anything.
- **About 20 of the 37 CUTLASS moves, and some CUDA ones, rest on kernel-layer conformance kernels** (split-K, grouped GEMM, aux outputs, the implicit-GEMM convolution family, attention), tagged `(kernel layer)` in the evidence files. They show the kernel IR can express the kernel, not that the plan layer has a node for it. If the index is read plan-layer-only, those rows revert; rolling back only the twelve CUTLASS rows the agent tagged gives **about 48%**, so the real floor is a little lower still. This note does not say which reading CireSnave means.
- **Some `Yes` rows are compositions**, not a node of their own (Slang product/CountBits as `ReduceWith`/`Reduce(Sum)`: no test uses a literal multiply combine). And a few rest on emitter-set metadata (`KernelInterface::cooperative`, CUDA row 12) that the kernel itself does not determine.
- **Equal weight per row.** A `Partial` counts as half a `Yes`; rows are not weighted by how much work a consumer needs. Eight rows that need a descriptor operand kind count the same as one that needs a flag.

### What was NOT re-verified

- **Nothing was run.** No `cargo` build or test was executed for this measurement (read-only, and the box's C compiler setup hangs). Every status rests on reading a code symbol and the name of the test that exercises it. A lead check confirmed that all 42 test files the evidence cites, and the 98 test or function names it cites, exist in the tree at `78c7d38e` (a name present is not a proof the test asserts the claim).
- **Rows that were already `Yes` were not re-checked for a downward move** beyond the tests the first measurement cited; no row was found to have moved down, and none was looked for exhaustively.
- **Rows the 89 commits since `a634b09` cannot touch** (host-side concerns, build tooling) were skipped, by reason class, in each survey file.
- **Several moves are judgement calls**: CuTe 45 (DSMEM store plus global-to-shared element copies counted as `Partial` for "bulk copies"), CUDA 12 and 14 (emitter-set interface fields), ThunderKittens 15 (warp-level MMA to `Yes`; transposes are an atom layout, not a flag), Slang 18 and 23, CUTLASS 7, and Triton 70 (the agent said `Yes`; this note lowered it to `Partial`, because `tl.target_info` has no IR node).

### Where the first measurement was wrong

The agents found rows the first measurement under-scored (its pieces predate `a634b09`), so part of the jump is correction, not twelve days of work: **CUDA 49, 54, 68**; **CUTLASS 7**; **Slang 3, 23, 37**; **Mojo 18**; **ThunderKittens 45** (cluster launch, already in the tree) and 5 (should have been `Partial`). Rows where the agents say the first score may have been **too high** (not changed): CuTe 1, 2, 4, 5 (the `Layout` is a vocab value whose only IR consumer is `MmaAtom`), Mojo 42 and 45.

### Row counts: 13 rows the first measurement did not count

The first measurement numbered 519 rows. Its generator skipped every survey row whose status cell carried a parenthetical (`Yes (3-pass)`, `No (UNVERIFIED API)`, `No (...)`, `--`, `n/a (frontend)`). By file order the surveys hold **532** rows. This pass keeps the 519 so the two numbers are comparable, and states the 13 here instead: ThunderKittens 2 (tril/triu, `Yes (tensor level)`; cross-device signal-wait, **No -> Yes**: `Scope::System` barrier, fence and atomic, `tests/kernel_device_peer.rs`), Mojo 3 (`Yes (2 reserved)`, `n/a (frontend)`, `Yes (single axis, inner only v1)`), Triton 3 (all `Yes`-like), Slang 5 (cp.async/TMA/cluster **No -> Partial**; denormal handling `Partial`; generics, autodiff limits and ray tracing unchanged `No`). Scoring all 532 (the 13 at the status the agents gave them, the three Triton ones taken as `Yes`: Yes 156, Partial 210, No 153, n/a 13) gives an index of about 50% as well. Row numbers in the evidence files follow the survey's file order for CUDA, CuTe, CUTLASS, Mojo and ThunderKittens, and the first measurement's numbering for Triton and Slang; each file says which.

### The biggest remaining groups (the 150 `No` rows)

The agents listed what each remaining `No` row would take; this regroups them by theme across the seven surveys. Counts are rows, so one capability can appear several times, and the regrouping is mine and approximate (the sum is within a couple of rows of 150):

1. **Bulk async movement, `mbarrier`, named barriers, TMA and tensor-map descriptors, block-pointer and descriptor memory models: about 32 rows.** ROADMAP U8c and U4 S6b-2/S7; scheduled, not declined; trigger: Baracuda's sm_90 emitter asks, or a descriptor operand kind lands.
2. **Scheduling, roles, launch policy and compiler facts** (persistent and stream-K schedulers, warp specialization and `setmaxnreg`, PDL, a schedule language, `constexpr`/autotune inputs): **about 32 rows.** Mostly P9 (parked) and U17; many are plan-level hints, not IR statements.
3. **Warpgroup, `tcgen05`, tensor memory and `ldmatrix`/`stmatrix` matrix units: about 24 rows.** U7 later slices and U8b (needs a second lane-table source; held with the `mma.sp` hold).
4. **Tile and tensor value types, operand packing and scale layouts, contraction generalisation: about 19 rows.** Includes CUTLASS's five plan-layer contraction rows, which are D2 (held on Fuel and KISS).
5. **Cache and access hints, hints, and debug effects** (`CopyAsync` cache and zero-fill forms, unroll, `printf`/`assert`, RNG, global scratch): **about 19 rows.** U13, U18.
6. **Slang's resource and pipeline types** (textures and samplers, pointer values, spec constants, autodiff, byte-offset load/store): **about 11 rows.** U9c, U21, U23.
7. **Wide dtypes and numerics modes** (`I128`/`F128`, `F8E3M4`, `Fp8FastAccum`, NaN/Inf mode, bit-count nodes): **about 7 rows.**
8. **Host-side or out of scope** (multi-kernel graphs, copy-engine runtimes, target tables): about 5 rows.

### Unblocked next rows, by size

All are ROADMAP rows owned by Unpopped that are scheduled, not held; none is started. Breaking ones batch into one release so emitters break once (the U36 ruling).

- **S:** U24 (`multimem` refusals as gaps; one PR when a consumer asks, low rank). **RE-DISPOSITIONED 2026-11-26 (PM via pm-kernels; CireSnave 2026-10-24, 2026-11-11, 2026-11-19): not a wait; unblocked work, ranked lower.**
- **S-M:** U18 (unroll hint on `Stmt::For`; BREAKING, batches with U13 and U36).
- **M:** U13 (`CopyAsync` predicated, zero-fill and cache-hint forms; BREAKING field), U15 (8- and 16-lane packed values; additive if new forms), U17 (programmatic dependent launch), U22 (launch grid as an expression of parameters; additive beside the static form), U23 (struct-typed scalar parameters).
- **L:** U21 (texture and surface memory; waits on a consumer under the 2026-11-02 ruling). **RE-DISPOSITIONED 2026-11-26 (PM via pm-kernels; CireSnave 2026-10-24, 2026-11-11, 2026-11-19): not a wait; unblocked work, ranked lower.**
- **XL:** U19 (C-family kernel-body frontend).
- **Open, trigger-gated:** U7 later slices (`wgmma`/`tcgen05`), U8b, U8c, U9c (trigger met 2026-11-12 by Baracuda's CUDA frontend).
- **Deferred by the PM:** U36 (asymmetric im2col padding), until a consumer needs it or the next release that breaks anyway. **RE-DISPOSITIONED 2026-11-26 (PM via pm-kernels; CireSnave 2026-10-24, 2026-11-11, 2026-11-19): not a wait; unblocked work, ranked lower.**

Held and not startable here: A5 backward attention, stochastic rounding, `Fp8FastAccum`, accumulate-into-C and a precision recipe list (CireSnave, boards 443/444), I1 and I5.

## First measurement (2026-11-03, `a634b09`, Unpopped 0.42.14)

Kept unedited below as the baseline; its totals are superseded by the table above. One correction to it is stated above ("Where the first measurement was wrong").

Status: **a dated measurement (the first of two)**, taken at `origin/main` `a634b09` (Unpopped 0.42.14, 2026-11-03; 0.42.15, the `Cluster` scope, is in review and is counted here only where stated). Docs only; no code changed. Asked by CireSnave (2026-11-03: "Are we getting closer to a proper union of the kernel languages in our IR?"), approved by the PM.

## What was measured, and what was not

The seven surveys in `docs/ir-union-surveys/` (519 capability rows) were written on 2026-10-24/25, when the IR had no kernel layer. They are raw snapshots and stay untouched. This note re-asks each row against the code at the ref above and records **only the rows whose answer changed**, with the test or code that carries the new answer. Every other row is unchanged **as scored then**; this pass did not re-verify the unchanged rows, and a row the surveys call "No" may be wrong in either direction. A row moves up only on evidence in the tree (an enum variant, a checker rule, a test that runs); an unconfirmed piece is named in the Evidence column and keeps the row at `Partial`.

"Yes" means the IR can express it and the oracle runs it; it does not mean an emitter for that vendor exists. The only emitter in this repo is the serial C reference (`unpopped-cpu-c`); real targets live in other repos. The union-coverage figure below counts a `Partial` as half a `Yes` over the rows that are not `n/a`; it is a rough index, not a promise.

## Totals

| Survey | Rows | Yes | Partial | No | n/a | Changed |
|---|---|---|---|---|---|---|
| CUDA C++ | 95 | 7 -> **18** | 15 -> **30** | 73 -> **47** | 0 | 31 |
| CuTe / CuTe DSL | 66 | 0 -> **7** | 15 -> **29** | 50 -> **29** | 1 | 22 |
| CUTLASS | 96 | 6 -> **10** | 19 -> **23** | 69 -> **61** | 2 | 9 |
| Mojo / MAX | 73 | 7 -> **19** | 18 -> **27** | 45 -> **24** | 3 | 25 |
| Slang | 61 | 0 -> **17** | 14 -> **19** | 44 -> **22** | 3 | 26 |
| ThunderKittens | 58 | 7 -> **8** | 11 -> **19** | 38 -> **29** | 2 | 10 |
| Triton | 70 | 2 -> **14** | 16 -> **21** | 51 -> **34** | 1 | 21 |
| **All seven** | 519 | 29 -> **93** | 108 -> **168** | 370 -> **246** | 12 | 144 |

**Union coverage index (Yes + half Partial, over applicable rows): 16% -> 35%.**

## What moved, by theme

- **Kernel body** (loops, branches, early exit, locals, calls and modules): CUDA, Mojo, Slang, Triton and CuTe control-flow rows moved to `Yes`/`Partial` (U2, U3).
- **Execution model**: thread/block/grid and lane/subgroup ids, launch geometry, subgroup size, static and dynamic shared memory, per-thread arrays (U3).
- **Synchronization**: `Subgroup`/`Block`/`Device` barriers with a uniformity rule, `Cluster` scope (0.42.15), fences, atomics with order, scope, old value and compare-and-swap (U4).
- **Collectives**: shuffle, vote, ballot, reduce, scan and broadcast at subgroup and block scope (U5).
- **Layout algebra**: the CuTe layout rows moved wholesale (U6, done): hierarchical layouts, composition, complement, divide, product, coalesce, inverses.
- **Numerics and dtypes**: `Fma`, `Cast`/`Bitcast` with rounding and saturation, rounding/ftz/approx qualifiers, fp4/fp6 storage, packed values (U1).
- **Reductions and scans**: bitwise, NaN-ignoring and arg-reductions; scans and windows over any axis; a running mean and a product window (U5, U2; four of these are Unpopped extensions, proposed in KISS#766-769).

## The remaining gaps, as a counted list

Rows still `No` after this pass: **246** of 519. They are not independent (one capability often appears in several surveys). Grouped by what it would take:

1. **Asynchronous data movement, pipelines and mbarrier-style synchronization**: 71 rows.
2. **Tensor-core and matrix-unit operations**: 37 rows.
3. **Resources, address spaces, cache policy and host-side or vendor concerns**: 36 rows.
4. **Scheduling, launch policy and whole-kernel recipes (GEMM/conv/attention schedules)**: 36 rows.
5. **Tile, fragment and partition abstractions**: 22 rows.
6. **Smaller numerics, dtypes and mixed-precision policy**: 19 rows.
7. **Escape hatches, metaprogramming and user-defined types**: 17 rows.
8. **Unclassified**: 8 rows.

Where the groups live in the ROADMAP: asynchronous movement is U4 S7 and U8 (open; named barriers are U4 S6b-2, waiting for a consumer); matrix units are U7 (open, now unblocked: U3 and U6 are done) and U8; tile and fragment abstractions are U7/U8 on top of the finished layout algebra; scheduling and whole-kernel recipes are mostly plan-level hints, not IR statements; resources are U3's open rows and partly out of scope for a kernel IR; numerics are the U1 remainder (packed arithmetic, wider index dtypes); escape hatches are mostly deliberately out of scope; custom-combine reductions and the effects model are U5/U2 D6.

The groups are assigned by keyword over each remaining `No` row's capability text (`docs/ir-union-rescore.md` generator logic: first matching group wins), so a row can land in a neighbouring group and the numbers are indicative; the 'Unclassified' line shows what the keywords missed. Where one capability appears in several surveys it is counted once per survey.

## Extensions awaiting a KISS ruling

Everything in the union that KISS does not yet have is marked "Unpopped extension, not KISS" and proposed upstream; none blocks Unpopped: KISS#766 (running mean scan), #767 (product pool), #768 (bitwise and NaN-ignoring reductions, collectives), #769 (synchronization and memory ordering).

## The changes, row by row

### CUDA C++ (`docs/ir-union-surveys/cuda.md`): 31 rows changed of 95

| # | Capability | Status | Evidence at `a634b09` |
|---|---|---|---|
| 2 | Memory spaces: static `__shared__` array | No -> **Yes** | `Place::Alloc` in `MemorySpace::Shared` with a static `AllocDecl`; `tests/kernel_launch.rs`, `kernel_atomic_shared.rs`; cpu-c spells it |
| 3 | Memory spaces: dynamic `extern __shared__` + opt-in >48KB (`cudaFuncAttributeMaxDynamicSharedMemorySize`) | No -> **Partial** | `Extent::Param`: a dynamic extent from a scalar argument (`kernel_exec.rs`); no opt-in size attribute |
| 4 | Memory spaces: `__constant__` memory, `__grid_constant__` params | No -> **Partial** | `MemorySpace::Constant` can be declared; `AllocDecl` has no initializer, so the oracle refuses to read one (shape only) |
| 5 | Memory spaces: local memory / per-thread arrays, register arrays with dynamic index | No -> **Yes** | `MemorySpace::Private` per-thread arrays (oracle: `UninitializedPrivate`, `tests/kernel_launch.rs`); cpu-c refuses them; oracle only; cpu-c refuses |
| 10 | Execution: thread/block/grid indexing (`threadIdx/blockIdx/blockDim/gridDim`) | Partial -> **Yes** | `ExecId::{ThreadIdx,BlockIdx,BlockDim,GridDim}` on three axes (`exec.rs`); `tests/kernel_exec.rs`, `kernel_launch.rs` |
| 11 | Execution: thread-block clusters (`__cluster_dims__`, `cudaLaunchKernelEx` cluster attr, cluster rank | No -> **Partial** | `LaunchShape::cluster` + `Cluster` barrier/atomic/fence scope (0.42.15); `tests/kernel_cluster.rs`; no cluster ids, no distributed shared memory |
| 12 | Execution: cooperative launch / `grid.sync()` | No -> **Partial** | `Barrier` at `Device` scope, every block resident (0.42.13; cpu-c phases 0.42.14); `tests/kernel_device_barrier.rs`, cpu-c `kernel_device_barrier_emit_agrees.rs`; grid.sync semantics; no cooperative-launch attribute |
| 18 | Sync: `__syncthreads()` / `cg::sync` | No -> **Yes** | `Stmt::Barrier` at `Subgroup` and `Block` scope + the uniformity rule (`kernel_uniform.rs`); `tests/kernel_barrier.rs`, `kernel_barriers_run.rs` |
| 20 | Sync: `__syncwarp(mask)` / `bar.warp.sync` | No -> **Partial** | `Barrier` at `Subgroup` scope; no lane mask |
| 24 | Sync: fences: `__threadfence{_block,_system}`, `fence.proxy.async`, `fence.proxy.tensormap`, `tc | No -> **Partial** | `Stmt::Fence` with order and scope (0.42.8); `tests/kernel_fence.rs`; no proxy/async/tensormap fences |
| 28 | Collectives: warp shuffle `__shfl_{,up,down,xor}_sync` | No -> **Yes** | `Stmt::Collective` shuffle/vote/ballot/reduce/scan/broadcast (0.42.3; cpu-c 0.42.4); `tests/kernel_collective.rs`, cpu-c `kernel_collective_emit_agrees.rs` |
| 29 | Collectives: vote/ballot/any/all, `__activemask`, `match_any/all` | No -> **Partial** | `Stmt::Collective` shuffle/vote/ballot/reduce/scan/broadcast (0.42.3; cpu-c 0.42.4); `tests/kernel_collective.rs`, cpu-c `kernel_collective_emit_agrees.rs`; ballot/any/all; no `activemask`/`match` |
| 30 | Collectives: `redux.sync` / `__reduce_{add,min,max,and,or,xor}_sync` (sm_80+) | Partial -> **Yes** | `Stmt::Collective` shuffle/vote/ballot/reduce/scan/broadcast (0.42.3; cpu-c 0.42.4); `tests/kernel_collective.rs`, cpu-c `kernel_collective_emit_agrees.rs`; Reduce Sum/Max/Min/And/Or/Xor |
| 31 | Collectives: cooperative_groups partitions: `tiled_partition<N>`, `labeled_partition`, `binary_partitio | No -> **Partial** | `Stmt::Collective` shuffle/vote/ballot/reduce/scan/broadcast (0.42.3; cpu-c 0.42.4); `tests/kernel_collective.rs`, cpu-c `kernel_collective_emit_agrees.rs`; `Subgroup` scope = a tiled partition of the launch subgroup size only |
| 32 | Collectives: `cg::reduce` w/ plus/less/greater/bit_and/bit_or/bit_xor; `reduce_update_async` | Partial -> **Yes** | `Stmt::Collective` shuffle/vote/ballot/reduce/scan/broadcast (0.42.3; cpu-c 0.42.4); `tests/kernel_collective.rs`, cpu-c `kernel_collective_emit_agrees.rs` |
| 59 | Dtypes: `half2` / `__nv_bfloat162` / `float2` / `float4` packed | No -> **Partial** | `ScalarExpr::{Pack,Extract,BitcastPacked}` and the packed `Select` mask (0.31.x); no vector arithmetic ops beyond lane-wise `Select` |
| 61 | Dtypes: `__nv_fp6_e2m3`, `__nv_fp6_e3m2` | No -> **Partial** | `ElementKind::{F4E2M1,F6E2M3,F6E3M2}`: storage and key identity (0.29.0), `Cast` rows (0.31.7); stored-result layout still open |
| 62 | Dtypes: `__nv_fp4_e2m1` (and NVFP4 = e2m1 + ue4m3 block scale) | No -> **Partial** | `ElementKind::{F4E2M1,F6E2M3,F6E3M2}`: storage and key identity (0.29.0), `Cast` rows (0.31.7); stored-result layout still open |
| 67 | Atomics/ordering: `atomicAdd` | Partial -> **Yes** | `Stmt::AtomicRmw`/`AtomicCas` with `old`, order and scope (0.41.x-0.42.7); `tests/kernel_atomic_{rmw,fetch,cas,order,shared}.rs` |
| 69 | Atomics/ordering: `atomicCAS`, `atomicExch`, `atomicAnd/Or/Xor`, `atomicInc/Dec` | No -> **Partial** | `Stmt::AtomicRmw`/`AtomicCas` with `old`, order and scope (0.41.x-0.42.7); `tests/kernel_atomic_{rmw,fetch,cas,order,shared}.rs`; CAS, exchange, and/or/xor; no `atomicInc/Dec` wrap ops |
| 70 | Atomics/ordering: `cuda::atomic<T,scope>`, `atomic_ref`, `fetch_add`, load/store with acquire/release; scope | No -> **Partial** | order+scope RMW (an `Or` of 0 reads); no atomic load/store statements; a spin on one is a named `CrossBlockWait` (0.42.7) |
| 71 | Atomics/ordering: cluster-scope atomics / DSM atomics | No -> **Partial** | `LaunchShape::cluster` + `Cluster` barrier/atomic/fence scope (0.42.15); `tests/kernel_cluster.rs`; no cluster ids, no distributed shared memory; cluster-scope atomics; no distributed shared memory |
| 73 | Control flow: data-dependent `while`/`for` loops, early `break/continue/return` | No -> **Yes** | `Stmt::{If,For,While,Break,Continue,Return}` and `Let`/`Assign` locals; `tests/kernel_layer.rs`, `kernel_resume_agrees.rs`, `kernel_eval_semantics.rs` |
| 74 | Control flow: predication / divergence (`if (tid<n)`), masks | Partial -> **Yes** | `Stmt::{If,For,While,Break,Continue,Return}` and `Let`/`Assign` locals; `tests/kernel_layer.rs`, `kernel_resume_agrees.rs`, `kernel_eval_semantics.rs` |
| 75 | Control flow: recursion, function calls, templates, `constexpr` | No -> **Partial** | `Stmt::Call`, `Module`, `check_module`, run in the oracle and spelled in cpu-c; `tests/kernel_module_run.rs`, `kernel_call_shape.rs`; no recursion, templates or `constexpr` |
| 82 | Numerics: `--use_fast_math` bundle: `-ftz`, `-prec-div`, `-prec-sqrt`, `-fmad` | No -> **Partial** | `ScalarExpr::Qualified` rounding/ftz/approx qualifiers (0.25.0); `docs/ir-numerics-qualifiers.md` |
| 83 | Numerics: explicit rounding-mode intrinsics (`_rn/_rz/_ru/_rd`: `__fadd_rn`, `__fmaf_rz`, `__fma_rd` | No -> **Partial** | `ScalarExpr::Qualified` rounding/ftz/approx qualifiers (0.25.0); `docs/ir-numerics-qualifiers.md` |
| 84 | Numerics: explicit FMA (`fma`, `__fmaf_rn`, `__fmaf_ieee_*`) vs contraction freedom (`-fmad`) | No -> **Yes** | `ScalarExpr::Fma` (0.23.0) |
| 85 | Numerics: approximate transcendental intrinsics (`__expf`, `__sinf`, `__cosf`, `__powf`, `__fdividef | No -> **Partial** | `ScalarExpr::Qualified` rounding/ftz/approx qualifiers (0.25.0); `docs/ir-numerics-qualifiers.md` |
| 87 | Numerics: explicit conversions with rounding/saturation (`cvt.rn.satfinite`, `cvt.pack`, `__float2ha | No -> **Partial** | `ScalarExpr::Cast`/`Bitcast` with `RoundMode` and `Saturation` (0.24.0); `docs/ir-typed-expressions.md`; no `cvt.pack` |
| 90 | Unique to CUDA: special registers (`%laneid`, `%warpid`, `%smid`, `%clock`, `%globaltimer`, cluster regs ` | No -> **Partial** | `ExecId::{LaneId,SubgroupId,SubgroupSize}`; `tests/kernel_exec.rs` (no `sm_id`, clock or other special registers) |

### CuTe / CuTe DSL (`docs/ir-union-surveys/cute.md`): 22 rows changed of 66

| # | Capability | Status | Evidence at `a634b09` |
|---|---|---|---|
| 1 | Layout: Layout = (Shape, Stride) pair with explicit integer strides as first-class IR value | No -> **Yes** | `unpopped-vocab/src/layout_algebra.rs`: hierarchical `IntTuple`/`Layout`, `composition`, `complement`, `logical_divide`, `logical_product`, `coalesce`, inverses, `swizzle`, `select/take/slice/tile_to_shape` with a property-test suite (U6, ROADMAP: done, issue #117 closed 2026-11-02) |
| 2 | Layout: Hierarchical/nested IntTuple shapes and strides (`(_5,(_2,_2)):(_16,(_80,_4))`), depth, co | No -> **Yes** | `unpopped-vocab/src/layout_algebra.rs`: hierarchical `IntTuple`/`Layout`, `composition`, `complement`, `logical_divide`, `logical_product`, `coalesce`, inverses, `swizzle`, `select/take/slice/tile_to_shape` with a property-test suite (U6, ROADMAP: done, issue #117 closed 2026-11-02) |
| 4 | Layout: Layout as function idx->offset; size/rank/depth/cosize | No -> **Yes** | `unpopped-vocab/src/layout_algebra.rs`: hierarchical `IntTuple`/`Layout`, `composition`, `complement`, `logical_divide`, `logical_product`, `coalesce`, inverses, `swizzle`, `select/take/slice/tile_to_shape` with a property-test suite (U6, ROADMAP: done, issue #117 closed 2026-11-02) |
| 5 | Layout: crd2idx / idx2crd (colexicographic, natural, R-D, h-D coordinates) | Partial -> **Yes** | `unpopped-vocab/src/layout_algebra.rs`: hierarchical `IntTuple`/`Layout`, `composition`, `complement`, `logical_divide`, `logical_product`, `coalesce`, inverses, `swizzle`, `select/take/slice/tile_to_shape` with a property-test suite (U6, ROADMAP: done, issue #117 closed 2026-11-02) |
| 6 | Layout: Structural ops: get/select/take/flatten/group/append/prepend/replace/slice/dice/filter/fil | No -> **Partial** | `unpopped-vocab/src/layout_algebra.rs`: hierarchical `IntTuple`/`Layout`, `composition`, `complement`, `logical_divide`, `logical_product`, `coalesce`, inverses, `swizzle`, `select/take/slice/tile_to_shape` with a property-test suite (U6, ROADMAP: done, issue #117 closed 2026-11-02); select/take/slice/flatten/tile_to_shape; group/append/prepend/replace/dice/filter not confirmed |
| 8 | Layout algebra: composition (A o B), by-mode composition with tilers | No -> **Yes** | `unpopped-vocab/src/layout_algebra.rs`: hierarchical `IntTuple`/`Layout`, `composition`, `complement`, `logical_divide`, `logical_product`, `coalesce`, inverses, `swizzle`, `select/take/slice/tile_to_shape` with a property-test suite (U6, ROADMAP: done, issue #117 closed 2026-11-02) |
| 9 | Layout algebra: complement(layout, cotarget) | No -> **Yes** | `unpopped-vocab/src/layout_algebra.rs`: hierarchical `IntTuple`/`Layout`, `composition`, `complement`, `logical_divide`, `logical_product`, `coalesce`, inverses, `swizzle`, `select/take/slice/tile_to_shape` with a property-test suite (U6, ROADMAP: done, issue #117 closed 2026-11-02) |
| 10 | Layout algebra: logical_divide / zipped / tiled / flat divide (tiling with tile + rest modes, round-up res | No -> **Partial** | `unpopped-vocab/src/layout_algebra.rs`: hierarchical `IntTuple`/`Layout`, `composition`, `complement`, `logical_divide`, `logical_product`, `coalesce`, inverses, `swizzle`, `select/take/slice/tile_to_shape` with a property-test suite (U6, ROADMAP: done, issue #117 closed 2026-11-02); logical_divide; the zipped/tiled/flat variants not confirmed |
| 11 | Layout algebra: logical_product / blocked / raked / zipped / tiled / flat product | No -> **Partial** | `unpopped-vocab/src/layout_algebra.rs`: hierarchical `IntTuple`/`Layout`, `composition`, `complement`, `logical_divide`, `logical_product`, `coalesce`, inverses, `swizzle`, `select/take/slice/tile_to_shape` with a property-test suite (U6, ROADMAP: done, issue #117 closed 2026-11-02); logical_product; the variants not confirmed |
| 12 | Layout algebra: left_inverse / right_inverse; max_common_layout/vector (basis of auto-vectorisation) | No -> **Partial** | `unpopped-vocab/src/layout_algebra.rs`: hierarchical `IntTuple`/`Layout`, `composition`, `complement`, `logical_divide`, `logical_product`, `coalesce`, inverses, `swizzle`, `select/take/slice/tile_to_shape` with a property-test suite (U6, ROADMAP: done, issue #117 closed 2026-11-02); left/right inverse; `max_common_layout/vector` not built |
| 13 | Layout algebra: make_ordered_layout, make_identity_layout, tile_to_shape, recast_layout (element bit-width | No -> **Partial** | `unpopped-vocab/src/layout_algebra.rs`: hierarchical `IntTuple`/`Layout`, `composition`, `complement`, `logical_divide`, `logical_product`, `coalesce`, inverses, `swizzle`, `select/take/slice/tile_to_shape` with a property-test suite (U6, ROADMAP: done, issue #117 closed 2026-11-02); `col_major`, `tile_to_shape`; `recast_layout`/`make_ordered_layout` not confirmed |
| 15 | Swizzle: `Swizzle<B,M,S>` XOR functor; ComposedLayout(swizzle o offset o layout); smem swizzle atom | No -> **Partial** | `swizzle` in `layout_algebra` (a functor over layouts); no shared-memory tile consumer, no `ComposedLayout` with smem atoms |
| 18 | Memory space: Pointer tagged with address space: gmem/smem/rmem/tmem/dsmem/cmem/generic | No -> **Partial** | `Place::Alloc` in `MemorySpace::Shared` with a static `AllocDecl`; `tests/kernel_launch.rs`, `kernel_atomic_shared.rs`; cpu-c spells it; `MemorySpace` on allocations; no tagged pointers |
| 19 | Memory: Static shared-memory allocation / `cute.struct` SharedStorage with MemRange and Align; dyn | No -> **Partial** | `Place::Alloc` in `MemorySpace::Shared` with a static `AllocDecl`; `tests/kernel_launch.rs`, `kernel_atomic_shared.rs`; cpu-c spells it; no `cute.struct`/alignment |
| 20 | Memory: Register fragments: make_fragment_like / make_rmem_tensor / TensorSSA (immutable vector va | No -> **Partial** | `MemorySpace::Private` per-thread arrays (oracle: `UninitializedPrivate`, `tests/kernel_launch.rs`); cpu-c refuses them |
| 28 | Index: Thread / block / cluster / warp identity values (thread_idx, block_idx, cluster_idx, lane_ | No -> **Partial** | `ExecId::{LaneId,SubgroupId,SubgroupSize}`; `tests/kernel_exec.rs` (no `sm_id`, clock or other special registers); thread/block/lane/subgroup ids; no cluster id |
| 49 | Sync: CTA barrier `sync_threads`, named barriers (IDs 0-15, arrive / arrive_and_wait), warp sync | No -> **Partial** | `Stmt::Barrier` at `Subgroup` and `Block` scope + the uniformity rule (`kernel_uniform.rs`); `tests/kernel_barrier.rs`, `kernel_barriers_run.rs`; CTA/warp barrier; no named barriers |
| 50 | Sync: Warp collectives: shuffle (idx/up/down/bfly), vote (ballot/any/all/uni), match, warp_redux | No -> **Partial** | `Stmt::Collective` shuffle/vote/ballot/reduce/scan/broadcast (0.42.3; cpu-c 0.42.4); `tests/kernel_collective.rs`, cpu-c `kernel_collective_emit_agrees.rs`; shuffle idx/up/down/xor, ballot/any/all; no match/uni |
| 51 | Sync: Cluster sync (arrive/wait/relaxed), cluster/CTA-in-cluster ids, distributed shared memory | No -> **Partial** | `LaunchShape::cluster` + `Cluster` barrier/atomic/fence scope (0.42.15); `tests/kernel_cluster.rs`; no cluster ids, no distributed shared memory; cluster barrier; no cluster ids or DSM |
| 53 | Sync: Memory fences: acq_rel at cta/cluster/gpu/sys; proxy fences (async/tensormap/generic); tme | No -> **Partial** | `Stmt::Fence` with order and scope (0.42.8); `tests/kernel_fence.rs`; no proxy/async/tensormap fences |
| 60 | Control: Per-thread control flow: if/for/while with loop-carried values; tile loops (k-tile mainloo | No -> **Yes** | `Stmt::{If,For,While,Break,Continue,Return}` and `Let`/`Assign` locals; `tests/kernel_layer.rs`, `kernel_resume_agrees.rs`, `kernel_eval_semantics.rs` |
| 62 | Launch: grid/block/cluster, fallback_cluster, dynamic smem, PDL (`use_pdl`), cooperative launch, m | No -> **Partial** | `LaunchShape::cluster` + `Cluster` barrier/atomic/fence scope (0.42.15); `tests/kernel_cluster.rs`; no cluster ids, no distributed shared memory; grid/block/cluster dims, dynamic shared extent; no PDL/fallback cluster |

### CUTLASS (`docs/ir-union-surveys/cutlass.md`): 9 rows changed of 96

| # | Capability | Status | Evidence at `a634b09` |
|---|---|---|---|
| 58 | Scheduling: Thread block clusters, cluster shape, preferred+fallback cluster, dynamic cluster | No -> **Partial** | `LaunchShape::cluster` + `Cluster` barrier/atomic/fence scope (0.42.15); `tests/kernel_cluster.rs`; no cluster ids, no distributed shared memory; cluster dims; no preferred/fallback or dynamic cluster |
| 66 | Data movement: Shared-memory tiles with swizzled layouts (Crosswise/Congruous, CuTe Swizzle<B,M,S>) | No -> **Partial** | `swizzle` in `layout_algebra` (a functor over layouts); no shared-memory tile consumer, no `ComposedLayout` with smem atoms |
| 74 | Compute: Saturating (satfinite) int output | No -> **Yes** | `ScalarExpr::Cast`/`Bitcast` with `RoundMode` and `Saturation` (0.24.0); `docs/ir-typed-expressions.md`; `Saturation` on integer outputs |
| 78 | Compute: fp6 (e2m3, e3m2), fp4 (e2m1) | No -> **Partial** | `ElementKind::{F4E2M1,F6E2M3,F6E3M2}`: storage and key identity (0.29.0), `Cast` rows (0.31.7); stored-result layout still open |
| 83 | Compute: Half2/bfloat162/vectorized packed math (HFMA2, FP32x2 packed) | No -> **Partial** | `ScalarExpr::{Pack,Extract,BitcastPacked}` and the packed `Select` mask (0.31.x); no vector arithmetic ops beyond lane-wise `Select` |
| 84 | Compute: Fast-math/approx transcendentals (`__expf`) as explicit "Approx" | No -> **Partial** | `ScalarExpr::Qualified` rounding/ftz/approx qualifiers (0.25.0); `docs/ir-numerics-qualifiers.md` |
| 85 | Compute: Warp/block collectives (shuffle, reduce) | Partial -> **Yes** | `Stmt::Collective` shuffle/vote/ballot/reduce/scan/broadcast (0.42.3; cpu-c 0.42.4); `tests/kernel_collective.rs`, cpu-c `kernel_collective_emit_agrees.rs` |
| 86 | CuTe/DSL: CuTe layouts algebra (composition, complement, logical divide/product, hierarchical shapes | No -> **Yes** | `unpopped-vocab/src/layout_algebra.rs`: hierarchical `IntTuple`/`Layout`, `composition`, `complement`, `logical_divide`, `logical_product`, `coalesce`, inverses, `swizzle`, `select/take/slice/tile_to_shape` with a property-test suite (U6, ROADMAP: done, issue #117 closed 2026-11-02) |
| 89 | CuTe/DSL: Control flow with dynamic loops/ifs inside kernel | No -> **Yes** | `Stmt::{If,For,While,Break,Continue,Return}` and `Let`/`Assign` locals; `tests/kernel_layer.rs`, `kernel_resume_agrees.rs`, `kernel_eval_semantics.rs` |

### Mojo / MAX (`docs/ir-union-surveys/mojo.md`): 25 rows changed of 73

| # | Capability | Status | Evidence at `a634b09` |
|---|---|---|---|
| 1 | Exec hierarchy: thread_idx / block_idx / block_dim / grid_dim / global_idx (x,y,z) | No -> **Yes** | `ExecId::{ThreadIdx,BlockIdx,BlockDim,GridDim}` on three axes (`exec.rs`); `tests/kernel_exec.rs`, `kernel_launch.rs` |
| 2 | Exec hierarchy: lane_id, warp_id, sm_id | No -> **Partial** | `ExecId::{LaneId,SubgroupId,SubgroupSize}`; `tests/kernel_exec.rs` (no `sm_id`, clock or other special registers) |
| 3 | Exec hierarchy: thread blocks clusters: cluster_idx, cluster_dim, block_id_in_cluster, block_rank_in_clust | No -> **Partial** | `LaunchShape::cluster` + `Cluster` barrier/atomic/fence scope (0.42.15); `tests/kernel_cluster.rs`; no cluster ids, no distributed shared memory; cluster dims and barrier; no cluster_idx/rank leaves |
| 4 | Exec hierarchy: Launch config via DeviceContext.enqueue_function (grid/block dim, dynamic shared bytes, la | No -> **Partial** | `Extent::Param`: a dynamic extent from a scalar argument (`kernel_exec.rs`); no opt-in size attribute; grid/block dims in `LaunchShape`; no dynamic shared bytes attribute |
| 6 | Memory spaces: AddressSpace (generic/global/shared/constant/local/shared-cluster) on pointers | No -> **Partial** | `Place::Alloc` in `MemorySpace::Shared` with a static `AllocDecl`; `tests/kernel_launch.rs`, `kernel_atomic_shared.rs`; cpu-c spells it; `MemorySpace` on allocations, not on pointers |
| 7 | Memory spaces: Static shared memory via stack_allocation / `unsafe_stack_allocation` | No -> **Yes** | `Place::Alloc` in `MemorySpace::Shared` with a static `AllocDecl`; `tests/kernel_launch.rs`, `kernel_atomic_shared.rs`; cpu-c spells it |
| 8 | Memory spaces: Dynamic/external shared memory (`external_memory`) | No -> **Partial** | `Extent::Param`: a dynamic extent from a scalar argument (`kernel_exec.rs`); no opt-in size attribute |
| 9 | Memory spaces: Registers/local arrays; register realloc (`warpgroup_reg_alloc/dealloc`) | No -> **Partial** | `MemorySpace::Private` per-thread arrays (oracle: `UninitializedPrivate`, `tests/kernel_launch.rs`); cpu-c refuses them |
| 13 | Sync: Block barrier (`barrier()`) | No -> **Yes** | `Stmt::Barrier` at `Subgroup` and `Block` scope + the uniformity rule (`kernel_uniform.rs`); `tests/kernel_barrier.rs`, `kernel_barriers_run.rs` |
| 14 | Sync: syncwarp(mask) | No -> **Partial** | `Barrier` at `Subgroup` scope; no lane mask |
| 19 | Sync: Memory fences: threadfence, `atomic.fence`, fence_async_view_proxy, fence_mbarrier_init, t | No -> **Partial** | `Stmt::Fence` with order and scope (0.42.8); `tests/kernel_fence.rs`; no proxy/async/tensormap fences |
| 20 | Warp collectives: shuffle_idx/up/down/xor | No -> **Yes** | `Stmt::Collective` shuffle/vote/ballot/reduce/scan/broadcast (0.42.3; cpu-c 0.42.4); `tests/kernel_collective.rs`, cpu-c `kernel_collective_emit_agrees.rs` |
| 21 | Warp collectives: warp sum/max/min/reduce, lane_group_{sum,max,min,reduce}, broadcast, prefix_sum | Partial -> **Yes** | `Stmt::Collective` shuffle/vote/ballot/reduce/scan/broadcast (0.42.3; cpu-c 0.42.4); `tests/kernel_collective.rs`, cpu-c `kernel_collective_emit_agrees.rs` |
| 22 | Warp collectives: vote (ballot mask), match_any, match_all, elect_one_sync | No -> **Partial** | `Stmt::Collective` shuffle/vote/ballot/reduce/scan/broadcast (0.42.3; cpu-c 0.42.4); `tests/kernel_collective.rs`, cpu-c `kernel_collective_emit_agrees.rs` |
| 23 | Block collectives: block.sum/max/min/broadcast/prefix_sum | Partial -> **Yes** | `Stmt::Collective` shuffle/vote/ballot/reduce/scan/broadcast (0.42.3; cpu-c 0.42.4); `tests/kernel_collective.rs`, cpu-c `kernel_collective_emit_agrees.rs`; Block-scope reduce/scan/broadcast |
| 36 | Layouts: Layout algebra: Layout(shape,stride) nested/hierarchical; composition, coalesce, complemen | No -> **Yes** | `unpopped-vocab/src/layout_algebra.rs`: hierarchical `IntTuple`/`Layout`, `composition`, `complement`, `logical_divide`, `logical_product`, `coalesce`, inverses, `swizzle`, `select/take/slice/tile_to_shape` with a property-test suite (U6, ROADMAP: done, issue #117 closed 2026-11-02) |
| 37 | Layouts: Swizzle / ComposedLayout, make_swizzle, make_ldmatrix_swizzle (bank-conflict avoidance, TM | No -> **Partial** | `swizzle` in `layout_algebra` (a functor over layouts); no shared-memory tile consumer, no `ComposedLayout` with smem atoms |
| 39 | Layouts: LayoutTensor ops: tile, distribute(thread layout), vectorize, slice, copy_from, async copy | No -> **Partial** | `unpopped-vocab/src/layout_algebra.rs`: hierarchical `IntTuple`/`Layout`, `composition`, `complement`, `logical_divide`, `logical_product`, `coalesce`, inverses, `swizzle`, `select/take/slice/tile_to_shape` with a property-test suite (U6, ROADMAP: done, issue #117 closed 2026-11-02); `slice`, `tile_to_shape`; `distribute`/`vectorize`/`copy_from` not built |
| 44 | Dtypes: float4_e2m1fn, float6_e2m3fn, float6_e3m2fn as DType | Partial -> **Yes** | `ElementKind::{F4E2M1,F6E2M3,F6E3M2}`: storage and key identity (0.29.0), `Cast` rows (0.31.7); stored-result layout still open |
| 60 | Atomics: fence(ordering) standalone | No -> **Yes** | `Stmt::Fence` with order and scope (0.42.8); `tests/kernel_fence.rs`; no proxy/async/tensormap fences |
| 63 | Reductions: Arg-reductions / custom (non-monoid) combine functions as user lambdas (`warp.reduce` take | No -> **Partial** | `ReduceOp::{Sum,Mean,Max,Min,Prod,And,Or,Xor,MaxNum,MinNum,ArgMax,ArgMin}`; `tests/reduce_bit_ops.rs`, `reduce_maxnum.rs`, `reduce_argmax.rs`, `kernel_reduction_agrees.rs`; arg-reductions yes; a user-lambda combine is not |
| 64 | Control flow: Data-dependent loops, early exit/return, while, runtime if/else (not select) with side eff | No -> **Yes** | `Stmt::{If,For,While,Break,Continue,Return}` and `Let`/`Assign` locals; `tests/kernel_layer.rs`, `kernel_resume_agrees.rs`, `kernel_eval_semantics.rs` |
| 65 | Control flow: Local mutable variables / registers, in-kernel arrays, scratch state across loop iteration | No -> **Yes** | `MemorySpace::Private` per-thread arrays (oracle: `UninitializedPrivate`, `tests/kernel_launch.rs`); cpu-c refuses them; oracle only for arrays; locals everywhere |
| 66 | Control flow: Function calls / `@always_inline` helper kernels, closures captured in `foreach`/`vectoriz | No -> **Partial** | `Stmt::Call`, `Module`, `check_module`, run in the oracle and spelled in cpu-c; `tests/kernel_module_run.rs`, `kernel_call_shape.rs`; no closures |
| 67 | Control flow: Bounds guards / predicated tails (tile remainder, `if tid < n`) | Partial -> **Yes** | `Stmt::{If,For,While,Break,Continue,Return}` and `Let`/`Assign` locals; `tests/kernel_layer.rs`, `kernel_resume_agrees.rs`, `kernel_eval_semantics.rs` |

### Slang (`docs/ir-union-surveys/slang.md`): 26 rows changed of 61

| # | Capability | Status | Evidence at `a634b09` |
|---|---|---|---|
| 1 | Memory: `groupshared` / workgroup array (named, sized, per-block) | No -> **Yes** | `Place::Alloc` in `MemorySpace::Shared` with a static `AllocDecl`; `tests/kernel_launch.rs`, `kernel_atomic_shared.rs`; cpu-c spells it |
| 2 | Memory: Dynamic shared memory size / opaque scratch | No -> **Partial** | `Extent::Param`: a dynamic extent from a scalar argument (`kernel_exec.rs`); no opt-in size attribute |
| 5 | Memory: Append/Consume structured buffer (atomic counter queue) | No -> **Partial** | fetch-add hands each thread a distinct slot (`tests/kernel_atomic_fetch.rs`): an append-queue counter, no `Append/Consume` resource |
| 11 | Memory: Coherence/cache qualifiers (`LoadCoherent`, `globallycoherent`, Vulkan memory model `vk_me | No -> **Partial** | `Stmt::AtomicRmw`/`AtomicCas` with `old`, order and scope (0.41.x-0.42.7); `tests/kernel_atomic_{rmw,fetch,cas,order,shared}.rs`; memory order and scope; no coherence qualifiers |
| 13 | Sync: Workgroup barrier `GroupMemoryBarrierWithGroupSync` (Fuel: 36 files) | No -> **Yes** | `Stmt::Barrier` at `Subgroup` and `Block` scope + the uniformity rule (`kernel_uniform.rs`); `tests/kernel_barrier.rs`, `kernel_barriers_run.rs` |
| 14 | Sync: Memory-only barriers: `GroupMemoryBarrier`, `DeviceMemoryBarrier[WithGroupSync]`, `AllMemo | No -> **Yes** | `Stmt::Fence` with order and scope (0.42.8); `tests/kernel_fence.rs`; no proxy/async/tensormap fences; `GroupMemoryBarrier` = Fence(Block), `DeviceMemoryBarrier` = Fence(Device), `...WithGroupSync` = Barrier |
| 15 | Sync: Per-thread control flow with barriers (uniformity analysis, `Treat Values as Uniform`) | No -> **Yes** | `kernel_uniform.rs` (barrier-uniformity analysis); `tests/kernel_barrier.rs` |
| 16 | Sync: Execution-model queries: `SV_GroupThreadID`, `SV_GroupID`, `SV_DispatchThreadID`, `SV_Grou | Partial -> **Yes** | `ExecId::{ThreadIdx,BlockIdx,BlockDim,GridDim}` on three axes (`exec.rs`); `tests/kernel_exec.rs`, `kernel_launch.rs`; SV_* ids are arithmetic over thread/block ids |
| 17 | Sync: `[numthreads(x,y,z)]`, generic numthreads, `[WaveSize(n)]` | No -> **Yes** | `LaunchShape { block, grid }` (the `[numthreads]` equivalent) and the subgroup range |
| 18 | Wave: Reductions `WaveActiveSum/Product/Min/Max/BitAnd/BitOr/BitXor/CountBits/AllTrue/AnyTrue/Al | No -> **Partial** | `Stmt::Collective` shuffle/vote/ballot/reduce/scan/broadcast (0.42.3; cpu-c 0.42.4); `tests/kernel_collective.rs`, cpu-c `kernel_collective_emit_agrees.rs`; Sum/Max/Min/And/Or/Xor and any/all; no product, no `CountBits` |
| 19 | Wave: Prefix: `WavePrefixSum/Product/Min/Max/BitAnd/..`, `WavePrefixCountBits` | No -> **Partial** | `Stmt::Collective` shuffle/vote/ballot/reduce/scan/broadcast (0.42.3; cpu-c 0.42.4); `tests/kernel_collective.rs`, cpu-c `kernel_collective_emit_agrees.rs`; scan Sum/Max/Min/And/Or/Xor; no product |
| 20 | Wave: Shuffles/broadcast: `WaveReadLaneAt/First`, `WaveBroadcastLaneAt`, `WaveShuffle`, `WaveGet | No -> **Yes** | `Stmt::Collective` shuffle/vote/ballot/reduce/scan/broadcast (0.42.3; cpu-c 0.42.4); `tests/kernel_collective.rs`, cpu-c `kernel_collective_emit_agrees.rs` |
| 21 | Wave: Ballot/vote/match: `WaveActiveBallot`, `WaveMatch`, lane masks (`WaveGetLaneEq/Ge/Gt/Le/Lt | No -> **Partial** | `Stmt::Collective` shuffle/vote/ballot/reduce/scan/broadcast (0.42.3; cpu-c 0.42.4); `tests/kernel_collective.rs`, cpu-c `kernel_collective_emit_agrees.rs`; ballot yes, `WaveMatch` no |
| 24 | Wave: `WaveGetNumWaves`/`WaveGetWaveIndex` | No -> **Yes** | `ExecId::{LaneId,SubgroupId,SubgroupSize}`; `tests/kernel_exec.rs` (no `sm_id`, clock or other special registers) |
| 28 | Atomics: `Interlocked{Add,And,Or,Xor,Min,Max,Exchange,CompareExchange,CompareStore}` on int32/int64 | Partial -> **Yes** | `Stmt::AtomicRmw`/`AtomicCas` with `old`, order and scope (0.41.x-0.42.7); `tests/kernel_atomic_{rmw,fetch,cas,order,shared}.rs` |
| 29 | Atomics: `Atomic<T>` with MemoryOrder {Relaxed,Acquire,Release,AcquireRelease,SeqCst}, load/store/e | No -> **Partial** | order+scope RMW (an `Or` of 0 reads); no atomic load/store statements; a spin on one is a named `CrossBlockWait` (0.42.7) |
| 31 | Atomics: 64-bit atomics (HLSL SM6.6, SPIR-V Int64Atomics) | No -> **Yes** | `Stmt::AtomicRmw`/`AtomicCas` with `old`, order and scope (0.41.x-0.42.7); `tests/kernel_atomic_{rmw,fetch,cas,order,shared}.rs`; integer atomics on `I64`/`U64` elements |
| 32 | Atomics: Atomics on groupshared | No -> **Yes** | `Stmt::AtomicRmw`/`AtomicCas` with `old`, order and scope (0.41.x-0.42.7); `tests/kernel_atomic_{rmw,fetch,cas,order,shared}.rs`; `tests/kernel_atomic_shared.rs` |
| 35 | Dtypes: Vector and matrix types (`float4`, `float3x3`, swizzles, vector ops, `mul`, `dot`, `cross` | No -> **Partial** | `ScalarExpr::{Pack,Extract,BitcastPacked}` and the packed `Select` mask (0.31.x); no vector arithmetic ops beyond lane-wise `Select` |
| 39 | Dtypes: Bitcast (`asfloat/asuint/asint/asdouble`), `reinterpret<T>` | Partial -> **Yes** | `ScalarExpr::Bitcast` (0.24.0) |
| 42 | Control flow: Loops (`for/while/do`), dynamic trip counts, `break/continue`, multi-level break, early re | No -> **Yes** | `Stmt::{If,For,While,Break,Continue,Return}` and `Let`/`Assign` locals; `tests/kernel_layer.rs`, `kernel_resume_agrees.rs`, `kernel_eval_semantics.rs` |
| 43 | Control flow: Divergent `if/else` with side effects (stores/atomics in branches) | Partial -> **Yes** | `Stmt::{If,For,While,Break,Continue,Return}` and `Let`/`Assign` locals; `tests/kernel_layer.rs`, `kernel_resume_agrees.rs`, `kernel_eval_semantics.rs` |
| 44 | Control flow: Function calls, recursion-free helper functions, `inline` | No -> **Yes** | `Stmt::Call`, `Module`, `check_module`, run in the oracle and spelled in cpu-c; `tests/kernel_module_run.rs`, `kernel_call_shape.rs` |
| 50 | Numerics: Fast vs precise (`-fp-mode`/`FloatingPointMode`), per-function precision; `precise` qualif | No -> **Partial** | `ScalarExpr::Qualified` rounding/ftz/approx qualifiers (0.25.0); `docs/ir-numerics-qualifiers.md` |
| 51 | Numerics: Rounding mode per op (RTZ/RTE on conversions, `OpFAdd` rounding decoration, cvt.rn) | No -> **Yes** | `ScalarExpr::Cast`/`Bitcast` with `RoundMode` and `Saturation` (0.24.0); `docs/ir-typed-expressions.md`; rounding on `Cast` and `Qualified` |
| 55 | Vulkan/SPIR-V: Subgroup size control / required subgroup size, `[WaveSize]` | No -> **Yes** | `LaunchShape::subgroup` range (`with_subgroup`) and `SubgroupSize`; `tests/kernel_exec.rs` |

### ThunderKittens (`docs/ir-union-surveys/thunderkittens.md`): 10 rows changed of 58

| # | Capability | Status | Evidence at `a634b09` |
|---|---|---|---|
| 4 | Tile types: Shared tile `st<T,R,C,swizzle,swizzle_bytes>`; 0/32/64/128B swizzle, auto-selected | No -> **Partial** | `swizzle` in `layout_algebra` (a functor over layouts); no shared-memory tile consumer, no `ComposedLayout` with smem atoms |
| 10 | Memory spaces: Register / shared / tensor-memory / global as distinct allocation classes | No -> **Partial** | `Place::Alloc` in `MemorySpace::Shared` with a static `AllocDecl`; `tests/kernel_launch.rs`, `kernel_atomic_shared.rs`; cpu-c spells it; global/shared/private/constant allocation classes; no tensor memory |
| 12 | Scope: thread / warp / warpgroup (4 warps) / block / grid scope on each op | No -> **Partial** | `LaunchShape::cluster` + `Cluster` barrier/atomic/fence scope (0.42.15); `tests/kernel_cluster.rs`; no cluster ids, no distributed shared memory; scopes `Subgroup`/`Block`/`Cluster`/`Device`; no warpgroup |
| 13 | Scope: Warp role/specialization (consumer = low warps, producer = high warps; `warpid()`, `warpgr | No -> **Partial** | `ExecId::{LaneId,SubgroupId,SubgroupSize}`; `tests/kernel_exec.rs` (no `sm_id`, clock or other special registers); branch on `SubgroupId`; no `setmaxnreg`, no named barriers |
| 14 | Scope: Leader election (`elect_leader`, "only lane 0 issues") | No -> **Partial** | `ExecId::{LaneId,SubgroupId,SubgroupSize}`; `tests/kernel_exec.rs` (no `sm_id`, clock or other special registers); "only lane 0 issues" is `if LaneId == 0` |
| 29 | Sync: Block / warp sync, cluster barrier, fences (`fence.proxy.async`) | No -> **Partial** | `LaunchShape::cluster` + `Cluster` barrier/atomic/fence scope (0.42.15); `tests/kernel_cluster.rs`; no cluster ids, no distributed shared memory; block/warp/cluster barrier + `Fence`; no proxy fences |
| 36 | Reduction: Warp-level reduction across lanes inside the group op (shuffle) is implicit | Partial -> **Yes** | `Stmt::Collective` shuffle/vote/ballot/reduce/scan/broadcast (0.42.3; cpu-c 0.42.4); `tests/kernel_collective.rs`, cpu-c `kernel_collective_emit_agrees.rs` |
| 38 | Conversion: shared<->register, global<->register direct loads/stores | No -> **Partial** | `Place::Alloc` in `MemorySpace::Shared` with a static `AllocDecl`; `tests/kernel_launch.rs`, `kernel_atomic_shared.rs`; cpu-c spells it; element-level loads/stores between spaces; no tile-level moves |
| 52 | Dtypes: fp4 e2m1 (packed `fp4e2m1_2`; raw rejected on SM10X) | No -> **Partial** | `ElementKind::{F4E2M1,F6E2M3,F6E3M2}`: storage and key identity (0.29.0), `Cast` rows (0.31.7); stored-result layout still open |
| 53 | Dtypes: Packed 2-element register types (`T2`: bf16_2, half_2, float2) | No -> **Partial** | `ScalarExpr::{Pack,Extract,BitcastPacked}` and the packed `Select` mask (0.31.x); no vector arithmetic ops beyond lane-wise `Select` |

### Triton (`docs/ir-union-surveys/triton.md`): 21 rows changed of 70

| # | Capability | Status | Evidence at `a634b09` |
|---|---|---|---|
| 1 | Program model: `tl.program_id(axis)`, `tl.num_programs(axis)` (grid of SPMD block programs) | No -> **Yes** | `ExecId::{ThreadIdx,BlockIdx,BlockDim,GridDim}` on three axes (`exec.rs`); `tests/kernel_exec.rs`, `kernel_launch.rs` |
| 4 | Program model: Device function calls (`tt.call`, `@triton.jit` helper fns, `combine_fn` lambdas) | No -> **Partial** | `Stmt::Call`, `Module`, `check_module`, run in the oracle and spelled in cpu-c; `tests/kernel_module_run.rs`, `kernel_call_shape.rs`; helper functions yes; `combine_fn` lambdas no |
| 20 | Reduce: `tl.sum/max/min/prod? (no prod; reduce combine_fn)/argmax/argmin/any/all/reduce_or/xor_sum | Partial -> **Yes** | `ReduceOp::{Sum,Mean,Max,Min,Prod,And,Or,Xor,MaxNum,MinNum,ArgMax,ArgMin}`; `tests/reduce_bit_ops.rs`, `reduce_maxnum.rs`, `reduce_argmax.rs`, `kernel_reduction_agrees.rs`; sum/max/min/prod/argmax/argmin/any/all/or/xor |
| 27 | Atomics: `atomic_add/max/min` (returns OLD value) with `sem` (acquire/release/acq_rel/relaxed) and  | Partial -> **Yes** | `Stmt::AtomicRmw`/`AtomicCas` with `old`, order and scope (0.41.x-0.42.7); `tests/kernel_atomic_{rmw,fetch,cas,order,shared}.rs` |
| 28 | Atomics: `atomic_cas`, `atomic_xchg`, `atomic_and`, `atomic_or`, `atomic_xor` | No -> **Yes** | `Stmt::AtomicRmw`/`AtomicCas` with `old`, order and scope (0.41.x-0.42.7); `tests/kernel_atomic_{rmw,fetch,cas,order,shared}.rs` |
| 29 | Atomics: `atomic_load`, `atomic_store`, `atomic_poll(expected, sem, scope, timeout_ns)` (inter-prog | No -> **Partial** | order+scope RMW (an `Or` of 0 reads); no atomic load/store statements; a spin on one is a named `CrossBlockWait` (0.42.7) |
| 32 | Dtypes: int1/int8/16/32/64, uint8/16/32/64 | Partial -> **Yes** | `ElementKind::{I8,I16,I32,I64,U8,U16,U32,U64,Bool}` |
| 36 | Dtypes: fp4: `float4e2m1x2` (two e2m1 packed in a byte) | No -> **Partial** | `ElementKind::{F4E2M1,F6E2M3,F6E3M2}`: storage and key identity (0.29.0), `Cast` rows (0.31.7); stored-result layout still open |
| 37 | Casts: `tl.cast(x, dtype, fp_downcast_rounding={"rtne","rtz"}, bitcast=False)` | No -> **Yes** | `ScalarExpr::Cast`/`Bitcast` with `RoundMode` and `Saturation` (0.24.0); `docs/ir-typed-expressions.md` |
| 38 | Casts: `tt.fp_to_fp` rounding attr, `tl.convert saturation / overflow_mode` (UNVERIFIED: overflow | No -> **Yes** | `ScalarExpr::Cast`/`Bitcast` with `RoundMode` and `Saturation` (0.24.0); `docs/ir-typed-expressions.md` |
| 40 | Packed/vector: `tl.map_elementwise(scalar_fn, *args, pack)`, `ttng.packed_arith` | No -> **Partial** | `ScalarExpr::{Pack,Extract,BitcastPacked}` and the packed `Select` mask (0.31.x); no vector arithmetic ops beyond lane-wise `Select` |
| 41 | Control: `for i in tl.range(...)` / `tl.static_range` loops with loop-carried tiles (the heart of e | No -> **Yes** | `Stmt::{If,For,While,Break,Continue,Return}` and `Let`/`Assign` locals; `tests/kernel_layer.rs`, `kernel_resume_agrees.rs`, `kernel_eval_semantics.rs` |
| 42 | Control: `if`/`while` on scalar (program-uniform) predicates, early return, data-dependent loop bou | No -> **Yes** | `Stmt::{If,For,While,Break,Continue,Return}` and `Let`/`Assign` locals; `tests/kernel_layer.rs`, `kernel_resume_agrees.rs`, `kernel_eval_semantics.rs` |
| 44 | Schedule: `num_warps` (default 4) | No -> **Partial** | `LaunchShape { block, grid }` (the `[numthreads]` equivalent) and the subgroup range; block size is in `LaunchShape`; `num_warps` as a compile hint is not |
| 46 | Schedule: `num_ctas` (cluster size, SM90+) | No -> **Partial** | `LaunchShape::cluster` + `Cluster` barrier/atomic/fence scope (0.42.15); `tests/kernel_cluster.rs`; no cluster ids, no distributed shared memory |
| 52 | Coop: Shared-memory buffers (`ttg.local_alloc/load/store/dealloc`, memdesc_index/subslice/trans/ | No -> **Partial** | `Place::Alloc` in `MemorySpace::Shared` with a static `AllocDecl`; `tests/kernel_launch.rs`, `kernel_atomic_shared.rs`; cpu-c spells it; `local_alloc/load/store`; no memdesc views |
| 53 | Coop: Shared-mem gather/scatter/atomic scatter (`ttg.local_gather/local_scatter/local_atomic_sca | No -> **Yes** | `Place::Alloc` in `MemorySpace::Shared` with a static `AllocDecl`; `tests/kernel_launch.rs`, `kernel_atomic_shared.rs`; cpu-c spells it; indexed shared loads/stores/atomics |
| 54 | Coop: CTA barriers `tl.debug_barrier`, `ttg.barrier` (CTA or warp scope), `__syncthreads` | No -> **Yes** | `Stmt::Barrier` at `Subgroup` and `Block` scope + the uniformity rule (`kernel_uniform.rs`); `tests/kernel_barrier.rs`, `kernel_barriers_run.rs` |
| 60 | Coop: Clusters / multi-CTA (`num_ctas`, cluster_barrier, multicast TMA, DSMEM, CLC launch contro | No -> **Partial** | `LaunchShape::cluster` + `Cluster` barrier/atomic/fence scope (0.42.15); `tests/kernel_cluster.rs`; no cluster ids, no distributed shared memory |
| 61 | Coop: Layouts: blocked / mma / dot_operand / shared (swizzle, linear layouts), `ttg.convert_layo | No -> **Partial** | `unpopped-vocab/src/layout_algebra.rs`: hierarchical `IntTuple`/`Layout`, `composition`, `complement`, `logical_divide`, `logical_product`, `coalesce`, inverses, `swizzle`, `select/take/slice/tile_to_shape` with a property-test suite (U6, ROADMAP: done, issue #117 closed 2026-11-02); layout vocabulary; no mma/dot-operand/linear-layout conversion op |
| 62 | Coop: Warp/block collectives (shuffle, ballot, vote) | Partial -> **Yes** | `Stmt::Collective` shuffle/vote/ballot/reduce/scan/broadcast (0.42.3; cpu-c 0.42.4); `tests/kernel_collective.rs`, cpu-c `kernel_collective_emit_agrees.rs` |

