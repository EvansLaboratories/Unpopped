# Re-score of CUDA C++ (`docs/ir-union-surveys/cuda.md`) at b21af35c (Unpopped 0.56.12)

Row numbers count the survey's table rows top to bottom (row 1 = global memory pointers, row 95 = shared-memory limits), as the 2026-11-15 file does; the survey has 95 rows and no parenthetical-status row, so the 13 uncounted rows of the overall write-up are not in this survey (none listed).

Status at a634b09: Yes/Partial/No/n-a = 18/30/47/0 = 95.
Status at 78c7d38e: Yes/Partial/No/n-a = 27/40/28/0 = 95 (reproduced from the survey statuses plus the two earlier change lists: 18/30/47 and 27/40/28 both recomputed exactly).
Status at b21af35c: Yes/Partial/No/n-a = 30/51/14/0 = 95.

Headline rule: a row is Yes only when IR = Yes and ORACLE = Yes. ORACLE column: Yes = a test in the oracle runs what the IR expresses; Partial = only part runs; No = expressible but nothing runs it; n-a = nothing is expressed to run (IR = No) or the row has no run-time meaning (launch bounds, host). cpu-c (the serial C reference) refusing a statement is NOT counted against ORACLE (it cannot block a thread: wgmma, tcgen05, mbarrier, CopyBulk, named barriers, ldmatrix are oracle-only by design).

Rows changed: 16 (15 up, 1 down): up 7, 9, 16, 19, 21, 22, 23, 39, 40, 41, 45, 46, 77, 79, 95; down 60. Rows whose status is unchanged but whose 11-15 evidence was wrong or stale: 4, 50, 59, 76.
Rows examined: all 95. The 65 rows not Yes at b21af35c and the 28 rows No at 78c7d38e were opened against the code. The 27 rows Yes at 78c7d38e were checked for existence of every cited symbol and test file/function (grep, listed in the last section), not re-read line by line. Skipped: nothing skipped by class; rows 8, 14, 16, 27, 92 (host or launch attributes) were judged on their kernel-visible part only.
Nothing was built or run. Every status rests on a symbol and a test name that was grepped in the b21af35c tree.

## 1. Rows changed or found wrong

| # | Capability | 78c7d38e -> b21af35c (IR / ORACLE) | Evidence (file:symbol ; test) |
|---|---|---|---|
| 7 | tensor memory (TMEM) alloc/ld/st/cp | No -> Partial (Partial / Yes) | `exec.rs:MemorySpace::Tensor`, `kernel.rs:Stmt::Tcgen05Ld`, `Tcgen05Mma` ; `tests/kernel_tcgen05.rs::every_lane_reads_its_row_of_the_reference_product`. alloc is static, no `tcgen05.st/cp/shift`, one `ld` shape (doc on `Tcgen05Mma`). |
| 9 | cache-policy / L2 residency | No -> Partial (Partial / Yes) | `kernel.rs:CacheHint`, `Stmt::CopyAsync{cache,l2_prefetch}` ; `tests/kernel_unroll_copy_forms.rs::cache_hints_and_l2_prefetch_do_not_change_the_result`. Only on `CopyAsync`; `Load`/`Store` have no hint field; no `createpolicy`/`prefetch`. Hints are non-semantic (oracle ignores them). |
| 16 | programmatic dependent launch | No -> Partial (Partial / Partial) | `kernel.rs:Stmt::GridDepWait`, `GridDepLaunchDependents` ; `tests/kernel_grid_dep.rs::a_kernel_with_the_statements_computes_what_it_computes_without_them`. Both are no-ops in the oracle (one launch has no prerequisite grid): the cross-grid rule is not modelled and not tested. |
| 19 | named/partial barriers (`bar.sync id,count`) | No -> Yes (Yes / Yes) | `kernel.rs:BarrierKind::{Named,NamedArrive}`, `oracle/kernel_named.rs:NamedState`, `kernel_check.rs` (named id/count rules) ; `tests/kernel_named_barrier.rs::a_producer_group_hands_a_tile_to_a_consumer_group_twice`, `::the_participants_must_agree_on_the_count`. |
| 21 | mbarrier / `cuda::barrier` | No -> Partial (Partial / Yes) | `kernel.rs:Stmt::{MbarInit,MbarArrive,MbarWait}`, `oracle/kernel_mbar.rs` ; `tests/kernel_mbar.rs::the_phase_completes_when_every_expected_arrival_has_come`, `::the_phase_parity_alternates`, `::a_phase_nobody_completes_is_a_deadlock`. Missing: arrive_and_drop, a non-blocking try_wait, completion function. Absent control: grep `MbarDrop|ArriveDrop|arrive_and_drop|MbarTry` in kernel.rs/exec.rs = 0; the same files hold 4 `MbarArrive` hits. |
| 22 | transaction-count barriers | No -> Yes (Yes / Yes) | `kernel.rs:Stmt::MbarArrive{tx}` (arrive.expect_tx), `Stmt::MbarCompleteTx` ; `tests/kernel_mbar.rs::the_transaction_count_holds_the_phase_open_until_the_bytes_land`. |
| 23 | split arrive/wait | No -> Yes (Yes / Yes) | `Stmt::MbarArrive` and `Stmt::MbarWait{parity}` are separate statements ; `tests/kernel_mbar.rs::a_consumer_blocks_until_the_producer_arrives_and_then_sees_its_data`. Judgement call: the wait is by phase parity, there is no token value (see section 5). |
| 39 | `ldmatrix` / `stmatrix` / `movmatrix` | No -> Partial (Partial / Yes) | `kernel.rs:Stmt::MatrixLoad`, `Stmt::MatrixStore` ; `tests/kernel_matrix_load.rs::every_lane_receives_the_element_ptx_says`, `tests/kernel_matrix_store.rs::every_tile_cell_holds_the_element_ptx_says`. `movmatrix` absent (grep `movmatrix` in src = 0; `ldmatrix` finds kernel.rs). The stmatrix row rule is by symmetry, not from a second source (kernel.rs doc, ROADMAP U8b). |
| 40 | `wgmma.mma_async` (+ fence/commit/wait) | No -> Partial (Partial / Yes) | `kernel.rs:Stmt::Wgmma`, `WgmmaCommit`, `WgmmaWait`, `exec.rs:Scope::Warpgroup`, `oracle/kernel_wgmma.rs` ; `tests/kernel_wgmma.rs::every_fragment_value_is_the_reference_product`, `::the_result_lands_only_at_the_wait_that_covers_its_group`. Not modelled (doc on `Wgmma`): descriptors, swizzle, register-A, transposed operands, 8-bit/tf32 inputs, `.sp`, f16 accumulate. No `wgmma.fence`. |
| 41 | `tcgen05.mma` (+ commit, TMEM accumulators) | No -> Partial (Partial / Yes) | `kernel.rs:Stmt::Tcgen05Mma`, `Tcgen05Commit`, `Tcgen05Ld` ; `tests/kernel_tcgen05.rs::the_result_needs_the_commit_and_an_acquire_on_its_barrier`, `::a_later_product_accumulates_when_scale_d_is_set`. Not modelled (doc on `Tcgen05Mma`): descriptors, swizzle, alloc/dealloc, TMEM-A, `cta_group::2`, `.sp`/`.ws`, 8-bit/tf32/block-scaled kinds. |
| 45 | `cuda::memcpy_async` / `cp.async` (4/8/16B, L1 bypass at 16B) | Partial -> Yes (Yes / Yes) | `kernel.rs:Stmt::CopyAsync{valid,cache,l2_prefetch}`, `kernel_check/async_copy.rs` (`.cg` only at 16 bytes) ; `tests/kernel_unroll_copy_forms.rs::valid_reads_that_many_elements_and_zero_fills_the_rest`, `::the_checker_refuses_what_a_copy_form_cannot_be`; `tests/kernel_async_copy.rs`. Not covered by this row: `cp.async.mbarrier.arrive`. |
| 46 | TMA 1D bulk / tensor / prefetch / reduce-copy | No -> Partial (Partial / Yes) | `kernel.rs:Stmt::CopyBulk` (1-D, global to shared, completes on an mbarrier) ; `tests/kernel_copy_bulk.rs::a_consumer_that_waits_sees_the_copied_tile`, `::bytes_announced_but_not_delivered_leave_the_phase_open`. No tensor-map copy, no prefetch, no reduce-copy, no shared-to-global store. |
| 60 | `__nv_fp8_e4m3/e5m2/e8m0` | Yes -> Partial (Yes / Partial) **found overstated** | e4m3 and e5m2 run (`oracle/value.rs:raw_to_f64`, `tests/fp8_codecs.rs`). `ElementKind::F8E8M0` exists but is a scale type the plan gate refuses (`plan/dtype_float.rs:check_dtype_is_a_compute_dtype`; `tests/oracle_reachable_on_every_admitted_dtype.rs`, "must be refused BY THE GATE") and `raw_to_f64` panics on it (arm `other => panic!`). No test decodes an E8M0 value; the block-scale test uses `F32` scales (`tests/kernel_mma_flags.rs:48`). Positive control: the same `raw_to_f64` handles `Fp8E5M2`. Stale at a634b09 and 78c7d38e, not a regression. |
| 77 | `#pragma unroll`, `__noinline__` | No -> Partial (Partial / Yes) | `kernel.rs:Unroll{None,Full,Count}`, `Stmt::For{unroll}` ; `tests/kernel_unroll_copy_forms.rs::the_unroll_hint_does_not_change_the_result`, `::the_unroll_hint_survives_serialisation_and_a_zero_count_is_refused`. `__noinline__`: no field (judgement call, section 5). |
| 79 | warp specialization (producer/consumer, `pipeline_role`, `setmaxnreg`) | No -> Partial (Partial / Yes) | Roles are a recipe by ruling (ROADMAP U8 S4); the recipe now runs: `tests/kernel_named_barrier.rs::a_producer_group_hands_a_tile_to_a_consumer_group_twice` (producer and consumer thread groups on named barriers), `kernel.rs:Stmt::NamedArrive` via `BarrierKind`, `exec.rs:Scope::Warpgroup`. `setmaxnreg` absent (grep `setmaxnreg` across crates: only `__maxnreg__` interface data in `kernel_interface.rs`). |
| 95 | shared-memory limits per arch | No -> Partial (Partial / No) **correction, predates a634b09** | `capability.rs:cuda_capabilities`, `TargetCapabilities::max_shared_mem_per_block/per_sm` (48 KiB sm_61 to 227 KiB sm_90, 99 KiB sm_120, table in git since 2026-08-27) ; unit tests `capability.rs::tests::the_table_discriminates_between_capabilities`, `::shared_memory_is_not_monotonic_in_the_architecture_number`. Nothing consults it: grep `max_shared_mem` outside capability.rs = 0, so no kernel's allocations are checked against it (ORACLE No). |
| 4 | `__constant__` memory, `__grid_constant__` params | Partial -> Partial (Partial / Partial), evidence updated | `__grid_constant__` struct params now exist: `kernel.rs:ParamKind::Struct`, `Param::structure`, `kernel_interface.rs` `params_block` ; `tests/kernel_struct_param.rs::a_kernel_reads_each_field_as_its_local`. `__constant__` still has no initializer: `AllocDecl` (exec.rs:283) has only name/space/elem/extent (grep `pub init` = 0, control `pub extent` = 1), oracle refuses a read (`oracle/kernel_mem_ops.rs:90`). |
| 50 | vectorized access | Partial -> Partial (Partial / Partial), evidence updated | 8/16-lane `Pack`/`BitcastPacked` type rules (`tests/packed_values.rs::eight_and_sixteen_lane_packs_are_packed_up_to_128_bits`); a packed LOCAL still traps in the kernel oracle (`tests/kernel_packed.rs::a_packed_local_is_a_pinned_gap`); no vector load/store statement. |
| 59 | packed `half2`/`float4` | Partial -> Partial (Partial / Partial), 11-15 note wrong | The 11-15 note "no vector arithmetic beyond lane-wise Select" is stale: lane-wise operators on packed values exist since 0.27.2 (commit d63137ff, 2026-10-26): `tests/packed_values.rs::lane_wise_arithmetic_keeps_the_packed_type`, `tests/kernel_packed.rs::a_lane_wise_operator_applies_to_the_same_lane`. Still Partial: packed locals trap, no packed memory access. |
| 76 | pointer arithmetic, aliasing, arbitrary address computation | Partial -> Partial (Partial / Yes), evidence updated | `kernel.rs:Place::{Raw,AllocRaw}` (byte-offset places) ; `tests/kernel_raw_place.rs::a_raw_load_reads_the_named_bytes_at_the_named_width`, `tests/kernel_alloc_raw_place.rs::a_raw_load_from_an_allocation_reads_the_named_bytes_at_the_named_width`. No pointer value type (grep `Place::Ptr` in kernel.rs = only the "deliberately not here" comment). |

Also stale, no status effect: the doc comment of `Stmt::Barrier` (`kernel.rs` lines 426-428) still says "Only `BarrierKind::Full` exists"; `BarrierKind` has `Named` and `NamedArrive` (line 1102).

## 2. Rows still not a full Yes at b21af35c (65)

PRs = estimated ~150k-diff-character PRs for the Unpopped lane. "with #n" = shares one implementation. Blocker wording for PATENT follows the brief ("outside as read" / "NOT CLEARED"); "no row" = no U-number in ROADMAP 6.7.

| # | Capability | IR | ORACLE | What is missing | Blocker | PRs |
|---|---|---|---|---|---|---|
| 1 | global pointers, `__restrict__` | Partial | No | restrict is emitter-set interface data (`BufferGuarantee::restrict`); no IR alias fact, no oracle check of a violation | READY, no row | 1 |
| 4 | `__constant__`, `__grid_constant__` | Partial | Partial | initializer on `AllocDecl` (grid-constant structs done) | TRIGGER: U40 "scheduled on demand; a lifted kernel or a lane needs a table" | 1-2 |
| 7 | tensor memory | Partial | Yes | `tcgen05.alloc/dealloc`, `st`, `cp`, `shift`, other `ld` shapes | PATENT: U7 S5 section, "outside as read" for the read claims, US10338919B2/WO2018217360A1/US12499052 claims unread, NOT CLEARED; re-read before alloc modelling | with #41 |
| 8 | managed/unified memory | No | n-a | `__managed__` is a host allocation attribute | OUTSCOPE: host allocation, not a kernel body concern | 0 |
| 9 | cache-policy / L2 residency | Partial | Yes | `createpolicy`, `prefetch`, `ld`/`st` cache hints (CopyAsync only today) | READY, no row (U13 built the CopyAsync part) | 1 |
| 11 | thread-block clusters | Partial | Yes | cluster rank/dims leaves (rank is `BlockIdx` arithmetic) | READY, no row | 1 |
| 13 | persistent kernels, tile schedulers, stream-K, CLC | No | n-a | no scheduler node or conformance kernel; CLC (`try_cancel`) | DECISION: P9 schedule language "parked by the PM 2026-10-24"; recipe-not-node ruling in U8 S4 | 1-2 |
| 14 | `__launch_bounds__`, `__maxnreg__` | Partial | n-a | hint on `KernelFn`, occupancy model | OUTSCOPE: performance attribute, interface data already carried | 0 |
| 15 | `setmaxnreg` | No | n-a | a `LaunchShape` field or hint | TRIGGER: `docs/options-u8-copy-atoms.md` "launch attributes ... ride on `LaunchShape` fields when a backend needs them" | 1 |
| 16 | programmatic dependent launch | Partial | Partial | cross-grid prerequisite rule; oracle has one launch | OUTSCOPE: the edge is a host stream/graph property (`kernel.rs` doc on `GridDepWait`); 1-2 PRs only if a multi-grid oracle is wanted | 0 |
| 17 | dynamic parallelism | No | n-a | device-side launch statement and nested-grid oracle | DECISION: union member vs scope, no ROADMAP row, no options note | 3 |
| 20 | `__syncwarp(mask)` | Partial | Yes | lane mask on the `Subgroup` barrier | READY, no row | 1 |
| 21 | mbarrier / `cuda::barrier` | Partial | Yes | arrive_and_drop, non-blocking try_wait; completion function (held: US11803380B2 claim 15 open, PATENT-SWEEP "Publish-prep searches") | READY, no row (U8c) | 1 |
| 24 | fences | Partial | Yes | `fence.proxy.async`, `.tensormap`, `tcgen05.fence` (only scope+order) | READY, no row | 1 |
| 25 | async groups | Partial | Yes | bulk groups (`cp.async.bulk.commit_group/wait_group`); cp.async and wgmma groups done | TRIGGER: U8c "Baracuda's sm_90 emitter asks, or a descriptor operand kind lands" | with #46 |
| 26 | `cuda::pipeline` object | Partial | Yes | acquire/release/roles/`quit` object (a recipe runs) | DECISION: PM 2026-11-04 Q1-Q6 ruled pipelines recipes, "stage objects stay a recipe" (U8 S4, U13); needs a reversal | 2 |
| 27 | memory sync domains | No | n-a | launch attribute | OUTSCOPE: host launch attribute | 0 |
| 29 | vote/ballot, `activemask`, `match` | Partial | Yes | `__activemask`, `match_any/all` | READY, no row | with #20 |
| 31 | cooperative_groups partitions | Partial | Yes | labeled/binary partition, `tiled_partition<N>` other than the launch subgroup | READY, no row | 2 |
| 34 | `invoke_one`, `elect.sync` | No | n-a | an elect predicate (`If` on `LaneId == 0` is the workaround) | READY, no row | with #20 |
| 35 | block-level reduce/scan/sort | Partial | Yes | block sort as a collective | READY, no row | 1 |
| 36 | `wmma` fragments | Partial | Yes | m16n16k16, m8n32k16, tf32 m16n16k8, u4/b1/f64 atoms (data) | READY: U7 (S2 atoms) | 2 (with #37) |
| 37 | `mma.sync` PTX | Partial | Yes | m16n8k8 tf32, fp8, f16-accumulate, k64 atoms | READY: U7 | with #36 |
| 38 | `mma.sp` | Partial | Yes | PTX `mma.sp` lane tables, `ordered_metadata` | TRIGGER: U7 S3 / U8b "second independent lane-table source" | 1 |
| 39 | ldmatrix/stmatrix/movmatrix | Partial | Yes | `movmatrix`; second source for the b16 8x8 distribution | TRIGGER: U8b "a second independent lane-table source is found" | 1 |
| 40 | `wgmma.mma_async` | Partial | Yes | descriptors, swizzle, register-A, transposed, 8-bit/tf32/sparse, f16 acc, `wgmma.fence` | PATENT: U7 S4 section, US20230289398A1 outside as read, US12204897B2/US12710961B2 NEEDS REVIEW (ruled note + decline), US12020035B2/US20230315655A1 unread, NOT CLEARED; re-read trigger descriptor/swizzle | 2-3 |
| 41 | `tcgen05.mma` | Partial | Yes | descriptors, swizzle, alloc, TMEM-A, `cta_group::2`, `.sp`/`.ws`, 8-bit/tf32/block-scaled kinds | PATENT: U7 S5 section (as #7) | 3 |
| 42 | block-scaled MMA | Partial | Yes | PTX `.kind::mxf4nvf4`/`scale_vec` constructor, checked scale lane table | TRIGGER: U7 S3 "no PTX sparse/scaled constructor until its lane tables are checked against a second source" | 1 |
| 44 | higher-rank/mixed-dtype/multi-K contractions | Partial | Yes | plan-layer `Access::Contraction` generalisation (kernel layer only today); pointer-array needs U9c | DECISION: D2 (Fuel/KISS), U9 "plan-layer generalization (B) stays held on D2" | 3 |
| 46 | TMA 1D/tensor/prefetch/reduce | Partial | Yes | `cp.async.bulk.tensor` (<=5D), prefetch, reduce-copy, shared-to-global store | TRIGGER: U8c (as #25); PATENT-SWEEP U8c slice 2 re-read trigger "a tensor-map copy is added" | 1 |
| 47 | TMA multicast, smem-to-smem cluster copy | No | n-a | multicast mask on a bulk copy; shared::cluster copy | TRIGGER: U8c | 1 |
| 48 | tensor maps | No | n-a | descriptor operand kind (`ParamKind` has Buffer/Scalar/Struct only), encode, `tensormap.replace` | TRIGGER: U8c "or a descriptor operand kind lands" | 2 |
| 49 | swizzle modes | Partial | Yes | swizzle as a `Place` addressing mode; TMA swizzle enum goes with #48 | READY, no row | 1 |
| 50 | vectorized access | Partial | Partial | vector load/store statement; packed locals | PATENT: U15 section, "NOT CLEARED, UNREAD" (US11262989B2, US9052888B2, WO2013070616A1 claims first) before any vector load/store lowering | 2 |
| 51 | `red.async`, `multimem`, `st.async` | Partial | Yes | `red.async`/`st.async` (multimem done) | READY, no row (extends U12) | 1-2 |
| 54 | hierarchical layouts | Partial | Yes | `View::Layout` over nested layouts (nested is declined) | READY, no row | 1 |
| 55 | row/col operand layout for MMA | Partial | Yes | operand major-ness on `Access::Contraction` | DECISION: D2 (as #44) | with #44 |
| 56 | interleaved layouts, nibble order | Partial | Yes | named interleave layouts; nibble order fixed low-first | LANE: KISS (U1b: packing is convention, UNVERIFIED against CUTLASS/Mojo/Triton) | 1 |
| 59 | packed `half2`/`float4` | Partial | Partial | packed locals trap; scalarize pass (~250 lines, ROADMAP) | TRIGGER: U15 "Parked 2026-11-21 (pm-kernels): no consumer; revive only if an emitter must keep a 128-bit vector as one access" | 1 |
| 60 | fp8 e4m3/e5m2/e8m0 | Yes | Partial | oracle decode of `F8E8M0` as a scale operand and a test | READY, no row (KISS-CLASSIFY 6.1-0013: a scale, never a compute dtype) | 1 |
| 61 | fp6 | Partial | Partial | decode exists only as a `Cast` intermediate (`oracle/value.rs`, "no buffer layout yet"); no kernel load/store | LANE: KISS (U1b packing unverified; fp4/fp6 rows are D5 territory) | 2 |
| 62 | fp4 / NVFP4 | Partial | Partial | as #61; NVFP4 scale pairing | LANE: KISS | with #61 |
| 64 | `__float128` | No | n-a | `ElementKind` row, oracle arithmetic | LANE: KISS (ElementKind rows are KISS-owned; grep `F128` in element.rs = 0, control `Complex128` = 12) | 1 |
| 65 | 128-bit integers (+128-bit atomics) | No | n-a | `I128`/`U128` | LANE: KISS (as #64) | 1 |
| 69 | `atomicInc/Dec` | Partial | Yes | no `AtomicOp::{Inc,Dec}` (enum is Add, Max, Min, And, Or, Xor, Exchange) | READY, no row | 2 (with #70, #72) |
| 70 | `cuda::atomic` load/store acquire/release | Partial | Yes | atomic load and store statements (only RMW and CAS carry order/scope) | READY, no row | with #69 |
| 72 | fp32 atomic add denormal flush | No | n-a | a numerics note/flag on atomic combine (grep `denormal|flush` in kernel.rs = 0) | READY, no row | with #69 |
| 75 | recursion, templates, `constexpr` | Partial | Yes | recursion is refused by design (`KernelError::CallCycle`, `tests/kernel_call_tuple.rs`); templates/constexpr are frontend-resolved | DECISION: recursion unruled; U19 Q2 option B (compile-time evaluator) pending | 2 |
| 76 | pointer arithmetic / aliasing | Partial | Yes | pointer values (`Place::Ptr`), alias analysis | PATENT: U9c section, "no claim found; NOT CLEARED", re-read before a pointer-valued type (EP2659362A2); ROADMAP U9c trigger already met | 3 |
| 77 | `#pragma unroll`, `__noinline__` | Partial | Yes | `__noinline__` hint | READY, no row | 1 |
| 78 | multi-stage software pipelining | Partial | Yes | `Schedule::Pipelined` / stages as a node (recipe runs) | DECISION: recipe ruling (U8 S4); P9 parked | with #26 |
| 79 | warp specialization, `setmaxnreg` | Partial | Yes | `setmaxnreg`; role node (recipe runs) | TRIGGER: as #15 | with #15 |
| 80 | tile shape hierarchy | No | n-a | `Schedule::Contraction` is a unit variant (`plan/types.rs:57`, control `n_stages` found) | DECISION: P9 parked and D2 | 2 |
| 81 | split-K / stream-K | Partial | Yes | plan-level `k_split` axis, stream-K work division (kernel layer runs) | DECISION: D2 (as #44) | with #44 |
| 82 | `--use_fast_math` bundle | Partial | Yes | kernel-wide mode (per-node `Qualified` only) | READY, no row | 1 |
| 83 | rounding-mode intrinsics | Partial | Yes | `Qualified` at F64 and half is refused (`TypeError::IllegalQualifier`, `tests/kernel_qualified_dtype.rs`) | READY: U43 (open) | 1 |
| 85 | approximate transcendentals | Partial | Yes | exact-spelling hint so a re-emit prints `__expf` | READY: U37 (scheduled) | 1 |
| 86 | ULP bounds | Partial | Partial | bound on the node; F64 ratings are f32 ratings | READY: U42 (open, size S) | 1 |
| 87 | `cvt` with rounding/saturation | Partial | Yes | `cvt.pack` | READY, no row | 1 |
| 88 | `dp4a`, `lop3`, `prmt`, popc/clz/brev, funnel, DPX | Partial | Yes | no such `UnaryOp`/`BinaryOp` (grep `Popc|Clz|Brev|Funnel|Dot4|Prmt` in ir.rs = 0, control `BitAnd` found) | READY, no row | 2 |
| 90 | special registers | Partial | Yes | `%smid`, clock, `%globaltimer`, cluster registers | DECISION: what the oracle returns for a nondeterministic register is unruled | 1 |
| 91 | `__nanosleep`, `clock()` | No | n-a | timing statements | DECISION: as #90 | with #90 |
| 92 | graphs, streams, events, green contexts, peer access | Partial | Yes | graphs/streams/events/green contexts (peer access done) | OUTSCOPE: host runtime | 0 |
| 93 | `printf`, `assert`, `trap` | No | n-a | device side-effect statements (`printf` appears only in `ondevice/*.cu` harnesses, not the IR) | DECISION: output channel and oracle meaning unruled | 1 |
| 95 | shared-memory limits per arch | Partial | No | check a kernel's static and dynamic shared size against `cuda_capabilities` | READY, no row | 1 |

Rows by blocker: PATENT 5 (7, 40, 41, 50, 76); DECISION 12 (13, 17, 26, 44, 55, 75, 78, 80, 81, 90, 91, 93); LANE 5 (56, 61, 62, 64, 65); TRIGGER 11 (4, 15, 25, 38, 39, 42, 46, 47, 48, 59, 79); OUTSCOPE 5 (8, 14, 16, 27, 92); READY 27 (1, 9, 11, 20, 21, 24, 29, 31, 34, 35, 36, 37, 49, 51, 54, 60, 69, 70, 72, 77, 82, 83, 85, 86, 87, 88, 95). Total 65.
PRs by blocker (rows counted once): PATENT 10-11; DECISION 15-16; LANE 5; TRIGGER 10-11; OUTSCOPE 0; READY 26-27. Sum 66-70.

## 3. Burn-down (each row once)

Order: READY themes first (no external wait, best rows per PR), then themes a PM/CireSnave action unblocks, then the rest. Breaking changes batch into one release (U36 ruling).

| Order | Theme | Rows | Blocker | PRs | Why here |
|---|---|---|---|---|---|
| 1 | Atomics completion: `AtomicOp::{Inc,Dec}`, atomic load/store, denormal note | 69, 70, 72 | READY | 2 | 3 rows for 2 PRs, additive-ish, no patent area |
| 2 | Warp-mask and group collectives: lane mask, `activemask`/`match`, elect, partitions | 20, 29, 34, 31 | READY | 3 | 4 rows, one `Stmt`/`CollectiveKind` batch with theme 3 |
| 3 | Small leaves and hints: alias fact, cache hints on load/store, cluster rank, mbarrier drop/try_wait, proxy fences, `__noinline__`, shared-size legality check | 1, 9, 11, 21, 24, 77, 95 | READY | 7 | one breaking batch; each is small, none needs a decision |
| 4 | Numerics gaps: U43 F64/half qualifiers, U37 exact-spelling hint, U42 F64 ratings, fast-math mode, `cvt.pack`, integer bit ops | 82, 83, 85, 86, 87, 88 | READY | 7 | rows already on ROADMAP (U37, U42, U43) |
| 5 | MMA atom data (wmma and extra PTX shapes) | 36, 37 | READY | 2 | data tables checked against PTX, as `tests/mma_atoms.rs` does |
| 6 | Layout/oracle odds and ends: nested `View::Layout`, swizzle `Place` mode, E8M0 oracle decode, block sort, `red.async` | 54, 49, 60, 35, 51 | READY | 5-6 | independent, each M or smaller |
| 7 | Tensor-map/TMA (descriptor operand kind, `cp.async.bulk.tensor`, multicast, bulk groups) | 25, 46, 47, 48 | TRIGGER (U8c) | 4 | waits on Baracuda's sm_90 emitter; re-read US12645455B2 before a prefetch statement |
| 8 | MMA flags needing a second lane-table source | 38, 39, 42 | TRIGGER (U7 S3, U8b) | 3 | clears together; the source is the trigger |
| 9 | Warp roles launch attribute | 15, 79 | TRIGGER | 1 | cheap once a backend asks |
| 10 | Constants and packed locals | 4, 59 | TRIGGER (U40, U15) | 2-3 | both parked on "no consumer" |
| 11 | Warpgroup / TMEM completion (descriptors, swizzle, alloc, CTA pairs) | 7, 40, 41 | PATENT (U7 S4, S5) | 5-6 | needs the unread adjacent claims read first |
| 12 | Vector load/store and pointer values | 50, 76 | PATENT (U15, U9c) | 5 | needs claims read of US11262989B2, US9052888B2, WO2013070616A1 and a re-read for a pointer type |
| 13 | Plan-layer contraction and schedule language (per-operand dtype, multi-K, k_split, tile hierarchy, persistent/stream-K, pipeline object) | 44, 55, 81, 80, 13, 26, 78 | DECISION (D2, P9, U8 S4) | 10-11 | the largest block; none starts without a ruling |
| 14 | Recursion, device launch, special registers, device effects | 75, 17, 90, 91, 93 | DECISION | 7 | scope questions first |
| 15 | KISS element rows (fp4/fp6 buffers, wide dtypes, interleave layouts) | 56, 61, 62, 64, 65 | LANE (KISS) | 5 | KISS writes the rows |
| - | Host or launch attributes | 8, 14, 16, 27, 92 | OUTSCOPE | 0 | not IR concerns |

## 4. Judgement calls and what I did not verify

- Nothing was built or run. Statuses are code symbols plus test names grepped at b21af35c; a test name existing is not proof it asserts the claim (I read the bodies only of `kernel_unroll_copy_forms.rs` cache/valid tests, `kernel_packed.rs::a_packed_local_is_a_pinned_gap` and the `mma.rs` block-scale constructor).
- Yes rows at 78c7d38e (27): for each I confirmed the cited symbol still exists after the BREAKING PRs and the cited test file (and, where 11-15 named one, the test function) exists: `Place::ClusterPeer`, `ClusterRankOutOfRange`, `KernelInterface::{of, cooperative, launch_bounds, max_regs}`, `ExecId`, `AtomicOp::{Max,Min}`, `AccumSpec::{Tf32,Tf32x3,Bf16x3,F16Accum,Int32,Int32Sat}`, `ExternDecl`/`BodyText`/`Stmt::ExternCall`, `ScalarExpr::Fma`, `Stmt::{While,Break,Return}` (3 uses in `kernel_layer.rs`), `Stmt::Barrier` Device scope (`kernel_device_barrier.rs`), collectives (`kernel_collective.rs`), cp.async groups. No symbol or test was missing. Only row 60 was found overstated. I did not re-read those tests' assertions.
- Row 23 Yes is a judgement call: a split arrive and wait exists, the wait is by phase parity and there is no returned token (the survey's "token-returning arrive"). A program that tracks the phase gets the same behaviour. Strict reading would give Partial (then Yes 29, Partial 52).
- Row 45 Yes: the row asks for 4/8/16B, L1 bypass at 16B, global to shared; all are covered. `cp.async.mbarrier.arrive` is a different instruction and not in the row.
- Row 77 Partial is strict: `#pragma unroll` is built, `__noinline__` has no field. A lenient reader gives Yes.
- Row 95 and row 60 changes are corrections of the earlier passes (the table predates a634b09; the E8M0 gate predates it), not work in the 28 commits.
- Rows 13 (persistent loops), 78 and 79: persistent and warp-role kernels are ordinary control flow by ruling (U8 S4). I scored 79 Partial because a producer/consumer test runs; row 13 stays No because no test runs a tile-scheduler loop (a grid-stride lifted kernel exists in `tests/kernel_parallel_grid_stride_agrees.rs`, which I did not count as a persistent scheduler). A lenient reader moves 13 to Partial.
- Oracle semantics are unconfirmed against a second source for the mbarrier phase/parity rule (`docs/options-u8-copy-atoms.md`, "second independent source ... not obtained"), the stmatrix row rule, and the wgmma/tcgen05 fragment layouts ("no ISA text"): rows 19-23, 39-41, 46 are IR Yes / ORACLE Yes only in the sense that a test executes the oracle's reading.
- Row 12 stays Yes on the emitter-set `KernelInterface::cooperative` flag; rows 27 (sync domains) and 14 are the same kind of launch attribute and were scored OUTSCOPE, which is inconsistent with 12 counting. Flagged, not changed.
- The U19 C-family frontend (#385-#391) adds no IR node, so it moved no row.
- Absent claims and their controls: `movmatrix` (control: `ldmatrix` found in kernel.rs); arrive_and_drop/try_wait (control: 4 `MbarArrive` hits); elect/ClusterRank/setmaxnreg/TensorMap/multicast mask in kernel.rs, exec.rs, kernel_interface.rs (controls: `ExecId` 4, `LaunchShape` 6, `CopyBulk` 2 hits in the same files); dynamic parallelism, sync domain, managed memory, F128/I128, denormal-atomic note, tile shape, `printf` across crates (controls in the same runs: `u128` hits for the 128-bit grep, `Complex128` 12 hits in element.rs, `printf` hits in `ondevice/*.cu` proving the grep sees such text); `AtomicOp` Inc/Dec (the enum body was read: 7 variants); Popc/Clz/Brev/Funnel/Dot4/Prmt in ir.rs (control `BitAnd` found); `pub init` on `AllocDecl` (control `pub extent`).
- Not examined: PATENT-SWEEP was read only for the sections named in the table; the U-rows were read from ROADMAP 6.7 as printed (the table cells were truncated to the status text I needed). PR estimates are guesses from the size of comparable merged PRs (#366 to #391), not measurements.
