# Re-score of `docs/ir-union-surveys/slang.md` at origin/main `b21af35c` (Unpopped 0.56.12)

Method as `docs/ir-union-rescore.md`: read-only, nothing was built or run; every status rests on a named code symbol and a named test whose
name I grepped and found in the tree. "Yes" in the headline means IR-yes AND oracle-yes (a test executes it). The serial C emitter
(`unpopped-cpu-c`) is the only emitter here and is not the criterion.

Counts (Yes/Partial/No/n-a = total), 61 counted rows, numbering exactly as `rescore-2026-11-15/slang.md` (table rows top to bottom, minus the five
parenthetical rows):

- `a634b09`: 17 / 19 / 22 / 3 = 61
- `78c7d38e`: 22 / 21 / 15 / 3 = 61 (reconstructed from the 11-15 file: Yes = 1,2,3,13,14,15,16,17,18,20,23,24,28,31,32,37,39,42,43,44,51,55)
- `b21af35c`: 22 / 24 / 12 / 3 = 61

Rows whose status changed: 6 (4 up, 2 down: 4, 6, 7, 12 up; 31, 51 down). Rows whose status is unchanged but whose evidence changed: 3 (33, 34, 35).
Rows examined: all 36 not-Yes rows (by the code enums plus the greps named below) and all 22 Yes rows (cited symbol and cited test name confirmed to exist;
bodies opened for 23, 31, 51; others by name only). Skipped: n/a rows 58, 59, 60 (front end). The five uncounted rows are reported at the end.

## Rows changed or found wrong

| # | Capability | 78c7d38e -> b21af35c (IR / ORACLE) | Evidence (file:symbol ; test) |
|---|---|---|---|
| 4 | Memory: ByteAddressBuffer raw load/store at byte offset | No -> **Yes** (Yes / Yes) | `kernel.rs:Place::Raw` and `Place::AllocRaw` (#384, U9c level b): `elem`-typed value at a byte offset, any whole-byte scalar, `Load`/`Store`/`MaskedLoad`/`MaskedStore`; `tests/kernel_raw_place.rs::a_raw_load_reads_the_named_bytes_at_the_named_width`, `::a_raw_store_writes_only_its_bytes_little_endian`, `::a_misaligned_or_negative_offset_is_a_typed_trap`; shared/private twin `tests/kernel_alloc_raw_place.rs::a_raw_load_from_an_allocation_reads_the_named_bytes_at_the_named_width`. Oracle only: cpu-c declines it by name (`unpopped-cpu-c/tests/kernel_raw_place_refused.rs`). Scalars only (no struct-typed load) |
| 6 | Memory: Pointers / buffer device address | No -> **Partial** (Partial / Partial) | Same symbols: base named by a buffer parameter plus an integer byte offset, `elem` punnable (`(elem*)((char*)buf + off)`). No pointer value: no pointer type, parameter, local, comparison or pointer-in-struct (`Place::Raw` doc comment says so; `tests/kernel_raw_place.rs::the_checker_applies_the_buffers_role_and_refuses_what_has_no_raw_rules`). Pointer values are ROADMAP U9c level (a), trigger met 2026-11-12, not started |
| 7 | Memory: Constant buffer / ParameterBlock / push constants | Partial -> **Yes** (Yes / Yes) | `kernel.rs:ParamKind::Struct`, `StructField`, `CallArg::Struct` (#382, #383, U23 slices 1-4); `kernel_interface.rs:ParamsBlock::of_struct` (natural C layout); `tests/kernel_struct_param.rs::a_kernel_reads_each_field_as_its_local` (has a control with other field values), `tests/kernel_call_struct.rs::a_struct_argument_binds_each_field_in_order`, `tests/kernel_struct_grid.rs::the_grid_comes_from_the_field_value`, `tests/kernel_params_block_derive.rs::padding_follows_the_c_rules`. Judgement: fields are scalars only (no array, vector or nested struct field), and the Push/Uniform/SpecConst class split is emitter metadata, not a node; cpu-c declines it (`kernel_struct_param_refused.rs`) |
| 12 | Memory: Address spaces / data layouts (std140, std430, scalar, C layouts) | No -> **Partial** (Partial / n-a) | `kernel_interface.rs:ParamsBlock::of_struct` derives the natural C layout of a struct parameter (`tests/kernel_params_block_derive.rs::padding_follows_the_c_rules`, `::tail_padding_rounds_the_size_to_the_alignment`); `artifact.rs:LayoutRule::{Std140,Std430,Scalar}` is carried in the declared resource map (`tests/artifact_and_capability_types.rs`, a type test only). The kernel IR carries no per-operand layout rule and nothing derives a std140/std430 layout. A layout is not something the oracle executes, so ORACLE is n-a. Judgement call; 78c7d38e did not look at `LayoutRule` or `ParamsBlock::of_struct` |
| 31 | Atomics: 64-bit atomics | Yes -> **Partial** (Yes / Partial) | STALE CLAIM. IR: `kernel_check.rs:check_atomic` admits any `int_range` element incl. `I64`/`U64`; oracle `oracle/kernel_eval.rs:atomic_apply` is generic over `int_range`. But no test executes a 64-bit atomic: the only `I64` hits in `tests/kernel_atomic_{rmw,cas,fetch}.rs` are a wrong-typed operand or a wrong-typed old local (refusal tests); `kernel_atomic_shared.rs`, `kernel_atomic_half.rs`, `kernel_cluster.rs`, `kernel_multicast.rs` have 0 hits. Control: the same grep finds those refusal uses. cpu-c tests: 0 `U64`/`I64` atomic hits. The 78c7d38e/a634b09 evidence ("integer atomics on I64/U64") named the checker rule, not a run |
| 51 | Numerics: Rounding mode per op (RTZ/RTE on conversions, `OpFAdd` rounding decoration, cvt.rn) | Yes -> **Partial** (Partial / Partial) | STALE CLAIM. Conversions: `ir.rs:RoundMode` + `Saturation` on `Cast` run at every dtype (`tests/typed_expressions.rs`, `oracle_bitcast_float_to_int.rs`). Arithmetic rounding decoration: `ScalarExpr::Qualified` runs in the oracle only at `F32`/`F32Strict`; #388 made `check_kernel` refuse it at `F64` and half types by name (`types.rs:TypeError::IllegalQualifier`, `tests/kernel_qualified_dtype.rs`; before #388 it passed the checker and the oracle panicked, so the old Yes overreached for `__dadd_rn`-style ops). ROADMAP U43, open |
| 33 | Dtypes: half, bf16, fp8, f16tof32/packHalf2x16 | Partial -> Partial (Partial / Partial), evidence only | 78c7d38e said `Pack`/`BitcastPacked` "never run end to end". Now `tests/kernel_packed.rs::a_lane_of_a_pack_is_that_lane`, `::a_lane_of_a_bitcast_is_sliced_from_its_bits_lane_zero_least_significant` run them in a launched kernel (F32/U8/U32 lanes). No test packs two `F16` lanes into a `U32` (the packHalf2x16 form); a packed LOCAL still traps (`::a_packed_local_is_a_pinned_gap`) |
| 34 | Dtypes: int8_t4_packed, dot4add | Partial -> Partial (Partial / Partial), evidence only | `tests/kernel_packed.rs` runs `BitcastPacked(word, U8, 4)` + `Extract` + lane-wise operators in a kernel (the `uint8_t4_packed` form). Still no `Dp4a` node, no `I8` lane run, no lane-wise dot test |
| 35 | Dtypes: vector and matrix types | Partial -> Partial (Partial / Partial), evidence only | Lane counts are now 2, 4, 8, 16 (8 and 16 within 128 bits): `tests/packed_values.rs::eight_and_sixteen_lane_packs_are_packed_up_to_128_bits` (type rule only, ROADMAP U15 slices 1-2, #380); 3 lanes still refused (`::lane_counts_other_than_two_and_four_are_refused` iterates n = 0,1,3,5; control: the 8/16 test passes). So no `float3`; still no `dot`/`cross`/matrix `mul` |

Yes rows spot-checked and still standing (symbol and test names exist): 1 (`Place::Alloc`+`MemorySpace::Shared`; `kernel_launch.rs::a_thread_may_use_shared_memory_as_its_own_scratch`),
2 (`exec.rs:Extent::Param`; `kernel_launch.rs::a_dynamic_extent_comes_from_its_argument`; cpu-c `kernel_dynamic_shared_emit_agrees.rs`; `kernel_interface.rs:DynamicShared`),
3 (`kernel.rs:BufferRole`, `KernelError::{WriteToReadOnly,ReadFromWriteOnly}`; `kernel_layer.rs::buffer_roles_are_enforced_in_both_directions`),
13 (`Stmt::Barrier`+`BarrierKind::Full`; `kernel_barriers_run.rs::a_barrier_lets_a_thread_read_a_neighbours_shared_store`),
14 (`Stmt::Fence`; `kernel_fence.rs::a_release_fence_and_an_acquire_fence_order_relaxed_message_passing`),
15 (`kernel_uniform.rs`; `kernel_barrier.rs::a_barrier_under_a_thread_dependent_if_is_rejected`; #388 added warp-uniform control, `kernel_warp_uniform.rs`),
16 and 24 (`exec.rs:ExecId::{LaneId,SubgroupId,SubgroupSize,...}`; `kernel_launch.rs::every_exec_id_reads_its_value`),
17 and 55 (`exec.rs:LaunchShape`, `with_subgroup`; `kernel_launch.rs::the_subgroup_size_can_be_chosen_inside_the_range`, `kernel_exec.rs::a_subgroup_range_is_accepted`),
18 (`kernel.rs:CollectiveKind::ReduceWith`, `kernel_check/collective.rs:check_combine`; `kernel_collective_combine.rs::an_addition_combine_agrees_with_the_built_in_sum`; no test uses a literal multiply combine, as before),
20 (`CollectiveKind::{Shuffle,Broadcast}`; `kernel_collective.rs::a_broadcast_gives_every_lane_one_lanes_value`),
23 (body opened: `a_shuffle_reads_another_lane_or_keeps_its_own` runs Idx, Up, Down, Xor 1, Xor 2; Xor 3 is composition),
28 and 32 (`Stmt::AtomicRmw`/`AtomicCas`; `kernel_atomic_rmw.rs::max_and_min_pick_the_extremum_and_add_wraps`, `kernel_atomic_cas.rs::exactly_one_thread_wins_a_cas_from_zero`, `kernel_atomic_shared.rs::eight_threads_count_into_one_shared_cell`; Interlocked on int64 is covered only by the row-31 gap),
37 (`kernel_subbyte_load.rs::a_sub_byte_element_widens_with_sign_or_zero_extension`, `sub_byte_packing.rs::every_i4_and_u4_value_round_trips`, `mx_elements.rs`),
39 (`ir.rs:ScalarExpr::Bitcast`; `oracle_bitcast_float_to_int.rs::a_float_read_as_i32_and_as_u32_gives_its_bit_pattern`, a test added by #388: before it an integer `Bitcast` of a float did not evaluate in a kernel, so 78c7d38e's oracle half was weaker than it read),
42, 43 (`Stmt::{If,For,While,Break,Continue,Return}`; `kernel_eval_semantics.rs::break_continue_and_return_do_what_they_say`, `kernel_layer.rs`; `Stmt::For` gained `unroll`, tests updated: `kernel_unroll_copy_forms.rs::the_unroll_hint_does_not_change_the_result`),
44 (`Stmt::Call`, `kernel_module.rs:check_module`; `kernel_module_run.rs`, `kernel_call_shape.rs`).
No cited symbol or test name of a Yes row has disappeared.

## Rows still not a full Yes at b21af35c (36 rows)

Blocker codes: PATENT sections are in `docs/PATENT-SWEEP.md`. Rule I used for PATENT: the next piece of work falls under a section that records a re-read trigger or "claims UNREAD" before building. Rows 19/21/22 sit near hit 23 (US8539204B2 / US9417875B2, section "Re-sweep addendum ... 2026-11-13", ruled A by CireSnave 2026-11-13, uncleared); I coded them DECISION/READY because the added forms do not change `Collective` barrier semantics, which is that hit's re-read trigger.

| # | Capability | IR | ORACLE | What is missing | Blocker | PRs |
|---|---|---|---|---|---|---|
| 5 | Append/Consume structured buffer | Partial | Partial | fetch-add append counter works (`kernel_atomic_fetch.rs::a_fetch_add_of_one_hands_every_thread_a_distinct_slot`); no resource with a hidden counter | TRIGGER: ROADMAP U3 remainder ("each when a consumer needs it", `docs/options-u3-remainder.md`); no consumer | 1 |
| 6 | Pointers / buffer device address | Partial | Partial | pointer type/param/local/compare, pointer in struct | PATENT: section "U9c raw byte-offset places (2026-11-21)", NOT CLEARED, re-read trigger "a pointer-valued Ty, parameter or local"; ROADMAP U9c (trigger met 2026-11-12) | 3-4 |
| 8 | Textures / samplers | No | No | image operand kind, Sample/Load/GetDimensions, LOD, gradient, gather. Absent: grep `texture\|sampler` in `kernel.rs`/`exec.rs` = 0 (control: `descriptor` hits there, on mma descriptors) | TRIGGER: ROADMAP U21 ("waits on a consumer": Fuel 0 of 192, FlashInfer 0) | 3-4 |
| 9 | Bindless handles | No | No | `Handle<T>`, descriptor heap, `nonuniform` (same grep) | TRIGGER: U21 | 1-2 after #8 |
| 10 | Tensor-shaped resources (runtime strides/rank) | Partial | Partial | runtime strides are now reachable (struct param fields as locals plus `Place::Raw` byte offsets) but no kernel shows it; no runtime-rank operand | READY: no row (a conformance kernel flips it by composition, as K1..K7 did) | 0.3 |
| 11 | Coherence/cache qualifiers | Partial | Partial | no coherent/volatile/nontemporal on `Load`/`Store` (`Stmt::Load` has only `local`, `place`; `CacheHint` is on `CopyAsync` only) | READY: no row (one change with #29; BREAKING on `Stmt`, batch per the U36 ruling) | 1 |
| 12 | Data layouts (std140/std430/scalar) | Partial | n-a | `ParamsBlock::of_struct` is natural-C only; no rule parameter, no std140 padding | READY: no row (stays inside the U23 layout-only guard; no re-read trigger) | 0.3 |
| 19 | Wave prefix | Partial | Partial | exclusive `Min`/`Max` refused (`kernel_check/collective.rs:fold_op`, "no identity for lane 0"), exclusive float non-`Sum` refused, no `Prod` scan | DECISION: lane-0 value of an exclusive Min/Max/Prod scan (precedent: exclusive `Mean` scan, lane 0 NaN, 0.42.10); no options note | 0.3 |
| 21 | Ballot/vote/match | Partial | Partial | `WaveMatch`, wide (uint4) ballot, `WaveGetLane*Mask` forms (lane masks are shifts of `LaneId`, composition). `CollectiveKind` has Shuffle/Vote(All,Any,Ballot)/Reduce/Scan/Broadcast/ReduceWith/ScanWith only; no Match (control: `ReduceWith` found by the same grep) | READY: no row (U5 extension; `CollectiveKind` is `non_exhaustive`, additive) | 0.5 |
| 22 | Clustered / multi-prefix | No | No | no cluster-size operand on `Reduce`/`Scan`; an xor-butterfly with `Shuffle{Xor}` is expressible (row 23 proves Xor runs) but no test does it | READY: no row (a butterfly test flips it by composition, consistent with row 23; a first-class operand would be a new `Collective` form near hit 23) | 0.1 |
| 25 | Coop matrix (CoopMat) | Partial | Partial | tile-level fill/transpose/reduce/MapElement/convertUse/arithmetic; WMMA 16x16x16, e4m3/e5m2 and Metal 8x8 atoms (only three PTX atom constructors, `mma.rs:nvidia_m16n8k16_f16_f32` etc.). New since 78c7d38e: `Stmt::MatrixLoad`/`MatrixStore` (ldmatrix/stmatrix, with `trans`; `tests/kernel_matrix_load.rs::the_transpose_is_not_the_identity`, `kernel_matrix_store.rs`), `Stmt::Wgmma`/`WgmmaCommit`/`WgmmaWait` (f16/bf16 -> f32, shared operands; `kernel_wgmma.rs::every_fragment_value_is_the_reference_product`), `Stmt::Tcgen05Mma`/`Commit`/`Ld` (`kernel_tcgen05.rs`); all oracle only, cpu-c refuses wgmma/tcgen05. They are PTX-specific and do not give a CoopMat tile value | DECISION: no ruling on a tile-valued fragment type (fragments are per-lane `Private` arrays); atom lane tables also wait on a second source (ROADMAP U8b/U7 S3 trigger) | 4-6 |
| 26 | Coop matrix clamp modes / TensorLayout | Partial | Partial | tile-load operand carrying layout and clamp mode; per-element only (`kernel_masked_access.rs`, `kernel_padding_modes.rs`); `TensorLayout` semantics still UNVERIFIED | DECISION: with #25 | with #25 |
| 27 | Coop vector (CoopVec) | No | No | no matvec node, no outer-product accumulate (Slang CoopVec target list UNVERIFIED in the survey) | READY: no row (low value; needs an options note first) | 2-3 |
| 29 | Atomic<T> | Partial | Partial | no atomic load/store statements, no inc/dec-with-bound (Stmt list has `AtomicRmw`, `AtomicCas`, no load/store variant) | READY: no row (with #11) | with #11 |
| 30 | Float atomics | Partial | Partial | float `Max`/`Min` refused by the checker (`kernel_atomic_rmw.rs::a_float_add_is_admitted_for_f32_and_f64_only_and_max_min_are_not`); F16/Bf16 `Add`/`Exchange` run (`kernel_atomic_half.rs`) | TRIGGER: ROADMAP U38a-2 PARKED 2026-11-22 ("revisit when the census shows CAS as the largest remaining wall", `docs/options-u38-cas-atomic-loops.md`) | 1 |
| 31 | 64-bit atomics | Yes | Partial | a test that runs `Add`/`Max`/`Cas` on `I64` and `U64` in the oracle | READY: no row | 0.1 |
| 33 | half/bf16/fp8, f16tof32/packHalf2x16 | Partial | Partial | a kernel that packs two `F16` lanes into a `U32`; packed locals | READY: no row (test; packed locals are U15, parked 2026-11-21, no consumer) | 0.1 |
| 34 | int8_t4_packed, dot4add | Partial | Partial | an `I8` lane run and a lane-wise dot (Extract/Mul/Add composition) or a `Dp4a` node | READY: no row | 0.2 |
| 35 | Vector and matrix types | Partial | Partial | 3 lanes, `dot`/`cross`/matrix `mul`, packed locals (scalarize pass, ~250 lines per ROADMAP U15) | PATENT: section "U15 8- and 16-lane packed values (2026-11-17)", verdict NOT CLEARED, UNREAD, claims read owed before any lowering; U15 also parked for want of a consumer | 2-3 |
| 36 | Struct/array/tuple/Optional/enum as values | Partial | Partial | struct params exist (#382) and tuples (`CallTuple`); no struct/aggregate local or value, no Optional/enum; arrays by id (D-R4-3) | DECISION: aggregate values reopen D-R4-3 ("arrays by id, no pointer values"); no options note | 3-4 |
| 38 | Integer bit ops | No | No | popcount/clz/ctz/bitreverse/bitfield. Absent: grep `popcount\|popcnt\|Clz\|Ctz\|BitReverse\|bitfield` over `unpopped/src` and `unpopped-cpu-c/src` = 0 hits outside `ldexpf`/`frexpf` text; control: `Lgamma` and `BitAnd` found | READY: no row | 1-2 |
| 40 | Math: fma, frexp, ldexp, isnan, lerp, saturate, modf, rint | Partial | Partial | `Fma` exists; `Frexp`/`Ldexp`/`IsNan`/`Lerp`/`Saturate`/`Modf` nodes absent (same grep style: 0; control: `Lgamma`); `Hypot` is ROADMAP U44 | READY: U44 (hypot), others no row | 1-2 |
| 41 | Multi-result intrinsics | Partial | Partial | `CallTuple` and `AtomicRmw.old` cover functions and Interlocked; `ScalarExpr` single-valued, so `sincos`/`modf`/`frexp` need the single-valued parts of #40 | READY: no row (with #40) | with #40 |
| 45 | throws/try, discard, printf | No | No | no trap/debug/printf statement (Stmt list has none; control: `AtomicRmw` found) | READY: no row (`docs/ir-union-design.md` D3: effects; options note first) | 1-2 |
| 46 | Link-time specialization | No | No | `ExternConst` leaf | OUTSCOPE: `ir-union-design.md` D3 "front-end, not IR" (Confirmed) | 0 |
| 47 | Specialization constants | No | No | `KernelInterface::spec_constants` is emitter-set metadata (`kernel_interface.rs:SpecConstant`); no IR leaf | OUTSCOPE: D3 (front-end) | 0 |
| 48 | Preprocessor macros, constexpr, static const | Partial | n-a | `Const` is compile-time; macros and constexpr are the front end's (U19 C-family walker handles function-like macros, #390; it adds no IR node) | OUTSCOPE: front-end by D3 | 0 |
| 49 | Autodiff | No | No | `Transform::Fwd/Bwd` over the kernel layer; loops and locals now exist; absent: grep `fwd_diff\|autodiff` in `unpopped/src` = only doc/decline-list text | DECISION: D3 places it as a transformation "over I2"; no ROADMAP row, no options note | 4-6 |
| 50 | Fast vs precise, per-function precision | Partial | Partial | per-op `Qualified` only (and now f32 only); no per-function mode, no contract/NaN assumption | LANE: KISS (ROADMAP I3 "numerics axes ... one synchronized cut", held on the sk4 seam) | 1 |
| 51 | Rounding mode per op | Partial | Partial | `Qualified` at F64 and half (oracle reference plus emitter spelling) | READY: U43 (open, owner Unpopped + emitters) | 1 |
| 52 | NaN/Inf preservation mode | No | No | flag on OpDef/KernelFn | LANE: KISS (ROADMAP C2 "belongs to KISS", parked; I3) | with #50 |
| 53 | Math precision of MMA | Partial | Partial | `AccumSpec::{Tf32,Tf32x3,Bf16x3,F16Accum,Int32,Int32Sat}` run (`contraction_accum_policy.rs`, `contraction_int32_accum.rs`); no FP8 multiply policy/test; wgmma/tcgen05 add only f16/bf16 -> f32 | READY: U7 (fp8 fast-accumulate listed in the row) | 1 |
| 54 | Inline SPIR-V, `[require]` atoms, `__target_switch` | Partial | Partial | `ExternCall`/`ExternDecl`/`TargetBody` per target run only via a `reference` kernel (`kernel_extern_run.rs::the_reference_is_what_the_oracle_runs`, `::a_declaration_with_no_reference_traps_with_a_typed_unsupported`); capability atoms live in `KernelInterface::requires`/`compiler_args`, set by the emitter | OUTSCOPE: capability atoms are target-emitter metadata and the oracle cannot run asm by design | 0 |
| 56 | Shader clock | Partial | Partial | only as opaque extern `ExternDecl::new("clock", ..)` (`kernel_extern_check.rs` line 100); no `ReadClock` leaf (`ExecId` has LaneId, SubgroupId, SubgroupSize, DeviceRank, DeviceCount, thread/block ids; no clock) | READY: no row | 0.5 |
| 57 | SPIR-V vendor extensions, demote-to-helper | No | No | capability atoms on nodes; helper invocations are a graphics concept | OUTSCOPE: emitter metadata, same as #54 | 0 |
| 61 | `printf` (Slang-unique) | No | No | as #45 | READY: no row (with #45) | with #45 |

Blocker counts: READY 18 (10,11,12,21,22,27,29,31,33,34,38,40,41,45,51,53,56,61), DECISION 5 (19,25,26,36,49), TRIGGER 4 (5,8,9,30), PATENT 2 (6,35), LANE 2 (50,52), OUTSCOPE 5 (46,47,48,54,57). Total 36.

PRs (~150k diff characters each), summed with "with #n" counted once: READY 10-14, DECISION 11-16, TRIGGER 6-8, PATENT 5-7, LANE 1, OUTSCOPE 0.

## Burn-down (each row once)

1. **Composition and conformance tests, no new IR** (22, 31, 33, 34, 10; READY; about 0.8 PR). First: cheapest, flips up to five rows by tests alone (31 only restores a Yes the old evidence over-claimed). Same judgement class as row 23 and row 18, so say so in the PR.
2. **Scalar numerics nodes** (38, 40, 41; READY; 2-4 PRs). Next: each node is oracle + checker + contract tier + cpu-c, and U44 (hypot) is already a row. 56 (clock leaf, 0.5) rides here as it is one more leaf.
3. **Rounding and accumulation** (51 via U43, 53 via U7; READY; 2 PRs). Fixes a Yes the old evidence overstated; U43 is already scheduled.
4. **Memory-model statements and collective forms** (11, 29 together 1 PR, BREAKING so batch; 21 at 0.5 PR READY; 19 DECISION 0.3 PR). One breaking cut; 19 needs a lane-0 ruling first.
5. **Layout derivation** (12; READY; 0.3 PR). Small, extends U23 inside its guard.
6. **Effects** (45, 61; READY; 1-2 PRs). Needs an options note first (D3 says effects).
7. **CoopVec** (27; READY; 2-3 PRs). Last of the READY rows: no Slang consumer in the corpus.
8. **Tile / cooperative matrix** (25, 26; DECISION; 4-6 PRs). Ask the PM for a ruling on a tile-valued fragment type before any code; the atoms part waits on a second lane-table source (U8b/U7 S3).
9. **Aggregates and autodiff** (36, 49; DECISION; 7-10 PRs). Biggest, and 49 needs a design; 36 reopens D-R4-3.
10. **Pointer values and wide vectors** (6, 35; PATENT; 5-7 PRs). Read the U9c and U15 claims first (both NOT CLEARED).
11. **Resource types** (5, 8, 9; TRIGGER; 5-8 PRs). Start only when a consumer appears (U21 / U3 remainder).
12. **Float Max/Min atomics** (30; TRIGGER; 1 PR). Parked by PM ruling 2026-11-22.
13. **Per-function numerics modes** (50, 52; LANE KISS; 1 PR). One synchronized cut with I3.
14. **Out of scope** (46, 47, 48, 54, 57; 0 PRs).

## Unnumbered rows (not in the 61; status at b21af35c)

- File line 78 (cp.async / TMA / mbarrier / cluster launch / DSMEM): Partial -> **Partial** (IR Partial / ORACLE Yes for the parts built). New since 78c7d38e: `Stmt::MbarInit/MbarArrive/MbarCompleteTx/MbarWait` (`tests/kernel_mbar.rs::a_consumer_blocks_until_the_producer_arrives_and_then_sees_its_data`), `Stmt::CopyBulk` 1-D bulk copy on an mbarrier (`tests/kernel_copy_bulk.rs::a_consumer_that_waits_sees_the_copied_tile`), `CopyAsync` `valid`/`cache`/`l2_prefetch` (`kernel_unroll_copy_forms.rs::valid_reads_that_many_elements_and_zero_fills_the_rest`), named barriers (`kernel_named_barrier.rs`), PDL `GridDepWait`/`GridDepLaunchDependents` (`kernel_grid_dep.rs`). Still absent: TMA tensor-map (descriptor) copies (ROADMAP U8c, open). Oracle only; cpu-c refuses mbarrier, bulk and named barriers.
- File line 69 (denormal handling): Partial, unchanged (`Qualified` ftz is per op and f32 only; no per-dtype mode on the kernel; LANE KISS, I3).
- File lines 62 (generics), 67 (autodiff limits), 75 (ray tracing): unchanged, No / -- / No (front end by D3; out of compute scope).

## Judgement calls and what I did not verify

- Nothing was built or run. A name found by grep is not proof the test asserts the claim; I opened bodies only for 23, 31 (grep of I64 sites), 33/34 (`kernel_packed.rs`), 51 (`kernel_qualified_dtype.rs` header) and the `Place`/`ParamKind` definitions.
- Yes rows were checked for a disappeared symbol or test, not re-executed. I did not open the bodies of the 78c7d38e Yes tests for 1, 2, 3, 13-17, 20, 24, 28, 32, 37, 42-44, 55 beyond their names and the enum definitions above.
- Downward moves rest on reading, not running: 31 (no 64-bit atomic test found; the oracle code is generic and probably correct, so this is a test gap, not a behaviour defect) and 51 (a definite restriction recorded in `kernel_qualified_dtype.rs` and ROADMAP U43). 51 could be argued as Yes for the conversion half of the row.
- Up moves: 4 is firm (Raw is exactly byte-offset typed access). 7 is a judgement call (scalar-field structs only; the Push/Uniform class split is emitter metadata). 6 is generous (a byte-offset special case, no pointer value). 12 is generous (a manifest type plus a derivation, nothing in the kernel).
- Row 22 is kept No although a Shuffle-Xor butterfly expresses clustered reduction; I treated "no test" as No, as the 11-15 pass did, but row 23 was promoted on the same kind of composition. The two are inconsistent; one test would settle 22.
- Row 18 and the Yes rows that are compositions (product, CountBits, AllEqual, quad ops) are unchanged and still compositions, not first-class nodes.
- Row 15: the checker now accepts a `Subgroup` collective under warp-uniform control (#388) and U41 records that `if (c) return;` before a barrier is checker-accepted; I did not re-score 13/15 for that, since both are Yes on the cited tests.
- The U19 C-family frontend (#385-#391) adds no IR node, so no row moved for it; rows 46-48 are scored OUTSCOPE because of D3, even though the walker now lifts function-like macros and `constexpr`-style constants.
- PATENT coding: I used it only where the sweep records a re-read trigger or unread claims for the next step (rows 6 and 35). By the broader reading "area marked NOT CLEARED" the collective rows (19, 21, 22; hit 23), the MMA rows (25, 26, 53; hits 22/26 and the wgmma/tcgen05 sections), the effects rows (45, 61) and 12 would be PATENT too. PM may recode.
- PR estimates are my sizing from the diff shape of comparable merged PRs (#380 for packed, #384 for Raw, #382-#383 for structs), not measured.
- 78c7d38e's per-row statuses were reconstructed from the 11-15 file (header counts matched: 22/21/15/3); I did not re-open `docs/ir-union-rescore.md` row by row.
