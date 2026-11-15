# Re-score: Triton (`docs/ir-union-surveys/triton.md`) at `78c7d38e` (Unpopped 0.47.3)

Status at a634b09 (Yes/Partial/No/n/a): 14 / 21 / 34 / 1 = 70
Status at 78c7d38e (Yes/Partial/No/n/a): 24 / 21 / 24 / 1 = 70 (row 70 lowered to Partial by the lead: the agent's own evidence says `tl.target_info` has no IR node)
Changed rows: 16 (10 up to Yes, 6 No -> Partial, none down).

Numbering note: row numbers follow `docs/ir-union-rescore.md` (70 rows). That numbering skips three raw-survey rows whose IR cell is not exactly Yes/Partial/No/n/a
("Yes (3-pass)" softmax, "Yes (extra)" Window, "Mostly Yes" Math). A plain count of every table row in file order would give 73 and shift rows after the softmax row by 2-3.
Checked against the first re-score at rows 1, 20, 27, 32, 44, 52, 62 (all line up).

Examined: 55 (every row not Yes at a634b09). Skipped: 15 (14 already Yes, 1 n/a). The 89 commits cannot lower a Yes row except through a regression, which I did not look for.

## Changed rows

| # | Capability | Status | Evidence at 78c7d38e |
|---|---|---|---|
| 4 | Program model: Device function calls (`tt.call`, `@triton.jit` helper fns, `combine_fn` lambdas) | Partial -> **Yes** | `CollectiveKind::{ReduceWith,ScanWith}{combine}` name a module function as the user combine; `Stmt::CallTuple` for multi-result callees (`kernel.rs`); `tests/kernel_collective_combine.rs`, `kernel_collective_tuple.rs`, `kernel_call_tuple.rs`; cpu-c `kernel_collective_combine_emit_agrees.rs`, `kernel_call_tuple_emit_agrees.rs` |
| 6 | Memory: Masked load `tl.load(ptr, mask, other)` (arbitrary boolean tile mask + fill value) | Partial -> **Yes** | `Stmt::MaskedLoad{local,place,mask,fill}`: any scalar mask, fill any value of the element type, memory untouched under a false mask; `tests/kernel_masked_access.rs` (`a_false_mask_gives_the_fill_and_never_reads_the_memory`, `the_fill_is_any_value_the_element_type_holds`); cpu-c `kernel_masked_access_emit_agrees.rs` |
| 7 | Memory: Masked store `tl.store(ptr, val, mask)` | Partial -> **Yes** | `Stmt::MaskedStore{place,value,mask}`; `tests/kernel_masked_access.rs` (`a_masked_store_writes_only_where_the_mask_holds`, `a_false_lane_leaves_no_race_record`); cpu-c `kernel_masked_access_emit_agrees.rs` |
| 16 | Matmul: `input_precision` = tf32 / tf32x3 / ieee / bf16x3 / bf16x6 | No -> **Partial** | `AccumSpec::{Tf32,Tf32x3,Bf16x3,F16Accum,Int32,Int32Sat}` + `WideFloat` (= ieee); oracle models; `tests/contraction_accum_policy.rs` (`the_oracle_matches_the_reference_for_every_f32_policy_and_step`), `contraction_int32_accum.rs`. Missing: `bf16x6` (grep bf16x6 over src/tests/docs = 0 hits; the same grep finds `Bf16x3`) |
| 18 | Matmul: `tl.dot_scaled(...)`; e2m1/e4m3/e5m2/bf16/fp16 formats; e8m0 scales group 32 along K | No -> **Partial** | `MmaAtom::with_block_scale(block, dtype, (sfa,sfb))` adds scale operands per K block, `Stmt::Mma{extra}` runs them in the oracle; `tests/kernel_mma_flags.rs` (`a_block_scaled_atom_applies_the_scale_of_each_k_block`); cpu-c `kernel_mma_extras_emit_agrees.rs`. Test atoms use `F32` elements and `F32` scales; no run with `F4E2M1` elements and `F8E8M0` scales at block 32, and no `lhs_k_pack` |
| 21 | Reduce: `tl.reduce(input, axis, combine_fn)` with tuple inputs (welford/argmax) and user combine_fn | No -> **Yes** | `Stmt::CollectiveTuple{kind: ReduceWith, values, results, scope}` over a tuple state, combine a module function of 2k params returning k values; `tests/kernel_collective_tuple.rs` (`an_argmax_reduce_keeps_the_first_of_the_largest_in_every_scope`, `a_welford_reduce_gives_every_lane_the_mean_the_m2_and_the_count`); cpu-c `kernel_collective_tuple_emit_agrees.rs`. Scope is Subgroup or Block (the thread-level form of a tile reduce) |
| 23 | Scan: `tl.associative_scan(input, axis, combine_fn, reverse)` tuple inputs, cumsum, cumprod | Partial -> **Yes** | `CollectiveKind::ScanWith` and `CollectiveTuple` scan (argmax running best, Welford merge): `tests/kernel_collective_tuple.rs` (`an_argmax_scan_is_the_running_best`, `a_welford_scan_...`), `kernel_collective_combine.rs` (`a_scan_gives_each_lane_the_inclusive_prefix_fold`); `Access::Scan` any axis since 0.42.11 (`tests/scan_axis.rs`) with `reverse`. Caveat: custom combine has no `reverse` and no exclusive form |
| 25 | Histogram: `tl.histogram(input, num_bins, mask)` | Partial -> **Yes** | per-block histogram in a `Shared` array by `AtomicRmw` then published to global: `tests/kernel_atomic_shared.rs` (`eight_threads_count_into_one_shared_cell`; races without the barriers are typed traps); a mask by `MaskedLoad` / `If` (`kernel_masked_access.rs`). Combination histogram+mask not in one test; the pieces are each run |
| 45 | Schedule: `num_stages` (software-pipelining depth) | No -> **Partial** | the pipeline is expressible as structure, not as a hint: `Stmt::{CopyAsync,AsyncCommit,AsyncWait{pending}}` give any stage depth; `tests/kernel_pipeline_double_buffer.rs`, cpu-c `kernel_pipeline_emit_agrees.rs`. No `num_stages` hint field (the row's IR addition); `Stmt::For` has no hints (U18 is scheduled, not built) |
| 46 | Schedule: `num_ctas` (cluster size, SM90+) | Partial -> **Yes** | `LaunchShape::cluster` plus `Place::ClusterPeer` (DSMEM), `Cluster` barrier/atomic/fence scope; `tests/kernel_cluster.rs`, `kernel_cluster_peer.rs`; cpu-c `kernel_cluster_peer_emit_agrees.rs`. Cluster rank is computed from `BlockIdx` and the cluster dims (`ranks_count_x_fastest_across_a_two_dimensional_cluster`) |
| 51 | Schedule: Persistent kernels (grid = #SMs, tile loop via `num_programs`), stream-K via atomics, CLC scheduler | No -> **Partial** | `ExecId::GridDim` + `Stmt::While`/`For` give the persistent tile loop (grid-stride loop in `tests/kernel_barrier.rs`, lifted forms in `kernel_parallel_grid_stride_agrees.rs`); stream-K style split-K with atomic fix-up: `tests/kernel_gemm_family.rs` (`split_k_across_blocks`, `split_k_with_fixup`, `tests/support/gemm.rs`). Not built: cluster launch control (`ttng.clc_*`), a persistent GEMM tile-scheduler kernel as a test |
| 55 | Coop: Async global->shared copy (`ttg.async_copy_global_to_local`, commit, wait N) = cp.async | No -> **Yes** | `Stmt::{CopyAsync{dst,src,elems},AsyncCommit,AsyncWait{pending}}`; oracle lands at wait, typed `KernelTrap::AsyncRace`/`AsyncUnwaited` (`oracle/kernel_async.rs`); `tests/kernel_async_copy.rs` (`wait_pending_lands_only_the_older_groups`, `reading_the_destination_before_the_wait_is_an_async_race`); cpu-c `kernel_async_copy_emit_agrees.rs`. Predicated/zero-fill and cache-hint forms are U13, not built |
| 64 | Escape: `tl.inline_asm_elementwise(asm, constraints, args, dtype, is_pure, pack)` | No -> **Yes** | `ExternDecl{params,results,effects: Effects{pure,volatile,..},bodies: [TargetBody{BodyText::Asm{template,constraints}}],reference}` + `Stmt::ExternCall`; tuple results allowed; `tests/kernel_extern_check.rs` (`inc()` has a PTX `Asm` body with constraints `=r,r`), `kernel_extern_run.rs` (`the_reference_is_what_the_oracle_runs`); cpu-c `kernel_extern_emit_agrees.rs`. The oracle runs the declared `reference` kernel, not the asm. No `pack` parameter |
| 65 | Escape: `ttg.inline_asm` (once per thread, explicit layouts), Gluon asm taking smem/tmem descriptors | No -> **Partial** | non-pure, `volatile`, `sync` extern calls exist (`Effects::opaque`, `with_sync`; `kernel_extern_check.rs` `a_sync_call_is_a_barrier_site`). `Effects.reads/writes` must be empty in v1, no pointer or memdesc operands, so smem/tmem-operand asm is not expressible |
| 66 | Escape: Extern libdevice / math library calls (`tt.extern_elementwise(libname, libpath, symbol, pure)`) | No -> **Yes** | `BodyText::Symbol{library,symbol}` per `TargetBody` (e.g. `rocm`/`ocml`, `cuda`/libdevice) on a pure `ExternDecl`; `tests/kernel_extern_check.rs` (`inc()` with a `Symbol` body, `a_body_must_be_of_a_sane_shape`), `kernel_extern_run.rs`. No `libpath` field (library name only) |
| 70 | Unique: `triton.language.extra` per-backend libdevice, `tl.target_info` / `tt.get_num_programs` | No -> **Partial** (agent said Yes; lowered by the lead) | per-backend libdevice = one `TargetBody` per target on an `ExternDecl` (row 66); `tt.get_num_programs` = `ExecId::GridDim`. `tl.target_info` (compile-time backend query) has no IR node; the survey's own IR addition was "covered by ExternCall" |

## Examined, unchanged (<= 25)

- 2: no tile value type; no join/split/interleave/cat/flip/arange ops (Layout algebra is vocabulary only).
- 8: computed multi-axis `Place::Logical` coords exist; pointer values and ptr<->int still absent (D-R4-3).
- 13: shuffle-Idx and shared indexed load gather per thread; no tile value type to gather over.
- 15: `Stmt::Mma` has C-in/D-out, i8->i32, f16/bf16->f32, 3D via grid; no fp8-operand dot run anywhere (grep Fp8 in mma tests = 0).
- 17: no fp8 promotion-interval accumulator; `k_step` exists only for tf32/f16/int policies.
- 19: `Stmt::Mma` is warp-level `mma.sync` only (`MmaAtomError::BadScope`); no wgmma, tcgen05, tmem.
- 24: `RowSort` still last-axis only; no sort-dim test.
- 26: integer `Select` landed 0.40.11 (before a634b09); no `swizzle2d` node or test.
- 29: atomic store/load only as `Exchange`/`Or 0`; spin is oracle `CrossBlockWait`; no poll/timeout.
- 35: no `float8e4b15` (grep B15 in vocab and core = 0).
- 36: only `I4`/`U4` kernel loads (`kernel_subbyte_load.rs`); no `F4E2M1` packed load in kernels.
- 40: lane-wise packed operators typed (`packed_values.rs`) but kernel oracle refuses packed values (`Unsupported("packed value")`).
- 44: `LaunchShape::block` is the thread count; no `num_warps` hint.
- 47: `KernelInterface::max_regs`/`launch_bounds` are emitter-set outputs, not an IR input.
- 52: `Place::Alloc` Shared incl. dynamic `Extent::Param` (cpu-c U16); no memdesc views.
- 60: DSMEM, cluster barrier, `num_ctas` yes; multicast TMA and CLC absent.
- 61: Layout/Swizzle canonical text exists; no `ConvertLayout`, no mma/dot-operand layout conversion op.
- 67: Print/Assert only as an opaque volatile extern; no test or oracle semantics for either.
- 68: backend list is a registry concern; Intel/CPU still unverified.

## Rows where the status at a634b09 looks WRONG

- 25 (Partial, understated): per-block shared histogram with atomics was already run (`kernel_atomic_shared.rs`) at a634b09; fixed above.
- 51 (No, understated): `GridDim` + loops + atomics existed at a634b09; fixed above.
- 40 (Partial, evidence misstated): the first re-score said "no vector arithmetic beyond lane-wise `Select`"; lane-wise operators and per-lane `Cast` were typed since 0.27.2 (`packed_values.rs`). Status stays Partial because the kernel oracle does not run packed values.
- 13 (Partial, possibly understated): lane gather by `Shuffle{Idx}` and shared indexed load run today; only the "register tile" wording keeps it Partial.
- 29 (Partial, possibly understated): `Exchange` (no old) is an atomic store and `Or 0` with `old` is an atomic load; only poll/timeout and a dedicated load/store are missing.
- 44 (Partial, arguably Yes in substance): `LaunchShape::block` fixes warps per block; the row's hint wording is what keeps it Partial.

## Remaining `No` rows (24), by what they would take

- Compile-time specialization and compiler facts, no IR channel (3, 5, 47, 48, 49): `constexpr` meta-params and `divisible_by` facts, `assume`/`max_contiguous`, a register-budget input, autotune key and config space, tile-size meta-params.
- Tensor-descriptor and block-pointer memory model (10, 11, 12, 30): block-pointer view with order/advance, a descriptor operand kind, tile-offset load/store, row gather/scatter, reducing store (ROADMAP U8c, U8 descriptors).
- TMA, mbarrier and async proxy (56, 57): `mbarrier` resource with arrive/expect/wait, TMA copies, proxy fences (U8c, U4 S7).
- Hopper/Blackwell tensor-core and memory (19, 59, 17): wgmma/tcgen05 atoms with warpgroup scope, tensor memory, fp8 promotion interval (U7 later slices).
- Warp specialization (58): per-partition warps/regs, named barriers (U4 S6b-2).
- Cache and access hints (9): per-access `CacheHint` (U13 covers `CopyAsync` only).
- Global scratch (14): a kernel-declared per-program global buffer (`MemorySpace::Global` is not declarable).
- Grid-level ordering (31): programmatic dependent launch statement or interface field (U17).
- Loop attributes (43): `Stmt::For` hints for unroll and friends (U18).
- Schedule rewrites (50): `swizzle2d` as a node or a conformance kernel (integer ops on `BlockIdx` likely suffice; no test).
- RNG (39): Philox counter-based generator node, or a conformance kernel plus `umulhi`.
- AMD-specific (63): buffer ops, TDM, scaled up/downcast, `in_thread_transpose`, `cond_barrier` (backend lowerings of generic ops).
- Debug effects (67): `Print`/`Assert` effect statements with oracle semantics.
- Profiling (69): Proton record ops (out of scope per the survey).
