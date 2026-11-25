# The IR union: asks to other lanes (LANE rows) and the consumers or sources that fire the TRIGGER rows

Written 2026-11-25 against `origin/main` b21af35c, from the third re-score (`docs/ir-union-rescore.md`, "Third measurement"; evidence in `docs/ir-union-evidence/rescore-2026-11-25/<survey>.md`). Docs only. Asked by the PM through pm-kernels. Row numbers are the survey's row numbers. Ownership marked "proposed" is the Unpopped lane's reading of the ROADMAP and the survey notes, for the PM to rule; where the records name an owner the owner is quoted.

## 1. LANE rows (23): who owns them, and the exact ask

Format: row | owning lane | the ask (what to build or expose, where, proven by what) | what it unblocks.

### 1a. KISS: element and numerics rows (9 rows; corrected 2026-11-25 against KISS's own RU.1, `origin/union-rows` ff4a9063)

- CUDA 56 | KISS | **corrected:** the `f4e2m1`, `f6e2m3` and `f6e3m2` tokens and their packing are already pinned (KISS-CLASSIFY-6.1-0014 and -0015, `conformance/src/derivation_vectors.rs:705`, tests `test_classify_mx_f4e2m1_format` and `test_classify_mx_f6_formats`), marked convention. What remains: read a second source (CUTLASS, Mojo or the OCP text) for the packing and any interleave layout, and pin it by a codec round-trip vector in `conformance/src/dtype.rs` (which has no f4/f6 codec today) with the source cited; Unpopped then adds the oracle sizing and kernel loads and stores | CUDA 56, 61, 62 and the fp4/fp6 GEMM rows (D5 is ruled in).
- CUDA 61 | KISS | **corrected:** the fp6 packing is pinned (6.1-0015); the remaining ask is the second-source check and the `dtype.rs` round-trip vector (same as CUDA 56); Unpopped's decode exists only as a `Cast` intermediate | CUDA 61.
- CUDA 62 | KISS | **corrected:** fp4 packing is pinned (6.1-0014); the remaining ask is the NVFP4 scale pairing (a UE4M3 scale per 16 elements: whether it fits the keyed scale dtype and block length, 6.6-0022/-0022a, is unread by KISS) plus the second-source check and round-trip vector | CUDA 62; with CUTLASS 21, 78, 79 and ThunderKittens 54.
- CUDA 64 | KISS | add the `F128` token to the dtype set and key codec; Unpopped then adds oracle arithmetic (a software binary128) and a conformance kernel | CUDA 64.
- CUDA 65 | KISS | add `I128`/`U128` tokens; Unpopped adds the oracle integer width and 128-bit atomic forms | CUDA 65, Mojo 48.
- Mojo 44 | KISS (owns the dtype tokens and key codec; Unpopped ROADMAP C10 is the codec-clause row) | add the `F8E3M4` token; U1 recorded "e3m4 not added" | Mojo 44.
- Mojo 48 | KISS | add `I128`/`U128`/`I256`/`U256` tokens (one request with CUDA 65) | Mojo 48.
- Slang 50 | KISS (Unpopped ROADMAP I3, "numerics axes ... one synchronized cut", whose status cell still reads `held(KISS, sk4 seam)`; KISS's schema is sk8, so that label is stale) | agree the per-function precision mode (fast vs precise, contract, NaN/Inf assumptions) as one key-affecting field; Unpopped adds the oracle rule and checker | Slang 50.
- Slang 52 | KISS (Unpopped ROADMAP C2 and I3; the row names C2, C10 and I3 are Unpopped's, not KISS's) | the NaN/Inf preservation mode as part of the same cut | Slang 52.

### 1b. Fuel and KISS: the plan-layer Contraction generalisation, ruled A now (D2; 11 rows)

The PM ruled Q3 to build it now. `Access::Contraction` is rank-2, single-K, with one batch role; the kernel layer already runs these kernels, so the work is the plan layer and the `StructureKey`. Proposed split (the PM rules): **KISS owns what changes the schema and the key; Fuel owns what changes how the planner and dispatch consume it.**

- CuTe 39 | KISS and Unpopped | **corrected:** rank above 2 and batched are largely keyed already (KISS-CLASSIFY-6.6-0023 pins every `ctr` cell of rank above 3 and a caller-declared batch set; `derivation_vectors` has a rank-4 batched cell). "`Access::Contraction` is rank-2" is Unpopped's plan-layer type, not KISS's. What remains: a rank-3 batched round-trip vector on the Unpopped side, and KISS checking that 6.6-0016's single M/N/K assignment extends | CuTe 39.
- CUTLASS 3 | KISS | more than one batch role and nested (hierarchical) modes | CUTLASS 3.
- CUTLASS 4 | KISS | more than one contracted K axis (GETT, tensor contraction) | CUTLASS 4.
- CUTLASS 5 | KISS | per-operand major-ness (N/T) for A, B, C and D | CUTLASS 5, CUDA 55.
- CUTLASS 31 | KISS | source C of a different dtype than D, and a compute dtype distinct from accumulator and output | CUTLASS 31, CUDA 44.
- CUTLASS 23 | KISS | an epilogue-produced scale-factor output (SFD) as an output role | CUTLASS 23.
- CUTLASS 15 | Fuel (consumes), KISS (schema) | structured operands (triangular, symmetric, Hermitian) as operand properties in the schema, and Fuel's planner accepting them | CUTLASS 15.
- CUTLASS 35 | Fuel | per-row [M] broadcast (per-token scale) accepted by the planner (it declines it today); the kernel layer already does it | CUTLASS 35, ThunderKittens 35.
- ThunderKittens 35 | Fuel | the same broadcast of a per-row vector operand across a tile | ThunderKittens 35.
- CUTLASS 38 | Fuel | TopK in the epilogue (softmax stays inside the attention kernel) | CUTLASS 38.
- CUTLASS 51 | Fuel | conv fused epilogue (bias, activation, per-channel scale) | CUTLASS 51.
- (CUDA 44, 55, 81 are the DECISION rows that move with this ruling: higher-rank, mixed-dtype and multi-K (KISS), operand major-ness (KISS), split-K and stream-K as a plan-level `k_split` axis and work division (Fuel).)

### 1c. Emitter lanes (2 rows) and Baracuda (1 row); Mojo 71 and 72 ruled by the PM 2026-11-25

- Mojo 71 | **ruled (PM 2026-11-25): HIP is Baracuda's (a new emitter crate, after its in-flight rungs and the sm_90 rows); Metal Shading Language is Vulkane's (a ranked row after its in-flight rungs)** | state whether a HIP/Metal emitter is planned; if yes, record the AMD/Apple atom constructors and the memory hints it needs; Unpopped then adds the `MmaAtom` constructors from published ISA tables | Mojo 71.
- Mojo 72 | **Baracuda, as the HIP emitter owner (PM 2026-11-25)** | state which AMD intrinsics (permlane, `ds_read_tr`, buffer resources) it wants as named operations rather than opaque `ExternCall`; Unpopped adds typed externs with declared effects | Mojo 72.
- ThunderKittens 58 | Baracuda (target registry and the sm_90 and sm_100 emitters) | the Hopper, Blackwell and Rubin target entries belong in the target registry, not in IR statements: record which targets Baracuda emits and the capability set of each in `unpopped-vocab`'s `target.rs` | ThunderKittens 58, CUTLASS 96.

## 2. TRIGGER rows (44): the trigger, whether the consumer exists, what makes it fire

Format: rows | the trigger | does the consumer exist today (yes/no, where) | what ask makes it fire.

### 2a. "Baracuda's sm_90 emitter asks, or a descriptor operand kind lands" (U8c): 16 rows

- CUDA 25, 46, 47, 48; ThunderKittens 9, 22, 23, 25, 26, 48; Triton 10, 11, 12, 30, 56, 60 | ROADMAP U8c: tensor-map (descriptor) copies, multicast, OOB zero-fill, descriptor atomics and reduces, block pointers | **partly: the consumer exists but has not asked.** `baracuda-cuda-emit` 0.56.12 (baracuda `origin/main`) emits kernel statements and spells `CopyAsync` (8 symbol hits) and `Place::ClusterPeer` (1); it spells none of `Wgmma`, `MbarInit`, `CopyBulk`, `MatrixLoad`, `Tcgen05Mma`, `GridDepWait`, or a named barrier (0 hits each; the control, `CopyAsync`, finds 8). tensor-map symbols appear in 10 Baracuda files and `wgmma` in 2 (driver bindings and kernel sources; not checked further here), so the capability is known to it | Baracuda records the requirements in section 2b; the PM rules the ask is made now. Unpopped then designs the descriptor operand kind to match what Baracuda needs to spell.

### 2b. The requirements Baracuda records (8 grouped lines)

1. **Statements the IR now emits that the emitter must accept** (all in `unpopped::kernel`, 0.56.x): `MbarInit`, `MbarArrive`, `MbarCompleteTx`, `MbarWait` (the wait is by phase parity, no token), `CopyBulk` (a 1-D bulk copy that completes on an mbarrier, `cp.async.bulk` with `complete_tx::bytes`), `CopyAsync` with `valid` (zero-fill or predication, the `src-size` operand), `cache` (`.ca`/`.cg`) and `l2_prefetch`, `MatrixLoad` and `MatrixStore` (`ldmatrix`/`stmatrix`, `.x1`/`.x2`/`.x4`, `.trans`), `Wgmma`, `WgmmaCommit`, `WgmmaWait` at `Scope::Warpgroup`, `Tcgen05Mma`, `Tcgen05Commit`, `Tcgen05Ld` with `MemorySpace::Tensor`, `BarrierKind::Named` and `NamedArrive`, `GridDepWait` and `GridDepLaunchDependents`, `Place::ClusterPeer` and the `Cluster` barrier scope. The emitter must either spell each or refuse it by name.
2. **What the IR does not model and the emitter must derive:** the `wgmma` shared-memory matrix descriptors (start address, leading and stride byte offsets, swizzle mode) from the allocation and its layout; tensor-memory allocation, deallocation and CTA-pair mode for `tcgen05`; the `mbarrier` transaction-byte count from the copy size. Baracuda records how it derives each so the IR does not duplicate it.
3. **Tensor maps (not yet in the IR):** state the shape it wants: a kernel parameter of `CUtensorMap` type built host-side with `cuTensorMapEncodeTiled`, the coordinates and box dimensions a copy statement supplies, and which of load, store, reduce, multicast and gather it needs first. Unpopped designs the descriptor operand kind to that shape (ROADMAP U8c; the re-read of US12645455B2 comes first).
4. **Interface data the emitter must carry out:** `KernelInterface` fields (launch bounds, cooperative flag, dynamic shared opt-in, cluster dimensions, register cap) as kernel attributes, and the `max_regs` it compiled with reported back.
5. **Semantics the oracle states without a second source** (`mbarrier` phase and parity, the `stmatrix` row rule, `wgmma` and `tcgen05` fragment layouts): Baracuda runs the emitted code under `ptxas` or on hardware and reports every disagreement with the oracle as a finding to Unpopped. That report is also the "second independent lane-table source" for 2c.
6. **A conformance harness:** one command that runs an Unpopped conformance kernel through the Baracuda emitter and compares against the oracle, so a new statement lands with a spelling and a test on both sides.
7. **Refusals by name:** a statement the emitter does not yet spell returns a typed refusal that names the statement, as the serial C reference does, so a gap is a row and never a silent miscompile.
8. **Version coupling:** `baracuda-cuda-emit` already tracks Unpopped's MAJOR.MINOR (0.56 now); a breaking IR release batches the new statements so the emitter breaks once.

### 2c. "A second independent lane-table source" (U7 S3, U8b): 7 rows

- CUDA 38, 39, 42; CuTe 36, 42; CUTLASS 25; Triton 18 | ROADMAP U7 S3 and U8b: no PTX sparse or scaled constructor, and the `ldmatrix`/`stmatrix`/`mma.sp` lane tables stay held, until the lane tables are checked against a second independent source | **a source, not a consumer, is what is missing, and it may exist:** candidate sources (named from memory, not checked for this note): CUTLASS's `cute/atom/mma_traits_sm80.hpp` family and LLVM's NVPTX tests; and Baracuda can produce one by running emitted code under `ptxas` or hardware (2b line 5) | PM ruling that the Unpopped lane may read those sources itself (work, not an ask to another lane), and Baracuda's report from 2b line 5.

### 2d. "No consumer yet; revive only if an emitter must keep a 128-bit vector as one access" (U15, parked 2026-11-21): 8 rows

- CUDA 59; CuTe 20, 48; CUTLASS 83; Mojo 52, 53, 54; ThunderKittens 55 | packed locals and SIMD values (8- and 16-lane packed values within 128 bits are built, 0.56.x; what is parked is the lowering that keeps a vector as one access) | **not today:** `baracuda-cuda-emit` has not asked for vector-as-one-access (`float4` loads) | Baracuda states whether it needs it; the claims of US11262989B2, US9052888B2 and WO2013070616A1 are read first (PATENT-SWEEP U15, NOT CLEARED, unread).

### 2e. Other triggers

- CuTe 62, CUTLASS 61, Triton 31 | U17: "a two-launch oracle is the follow-up if a consumer needs the ordering checked" | no: the oracle has one launch and PDL is a host-side edge | a consumer (Fuel dispatch or a Baracuda host harness) asks for cross-launch ordering to be checked; 1 to 2 PRs.
- Slang 8, 9 | U21 textures and bindless: "waits on a consumer" | no: Fuel uses 0 of 192 kernels, FlashInfer 0 (ROADMAP U21) | Fuel or fishstringer names a texture kernel it needs.
- Slang 5; Triton 2, 13 | U3 remainder: `ProgramModel`/`BlockSpmd` lowering, append/consume buffers, block tensors and `tl.gather`, "each when a consumer needs it" | no: no Triton front end exists in the portfolio | a Triton-source frontend lane (not started) or fishstringer asks.
- CUDA 4 | U40 constant tables: "a lifted kernel or a lane needs a table" | possibly: the U19 census is measurable | the lane measures how many census kernels carry a `__constant__` table; the lane can fire this itself.
- CUDA 15, 79 | `setmaxnreg` and warp-specialization launch attributes, "ride on `LaunchShape` fields when a backend needs them" | no: the Baracuda emitter does not spell them | Baracuda records `setmaxnreg` among its requirements (2b line 1).
- Mojo 49 | **superseded, now READY (PM 2026-11-25, ROADMAP U58):** the 2026-10-28 seam hold on the index-operand dtype of gather and scatter is lifted; CireSnave's 2026-10-28 sentence is an ordering, never an exclusion | none: Unpopped builds it | nothing to fire.
- Slang 30 | U38a-2 float atomics (CAS loops), parked 2026-11-22: "revisit when the census shows CAS as the largest remaining wall" | measurable | the lane re-runs the census; it can fire this itself.

Row count: 16 + 7 + 8 + 3 + 2 + 3 + 1 + 2 + 1 + 1 = 44 as first counted; Mojo 49 has since become READY, so 43 rows still wait on a trigger.
