# Unpopped roadmap

What this lane is doing, in what order, and **what each item waits on**. Read at
`origin/main` `2b94d5d` (0.22.10) on 2026-10-24; every row names the ref it was
read at, so re-read it before acting on it. History is in `CHANGELOG.md`;
knowingly-undone work and its reasoning is `docs/deferred.md`. This file is the
queue, not the archive: delete a row when it lands.

## 1. In flight

| # | Item | State | Waits on |
|---|---|---|---|
| 2 | **Run `emitted_slang_compiles_with_slangc` on the self-hosted runner** (CireSnave 5.1g/5.1h: an environment-bound test is run there, never ignored). It stays `#[ignore]` meanwhile: a documented, counted last resort. Owner: this lane. Detector: the runner job, once it exists (no hosted job can run it) | `slangc` exists on the PM's PC (`C:/VulkanSDK/1.4.350.0/Bin`) | PM: runner registered for Unpopped, which is a **PUBLIC** repo, so the **ephemeral** runner (mandatory for public repos; persistent is private-only) under 5.1h (push-to-main, schedule, dispatch; same-repo PRs only behind the environment gate; fork PRs never run on the runner, so fork code is never executed there) + `slangc` on the `gh-runner` PATH. Then this lane adds the job and drops the attribute |
| 3 | **Harness family**: external test/profile harnesses wrapping Unpopped, now in its own public repo `ThinkersJournal/unpopped-harness` (crate `unpopped-harness` 0.0.13 on main at `82296d20`, 0.0.14 pending; lanes cannot push there, the PM pushes by SHA; protection requires verify / cargo deny / SPDX). Targets: cpu-c (reference), CUDA, Vulkan (Slang). Done and measured there: exhaustive 2^32 f32 relu on all three; CUDA `Max(x,0)` == Relu; Slang relu is exact on AMD 610M and RTX 4070 only with `-denorm-mode-fp32 preserve` (Unpopped#84; PM ruling 2026-10-21: a kernel whose measured result flushes declines Exact). Design: `docs/harness-family.md`. Owner: this lane | Next tracks (binary ops, f64, chains on Vulkan, timing) are this lane's call. GPU and Vulkan tests need a self-hosted runner (PM list) |

## 2. KISS items this lane consumes

| Item | State | Where |
|---|---|---|
| Derivation vectors, KISS `026feba` (0.4.3) | **63 of 63 match**; `KNOWN_GAP` is empty and asserted exactly | `crates/unpopped-vocab/tests/kiss_derivation_vectors.rs` |
| M/N/K and batch roles (§6.6-0016, KISS#543) | **closed** at 0.19.0 behind `CellFacts::contraction_roles`; `ContractionRoles` since 0.20.0 | #64, #66 |
| Weight role (§6.6-0019) | **closed** in 0.17.3; `WeightRoleNotDeclared` and `ContractionRolesNotDeclared` are this crate's decline names, not KISS's | `CellFacts::weight_operand` |
| §6.6-0021 | the infallible `structure_key` still keys a scale dtype at operand 0; `try_structure_key` declines. Deprecating the infallible form means migrating ~200 callers, which the PM ruled is **not this lane** (bulk mechanical work, CireSnave's cost rule) | `CHANGELOG.md` |
| An FMA SIMT cell can never serve an `st` key (§6.17-0002) | **known gap**, told to the KISS lane; a caller wanting bit-exact F32 gets a decline until KISS adds a "third value" row | |

KISS `main` moves faster than this lane re-vendors. Re-vendor when KISS says a
blob changed, not on a schedule, and record the commit in `kiss/README.md`.

## 3. Next

1. **`Approx` qualifier** (`docs/idiom-lifting-design.md` §4.3): the full fix for
   `__expf` / `__logf`, which #57 only made refuse. Needs KISS-coordinated
   numerics axes.
2. **Re-vendor on KISS change**, with the derivation vectors re-run
   (`cargo test -p unpopped-vocab --test kiss_derivation_vectors -- --nocapture`).

## 4. Held or blocked

| Item | Waits on |
|---|---|
| **Leg B**: on-device `__expf` round-trip (`docs/idiom-lifting-design.md` §5). baracuda's pin at `baracuda-cuda-parse/tests/convert.rs:84` fails when they take 0.15.0 (PM relayed it) | a rig: no P40 yet |
| Joint GPU milestone plan, `docs/joint-gpu-milestone-plan.md`. Ownership: Unpopped owns capability rows and per-sm plan decisions, baracuda owns CUDA spelling per sm. U1 (#40) and U2 (#42) are done; M4/M5 are blocked on the P40 | PM, before more U-items |
| Moving `ArchSku` into `baracuda-cuda-vocab` | baracuda's KISS registry repoint (`deferred.md:33-34`), then PM says go; see P2 |
| Groupshared / cooperative scope call (`docs/groupshared-sizing.md`) | CireSnave or PM |
| CpuC declines `Select` for every narrow float | parity gap, not a bug; no owner |

## 5. Standing constraints

- Never run, launch or spawn anything on Opus; subagents carry an explicit
  `model: sonnet|haiku`.
- Every pushed change moves the version, docs and CI included; the PM allocates
  the number at gate. Pre-1.0, a breaking change moves the second number.
- kiss-ref enums are `#[non_exhaustive]` from 0.4.0: a new `match` over one
  needs a wildcard arm.
- `cargo public-api` and `cargo semver-checks` are not installed here; an API
  diff in a PR is "read, not a tool run".

## 6. Spec-to-roadmap audit (PM task, CireSnave's idea, 2026-10-24)

Every undone item found in a spec or design doc and not already a row above, read at
`origin/main` `2b94d5d` (0.22.10). **Rows are recorded, not started.** Status: **open** (this lane
could start it), **held(X)** (waits on X), **done** (with the PR). "Call" means it needs a CireSnave/PM
decision before anyone starts; each is an `[ASK]` to the PM. "no proof" means the closing PR or code
was not found in `git log`, `CHANGELOG.md` or the tree, which is not proof of absence.

### 6.1 Emitters and backends

| ID | Work | Source | Status |
|---|---|---|---|
| E1 | `unpopped-cpu-c` and `unpopped-slang` accept only `Schedule::Scalar`; Vectorized, Strided, Reduction, RowReduce, Contraction, Scan, Window, RowSort, Im2Col are listed "follow-ups" | groupshared-sizing:113; slang `lib.rs:52,330`; cpu-c `lib.rs:37,186` | open. parked by the PM 2026-10-24 (was Call 1; the row stays recorded): how far the reference emitters should grow |
| E2 | Slang declines 8 unary ops, "need polyfills": erf, gelu, erfc, asinh, acosh, atanh, cbrt, lgamma (measured by harness 0.0.14; cpu-c and cuda emit all 39); and 2 binary ops, `copysign` and `nextafter` (harness 0.0.15; cpu-c and cuda emit all 15) | Slang decline text; harness CHANGELOG 0.0.14, 0.0.15 | open. The Slang `INFINITY`/`NAN` constant spelling is fixed (#106, 0.22.14); the declined ops here are a different gap |
| E3 | Slang preludes for c64/c128, both FP8s, i4/u4/b1, f16/bf16: the 9 `NotYet` cells, "no current consumer" | deferred:655,678-710; `dtype_lowering_coverage.rs:162-193` | open, no consumer |
| E4 | Slang has no recorded NaN-ternary (N2) survival check; every new target needs its own | conformance:115-124; deferred:1059 | open |
| E5 | `GeneratedKernel` has no structured ABI manifest and `source` is a `String`, not a word stream (SPIR-V `[u32]`); "editor's position, open to challenge" | README:58-64; `backend.rs:97-110` | open, no demand yet |
| E6 | `emitter_version` is a producer-supplied string; a `Backend` method for it is a candidate | CHANGELOG 0.21.0 "Not here" | open |
| E7 | Vulkan bf16 has no `arith` token: the first vulkan backend gating halves needs a permanent decline or a cooperative-matrix route | deferred:150-185 | open, latent |
| E8 | Temp-binding pass: truncation must be settled before `Sqr`/`Relu`/`Gelu`/`Silu`/`Sign`/`Max`/`Min` extend to 8/16-bit | deferred:640-654 | open, latent (trigger: narrow-int extension) |
| E9 | `cpu_end_to_end` tests the first C compiler found, not every one | deferred:1336; `cpu_end_to_end.rs:84` | open |
| E10 | `k <= 1024` row-sort bound is in doc strings only; `validate_row_sort` does not check it | deferred:1245 | open, "not a defect" per the doc |
| E11 | Oracle cannot tell `Skip` from `ZeroFill` (`alloc_output` zeroes) | deferred:919 | open, recorded limitation |
| E12 | Slang front end: re-run `slang_corpus_report` on a current Fuel snapshot | slang-corpus-sizing:143-151; CHANGELOG ~802 | **done** 2026-10-24 (Unpopped 0.22.14 on fuel `ac29b53c9`, 147 `.slang` files read from `git archive` of Fuel's `origin/main`): **0 accepted**. 94 Unrecognized (44 NotElementwise, 18 read of a buffer not named `inputK`, 18 unknown call, 14 unknown identifier); 53 Inexpressible (36 `groupshared`, 16 `InterlockedCompareExchange`, 1 `RWByteAddressBuffer`). Fuel's kernels use their own binding layout, so 0 is this front end's yield on Fuel as-is, the baseline for E13, not a defect. First measurement since #22/#25 |
| E13 | Slang front end: declared-type buffer-role discovery (4/147 yield), strided/broadcast access (up to 70/147 files), more idioms | slang-naming:103-105; slang-second-blocker:106-127; corpus:127-141 | open. parked by the PM 2026-10-24 (was Call 6; the row stays recorded): worth the cost at that yield? |

### 6.2 Idiom lifting and the IR (`docs/idiom-lifting-design.md`)

| ID | Work | Source | Status |
|---|---|---|---|
| I1 | `Lifted.hints` and `Frontend` intrinsic/idiom fields; move `unary_fn`/`binary_fn` out of core; `#[non_exhaustive]` on `Frontend` and `Lifted` (neither is today). Breaking | idiom:311-332,802; `convert.rs:121`, `lift.rs:113` | open; PM allocates the breaking version |
| I2 | `Origin`, `Hint`, `Provenance.techniques` (none exist) | idiom:335-369,804 | open |
| I3 | Numerics axes: `AccumSpec`/`ReductionAccum` variants, `denormal` on `OpDef`; one synchronized cut | idiom:163-176,794-801 | held(KISS, sk4 seam) |
| I4 | Round-trip harness: per-frontend recognizer-fixture census, Leg A for the in-tree Slang front end, Leg C (only Leg B is in §4) | idiom:381-448,807-811 | open; CUDA legs held(baracuda) |
| I5 | Intent gaps: block-dequant read-through `View`, RowReduce second input (gamma), mixed-dtype `Contraction`, `Access::Attention` (absent from `ir.rs`). Needed by flash_attention, matmul_q4_0_tiled, qmatvec_q4_0 | idiom:195-226; groupshared:115-117 | open, behind the §4 groupshared ruling |
| I6 | Slang RowReduce schedule (needs E1), then the 4070 CUDA-vs-Vulkan A/B | idiom:764-787 | open, behind E1 and the §4 ruling |
| I7 | Technique registry (`techniques/registry.json`, `REGISTRY`, `verify_rows`, mutation-controlled test) and `"transferable": false` marks; the doc says stop if no consumer reads it | idiom:515-731 | open. parked by the PM 2026-10-24 (was Call 2; the row stays recorded): name a consumer or do not build |
| I8 | Autotuner (§9) | idiom:733-762 | held(each runtime owner); not Unpopped code |
| I9 | KISS-Consume idiom rules, only if KISS-Consume needs them | idiom:828-831 | open, conditional |
| I10 | `cuda:sm61` token is baracuda-cuda-vocab's | idiom:462-469 | held(baracuda), unverified |
| I11 | Telemetry `merge_reports` drop / mint a `TargetId` | idiom:475-479 | **done**, #36 (0.14.3) |
| I12 | `__expf`/`__logf` refuse | idiom:790-793 | **done**, #57 |
| I13 | sm_61 capability row | idiom:493-498 | **done**, #34 (0.14.1) |

### 6.3 Conformance, accuracy and the KISS seams

| ID | Work | Source | Status |
|---|---|---|---|
| C1 | Accuracy key: dtype and precision-mode axes, and a Vulkan table (until then a Vulkan target has no band). The target axis is **done** (95d2ed6) | conformance:285-389; `contract.rs:1281-1298` | open. parked by the PM 2026-10-24 (was Call 4; the row stays recorded): the key's shape; the doc says do not collect values first |
| C2 | N2 NaN propagation: gate or weaken for targets that cannot preserve NaN (belongs to KISS) | conformance OPEN-4:421-461 | parked (PM 2026-10-24, on KISS's recommendation): gate-decline later, and no spec text until some device is measured not preserving NaN. The unblocker is C14 |
| C3 | Before chaining FloatControls check `min(instance, device apiVersion) >= 1.2` | conformance:449-461 | open, latent (no gate exists) |
| C4 | Comparator semantics for RowSort NaN-greatest, TopK, gather/scatter OOB, FP atomicAdd | conformance OPEN-3:463-469 | open |
| C5 | Nothing compels callers to use `required_fidelity`; `compare` takes any `Fidelity`. The docs disagree on its non-test callers (one vs zero), uncounted | conformance:266-270; `contract.rs:1297` | open |
| C6 | `vec_width`: `vbytes <= 16` is CUDA's `float4` limit inside the neutral keyer; the right parameter is `(component_count, element_type)`; moves CUDA keys | normative-seams:94-124; deferred:1128-1180; `structure_key.rs:2320` | held(baracuda #252). Then this lane writes the corpus-regen plan and version note and sends it to the PM (PM 2026-10-24) |
| C7 | Questions never put to KISS: is the move/compute boundary KISS-stated; are the four "computable" dtypes KISS's; the rest of the seam list | normative-seams:87-90,173-177,336-337 | open (file them) |
| C8 | KISS filing: a home for the variant-relative "bit-identical to the default lowering" axis | deferred:462 | held(KISS) |
| C9 | FKC-to-KISS-Contract port; no home yet for `link_registry`, multi-output `aliasing`, `blurb`/`precision.notes`; `layout_guarantee` | fkc-mapping:73-146 | open; `layout_guarantee` held(KISS R1.13, KISS#316) |
| C10 | Proposed §6.7 codec clause (version prefix inseparable from dtype sub-tokens); c64 meaning-flip migration note | dtype-spelling-delta:80-98 | open; clause is KISS's call; migration note no proof |
| C11 | Panic text as a cross-repo API, REPORTED and not re-measured | normative-seams:205-209 | open |
| C12 | Per-device decline step 3: adopter integration (ledger location, measure-on-first-run, apply `UseArgs`) | design-per-device-decline:113,121-123 | held(Fuel) |
| C13 | Derivation vectors, weight role, contraction roles | kiss/README, §2 | **done**, #64, #66, 0.17.3 |
| C14 | Measure a Vulkan device's NaN, infinity and signed-zero behaviour with the harness: the unblocker for C2 | PM [FIX] 2026-10-24; conformance OPEN-4:449-461 | **done**, `unpopped-harness` 0.0.16 (#16): AMD Radeon 610M and RTX 4070 Laptop both 0 mismatches over all 2^32 f32 inputs on `isnan`, `isinf` and `zero_sign`. Neither is measured NOT preserving, so C2 stays parked. Not measured: the device's claimed `shaderSignedZeroInfNanPreserveFloat32` (`vulkane` 0.16 does not expose it), other devices, f16/f64 |
| C15 | `f8e6m2` was counted usable (22 vs KISS's 21 usable, 3 reserved) | KISS-CLASSIFY-6.1-0001 | **done**, #104 (0.22.12) |

### 6.4 Ownership, parsers and the cross-project plan

| ID | Work | Source | Status |
|---|---|---|---|
| P1 | `FrontendRegistry` parser registry; where the neutral half of `lift.rs` lives; no code written | parser-ownership:57-64,162-187 | open (design call at the PM gate) |
| P2 | `baracuda-cuda-parse` donation: the CUDA slice of `convert.rs`, then evict the `cuda:` token and `KernelSku.arch` | deferred:33-34; `sku.rs:281` | held(baracuda, KISS registry repoint (b)); `KernelSku.arch` held(Fuel) |
| P3 | Parse ownership for Slang with Fuel; whether GLSL/Metal are in scope; owners for OpenCL/HIP/ISPC | parser-ownership:106-115,195-205 | open. parked by the PM 2026-10-24 (was Call 8; the row stays recorded) |
| P4 | Joint GPU plan beyond U1/U2: M2 lightbulb off the fuel git pin; B1 NVRTC sm_61 compile; M3; M4/M5 (P40); M6 (Qwen 27B-class); AMD/Intel via Vulkan (66/167 kernels have SPIR-V); no CUDA CI runner | joint-gpu-plan:18-19,95-145,203-223 | held(baracuda B1, fuel, lightbulb, P40 arrival and LAN). **Call 7**: exact model names, 27B quant, a CUDA runner |
| P5 | qwen3_5 decode: planning test like `qwen3_dense_ops_plan_for_sm61.rs`; causal conv1d is 9 operands over `MAX_OPERANDS = 8`; triangular-solve intent; matrix-state recurrence | qwen3_5-assessment:112-141; `ir.rs:1740` | held(P40, M6) |
| P6 | Harness B and C (stage profile, hotspot map) | harness-family:47-50 | **done** in `unpopped-harness` 0.0.14 (#14, unary) and 0.0.15 (binary and ternary corpora, `7ca0068a`, PM push pending); no emit hotspot in any corpus |
| P7 | Harness targets AOCL and oneMKL (oracles for contraction, not elementwise); the Fuel corpus reuse and survey decisions | harness-family:40-42,59-66 | open, no owner; contraction phase |
| P8 | Candidate generation precondition: Fuel GAP-244 (`"unpopped"` missing from `kernel_source_intern`) | deferred:1083 | held(Fuel), unverifiable here |
| P9 | Schedule language (tiling, shared-memory staging, unroll) and a search over it | deferred:614 | open. "CireSnave's spend": parked by the PM 2026-10-24 (was Call 9; the row stays recorded) |

### 6.5 The on-device harnesses (`crates/unpopped/ondevice/README.md`, 2,697 lines)

A hand-written run book and measurement log: 29 manual `nvcc` harnesses, every "Last run" on one
RTX 4070 in 2026-07/08. The generators and emit examples live in baracuda.

| ID | Work | Source | Status |
|---|---|---|---|
| D1 | None of the 29 harnesses runs in CI; CireSnave 5.1g says run them on the self-hosted runner. They need a sibling baracuda checkout, which touches the 2026-10-07 sources rule | ondevice:3,12 | deferred by the PM 2026-10-24: needs a device runner for a public repo (board 241) and is a portfolio ownership move |
| D2 | Contract-less families (contraction, reduction, multi-output, view, gather, scatter, scan, window, sort, topk, offset, dropout, im2col) need Fuel propose-first items: per-operand-dtype key (`STRUCTURE_KEY_VERSION` bump), layout/gather/offset grammar, bundle ABI, new `OpKind`s; includes `start_offset` and the `[U8,T,T]` `where` | ondevice:705-2584 | held(Fuel, KISS, PM). parked by the PM 2026-10-24 (was Call 11; the row stays recorded): pursue or park |
| D3 | f16/bf16 scalar-param ctype: "numerics decision with no oracle" | ondevice:2380-2382 | open. Part of parked by the PM 2026-10-24 (was Call 11; the row stays recorded) |
| D4 | Baracuda-side de-scopes recorded in this README, not Unpopped code: packed transcendentals, dequant-fused matmul, scan (cummax, LogCumsumExp), window (`causal_conv1d`, N-D), sort (argsort I64, >1024), fused rope, offset limits, im2col (col2im, grouped, padding modes), embedding `padding_idx`, JIT f64 params | ondevice:268-2596 | held(baracuda); f16/bf16 explicit block-scan launches open |
| D5 | DAG follow-ups: fused-reduction epilogue dedup, DAG-based flops count, `region_to_op` hash-cons | ondevice:235-238 | held(baracuda; the seam moved) |

### 6.6 Doc defects found by the audit (doc contradicts code)

| ID | Defect | Status |
|---|---|---|
| X1 | `ulp_bound` "takes no target": conformance:272, normative-seams:145, deferred:425, and the `contract.rs` KNOWN LIMIT comment | **fixed in this PR** (claim replaced, not caveated) |
| X2 | README published versions (`unpopped` 0.2.0, `unpopped-vocab` 0.3.0, emitters 0.14.4); "Eight public breaks" over nine rows | **fixed in this PR** (versions now point at crates.io) |
| X3 | `ondevice/README.md`: snippets import `unpopped::Cuda` (it is in `baracuda-cuda-emit`); `cargo run -p unpopped --example emit_*` (those examples are in baracuda); test names such as `contract::tests::scan_is_an_honest_miss_no_contract` do not exist where cited; stale line cites; `reduction_int8_validate.cu` undocumented; rope "STILL DEFERRED" vs "3 of 3 blockers CLOSED" | open (with D1's home call) |
| X4 | `groupshared-sizing:4-6` says the scope question is "answered"; §4 holds it | **fixed in 0.22.19** (answered: yes, the IR models cooperation) |
| X5 | `joint-gpu-plan:28` "plan.rs never reads the target" (false since #40); `slang-second-blocker:77-79` says Slang lowers `Select` for all dtypes (float-only); `normative-seams:292` `supports_dtype` returns bool (now `DtypeSupport`); Slang `lower()` error text says i16/u16 wait on vocabulary v5 | **fixed in 0.22.19 except the Slang `lower()` message**, which is error text adopters match (a cross-repo API): fix with a version note in a code increment |
| X6 | `deferred.md:33-34` says moving `ArchSku` waits on baracuda's registry repoint; §4 said "PM says go" | §4 corrected in this PR |
| X7 | Line cites drifted in slang-second-blocker, `ir.rs`, `structure_key.rs`, `convert.rs`, `contract.rs:245`, `pattern.rs:229` | open, cosmetic |

### 6.7 The IR as the full union of kernel languages (`docs/ir-union-design.md`, `docs/ir-union-capability-grid.md`)

CireSnave, 2026-10-24: *"I want the IR to cover the full union of the capabilities of every kernel language in common use today including, but not limited to, CUDA, CUTLASS, Slang, CuTe, Mojo, Triton, and ThunderKittens."* One tracking issue per increment; the lane that lands an increment closes its issue. Decisions D1-D4, D6-D8 are ruled (design doc section 7); **D5 (fp4/fp6 in `ElementKind`) is CireSnave's, board 290: nothing touching D5 starts.**

| ID | Work | Issue | Status |
|---|---|---|---|
| U0 | Replace idiom-lifting 1a, its "not transferable" table, and `groupshared-sizing`'s premise with the cooperation ruling; fix the stale claims in 3.22 docs; record the ownership split. Docs only | #111 | **done**, 0.22.19 |
| U1 | `Cast`/`Bitcast` with rounding+saturation; `Fma`; rounding/ftz/approx qualifiers; new `ElementKind` rows for fp4/fp6/e3m4 and MX scales (reverses the 2026-08-08 ruling, **D5**); packed/vector values; integer `Coord`/`Select`; wider index dtypes | #112 | in progress: `Fma` landed in 0.23.0, typed expressions with `Cast`/`Bitcast` in 0.24.0 (#112); `Qualified` rounding/ftz/approx qualifiers landed in 0.25.0 (`docs/ir-numerics-qualifiers.md`); `CoordInt`/`ConstInt`/integer arithmetic landed in 0.26.0 (`docs/ir-typed-expressions.md` section 8); rest open (packed values, wider index dtypes); the fp4/fp6/MX part waits on D5 |
| U2 | Kernel layer v0: `Let`, assignment, `For`/`While`/`If`/`Break`/`Return`, calls, effects model; lifting of today's `OpDef` into it (so the layers agree on every existing op) | #113 | open, after U1 |
| U3 | Execution scopes and ids; memory spaces and allocation; typed load/store; pointers; resources (textures/samplers/structs); `ProgramModel` | #114 | open, after U2 |
| U4 | Barriers, fences, mbarrier, async groups; full atomics with order/scope/old-value, on shared memory too | #115 | open, after U3 |
| U5 | Collectives as nodes; tuple-state reductions/scans with custom combine; arg-reductions; non-propagating NaN reduce | #116 | open, after U3, U4 |
| U6 | Layout value + algebra (hierarchical shape:stride, composition/complement/divide/product/coalesce/swizzle) **as pure vocabulary with a property-test suite**; `View` re-expressed over it | #117 | open, after U1 |
| U7 | MMA atoms (`mma.sync`, WGMMA, tcgen05/UMMA, sparse, block-scaled) and fragments; `AccumSpec` variants (TF32, 3xTF32, int32, f16, fp8 fast-accumulate); mixed-dtype contraction; accumulate-into-C | #118 | open, after U3, U6 |
| U8 | Copy atoms (cp.async, TMA family, ldmatrix); pipelines, roles, persistent schedulers | #119 | open, after U4, U6, U7 |
| U9 | EVT-style epilogue trees, grouped/ptr-array, split/stream-K, sparse, block-scaled GEMM; implicit-GEMM conv | #120 | open, after U7, U8 |
| U10 | `Access::Attention` incl. online-softmax state, masks, paged KV (block-table addressing over `ReadIndex`), backward | #121 | open, after U5, U7, U9 |
| U11 | Typed `Extern`/inline-asm with declared effects and a declared per-target body | #122 | open, after U3 |
| U12 | Peer/multicast scopes and multimem ops | #123 | open, last (D4) |


**Flight of fancy (CireSnave, board 282 item 81):** direct-to-PTX emitter: a PTX-capable lowering in the union IR is its prerequisite; flight of fancy, no schedule.
