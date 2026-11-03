# Options note: MMA atoms, fragments and accumulation policy (ROADMAP U7)

Status: **proposal, nothing built.** Asked by the PM (2026-11-03: "U7 options note approved, docs first; then bring decisions to
me"). Read at `origin/main` `a634b09` (0.42.14). U7 was waiting on U3 (kernel layer, execution scopes, shared memory) and U6
(layout algebra); both are done. The re-score (`docs/ir-union-rescore.md`) puts the matrix-unit group at **37 rows still `No`**, the
second-largest gap after asynchronous data movement (71 rows, which needs U8's payload).

## What exists

- `Access::Contraction { axes, accum, epilogue }`: batched dense `D = epilogue(sum_k A.B)`, one contracted group, one dtype
  for every operand (`StructureKey::dtype` is "primary dtype, operand 0"; "mixed-dtype ops fold per-operand dtype in a follow-up").
  The oracle (`eval_contraction`) accumulates in `f64` in ascending `k` and rounds once at the store.
- `AccumSpec` has **one** variant, `WideFloat`. The vocabulary says `f32` GEMM is *declared* to route through TF32 tensor cores
  (`PrecisionGuarantee::math_precision`), but the IR does not carry it: "the routing is the consuming runtime's".
- A kernel layer with locals, loops, shared and `Private` arrays, barriers at four scopes, collectives (a rendezvous over a scope with
  full participation, `oracle/kernel_collective.rs`), and `Layout` values with composition, complement, divide, product and inverses
  (`unpopped-vocab/src/layout_algebra.rs`). The cpu-c serial C reference spells all of it except `Private` arrays.
- **KISS has no clause for any of it.** `spec/ops.md` has `matmul` as a contraction op; KISS-OPS-6.0-0004 makes any float contraction
  class *order-invariant/nondeterministic* (compared under a tolerance, "neither a canonical reduction order nor a canonical
  accumulator width is pinned"); KISS-OPS-6.17 owns a two-member `MathFidelity` `{bit-stable, reduced-mantissa-permitted}`. There is
  no atom, fragment, tensor-core or accumulate-into-C clause. Everything below is therefore an **Unpopped extension, not KISS**, to be
  proposed upstream when it lands (as KISS#768/#769 were).

## What the surveys ask for (the 37 rows, grouped)

1. **Accumulation policy of a whole contraction**: TF32 and 3xTF32, bf16x3/x6, f16 and int32 accumulators, fp8 fast-accumulate with a
   promotion interval, saturating int output, accumulate-into-C, `input_precision`, mixed A/B dtypes (CUDA 43-44, CUTLASS 71-76,
   Triton 15-17, CuTe 38-39, Mojo 31).
2. **Register-fragment atoms**: `mma.sync` m16n8k8/16/32, `wmma`, AMD MFMA/WMMA, Metal `simdgroup_matrix` (CUDA 36-37, CuTe 29-32, Mojo 25,
   Slang 25-27, ThunderKittens 15): a warp-collective multiply-accumulate on per-lane fragments whose lane layout is part of the atom.
3. **Descriptor-driven, asynchronous atoms**: WGMMA (warpgroup, shared-memory operand descriptors, commit/wait groups), `tcgen05`/UMMA
   (single-thread issue, tensor-memory accumulator, CTA pairs) (CUDA 40-41, CuTe 33-34, Mojo 27-30, ThunderKittens 16-18).
4. **Operand variants**: 2:4 structured sparsity with a metadata operand, block-scaled MX/NVFP4 with scale operands, complex
   (CUDA 38/42, CuTe 36-37, ThunderKittens 19-20).
5. **Fragment movement**: `ldmatrix`/`stmatrix` (CUDA 39, CuTe 42, Mojo 26); this is U8's (copy atoms).

Group 3 and the movement in group 5 need shared-memory descriptors, tensor memory, commit/wait groups and `mbarrier`, which are U8
and U4 S7. They cannot be built honestly before those exist, so they are staged last.

## Proposal: two layers, like everything else in this IR

**Layer 1, the plan layer (`Access::Contraction`): what the answer is.** Grow the node, not the kernel language.
- `AccumSpec` gains variants: `Tf32`, `Tf32x3`, `Bf16x3`, `F16`, `Int32` (wrap) / `Int32Sat`, `Fp8FastAccum { interval }`. Each is a
  *model*: how inputs are converted (TF32: keep 10 mantissa bits, bf16x3: split into three bf16 terms), what the accumulator type is, and
  the order the oracle uses. The oracle models the structure, not a particular chip: converted inputs, exact products, a sum in `f64`,
  rounded to the accumulator type **once per atom step** (a step is `k_atom` elements, ascending), so the bytes are deterministic.
- Float results are compared to emitters under a **declared bound** (`gamma_K . sum|a_i b_i|` in the accumulator's unit roundoff,
  the form already used for float sums in `kernel_parallel_reduction_float_sum_bound.rs`), not byte-exact: KISS-OPS-6.0-0004.
  **Integer** accumulation (`s8/u8/s4/b1` into `s32`) is exact, so it is compared byte for byte.
- Accumulate-into-C: a `C` operand read by the epilogue (`D = epilogue(A.B, C)`), which also lands CUTLASS's `alpha.AB + beta.C`.
- Mixed A/B dtypes need `StructureKey` to carry per-operand dtypes. That is a **key schema change** (`STRUCTURE_KEY_VERSION`) seen by every
  consumer (Baracuda, Fuel, Vulkane), so it is its own slice and needs your ruling (question 3).

**Layer 2, the kernel layer (`Stmt::Mma`): how a thread group computes it.** A collective, like `Stmt::Collective`: all lanes of the
atom's scope arrive, the oracle gathers the lanes' fragments into logical `M x K`, `K x N`, `M x N` matrices through the atom's layouts,
computes `D = A.B + C` under the atom's `AccumSpec`, and scatters `D` back to the lanes' fragment locals. Same rendezvous machinery as
collectives; same checker rule (in uniform control, at a scope, full participation).
- **Fragments** are per-thread `Private` arrays (they exist; the cpu-c reference must learn them, small).
- **The atom is data, not an enum**: `MmaAtom { m, n, k, a, b, acc: ElementKind, scope, thr_val: [Layout; 3] }`, with the thread-value
  layouts for A, B and C/D as U6 `Layout` values (this is exactly what U6 enables). NVIDIA `mma.sync` m16n8k16 (f16/bf16 into f32),
  m16n8k8 (tf32) and m16n8k32 (s8 into s32) ship as constructors; AMD MFMA/WMMA and Metal `simdgroup_matrix` are further data, so adding
  a vendor never touches `Stmt`.
- Checks: fragment arrays have `v` elements of the atom's dtype; the launch subgroup size equals the atom's scope size; operand
  dtypes match the atom.
- **Property test for every atom**: the layout maps `(lane, value)` to `(row, col)` bijectively over the whole fragment (a plain
  round-trip over the atom): a wrong lane layout is the likeliest defect in an atom definition and nothing else would notice it.

## Staging (each slice additive and held to the usual bar: red-first tests, a Rust reference, seeded mutants, a C differential)

| Slice | Content | Needs |
|---|---|---|
| S1a | `AccumSpec` variants, accumulate-into-C, saturating int output; oracle models + declared bounds; cpu-c reference | nothing new |
| S1b | per-operand dtypes in `StructureKey` (mixed-dtype contraction) | **ruling**: key version bump, breaking for consumers |
| S2 | `Stmt::Mma` + `MmaAtom` + warp-scope register-fragment atoms (3 NVIDIA atoms); cpu-c `Private` arrays | S1a |
| S3 | block-scaled (scale operands) and 2:4 sparse (metadata operand) as atom flags | S2 |
| S4 | WGMMA: a warpgroup scope, shared-memory operand descriptors, commit/wait | S2 + U4 S7 + U8 smem/async |
| S5 | `tcgen05`/UMMA: tensor-memory space, single-thread issue, CTA pairs | S4 + a TMEM space |

S1a, S2 and S3 address groups 1, 2 and 4 above; S4 and S5 address group 3 and are blocked on U8 and U4 S7, and say so. The number of survey rows each slice would flip is not computed here.

## Needs a ruling

1. **Layering**: plan-level accumulation policy first (S1a), then the kernel-level `Stmt::Mma` over data-driven atoms (S2)? (recommended: yes.)
2. **Float MMA contract**: a deterministic oracle model (conversion, exact products, `f64` sum, one rounding per atom step) with emitters held to a
   declared relative bound; integer MMA byte-exact. (recommended: yes.) The alternative is a looser "within a tolerance" with no stated model, which
   leaves nothing to test an emitter against.
3. **Mixed-dtype contraction** (S1b): do it as a `StructureKey` version bump now, or defer until a consumer asks? (recommended: defer;
   S1a and S2 do not need it, and a key change is the one thing in this plan that breaks other repos.)
4. **Atoms as data** (`MmaAtom` with U6 layouts) rather than an enum of named hardware atoms. (recommended: data.)
5. **Stop at S3 for now** and leave S4/S5 until U8 and U4 S7 exist. (recommended: yes.)
6. **KISS**: file the extension as an RFC when S1a lands (accumulation policy, accumulate-into-C), as for #768/#769? (recommended: yes.)
