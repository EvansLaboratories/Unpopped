# Re-score of `docs/ir-union-surveys/slang.md` at origin/main `78c7d38e` (Unpopped 0.47.3)

Slang. Status at `a634b09` (Yes/Partial/No/n/a): 17 / 19 / 22 / 3 = 61. At `78c7d38e`: 22 / 21 / 15 / 3 = 61.

Row numbering: the first re-score's numbering, reproduced exactly. It numbers the table rows of the raw survey but leaves out the five rows whose IR cell
is `No (...)` with a parenthesis or `--` (file lines 62 generics, 67 autodiff limits, 69 denormal, 75 ray tracing, 78 cp.async/TMA/cluster); that gives the 61
rows (3 n/a = lines 79-81, the `Slang-unique` rows). Those five rows are not counted here; the last section says what moved among them.

Examined: 41 rows (every row not Yes at `a634b09`: 19 Partial + 22 No). Skipped: 17 Yes rows and 3 n/a rows (skip class: already Yes, and nothing in the 89 commits
removes any of those features; I did not re-run them looking for a downward move) and the five unnumbered rows.

## Changed rows

| # | Capability | Status | Evidence at `78c7d38e` |
|---|---|---|---|
| 2 | Memory: Dynamic shared memory size / opaque scratch | Partial -> **Yes** | `Extent::Param` on a `Shared` `AllocDecl` (`exec.rs:Extent`); oracle runs it (`tests/kernel_exec.rs` ~l.267-281, `kernel_launch.rs` ~l.385); cpu-c spells it (`unpopped-cpu-c/tests/kernel_dynamic_shared_emit_agrees.rs`, U16); `KernelInterface::of` reports it as `DynamicShared` with the size parameter and the max-dynamic-size note (`kernel_interface.rs`, `tests/kernel_interface.rs`). The opt-in attribute call is the consumer's, not an IR matter |
| 3 | Memory: Structured/RW buffers (`StructuredBuffer`, `RWStructuredBuffer`) | Partial -> **Yes** | Scoring correction, not new work: `BufferRole::{Read,Write,ReadWrite}` on `ParamKind::Buffer` since 0.31.0 (`kernel.rs:BufferRole`), enforced by `check_kernel` (`WriteToReadOnly`, `ReadFromWriteOnly`; `tests/kernel_layer.rs`, `kernel_masked_access.rs`); atomics need `ReadWrite` (`tests/kernel_atomic_rmw.rs`). The raw claim "no read-only vs read-write qualifier" was false. `restrict` is an emitter-set `BufferGuarantee` in `KernelInterface`; coherence is row 11 |
| 18 | Wave: Reductions `WaveActiveSum/Product/Min/Max/BitAnd/BitOr/BitXor/CountBits/AllTrue/AnyTrue/AllEqual` | Partial -> **Yes** | Product: `CollectiveKind::ReduceWith { combine }` runs a user combine function in the oracle (`kernel_check/collective.rs:check_combine`; `tests/kernel_collective_combine.rs`: add, first, last, max-abs combines; no test uses a literal `Mul` combine, same path as `add`). CountBits = `Reduce(Sum)` of a 0/1 value, AllEqual = `Broadcast` + `Vote(All)` (`tests/kernel_collective.rs`). Judgment: product and CountBits are compositions, not first-class ops |
| 23 | Wave: Quad ops `QuadReadAcrossX/Y/Diagonal`, `QuadReadLaneAt` | No -> **Yes** | Correction of the `a634b09` score (shuffles were in since 0.42.3 and row 20 moved, this one was missed): `CollectiveKind::Shuffle { mode: Xor/Idx, operand }` over `ExecId::LaneId` arithmetic; `tests/kernel_collective.rs::a_shuffle_reads_another_lane_or_keeps_its_own` runs Xor 1, Xor 2 and Idx. AcrossX = Xor 1, AcrossY = Xor 2, Diagonal = Xor 3, LaneAt = Idx over `(lane & ~3) + k`. Assumes the linear lane-to-quad mapping of compute |
| 25 | Coop matrix: `linalg::CoopMat<T,Scope,M,N,Use>`: fill/Load/Store/MatMulAdd, Transpose, convertUse, ReduceRow/Column/2x2, MapElement, arithmetic; ... | No -> **Partial** | `Stmt::Mma { atom, a, b, c, d, extra }` over `Private` fragment arrays (`kernel.rs:Stmt::Mma`, `mma.rs:MmaAtom`, with 2:4 sparse and block-scale flags); oracle computes D = A.B + C through the atom's layouts: `tests/kernel_mma.rs` (f16, bf16, s8 PTX m16n8 atoms, wrap, trap, checker), `tests/mma_atoms.rs`, `tests/kernel_mma_flags.rs`. Missing: tile-level Load/Store/Fill/Transpose/Reduce/MapElement/convertUse as ops (fragments are per-lane arrays); no WMMA 16x16x16, e4m3/e5m2 or Metal 8x8 atom is built or tested (`MmaAtom::new` takes any shape, only three named constructors) |
| 26 | Coop matrix: Tensor addressing / clamp modes (`CoopMatClampMode`, `TensorLayout`) | No -> **Partial** | Per-element only: `Stmt::MaskedLoad`/`MaskedStore` with a fill value (`tests/kernel_masked_access.rs`) plus index remap give constant, NaN, replicate (clamp) and reflect (`tests/kernel_padding_modes.rs`). No tile-load operand carrying a layout and a clamp mode; `TensorLayout` semantics were UNVERIFIED in the raw survey and still are |
| 36 | Dtypes: Struct, array, tuple, `Optional<T>`, enum, user types as values | No -> **Partial** | Arrays: `Place::Alloc` over `AllocDecl` in `Private`/`Shared` (`tests/kernel_module_alloc.rs`, `kernel_launch.rs`). Tuples: `KernelFn::results` + `Stmt::CallTuple` (`tests/kernel_call_tuple.rs`) and `Stmt::CollectiveTuple` (`tests/kernel_collective_tuple.rs`). No struct, `Optional`, enum or user type as a value; arrays are by id, no array values |
| 37 | Dtypes: 128-bit ints / u4, MX scale formats | Partial -> **Yes** | Scoring correction: the raw Partial was only "no int128", which the row says Slang lacks too. `I4`/`U4` load/cast/dequant run in the oracle (`tests/kernel_subbyte_load.rs`, `kernel_dequant_gemm.rs`, `sub_byte_packing.rs`); MX element formats and `F8E8M0`/`F8E6M2` (`tests/mx_elements.rs`). `B1` not re-checked |
| 41 | Math: Multi-result intrinsics (`sincos`, `modf`, `frexp`, `InterlockedX` out-params) | No -> **Partial** | A function with several results: `Stmt::CallTuple`, `ExternDecl::results` as a tuple (`tests/kernel_call_tuple.rs`, `kernel_extern_run.rs`); `AtomicRmw`/`AtomicCas` `old` is the Interlocked out-param (`tests/kernel_atomic_fetch.rs`). `ScalarExpr` is still single-valued and there is no `frexp`/`modf`/`ldexp` node |
| 54 | Vulkan/SPIR-V: Inline SPIR-V (`spirv_asm`), `[require(spirv,...)]` capability atoms, `__target_switch`, `-capability` | No -> **Partial** | `Stmt::ExternCall` + `ExternDecl` with `TargetBody { target, BodyText::Asm / Symbol }`, one body per target, checked by `check_module` (`tests/kernel_extern_check.rs`: operand placeholders, empty template, one body per target). The oracle cannot run asm: it runs only the declaration's `reference` kernel (`tests/kernel_extern_run.rs`). Capability atoms live only as the emitter-set `KernelInterface::requires` / `compiler_args`, not on a node |
| 56 | Vulkan/SPIR-V: Shader clock (`clockARB`, `getRealtimeClock`) | No -> **Partial** | Only as an opaque extern: `ExternDecl::new("clock", [], [U32], Effects::opaque())` is accepted and held to its effects (`tests/kernel_extern_check.rs`, the divergent-clock-into-barrier refusal). No `ReadClock` leaf; with no reference kernel the oracle traps `Unsupported` |

## Examined, unchanged

- 5 Partial: fetch-add gives an append counter; no Append/Consume resource with a hidden counter.
- 7 Partial: scalar and buffer params only; struct params are ROADMAP U23, `ParamsBlock` is emitter metadata.
- 10 Partial: runtime strides only through scalar params and index math; no runtime-rank operand.
- 11 Partial: order and scope exist; no coherent/volatile load qualifiers (`Effects::volatile` is extern only).
- 19 Partial: inclusive `ScanWith` exists; exclusive Min/Max/Prod refused (`tests/kernel_collective.rs` asserts the refusal).
- 21 Partial: `VoteKind::Ballot` is a U32 mask, no `WaveMatch`, no wide (uint4) ballot.
- 29 Partial: every order and scope on RMW/CAS; no atomic load/store statements, no inc/dec-with-bound.
- 30 Partial: F16/Bf16/F32/F64 `Add`/`Exchange` run (`kernel_atomic_half.rs`); float `Max`/`Min` still refused by `check_atomic`.
- 33 Partial: types and casts run (`fp8_codecs.rs`, Mma f16/bf16); `Pack`/`BitcastPacked` are typed, never run end to end.
- 34 Partial: packed I8x4 exists but only type-checked (`packed_values.rs`); no `Dp4a` node, no run of a lane-wise dot.
- 35 Partial: only 2 or 4 lanes (no `float3`, ROADMAP U15 for wider), no `dot`/`cross`/matrix `mul`.
- 40 Partial: `ScalarExpr::Fma` exists; `frexp`/`ldexp`/`isnan`/`lerp`/`saturate`/`modf` still absent as nodes.
- 50 Partial: per-op `Qualified` round/ftz/approx only; no per-function mode, no NaN/contract assumption.
- 53 Partial: `AccumSpec::{Tf32,Tf32x3,Bf16x3,F16Accum,Int32,Int32Sat}` run (`contraction_accum_policy.rs`, `contraction_int32_accum.rs`) but no FP8 multiply test.
- 12 No: `Layout`/`Swizzle` text is a CuTe layout algebra, not std140/std430 struct layouts.
- 4 No: sub-byte `Load` is nibble indexing, not a typed load at a byte offset.
- 27 No: no matrix-vector node; only writable as scalar loops, so not counted.
- 38 No: no popcount/clz/ctz/bitreverse/bitfield nodes (`Shl/Shr/And/Or/Xor` only).
- 22 No: `Reduce` is whole-subgroup or whole-block; no cluster-size operand, none tested by xor butterflies.
- 47 No: `KernelInterface::spec_constants` names only block-dim/dynamic-shared and no emitter sets it; no IR `SpecConst` leaf.
- 8, 9 No: textures and bindless are ROADMAP U21, scheduled, not built.
- 6 No: `Place::Ptr` is ROADMAP U9c/U3, scheduled; `Place::{Peer,ClusterPeer,Multicast}` are not pointers.

Rows where I think the status at `a634b09` is WRONG:
- 23 (No, should have been Yes): Shuffle Xor/Idx existed at 0.42.3 and row 20 moved on the same evidence.
- 3 (Partial, should be Yes): `BufferRole` has existed since 0.31.0; the stated gap was untrue.
- 37 (Partial, should be Yes): the only gap named (int128) is not a Slang feature.
- 18 (Partial -> Yes now): "no product" assumed no user combine; `ReduceWith` (post-`a634b09`) covers it. A real change, not an error.
- 33, 34 (Partial, reason stale): "no pack/unpack nodes" is false (`Pack`/`Extract`/`BitcastPacked` since 0.27-0.28); the real gap is no run evidence. Left at Partial.
- 35 (Partial): fine, but its evidence does not mention that lanes are 2 or 4 only.

## Remaining No rows (15), by what they would take

- A resource or memory-model type: 4 (byte-offset typed load/store), 6 (pointer values, U9c), 8 textures/samplers and 9 bindless handles (U21), 12 (data layouts on struct operands, U23).
- A new collective form: 22 (clustered / multi-prefix with a cluster size; WaveMatch for 21 is the same family).
- A cooperative vector op: 27 (`CoopVec` matvec, outer-product accumulate).
- Integer bit nodes: 38 (popcount, clz/ctz, bit-reverse, bitfield extract/insert).
- Debug / error statements: 45 (`throws`/`try`, `discard`, `printf`) and 61 (`printf`): a `Debug`/trap statement.
- Link-time and pipeline-time constants: 46 (`ExternConst` leaf), 47 (`SpecConst` leaf in the IR).
- A derivative transform: 49 (autodiff `Transform::Fwd/Bwd`, needs a design; loops and locals now exist).
- A numerics mode: 52 (NaN/Inf preservation flag).
- Capability atoms on nodes: 57 (`SPV_KHR_cooperative_matrix`, `SPV_NV_cooperative_matrix2`, demote-to-helper).

## Unnumbered rows (not in the 61), what moved

- File line 78 (cp.async / TMA / mbarrier / cluster launch / DSMEM): No -> Partial. `CopyAsync`/`AsyncCommit`/`AsyncWait` (`tests/kernel_async_copy.rs`), `Scope::Cluster` barrier and cluster launch dims (`kernel_cluster.rs`), `Place::ClusterPeer` DSMEM (`kernel_cluster_peer.rs`). TMA, tensormap descriptors and `mbarrier` are absent (ROADMAP U8c, scheduled).
- File line 69 (denormal handling): `ScalarExpr::Qualified` `ftz` is per op (`tests/numerics_qualifiers.rs`), so Partial if it were counted; no per-dtype mode on the kernel.
- File lines 62, 67, 75: unchanged (monomorphic IR, no data, ray tracing out of scope).
