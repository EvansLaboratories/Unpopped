# Re-score at 78c7d38e: CuTe / CuTe DSL (`docs/ir-union-surveys/cute.md`)

Status at a634b09 (Yes/Partial/No/n/a): **7 / 29 / 29 / 1** = 66. Status at 78c7d38e: **10 / 35 / 20 / 1** = 66. Changed rows: 9 (3 to Yes, 6 to Partial, none down).

Row numbers count table rows top to bottom (1a = 1-16, 1b = 17-28, 1c = 29-48, 1d = 49-61, 1e = 62-66). Paths are under `crates/`; `u` = `unpopped/src`, `ut` = `unpopped/tests`.

## Changed rows

| # | Capability | Status | Evidence at 78c7d38e |
|---|---|---|---|
| 25 | Partition: TV-layout (thread,value)->coord and `composition(tensor, tv)` then slice by thread | No -> **Partial** | The (thread,value)->coord layout exists as data: `u/mma.rs:MmaAtom` stores a `Layout` per operand, mapping `lane + lanes*value` to a logical id, checked as a bijection in `MmaAtom::new`; thread index is `ExecId::LaneId`/`ThreadIdx`. Tests `ut/mma_atoms.rs`, `ut/kernel_mma.rs`. No tensor value, so no `composition(tensor, tv)` and no slice by thread. |
| 29 | MMA: MMA_Atom + MMA_Traits (ValTypeA/B/C/D, Shape_MNK, ThrID, A/B/C layouts) | No -> **Yes** | `u/mma.rs:MmaAtom` + `AtomSpec` (name, `mnk`, `dtypes: (A, B, acc)`, `scope`, `lanes`) with TV `Layout` for A, B, C; PTX constructors `nvidia_m16n8k16_f16_f32`, `_bf16_f32`, `nvidia_m16n8k32_s8_s32`. `Stmt::Mma` in `u/kernel.rs`, checker `u/kernel_check/mma.rs`, oracle `u/oracle/kernel_mma.rs`. Tests: `ut/mma_atoms.rs` (fragment tables vs PTX), `ut/kernel_mma.rs` (D vs triple loop, wrap, refusals). |
| 31 | MMA: cute.gemm over mode patterns: (V),(M)x(N),(M,K)x(N,K),(V,M,K)x(V,N,K) | No -> **Partial** | `Stmt::Mma` is an atom-level `D = A.B + C` statement on `Private` fragments, run in a K loop in the K6 GEMM kernel (`ut/kernel_gemm_family.rs`, `ut/support/gemm.rs`) and in `ut/kernel_attention_mma.rs`. No `Gemm` over tensor values or mode patterns; no tensor value exists. |
| 35 | MMA: Runtime-settable MMA fields (negate A/B, accumulate flag, scale factors, sparsity metadata, disable-output-lane, collector op) | No -> **Partial** | Sparsity metadata and scale factors are runtime operands: `MmaAtom::with_sparse_24`, `with_block_scale`, `Stmt::Mma::extra`; `ut/kernel_mma_flags.rs`. Negate A/B, accumulate flag, disable-output-lane, collector op are not in the IR. |
| 37 | MMA: 2:4 structured sparse MMA with metadata operand `[a, e]` | No -> **Yes** | `u/mma.rs:MmaAtom::with_sparse_24` (compressed `M x K/2` A, `M x K/4` metadata, `ExtraRole::Metadata`); oracle `expand_sparse` in `u/oracle/kernel_mma.rs`. Tests in `ut/kernel_mma_flags.rs`: `a_2_4_sparse_atom_expands_a_through_the_metadata` (with a dense-read control), `a_metadata_entry_with_unordered_positions_traps`, `a_sparse_and_scaled_atom_composes_both`. Only toy atoms; no PTX `mma.sp` constructor (its lane tables are unchecked, per `u/mma.rs` docs). |
| 40 | Copy atom: Copy_Atom/CopyTraits (Src/Dst/Ref layouts), TiledCopy via make_tiled_copy, `copy`, `copy_if`, `autovec_copy`, axpby/fill/clear | No -> **Partial** | A copy statement now exists in one form: `Stmt::CopyAsync{dst, src, elems}` (G2S, 4/8/16 B) and `Stmt::MaskedLoad/MaskedStore` as a per-element `copy_if` (`ut/kernel_async_copy.rs`, `ut/kernel_masked_access.rs`). No `CopyAtom`/`TiledCopy`/thread-value layout for copies, no `Fill`/`Axpby`. |
| 41 | Copy atom: SM80 `cp.async` (G2S, 4/8/16 B, cache mode), commit_group/wait_group | No -> **Partial** | `u/kernel.rs:Stmt::{CopyAsync, AsyncCommit, AsyncWait}`; checker `u/kernel_check/async_copy.rs`; oracle `u/oracle/kernel_async.rs` (`KernelTrap::AsyncRace`, `AsyncUnwaited`); tests `ut/kernel_async_copy.rs`, `ut/kernel_pipeline_double_buffer.rs`. Held at Partial only for the "cache mode" item in the row (no `CacheHint`, row 21). |
| 45 | Copy atom: Bulk (non-tensor) copies G2S, S2G, S2S (cluster), byte-masked S2G, DSMEM store | No -> **Partial** | DSMEM store: `Stmt::Store` to `Place::ClusterPeer` (`u/kernel.rs`; `ut/kernel_cluster_peer.rs`, `ut/support/cluster_peer.rs`). G2S element copies via `CopyAsync`. No bulk (`cp.async.bulk`, mbarrier-completed) copy, no S2G bulk, no byte mask. |
| 54 | Sync: cp.async / bulk-async group commit and wait_group(N) | No -> **Yes** | `Stmt::AsyncCommit` and `Stmt::AsyncWait{pending}` count committed groups per thread (empty commit still makes a group); oracle `u/oracle/kernel_async.rs`; `ut/kernel_async_copy.rs`, `ut/kernel_pipeline_double_buffer.rs` (a too-small wait count reads a tile that has not landed, which the oracle traps). Covers the cp.async group; a TMA/bulk-async group has no producer (rows 43, 45). |

## Examined, unchanged (max 25)

- 3: layout shapes/strides stay `i64`; only `Extent::Param` is dynamic (allocation sizes).
- 6: `select/take/slice/flatten/tile_to_shape` exist; group/append/prepend/replace/dice/filter do not.
- 7: `Layout::coalesce` exists; by-mode coalesce absent.
- 10: `logical_divide` only; zipped/tiled/flat variants absent.
- 11: `logical_product` only; blocked/raked/zipped variants absent.
- 12: left/right inverse exist; `max_common_layout/vector` absent.
- 13: `col_major`, `tile_to_shape` only; `make_ordered_layout`, `recast_layout` absent.
- 15: `Swizzle` + `ComposedLayout` + canonical text (`layout_text_round_trip.rs`); no smem atoms, no kernel use.
- 18: `Place::ClusterPeer` adds dsmem addressing; still no Tmem or generic space, no tagged pointers.
- 19: static and `Extent::Param` shared allocs; no struct, no alignment field in `AllocDecl`.
- 20: `Private` arrays and typed Mma fragments; no vector/fragment value type or layout.
- 28: `ExecId` gained `DeviceRank/DeviceCount`; no cluster-rank or `smid` leaf (rank is hand-computed).
- 32: f16/bf16/s8 atoms and `AccumSpec::Tf32` exist; fp8 atom and tf32 atom absent.
- 36: scale operands run (toy, F32 scales); F4/F6 are storage-only, no MX atom or E8M0 test.
- 38: `AtomSpec::dtypes` has distinct A and B and the oracle is type-blind; no A != B test, so stays No.
- 47: `MaskedLoad/Store` give per-element predication; no predicate or identity tensors.
- 49: only `BarrierKind::Full`; no named barriers or masked warp sync.
- 50: shuffle/vote/ballot/reduce/scan/ReduceWith exist; no match, elect, uni.
- 51: cluster barrier and DSM exist; no split arrive/wait, no cluster-id leaf.
- 53: scopes now include `System`; no proxy/async/tensormap fences.
- 59: And/Or/Xor/Exchange/Cas, old value, shared, F16/Bf16 Add; float Max/Min (fmax/fmin) refused (`kernel_check.rs:check_atomic`).
- 61: no popc/clz/bfe/brev/mul_hi in `UnaryOp/BinaryOp` (`u/ir.rs`); packed forms exist (`Pack`/`Extract`).
- 62: cluster, devices, dynamic extent; `Device` barrier implies cooperative; no PDL, fallback cluster, occupancy hint.
- 48: I4/U4 loads, Cast, dequant GEMM (`ut/kernel_dequant_gemm.rs`); no sub-byte or fp4/fp6 MMA operand.

## Rows where status at a634b09 may be wrong

- 5 (Yes): `crd2idx/idx2crd` are vocab functions (`unpopped-vocab/src/layout_algebra.rs`), not IR nodes; the row's addition asks for integer index-expression nodes. Defensible only because `MmaAtom` runs `Layout::call` in the oracle; arguably Partial.
- 1, 2, 4 (Yes): the Layout is a vocab value; `AllocDecl` and `Place` carry no layout. Not wrong, but after this pass the only IR consumer of a `Layout` is `MmaAtom`; operand and buffer layouts are still absent.
- 7 (Partial): fine, but `Layout::coalesce` was already in tree at a634b09, so the earlier note "key does it, not an IR op" is stale.

## Remaining No rows (20), by what each would take

- Tensor values with a layout and memory space (17, 23, 24, 26, 27): a `Tensor{space, elem, layout}` SSA value, then `LocalTile/LocalPartition/PartitionMma/PartitionCopy/Retile` over it.
- TiledMma (30): an atom-layout x permutation-tiler value on top of `MmaAtom`.
- Cache and eviction hints on load/store (21): `CacheHint`/`MemOrder` attributes on `Load/Store`.
- Hopper/Blackwell MMA (33, 34): async warpgroup `MmaAtom` (scope other than `Subgroup`, smem descriptors, commit/wait group); tcgen05 plus a `Tmem` space and commit-to-mbarrier.
- Mixed A/B dtype (38): no new IR (the type allows it); needs a test with A != B.
- ldmatrix/stmatrix (42), TMA and descriptors (43), TMA basis strides (44), TMEM copies (46): ROADMAP U8b/U8c (scheduled, not built); 44 also needs a basis-element stride type in `Layout`.
- mbarrier with phase and tx-count (52): ROADMAP U4/U8c; no `MBarrier` object.
- Pipelines and roles (55, 56, 57, 58): `PipelineState`, producer/consumer variants, CLC and persistent schedulers, warp roles with register budgets; ROADMAP U8 S4 says recipes only, no IR node. Named barriers (row 49) block role synchronisation.
- Multi-kernel host graph (63): host-side graph over kernels; `Module` only holds device functions.

## Coverage

Examined 58 rows (every row not already Yes and not n/a); skipped 8: rows 1, 2, 4, 5, 8, 9, 60 (already Yes, only cross-checked for overstatement) and 65 (n/a, a build-tooling concern the 89 commits cannot touch). Of the 58, rows 14, 16, 22, 64 and 66 were checked only for a change in the touched files (`ir.rs`, `shape_expr.rs`, `view_layout.rs`: no layout, shape or view change since a634b09) and not individually re-derived. Method: read `kernel.rs`, `exec.rs`, `mma.rs`, `kernel_check/mma.rs`, `oracle/kernel_mma.rs`, `layout_algebra*`, and the named tests; no test was run (read-only, no cargo).
