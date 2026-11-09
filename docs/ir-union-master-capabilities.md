# Master capability list: everything the seven kernel languages can do, against the Unpopped IR

Promoted from `docs/ir-union-capability-grid.md` (46 rows, A1..H6). **It does two jobs:** it lists what the IR should hold, and it is the test of whether KISS is wide enough. **Unpopped owns this list; KISS owns its verdict column** (`docs/union-coverage-master.md`, keyed by the row ids here). Row ids are stable: a new row gets a new id, none is renumbered.

Legend: ✅ supported as a first-class named construct, in the language or its official standard library, covering the whole row; 🟡 partial (a subset, fixed variants only, an experimental or vendor-extension path, or only in a separate library such as CUB); ❌ documented absent (an absence is a source search with a positive control, written beside the cell); ❔ **the primary documents are silent after a search**, never "not looked at". **schedule-only** = the row is scheduling or placement, not what is computed; KISS does not describe scheduling.

## Status

| Part | Rows | Cells checked against primary sources | ❔ |
|---|---|---|---|
| R5.7 rows (below) | 9 rows: 7 from the grid (F3 G3 G4 D4 C4 F4 F7) and 2 new (C4b F9) | 63 of 63 | 0 |
| Remaining 39 rows (second table) | 39 | 273 of 273, plus the Unpopped column 39 of 39 | 0 |
| New row I1 (norm family) | 1 | 7 of 7, plus Unpopped | 0 |

Every cell in the R5.7 table cites its primary source in `docs/ir-union-evidence/<language>-wave1.md` (version, URL or file path with line, quoted text or identifier). The cells of the second table carry the same evidence in the wave-2 files.

## Versions each column was read at

| Column | Source read | Version |
|---|---|---|
| Unpopped | `crates/unpopped/src` at `origin/main` 5c12a76b (R5.7 PRs #319 to #326 merged) | 5c12a76b |
| CUDA | PTX ISA 9.5; CUDA Math, Driver, nvcc, Programming Guide 13.5; CCCL (CUB) | fetched 2026-11-06; CCCL v3.5.0; cuBLAS 13.5 and cudnn-frontend (v1.32.2 for I1, commit ad4e0035 for F8) where a row names them |
| CUTLASS | `NVIDIA/cutlass` headers | v4.8.1 (tag object 2c85f497) |
| CuTe | `include/cute` (C++) and `python/CuTeDSL`, same repo | v4.8.1, commit 2c85f497547eb441f0fe80bfb0fb2e6f9add1719 |
| Slang | `shader-slang/slang` source, prelude, standard modules, docs | v2026.21, commit e73d2689 |
| Mojo | `modular/modular` source (stdlib, `max/mojo`, kernels); docs site not used | main @6b93ed4 (most rows); A-C rows of wave 2 at tag max/v26.7.0 = 527d069; nearest tag max/v26.7.0 |
| Triton | `triton-lang/triton` source at the tag (rendered docs not used) | v3.9.0, commit b7ddcf86 |
| ThunderKittens | `HazyResearch/ThunderKittens` `include/` | commit 67845f5f |

## R5.7 rows (verified)

Order and ids as agreed with the KISS lane. Marks are the reviewed ones; where a reading agent marked a cell differently and I changed it, the change is listed under the table.

| ID | Capability | schedule-only | Unpopped | CUDA | CUTLASS | CuTe | Slang | Mojo | Triton | TK | Old grid (CUDA..TK) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| F3 | Accumulator-precision selection (TF32, 3xTF32, int32/fp16 accumulate, fp8 fast-accumulate) | no | 🟡 | 🟡 | ✅ | 🟡 | 🟡 | 🟡 | ✅ | 🟡 | ✅ ✅ 🟡 🟡 🟡 ✅ 🟡 |
| G3 | Cast / convert with explicit rounding and saturation | no | ✅ | ✅ | ✅ | 🟡 | ❌ | 🟡 | 🟡 | 🟡 | ✅ ✅ ✅ ❔ 🟡 🟡 🟡 |
| G4 | Fma + rounding-mode / ftz / approx-vs-exact numerics controls | no | ✅ | ✅ | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | ❌ | ✅ 🟡 🟡 🟡 🟡 🟡 ❔ |
| D4 | Reductions / scans with a custom combine function or tuple state | no | ✅ | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | ✅ | 🟡 | 🟡 🟡 🟡 ❔ 🟡 ✅ 🟡 |
| C4 | Masked / predicated load and store with a user fill value | no | ✅ | 🟡 | 🟡 | 🟡 | 🟡 | ✅ | ✅ | 🟡 | 🟡 ✅ ✅ ❔ 🟡 ✅ 🟡 |
| C4b | Out-of-bounds / padding fill modes beyond one constant (zero, NaN, constant, clamp, reflect, replicate) | no | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | new row |
| F4 | Async copy (cp.async, TMA tile/bulk/multicast/reduce/im2col, ldmatrix) | partly | 🟡 | ✅ | ✅ | ✅ | ❌ | ✅ | 🟡 | 🟡 | ✅ ✅ ✅ ❌ ✅ ✅ ✅ |
| F7 | Implicit-GEMM convolution | no | 🟡 | 🟡 | ✅ | 🟡 | ❌ | ✅ | 🟡 | ❌ | ❌ ✅ 🟡 ❌ 🟡 ❌ ❌ |
| F9 | MMA operand count and kinds (A, B, C, D, accumulate flag, block-scale operands, 2:4 sparsity metadata) | no | ✅ | ✅ | ✅ | ✅ | 🟡 | 🟡 | 🟡 | 🟡 | new row |

Reviewer changes to agent marks: **CuTe C4** ✅→🟡 (fill is zero-only per op; arbitrary fill is `fill()` then a predicated copy, the same reason CUTLASS is 🟡); **Triton F4** ✅→🟡 (im2col and async copy are in experimental Gluon, not `tl.*`). **Changes against the old grid:** F3 CUDA ✅→🟡, CUTLASS ✅ (unchanged), Triton ✅; G3 CuTe ✅→🟡; G4 CUDA ✅, ThunderKittens ❔→❌, Slang/Mojo/CuTe/CUTLASS unchanged 🟡; C4 CUTLASS ✅→🟡, Mojo 🟡→✅, Slang ❔→🟡; D4 Slang ❔→🟡; F4 Triton ✅→🟡 (see above), ThunderKittens ✅→🟡; F7 CUDA ❌→🟡, Mojo 🟡→✅, Triton ❌→🟡. **Judgement calls an owner may revisit:** CUDA F3/F7 (🟡 because 3xTF32 and a conv operator are absent from PTX; ✅/❌ if the row meant only "choose the accumulator type" / "language-level conv"); Mojo C4/F7 (library code, not syntax).

### Per-row notes

- **F3** (Accumulator-precision selection (TF32, 3xTF32, int32/fp16 accumulate, fp8 fast-accumulate)): Unpopped: the accumulator is an `MmaAtom` field (f32/f16/int, and `AccumSpec::Int32`/`Int32Sat` for s8/u8 into s32 since #320), oracle sums exactly in f64 then rounds once; still no TF32 input mode, no 3xTF32, no fp8 fast-accumulate flag (the last has no documented model, held), so the cell stays 🟡. CUDA: PTX selects f16/f32/s32 but has no 3xTF32 or fp8 fast-accumulate flag (CUTLASS software techniques).
- **G3** (Cast / convert with explicit rounding and saturation): Unpopped: `ScalarExpr::Cast(CastSpec{to,round,sat})`, four rounding modes, Wrap/Clamp/ClampFinite (stochastic rounding not yet). Triton has `fp_downcast_rounding` rtne/rtz only and no saturation parameter at v3.9.0. CuTe C++ has no conversion API (NumericConverter is CUTLASS). Slang: no rounding control found.
- **G4** (Fma + rounding-mode / ftz / approx-vs-exact numerics controls): Unpopped: `ScalarExpr::Qualified{QualOp, Numerics{round,ftz,approx}}` over Add/Sub/Mul/Div/Fma/Sqrt/Rsqrt/Recip/Exp2/Log2. CUDA PTX carries rnd/ftz/sat per instruction. Triton ftz and fusion are per-kernel compile options. ThunderKittens has no selector; its generic `fma_AxBtC` is a separate mul then add.
- **D4** (Reductions / scans with a custom combine function or tuple state): Unpopped: `CollectiveKind::ReduceWith`/`ScanWith` (a module function as the combine, #321) and `Stmt::CallTuple`/`Stmt::CollectiveTuple` with `KernelFn::results` for a tuple state (#326), in the oracle and `unpopped-cpu-c`; tested with argmax `(value, index)` and Welford `(mean, m2, count)`. The author asserts associativity; there is no exclusive scan (compose it from an inclusive scan). Only Triton (`reduce`/`associative_scan` with `combine_fn`, tuple operands) is first-class for both. CUDA gets 🟡 because it lives in CUB, a separate library.
- **C4** (Masked / predicated load and store with a user fill value): Unpopped: `Stmt::MaskedLoad`/`MaskedStore` with a user fill value (#322), oracle and `unpopped-cpu-c`. Fill is hardware-zero only in CUDA (cp.async), CUTLASS (zfill, or register untouched), CuTe, ThunderKittens (hard-coded zeros); Mojo `masked_load(addr, mask, passthrough)` and Triton `tl.load(mask, other)` take any value; Slang only on CoopMat tensor-addressed loads (NV extension).
- **C4b** (Out-of-bounds / padding fill modes beyond one constant (zero, NaN, constant, clamp, reflect, replicate)): New row. No language has reflect/replicate; the closed sets are zero (all), NaN-request (TMA `OOB_FILL_NAN_REQUEST_ZERO_FMA`, Triton descriptor `padding_option` zero/nan). Triton removed block pointers and `boundary_check` at v3.9.0. Unpopped: zero, NaN, constant and -inf fill are a `Stmt::MaskedLoad`; replicate (clamp) and reflect are an index remap (Min, Max) in front of a load, pinned by `kernel_padding_modes` (#322). They are composition, not a named mode, so the cell stays 🟡; the `Access::Im2Col` lift still pads with zero (`pad` per axis).
- **F4** (Async copy (cp.async, TMA tile/bulk/multicast/reduce/im2col, ldmatrix)): Unpopped: kernel-layer async copy exists (`kernel_check/async_copy.rs`); no TMA im2col mode. Triton: TMA im2col only in experimental Gluon, `tl.*` descriptor loads lower to TMA implicitly. ThunderKittens: cp.async, TMA tile 4d/5d, multicast, reduce, bulk, ldmatrix, but 0 hits for im2col. The copy itself is data movement; where and when it is issued is schedule.
- **F7** (Implicit-GEMM convolution): Unpopped: `Access::Im2Col` lift plus kernel-layer MMA (A6 shows a GEMM; no fused conv op); the fprop, dgrad and wgrad family with stride, dilation and padding is written as kernels and held to the oracle and the C emitter by `kernel_conv_family` (#323), so the cell stays 🟡. CUDA: no conv construct, but PTX 9.5 has TMA im2col and tcgen05 "Convolution MMA" (`.ashift`). Mojo: shipped sm100 `conv2d` "implemented as implicit GEMM" (forward verified only). Triton: Gluon tutorial/example kernels only. ThunderKittens: only an FFT conv.
- **F9** (MMA operand count and kinds (A, B, C, D, accumulate flag, block-scale operands, 2:4 sparsity metadata)): New row. Unpopped: `MmaAtom::with_sparse_24` / `with_block_scale` add `extras` operands (`Stmt::Mma.extra`), accumulator of f32/f16/int. CUDA: operand list differs per family (mma, mma.sp adds e,f; block-scale adds scale-a/b-data; wgmma adds scale-d, imm scales/transposes; tcgen05 adds idesc and TMEM scale operands). Mojo/Triton/ThunderKittens/Slang: dense or no sparse metadata found.

### Pending rulings that touch rows

- **D3, D6 (stability, keepdim):** CireSnave 2026-11-07 (relayed by the KISS lane): keepdim and stability become real flags, default 1. Provisional IR needs: D3 a reduce whose output omits the reduced axes (or reduce plus a rank-reducing `Reshape`; all-axes drop = rank 0); D6 a sort marked unstable (any tie order). The argmax/argmin tie rule is awaiting a ruling. Recorded here; no mark changes until it is ruled.

## Remaining 39 rows (verified)

Same columns and order as above. Every cell was read against the primary sources named in the version table, by one reading agent per language and row group; its evidence (version, path or URL with line, quote, notes) is in `docs/ir-union-evidence/<language>-wave2{a,b,c}.md` (A-C rows, D-E rows, F-H rows), and for Unpopped in `unpopped-wave2{a,b}.md`. Seven cells that the first readers left unknown were re-searched: `resolved-unknowns.md`. The last column is the earlier survey-derived grid, kept so the movement is visible. These cells had one pass by a reading agent and my read of its report; they have not had the cell-by-cell reviewer pass the R5.7 rows had.

| ID | Capability | schedule-only | Unpopped | CUDA | CUTLASS | CuTe | Slang | Mojo | Triton | TK | Old (Unpopped; CUDA..TK) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A1 | Elementwise scalar expressions over tensor inputs (arith, math functions, select) | no | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅; ✅ ✅ ✅ ✅ ✅ ✅ ✅ |
| A2 | Local variables / mutation / structured control flow (if, for, while, break, return) | no | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 🟡 | ✅ | ❌; ✅ ✅ ✅ ✅ ✅ ✅ ✅ |
| A3 | Device function calls / user-defined functions / recursion-free call graph | no | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌; ✅ ✅ ✅ ✅ ✅ ✅ ✅ |
| A4 | Generics / templates / compile-time specialization (comptime, constexpr) | yes | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌; ✅ ✅ ✅ ✅ ✅ ✅ ✅ |
| B1 | Thread/lane/warp/block/grid ids and dims (SIMT explicit) | no | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ❌; ✅ ✅ ✅ ✅ ✅ ❌ ✅ |
| B2 | Block-level SPMD program ids (program_id style, no explicit threads) | no | 🟡 | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌; ❌ ❌ ❌ ❌ ❌ ✅ ❌ |
| B3 | Cluster / thread-block-cluster scope and distributed shared memory | partly | 🟡 | ✅ | ✅ | ✅ | ❌ | ✅ | 🟡 | ✅ | ❌; ✅ ✅ ✅ ❌ ✅ 🟡 ✅ |
| B4 | Persistent kernels / tile schedulers (static, stream-K, cluster launch control) | yes | 🟡 | 🟡 | ✅ | 🟡 | ❌ | 🟡 | ✅ | ✅ | ❌; 🟡 ✅ ✅ ❌ 🟡 ✅ ✅ |
| C1 | Shared / workgroup memory (static + dynamic) | no | ✅ | ✅ | ✅ | ✅ | 🟡 | ✅ | 🟡 | ✅ | 🟡; ✅ ✅ ✅ ✅ ✅ 🟡 ✅ |
| C2 | Register-level fragments / explicit register tiles | no | 🟡 | 🟡 | ✅ | ✅ | 🟡 | ✅ | 🟡 | ✅ | ❌; 🟡 ✅ ✅ 🟡 ✅ 🟡 ✅ |
| C3 | Typed load/store, pointers, byte-offset access, bitcast | no | 🟡 | ✅ | ✅ | ✅ | 🟡 | ✅ | ✅ | 🟡 | ❌; ✅ ✅ ✅ 🟡 ✅ ✅ 🟡 |
| C5 | Gather/scatter (data-dependent indexing) | no | ✅ | ✅ | ✅ | 🟡 | ✅ | ✅ | ✅ | 🟡 | ✅; ✅ ✅ 🟡 ✅ 🟡 ✅ 🟡 |
| C6 | Textures / samplers / images / resource handles | no | ❌ | ✅ | ❌ | 🟡 | ✅ | ❌ | 🟡 | ❌ | ❌; ✅ ❌ ❌ ✅ ❔ ❌ ❌ |
| C7 | Structs / arrays / vectors / matrices as values | no | 🟡 | ✅ | ✅ | ✅ | ✅ | ✅ | 🟡 | ✅ | ❌; ✅ ✅ ✅ ✅ ✅ 🟡 ✅ |
| D1 | Hierarchical layout algebra (shape:stride, composition, divide, product, coalesce) | no | 🟡 | ❌ | ✅ | ✅ | ❌ | ✅ | 🟡 | ❌ | 🟡; ❌ ✅ ✅ ❔ ✅ 🟡 ❌ |
| D2 | Swizzle | partly | 🟡 | 🟡 | ✅ | ✅ | ❌ | ✅ | 🟡 | ✅ | ❌; 🟡 ✅ ✅ ❔ ✅ 🟡 ✅ |
| D3 | Reductions (sum/max/min/arg-, row and tile) | no | ✅ | 🟡 | 🟡 | 🟡 | 🟡 | ✅ | ✅ | 🟡 | ✅; 🟡 🟡 🟡 🟡 ✅ ✅ ✅ |
| D5 | Scans (prefix)  | no | ✅ | 🟡 | ❌ | 🟡 | 🟡 | ✅ | ✅ | ❌ | ✅; 🟡 ❌ ❔ 🟡 ✅ ✅ ❌ |
| D6 | Sort / top-k | no | ✅ | 🟡 | 🟡 | ❌ | ❌ | ✅ | ✅ | ❌ | 🟡; 🟡 🟡 ❔ ❌ ❔ ✅ ❔ |
| E1 | Block / named barriers | no | 🟡 | ✅ | ✅ | ✅ | ✅ | ✅ | 🟡 | ✅ | ❌; ✅ ✅ ✅ ✅ ✅ 🟡 ✅ |
| E2 | mbarrier-style split arrive/wait with transaction counts | no | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ | 🟡 | ✅ | ❌; ✅ ✅ ✅ ❌ ✅ 🟡 ✅ |
| E3 | Memory fences / scoped memory model (acquire/release, scope) | no | ✅ | ✅ | 🟡 | ✅ | 🟡 | ✅ | 🟡 | 🟡 | ❌; ✅ ✅ ✅ 🟡 ✅ 🟡 ✅ |
| E4 | Atomics (add, min/max, CAS, exchange, returned old value, on shared memory) | no | ✅ | ✅ | 🟡 | ✅ | ✅ | 🟡 | ✅ | 🟡 | 🟡; ✅ ✅ ✅ ✅ ✅ ✅ 🟡 |
| E5 | Warp/subgroup collectives (shuffle, vote, ballot, match, elect) | no | 🟡 | ✅ | 🟡 | ✅ | ✅ | ✅ | ❌ | 🟡 | 🟡; ✅ ✅ ✅ ✅ ✅ ❌ ✅ |
| E6 | Block-level collectives (block reduce / scan / broadcast / cooperative groups) | no | ✅ | 🟡 | 🟡 | 🟡 | ❌ | ✅ | ✅ | 🟡 | 🟡; 🟡 🟡 🟡 ❌ ✅ ✅ ✅ |
| F1 | Tensor-core MMA atoms (mma.sync, WGMMA, tcgen05/UMMA) | no | 🟡 | ✅ | ✅ | ✅ | 🟡 | ✅ | ✅ | ✅ | ❌; ✅ ✅ ✅ 🟡 ✅ ✅ ✅ |
| F2 | Mixed-dtype / block-scaled (MX, NVFP4) / 2:4 sparse MMA | no | 🟡 | ✅ | ✅ | ✅ | 🟡 | 🟡 | 🟡 | 🟡 | 🟡; ✅ ✅ ✅ 🟡 🟡 🟡 🟡 |
| F5 | Software pipelining: stages, producer/consumer warp specialization, setmaxnreg | yes | 🟡 | 🟡 | ✅ | ✅ | ❌ | 🟡 | ✅ | ✅ | ❌; 🟡 ✅ ✅ ❌ 🟡 ✅ ✅ |
| F6 | GEMM epilogue fusion trees (EVT), grouped / ptr-array GEMM, split-K / stream-K | partly | 🟡 | 🟡 | ✅ | 🟡 | ❌ | 🟡 | 🟡 | ❌ | ❌; ❌ ✅ 🟡 ❌ ❔ 🟡 ❌ |
| F8 | Attention as a first-class op (online softmax, masks, paged KV, backward) | no | 🟡 | 🟡 | 🟡 | ❌ | ❌ | 🟡 | ❌ | 🟡 | ❌; ❌ ✅ ❌ ❌ ❔ ❌ 🟡 |
| G1 | Narrow float dtypes: fp8 (e4m3/e5m2), fp4 e2m1, fp6, MX scale types | no | 🟡 | ✅ | ✅ | ✅ | 🟡 | ✅ | ✅ | 🟡 | 🟡; ✅ ✅ ✅ 🟡 ✅ ✅ ✅ |
| G2 | Packed / vector value types (half2, float4, 128-bit) | no | 🟡 | ✅ | ✅ | ✅ | ✅ | ✅ | 🟡 | ✅ | ❌; ✅ ✅ ✅ ✅ ✅ 🟡 ✅ |
| G5 | NaN/Inf-preserving semantics control | no | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | ❌ | 🟡; 🟡 ❔ ❔ ❔ ❔ 🟡 ❔ |
| H1 | Inline asm / extern device libraries / typed escape hatch | no | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌; ✅ ✅ 🟡 ✅ ✅ ✅ ✅ |
| H2 | Automatic differentiation | no | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌; ❌ ❌ ❌ ✅ ❌ ❌ ❌ |
| H3 | RNG (Philox etc.), device assert, device print (side-effecting ops) | no | ❌ | 🟡 | 🟡 | 🟡 | 🟡 | ✅ | ✅ | 🟡 | ❌; ✅ 🟡 ❔ 🟡 ❔ ✅ ❔ |
| H4 | Autotuning configs (num_warps, num_stages, tile config) as schedule hints | yes | 🟡 | 🟡 | ✅ | 🟡 | ❌ | 🟡 | ✅ | ❌ | 🟡; 🟡 ✅ 🟡 ❌ 🟡 ✅ ❌ |
| H5 | Multi-GPU: peer pointers, multicast, multimem | partly | ❌ | ✅ | 🟡 | 🟡 | ❌ | 🟡 | 🟡 | ✅ | ❌; ✅ 🟡 🟡 ❌ 🟡 ❌ ✅ |
| H6 | Emits to / parses from multiple targets (portability across vendors) | no | 🟡 | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ | ✅; ❌ ❌ ❌ ✅ ✅ 🟡 🟡 |

Cells that differ from the old grid: 63 of 312 (39 rows; Unpopped and the seven languages).
**Unknown (❔) cells in this table: 0.** 

Judgement calls the readers flagged (a different reading moves the mark one step): CUDA F6 and F8 are 🟡 because the support is in separate NVIDIA libraries (cuBLAS grouped GEMM and split-K, cuDNN SDPA), the same rule that puts CUB in 🟡; CUTLASS and CuTe E3/E4/E5 are 🟡 (the library wraps only part of the primitive and the rest is raw CUDA inside kernels); Triton B4 ✅ (Blackwell cluster launch control only, no tile scheduler or stream-K), E2 🟡, D6 ✅; Slang C1 🟡 (no dynamic shared memory found); Unpopped A4 ❌ (generics are front-end by design; 🟡 is arguable), D1 🟡 (the layout algebra lives in `unpopped-vocab`, not as a kernel-IR leaf), B2 and C2 🟡 (by composition). Many ❌ cells are source searches, each with a positive control in its evidence file; few are a sentence in the docs saying "unsupported".

Stale text found while reading Unpopped (to fix separately): doc comments in `kernel.rs` on `Stmt::Call` (about lines 15-17 and 407-420) and on `AtomicCas` say the checker refuses it; `docs/options-u11-extern-and-inline-asm.md` says "nothing built".

## New row (verified)

| ID | Capability | schedule-only | Unpopped | CUDA | CUTLASS | CuTe | Slang | Mojo | Triton | TK |
|---|---|---|---|---|---|---|---|---|---|---|
| I1 | Normalisation family as an operation: softmax, log_softmax, rms_norm, layer_norm, over several trailing axes (PyTorch `normalized_shape`), affine gamma/beta | no | 🟡 | 🟡 | 🟡 | ❌ | ❌ | 🟡 | 🟡 | ❌ |

Evidence: `docs/ir-union-evidence/norm-family-I1.md` (all seven languages plus Unpopped). Findings: Unpopped has no kernel-IR norm op (softmax and rms_norm are composed from reduce and elementwise); its SKU vocabulary layer has `NormalizationKind` (RMSNorm, LayerNorm, GroupNorm, BatchNorm training-only, InstanceNorm) and `SoftmaxKind` (Softmax, LogSoftmax) with a **multi-axis bitmask for RMSNorm/LayerNorm, the normalized axes a suffix of the input shape** (`unpopped-vocab/src/ops.rs:663-694`); the gamma/beta shape is **not specified in this repo** (only `* gamma`, `* gamma + beta` in the docs; the plan types and launcher live in Baracuda). Softmax is "along a single axis". Of the seven languages only cuDNN (CUDA column, separate NVIDIA library) accepts more than one normalized axis, with scale and bias of input rank (full extent on the normalized axes, 1 elsewhere); CUTLASS, CuTe and ThunderKittens have only examples or host utilities (single axis, gamma and beta of shape `[n]`); Mojo has library ops with one comptime reduce axis and rank-1 gamma; Triton has `tl.softmax` (one `dim`) and a layer-norm tutorial that ignores `normalized_shape`; Slang has none.

## How this list is kept

- A cell changes only with a citation (version, URL or path, quote) in the language's evidence file. A language release does not silently change a mark: re-read it, bump the version in the table above, and say so in the PR.
- A `❔` means the primary documents were searched and are silent; the evidence file names what was searched.
- Bulk reading is done by Sonnet or Haiku agents; every cell that goes in was reviewed by the owner.

