# Capability grid: Unpopped's IR against the seven kernel languages

Companion to `docs/ir-union-design.md`. One row per capability, one column per language; the first column is
Unpopped's IR **as it is today**, so the distance to the union is visible at a glance.

Legend: ✅ supported (first-class or native), 🟡 partial (library-only, restricted, or only some variants),
❌ not supported, ❔ the survey does not say.

**How it was made, and its limits.** The Unpopped column is the verified baseline of design doc section 1
(`origin/main` `a575e66`). Each language column was filled by a read-only Sonnet agent from that language's
survey in `docs/ir-union-surveys/` (46 rows, one column each; the agents' per-cell notes are not kept here). The
surveys themselves have not been reviewed line by line, no cell was checked against the primary
documentation, and ❔ means "the survey is silent", **not** "the language lacks it". The agents shared one brief
and one row list, so their agreement is weaker evidence than independent studies. Where this grid and the
design doc disagree, the design doc wins. Row IDs here (A1..H6) are this grid's own and do not map one to
one onto the design doc's gap rows (3.1..3.22).

| ID | Capability | Unpopped IR | CUDA | CUTLASS | CuTe | Slang | Mojo | Triton | TK |
|---|---|---|---|---|---|---|---|---|---|
| A1 | Elementwise scalar expressions over tensor inputs (arith, math functions, select) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| A2 | Local variables / mutation / structured control flow (if, for, while, break, return) | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| A3 | Device function calls / user-defined functions / recursion-free call graph | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| A4 | Generics / templates / compile-time specialization (comptime, constexpr) | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| B1 | Thread/lane/warp/block/grid ids and dims (SIMT explicit) | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ |
| B2 | Block-level SPMD program ids (program_id style, no explicit threads) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ |
| B3 | Cluster / thread-block-cluster scope and distributed shared memory | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ | 🟡 | ✅ |
| B4 | Persistent kernels / tile schedulers (static, stream-K, cluster launch control) | ❌ | 🟡 | ✅ | ✅ | ❌ | 🟡 | ✅ | ✅ |
| C1 | Shared / workgroup memory (static + dynamic) | 🟡 | ✅ | ✅ | ✅ | ✅ | ✅ | 🟡 | ✅ |
| C2 | Register-level fragments / explicit register tiles | ❌ | 🟡 | ✅ | ✅ | 🟡 | ✅ | 🟡 | ✅ |
| C3 | Typed load/store, pointers, byte-offset access, bitcast | ❌ | ✅ | ✅ | ✅ | 🟡 | ✅ | ✅ | 🟡 |
| C4 | Masked/predicated load/store with fill value | ❌ | 🟡 | ✅ | ✅ | ❔ | 🟡 | ✅ | 🟡 |
| C5 | Gather/scatter (data-dependent indexing) | ✅ | ✅ | ✅ | 🟡 | ✅ | 🟡 | ✅ | 🟡 |
| C6 | Textures / samplers / images / resource handles | ❌ | ✅ | ❌ | ❌ | ✅ | ❔ | ❌ | ❌ |
| C7 | Structs / arrays / vectors / matrices as values | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | 🟡 | ✅ |
| D1 | Hierarchical layout algebra (shape:stride, composition, divide, product, coalesce) | 🟡 | ❌ | ✅ | ✅ | ❔ | ✅ | 🟡 | ❌ |
| D2 | Swizzle | ❌ | 🟡 | ✅ | ✅ | ❔ | ✅ | 🟡 | ✅ |
| D3 | Reductions (sum/max/min/arg-, row and tile) | ✅ | 🟡 | 🟡 | 🟡 | 🟡 | ✅ | ✅ | ✅ |
| D4 | Reductions/scans with custom combine function or tuple state | ❌ | 🟡 | 🟡 | 🟡 | ❔ | 🟡 | ✅ | 🟡 |
| D5 | Scans (prefix)  | ✅ | 🟡 | ❌ | ❔ | 🟡 | ✅ | ✅ | ❌ |
| D6 | Sort / top-k | 🟡 | 🟡 | 🟡 | ❔ | ❌ | ❔ | ✅ | ❔ |
| E1 | Block / named barriers | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | 🟡 | ✅ |
| E2 | mbarrier-style split arrive/wait with transaction counts | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ | 🟡 | ✅ |
| E3 | Memory fences / scoped memory model (acquire/release, scope) | ❌ | ✅ | ✅ | ✅ | 🟡 | ✅ | 🟡 | ✅ |
| E4 | Atomics (add, min/max, CAS, exchange, returned old value, on shared memory) | 🟡 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 🟡 |
| E5 | Warp/subgroup collectives (shuffle, vote, ballot, match, elect) | 🟡 | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ |
| E6 | Block-level collectives (block reduce / scan / broadcast / cooperative groups) | 🟡 | 🟡 | 🟡 | 🟡 | ❌ | ✅ | ✅ | ✅ |
| F1 | Tensor-core MMA atoms (mma.sync, WGMMA, tcgen05/UMMA) | ❌ | ✅ | ✅ | ✅ | 🟡 | ✅ | ✅ | ✅ |
| F2 | Mixed-dtype / block-scaled (MX, NVFP4) / 2:4 sparse MMA | 🟡 | ✅ | ✅ | ✅ | 🟡 | 🟡 | 🟡 | 🟡 |
| F3 | Accumulator-precision selection (TF32, 3xTF32, int32, fp16 accumulate, fp8 fast-accumulate) | 🟡 | ✅ | ✅ | 🟡 | 🟡 | 🟡 | ✅ | 🟡 |
| F4 | Async copy (cp.async, TMA tile/bulk/multicast/reduce/im2col, ldmatrix) | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ |
| F5 | Software pipelining: stages, producer/consumer warp specialization, setmaxnreg | ❌ | 🟡 | ✅ | ✅ | ❌ | 🟡 | ✅ | ✅ |
| F6 | GEMM epilogue fusion trees (EVT), grouped / ptr-array GEMM, split-K / stream-K | ❌ | ❌ | ✅ | 🟡 | ❌ | ❔ | 🟡 | ❌ |
| F7 | Implicit-GEMM convolution | 🟡 | ❌ | ✅ | 🟡 | ❌ | 🟡 | ❌ | ❌ |
| F8 | Attention as a first-class op (online softmax, masks, paged KV, backward) | ❌ | ❌ | ✅ | ❌ | ❌ | ❔ | ❌ | 🟡 |
| G1 | Narrow float dtypes: fp8 (e4m3/e5m2), fp4 e2m1, fp6, MX scale types | 🟡 | ✅ | ✅ | ✅ | 🟡 | ✅ | ✅ | ✅ |
| G2 | Packed / vector value types (half2, float4, 128-bit) | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | 🟡 | ✅ |
| G3 | Cast / convert with explicit rounding and saturation | ❌ | ✅ | ✅ | ✅ | ❔ | 🟡 | 🟡 | 🟡 |
| G4 | Fma + rounding-mode / ftz / approx-vs-exact numerics controls | ❌ | ✅ | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | ❔ |
| G5 | NaN/Inf-preserving semantics control | 🟡 | 🟡 | ❔ | ❔ | ❔ | ❔ | 🟡 | ❔ |
| H1 | Inline asm / extern device libraries / typed escape hatch | ❌ | ✅ | ✅ | 🟡 | ✅ | ✅ | ✅ | ✅ |
| H2 | Automatic differentiation | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ |
| H3 | RNG (Philox etc.), device assert, device print (side-effecting ops) | ❌ | ✅ | 🟡 | ❔ | 🟡 | ❔ | ✅ | ❔ |
| H4 | Autotuning configs (num_warps, num_stages, tile config) as schedule hints | 🟡 | 🟡 | ✅ | 🟡 | ❌ | 🟡 | ✅ | ❌ |
| H5 | Multi-GPU: peer pointers, multicast, multimem | ❌ | ✅ | 🟡 | 🟡 | ❌ | 🟡 | ❌ | ✅ |
| H6 | Emits to / parses from multiple targets (portability across vendors) | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ | 🟡 | 🟡 |
