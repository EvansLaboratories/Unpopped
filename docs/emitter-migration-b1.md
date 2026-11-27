<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
# B1 emitter migration note: what an emitter (Baracuda, Vulkane, Fuel, cpu-c) must handle

For: `baracuda-cuda-emit`/`-parse`/`-seam`, `vulkane-*`, Fuel's consumers, and our own `unpopped-convert` and `unpopped-cpu-c`.
Scope: the B1 breaking batch, one section per item as each lands. Authority for every behaviour is the doc comment on the item in `crates/unpopped/src/kernel.rs` and `exec.rs`.

## T2: opaque tiles (this PR)

| Item | Where | Kind |
|---|---|---|
| `Stmt::TileLoad { d, src, elements_per_row, origin }`, `TileMma { d, a, b, c }`, `TileFill { d, value }`, `TileStore { a, dst, elements_per_row, origin }` | `Stmt` | four new variants: an exhaustive `match` stops compiling |
| `MemorySpace::Tile` | `MemorySpace` (`non_exhaustive`) | new case: **a wildcard arm hides it**, so search your `MemorySpace` matches and state the decline |
| `AllocDecl::tile: Option<TileShape>` | `AllocDecl` (`non_exhaustive`) | new field, `serde(default)`; build with `AllocDecl::tile(..)` |
| `KernelError::Tile { alloc, why }` | `KernelError` (`non_exhaustive`) | new case |

An emitter that does not spell tiles **declines by name**: refuse a `MemorySpace::Tile` allocation and each of the four statements with an error that says "Tile" (cpu-c: `kernel_tile_declines.rs`). Never lower a tile statement to a per-lane fragment `Mma` or drop it. Meaning: a tile is held by the whole subgroup, no element is addressable, `TileMma` is `D = A.B + C` (exact products, ascending `k`, one rounding to the accumulator), `TileStore` is one writer, every tile statement is reached in uniform control. **No `transpose` option exists and none is to be added** (patent hold, US12405770B2). An emitter for Metal or WMMA that lowers tiles lowers `TileLoad`/`TileStore` to `simdgroup_load`/`store` or `wmma::load_matrix_sync`/`store_matrix_sync` and `TileMma` to `simdgroup_multiply_accumulate` or `mma_sync`, and compares with the oracle within a tolerance, never by bytes.
