# Design note: a tile-level MMA over an opaque tile (Apple `simdgroup_matrix`; Mojo 71 / 72; Q1 evidence)

Status: **design only; nothing is built, and nothing will be until CireSnave rules Q1 (tensor/tile)** (pm-kernels, 2026-11-25: the statement is a tile value type, so it is Q1-covered). Written 2026-11-25 by the Unpopped lane on pm-kernels' task, as the evidence for the Q1 brief and the answer to Vulkane's `[ASK]` (row 8o, the Metal emitter). Docs only. No legal advice.

## 1. The problem

`Stmt::Mma` (`crates/unpopped/src/kernel.rs`, "A warp-level matrix multiply-accumulate (ROADMAP U7 S2)") is a **per-lane** statement: every lane of the atom's scope supplies `Private` fragments `a`, `b`, `c` and receives `d`, and the atom (`MmaAtom`, `crates/unpopped/src/mma.rs`) carries the **data layouts** that say which element of the logical matrix each lane's fragment slot holds. That is the shape of PTX `mma.sync`, whose fragment layout NVIDIA fixes.

Metal does not fix one. Vulkane (the Metal emitter lane) reports, from Apple's *Metal Shading Language Specification* v4.1 (2026-06-04), section 2.4 p38: **"The mapping of matrix elements to threads in the SIMD-group is unspecified."** and, section 6.8 pp212-213, the only matrix operations: `simdgroup_load(d, src, elements_per_row, matrix_origin, transpose_matrix)`, `simdgroup_store(a, dst, ...)`, `simdgroup_multiply_accumulate(d, a, b, c)` ("Returns d = a * b + c"), on `simdgroup_matrix<T,Cols,Rows>` with `T` = `half`, `bfloat` (MSL 3.1+) or `float` and `Cols = Rows = 8` (section 2.4), under uniform control flow. *These quotations are Vulkane's, of Apple's public PDF; this lane has not read that PDF, and records them as relayed.* So there is no honest Apple fragment layout to put in an `MmaAtom`, and an `Mma` flag would assert a layout nobody documents.

Mojo rows 71 and 72 (`docs/ir-union-evidence/rescore-2026-11-25/mojo.md`) name the Apple 8x8 simdgroup MMA atom ("shipped AMD/Apple MMA atoms and emitters").

## 2. The statements

Three statements and one allocation kind, over an **opaque tile** (a value with a shape and an element type and **no per-lane mapping**):

```text
Allocation space  MemorySpace::Tile   // one 8x8 (or m x n) tile per subgroup; no element is addressable, only the tile statements reach it
Stmt::TileLoad  { d: u32, src: Place, elements_per_row: ScalarExpr, origin: [ScalarExpr; 2], transpose: bool }
Stmt::TileMma   { d: u32, a: u32, b: u32, c: u32 }          // D = A.B + C on tiles; d may be c
Stmt::TileStore { a: u32, dst: Place, elements_per_row: ScalarExpr, origin: [ScalarExpr; 2], transpose: bool }
```

The tile's element type and shape live on its `AllocDecl` (`TileShape { m, n, k, elem, acc }`); Apple's is 8 x 8 x 8 with `half`/`bfloat`/`float`. `src`/`dst` are `Shared` or `Global` places (MSL's `threadgroup` and `device` address spaces). All three are rendezvous statements reached in uniform control by the whole subgroup (a `TileMma` on divergent lanes is the same kind of error as a divergent `Mma`). `origin` and `elements_per_row` carry MSL's `matrix_origin` and `elements_per_row` (default `Cols`); `transpose` carries `transpose_matrix`.

**What the oracle computes.** The tile is a small matrix held once per subgroup, never per lane. `TileLoad` reads `m x n` elements from `src` (row `r`, column `c` at `src + (origin.y + r) * elements_per_row + origin.x + c`, or the transpose) as ordinary logged loads **attributed to the subgroup's first lane** (so the race detector sees one reader, not 32; a barrier orders it like any other); `TileMma` forms `D = A.B + C` with exact products summed in `f64` and rounded once to the accumulator type, exactly as `Mma`'s oracle does; `TileStore` writes the tile back as logged stores by the first lane. No thread-to-element mapping appears anywhere, so the CPU oracle runs it and a device may use whichever distribution it likes: the contract is the *result tile*, compared within the `Mma` tolerance for floats.

**Differences from `Mma`.** `Mma`: per-lane `Private` fragments, layouts in the atom, the program can read and write individual fragment slots (the epilogue code that scales or converts `D` lane by lane). `TileMma`: no fragment slots, the program cannot touch an element between `TileLoad` and `TileStore` (Metal offers no way to either: `simdgroup_matrix` has no documented element access). Elementwise work on an Apple tile must go through `Shared` (store, `Barrier`, load), which is what Metal code does.

## 3. Does CUDA's WMMA share it?

CUDA's C++ `nvcuda::wmma` API (`fragment`, `load_matrix_sync`, `mma_sync`, `store_matrix_sync`) is the same shape of API: an opaque fragment object loaded from and stored to memory with a leading dimension. As far as this lane knows, the CUDA C++ Programming Guide says the mapping of fragment elements to threads is unspecified for it as well. **That is from memory and unread here: the union rows "wmma shapes" (`docs/ir-union-rescore.md` group 3) need that section read before this note claims it.** If it holds, one tile family (`TileShape` with `m`, `n`, `k`, the element and accumulator types) carries both Apple's 8x8x8 and WMMA's 16x16x16 and its `m32n8k16` / `m8n32k16` shapes, and the per-lane `Mma` stays the `mma.sync` form. ThunderKittens' register tiles and CuTe's `Tensor`/`TiledMma` are different: they carry a `Layout` and per-thread partitions, so they are the larger part of Q1 and are not what this note proposes.

## 4. What each Q1 option needs (the evidence for the brief)

Q1 (`docs/ir-union-rulings.md`): **A** a tensor/tile value in the IR; **B** recipes over `Alloc` + `Layout` with a scoring rule; **C** a layout on allocations plus partition vocabulary (the recommended C, then A only for rows C cannot reach).

| Option | Does it carry the opaque tile statements? | Cost |
|---|---|---|
| **A** tensor/tile value | **Yes, naturally**: the Apple tile is a tile value whose layout is "opaque"; `TileLoad`/`TileMma`/`TileStore` are the tile-granular moves and `TiledMma` of A, with the layout leaf left opaque. | none beyond A (the three statements become instances of A's general ops) |
| **B** recipes | **No**: a recipe over `Alloc` + `Layout` has no object that exists only inside registers with no addressable element; the tile cannot be expressed as an allocation with a layout. | it would need an exception: a fixed "opaque tile" statement family outside the recipe scoring |
| **C** layout on allocations + partition | **No**: C describes where elements live; the opaque tile has no elements to place. | same exception as B: the three statements and `MemorySpace::Tile` as a stand-alone family |

**So the statement family is not common to every Q1 option**: under A it is a case of the general design; under B or C it is a small, self-contained exception (three statements, one space). **It needs A, or an explicit exception under B/C; it cannot be reached from a layout vocabulary alone.** Building the exception first would not foreclose A (the three statements are what A's general ops reduce to for an opaque layout), but it would make the Q1 answer matter less for Metal and leave the CUDA WMMA and ThunderKittens question as it is, which is why pm-kernels held the build for the ruling.

## 5. Cost and breakage

New `Stmt` variants (3) and a new `MemorySpace` variant: **BREAKING** for every exhaustive match (`unpopped-convert`, `unpopped-cpu-c`, the Baracuda and Vulkane emitter crates); batch with the U8c lineage (Q14 A). Slices, red-first: (1) `MemorySpace::Tile` + `TileShape` + checker (uniform control, subgroup of 32, element types, `Shared`/`Global` places); (2) oracle for the three statements with the differential against a hand-written product; (3) the serial C (the CPU reference can run it: a tile is a small array); (4) the lift from the C frontend (`simdgroup_load` and friends are Metal source, which the C-family walker does not read today: a Vulkane/Slang-side matter).

## 6. Patent line

None of the 38 PATENT rows of the union re-score names Apple or Metal simdgroup matrices, and `docs/PATENT-SWEEP.md` records only Apple's unified-IR and language-library families (hits 1, 13, 14, "outside as read": graphics operations in the IR, shaders in the binary, a pre-compiled library), none about matrix operations. No search was run for this note: it is a design, not a build. Before building, a patent sub-lane search for "simdgroup matrix multiply accumulate" and "tile load store opaque matrix fragment" is read-only work and is proposed as the first slice. Read-only; never "clear".

— lane: unpopped
