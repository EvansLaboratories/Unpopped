# CUDA Tile IR reader spike (PM [TASK] 2026-11-05): parse and validate only, no lowering

Status: **spike report.** Read at NVIDIA/cuda-tile `0192668` (2026-11-03, aligned to CUDA 13.1, bytecode versions 13.1 to 13.4). Licence file
read back: `LICENSE.txt` sha `6aa3392`, Apache License 2.0 **with LLVM Exceptions**, as the brief says. Nothing was copied from the
repository; the throwaway probe `tools/spikes/cuda_tile_probe.py` (standard library only) reads it and was never allowed to execute
anything from it. No lowering, no patent screen (the brief puts both after this).

## Verdict, early

**Not a no-go: the open parts are consumable without NVIDIA's closed `tileiras`.** The dialect (TableGen), the bytecode format (documented
in the reader and writer sources) and the test inputs are all in the repository. Three corrections to the brief, because they change what
a lowering would have to be:

1. **The suite is not a lowering oracle.** Its 92 `.mlir` tests and 29 bytecode files check parsing, printing, verification, bytecode
   round trips, versioning and transform passes, driven by lit and FileCheck through `cuda-tile-opt` and `cuda-tile-translate`. Nothing in
   it executes a kernel or states what a kernel computes on hardware; the only backend that does is `tileiras`, which is closed. "A lowering
   that passes it is correct by NVIDIA's own definition" holds for *reading* Tile IR, not for lowering it.
2. **The dialect has no NVIDIA machine ops to translate away.** `mbarrier`, `wgmma`, `cp.async`, `tcgen05` and cluster barriers do not appear
   in `Ops.td`, `AttrDefs.td` or any test (0 hits each). TMA appears only in the hint names and text (`allow_tma`, the worker-warps hint) and in one diagnostic and one comment about view-based atomic reductions ("use 'tl_blk' or 'device' for TMA compatibility"). Those are what `tileiras` *lowers to*;
   the dialect is one level above them (tile-level `mmaf`, view loads and stores, token-ordered memory). So the hard part of a port is not
   un-NVIDIA-ing the IR but writing the compiler that maps tiles to threads, async copies and matrix units, which is the closed part.
3. **Running the suite as written needs a build this machine does not have.** `cuda-tile-opt` links MLIR at the pinned LLVM commit
   `d6a4b2ad0b90fad43565a25f1cbcbac6e6857dde`, with CMake (present), Ninja and a C++17 compiler (neither present here; WSL not tried).
   I did not run it. The suite's *inputs* were used as data instead (below).

## What the probe parsed and checked

Inventory read from the TableGen files: **105 operations** (92 first available in 13.1, 1 in 13.2, 7 in 13.3, 5 in 13.4), 35 type
definitions, 38 attribute definitions, 108 `PublicOpcode` assignments in `BytecodeOpcodes.td` (bytecode opcodes 0x0 to 0xFFF).

Bytecode (structural level, the part that is a page of rules): header `0x7F "TileIR" 0x00`, then major (u8), minor (u8), tag (u16);
sections `id | length | [alignment, 0xCB padding]` with ids 0 end, 1 string, 2 func, 3 debug, 4 constant, 5 type, 6 global, 7 producer;
the string section decoded. Over all 29 bytecode files of the suite:

| Result | Count | Files |
|---|---|---|
| Parsed, structure well formed | 25 | 13 encoding-stability goldens, the versioned inputs of 13.1 to 13.4, `legacy_i1_splat_0xff` and 3 payload-invalid files below |
| Rejected for the class of reason the suite expects | 4 of 4 | invalid magic, unsupported version (18.0.0, reader range 13.1 to 13.4), unknown section id 127, end section not last |
| Parsed here but the suite expects a rejection | 3 | `constant_rawdatasize_overflow`, `invalid_attribute_name`, `invalid_dense_map_value`: the fault is in the payload, so **not checked** without a full decoder |

Text corpus: the 92 `.mlir` tests use **96 of the 105 operations**; never used: `alloca`, `atomic_red_view_tko`, `gdc_launch_dependents_tko`,
`gdc_wait_tko`, `make_gather_scatter_view`, `memory_fence_alias_tko`, `pack`, `ptr_to_ptr`, `unpack` (they are tested in other
forms or only by the dialect verifier). Element types seen in `tile<...>`: i1/i8/i16/i32/i64, f16/bf16/f32/f64 and pointers, in shapes
from scalar to `4x8`.

A full decoder is a much larger job than the probe: the reference reader is 3,044 lines of C++ (`lib/Bytecode/Reader/BytecodeReader.cpp`);
the structural part above is about 70 lines of Python. A *text* reader is larger still (105 custom assembly formats), which is why the
bytecode, not the text, is the right thing to consume if this goes on.

One small spec inconsistency found while reading: the reader's format comment says alignment padding bytes are `0xCF`; the code and the
files use `0xCB`.

## What our IR would need to represent

Mapping of Tile IR concepts to the IR union (`docs/ir-union-design.md`); "expected" means by reading the descriptions, not by a per-op check.

| Tile IR | Our IR today | Gap |
|---|---|---|
| `tile<shape x elem>` value, shapes up to rank 3+, pointer tiles | scalars per thread (`ScalarExpr`), buffers by `Place`; packed values open (U1) | **No first-class tile value** in the kernel layer. Tile IR is a block-SPMD model where the compiler owns the threads: the landing place is the open `ProgramModel` / `BlockSpmd` lowering of U3, not a statement |
| ~60 element-wise ops (`addi`..`xori`, `sin`, `tanh`, `fma`, `ftof`, `cmpf`, `select`, shifts, `mulhii`, `fpowf`/`fpowi`, `rsqrt`) | `ScalarExpr` and `BinaryOp`/`UnaryOp`, `Cast`, `Fma`, `Select` | expected to map one to one per element; per-op coverage (`atan2`, `sinh`, `cosh`, `tan`, `mulhii`, `exp2`, `log2`, `fpowi`) not yet checked against the tables |
| `for`, `loop`, `if`, `break`, `continue`, `yield`, `return` | `Stmt::{For, While, If, Break, Continue, Return}` | SSA region results (`yield`) become assigned locals: a conversion, not a gap |
| `mmaf`, `mmai` (tile level: `MxK . KxN + MxN`, optional batch) | `Stmt::Mma` is warp-fragment level on an `MmaAtom`; plan-layer `Contraction` | **Level mismatch:** a tile `mma` has no lanes or fragments; the lowering that splits it into atoms is the closed compiler's work |
| `mmaf_scaled` (13.3) | block-scaled atoms (S3, 0.42.30) | shapes and scale layouts to compare |
| `reduce`, `scan` (variadic, combiner region) | `Stmt::Collective`, `ReduceOp`, scans; custom combine and tuple state are open (U5) | custom combiner regions |
| `load_view_tko` / `store_view_tko` over `partition_view`, `strided_view`, `gather_scatter_view` of a `tensor_view`; padding values; latency and `allow_tma` hints | `Place::Logical`, `View`, the layout algebra (U6); hints have no home | views map to layout algebra plus masks; **hints (`allow_tma`, `latency`, `occupancy`, `num_cta_in_cga`, `num_worker_warps_per_cta`) have no field** |
| tokens: `make_token`, `join_tokens`, `*_tko` ops | order and scope on atomics and fences, barriers (U4) | **Explicit token dependences are a different memory-ordering vocabulary**; a reader would have to decide token chains into barriers or ordering edges |
| `atomic_rmw_tko`, `atomic_cas_tko`, `atomic_red_view_tko`; `memory_ordering_semantics` weak/relaxed/acquire/release/acq_rel; `memory_scope` | `AtomicRmw`, `AtomicCas`, `MemoryOrder`, `Scope` | expected to map; view-based atomic reduction new; `memory_fence_alias_tko` (proxy alias fence) has no counterpart |
| `get_tile_block_id`, `get_num_tile_blocks` | `ExecId::BlockIdx`, grid extent | maps; no thread ids exist in Tile IR |
| `entry`, `global`, `get_global`, `alloca` | `KernelFn`, `AllocDecl`, `Module` | `global` and `alloca` need module-level allocation with initializers (Constant space has none today) |
| `int_to_ptr`, `ptr_to_int`, `ptr_to_ptr`, `offset`, pointer tiles | none (pointer values held: ROADMAP U9c, U3 `Place::Ptr`) | pointer values |
| `pack`, `unpack` (13.3) | `Bitcast`; packed values (U1, open) | packed values |
| `assert`, `print_tko`, `assume` | none | a trap statement and a print; `assume` is an annotation |
| `gdc_launch_dependents_tko`, `gdc_wait_tko` (programmatic dependent launch) | none | launch-level dependency between kernels; a different scope than a statement |
| element types incl. `f8E4M3FN`, `f8E5M2`, `f8E8M0FNU`, `f4E2M1FN`, `f8E5M3FNU`, tf32 | `ElementKind` (MX rows from 0.29.0; tf32 operand notion missing for atoms, U7) | `f8E5M3FNU` not checked against our rows |

Version scale: 92 of 105 operations are in 13.1, so a reader that accepts 13.1 and ignores the 13 newer ones still covers the core; the
bytecode carries its version, and the reader range is 13.1 to 13.4.

## Gaps that decide the next step

1. No first-class tile value in the kernel layer, and the block-SPMD landing place (U3 `ProgramModel`/`BlockSpmd`) is still open.
2. Tile-level `mma` and views need a thread-and-fragment mapping we would have to own; that mapping is `tileiras`.
3. Token ordering, the five optimization hints and `gdc_*` have no home.
4. No semantic oracle: a Tile program's meaning on hardware cannot be checked against anything open. Our own oracle plus a host loop
   (as in the U9 kernels) would have to play that role, with Tile programs written or hand-translated for it.

## Options for the PM

| # | Option | Cost | Recommendation |
|---|---|---|---|
| A | Park here. This note and the probe are the whole spike | none | **Yes, unless a consumer wants Tile as a source language.** Reason: items 1 and 2 above are prerequisites, and the suite cannot judge a lowering |
| B | A full bytecode decoder in Rust (types, constants, functions) into an in-memory Tile IR, checked against the 29 files and the `.mlir` twins of the goldens | a few focused PRs; no GPU, no tileiras | Only with a named consumer; it would also close the 3 "not checked" files |
| C | Build `cuda-tile-opt` at the pinned LLVM on a machine with a C++17 compiler and Ninja, run the suite for real | a long build, no code | If B is chosen: run the suite once for the record before trusting the probe |

Provenance rule: if a later reader is derived from NVIDIA's C++ reader (not merely from the format), it carries their Apache-2.0 WITH
LLVM-exception notice and a NOTICE entry; the spike derives from the format only. The patent screen on any lowering (tiled compilation,
TMA-style async orchestration, the wgmma model) comes after a go, not now.
