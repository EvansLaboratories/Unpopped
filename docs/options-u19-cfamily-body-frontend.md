# Options note: the C-family kernel-body frontend (ROADMAP U19)

Status: **options, no code** (pm-kernels [TASK] 2026-11-21: first slice as an options doc, no code before an ack). Read at `origin/main`
`df1e5b14` (0.56.6). ROADMAP U19: CST to `KernelFn` (statement spine, indexing to `Place`, builtins to `ExecId`/`LaunchShape`, every
unsupported node a typed decline). Goal test (baracuda#356 P11): lift FlashInfer's paged-decode instantiation, re-emit, match
`paged_decode_vs_reference`.

**Prerequisites: both done.** `Place::Multicast` (#332) and `MulticastLoadReduce` (U12 S4, 0.45.0) are merged, so the `Stmt` and `Place`
vocabulary a CUDA body needs is in the tree except what a slice below names.

## What exists, and what does not

- `unpopped-convert` is a tree-sitter walker for **operator-shaped** kernels only: `lift_elementwise`, `lift_reduction`, `lift_scan`,
  `lift` produce an `OpDef`, never a `KernelFn`. `Frontend` carries a `parse: fn(&str) -> Option<Tree>`, so the grammar stays in
  `baracuda-cuda-parse` / `vulkane-slang-parse`; the crate bundles none. Its own header says the shared walker accepted 0/147 of Fuel's
  real Slang kernels, so nothing there is evidence a statement-level walker works.
- `KernelFn` (`kernel.rs`) has the statement vocabulary a body needs (`Let`, `Assign`, `If`, `For`, `Store`, atomics, barriers, `Call`,
  the U9c `Place::Raw`, struct params from U23). Nothing builds one from text. Kernel-layer lifts go `OpDef -> KernelFn` only.
- **Not measured yet:** how much of FlashInfer's paged decode is template metaprogramming rather than statements. That decides Q2.

## Questions and options

**Q1. Where does the walker live, and who owns the grammar?**
- **A (recommended):** a new module in `unpopped-convert` (it already depends on `tree-sitter` and reads `tree-sitter-cpp`-shaped CSTs),
  taking the existing `Frontend.parse`. One walker, per-language `Frontend` supplies builtin spellings. No grammar in Unpopped.
- B: a new crate `unpopped-kernel-lift`. Cleaner split from the operator lifts, one more crate to version and stamp.
- C: a CST trait (kind, field, children, text) so no crate depends on tree-sitter. Most reusable, and the one that costs a trait
  design before any statement lifts. Not worth it before a second consumer needs a non-tree-sitter parser.

**Q2. Who resolves templates and macros?** FlashInfer is C++ templates; a CST holds `template<...>` text, not instantiations.
- **A (recommended):** the consumer hands over a **concrete instantiation** (Baracuda runs the preprocessor and instantiates; the walker
  sees a plain `__global__` function). A template parameter left in the source is a typed decline `UnresolvedTemplate`. Keeps the
  walker a statement mapper, the size the ROADMAP row implies.
- B: the walker substitutes template arguments itself (a small evaluator over `constexpr` integers and types). Roughly doubles the
  size, and a wrong substitution is silent.
- Needs Baracuda's answer: can it produce the instantiated source for the goal test? **[ASK] to baracuda.**

**Q3. The decline contract.** `KernelLiftError::Unsupported { node_kind, span, reason }`, one variant per construct class (not one
per node kind), never a best-effort guess. A construct the IR can express but the walker does not is `NotYetLifted`; one the IR cannot
express is `OutsideIr` and files an IR row. Recommend this split: it turns every decline into either a walker slice or an IR backlog
row, with no third state. Every unsupported node declines the **whole kernel**, never a partial `KernelFn`.

**Q4. Slicing** (each its own commit; all additive, no `Stmt` change expected):
1. **Spine:** params to `Param`, `Let`/`Assign`/`If`/`For`/`While`/`Return`, scalar arithmetic and comparisons, `?:`, casts, buffer
   `a[i]` to `Place::Logical`, pointer-offset forms to `Place::Raw`. Goal: a hand-written saxpy and a bounds-checked copy.
2. **Builtins:** `threadIdx`/`blockIdx`/`blockDim`/`gridDim` to `ExecId`, `__syncthreads` and fences, `__shared__` to allocs,
   `LaunchShape` from `__launch_bounds__`.
3. **Atomics, intrinsics, warp ops:** through the existing `Stmt` forms; each unmapped intrinsic a typed decline naming it.
4. **Structs and vector types** (`float4`, U15 packed values, U23 struct params).
5. **Goal test:** the paged-decode instantiation; every decline it hits becomes a row.

**Q5. How each slice is verified.** Differential, not golden text: lift a source, run the **oracle** on the `KernelFn`, compare to the
same kernel written directly as `KernelFn` and to a Rust reference, on distinct-byte data; seeded mutants (applied-count asserted) on
the walker. A second leg compiles the source with `nvcc`/a C compiler where one is available and compares outputs (Baracuda's runner,
or cpu-c on the neutral subset). A green suite proves only the idioms in it, so each slice states the construct list it covers.

## Recommendation and what I need

Q1 A, Q2 A, Q3 as above, Q4 order as listed, Q5 as listed. **Rulings needed:** Q1 (PM), Q2 (Baracuda's yes or no on supplying
instantiated source). Nothing is built until pm-kernels acks. Size stays XL; slices 1 to 3 are M each.

— lane: unpopped
