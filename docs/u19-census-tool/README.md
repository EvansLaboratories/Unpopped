# The U19 decline-census tool (scratch, not part of the workspace)

`docs/options-u19-census.md` was measured with this. It is **documentation of the method**, not a crate: its `Cargo.toml` is stored as
`Cargo.toml.scratch` so cargo does not pick it up. To reproduce: copy this directory to a scratch location, rename
`Cargo.toml.scratch` to `Cargo.toml` and each `*.rs.txt` to `src/*.rs` (stored as text so the repo's analysers do not read a throwaway tool as source), fix its `path =` lines to your checkouts (it depends on this repo's `unpopped*` crates, on
`baracuda-cuda-parse` for `parse_cuda`, and on `fishstringer-slang-parse` for `SLANG`), `git archive` each corpus from `origin/main` into a
scratch `corpus/` directory (never lift from a working tree), then:

    census <cuda|slang> <label> <dir> [path-substring] [--instantiate] [--args] [--includes]

It finds kernels by **text** (`__global__` for CUDA, `[numthreads]` for Slang), lifts each with `lift_kernel_in` under one generic context
(block 128, dynamic shared 49152 B, `__CUDACC_VER_MAJOR__` 13, `__CUDA_ARCH__` 800), and prints the first decline per kernel. `--instantiate`
is a HEURISTIC (template type parameters become `float`, non-type ones `128`, file-wide); it shows what is behind a template, it is not an
instantiation. **`--args`** is the per-kernel version of that heuristic made through the caller's API: each template parameter of the kernel is given to
`LiftContext::with_type` (`float`) or `with_const` (the literal default or `128`). **`--includes`** resolves each `#include "x"` of a file against the other
files of the corpus and hands the text to `LiftContext::with_include` (nearest path, `CENSUS_OLD_INCLUDES=1` restores the first-found resolution). The 141, 146,
157 and 166 of `docs/options-u19-census.md` and the figures below were measured with `--args --includes`; the stored tool used to be an older copy that
had neither flag (a reviewer's catch on #386), and is now the version that produced the figures below. `baracuda-cuda-parse@ce2911956` pins `unpopped-convert ~0.47`, so its `CUDA` constant cannot be passed to this walker; the tool
builds the same `Frontend` from its exported `parse_cuda` and a copy of its (private) residue list.

## What this tool measured at the head of #386 (2026-11-23), with its control

Same tool (the files in this directory), same flags `--args --includes`, same corpus (a `git archive` of Baracuda `origin/main` taken 2026-11-21 evening and not
re-archived since: 564 files scanned, 94 with a kernel, 535 kernels of which 408 are templates and 11 prototypes only; its `baracuda_loss.cuh` equals Baracuda main at
`f7fd28d74`), `baracuda-cuda-parse` as of Baracuda `ce2911956`, run as `census cuda <label> <corpus>/baracuda --args --includes`:

| walker at | LIFTED (`lift_kernel_in` returned `Ok`, the checker included) | of which template kernels |
|---|---|---|
| `6ba67c7c` (the head of #386, no `.rs` file differs from `1a55b0e6`: the stamp commit moves the changelog fragments and bumps version strings) | **253 of 535** | 228 |
| `ebf5342e` (the base of the last six slices; the control) | 205 of 535 | 194 |

The control differs from the head by +48 with the tool, the flags and the corpus held fixed, so the difference is the walker's. It replaces the figures the PR
description used to carry (205 / 240 / 247, which came from earlier copies of the tool and a second include-resolution method and are not reproducible from
this directory). Reading the headline: it counts kernels whose lift returned `Ok`; it is not a count of kernels that are correct on a device. A run of
the same census on `lane/u19-walls-2` (a later branch) uses `--real`, which is a different instrument (real instantiations from call sites) and is not comparable
with these figures.
