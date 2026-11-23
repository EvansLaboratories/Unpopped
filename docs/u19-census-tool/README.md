# The U19 decline-census tool (scratch, not part of the workspace)

`docs/options-u19-census.md` was measured with this. It is **documentation of the method**, not a crate: its `Cargo.toml` is stored as
`Cargo.toml.scratch` so cargo does not pick it up. To reproduce: copy this directory to a scratch location, rename
`Cargo.toml.scratch` to `Cargo.toml` and each `*.rs.txt` to `src/*.rs` (stored as text so the repo's analysers do not read a throwaway tool as source), fix its `path =` lines to your checkouts (it depends on this repo's `unpopped*` crates, on
`baracuda-cuda-parse` for `parse_cuda`, and on `fishstringer-slang-parse` for `SLANG`), `git archive` each corpus from `origin/main` into a
scratch `corpus/` directory (never lift from a working tree), then:

    census <cuda|slang> <label> <dir> [path-substring] [--instantiate] [--args] [--includes] [--real]

It finds kernels by **text** (`__global__` for CUDA, `[numthreads]` for Slang), lifts each with `lift_kernel_in` under one generic context
(block 128, dynamic shared 49152 B, `__CUDACC_VER_MAJOR__` 13, `__CUDA_ARCH__` 800), and prints the first decline per kernel. `--instantiate`
is a HEURISTIC (template type parameters become `float`, non-type ones `128`, file-wide); it shows what is behind a template, it is not an
instantiation. **`--args`** is the per-kernel version of that heuristic made through the caller's API: each template parameter of the kernel is given to
`LiftContext::with_type` (`float`) or `with_const` (the literal default or `128`). **`--includes`** resolves each `#include "x"` of a file against the other
files of the corpus and hands the text to `LiftContext::with_include` (nearest path, `CENSUS_OLD_INCLUDES=1` restores the first-found resolution). The 141, 146,
157 and 166 of `docs/options-u19-census.md` and the figures below were measured with `--args --includes`; the stored tool used to be an older copy that
had neither flag (a reviewer's catch on #386), and is now the version that produced the figures below. `baracuda-cuda-parse@ce2911956` pins `unpopped-convert ~0.47`, so its `CUDA` constant cannot be passed to this walker; the tool
builds the same `Frontend` from its exported `parse_cuda` and a copy of its (private) residue list.

**`--real`** (a later mode of the same tool; `real.rs.txt` is its file, and `main.rs` `include!`s it) reads each template kernel's REAL instantiations from the corpus
(a launch-site call, an explicit specialisation, a macro use that gives arguments) and lifts each; it prints, per kernel, ALL / MIXED / NONE (every, some or no real
instantiation lifts), the count of real instantiations that lift, and the kernels for which no instantiation can be found (`no instantiation`). `RAWINST` lines on stderr
(`CENSUS_RAW=1`) give the raw first decline of each failing instantiation. At the head of #386 there is no `LiftContext::with_struct_layout` (U38d lands later), so the
stored `with_gguf_layouts` is a stub that adds none.

## What this tool measured at the head of #386 (2026-11-23), with its control

Same tool (the files in this directory), same corpus (a `git archive` of Baracuda `origin/main` taken 2026-11-21 evening and not re-archived since: 564 files scanned, 94
with a kernel, 535 kernels of which 408 are templates and 11 prototypes only; its `baracuda_loss.cuh` equals Baracuda main at `f7fd28d74`), `baracuda-cuda-parse` as of Baracuda
`ce2911956`. Two instruments, **not comparable with each other**:

* **guess mode** (`--args --includes`): every template parameter of a kernel is set to `float` (type) or `128`/its default (value), whatever the real call sites use. A kernel
  counts as lifted when it lifts with those guesses, so this is an **upper bound**: the figures 141, 146, 157, 166 and 253 quoted so far (the changelog, the census note, the PR
  description) are all guess-mode figures.
* **real mode** (`--real --includes`): each kernel is lifted at its real instantiations. This is the instrument that says what lifts in the corpus as it is built.

| walker at | guess mode: kernels lifted (of them templates) | real mode: kernels lifted (templates) | real mode: ALL / MIXED / NONE of the 433 kernels with an instantiation | real mode: instantiations lifted | no instantiation found |
|---|---|---|---|---|---|
| `6ba67c7c` (head of #386; no `.rs` file differs from `1a55b0e6`: the stamp commit moves changelog fragments and bumps version strings) | **253 of 535** (228) | **201 of 535** (176) | 130 / 71 / 232 | 748 of 2014 (37.1%) | 102 |
| `ebf5342e` (base of the last six slices; the control, same tool, flags and corpus) | 205 of 535 (194) | 163 of 535 (152) | 75 / 88 / 270 | 472 of 2014 (23.4%) | 102 |

The control differs from the head with the tool, flags and corpus fixed, in both instruments, so the differences are the walker's. The figures the PR description used to carry
for the control (205 / 240 / 247) came from earlier copies of the tool and a second include-resolution method and are withdrawn; 205 is now the control's guess-mode figure,
measured with this directory. Reading a headline: it counts kernels or instantiations whose lift returned `Ok` (the checker included); it is not a count of kernels that are correct on a
device. A run of this tool on `lane/u19-walls-2` (a later branch, which adds struct layouts) is a different tree and is not comparable with these rows.
