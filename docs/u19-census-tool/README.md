# The U19 decline-census tool (scratch, not part of the workspace)

`docs/options-u19-census.md` was measured with this. It is **documentation of the method**, not a crate: its `Cargo.toml` is stored as
`Cargo.toml.scratch` so cargo does not pick it up. To reproduce: copy this directory to a scratch location, rename
`Cargo.toml.scratch` to `Cargo.toml`, fix its `path =` lines to your checkouts (it depends on this repo's `unpopped*` crates, on
`baracuda-cuda-parse` for `parse_cuda`, and on `fishstringer-slang-parse` for `SLANG`), `git archive` each corpus from `origin/main` into a
scratch `corpus/` directory (never lift from a working tree), then:

    census <cuda|slang> <label> <dir> [path-substring] [--instantiate]

It finds kernels by **text** (`__global__` for CUDA, `[numthreads]` for Slang), lifts each with `lift_kernel_in` under one generic context
(block 128, dynamic shared 49152 B, `__CUDACC_VER_MAJOR__` 13, `__CUDA_ARCH__` 800), and prints the first decline per kernel. `--instantiate`
is a HEURISTIC (template type parameters become `float`, non-type ones `128`, file-wide); it shows what is behind a template, it is not an
instantiation. `baracuda-cuda-parse@ce2911956` pins `unpopped-convert ~0.47`, so its `CUDA` constant cannot be passed to this walker; the tool
builds the same `Frontend` from its exported `parse_cuda` and a copy of its (private) residue list.
