kind: CUDA Tile IR reader spike (a report and a throwaway probe; docs and tools only)
---
`docs/cuda-tile-ir-reader-spike.md` and `tools/spikes/cuda_tile_probe.py`: NVIDIA/cuda-tile read at `0192668`, parse and validate only,
no lowering. 105 ops, 29 bytecode files (25 parsed, 4 of 4 structural rejections matched), 96 of 105 ops used by the 92 text tests;
the suite is not a lowering oracle and the open dialect has no `mbarrier`/`wgmma`/`cp.async` ops. Nothing copied from the repository.
