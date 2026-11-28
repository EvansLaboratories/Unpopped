kind: docs, ROADMAP U72 to U75, the tile rules the WMMA and Apple shapes cannot meet yet
---
ROADMAP gains U72 (an `F16` accumulator for `TileMma`), U73 (TF32 as a tile element), U74 (integer `TileMma`, `s8`/`u8`/`s4`/`u4` into `s32`) and U75 (1-bit tiles and atoms: the atoms are built, the packed/vector popc statement stays held, US10452398B2), from the coverage table of `unpopped-mma`'s #525 (`docs/mma-atom-coverage.md`). Each names its path; none is declined for lack of a consumer. No code change.
