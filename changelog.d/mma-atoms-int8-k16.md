kind: additive: NVIDIA mma.sync integer atoms m16n8k16 s8, u8, mixed s8/u8 (U7 theme 3)
---
Four `MmaAtom` constructors as data (`mma_sm80_8bit.rs`): `nvidia_m16n8k16_{s8,u8,s8_u8,u8_s8}_s32`. Layouts from the public PTX ISA 9.5 section 9.7.16.5.9 (Figure 87 p. 568, Figure 88 p. 570, Figure 89 p. 571); single source (CUTLASS not read for this shape). Patent column US12314717B2: applies 519 B; wrapper read 2026-11-27; not a clearance; no attorney; findings not advice. No `Stmt`, `ScalarExpr` or `ElementKind` change.
