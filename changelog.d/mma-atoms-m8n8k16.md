kind: additive: NVIDIA mma.sync integer atoms m8n8k16 s8, u8, mixed s8/u8 (U7 theme 3)
---
Four `MmaAtom` constructors as data (`mma_m8n8.rs`): `nvidia_m8n8k16_{s8,u8,s8_u8,u8_s8}_s32`. Layouts from the public PTX ISA 9.5 section 9.7.16.5.3 (A Figure 59 pp. 539-540, B Figure 60 pp. 540-541, C Figure 61 p. 542); single source (CUTLASS not re-read). Patent column US12314717B2: applies 519 B; wrapper read 2026-11-27; not a clearance; no attorney; findings not advice. No `Stmt`, `ScalarExpr` or `ElementKind` change.
