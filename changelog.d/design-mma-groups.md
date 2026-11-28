kind: docs: design note `docs/design-mma-groups.md` for a `groups` field on `MmaAtom` (ROADMAP U70, f16 `m8n8k4`)

Docs only. Four independent 8-lane MMAs per warp (PTX ISA 9.5 section 9.7.16.5.1, Figures 49-55 cited by figure, text formulas read; single source); option A accepted by `unpopped`: additive `groups` plus a `(group, local lane) -> lane` map on `MmaAtom`, `MmaAtom` stays free-form. Applies 519 B; US12314717B2 wrapper read 2026-11-27; not a clearance; no attorney; findings not advice.
