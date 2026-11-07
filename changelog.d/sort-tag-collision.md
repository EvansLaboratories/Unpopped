kind: fix, a flaky cpu-c row-sort test (two tests wrote the same C file name in parallel)
---
`kernel_emit_row_sort_agrees`: `degenerate_rows_agree` and `full_sorts_agree_in_every_form` built the same tag (`sort_F32_Asc_Both_0_2`, `..Desc..`), so the parallel tests overwrote each other's `.c` and `.exe`. The tag now includes the shape. Test-only.
