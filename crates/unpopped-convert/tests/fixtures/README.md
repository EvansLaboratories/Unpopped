# Fixtures for `kernel_body_decode_goal.rs`

`flashinfer_paged_decode_f16_d128_g1_flat.cu` is a **verbatim copy** of
`docs/planning/u19/flashinfer_paged_decode_f16_d128_g1_flat.cu` at Baracuda commit
`ce29119567c7041360c83e84728091bcaf9658e9` (Baracuda #419), byte-identical (checked with `cmp`).

It is a hand-flattened single configuration (half, HEAD_DIM 128, group size 1, no positional encoding) of
**FlashInfer** `BatchDecodeWithPagedKVCacheKernel`, copyright (c) 2023 FlashInfer team, **Apache License 2.0**; upstream commit
`eee0d75f91f64c520bfaed07e39a850ea4ddde23`. The file's own header carries the notice and the statement of what was changed.
`FLASHINFER-LICENSE` and `FLASHINFER-NOTICE` are the licence and notice of the FlashInfer tree Baracuda vendors
(`crates/baracuda-kernels-sys/vendor/flashinfer/{LICENSE,NOTICE}` at the same commit), copied verbatim.

It is a **test input**, never compiled and not part of any published crate. The flattening was written by the Baracuda lane
and checked bitwise against the original by Baracuda's own harness (`docs/planning/u19/flashinfer_flat_check_harness.cu`).
