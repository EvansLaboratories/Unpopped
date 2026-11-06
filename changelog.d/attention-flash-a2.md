kind: flash-style attention conformance kernel A2 (ROADMAP U10; tests only)
---
`kernel_attention_flash.rs` and `unpopped-cpu-c/tests/kernel_attention_flash_emit_agrees.rs` pin an online-softmax attention written with the kernel layer alone: no score matrix, per row and output column a running max, sum and accumulator, rescaled by `exp(m - m')` when a key raises the max. The oracle result matches A1's naive kernel and the host loop; the C matches `run_launch` within 2 ulps. See `docs/options-u10-attention.md`.
