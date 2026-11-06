kind: masked attention conformance kernel A3 (ROADMAP U10; tests only)
---
`kernel_attention_masked.rs` and `unpopped-cpu-c/tests/kernel_attention_masked_emit_agrees.rs` pin A2's online-softmax attention with a per-key visibility test (`j + lag <= row`; `lag = 0` is causal), an optional ALiBi bias `slope * (row - j)` on visible scores, and the defined result for a row that sees no key: zeros, not NaN (options note Q4). Oracle against a host loop in the same `f32` order; C against `run_launch` within 2 ulps. See `docs/options-u10-attention.md`.
