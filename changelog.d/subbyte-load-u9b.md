kind: sub-byte loads and the dequantizing GEMM: findings, pinning tests and an options note (ROADMAP U9b; docs and tests only)
---
`docs/options-u9b-subbyte-dequant.md`, `tests/kernel_subbyte_load.rs`, `unpopped-cpu-c/tests/kernel_subbyte_load_refused.rs`. The U9b trigger
half-fired: a kernel `Load`/`Store` of an `I4` buffer element round-trips exactly in the oracle (low nibble first), but a `Cast` from `I4` is a
typed refusal ("not defined yet") and the C emitter does not spell an `I4` buffer. Both refusals are pinned; the note proposes extending the
`Cast` endpoints to `I4`/`U4`, spelling them in the emitter, then the dequant kernel, and keeping sub-byte buffers read-only in kernels.
