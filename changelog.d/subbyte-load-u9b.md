kind: sub-byte `Cast` endpoints `I4`/`U4` (ROADMAP U9b S1; additive: a refusal becomes an acceptance)
---
Ruled by the PM 2026-11-05 (`docs/options-u9b-subbyte-dequant.md`). `check_cast` now accepts `I4` and `U4` against an integer endpoint with `Saturation::Wrap`
(widening is exact: `I4` sign-extends, `U4` zero-extends; narrowing wraps modulo 16); `types::{is_subbyte_int, subbyte_range}` are new. The oracle evaluates
a sub-byte value as a local, a constant or a narrowing `Cast`. The checker refuses, typed, a `Store` of a sub-byte element (sub-byte buffers are read-only in
kernels in v1: a store is a read-modify-write of a byte two threads share) and any operator over a sub-byte operand (widen it with a `Cast` first); a
saturating cast or one against a float endpoint is still refused. `int_range` is unchanged (47 call sites treat it as "an arithmetic integer"). The C emitter
still does not spell an `I4` buffer (slice S2, pinned by `kernel_subbyte_load_refused.rs`). Tests: `kernel_subbyte_load` (6); nine seeded mutants caught.
