kind: **Fixed**: the oracle's `Lgamma` followed no special values (NaN at `+inf`)
---
`oracle::scalar_ops::lgamma` computed `inf - inf` at `+inf` (NaN; C `lgamma(+inf)` is `+inf`), returned a large finite value near a negative integer where C has a pole, and was off by ~1e-15 at 1 and 2. It now follows C99 Annex F.9.5.3: `+-inf`, `+-0` and negative integers give `+inf`, NaN stays NaN, `lgamma(1)` and `lgamma(2)` are `+0`. Found by openclownfish's device test (RTX 4070 Laptop, `device_agrees_float_transcendental.rs`). Pinned by `oracle::scalar_ops::lgamma_special_values`.
