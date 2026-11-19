kind: the oracle's F64 atanh is accurate near -1 (bug fix)
---
`UnaryOp::Atanh` at `F64` used `f64::atanh`, which is `0.5 * ln_1p(2x / (1 - x))` and cancels for `x` near -1: 1172 ULP off at `x = -0.999956949180724`, 1290 at `-0.99995695`, 37618 at `-0.999999`, while the same magnitudes positive were exact. Every emitter's ULP claim for `F64` `atanh` is measured against this oracle, so it inherited the error (openclownfish reported a device result 20 ULP from the oracle on NVIDIA). The oracle now evaluates `atanh(|x|)` and restores the sign, since `atanh` is odd (NaN, +-0, +-1 and out-of-domain values unchanged). Values change for negative `x` near -1; anything pinning the old oracle values there was pinning the error.

Verified: `tests/oracle_atanh_f64_accuracy.rs`, red first (1172 ULP, the reported figure), against correctly rounded 80-digit `Decimal` values; `asinh` and `acosh` swept over 20000 points the same way, worst 2 ULP, no change.
