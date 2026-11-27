kind: tests: wide-integer idioms in the oracle (64x64->128, 128-bit dot accumulator, 128x128->256)
---
`kernel_wide_int_idioms`: the widening multiply, an `I128` accumulator over `I64` products, and the `U128` x `U128` -> `U256` product, each a kernel of ordinary ops compared with Rust's `u128`/`i128` or an independent 32-bit-limb schoolbook; each has a one-width-too-small mutant that changes the value. Oracle only; no carry-chain lowering. Stacked on `lane/e2-wide-int-r3`.

— lane: unpopped-conformance
