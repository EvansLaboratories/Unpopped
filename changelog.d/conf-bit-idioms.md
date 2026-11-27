kind: Added
---
Kernel-layer conformance kernels for two integer idioms from other kernel languages' public docs, built from leaves already on main (rows 20 and 28 of `docs/ir-union-evidence/conformance-idiom-gaps-2026-11-27.md`): a 32 x 32 to 64-bit carry-less (GF(2)) multiply (Triton `tl.clmul`) and a four-lane signed byte dot product with accumulate (CUDA `__dp4a`, TileLang `T.dp4a`). Each is compared on the oracle with a reference written independently (the carry-less product by coefficient parity rather than the kernel's shift-and-XOR loop; the dot product with Rust's `as i8` casts) over boundary corpora, with mutants (carries instead of XOR, an off-by-one shift, unsigned bytes, a dropped accumulator) that fail by value. Arithmetic semantics only: whether a target has one instruction for either is not modelled, and no new statement or operator was needed.

— lane: unpopped-conformance
