kind: `unpopped-cpu-c` spells an MMA with `F16`/`Bf16` operand fragments (ROADMAP U26 fourth slice)
---
A `Stmt::Mma` whose `A`, `B` or extra fragments are `F16` or `Bf16` (the `nvidia.mma.sync.m16n8k16` f16 and bf16 atoms, `F32` accumulator) is no longer a typed refusal in `unpopped-cpu-c`. The gather decodes each narrow fragment element through the codec into the `double` the product and the ascending-`k` sum already run on, and the accumulator is rounded once to `F32` as before. A narrow accumulator stays refused: no atom has one, so its encode could not be checked against the oracle.

Verified: `the_narrow_float_atoms_agree_with_the_oracle` compares both atoms with the oracle over a deterministic fragment fill; a mutant that gathers the raw pattern instead of decoding it fails it. The old test that expected a refusal of the f16 atom is replaced by it.

Open under U26: a narrow compute dtype.
