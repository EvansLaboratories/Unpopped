kind: `AtomicOp::Inc`/`Dec` (CUDA `atomicInc`/`atomicDec`)
---
Added `AtomicOp::Inc` (`old >= v ? 0 : old + 1`) and `Dec` (`(old == 0 || old > v) ? v : old - 1`), `U32` only (the checker refuses other elements by name): CUDA C++ Programming Guide `atomicInc`/`atomicDec`, PTX ISA `atom.inc.u32`/`atom.dec.u32`. Oracle, cpu-c and the CUDA walker (`atomicInc`/`atomicDec`) spell them. Pinned by `kernel_atomic_incdec.rs` (an independent loop from the CUDA definition) and the cpu-c differential `the_wrapping_inc_and_dec_agree`.
