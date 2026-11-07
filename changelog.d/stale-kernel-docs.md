kind: docs, doc comments that said Call, AtomicCas and the module type were unchecked, and the U11 note that said nothing was built
---
`kernel.rs` said `check_kernel` refuses `AtomicCas` and that nothing checks or runs `Stmt::Call` or `Module`; both are checked and run. `docs/options-u11-extern-and-inline-asm.md` said "nothing built" after typed `Extern` and `Stmt::ExternCall` landed (0.43.0). Comments only.
