# Question for CireSnave: what should the oracle promise for atomic order and scope, and for fences? (ROADMAP U4 S4, S5)

Status: **ruled A by CireSnave 2026-11-03** ("370: A. We may switch to C later or at least expand this more but A is fine for now."), with both sub-answers yes; **S4 (order and scope) built in 0.42.7**, S5 (`Fence`) next. Originally: a question, nothing built. Asked by the PM (2026-11-02): write this when U4 reaches S4/S5. The U4 slices
that did not need an answer are built (`docs/options-u4-sync-and-atomics.md`: S1 old value, S2 more combines, S3
`AtomicCas`; the shapes of S4 and S5 are already in `Stmt` and refused with a typed error). These two cannot be
built without a decision on what "correct" means, and it is yours.

## What the oracle promises today

The interpreter runs the threads of a block in lockstep phases between barriers, and blocks one after another. It
checks a **race rule**, not an execution order:

- Two accesses to one element, at least one a write, with no barrier between them that orders them for the two
  threads, are a `DataRace`. A `Block` barrier orders the threads of the block; a `Subgroup` barrier those of the
  subgroup. Accesses by different blocks to a global buffer are never ordered (the oracle has no way to order them).
- An atomic races only with a **plain** access to the same element; two atomics never race (they are `Relaxed` and
  `Device` scope, the only form accepted today).

That is the whole model, and it is deliberately a classification of programs, not a simulation of a machine: the
interpreter is one valid schedule, so it cannot *show* a weak-memory outcome; it can only *refuse a program that
needs a guarantee the program does not state*.

## The question

`AtomicRmw`/`AtomicCas` already carry `order` (`Relaxed`, `Acquire`, `Release`, `AcqRel`, `SeqCst`) and `scope`, and
`Stmt::Fence { scope, order }` exists. What should the oracle do with them?

**A. Race classification with happens-before edges (recommended).** An `Acquire` that reads a value written by a
`Release` (same element, or through a fence pair) creates a happens-before edge between the two threads, and plain
accesses ordered by it do not race. A message-passing kernel (write data; release flag | acquire flag; read data) is
race-free with release/acquire and a `DataRace` with `Relaxed`. A `Fence` of `Release`/`Acquire` order turns the
adjacent relaxed atomics into the release/acquire of the C++ fence rules. `SeqCst` is treated as `AcqRel` (the oracle
does not model the single total order). Scope: two atomics at a scope narrower than the relation between their
threads (a `Block`-scope atomic used by two blocks) are a `DataRace`.
*Cost:* a per-element release/acquire log in `kernel_mem.rs` and a few dozen lines of edge computation; tests are
message-passing, store-buffer-shaped and scope-mismatch kernels, each with a seeded mutant that drops the edge.
*Limit:* it admits every program that is correctly synchronized and rejects those that are not, but it cannot say what a
racy program would print on a device.

**B. Ignore order.** Treat every atomic as sequentially consistent and `Fence` as a no-op. Smallest, but it accepts
kernels that are wrong on hardware (relaxed message passing) and gives a cross-lane emitter no way to be checked on
the property it most needs.

**C. Explore outcomes (a litmus-test model checker).** Enumerate the memory-model-allowed outcomes. Exact, but it is a
research-sized piece of work and the oracle's job (a reference for the emitters' output) does not need it.

## Two sub-questions that A forces

1. **Cross-block communication.** The oracle runs blocks one after another. A consumer block that spins on a flag a
   *later* block writes never finishes (`StepLimit`). Recommended: until the `Device` barrier / residency model of
   S6 exists, a wait on another block's write is a typed trap (`CrossBlockWait`), not a hang, and only the
   producer-before-consumer order is run.
2. **`SeqCst`.** Recommended: accepted, treated as `AcqRel`, and the CHANGELOG says so, so nobody reads "checked
   sequentially consistent" into it.

## What I will do on each answer

- **A** (default if you say nothing else): build S4 then S5 as two additive slices, each with its tests and
  mutants, the first consumer being the message-passing differential in the cpu-c reference.
- **B**: lift the refusals for `order` and `Fence` as accepted-and-ignored, with a CHANGELOG line that the oracle does
  not check them.
- **C**: not started without a separate plan; I would write that plan first.

Reply with A, B or C, and with yes/no on the two sub-questions (recommended: yes to both).
