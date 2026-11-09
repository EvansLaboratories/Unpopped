# Options note: collectives, arg-reductions, bit reductions, NaN-ignoring reductions, custom combines (ROADMAP U5)

Status: **ruled 2026-11-02 (A staged; P4 at the kernel layer; P3 PyTorch; P5 full participation; P1 `MaxNum`/`MinNum`, all-NaN gives NaN); slice 1 built in 0.42.0, P1 (`MaxNum`/`MinNum`) in 0.42.1, P3 (arg-reductions, as `ReduceOp::{ArgMax, ArgMin}` rather than a new `Access` variant) in 0.42.2, P5 (collectives, in the oracle) in 0.42.3 and the cpu-c reference in 0.42.4.** Written as a proposal, nothing built, before the rulings. Asked by the PM (`[TASK]`, 2026-11-02: "U5 options note then build", ROADMAP
demand order U2, U3, U6, U4, U5). Gap rows 3.6 (collectives) and 3.16 (reductions, scans and sort with richer
state) of `docs/ir-union-design.md`; increment I5. Read at `origin/main` `265e642` (0.41.13). The surveys it leans on
are `docs/ir-union-surveys/{cuda,cute,mojo,triton}.md`; their cell contents are quoted, not re-measured.

## What exists

- **Intent layer.** `ReduceOp` is a closed, **exhaustive** enum: `Sum`, `Mean`, `Max`, `Min`, `Prod` (`ir.rs`).
  `Max`/`Min` propagate NaN. `Access::Reduction`, `RowReduce`, `Scan` (monoids `Sum`/`Prod`/`Max`/`Min`) and `Window`
  use it. There is no index-carrying reduction, no bit or logical reduction, no NaN-ignoring `Max`/`Min`, and no
  tuple state. Fuel's `lower_op_kind` table (snapshot in `contract.rs`) accepts `ArgMaxDim` and `ArgMinDim`; the
  lifting recipes know `argmax`, `any` and `all` by name (`recipe.rs`), so the names have a consumer.
- **Kernel layer (U2 to U4).** `Let`/`Assign`/`For`/`While`/`If`, `Place::Alloc` in `Shared` and `Private`,
  `Barrier`, atomics, and (since 0.41) calls. A *sequential* fold with any state tuple, including a custom combine
  and an arg-reduction, is therefore **already expressible** as a loop over locals; what is not expressible is a
  *cooperative* one.
- **Collectives.** Absent as user-visible nodes. The CUDA emitter lowers `Reduction`/`RowReduce`/`Scan` to
  `__shfl_down_sync` trees as a schedule (`idiom-lifting-design.md`), which a kernel parsed from CUDA, HIP or Slang
  cannot name. The surveys list the same gap in every language: CUDA `__shfl_*`, `__ballot`, `__reduce_*_sync`,
  cooperative-groups partitions; CuTe `Shuffle/Vote/Match/Redux/ElectOne`; Mojo warp and block collectives; Triton
  `tl.reduce` with a user combine and `argmax`/`argmin`/`any`/`all`/`xor_sum`.
- **Demand.** ROADMAP ranks U5 fifth from Fuel's 147-kernel Slang corpus (regex counts in Fuel#514, which this
  repository does not carry, so the counts are not repeated here). Nobody has measured CUDA-side demand.

## The pieces

| id | shape | breaking? | oracle semantics it needs |
|---|---|---|---|
| P1 | `ReduceOp::MaxNum` / `MinNum`: NaN is ignored; a row of only NaN gives NaN | yes: `ReduceOp` is exhaustive, so every exhaustive `match` on it breaks (pre-1.0: second number) | IEEE `maxNum`/`minNum` fold; a differential against the cpu-c reference (`fmaxf`/`fminf` fold); scans and windows take the same op |
| P2 | `ReduceOp::{And, Or, Xor}` over integers, and `Any`/`All` (a logical fold of a non-zero test) | yes (same enum) | bit folds; integer-only (typed refusal for a float input); the `Any`/`All` output is the compute type's 1 or 0, as comparisons are |
| P3 | `Access::ArgReduce { op: Max \| Min, axis, keepdim }` producing an index of a stated integer dtype | no for downstream (`Access` is `#[non_exhaustive]`); the plan gate and the oracle in this crate gain an arm | first occurrence on ties; NaN rule below; output dtype `I64` by default (PyTorch) with `I32` selectable |
| P4 | tuple-state reduction with a custom combine: `ReduceOp::Custom { state, combine, identity }` | yes | a left fold in the oracle; a parallel emitter reassociates, so the claim "associative" is the author's and the differential is a tolerance, not bytes |
| P5 | collectives as **kernel-layer statements**: `Stmt::Collective { kind, value, result, scope }` with `kind` in `Shuffle{Idx,Up,Down,Xor}`, `Vote{All,Any,Ballot}`, `Reduce{op}`, `Scan{op,exclusive}`, `Broadcast`; `scope` in `Subgroup`, `Block` | **yes: `Stmt` is exhaustive on purpose** (`kernel.rs`: "a new statement breaks every consumer at compile time"), so a new statement is a break, exactly as `AtomicCas`/`Fence`/`Call` were in 0.41.0 | a rendezvous like a barrier: see "How the oracle runs a collective" |

### P3's rule, stated

**KISS wins (PM, 2026-11-02, after KISS-OPS-6.11-0007 was read; the first ruling, PyTorch's, was made without it).** NaN is
the **greatest** value and ties go to the lower index: `ArgMax` of a row with a NaN is the first NaN's index, and
`ArgMin` **never selects a NaN while a non-NaN exists** (index 0 if every element is NaN). That differs from PyTorch and
NumPy, whose `argmin` returns the NaN's index; 0.42.2 shipped the PyTorch `ArgMin`, a defect against KISS fixed in
0.42.5. A zero-length axis is a typed refusal, as it is for `Max`/`Min` today. Fuel's `ArgMinDim` is PyTorch's and
must be told. `ArgMax`/`ArgMin` are KISS-Ops's `argmax`/`argmin` (§6.13 table; KISS-OPS-6.11-0007 is their authority).

### How the oracle runs a collective (P5)

A collective is a `Stmt`, not a `ScalarExpr`: it needs every lane of its scope to have *reached it with its input
value* before any lane can have its result, which is exactly a barrier. So it reuses the machinery that exists:

- The resumable evaluator stops each thread at the statement (as it does at `Barrier`), recording the thread's input
  value; when every thread of the scope has arrived, the driver computes all the lanes' results (a shuffle is a
  permutation of the inputs, a reduce is a fold, a scan a prefix fold, a vote a predicate over the inputs) and
  releases the threads, each of which stores its own result in its `result` local.
- The **uniformity rule is the barrier's** (`kernel_uniform.rs`): a collective under divergent control is
  `BarrierDivergent`-class (a new `CollectiveDivergent`), and `result` is divergent. Full participation of the scope is
  the v1 rule; CUDA's partial `mask` argument is deferred, because a partial mask needs a participation-set semantics
  that `Barrier` has not needed.
- A thread that finishes while its scope waits is the existing `BarrierMismatch`.
- In the cpu-c thread-loop reference a collective is a **block-level** construct between two thread loops, reading
  and writing the per-thread local arrays `lN[t]` directly; that is simpler than the device spellings and is the
  differential the device emitters are held to.

## Options

**A. Everything, in demand order, one piece per PR.** P1 and P2 first (a few lines of op each, plus their emitters),
then P3, then P5 (subgroup scope first, then block), then P4. P5 then breaks `Stmt` consumers a second time.

**B. Kernel layer only.** Do P5 (collectives) and let P3 and P4 be *expressed* as kernel loops and collectives, with
no intent-layer growth. Smallest surface, but the intent layer stays unable to name `argmax`, which Fuel asks for by
name, and every consumer would have to emit a kernel for what is one op.

**C. Intent layer only.** Do P1 to P4 and leave collectives as an emitter schedule. Fails the CUDA, HIP and Slang
parsers, which must lift `__shfl_sync` and `WaveActiveSum` into *something*; the union-IR directive is that the IR
carries what the parsers need.

## Recommendation

**A, staged the way U4 was** (`options-u4-sync-and-atomics.md`, option B): break the exhaustive enums **once**, behind
typed refusals, so the later slices are additive.

1. **One breaking release** carrying `ReduceOp::{And, Or, Xor}` (P2, built in the same PR, since they need no ruling),
   the *shape* of `ReduceOp::{MaxNum, MinNum}` (P1), and `Stmt::Collective { kind, value, result, scope }` (P5);
   `check_kernel` and the plan gate refuse the shapes whose semantics has not landed (typed `CollectiveUnsupported`,
   `UnsupportedReduceOp`), as 0.41.0 did.
2. Then each additive, red-first, with its oracle test and seeded mutants: P1 (the NaN rule), P3 (`ArgReduce`, additive
   on its own), P5 in two steps (`Subgroup` scope, then `Block`), the cpu-c block-level reference first.
3. **P4 stays at the kernel layer.** *Built 2026-11-09 (ROADMAP R5.7 D4, PM ruling the same day: "build the kernel-layer combine via a callee"):* `CollectiveKind::{ReduceWith, ScanWith} { combine }` fold a `Subgroup` or `Block` with a plain module function `(a: T, b: T) -> T` (no barrier), left to right in lane order; `check_module` resolves it, `check_kernel` alone refuses it; the oracle runs it on a machine of its own; `unpopped-cpu-c` spells a call per lane. The author asserts the function is associative (a device may reassociate it), so a float combine is compared within a tolerance. **Not built:** an exclusive scan (a user combine has no identity), a tuple state (the callee takes and returns one scalar; a packed state needs packed parameters and results, a later slice), and any intent-layer `ReduceOp::Custom`. The original text follows.

3. **P4 stays at the kernel layer.** A custom combine is a loop over locals (it already is); an intent-layer
   `ReduceOp::Custom` is a large DAG-with-state feature nobody has asked for. The Triton survey's
   `tl.reduce(combine_fn)` is the only user, and no lane has a Triton frontend.

Why: P1, P2 and P5 each break an exhaustive enum, so taking the break once costs downstream one `match` update
instead of three; P3 and the semantics of P1 and P5 then land additively.

## Needs a ruling

1. A (recommended) versus B or C.
2. P4: kernel layer only, as recommended, or an intent-layer `ReduceOp::Custom` now.
3. P3's rule: KISS-OPS-6.11-0007's order (ruled 2026-11-02 first as PyTorch's, then corrected to KISS).
4. P5 v1: full participation of the scope, with the mask form deferred, as recommended.
5. P1's name and rule: `MaxNum`/`MinNum` with "all-NaN gives NaN", as recommended (the alternative, returning the
   identity for an all-NaN row, needs an infinity literal the current ops avoid on purpose).

Until a ruling arrives I will not build; once A is ruled, the first slice is the breaking release in step 1 with P2's
integer `And`/`Or`/`Xor` reductions built and `MaxNum`/`MinNum` and `Collective` as refused shapes: it needs only
rulings 1 and 2, and the shapes in it do not fix the rules of 3 to 5 (P1 waits for 5, P3 for 3, P5's semantics for 4).
