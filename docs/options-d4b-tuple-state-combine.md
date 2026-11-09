# Options note: a tuple state for a combining collective (ROADMAP R5.7 D4, slice b)

Status: **proposal; no code.** Written 2026-11-09 against `origin/main` after #321 (`CollectiveKind::{ReduceWith, ScanWith}`, a scalar combine function). PM 2026-11-09: slice b (tuple state) goes ahead after C4, C4b and F7; this note picks the representation before any code, because the obvious one does not work today.

## What is needed

Triton's `tl.reduce` and `tl.associative_scan` fold a **tuple** with a user combine (`combine_fn` over tuples): argmax carries `(value, index)`, Welford carries `(mean, m2, count)`. A scalar combine (#321) cannot express them; the kernel-layer loop can, but a loop is not a declaration that the fold is associative, so an emitter cannot turn it into a shuffle tree.

## What the code does not do today (measured at `origin/main`)

- The kernel evaluator **refuses a packed value**: `oracle/kernel_eval.rs:559` returns `Unsupported("packed value")` for `Ty::Packed`. `LocalDecl::packed` and `ScalarExpr::Pack`/`Extract` exist, but only the intent layer's lane-wise operators evaluate them.
- A callee has **one result**: `KernelFn::result: Option<u32>`; `Stmt::ExternCall` names several `results` but `exec_extern` passes only the first (`oracle/kernel_call.rs:106`).
- `ParamKind` is `Buffer` or `Scalar`; there is no packed parameter.

## Options

| | State carried as | New in the IR | Evaluator and emitter work | Reuses |
|---|---|---|---|---|
| **A** | a packed local (2 or 4 lanes of **one** element type) | `ParamKind::Packed`, a packed `result` local, packed `CallArg` | the oracle and cpu-c must hold packed values everywhere a local can be (the largest change) | `Pack`/`Extract`, `LocalDecl::packed` |
| **B** | `k` scalar values, each its own type | `KernelFn::results: Vec<u32>` (extra results beyond `result`), `Stmt::CollectiveTuple { kind, values, results, scope }` as a **new variant** (no existing struct changes), a combine of `2k` scalar parameters and `k` results | multi-result calls in the oracle and cpu-c (a C function writing `k` out-pointers), a rendezvous that carries `k` values per lane | scalar `Call`, #321's driver |
| **C** | a `Private` array of `k` elements | callee arguments by alias (`CallArg::Alloc`, refused today for `Private` in a launch: `cpu-c` `call_actuals`) | lifting that refusal | allocations |

## Recommendation: B

- **Mixed types** (a `f32` value with a `i32` index) are the common case; A is homogeneous and would force a bitcast of the index.
- It **mirrors what the emitters write**: a struct of scalars passed to a combine function (CUDA, Triton's lowering). The state is explicit in the IR instead of hidden in a packed lane.
- It touches **no existing variant**: `Stmt::CollectiveTuple` and `KernelFn::results` are additions (`Stmt` is `#[non_exhaustive]`; `KernelFn` gains a defaulted field, a breaking change for a literal `KernelFn { .. }` outside this repo, which the PM stamps as the second number).
- It keeps #321's rule: the author asserts associativity, the oracle folds left to right in lane order, a float state is compared within a tolerance; no exclusive scan (no identity).

## Slices if B is ruled

1. `KernelFn::results` and multi-result `Stmt::Call { results: Vec<u32> }` in the checker, oracle and cpu-c (a plain function with two results, no collective yet: argmax over a loop calling it).
2. `Stmt::CollectiveTuple` reduce and scan, oracle and cpu-c, with the Triton `argmax` and a Welford combine as the conformance fixtures, hand-worked expectations, cpu-c differential and seeded mutants.

## Question for the PM

B as above, or A (packed state, homogeneous, the larger evaluator change)? Recommend B.
