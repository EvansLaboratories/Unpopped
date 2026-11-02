# Options note: `Stmt::Call` and the module checker (ROADMAP U3 R1, ir-kernel-layer unknown U-2)

Status: **Q2 ruled 2026-11-02 (option b); M1 built in 0.41.4 (#244); M2 follows.** Asked by the PM (2026-11-02,
`[NOTICE] topic=Unpopped-next`): what a callee's check, evaluation and refusal rules are, as a note before the build,
because real CUDA/HIP/SYCL/Slang kernels call device functions and the parsers cannot lift them without it. Read at
`origin/main` `b5eb5d3` (0.41.1); the status line and the M1 row were updated at `0583d20` (0.41.4). Q1 and Q3-Q7 below
are still **recommendations**, not rulings.

## What exists

0.41.0 landed the shapes and the refusal: `Stmt::Call { callee: u32, args: Vec<ScalarExpr>, result: Option<u32> }`,
`Module { fns: Vec<KernelFn> }`, `KernelError::CallUnsupported`. `options-u3-remainder.md` ruled the callee a
`KernelFn` of the module (R1a), an acyclic call graph, and a callee holding a `Barrier` called from uniform control
only. Nothing yet says how a callee returns a value, receives a buffer, or runs. Those are the questions below.

**Q2 was settled before the publish.** 0.41.x had not been published, so the PM ruled to reshape the `Call` of 0.41.0
before it reached crates.io (`args: Vec<CallArg>`, `KernelFn.result`; 0.41.4, #244) and to publish 0.41.x only after that
merged: `args: Vec<ScalarExpr>` was never public and no consumer saw a break.

## Questions, options, recommendation

**Q1. How does a callee return a value?** `KernelFn` has no result and `Stmt::Return` carries none.
(a) a result local on the callee: `KernelFn.result: Option<u32>` (additive, `serde(default)`), the value that local
holds at `Return` or at the end; the caller's `Call { result }` receives it. (b) `Return { value: Option<ScalarExpr> }`
(breaks `Stmt` again; every `Return` match). (c) callees write through a buffer only, no scalar result.
**Recommend (a):** additive, the checker already types locals, and a multi-return callee assigns the one local then
`Return`s. Its type is `locals[result].ty`.

**Q2. How does a callee receive a buffer or an allocation?** The device functions that matter take arrays and
pointers (`__device__ void f(float* p, int n)`; Slang `RWStructuredBuffer`); `args: Vec<ScalarExpr>` passes scalars
only. (a) scalar arguments only: simple, and covers pure math helpers, but a callee cannot touch memory, so most real
helpers stay unliftable. (b) `args: Vec<CallArg>` with `CallArg::Scalar(ScalarExpr) | Buffer(u32) | Alloc(u32)`: a
buffer argument binds the callee's `ParamKind::Buffer` to the caller's buffer parameter (or `Shared`/`Private`
allocation), by reference; role, element type and rank must agree. (c) pointer values (`Place::Ptr`, R2): the general
answer, but it brings an aliasing story nobody has asked for. **Recommend (b); RULED and built in 0.41.4:** it
changed the `Call` shape while that was still free. A buffer argument is an *alias*, so the race detector keys on the
caller's buffer and `ReadOnly` roles must hold through the call (a callee `Store`s only to a `Write`/`ReadWrite`
binding). Two buffer arguments naming the same caller buffer are allowed (no `restrict`); the race detector already
handles that, it sees one buffer.

**Q3. What is the callee's environment?** A callee is a device function: it runs on the calling thread, inside the
caller's launch. So: its `launch` must be `None` (else `CallUnsupported`); `Exec` ids (`ThreadIdx`, ...) read the
calling thread's; its scalar parameters are fresh locals bound by value (never assigned, as for a kernel); its own
`locals` are private to the activation. Its `allocs`: `Private` allowed (a per-activation array); `Shared` allocations
declared *in a callee* are refused in v1 (a block-wide object belongs to the kernel, and two activations would have to
agree on identity); `Constant` likewise. **Recommend** exactly this.

**Q4. How does it run?** (a) *define* a call as inlining, and implement it as an inlining pass before checking and
running: no new machine, but a `Return` inside a loop of the callee must become a flag-and-break ladder, which is the
bulk of the work and a place for bugs. (b) *native frames:* the resumable evaluator (`kernel_resume.rs`) already keeps an
explicit frame stack; add a `Frame::Call { caller_locals, result }` that saves the caller's locals, runs the callee
body with fresh ones, and restores them on `Return`/end. A `Barrier` inside a callee then suspends the thread
mid-call with no extra work, which is what a barrier-holding callee needs. The recursive evaluator (`run_kernel`,
unlaunched kernels) needs a plain recursive call. **Recommend (b):** the semantics is then the call, not a rewrite,
and the two evaluators can be held to each other by the existing `the_two_evaluators_agree` differential. Inlining
stays available to an emitter as an implementation choice.

**Q5. `check_module`: what is checked?**
1. every `Call.callee` indexes `fns`; arity and kinds match the callee's `params` (scalar types equal, buffer
   element type and rank equal, role compatible per Q2);
2. the call graph is acyclic (a DFS with a visiting set, `CallCycle { path }`); a recursive kernel is refused, since
   GPU targets mostly forbid it and the frame stack would need a depth bound;
3. each function passes `check_kernel`, which now *accepts* `Call` when a module context is supplied
   (`check_kernel(&k)` alone keeps refusing it: no module, nothing to resolve);
4. `result` local type equals the callee's result type; a callee without a result and `result: Some` is an error;
5. the entry kernel has the launch; callees do not (Q3).
Surface: `check_module(&Module, entry: u32) -> Result<(), KernelError>`, new variants `CallCycle`, `CallArity`,
`CallArgMismatch`, `BadCallee` (the enum is non-exhaustive, additive).

**Q6. Barriers and divergence through a call.** Compute, per function, `has_barrier` transitively (callees first,
which the acyclic graph permits). A `Call` to such a function is a barrier site: it must sit in uniform control, and
`kernel_uniform.rs` treats it as the `Barrier` arm does. A call's `result` local is thread-divergent unless the
callee is proven uniform; **recommend always divergent** in v1 (conservative: it can refuse a program, never admit a
bad one). Atomics and accesses in a callee are logged with the calling thread, so the race detector needs no change.

**Q7. Effects.** The effect set (ROADMAP U3) is "computed when a consumer asks". A call's effect is the union of its
callee's; stating that now costs nothing and nothing computes it yet.

## Slices (each additive after the Q2 reshape, each red-first with seeded mutants)

| slice | content | breaks? |
|---|---|---|
| M1 | `CallArg`, `KernelFn.result` (Q1, Q2 shapes) | **built, 0.41.4 (#244)**; changed the `Call` shape before the publish |
| M2 | `check_module` + the new `KernelError`s; `check_kernel` still refuses a bare `Call` | additive |
| M3 | recursive evaluator runs a module (`run_module`) | **built, 0.41.7** (scalar and buffer arguments; allocations through a call are refused) |
| M4a | resumable evaluator `Frame::Call`; a launched entry calls plain functions (`run_launch_module`) | **built, 0.41.8** |
| M4b | barrier in a callee; transitive divergence (Q6) | **built, 0.41.9** |
| M4c | allocations through a call (`CallArg::Alloc`, a callee's `Private` allocations) | additive |
| M5 | `unpopped-cpu-c` spells a call as a C function (serial reference), the differential for M3/M4 | **built, 0.41.10** (`lower_module`, unlaunched) **and 0.41.11** (launched: a barrier-free callee reads the thread's ids as arguments) |

Mutants to seed: arity unchecked; cycle undetected; callee locals not saved or not restored; result written to the
wrong caller local; a buffer argument bound to the wrong parameter; a `Store` through a `Read` binding admitted; a
callee's `Barrier` not counted for the caller's divergence check.

## Decisions for the PM

1. ~~Q2(b): reshape `Call` to `CallArg` before the publish~~ **ruled (b) and built, 0.41.4.**
2. Q1(a) result local (the shape landed with M1), Q3 (callee `Shared` allocations refused), Q4(b) native frames, Q6
   (call results divergent): accept as recommended, or say which to change. M2 (`check_module`) enforces Q3 and the
   no-barrier-callee rule as typed refusals either way, so these two rulings gate M3/M4, not M2.

Not measured: how many of Fuel's 147 kernels call a device function that takes a buffer. The U3 spec says
device-function calls are "needed before most real Slang kernels lift"; no sibling has counted buffer-taking callees.
If Fuel can count them, it settles Q2(a) against (b).
