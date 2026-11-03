# Unpopped's IR as source material for a KISS "kernel body" sub-standard: a concept map

Status: **a map, not a proposal.** Written for the KISS lane (`jkkstudy`), who author the scope-and-process note (request relayed by
the PM, 2026-11-03: CireSnave agreed that Unpopped's IR is the source material for a new KISS normative layer, Unpopped being one
implementation of it; due 2026-11-06). Read at `origin/main` `aae2739` (0.42.18). Every claim names the type, rule or test that carries
it; where Unpopped *decided* something KISS might decide differently, it is in section 12, not buried.

**Legend.** *Common* = a rule every conforming reader/writer must share for programs to mean the same thing (candidate normative text).
*Target* = a rule that differs between devices or languages, so the standard should say "a target states it" or pin a reference
behaviour. *Evidence* = what in this repo checks the claim. "Oracle" is `unpopped::oracle` (a CPU interpreter); "C reference" is
`unpopped-cpu-c` (a serialized thread loop compiled with a host C compiler and compared byte for byte). **No construct below has been
run on a device by Unpopped.** The evidence is: an interpreter, an independent Rust reference per feature, a compiled-C differential, and
seeded mutants (a deliberately broken oracle must fail a test). It is evidence of *self-consistency and of the stated rules*, not of
agreement with any GPU.

## 0. Where the kernel body sits

Two layers, one above the other. The **ops layer** (`Access::{Elementwise, Reduction, RowReduce, Contraction, Scan, Window, RowSort,
Im2Col}`) says *what is computed* over whole tensors and is KISS-Ops territory. The **kernel body** (`KernelFn`, `Stmt`) says *what one
thread does*, in a launch of many; the ops layer lifts into it (`kernel_lift`), and the oracle checks that a lifted body and the op agree
(`tests/kernel_*_agrees.rs`, 57 files). The standard being discussed is the second layer. Its machine model is **one logical thread per
body**, replicated over a launch; there is no thread-private view of other threads except through the constructs in sections 5 to 7.

## 1. Function, launch, ids

| Construct | Observable semantics | Common / Target | Evidence |
|---|---|---|---|
| `KernelFn { params, locals, allocs, body, launch, compute }` | A list of typed scalar and buffer parameters, declared typed locals, declared memory allocations, a statement block, an optional launch shape, a compute type for unannotated arithmetic. | Common | `check_kernel` rules 1 to 6 (`docs/ir-kernel-layer.md` section 4); `tests/kernel_layer.rs`, `kernel_exec.rs` |
| `Param::Buffer { elem, role, rank }`, roles `Read`/`Write`/`ReadWrite` | A store to a `Read` buffer, or a load from a `Write` one, is ill-formed. A buffer is addressed by *logical coordinates*, not a pointer. | Common (the role discipline); aliasing between buffers is **not** stated | `WriteToReadOnly`, `BadPlace` tests |
| `Param::Scalar` | Readable as a local, never assigned (`ParamAssigned`). | Common | `kernel_exec.rs` |
| `LaunchShape { block, grid, subgroup, cluster }` | `block` threads per block and `grid` blocks per axis, three axes, no zero extent; `subgroup` is a *required size range* (the oracle runs the range's minimum unless told otherwise); `cluster` is blocks per cluster, the grid a multiple of it. | Common: the shape; **Target**: which sizes exist (max threads, subgroup sizes, cluster support) | `check_launch` (`BadLaunch`); `tests/kernel_cluster.rs`; cpu-c refuses blocks over 1024 threads (its own limit) |
| `ExecId::{ThreadIdx, BlockIdx, BlockDim, GridDim}` (axes 0 to 2), `LaneId`, `SubgroupId`, `SubgroupSize` | `U32` leaves with the usual meaning; lane/subgroup ids need a launch that names a subgroup size. No cluster id, no `%smid`, no clock. | Common (the ids); Target (the register file behind them) | `tests/kernel_exec.rs`, `kernel_launch.rs` |
| `Module`, `Stmt::Call { callee, args, result }`, `CallArg::{Scalar, Buffer, Alloc}` | A plain function called from a kernel; buffers and arrays are passed by reference, scalars by value; a result is one scalar. Recursion is rejected (`CallCycle`). A callee that reaches a barrier or collective is a barrier site (the caller must be in uniform control). | Common | `tests/kernel_module_*.rs`, `kernel_call_shape.rs`, `docs/options-u3-calls.md` |

## 2. Types and expressions

`ScalarExpr` leaves valid in a kernel body: `Const`, `ConstInt`, `Local`, `Exec`. The ops-layer ("intent") leaves `Input`, `Param`,
`Reduced`, `Coord`, `CoordInt` are rejected there (`KernelError::IntentLeaf`); the lift replaces them with loads, locals and ids. Nodes: `Add`/`Sub`/`Mul`/`Div`, `Unary` (39 functions), `Binary` (24 incl. comparisons,
bit ops, shifts, `Max`/`Min`/`FmaxIeee`/`FminIeee`/`Pow`/`Rem`/...), `Select`, `Fma`, `Cast`, `Bitcast`, `Qualified`, `Pack`/`Extract`/`BitcastPacked`.

| Concept | Behaviour | Common / Target | Evidence |
|---|---|---|---|
| Typing | Every expression has a type; no implicit conversion (`Cast` is explicit); `Let`, `Assign`, `Load` need type equality. | Common | `infer_ty_in`; `TypeMismatch` tests |
| Integers | Wrap at the type's width; **division is total** (a zero divisor gives the dividend, so does `MIN / -1`); shift counts are read as unsigned bits modulo the promoted width. A comparison yields the compute type's 1 or 0. | Common *as Unpopped defines it*; **Target**: hardware traps or leaves it undefined | `docs/int-div-shift-semantics.md` (PM ruling 2026-10-31), `tests/kernel_int_div_shift_semantics.rs` |
| Floats | Compute type `F32`/`F64` (and `F16`/`Bf16` storage); `Max`/`Min` propagate NaN, `FmaxIeee`/`FminIeee` suppress it; ties on a signed zero are pinned (operand `a`). | Common (the NaN and zero rules, aligned with KISS-OPS-6.15); **Target**: transcendental accuracy (ULP tiers), fused multiply-add contraction | `tests/kernel_eval_semantics.rs`; KISS-OPS-6.0 classes |
| `Fma` | A single rounding of `a*b+c`. | Common | 0.23.0 tests |
| `Cast` with `RoundMode` and `Saturation` | Converts with the stated rounding and overflow behaviour (`Wrap`, clamp-finite, ...); NaN handling per row. | Common (the semantics), Target (instruction availability) | `docs/ir-typed-expressions.md`; `tests/typed_expressions.rs` |
| `Bitcast` | Reinterprets bits between equal-width types. | Common | 0.24.0 |
| `Qualified` (rounding / flush-to-zero / approximate) | Qualifies an op with a rounding mode, flush-to-zero or an approximate form (the bounds are in the doc). | Common (the qualifier vocabulary), **Target** (what "approx" costs) | `docs/ir-numerics-qualifiers.md` |
| Packed values `Pack`/`Extract`/`BitcastPacked`, packed `Select` mask | Small fixed-width vectors of one element type. No lane-wise arithmetic operators beyond `Select` yet. | Common (shape), incomplete | 0.31.x tests |

## 3. Statements

`Let`, `Assign`, `Load`, `Store`, `If`, `For`, `While`, `Break`, `Continue`, `Return`, `Call`, plus the synchronizing ones in sections 5 to 7.

- **Declaration before use, one declaration per id, block scoping, no statement after an unconditional `Break`/`Continue`/`Return`**
  (`UseBeforeLet`, `OutOfScope`, `Redeclared`, `Unreachable`). Common.
- **`For`**: bounds and step evaluated once; half-open; `step == 0` traps (`ZeroStep`); the counter cannot be assigned in the body
  (`CounterAssigned`); a counter that would wrap ends the loop. Common. `While` re-evaluates its condition.
- **Conditions** are scalars of any type, tested `!= 0`. Common.
- **Two stores to one element in one thread**: the later wins. Across threads, see the race rule (section 6).
- The interpreter's `StepLimit` (default 10 million statements) bounds the *interpreter*, not the language.
- Evidence: `docs/ir-kernel-layer.md` section 5; `tests/kernel_layer.rs`, `kernel_resume_agrees.rs` (the resumable per-thread evaluator agrees
  with the straight one), `kernel_layers_agree.rs`.

## 4. Memory spaces and places

| Construct | Behaviour | Common / Target | Evidence |
|---|---|---|---|
| `Place::Logical { buffer, coords }` | A global-buffer element by logical coordinates through the buffer's strides; out of range is `OutOfBounds`, never wrapped. | Common | `kernel_exec.rs`, `kernel_launch_traps.rs` |
| `AllocDecl` in `MemorySpace::{Shared, Private, Constant}`, `Extent::{Static, Param}` | `Shared`: one instance per block; `Private`: per thread; `Constant`: declarable, **no initializer**, so the oracle refuses to read one; extent static or from an integer scalar argument. | Common (the spaces and scoping); **Target**: sizes, banks, alignment, opt-in limits | `tests/kernel_launch.rs`, `kernel_atomic_shared.rs`, `kernel_module_alloc.rs` |
| Initialization | A read of a `Shared` or `Private` element nothing stored is a trap (`UninitializedShared`/`Private`): shared memory is **not zero-filled**, so a missing-barrier bug cannot hide. | **Unpopped's choice**; devices hand back garbage | `kernel_launch_traps.rs` |
| Index outside an allocation | `AllocOutOfBounds`, never wrapped. | Common | same |

## 5. Barriers

`Stmt::Barrier { scope, kind: Full }` with scope `Subgroup`, `Block`, `Device` (the whole grid), `Cluster` (needs `LaunchShape::cluster`).
`Thread` has no meaning and is refused.

- **Semantics.** Every thread of the scope arrives, then every thread leaves; everything a thread did before is ordered before everything the
  scope's threads do after. `Device` assumes **every block is resident** (a cooperative launch); the oracle runs the grid in phases.
- **Definedness (the uniformity rule).** A barrier is defined only if every thread of its scope reaches it. `check_kernel` enforces a
  *conservative* taint analysis: a barrier under a branch, loop bound or `Break` that may differ between threads of the scope is
  `BarrierDivergent`; for a `Device` or `Cluster` barrier `BlockIdx` also counts as divergent. It rejects some correct programs and
  never accepts a divergent one. A thread that finishes while peers wait, or peers at different barriers, is `BarrierMismatch` (run time).
- **Common**: the semantics and the divergence rule (the *conservatism* is a choice a standard could replace with "undefined if not
  uniform" plus a checker note). **Target**: which scopes exist; `Device` needs cooperative launch; `Cluster` needs cluster hardware;
  subgroup size differs per device (CUDA 32, others 4 to 128).
- **Not covered (open)**: named/subset barriers with a participant count (`docs/options-u4-s6b-cluster-and-named-barriers.md`, S6b-2), split
  arrive/wait, `mbarrier`.
- Evidence: `tests/kernel_barrier.rs`, `kernel_barriers_run.rs`, `kernel_device_barrier.rs`, `kernel_cluster.rs`; C reference
  `kernel_launch_emit_agrees.rs`, `kernel_device_barrier_emit_agrees.rs` (Block and Device phases; Cluster spelled as a grid barrier, a
  valid stronger schedule).

## 6. Atomics, fences, the race rule

| Construct | Behaviour | Common / Target | Evidence |
|---|---|---|---|
| `AtomicRmw { op, place, value, old, order, scope }` | One indivisible `*p = *p op v`; `old` gets the value just before this thread's step in *some* serialization consistent with program order (**which one is unspecified**). Ops `Add`, `Max`, `Min`, `And`, `Or`, `Xor`, `Exchange`; integer elements all, `F32`/`F64` only `Add` and `Exchange`. `Add` wraps. | Common (the ops, the old-value contract); **Target**: float-atomic precision and denormals, 64-bit support | `kernel_atomic_rmw/fetch/shared.rs` |
| Float `Add` | Defined **only inside a band** `|r - exact| <= gamma_(n-1) * sum|x_i|` (order independent), never as bytes. | Common (the band); the band differs from KISS-CONTRACT-6.8-0012 as clause-pinned (see the note in `docs/ir-kernel-layer.md` 5a) | `kernel_parallel_float_scatter_band.rs` |
| `AtomicCas` | Stores `desired` on a match, always yields the value found; integer only; which thread wins is unspecified. | Common | `kernel_atomic_cas.rs` |
| `order` (`Relaxed`, `Acquire`, `Release`, `AcqRel`, `SeqCst`), `scope` (`Subgroup`, `Block`, `Cluster`, `Device`) | Happens-before edges: a release-class atomic then an acquire-class one on the same element orders the plain accesses around them; a relaxed RMW continues the release sequence; the edge needs the *narrower* of the two scopes to contain both threads. **`SeqCst` is accepted and treated as `AcqRel`** (no total order is modelled). | **Unpopped's model** (C++-fence-like); Target: hardware orders/scopes available | `kernel_atomic_order.rs` (13) |
| `Fence { scope, order }` | A release fence before a relaxed atomic publishes what the thread knew; an acquire fence after a relaxed read takes in what that read saw. A `Relaxed` fence is refused. No proxy/async fences. | same | `kernel_fence.rs` (8) |
| The race rule | Two accesses to one element, at least one a write, with nothing ordering them (a barrier of a scope containing both, or an acquire/release edge) are a **data race**; an atomic races only with a plain access, or with another atomic whose narrower scope does not contain both threads. The oracle *classifies* (`DataRace`); it does not simulate a weak machine. | Common as a classification; **the standard decides** whether a racy program is undefined or must be flagged | `kernel_launch_traps.rs` and the order/scope tests |
| `CrossBlockWait` | The oracle runs blocks in order; a block that spins after an acquire on a later block's write is named, not a hang. An oracle artefact, not a language rule. | Not language | `kernel_atomic_order.rs`, `kernel_fence.rs` |

## 7. Collectives

`Stmt::Collective { kind, value, result, scope }`, a rendezvous that exchanges registers (it is **not** a fence: it orders no memory).
Scopes `Subgroup` or `Block`, full participation required (in uniform control, like a barrier).

| Kind | Behaviour | Common / Target |
|---|---|---|
| `Shuffle { mode, operand }` | `Idx`: lane `operand mod n`; `Up`/`Down`/`Xor`: lane `i - operand`, `i + operand`, `i ^ operand`, **or the lane's own value when there is no such lane**. | Common: the in-range rule; **Target: the out-of-range rule** (this is CUDA's; other targets leave it undefined or wrap; the standard must pick or declare it target-stated) |
| `Broadcast { lane }` | Lane `lane mod n` to all. | Common |
| `Vote(All|Any|Ballot)` | `All`/`Any` give 1/0 as `U32`; `Ballot` the `U32` mask, a **subgroup** operation of at most 32 lanes in the oracle. | Common; **Target**: mask width above 32, inactive-lane semantics (Unpopped requires full participation, so none) |
| `Reduce(op)`, `Scan { op, exclusive }` | Left fold in lane order, every lane gets it / prefix fold. `Sum`, `Max`, `Min`, plus `And`/`Or`/`Xor` on integers; an exclusive `Max`/`Min` has no identity and is refused. A float `Sum` is the left fold, which a device may reassociate (compare within a bound). | Common (the semantics); **Target**: float order, subgroup size |

Evidence: `tests/kernel_collective.rs` (12), `kernel_collective_shape.rs`; C reference `kernel_collective_emit_agrees.rs` (5, byte for byte
on shuffles of every mode with out-of-range operands, votes, reductions, scans, a NaN in one subgroup).

## 8. MMA atoms and accumulation policies (honest status)

**Built:** only the *plan-level* part: `Access::Contraction` with a precision policy (`AccumSpec::{WideFloat, Tf32, Tf32x3, Bf16x3,
F16Accum}`, a `k_step` each) and a deterministic model (converted inputs, exact products, a narrow accumulator rounded every `k_step`
elements), with a declared error bound per emitter (`docs/accum-policies.md`, 0.42.19, under review). **Not built:** the kernel-level
`Stmt::Mma` over fragment locals with `MmaAtom` descriptors (U7 S2), WGMMA, `tcgen05`, sparse, block-scaled, fragment movement
(`docs/options-u7-mma-atoms.md`: proposal, six rulings taken). So this section has a *design* but no kernel-body *evidence yet*. KISS has
no clause for any of it (matmul is an order-invariant contraction; `MathFidelity` is two-valued).

## 9. Source leaves (what a lifter produces)

The kernel body is also the target of recognising source code (KISS-Consume's side). `unpopped::convert` and `cfamily` turn C-family
source idioms into ops-layer nodes, and `kernel_lift` turns those into bodies whose leaves are the four valid ones above
(`Const`/`ConstInt`/`Local`/`Exec`); what is recognised is idiom by idiom (`docs/idiom-lifting-design.md`), not a grammar of a language.
The standard's interest is the *leaf vocabulary and the statement set*, not the recognizers.

## 10. Evidence inventory

- **Oracle**: `unpopped::oracle` (`run_kernel`, `run_launch`, `run_launch_module`); a resumable per-thread evaluator, a launch driver with
  phases (block, subgroup, device, cluster), a memory log for the race rule.
- **Independent references**: each new feature's test holds the oracle to a reference written separately (e.g. a Rust scan over explicit
  coordinates, a product written from the policy text, hand-computed arrays for collectives).
- **C reference differential**: `unpopped-cpu-c` compiles to serial C and the bytes are compared with the oracle (blocks sequential, threads
  sequential, the thread loop split at barriers; collectives and Device phases spelled as block/grid-level code).
- **Seeded mutants**: every slice lists the mutants it must fail (e.g. 11 for the Cluster scope, 15 for the accumulation policies); two
  "survived" cases are recorded in the CHANGELOG as test gaps fixed, not hidden.
- **Counts**: 118 test files in `crates/unpopped/tests` (57 named `kernel_*`), 27 in `unpopped-cpu-c/tests`.
- **Not evidenced**: any device run; the behaviour of any specific GPU for the *Target* rows; concurrency beyond one valid schedule.

## 11. Who could be a second implementer

Unverified suggestions; the lanes know their own state.

| Candidate | What would make it a real second implementation | Pieces it would need first |
|---|---|---|
| **Fuel** | A consumer of the C reference already; running the same differential against its own backend. | Barriers at `Block`, atomics, collectives in its emitters. |
| **Baracuda** | CUDA emitter output compared with the oracle (the design lists "CUTLASS emitter output compared"). | Device barrier needs cooperative launch; cluster scope; memory-order mapping to PTX. |
| **Vulkane** (Slang/SPIR-V) | Its Slang emitter already honours the typed-decline default of `Backend::lower_kernel`. | Subgroup ops mapping, `GroupMemoryBarrier*`, atomics with SPIR-V memory semantics. |
| **Bonefish** (SYCL), **openclownfish** (OpenCL C) | Soft-stopped / early (ROADMAP notes). | `atomic_ref` order and scope, sub-group functions. |

## 12. Where Unpopped decided something KISS might decide differently

1. `SeqCst` accepted as `AcqRel` (no total order). 2. Shuffle out of range keeps the lane's own value (CUDA's rule). 3. Integer division
and shifts are *total*. 4. Race = classification, with a named trap, not undefined behaviour. 5. Shared and private memory reads of
unwritten elements trap. 6. The barrier divergence rule is a conservative analysis, not "undefined if not uniform". 7. A `Device` barrier
assumes residency. 8. The old value of an atomic is any consistent serialization. 9. Float atomic add is a band, not bytes (and differs
from the pinned KISS-CONTRACT-6.8-0012 form). 10. `Constant` memory is shape-only (no initializer). 11. Named barriers, `mbarrier`, async
copies, tensor memory and MMA atoms are **absent**. The synchronization and memory-ordering items are already proposed as KISS#769;
reductions/collectives as KISS#768.
