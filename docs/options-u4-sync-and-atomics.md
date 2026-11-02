# Options note: U4, synchronization and atomics (ROADMAP U4, gaps 3.5 and 3.10)

Status: **options, nothing built.** Asked by the PM (2026-11-02): an options note naming the IR shapes and which
emitter lanes each unblocks, then the smallest slice a named emitter lane can use. CireSnave's goal for the union
IR: it carries everything the parsers and emitters need (`ir-union-design.md`). Read at `origin/main` `59d25bd`
(0.40.20); the sibling counts are `git grep` over each repo's `origin/main` and say "no file matches", not "never
planned".

## What exists

- `Stmt::Barrier { scope }`: `Subgroup` or `Block` only. `check_kernel` rejects one under thread-divergent control
  (`KernelError::BarrierDivergent`). The interpreter runs a launch in lockstep with a resumable evaluator and
  reports `DataRace` between two accesses to one element that no barrier orders (`oracle/kernel_mem.rs`).
- `Stmt::AtomicRmw { op, place, value }`: `op` in `Add`/`Max`/`Min`, **relaxed**, **device-wide**, **no returned
  value**, on a `ReadWrite` global buffer element or a `Shared` allocation element, an integer element type or an
  `F32`/`F64` `Add` (`ir-kernel-layer.md` 5a). Two atomics on one element never race; an atomic against a plain
  access races unless a barrier orders them.
- `Scope` already has `Thread`, `Subgroup`, `Block`, `Cluster`, `Device` (`exec.rs`), but `Barrier` accepts two.
- Nothing for: fences, mbarrier, named or cluster or device barriers, async groups, a returned old value, memory
  order, atomic scope, `And`/`Or`/`Xor`/`Exchange`/`CompareExchange`/`Inc`/`Dec`, float `Min`/`Max`.

## What the surveys say is missing

`ir-union-design.md` rows 3.5 (synchronization) and 3.10 (atomics and the memory model):

| gap | CUDA | CUTL | CuTe | Slang | Mojo | Trit | TK |
|---|---|---|---|---|---|---|---|
| 3.5 barriers, mbarrier (phase, expect_tx, split arrive/wait), scoped and proxy fences, async commit/wait | X | X | X | X | X | X | X |
| 3.10 CAS, exchange, and/or/xor, inc/dec, float min/max, returned old value, order, scope, shared, 64-bit and unsigned | X | | X | X | X | X | |

(The matrix is "as reported by the surveys" and each increment re-reads its rows against the source; blank means
not listed.) Language facts, not measured here: CUDA `atomicAdd` and HIP `atomicAdd` return the old value;
HLSL/Slang `InterlockedAdd` has an `original_value` out parameter; OpenCL C 2.0 and SYCL `atomic_ref` take a
memory order and a memory scope.

## Who could use it (measured at each repo's `origin/main`)

| lane | kernel-layer atomics/barriers today |
|---|---|
| Unpopped `unpopped-cpu-c` | refuses both on purpose: `kernel.rs` "needs a launch" (it emits no launches) |
| bonefish (SYCL) | `lower_kernel` not overridden; its doc names the future spelling, `sycl::atomic_ref` `fetch_add`/`fetch_max`/`fetch_min`, relaxed, global; a test pins the typed decline |
| vulkane (Slang) | `lower_kernel` default only, with a test of the typed decline |
| baracuda, openclownfish, rockfish, fuel | no file mentions `AtomicRmw`, `Stmt::Barrier` or `AtomicOp` |

So **no emitter implements the kernel-layer path for these statements today.** "A lane that can use a slice"
means a lane that will write `lower_kernel`: bonefish has said it will (SYCL `atomic_ref` carries order and scope
as arguments, so it is the natural first consumer of both), and the in-tree reference (a serialized thread loop in
`unpopped-cpu-c`, the oracle promised by `ir-union-design.md` increment I3 and not built) is ours to write.

## The break-once problem

`Stmt` is exhaustive on purpose, so a new variant, or a new field on `AtomicRmw`, is a **breaking** change
(second number) for every `match` on it, as 0.40.0 was. `AtomicOp` is `#[non_exhaustive]`, so a new combine is
additive. Done slice by slice, U4 would break `Stmt` consumers several times. The options below differ mainly on
this.

## Shapes (what a slice adds)

| id | shape | breaking? | oracle semantics it needs |
|---|---|---|---|
| S1 | `AtomicRmw` returns the old value: `old: Option<LocalId>` | yes (field) | the old value is the value before this thread's step in *some* serialization of all atomics on that element consistent with program order; the interpreter uses ascending thread order; conformance asserts properties (a fetch-add of 1 over `n` threads returns a permutation of the initial value plus `0..n`), never the serial bytes. Same discipline as the float scatter band (`options-float-scatter-add.md`) |
| S2 | more combines: `And`, `Or`, `Xor`, `Exchange`, `Inc`/`Dec` (wrapping) | no (`AtomicOp` is non-exhaustive); float `Min`/`Max` stay refused (no native CUDA op) | pure functions of old and operand; associative and commutative except `Exchange` (order-dependent like S1) |
| S3 | `Stmt::AtomicCas { place, expected, desired, old }` | yes (variant) | compare and set as one step; needs S1's serialization rule |
| S4 | `order: Relaxed/Acquire/Release/AcqRel/SeqCst` and `scope: Scope` on atomics | yes (fields) | order: happens-before edges in the race detector (an acquire reading a release creates one), so a message-passing kernel is race-free with release/acquire and a `DataRace` with relaxed. Scope: two atomics narrower than the relation between their threads are a race (`atomic_block` across blocks). The lockstep interpreter cannot *violate* an order, so these are checked as race classification, not as litmus outcomes |
| S5 | `Stmt::Fence { scope, order }` | yes (variant) | as S4's edges; also what makes cross-block communication expressible without `Device` barriers |
| S6 | barrier forms: `Cluster` and `Device` scope, named barriers with a participant count | extend `Barrier` (variant or field) | `Device` needs every block resident at once (the interpreter runs blocks independently today); named barriers need a thread-subset rule and a changed divergence check |
| S7 | `mbarrier` (phase, expect_tx, split arrive and wait) and async groups (`cp.async` commit/wait) | yes | a stateful object in `Shared`, driven by the resumable evaluator; its payload (tracking async copies) is U8's, so only the sync-only subset can land first |

## Options

**A. One slice per PR, in demand order (S2, S1, S3, S4, S5, S6, S7).** Each PR is small and reviewable. `Stmt`
consumers break up to five times in a pre-1.0 range, and S1 then S4 changes `AtomicRmw` twice.

**B. Break once, stage the semantics.** One breaking release changes the *shapes* of everything in S1, S3, S4 and
S5 together: `AtomicRmw` gains `old`, `order` and `scope`; `AtomicCas` and `Fence` are added; `Barrier` takes a
non-exhaustive `kind`. `check_kernel` refuses every form whose semantics has not landed (typed
`AtomicUnsupported` / `BarrierUnsupported`), and the interpreter implements only `Relaxed`, `Device`, no old
value, as today. Each later slice lifts a refusal and is **additive**: S1 first (old value by the serialization
rule), then S2, S3, S4/S5, S6. A downstream `match` is touched once.

**C. Memory model first, then everything.** Write the happens-before model (S4, S5) as its own note and ruling, then
build all of it. Cleanest model, slowest first consumer.

## Recommendation

**B**, with S1 and S2 as the first slice that lifts a refusal, and bonefish's `sycl::atomic_ref` path plus the
cpu-c thread-loop reference as its two consumers. Reasons: it breaks `Stmt` consumers once instead of up to five
times; it lets an emitter lane write its `lower_kernel` against the final shape while the semantics arrive; and the
refusals are typed, so an emitter that sees a form it cannot spell declines in-band, as the plan gate does. B does
not decide the memory model: it only fixes the field names.

What it costs: one breaking release (0.41.0) that is mostly refusals, and the discipline that a refusal is lifted
only with its oracle test and its mutants, as every slice here has been.

## Needs a ruling

1. B (break once, stage semantics) versus A (slice by slice). I recommend B.
2. Whether the first consumer is bonefish's SYCL path, the cpu-c thread-loop reference, or both. I recommend both:
   the reference gives the differential, bonefish gives a real spelling.
3. The old-value rule (S1): "some serialization, ascending in the oracle, properties asserted" as for the float
   scatter band, or a fixed ascending order promised by the IR. I recommend the unspecified-order rule, because a
   promised order is false on a device.
