# U3 row 4: execution scopes, memory spaces and the per-thread driver (design note)

Status: **design only, no code.** Written because `u3-program-model-and-seam.md` sections 2.1-2.2 name the
shapes ("`Scope` enum, id leaves, `Alloc`, `Place::Ptr`") without fixing them, and the PM's task for this row
(2026-10-29) said to land a design note before code when the leaves are not already specified. Every choice
below marked **DECISION** is open to the PM's ruling; the rest follows from the spec and the code at
`origin/main` 4b5bedc (0.32.24).

Scope of the row: the kernel layer (`kernel.rs`, `ScalarExpr`) gains execution ids, a launch shape, memory
spaces with allocations, and a barrier; the interpreter gains a "run every thread" driver. **Out of scope:**
`ProgramModel`/`BlockSpmd` lowering, `Stmt::Call`, the resource map and `Frontend` change (row 3, held), any
emitter change (the seam hold stands: no new `Frontend`/`Backend` items).

## 1. What changes, and why it is one breaking increment

| Surface | Change | Breaking? |
|---|---|---|
| `ScalarExpr` | **one** new variant, `Exec(ExecId)` (section 2) | Yes: every exhaustive `match` on `ScalarExpr` (about 30 sites in `ir`, `types`, `optimize`, `plan`, `oracle`, `contract`, `cfamily`, `jit`, `pattern`, `backend`, `kernel`, `kernel_lift`, plus `DagNode`/`DagKey`) gets an arm. The count was taken from `grep CoordInt` over `crates/`, the nearest precedent leaf. |
| `Stmt` (exhaustive on purpose) | `Barrier { scope }` | Yes (kernel consumers only). |
| `KernelFn` (`#[non_exhaustive]`) | fields `launch: Option<LaunchShape>`, `allocs: Vec<AllocDecl>` | No. `KernelFn::new` keeps its signature and sets both empty. |
| `Place` (`#[non_exhaustive]`) | variant `Alloc { alloc, index }` | No. |
| `LocalDecl` (`#[non_exhaustive]`) | none (private is the only space for a scalar local; see 3.1) | No. |

All the breaking items land in **one** release, 0.33.0 (decision D8: one pre-1.0 minor). The new variant is
inert for every existing producer: no lift emits it, `plan::build_plan` and every `Backend` reject it with a
typed decline, so no existing op or emitter changes meaning.

**Why a single `Exec` leaf and not one variant per id.** Eight ids (`ThreadIdx`, `BlockIdx`, `BlockDim`,
`GridDim`, `LaneId`, `SubgroupId`, `SubgroupSize`, cluster ids) as eight variants is eight arms at each of the
~30 sites; as one leaf with an exhaustive payload enum it is one arm per site and the payload is matched in one
place (`types`, `oracle`). A new id later is then an additive change to a `#[non_exhaustive]` payload enum, not
another breaking release. **DECISION D-R4-1:** one leaf. The alternative (one variant per id) has no advantage
I can find except `match` ergonomics.

## 2. Execution ids and the launch shape

```rust
#[non_exhaustive] pub enum Scope { Thread, Subgroup, Block, Cluster, Device }   // Peer later

#[non_exhaustive] pub enum ExecId {
    ThreadIdx(u8),   // index of the thread in its block, axis 0..=2
    BlockIdx(u8),    // index of the block in the grid
    BlockDim(u8),    // extent of the block
    GridDim(u8),     // extent of the grid
    LaneId,          // index in the subgroup
    SubgroupId,      // index of the subgroup in the block
    SubgroupSize,    // lanes per subgroup (a range at plan time; a value in one run)
}
ScalarExpr::Exec(ExecId)        // type: U32 always
```

- **Type is `U32`** for every id. CUDA's `threadIdx.x` is `unsigned int`, Slang's `SV_GroupThreadID` is
  `uint3`, and the interpreter already wraps `U32` exactly (`ir-typed-expressions.md` section 8). A kernel that
  wants a signed index casts, explicitly, as everywhere else in the typed IR (no implicit promotion).
  **DECISION D-R4-2:** `U32`, not `I32`.
- `Exec` is a leaf like `Local`: opaque to the e-graph optimizer (never an `Input`, never const-folded, hashed
  by its payload), and rejected by `plan::build_plan` and every plan-reaching walker the way `Local` is.
- **`LaunchShape`** is a property of the kernel, not of a statement:

```rust
pub struct LaunchShape {
    pub block: [u32; 3],            // threads per block
    pub grid: [u32; 3],             // blocks per grid
    pub subgroup: Option<SubgroupSize>,   // None = the target decides; Some = the kernel requires it
}
pub struct SubgroupSize { pub min: u32, pub max: u32 }   // a range, never one number (Fuel, spec 2.5)
```

  `KernelFn::launch: Option<LaunchShape>`. `None` means "one logical thread", the whole of U2, unchanged.
  Cluster dimensions are **deferred** (nothing in the corpus the lane has measured needs them; adding a field
  to the `#[non_exhaustive]` struct later is additive).
- **Checker rules** (each a `KernelError` variant, each pinned by a known-bad mutation paired with the
  unmutated program passing, as in `tests/kernel_layer.rs`):
  - `ExecWithoutLaunch`: any `Exec` leaf in a kernel whose `launch` is `None`.
  - `ExecAxisOutOfRange`: axis > 2.
  - `BadLaunch`: a zero extent, or `subgroup.min > subgroup.max`, or `subgroup.min == 0`.
  - `SubgroupIdWithoutSubgroup`: `LaneId`, `SubgroupId` or `SubgroupSize` when `launch.subgroup` is `None`.
    (A kernel that reads the lane must say what it requires; this is Fuel's "subgroup size is not a constant".)

## 3. Memory spaces and allocations

### 3.1 No raw pointers in this row

The spec (2.2) sketched `Place::Ptr { base, offset }`. **DECISION D-R4-3: this row adds `Place::Alloc`, not
`Place::Ptr`.**

```rust
#[non_exhaustive] pub enum MemorySpace { Global, Shared, Private, Constant }   // Texture, Tensor later

pub struct AllocDecl { pub name: String, pub space: MemorySpace, pub elem: ElementKind, pub extent: Extent }
pub enum Extent { Static(u32), Param(u32) }        // Param(local): a scalar param, fixed per launch
Place::Alloc { alloc: u32, index: ScalarExpr }     // flat element index into KernelFn::allocs[alloc]
```

Reason: every array the corpus the lane has seen needs (Fuel's `groupshared` tiles, CUDA `__shared__`
buffers) is an array with a name and an extent, indexed by an integer. A pointer-typed *value* (stored in a
local, passed, offset, aliased) is a much larger commitment: it needs a pointer `Ty`, an aliasing story for
the effects model, and a rule for what `Place::Logical` means through one. None of the seven source languages
the union IR must cover needs a pointer value to express a shared-memory tile; CuTe and Triton do, and they
are not in this row. So: arrays by id now, pointer values when a consumer needs them, as an additive
`Place` variant (`Place` is `#[non_exhaustive]`). The cost of deferring is a possible second look at
`Place::Alloc` then; the cost of building `Ptr` now is guessing its aliasing rules with no consumer.

- `Place::Logical` is untouched and keeps meaning "through the operand's view" (U2 section 8, item 1). This
  row cannot break the lifting differential because it adds beside it.
- **`Global`** allocations are not declared here: global memory is the buffer *parameters*, as today. An
  `AllocDecl` in `Global` is rejected (`AllocSpace`), so there is one way to name a global buffer.
- **`Shared`**: one instance per block. `Extent::Static(n)` is compile-time; `Extent::Param(local)` is a
  launch-time scalar (CUDA dynamic shared memory), whose local must be a scalar parameter local of an
  integer type (checker: `AllocExtent`).
- **`Private`**: one instance per thread (a local array).
- **`Constant`**: read-only after launch; a `Store` to it is `WriteToConstant`.
- Rules: `Place::Alloc` index must be an integer expression; the allocation id must exist (`BadPlace`
  reused); a store's value must have the allocation's element type (`TypeMismatch`, reused). A `Load` from
  or `Store` to a `Shared` allocation in a kernel with no launch is `ExecWithoutLaunch`'s sibling,
  `SharedWithoutLaunch` (shared memory is per block; a block needs a launch).
- The oracle/interpreter bounds rule: an index outside the extent is `KernelTrap::OutOfBounds`, never wrap
  and never silent (the same stance as `ZeroStep`).

### 3.2 `LocalDecl` does not change

A scalar local is private by construction; the spec's "`space` on a `LocalDecl`" is therefore redundant with
`AllocDecl::space` for arrays. **DECISION D-R4-4:** leave `LocalDecl` as is. (If a fragment-typed local
arrives with `BlockSpmd`, that is its own increment.)

## 4. Barrier and the uniformity rule

Shared memory without a barrier cannot express a tile load followed by a read of a neighbour's element, which
is the reason the memory space exists. `Stmt::Barrier { scope: Scope }` with `scope` in `{Subgroup, Block}`
(`Thread`/`Device`/`Cluster` rejected: `BadBarrierScope`).

A barrier is only defined if every thread of the scope reaches it. **DECISION D-R4-5:** the checker enforces
a *conservative uniformity rule*: a `Barrier` may not appear under an `If` or `While` whose condition (or a
`For` whose bound) reads an `Exec` leaf, or a local assigned from one, or a `Load` from a `Shared`/`Private`/
global place. Everything else (a loop bounded by a scalar parameter or a constant, as in a tiled matmul) is
accepted. This is a taint analysis over the already-scoped walk (`BarrierDivergent`), conservative: it
rejects some correct programs (a data-dependent but in-fact-uniform trip count) and will never accept a
divergent barrier. The alternative (no rule, document "undefined if divergent") would let the interpreter
deadlock or silently run a program no GPU defines.

## 5. The per-thread driver (interpreter)

`run_kernel` stays as is for `launch == None`. A new public `run_launch(kernel, args)` runs every thread:

1. For each block of the grid, in row-major order, allocate its `Shared` instances (zeroed? **no**: filled
   with a poison pattern per dtype and the checker-visible rule "reading a shared element no thread stored is
   a `KernelTrap::UninitializedShared`"; a zero fill would hide the missing-barrier bug the memory space
   exists to catch).
2. Run the block's threads in **lockstep phases between barriers**: each thread is a resumable evaluation
   (an explicit frame stack in place of Rust recursion, because a barrier can sit inside a loop); a phase
   runs every live thread until it reaches a `Barrier` or `Return`; when all live threads of the scope are
   at the same barrier they all advance. A thread at a different barrier than its peers, or one that
   `Return`s while peers wait, is `KernelTrap::BarrierMismatch`. The uniformity rule (section 4) makes this
   rare, not impossible (a `Return` before a barrier is still allowed by the checker).
3. Global buffers are shared by all threads. **Two threads storing one global element in one phase is
   `KernelTrap::DataRace`**, not "later wins": v0's "later store wins" was sound only with one thread. Atomics
   (`AtomicAdd` and friends exist on the intent layer's scatter) are a separate statement, out of this row;
   until they land, a kernel that needs them cannot run in `run_launch`.
4. Subgroups are `launch.subgroup.min` lanes (the interpreter picks `min`; a test may pass another size
   inside the range through an argument to `run_launch`).

This is the largest part of the row and is sized as its own PR: it replaces the recursive evaluator with a
resumable one for the launch path only, leaving `run_kernel` and the lifting differential untouched.

## 6. Order of work (each its own PR; one is breaking)

| # | PR | Version | Contents |
|---|---|---|---|
| 0 | **this note** | patch, PM allocates | docs only |
| 1 | types, leaf, checker | **0.33.0** (breaking) | `ScalarExpr::Exec`, `Stmt::Barrier`, `KernelFn::{launch, allocs}`, `Place::Alloc`, the section 2-4 checker rules each with a known-bad mutation and positive control; the new arms at every exhaustive site, each declining or rejecting the new node; `cargo clippy --workspace --all-targets --all-features` (the lesson from earlier variants: `unpopped-slang`/`cpu-c` have matches `-p unpopped` does not see). The interpreter traps `Unsupported` on the new nodes. |
| 2 | `Exec` ids and shared memory in `run_launch`, no barriers | 0.33.x additive | the driver of section 5 steps 1, 3, 4 for barrier-free kernels |
| 3 | `Barrier` in `run_launch` | 0.33.x additive | the resumable evaluator, `BarrierMismatch`, the `DataRace` check |

PR 1 deliberately lands types the interpreter cannot yet run, because the breaking change is the expensive
thing to batch and the adopters (Baracuda, Fuel) re-vendor once. The honest cost: for one release a kernel can
*check* and not *run*; the `Unsupported` trap says so by name.

## 7. Release gates attached to this row

- **Patent hits 19 and 20** (`PATENT-SWEEP.md`, "not covered: U3 not built"): the independent claims of
  US10726606, USPTO 12412230 (variant selection) and WO2013070636A1, US20100122264A1 (address spaces) are
  read before PR 1 merges to a release. This lane's earlier sweeps read abstracts and summaries, not claim
  text; this one reads claims. Verdict goes in `PATENT-SWEEP.md` in PR 1.
- **Adopters** (Baracuda, Fuel) are told at PR 1's [READY]; they re-vendor for the breaking variant.
- **No `Frontend`/`Backend` trait item** is added by any PR above; the seam hold is respected.

## 8. Not decided here (named so they are not read as decisions)

- `BlockSpmd` and tile-typed locals (`Ty` gains a tile type): U3 row after this.
- Pointer values and `Place::Ptr` (D-R4-3 defers them).
- Atomics as kernel statements, cluster and distributed-shared memory, texture/tensor spaces.
- Cooperative matrix (Fuel R10): UNVERIFIED whether it fits `CapabilitySet` + MMA atoms or needs its own seam.
