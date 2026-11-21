# Options: pointer values for kernels (ROADMAP U9c, level (b))

*2026-11-21, unpopped lane. Scope set by pm-kernels (TASK 2026-11-21); the consumer is Baracuda's CUDA frontend
(baracuda#356, its G3). Ack from pm-kernels before any code.*

## What the consumer needs (measured by the Baracuda lane, 2026-11-21, read-only grep, not a full walk)

FlashInfer paged decode (`decode.cuh`, `cp_async.cuh` at baracuda `57e7294ff`): params are base buffers (`decode.cuh:221-228`);
every access is `base + expr` (`:282-286`); shared memory is one dynamic buffer cast to typed views at byte offsets
(`:244-245`, `:433-434`); `sizeof(DTypeKV*)` (`:783`) is host-side sizing only. **No pointer-typed parameter or local was
found.** Three leftovers: (1) nullable pointers: `q_rope_offset = nullptr` reassigned conditionally (`:446-450`),
`lse != nullptr` (`:365`, `:601`), `rope_pos_offset == nullptr` (`:542`); (2) `vec_t::load/store(ptr)` and
`paged_kv.protective_get_*` bodies, **not read**; (3) `cp.async` takes a 32-bit shared address and a global pointer plus size,
predicate and fill mode (`cp_async.cuh:77-80`); `griddepcontrol` asm has no operands (`:461,:469,:606`).

## Finding that changes the cost: `Place` is `#[non_exhaustive]`, and every memory statement takes a `Place`

`Load`, `Store`, `MaskedLoad`, `MaskedStore`, `AtomicRmw`, `AtomicCas` and `CopyAsync` (`kernel.rs`) all carry a `Place`. A new
`Place` variant therefore adds raw byte addressing to all of them **without adding one `Stmt` variant**. `Stmt` and
`ScalarExpr` are exhaustive; `Place`, `ParamKind`, `KernelArg` are not. So "RawLoad/RawStore" of `docs/options-u3-remainder.md` R2
is `Stmt::Load`/`Stmt::Store` over a new place, not two new statements.

## Option B1 (recommended): two new `Place` variants

```text
Place::Raw      { buffer: u32, byte_offset: ScalarExpr, elem: ElementKind }
Place::AllocRaw { alloc: u32,  byte_offset: ScalarExpr, elem: ElementKind }
```

* `Raw` names the `elem`-typed value at `byte_offset` bytes into the storage of buffer parameter `buffer`; `AllocRaw` the same
  into an allocation (the dynamic-shared carve of `:244-245`). `elem` need not equal the buffer's declared element type: that is
  the type pun a `(T*)` cast performs. It must be a scalar type with a defined size (not a sub-byte or untyped kind: typed refusal).
* **Rules** (each a typed trap, never UB): the offset is an integer expression; `byte_offset % size_of(elem) == 0`
  (misaligned access traps); `[offset, offset + size)` lies inside the buffer's bytes (the bound argument's byte length; for an
  allocation, its byte size, `DynamicShared` resolved at launch); the buffer's `BufferRole` gates read/write as for `Logical`;
  a view is **ignored** (raw is bytes of the storage, not "through the view", and `Param::rank` is not consulted).
* **Races and bounds** are recorded in bytes (`[offset, offset + size)`); today's `note_global` records elements, so the oracle's
  race detector needs a byte-range overlap for `Raw` (an element access and a raw access to the same bytes must conflict).
* **`CopyAsync`** takes `Raw` as `src` and `AllocRaw` as `dst` (`elems` counts `elem`-typed units; bytes = `elems * size_of(elem)`,
  which must be 4, 8 or 16 as the PTX form requires). That covers P4 with no new statement.
* **cpu-c**: `memcpy` through `unsigned char*` at the byte offset (no aliasing UB, no alignment assumption beyond the checked
  one). Other emitters decline `Raw` by name until taught (the typed-decline discipline of U4 option B).
* **Break shape: additive** for downstream crates (`Place` is non-exhaustive). In this workspace every exhaustive `match` on `Place`
  gains an arm: **4** in the event (checker, uniformity; the first estimate here, "about 25", was wrong: it counted `Place::Alloc`
  mentions, not exhaustive matches); `cargo check --workspace` finds them all. The race detector is **conservative at element
  granularity** (two threads writing different bytes of one wider element are reported); built and pinned in slice 1.
  **Built 2026-11-21:** slice 1 `Place::Raw`, slice 2 `Place::AllocRaw` and a `Raw` source for `CopyAsync` (a raw or `AllocRaw`
  *destination* of a copy is declined by name: landing a pun in the async queue is not modelled). Slice 3: cpu-c declines both by
  name; lowering needs the storage length in the C ABI first (see the ROADMAP row). **Exact new `Stmt` variants: none. New `ScalarExpr` nodes: none.** pm-assist stamps a minor bump.
* **Not built here**: pointer-typed `Ty`, a pointer as a parameter, local, return or struct field, pointer comparison and
  arithmetic as expressions, aliasing analysis. That is option C and stays deferred (D-R4-3).

## The nullable-pointer idiom (leftover 1): three ways

* **A (recommended): no IR change.** The lift turns `T* p = nullptr` into a buffer parameter that is always bound (the host passes
  an empty or dummy buffer) **plus a scalar `U32` flag parameter**, and each `p != nullptr` into `flag != 0`; an access under the
  guard is a `Stmt::If` on the flag (or `MaskedLoad`/`MaskedStore` with the flag in the mask, which already skips the bounds
  check and the race record when false). Additive; the cost is the lift rewriting the idiom and the launcher passing the flag.
* **B: `ScalarExpr::BufferPresent(buffer)`.** Cleaner for the consumer, **BREAKING** (`ScalarExpr` is exhaustive: every lowering,
  oracle arm and optimizer rule must answer), and needs an optional-buffer parameter. Not recommended for one idiom.
* **C: a null value of a pointer type.** Needs option C above. Deferred.

## What this does not settle

* **Vector accesses** (`vec_t::load`, 128-bit `float4`-style): body unread. The lift can emit `vec_size` consecutive scalar `Raw`
  loads (what `CopyAsync` does for `elems`), which needs nothing more. A single 128-bit load into a **packed local** is the U15
  scalarize gap (the interpreter traps `Unsupported("packed value")`; `tests/kernel_packed.rs`): scalarize is revived only if
  an emitter must keep the vector as one access, and that also meets the U15 patent hold (lowering to a vector load or store).
* **Patent read** (owed before code, per the portfolio rule): raw byte addressing is the CUDA/PTX `(T*)` cast and `ld`/`st`
  with a byte address, public-domain mechanism, but no claims have been read; `docs/PATENT-SWEEP.md` gets a U9c section in the
  code PR, with the U23 guard (layout only, no thread-set or memory-kind annotation) kept.
* **Host side**: `KernelInterface` needs no field for `Raw` (buffers bind by position); the flag parameter of option A is an
  ordinary scalar parameter.

## Slices, if acked

1. `Place::Raw` in `kernel.rs`, `check_kernel` rules, oracle load/store/atomics with byte bounds and races, oracle tests (red
   first; asymmetric data so a wrong offset or a type pun shows; seeded mutants for alignment, bounds and role).
2. `Place::AllocRaw`, and `CopyAsync` with both.
3. cpu-c lowering and `kernel_*_emit_agrees` differentials; the other emitters' typed decline.
4. `docs/PATENT-SWEEP.md` U9c section; ROADMAP U9c row; changelog fragment.
