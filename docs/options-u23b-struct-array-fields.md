# Options: struct parameters with array fields (U23b, census slice 5)

Status: **option A implemented (branch `lane/u19-templates`; pm-kernels approved 2026-11-22)**; see *Result* at the end. Originally: options note for pm-kernels' ruling, no code (`[TASK]` 2026-11-22). Source: `docs/options-u19-census.md`, which stopped **~67 of Baracuda's 535
kernel definitions** (heuristically instantiated) on "the field `shape.v` of a struct parameter: only scalar fields and nested structs of them are allowed".

## The real shape

Baracuda's strided kernels take `DimsI32 shape, DimsI64 stride_a, DimsI64 stride_y ...`:

    inline constexpr int MAX_RANK = 8;
    struct DimsI32 { int32_t v[MAX_RANK]; };   struct DimsI64 { int64_t v[MAX_RANK]; };
    for (int d = rank - 1; d >= 0; --d) { int32_t s = shape.v[d]; ... off_a += coord * stride_a.v[d]; ... }

Three facts that decide the options: the array is **small and fixed** (8); it is read **only** (a `const` by-value `__grid_constant__`-style parameter);
and it is indexed by a **runtime** value (`d`, bounded by the runtime `rank`), so naming each element is not enough.

What the IR has today: `ParamKind::Struct { fields }` of `StructField { name, ty, local }`: **one scalar per field, each its own local**, never assigned, no index
(`kernel.rs`). `ParamsBlock::of_struct` lays a struct out by the natural C rule (each field aligned to its size, the block to its widest field); an `int32_t v[8]`
is eight consecutive `int32_t`, which that rule produces for eight scalar fields **byte for byte**. `ScalarExpr` is **exhaustive on purpose** (the comment at its definition:
a new node must break every lowering, oracle arm and optimiser rule at compile time).

## Options

**A. Walker only, no IR change (recommended first).** The walker flattens an array field into `N` scalar fields (`shape.v[0]` ... `shape.v[7]`, dotted names, the
existing `ParamKind::Struct`) and lowers `shape.v[e]` as a `Select` ladder on `e`: a constant index is the scalar itself, a runtime index is
`Select(e == 0, v0, Select(e == 1, v1, ... v7))`, the last arm `0` for an index outside `0..N` (undefined in C; a total result is a refinement, as the integer rules do).
- **Oracle, checker, `unpopped-cpu-c`:** nothing to change; they already run `Select` and struct parameters (`KernelArg::Struct` takes the flattened values in order).
- **`params_block`:** `of_struct` over the flattened fields equals the C layout of the array byte for byte (consecutive, same alignment), so a launcher that fills the C
  struct fills the block.
- **Cost:** 8 `Select` nodes per access for N = 8 (`param_array_read` starts from the `0` base and wraps once per element, so the last arm is `0`, not `v7`), and emitters see ladders, not an index. **Correctness is exact**; speed is not the oracle's concern, and an emitter that wants
  the index back can pattern-match the ladder or wait for B. Eight elements, four reads per loop trip in Baracuda's strided kernels.
- **Unlocks:** the ~67 kernels stopped here (each then meets the next layer: re-measure).

**B. IR: an indexed read of a struct-parameter array.** `StructField` gains an extent (`extent: u32`, default 1, additive on a `#[non_exhaustive]` struct; the field's `N`
locals become consecutive or a `Vec<u32>`), plus a way to read element `e`: **a new `ScalarExpr` node** (for example `ParamIndex { param, field, index }`).
- **This is the stop condition: `ScalarExpr` is exhaustive, so the new node breaks every `match` in `unpopped`, `unpopped-cpu-c`, `baracuda-cuda-emit`,
  `fishstringer-slang-emit`, `vulkane-*`, `rockfish-hip-emit` and `openclownfish-clc` at compile time, by design.** A breaking change (0.n.0 under the versioning rule), with
  every emitter lane's work attached.
- Oracle: bounds-checked, a typed trap for an index outside the extent (stronger than A's total `0`). Checker: the index is an integer, the field has an extent.
  cpu-c / CUDA / Slang: `p.shape.v[i]`, natural. Layout: `of_struct` handles an extent (aligned to the element, `extent` times its size).
- **Buys:** the emitter spells the index directly; the oracle can trap an out-of-range index. **Costs:** an exhaustive-enum break across the portfolio, for a
  read that A already expresses exactly.

**C. A `Constant` allocation initialised from the argument.** `AllocDecl` has no initializer and the oracle refuses to read a `Constant` allocation
(`kernel_mem_ops`: "AllocDecl has no initializer"), so this needs an initializer first. Larger than B and no closer to the need. Not recommended.

**D. Pass the struct as a read-only buffer by address (`Place::Raw` loads).** Changes the launch ABI (a pointer to the struct instead of its fields as arguments) and
asks every launcher to change. Not recommended.

## Recommendation

**A now**, as a walker slice (the census order's slice 5 then needs no ruling beyond this note): it unlocks the kernels at no cost to any other lane, the layout is
exact, and nothing about it forecloses B. Take **B only if an emitter lane reports the ladder is a problem**, as its own `[ASK]` with those lanes in the loop,
because it is the one that touches the exhaustive enum.

Out of scope here: a struct array field **of structs** (`Cell v[4]`), a pointer or bit field, and a **non-const** array (an assignable array field is a local array, which the
walker already lowers as a `Private` allocation, not a parameter).

**Patent guard:** an array extent and a read are layout and access; neither adds a thread-set, memory-kind or copy-count annotation to the struct
(`docs/PATENT-SWEEP.md`, U23; `kernel_interface::layout_only_guard`). Option A adds nothing to the IR at all.


## Result (2026-11-22)

Option A as written: an array field of a struct parameter flattens to `N` scalar fields (`shape.v[0]` .. `shape.v[N-1]`, the one `ParamKind::Struct`), a constant index is the element and a runtime index a `Select` ladder. **Decisions recorded:** an index outside `0..N` (constant or not, negative, or a huge unsigned) reads **`0`**; a kernel that really indexes out of range reads whatever memory held under C and `0` through this lift, so a consumer sees a difference exactly there and nowhere else, and the lift never traps. Arrays above **64** elements decline (`NotYetLifted`, "at most 64"), as do arrays of structs and assignment to an element. The layout equals the C array's byte for byte (`ParamsBlock::of_struct` over eight `int64_t`: offsets 0, 8 .. 56, size 64, align 8). No IR change; option B stays held unless an emitter measures the ladder as a cost. Tests: `tests/kernel_body_param_arrays.rs`. **One thing the note missed:** Baracuda's strided kernels iterate `for (int d = rank - 1; d >= 0; --d)`, which the walker did not lift (counted loops counted up only), so the slice would not have unlocked them; counted loops may now count down (`tests/kernel_body_descending_for.rs`). The census has not been re-run on this branch, so how many of the ~67 kernels now lift is **not measured**.

— lane: unpopped
