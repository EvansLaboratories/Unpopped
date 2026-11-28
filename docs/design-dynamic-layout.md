# Design: a layout whose modes may be kernel scalar parameters (CuTe 3 and 66)

Written 2026-11-27 by `unpopped-leaves` at pm-kernels' [TASK] `dynamic-layout-design-note`. **Design only: no code, no `kernel.rs` edit.**
Anything not read at a named source is marked UNVERIFIED. Row sources: `docs/ir-union-evidence/rescore-2026-11-25/cute.md` rows 3 and
66 ("no `Param` mode in `Layout`"), `docs/ir-union-rulings.md` option C ("an allocation may carry a `Layout` with dynamic modes").

## 1. What exists, read at `origin/main`

- `Layout` (`unpopped-vocab/src/layout_algebra`): every mode is a static `i64`; `"(m,n):(lda,1)".parse::<Layout>()` is an error.
- `AllocDecl.layout: Option<AllocLayout { layout: Layout, swizzle }>` (T1, #479) and `check_alloc_layout`
  (`kernel_check.rs`), which **enumerates** the layout to prove it injective and inside the extent, capped at 4 Mi elements
  (`LAYOUT_ENUMERATION_CAP`), and refuses a `Param` extent when a layout is present.
- `Extent::Param(local)`: dynamic shared memory sized by a scalar parameter, fixed per launch. `GridDim::Param` likewise.
- The recipe that works today needs no type: index arithmetic on scalar parameters (`kernel_c_recipes_tiled.rs`,
  `cute_3_66_dynamic_extents_and_leading_dimension`, on `lane/conf-c-recipes-2`).

## 2. The decision: what is dynamic

**Shape and stride both**, and nothing else. The two real cases are a runtime extent (`m`, `n`) and a runtime leading dimension
(`lda`, a stride). The swizzle stays static (its bits are a hardware pattern); the rank and the profile (the nesting of the
tuple) stay static, so the algebra's structure never depends on a value.

```
DynLayout { shape: DynTuple, stride: DynTuple }        // same profile
Dim = Static(i64) | Param(u32)                          // Param(local): an integer scalar parameter, as Extent::Param
DynLayout::bind(&[(u32 local, i64 value)]) -> Result<Layout, LayoutError>   // every Param replaced: a static Layout
```

`bind` is the whole bridge: a dynamic layout **is** a family of static layouts indexed by the launch's scalar arguments. That keeps
T1's `Layout`, `check_alloc_layout` and the T1b tile/partition functions untouched (point 3) and gives them a dynamic input only
after a launch has fixed the values.

## 3. How the range and injectivity are proved without enumerating at check time

**Not symbolically, and not at `check_kernel`.** A symbolic bound is polynomial in the parameters (`(m-1)*lda + n-1`) and the
injectivity condition (`lda >= n` for row-major) is a relation between parameters; a sound symbolic prover is a new
decision procedure, which is not worth building for two use cases. The proof is a **launch-time check**, in two parts:

1. **`check_kernel` (static, cheap):** every `Param(local)` names an integer scalar parameter of the kernel; every `Static` dim is
   positive for a shape and non-negative for a stride; shape and stride have the same profile; the allocation carries
   `dyn_layout` or `layout`, never both; the space rule of T1 (a layout only on `Shared`) applies unchanged; a `DynLayout` allocation's
   extent is `Static(n)` (the launch checks `size == n`) or `Param(e)` (the launch checks `size == value of e`).
2. **At the launch, before any thread runs (the oracle; a host helper for emitters):** `bind` the scalar arguments, then run **the
   existing enumerating checker on the bound static `Layout`** (size equals the extent, cosize inside it, injective, the same
   `LAYOUT_ENUMERATION_CAP`). A failure is a typed trap or error naming the rule: `KernelTrap::DynLayout { why }` in the oracle
   (new variant, `non_exhaustive`), `LayoutError` from the host helper. Nothing is checked twice and nothing new is trusted.

## 4. The refusals (never silent over-reach: the `local_tile` lesson, #486)

Each has its accepted control beside it in the tests. Refused by name: a `Param` that is not an integer scalar; a negative or
zero bound extent; a bound shape whose size is not the extent; a bound layout that is not injective or whose cosize exceeds the
extent; a bound layout over the enumeration cap ("too large to prove injective", as T1's checker says); both `layout` and
`dyn_layout` on one allocation; a `DynLayout` on a non-`Shared` space (as T1); **a swizzle on a `DynLayout`** (the period must divide
the extent, which is a value here): **refused until section 8, which makes it a launch-time check**).
The layout algebra is **not** applied to an unbound `DynLayout`: `DynLayout` has no `composition`, `zipped_divide` or `slice`; a
caller binds first and gets T1b's `NotDivisible` refusal on a non-dividing tile.

## 5. What each consumer does

- **Oracle:** binds at launch (it has the arguments), runs the check of point 3, then maps a logical `Place::Alloc` index through the
  bound `Layout` exactly as it does for T1's static layout. A violation traps before the first statement.
- **cpu-c:** declines a kernel with a `dyn_layout` by name at first (`KernelError`-style refusal text naming `dyn_layout`), as it
  declines `spec_consts`; emitting the offset as C arithmetic over the parameters is the follow-up row, and the launch-time check
  is then the host helper, which the cpu-c runner calls before it invokes the function.
- **Other emitters:** unchanged until they opt in; a `non_exhaustive` field they do not read is a decline by name.

## 6. Additive, and what it flips

- **Additive:** a new type in `unpopped-vocab`; `AllocDecl.dyn_layout: Option<DynLayout>` with `#[serde(default,
  skip_serializing_if = "Option::is_none")]` (a kernel without one serializes as before, so the revision hash is unchanged); one new
  `non_exhaustive` `KernelTrap` variant. No `Stmt`, `ScalarExpr` or `ElementKind` change, no change to `Layout`.
- **Red-first record:** conformance's `declines_a_dynamic_mode_in_the_layout_algebra_by_name`
  (`crates/unpopped/tests/kernel_c_recipes_tiled.rs`, `lane/conf-c-recipes-2` 6a3ad78c) asserts that `"(m,n):(lda,1)"` does not
  parse as a `Layout` and that the static `(5,6):(8,1)` does. **The first half stays true** (the static `Layout` still refuses a symbolic
  mode, by design); the build flips the finding: a new test beside it parses the same text as a `DynLayout`, binds `(m,n,lda) = (5,6,8)`,
  and gets `(5,6):(8,1)`, and a bind with `lda = 3` (not injective) is refused by name.
- **Mutants to carry:** `bind` substituting the wrong parameter (swap `m` and `n`); skipping the launch-time injectivity run.

## 7. Patent

This adds a layout type and a launch-time check and **no statement**; no patent read is claimed for it.

## 8. Addendum 2026-11-28: a swizzle on a `DynLayout` (a row now, at pm-kernels' [TASK] `next-rows`)

Section 4 refused a swizzle on a `DynLayout` "until a row needs it". That reading lets the consumer decide whether the row exists; the row exists.
Design only in this section; the build follows pm-kernels' go on the patent grep (below).

**The rule that was static becomes a launch-time check.** For a static `Layout`, `check_alloc_layout` requires the swizzle's period (in
elements, `1 << (base + |shift| + bits)` of the element-form swizzle) to **divide the extent**, so a swizzled offset cannot leave it. With a
`DynLayout` the extent is a value (`Static(n)` or the argument of `Extent::Param`), so the same check runs at the launch on the bound value, in
`bind_dyn_layouts`, **after** the size/range/injectivity run and before the first statement:

| rule | where | refusal |
|---|---|---|
| swizzle base is at least the element width (and the width is a power of two) | `check_kernel` (needs no value) | `KernelError::AllocLayout` "the swizzle base is below the element width ..." (as T1) |
| `align_bytes` is stated and a multiple of the swizzle's byte period | `check_kernel` (needs no value) | `KernelError::AllocLayout` (as T1) |
| the period divides the **bound** extent | launch (oracle: `bind_dyn_layouts`; C: the launch check, status 4) | `KernelTrap::DynLayout { why: "the swizzle period does not divide the extent, so a swizzled offset could leave it" }` |

**When it does not divide, the launch refuses by name; nothing is rounded, clamped or run.** (The `local_tile` lesson, #486: no silent over-reach.)
Accepted control beside it in the tests: `swz(2,2,3)` on 4-byte elements has a 128-byte period (32 elements); an extent of 64 binds and runs
(the reversal kernel of `tests/kernel_dyn_layout.rs` computes the same bytes as the unswizzled layout), an extent of 48 traps.

**Representation (additive).** `DynLayout` gains `swizzle: Option<Swizzle>` (a byte swizzle, as `AllocLayout`), set by `DynLayout::with_swizzle`; its text is
the canonical `swz(B,M,S)(shape:stride)` of `AllocLayout`, so an existing `dyn_layout` text is unchanged and a kernel without a swizzle
serializes as before. `DynLayout::bind` still returns the static `Layout` (the algebra never sees a swizzle); the oracle builds
`AllocLayout::new(bound).with_swizzle(sw)` and the existing `physical` does the rest. No new `AllocDecl` field, no `Stmt`, `ScalarExpr` or `ElementKind` change.

**Emitters.** The oracle as above. cpu-c: the C offset gains the swizzle's XOR on the element-form offset, and its launch check gains the
period-divides-extent test (status 4), checked against the oracle over the same sweep; a swizzle in a kernel that bypasses the layout is
refused by name as a `dyn_layout` already is. The serial C does not place memory, so cpu-c accepts the `align_bytes` a swizzled `dyn_layout` requires (a swizzle changes which element holds a value, not what the value is). The swizzle's exact shift constant is not observable through logical accesses (any bijection gives the same bytes, as for the static swizzle), so the mutants that matter break the bijection or the period rule.

**Mutants to carry:** the divisibility test dropped (an extent of 48 must trap); the swizzle applied in bytes instead of elements; the
`align_bytes` rule dropped.

**Patent.** Cluster ruling by pm-kernels (2026-11-28, findings not advice, not a clearance, no attorney), read at origin/main: `docs/PATENT-SWEEP.md`
U6 addendum (lines 86-104): US20230289304A1, US12141082B2, US20240176663A1 and US20210064987A1 are each **outside as read, findings not advice, not a
clearance** (US12141082B2: the PM read all five independent claims; they need memory-access hardware circuits transferring data. US20240176663A1: tensor
acceleration circuits and tensor maps in a cache, low-medium confidence, current claims unread: **re-read when it grants**, a task for the patents lane,
on the board row and not this lane's). U7 re-read (lines 680-696): swizzle appears 0 times in US12204897B2 and US12710961B2; US20230289398A1 has two
"descriptor" hits in a work-distributor claim, not a swizzle. The static XOR `Swizzle` already ships (published prior art, CuTe). A swizzle on a `DynLayout`
adds a type and a launch-time divisibility check: no hardware, no asynchronous transfer, no statement. **Conditions, kept true in the code:**
(1) the swizzle stays XOR-bit data (`Swizzle { bits, base, shift }`): no lookup-table (8x8) encoding of the pattern, no tensor-map or descriptor object, no
cache of maps; (2) no emitted statement performs a swizzled copy or transfer: it is an offset function only (an XOR on the element offset). If a design
needs either (1) or (2), stop and ask pm-kernels.

## 9. Addendum 2026-11-28: a layout (static or dynamic) on a `Private` allocation; what a `Global` one is (pm-kernels' [TASK] `next-rows`)

Sections 3 and 4 and T1 put a layout on a `Shared` allocation only ("the other spaces keep the identity until a row asks"). The row asks. Design only in this section.

| space | layout (static or `dyn_layout`) | why |
|---|---|---|
| `Shared` | accepted (T1, section 3) | one array per block |
| **`Private`** | **accepted without a swizzle** | one array **per thread** (`extent` elements each, lifetime the kernel); the layout maps the thread's own index to its own physical element, so no two threads ever share a physical element and no race is possible; injectivity inside one thread's extent is what keeps two of its indices from aliasing |
| `Private` **with a swizzle** | **refused by name**: "a swizzle is a shared-memory bank pattern; a per-thread array has no banks, so this IR gives it no meaning" | not defined, not guessed |
| `Global` | **refused by name, already**: `KernelError::AllocSpace` (a kernel does not declare global memory); a global array is a **buffer parameter**, whose shape and strides are already run-time values of its argument | nothing to add: the dynamic-stride case is the buffer ABI, which exists |
| `Constant`, `Tensor`, `Tile` | refused by name ("a layout is only on a Shared or Private allocation") | `Constant` has no initializer; `Tensor` and `Tile` are not addressable element arrays |

**What changes.** The space check of `check_alloc_layout` and `check_dyn_layout` accepts `Shared` or `Private` (the text of the refusal names both and says why `Global` is not one), and a swizzle on a `Private` allocation is refused with the reason above; everything else (size equals the extent, range, injectivity, the parameter rules, the launch-time bind) is unchanged because it is about one array of `extent` elements, and a `Private` array is that, once per thread. The oracle already keys storage and the race log on the physical element per thread, so it needs no change beyond the check. cpu-c: a `Private` array is `sh<i>[t * extent + offset]`, so the offset arithmetic replaces the bare index inside the thread's slice; the launch check is unchanged.

**Aliasing and lifetime.** A `Private` layout aliases nothing across threads by construction (separate storage) and nothing within a thread (injective); it lives and dies with the thread's run; nothing is carried between threads or blocks. A `Global` layout of a buffer is not introduced.

**Patent.** The same conditions as section 8 (pm-kernels' cluster ruling 2026-11-28, findings not advice, not a clearance, outside as read): XOR-bit data only (and this row **refuses** the swizzle on `Private`), no lookup-table encoding, no tensor-map or descriptor object, no cache of maps, no emitted statement that copies or transfers: an offset function only. No hardware and no statement is added. If a design needs any of those, stop and ask pm-kernels.

## 10. Addendum 2026-11-28: a layout (static or dynamic) on an allocation that is passed to a callee (pm-kernels' [TASK] `next-rows`, item 4)

Design only. A missing consumer is not a reason to exclude the row (EXPECTATIONS 6.31).

**What already holds in the IR and the oracle (measured, `tests/kernel_module_layout.rs`).** A callee never sees a layout: it declares a one-dimensional
buffer parameter (`alloc_fits`: rank 1) and reads and writes it by logical index; the entry's `CallArg::Alloc(a)` binds that parameter to the
allocation, and the oracle's `alloc_bound` turns the callee's access into the entry's `Place::Alloc` access, so **the allocation's layout, static or the
launch-bound `dyn_layout`, applies to the callee's accesses exactly as to the entry's**. The test fills an array through a callee and reads it
back through `Place::Alloc` over a non-identity layout (`(2,4):(4,1)` and `(?1,?2):(?2,1)` bound to `(2,4)`): the two agree only because both apply the layout; a bind the
launch refuses traps `KernelTrap::DynLayout` before the call. So the IR needs no change and no new `ParamKind`: **what crosses the call, in the IR, is
the logical index only**; the rank (1) and the profile of the layout are the allocation's and static; the extents and strides stay the entry's own parameters.

**The gap is cpu-c's C ABI.** The serial C passes a buffer as `(ptr, shape, stride, base)`, so a rank-1 allocation reaches a callee as `sh<i>, ash<i>, ast<i>, 0LL` (shape `{n}`,
stride `{1}`): an affine address `base + i * stride`. A layout's offset (colexicographic coordinates times strides, then a swizzle's XOR) is not affine, so
passing the bare pointer would **bypass the layout silently**. Today `dyn_layout::admit` refuses by name any kernel with a layout and a call (the `bypasses_the_layout` walker).

**Design (what to build, in order).**
1. **Layout parameters beside the buffer, only where needed.** At emit time, over the whole module, find every callee buffer parameter that some call site binds
   to a laid-out allocation (directly or through a nested `CallArg::Buffer` forward). Only those parameters get a companion argument
   `const unpopped_layout_params *lay<i>` (a small C struct of the flattened `shape[]`, `stride[]`, `modes`, and the element-form swizzle's `src` mask and `shift`; **XOR-bit data
   only**), `NULL` for an identity binding; every other parameter and every other callee is emitted exactly as today. The callee's access through such a parameter calls one
   inline offset routine over the layout parameters (the same arithmetic `dyn_layout::offset_lines` emits for the entry); a forward to a nested callee passes the layout parameters on.
2. **The layout parameters are built at the call site from the allocation**: static layouts from constants, a `dyn_layout`'s leaves from the entry's scalar parameters (the same C
   expressions `offset_lines` already spells). The launch-time proof (size, range, injectivity, period) stays once, at the top of the entry, exactly where it is.
3. **Tests (red first):** a module whose callee fills and another that mirrors an allocation under (a) a static layout, (b) a `dyn_layout` over an argument sweep including
   a refused bind (status 4), (c) a swizzled one, the C against the oracle; mutants: the layout parameters dropped (bytes change), one stride leaf of them wrong, the swizzle
   XOR in the layout-parameter path turned into an OR (breaks the bijection), a forward to a nested callee that passes `NULL`.

**Refusals by name that stay.** A `Private` allocation passed to a callee (already refused: one array per thread has no shared binding); a callee with allocations of its own
(`a callee with allocations`, unchanged); a layout in a kernel with a Device barrier, a peer access or several devices (unchanged); `Mma`/`MatrixLoad`/`MatrixStore` over a laid-out
array (they address the storage, not the layout; unchanged); a callee parameter of rank other than 1 bound to an allocation (`alloc_fits`, unchanged).

**Additive.** cpu-c only (the ABI of the generated C is internal to it); no `Stmt`, `ScalarExpr`, `ElementKind`, `ParamKind` or `CallArg` change; the oracle needs none.
**Patent.** The same conditions as section 8 (pm-kernels' cluster ruling 2026-11-28, findings not advice, not a clearance, outside as read): the layout parameters are a small struct of
integers and XOR-bit constants passed to an offset function: **no lookup table, no tensor-map or tensor-descriptor object of the hardware kind, no cache of maps, no emitted copy or transfer**.
If a design needs any of those, stop and ask pm-kernels. (They are named layout parameters, not a descriptor: a C struct of layout integers, **not** a tensor descriptor or TMA map, per pm-kernels' naming note.)

— lane: unpopped-leaves
