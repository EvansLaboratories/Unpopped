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
the extent, which is a value here: deferred until a row needs it, refused until then).
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

— lane: unpopped-leaves
