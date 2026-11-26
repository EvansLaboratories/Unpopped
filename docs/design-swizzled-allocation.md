# Design: a layout and a swizzle on an allocation (T1, the `Swizzle` kernel consumer)

Written 2026-11-26 by `unpopped-leaves` at `unpopped`'s answer (option b): this note states what the `kernel.rs` and
`kernel_check` edit must do; **the edit is the primary's and rides T1** of `docs/design-tile-and-schedule-plan.md`
(section 4, "T1 `TileTy` and a layout field on allocations (serde default), checker rules"). Nothing here edits those files.
`cute.md` row 15 (CuTe `Swizzle` / `ComposedLayout`): both exist in `crates/unpopped-vocab/src/layout_algebra/swizzle.rs`
and have no `AllocDecl` or kernel consumer. Anything not read at a named source is marked **UNVERIFIED**.

## 1. What exists, read at `origin/main`

- `Swizzle::new(bits, base, shift)`, `apply(x)`: XOR of the `bits` bits at `base + max(shift,0)` into the `bits` bits `|shift|`
  positions away; a bijection on non-negative integers and its own inverse (requires `|shift| >= bits`).
  `Swizzle::new(3, 4, 3)` is the PTX 128-byte pattern: bits `[7,10)` XORed into bits `[4,7)` of the **byte** address
  (`docs/design-wgmma-descriptors.md` section 5c, read from PTX ISA 9.5 Figures 159-168; `Swizzle<B,4,3>`, `B` = 1, 2, 3 for
  the 32, 64, 128-byte modes, same document line 19).
- `ComposedLayout { swizzle, layout }`: `i -> swizzle(layout(i))`; canonical text `swz(B,M,S)(shape:stride)`
  (`tests/layout_text_round_trip.rs`), which is the wire form below.
- `AllocDecl { name, space, elem, extent }` (`exec.rs`, `non_exhaustive`, built by `AllocDecl::new`); `Place::Alloc { alloc,
  index }` names a **flat element index**; `Place::AllocRaw` names a **byte offset** into the allocation's own storage.

## 2. The field

```
AllocDecl { ..., layout: Option<AllocLayout> }          // #[serde(default, skip_serializing_if = "Option::is_none")]
AllocLayout { layout: Layout, swizzle: Option<Swizzle> }   // serde as the canonical text, one string
```

- Additive: `None` is today's meaning (identity: logical index = element offset), so every existing kernel and JSON is
  unchanged. `AllocDecl::new` keeps its signature; add `with_layout(AllocLayout)`.
- `layout` maps a **logical index** (the `index` of `Place::Alloc`) to an **element offset** in the allocation. `swizzle`
  then permutes that offset. The two together are a `ComposedLayout` over *element* units.
- **Units.** The hardware swizzle is on the byte address; a layout is in elements. For an element of `w = size_of(elem)`
  bytes with `w = 2^k`, a byte swizzle `Swizzle<B,M,S>` is the element swizzle `Swizzle<B,M-k,S>` provided `M >= k`; if
  `M < k` or `w` is not a power of two the field is **refused by name** (`swizzle base below the element width`). The
  field stores the **byte** form (`swz(3,4,3)` means what the PTX figure means for every element type); the element form is
  derived. CuTe does this with an `upcast` of the layout atom (**UNVERIFIED** against `cute/layout.hpp` and
  `cute/swizzle.hpp`, not read). The derivation is a pure vocabulary function, `Swizzle::in_elements(width_bytes)`, which this
  lane can add to `unpopped-vocab` with its tests ahead of T1 (say so and it is the next PR).

## 3. Checker rules (for `kernel_check`, the primary's)

Each rule has an accepted control beside its refusal in the tests.

1. **Space.** A swizzle is accepted only on `Shared` (the bank pattern it exists for); a layout without a swizzle on any
   space `AllocDecl` accepts today. A layout on `Constant` or `Global` is refused (those are not decided here: **UNVERIFIED**,
   `unpopped` to confirm).
2. **Size.** `layout.size()` equals the allocation extent. A `Param` extent (dynamic shared memory) is **refused** with a
   layout: the size is not known at check time.
3. **Range.** `layout.cosize() <= extent`, and for a swizzle the **swizzled** offsets stay below `extent`: it holds when the
   byte extent is a multiple of `2^(base + |shift| + bits)` (the highest bit the XOR can touch), which the checker states
   instead of enumerating. A non-multiple is `SwizzleOutOfRange`.
4. **Injective.** No two logical indices share an offset (a stride-0 mode of extent greater than 1 aliases; `filter_zeros` of
   `Layout`, merged in #421, finds them). Brute force over `size()` is acceptable for a static extent, capped (a large
   extent uses the structural test `complement` already applies); the swizzle is a bijection, so injectivity is the
   layout's alone.
5. **Alignment.** A swizzle's pattern repeats every `2^(base + |shift| + bits)` bytes (1024 for the 128-byte mode) and the
   pattern is anchored at **address** zero, not at the allocation: the allocation's base must be a multiple of that.
   `AllocDecl` has no alignment attribute today (grep `align` in `exec.rs`: none), so this note asks for one
   (`align_bytes: Option<u32>`, serde default) or states the oracle's rule below. This is the same alignment rule
   `docs/design-wgmma-descriptors.md` states for the descriptor's base offset.

## 4. The oracle's meaning

- A `Place::Alloc { alloc, index }` access reads or writes **physical element** `swizzle(layout(index))`. Storage is the
  physical array. The race detector and the initialised-element tracking key on the **physical** element, so two logical
  elements never alias (rule 4) and a swizzle never hides or creates a race.
- **A layout is a permutation of storage, so it changes no result of a kernel that only uses `Place::Alloc`.** That is the
  property the tests pin (section 5). It is not true of `Place::AllocRaw`: a byte offset names **physical** bytes, so a raw
  access sees the permutation. Stated, not hidden; whether `AllocRaw` on an allocation with a layout is refused or reads
  physical bytes is the primary's decision (recommend: reads physical bytes, since that is what the hardware does).
- A statement that names "the first element of this tile" (`MatrixLoad`, `MatrixStore`, `Wgmma`, `Tcgen05Mma`; `kernel.rs` lines 739, 819,
  866 to 868, 934 to 936 at `origin/main` 46d407cd) takes a **logical** `Place::Alloc`; how those operand layouts compose with an allocation layout is
  T2/T3 and is **not decided here** (`docs/design-wgmma-descriptors.md` sections 3 and 5c own the descriptor side).
- Bank conflicts are not modelled; the oracle says nothing about speed.

## 5. Property tests (to land with T1)

1. **Permutation invariance (the differential).** Take every existing `Shared`-allocation kernel test, give each allocation a
   random injective layout and a valid byte swizzle, and require the final buffers equal the unswizzled run's. Mutant: an
   oracle that applies the layout on stores but not loads (or the reverse) fails it.
2. **Layout is observable through `AllocRaw` only.** One kernel stores through `Place::Alloc` and reads through
   `AllocRaw`; the bytes must sit at `swizzle(layout(i))`, worked by hand for `swz(3,4,3)` over a 1024-byte `U8` allocation
   (index `0x80` lands at byte `0x90`: bit 7 XORed into bit 4).
3. **Each checker rule: refusal plus accepted control** (rules 1 to 5), the control differing from the refusal in exactly the
   named property. Mutant: drop rule 4 and the aliasing layout `(2,2):(1,0)` is accepted.
4. **Round trip.** `AllocDecl` JSON with and without the field; a document written before the field loads with `None`
   (the serde default is exercised, not assumed).

## 6. Emitters and the plan's decline rule

`unpopped-cpu-c` honours no layout on an allocation beyond the identity and **declines a `Some(layout)` by name**
(`design-tile-and-schedule-plan.md` section 3's rule: never silently ignored). An emitter that does not know the field
fails the same way. A new field on a `non_exhaustive` struct is not a breaking release (plan section 4, P1); T2's
statements are.

## 7. Patents and what this does not claim

The layout algebra is "outside as read" (`docs/PATENT-SWEEP.md`, "Addendum: layout algebra"). This note adds a declared
property of an allocation and **no statement**, so no standing hold is touched. It says nothing about whether any use of a
swizzle in a descriptor is outside any claim: the descriptor and swizzle parts are the re-read trigger recorded in
`docs/ir-union-rulings.md` (the `wgmma` families), and this note does not clear them.

## 8. Open points for `unpopped`

1. Is `AllocRaw` on an allocation with a layout refused or physical (section 4)? Recommend physical.
2. `align_bytes` on `AllocDecl`, or an implicit alignment of the swizzle's period for any swizzled allocation? Recommend the
   field: it is checkable and an emitter needs it.
3. Layout on `Private` allocations: allowed (a register-array permutation) or refused for now? Recommend refused until a row asks.
4. Sources not read for this note: `cute/layout.hpp`, `cute/swizzle.hpp`, the CUTLASS smem layout atoms.
   Everything above about the byte-to-element conversion is from the PTX figures in `design-wgmma-descriptors.md` and from
   `swizzle.rs`, not from CUTLASS.

— lane: unpopped-leaves
