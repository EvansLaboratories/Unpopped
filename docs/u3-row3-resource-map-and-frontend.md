# U3 build row 3 design note: the resource map and the `Frontend` change

Status: **design note, no code.** Row 3 of `u3-program-model-and-seam.md` section 5 ("the resource map and
the `Frontend` change together with I2's kernel-layer path"). The seam hold was lifted for this row by the
PM on 2026-10-29 after Fuel and Baracuda both answered yes to one bump. This note is the PR that comes
first, as the PM required; the code PR follows it.

**Requirements from the two consumers, as relayed by the PM (a-d), and where this note answers each.**

| | Requirement | Answer |
|---|---|---|
| a | `const fn Frontend::new(..)` so `Frontend` can go `#[non_exhaustive]` without breaking the one cross-crate literal (`baracuda-cuda-parse/src/lib.rs:72`, `pub const CUDA: Frontend = Frontend { .. }`) | 2.1 |
| b | a resource-map parameter on the `lift_*` entry points | 2.2, 2.3 |
| c | a CHANGELOG migration note naming any change to `ElementKind` / `OperandDesc` / `ArchSku` (Fuel's whole contact) and every new `ScalarExpr` leaf (Baracuda adds an arm or a typed decline for each) | 4 |
| d | std140 versus scalar: is the binding layout a declared `Frontend` input? Open at `SLANG-SEAM-REQUIREMENTS.md:192` | 3 |

## 1. What is read from `origin/main` (4b5bedc, 0.32.24)

- `Frontend` is `#[derive(Clone, Copy, Debug)]`, six `pub` fields, no `#[non_exhaustive]`
  (`convert.rs:103`). `SLANG` is a `const` literal of it, and one in-crate test builds it by literal
  (`unpopped-slang` only mentions it in a doc comment); Baracuda's `CUDA` is the one cross-crate literal.
- The four entry points are `lift_elementwise`, `lift_reduction`, `lift_scan`, `lift`, each
  `(fe: &Frontend, src: &str, name: &str, dtypes: &[ElementKind]) -> Result<Lifted, LiftError>`
  (`convert.rs:161-217`). Buffers are recognised **only by name**: `fe.out_name`, `fe.in_prefix` + `K`.
- `Lifted { op, n_inputs }` is not `#[non_exhaustive]` (`lift.rs:114`).
- The row 1-2 types exist and are unwired: `ParamField`, `ParamsLayout`, `LayoutRule`, `Binding`,
  `BindingKind`, `BindingRole`, `Packing`, `KernelManifest::check` (`artifact.rs`).
- The kernel-layer lifts (`lift_*_opdef`, `kernel_lift/*`) go `OpDef -> KernelFn`. They take **no**
  `Frontend` and no source text; nothing about them changes in this row.

## 2. Proposal

**2.1 `Frontend` becomes `#[non_exhaustive]` with a `const fn new`.**

```rust
impl Frontend {
    pub const fn new(
        name: &'static str, kernel_marker: &'static str, residue: &'static [&'static str],
        out_name: &'static str, in_prefix: &'static str, parse: fn(&str) -> Option<Tree>,
    ) -> Self
}
```

The six arguments are exactly today's six fields, in declaration order, so a literal converts
mechanically. `SLANG` and the in-crate test move to `new`. No field is added to `Frontend`: it stays
`Copy` and `const`-constructible, and a resource map (which owns `Vec`s) cannot live in it. Baracuda's
change is one expression (`Frontend { name: "cuda", .. }` to `Frontend::new("cuda", ..)`); A grep of `fuel/crates` (primary checkout, 2026-10-29) finds no `Frontend`
use, so Fuel builds none. After this, adding a field is non-breaking.

**2.2 `ResourceMap` is a per-call input, not a `Frontend` field.**

```rust
#[non_exhaustive]
pub struct ResourceMap {
    pub buffers: Vec<DeclaredBuffer>,   // DeclaredBuffer { name, role: BindingRole, set, index, elem, packing }
    pub params: Option<ParamsLayout>,   // the Params members, by name; carries its own LayoutRule
}
```

`ResourceMap::new(buffers, params)` plus `check()` (duplicate names, duplicate `(set, index)`, a `params`
whose `ParamsLayout` fails its own check, an output buffer that is `Read`-only). It reuses the row 1
types rather than restating them: the map is the **input** half of what a `KernelManifest` entry later
reports, so the two cannot drift apart in vocabulary. Fuel's condition (item 2, spec section 3) holds:
the map covers `Params` fields, not only buffers.

**2.3 Entry points: new `*_with`, old ones kept.**

`lift_elementwise_with(fe, src, name, dtypes, resources: Option<&ResourceMap>)` and the same for
`reduction`, `scan`, `lift`. The existing four keep their signatures and call the `_with` form with
`None`. Recommendation over changing the four in place: the minor bump is already breaking for the
`Frontend` literal, and there is no reason to add a second breakage (every `lift_*` caller in Baracuda and
Fuel) for no behavioural gain. If the PM prefers one name, changing in place is a mechanical follow-up.

With `Some(map)` the walk resolves a buffer name through the map: the output is the one `Write` /
`ReadWrite` buffer, operand `K` is the `K`th `Read` buffer **in declared order** (not `in{K}` by name),
and a `Params` field referenced in the body resolves to a scalar parameter by name. A name the map does not
declare is a typed `LiftError` (new variant `UndeclaredResource(String)`), not a fallback to inference:
a declared map that is wrong must be loud. With `None` the behaviour is byte-identical to today.

**2.4 Provenance.** `Lifted` gains `provenance: LiftProvenance { Declared, Inferred }` and becomes
`#[non_exhaustive]` (a `Lifted` is produced here; a grep of `baracuda/crates` for `Lifted {` finds no consumer
literal, and the code PR repeats it). `None` gives `Inferred`; `Some(map)` gives `Declared`. Fuel's condition: a kernel lifted
by name inference is counted as inferred and cannot pass as declared. `Lifted` also gains
`resources: Option<ResourceMap>` echoing the map used, so a consumer can build the manifest without
re-deriving it.

**2.5 What this row does not do.** It does not read `[[vk::binding]]` or the `Params` struct out of the
CST (that is a Slang-grammar recogniser, Fuel's emitter side, and a later increment against these types).
It does not touch `GeneratedKernel`, `Backend`, the kernel-layer path, `ScalarExpr` or any element type.
"Together with I2's kernel-layer path" is read as: the `Frontend` non-exhaustive change and the
resource-map entry points land in the same minor as the kernel-layer-returning `Frontend` work I2 plans
(`ir-union-design.md` section 4); this row supplies the types that path will take, and the kernel-layer
return itself stays I2's. If the PM means the code to land in one PR, say so at the gate.

## 3. Decision (d): is the binding layout a declared `Frontend` input?

**Decision: yes, as data in the `ResourceMap`, with no default, and not as a `Frontend` field.**

- The layout rule lives in `ResourceMap.params: Option<ParamsLayout>`, whose `LayoutRule`
  (`Std140 | Std430 | Scalar`) is required. A map with `Params` fields and no rule cannot be built.
  This is the row 1 rule ("a manifest that does not say is not a manifest") applied to the input.
- It is per map, not per `Frontend`, because it is a property of a pipeline-layout family
  (`2s1u`, `3s1u`), which is Fuel's dialect, not of the language (spec section 3, item 5).
- Fuel measured every `Params` field in the corpus as scalar (`SLANG-SEAM-REQUIREMENTS.md:74` region), and
  for scalar-only members the three rules give the same offsets. They differ only for **array members**,
  which the corpus has in 4 kernels. Which rule Fuel's host struct assumes for those 4 stays
  **UNVERIFIED and Fuel's to measure** (spec section 6); this row makes it a stated input instead of an
  implicit one, and `ResourceMap::check` rejects an offset that is not valid under the stated rule for the
  member's type (so a Std140 map with an unpadded array fails, which is the discriminating case).
- The cost of getting it wrong is bounded: the lift does not use the offsets (it resolves names);
  only the manifest built from the map carries them.

## 4. CHANGELOG migration note (requirement c), drafted

Row 3 as proposed changes **no** `ElementKind`, `OperandDesc`, `ArchSku` or `ScalarExpr` item. The code
PR's CHANGELOG entry will say that in those words, with the evidence being a public-item diff (and a
`ScalarExpr` variant count before and after). It will name what does change, for the two consumers:

- Baracuda: `CUDA` literal to `Frontend::new(..)`; no `ScalarExpr` arm needed (no new leaf).
- Fuel: nothing it compiles against changes; `Lifted` gains fields (it reads, never builds, it).
- Both: new `LiftError::UndeclaredResource`; `LiftError` is already `#[non_exhaustive]`
  (`lift.rs:27`), so a consumer's `match` already has a wildcard.

Row 4 (scopes and memory spaces) will add `ScalarExpr` leaves and carries its own migration note; the PM
orders the row 3 and row 4 numbers at the gate (one minor bump, 0.33.0).

## 5. Test and gate plan for the code PR

1. Round trip: for each existing frontend test kernel, lift with `None` and with a map declaring the same
   buffers; the `OpDef`s are equal and the provenances are `Inferred` / `Declared`.
2. The case inference cannot do: a Slang kernel with buffers named off-convention lifts with a map and
   fails (`NotElementwise`/`Unrecognized`) without one. Operand order follows the map, shown by a map that
   declares the buffers in the reverse of source order.
3. An undeclared name with a map present is `UndeclaredResource`, not a silent fallback.
4. `ResourceMap::check` on each defect, plus the Std140-array case from section 3.
5. Mutation-checked, each mutant named: operand order by declaration swapped to by-name, provenance
   hard-coded `Declared`, fallback-to-inference on a miss, the rule check skipped.
6. `tests/neutral_core_names_no_cuda_grammar.rs` still passes (the map names no language).
7. Patent sweep: a row for this note's surface ahead of release (PM requirement). Declared resource maps
   and a layout-rule input are passive data plus name resolution; no variant selection and no address
   spaces, so hits 19 and 20 stay row 4's. The note goes in `PATENT-SWEEP.md` with the code PR.

## 6. PM rulings (2026-10-29), recorded

1. `*_with` siblings; the old `lift_*` delegate to them with no declared map (`None`, giving `Inferred`).
   Additive for callers; only the `Frontend` `#[non_exhaustive]` change breaks.
2. "With I2's kernel-layer path" means the same **release**, not the same PR: separate PRs, merged
   back-to-back, published together. This lane's code PR is 0.33.0 (first breaking; Baracuda needs the
   `Frontend` constructor); lane 1's row 4 is 0.33.1; one 0.33.x release.
3. Still open and Fuel's: std140 or scalar for the 4 array-member kernels (section 3).
