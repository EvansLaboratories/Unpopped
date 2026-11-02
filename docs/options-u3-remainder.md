# Options note: what remains of U3 (ROADMAP U3, gaps 3.1, 3.2, 3.3, 3.18)

Status: **ruled 2026-11-02; R1's shape is built (0.41.0, #241), the rest waits for a consumer.** Written as an options note, with nothing built, before the rulings below. Asked by the PM (2026-11-02): the U3 remainder as a note, after the U4 note
(`docs/options-u4-sync-and-atomics.md`, ruled the same day: break `Stmt` once, stage the semantics). Read at
`origin/main` `59d25bd` (0.40.20) plus the unmerged U4 note branch; sibling counts are `git grep` over each repo's
`origin/main`.

## What is done

The spec (`u3-program-model-and-seam.md`), the artifact and capability types (`capability.rs`, `artifact.rs`,
including `BindingKind::{Storage, Uniform, Texture, Sampler}` in the **manifest**), the resource map and `Frontend`
change (`resource_map.rs`), and row 4: scopes, execution ids, `Place::Alloc` in `Shared`/`Private`/`Constant`,
`Stmt::Barrier`, the per-thread driver (`u3-row4-scopes-and-memory-spaces.md`). `kernel.rs`'s header names what is
deliberately not there: `Stmt::Call` (needs a module type, spec unknown U-2), pointer values (`Place::Ptr`), the
effect set.

## What is open, and what each costs

| item | gap rows (languages listing it) | measured need | breaks? |
|---|---|---|---|
| R1 `Stmt::Call` + a module type | 3.1 statement layer (all 7 list "device function calls") | the U3 spec says it is "needed before any device-function call, and so before most real Slang kernels lift" (Fuel's corpus is 147 kernels) | a new `Stmt` variant: breaking, **unless it rides U4's one break** |
| R2 pointer values, `Place::Ptr` | 3.3 (all 7); the Slang survey's item 7 names `RWByteAddressBuffer` (1 Fuel file) | no lane has asked; D-R4-3 defers them "when a consumer needs them" | `Place` is `#[non_exhaustive]`: additive |
| R3 `Texture`/`Tensor` memory spaces, texture/sampler/struct resources | 3.18 lists **Slang only**; Slang survey item 9 "zero IR representation" | unmeasured: the 147-kernel count covers `ConstantBuffer<Params>` (147 of 147), not textures | `MemorySpace` is non-exhaustive: additive; sampling operations are new `ScalarExpr` leaves: breaking |
| R4 `ProgramModel` / `BlockSpmd` lowering | 3.2 (all 7 list the execution model; `BlockSpmd` is Triton's) | no Triton frontend exists in any lane: the only `triton` mentions in baracuda are comments naming reference implementations | `KernelFn` is non-exhaustive: a `program_model` field is additive; the lowering pass is new code |

### R1, `Stmt::Call`

U-2 (`ir-kernel-layer.md`) leaves open what a callee is. Options for the callee: (a) another `KernelFn` in a
`Module { fns }` (b) an intent-layer `OpDef` body (c) an opaque extern with declared effects (that is U11).
Recommend (a): a callee that is a `KernelFn` reuses the checker, the interpreter and the lift, and U11 covers
(c) later. Rules to state with it: the call graph is acyclic (no recursion; GPU targets mostly forbid it), and a
callee containing a `Barrier` must be called from uniform control, so the divergence check becomes transitive.
**Recommendation: fold the `Stmt::Call { callee, args, result }` shape into the U4 break** (a `Module` type and
`check_module`, with `check_kernel` refusing a `Call` until they land, the same typed-refusal discipline as U4's
option B). That makes the 0.41.0 `Stmt` break carry both and spares downstream a second one.

### R2, pointer values

Three levels, cheapest first: (a) wait; (b) **byte-offset typed load and store** (`RawLoad`/`RawStore` with a
`Bitcast`), which covers `RWByteAddressBuffer` and needs no pointer type and no aliasing story; (c) a pointer
`Ty` with an aliasing story (what CuTe and Triton need for a shared-memory tile). Recommend (a) now and (b) as the
first step when a consumer asks, because (c) is a much larger commitment (D-R4-3) and nothing consumes it.

### R3, textures and resources

The **manifest** already names texture and sampler bindings (`BindingKind`); what is missing is the *IR* side:
a texture value, sampling operations and a `Texture` space. Only the Slang column lists it. **Measured (Fuel
lane, `fuel@1bdd4c3e0`, relayed by the PM 2026-11-02): 0 of 192 kernel files sample a texture or a sampler**
(149 Slang, 27 GLSL, 16 Metal, 0 CUDA; `git grep` with a positive control that hit; Baracuda's CUDA kernels were
not scanned). So Fuel needs no texture or sampler resources, and nothing consumes them: build nothing until a
kernel that samples is a lift target.

### R4, `ProgramModel`

D6 ruled that `BlockSpmd` lowers to `ThreadSpmd` inside Unpopped. It needs a tile type (`TileTy`), which belongs
with U6/U7/U10, and a Triton frontend, which nobody is writing. Recommend: add only the declaration, a
`program_model` field on `KernelFn` with the single value `ThreadSpmd` (additive), so a kernel states its model;
the lowering waits for a Triton frontend lane.

## Recommendation, in order

1. **R1 rides U4's break** (shape now, semantics staged). It is the only item a measured need points at.
2. R4's declaration field, additive, whenever convenient.
3. R2(b) and R3 only when a consumer asks (the Fuel count for R3 is 0).

## Needs a ruling

1. Callee = `KernelFn` in a `Module` (R1a), and the `Call` shape folded into the U4 break.
2. ~~Ask Fuel for the texture/sampler kernel count (R3)~~: answered, 0 of 192.
3. Confirm "when a consumer needs it" stays the rule for R2, R3 and R4's lowering.

## Rulings (PM, 2026-11-02), recorded after the note merged

1. **Callee = a `KernelFn` in a `Module`; the `Stmt::Call` shape rides the 0.41.0 break: YES, and built.** `Stmt::Call { callee, args, result }` and
   `Module { fns }` landed in 0.41.0 (#241); `check_kernel` refuses a `Call` with `KernelError::CallUnsupported` until `check_module` (acyclic call graph,
   a callee holding a `Barrier` called only from uniform control) and the call semantics land as an additive slice.
2. **Texture count:** the Fuel lane measured 0 of 192 kernel files (section R3), so nothing consumes texture resources.
3. **"When a consumer needs it" stays the rule** for pointer values, textures and the `BlockSpmd` lowering. The `program_model` declaration field (R4) is
   additive and not yet built.
