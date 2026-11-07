# Options note: typed `Extern` and inline asm, with declared effects and per-target bodies (ROADMAP U11, I11)

Status: **proposal; typed `ExternDecl` and `Stmt::ExternCall` are built and checked (0.43.0, #288), the rest is not.** Docs first, as for U7. Read at `origin/main` `dcc3174` (0.42.30). U11 waits only on U3 (kernel
layer, `Module`, `Stmt::Call`), which is done. Closes gap row 3.19 of `docs/ir-union-design.md`; decision D2 there already says **yes,
with declared effects and per-target bodies, opaque to the optimizer by construction**. This note settles the shape.

## What exists

- Nothing for an escape hatch. `ScalarExpr::Unary` is a closed set (about 40 functions); `Stmt` has no asm or foreign call. The only
  call is `Stmt::Call` into another `KernelFn` of the same `Module` (checked, run by the oracle, spelled by cpu-c).
- `Stmt` is **exhaustive on purpose** (`kernel.rs`): a new statement breaks every consumer at compile time. So a new variant is a
  breaking minor (pre-1.0: the second number), allocated by the PM.
- Rendezvous statements (`Barrier`, `Collective`, `Mma`) must be reached in uniform control; a call that reaches one is a barrier site.
  Any effect vocabulary has to reuse that rule, not invent a second one.

## What the surveys ask for (the escape-hatch rows; the rescore's group 7, "escape hatches, metaprogramming and user-defined types", has 17 rows, of which only part is this)

| Source | Construct | What it carries |
|---|---|---|
| CUDA | `asm volatile("..." : "=r"(x) : "r"(y))`, `cuda::ptx::*` | a text template, output/input operand constraints, `volatile` |
| Triton | `tl.inline_asm_elementwise(asm, constraints, args, dtype, is_pure, pack)` | elementwise over a tensor, tuple result, **declared purity**, `pack` (elements per asm instance) |
| Triton | `tt.extern_elementwise(libname, libpath, symbol, pure)`; `tl.extra.libdevice.*` | a library symbol, per backend (CUDA libdevice, AMD ocml) |
| Triton | Gluon `inline_asm_elementwise` with smem/tmem descriptors | **not pure** (touches memory) |
| Mojo | `inlined_assembly`, `__mlir_op` | asm text; MLIR is a different escape (see below) |
| Slang | `spirv_asm`, `__target_switch`, `[require(...)]` | **a different body per target** in one function; capability atoms |
| Slang | `extern static const` / link-time constants | a name resolved at link, not a call |

The survey files mark a few of these `UNVERIFIED` (`ir-union-surveys/triton.md`, `slang.md`: the `spirv_asm` page was not retrievable).
Nothing below depends on the unverified details; the places that do are marked.

## Proposal

### 1. One declaration, two uses

An `ExternDecl` (stored in the `Module`, by index, like a callee) says **what the opaque thing is** and a use site says **where it is
called**:

```text
ExternDecl {
    name,                       // for diagnostics and the text form
    params:  Vec<ElementKind>,  // scalar operands only in v1
    results: Vec<ElementKind>,  // 0, 1 or a tuple (Triton's tuple dtype)
    effects: Effects,           // section 2
    bodies:  Vec<TargetBody>,   // section 3
    reference: Option<u32>,     // index of a KernelFn in the same Module: the semantics the oracle runs (section 4)
}
Stmt::ExternCall { decl: u32, args: Vec<ScalarExpr>, results: Vec<u32> }   // locals that receive the results
```

A statement, not a `ScalarExpr`: an effectful call has an order, and `Stmt` is where order lives. A pure one can still be used in
straight-line code; a later `ScalarExpr::ExternPure` is possible if a consumer needs one inside an expression (not asked for yet).

### 2. Effects are declared, closed, and checked

```text
Effects {
    pure: bool,                 // no effect at all: may be reordered, duplicated or dropped if unused
    reads:  Vec<MemSpace>,      // spaces it may read through a pointer argument (v1: empty; no pointer operands)
    writes: Vec<MemSpace>,
    sync: Option<Scope>,        // rendezvous at this scope: a barrier site, uniform control only
    volatile: bool,             // must not be moved or removed (CUDA `asm volatile`)
}
```

The checker holds a declaration to its own words, which is the testable half of "declared effects are what is tested": `pure` excludes
`reads`, `writes`, `sync` and `volatile`; `sync` makes the call a barrier site under the existing uniform-control rule; a pure
declaration with no result is refused (it could only be dead). **v1 has no pointer operands**, so `reads`/`writes` stay empty and
only `pure`, `sync` and `volatile` are live. Memory-touching asm (Gluon's) waits for the pointer/descriptor operands U8 brings, and
the field exists now only so that adding it later is not a second break of `ExternDecl`.

### 3. Per-target bodies

```text
TargetBody { target: TargetId, text: BodyText }
BodyText::Asm   { template, constraints, ... }   // PTX, SPIR-V asm, AMDGCN: opaque text, never parsed by Unpopped
BodyText::Symbol { library, symbol }              // libdevice / ocml / a linked object
```

`TargetId` is a string key matching the backend's own name (`"cuda-ptx"`, `"spirv"`, `"cpu-c"`, ...); a body for a target nobody
has is simply never used. Unpopped checks only shape: the template's operand placeholders match `params.len() + results.len()`, a
target appears at most once. **It never checks that the asm is correct** (it cannot; that is the point of an escape hatch), and a
backend with no body for its own target must refuse with a typed error naming the declaration, never fall back silently.

### 4. What the oracle and cpu-c run: the `reference`

cpu-c cannot run PTX. The design doc's answer is "legs per target; the declared effects are what is tested". To make the **result**
testable too, a declaration may name a `reference`: an ordinary `KernelFn` of the module with the same signature. The oracle and cpu-c
run the reference. A declaration without one is allowed, but the oracle then traps `Unsupported("an Extern with no reference body")`
exactly as it does for any statement it cannot run, so such a kernel is still checkable and emittable on the target that has a body, just
not runnable by the oracle. The device leg (self-hosted runner, as for `kernel_*` device tests) compares the target body with the
reference, within the tolerance the declaration states if it is floating point. That comparison is the only evidence the asm means
what its reference means, and the note says so in the docs of the type.

### 5. What it is not

- **Not an optimizer hint.** A call is opaque to every pass by construction: no fusion across it, no CSE unless `pure`, no hoisting
  past a `sync`.
- **Not MLIR or front-end metaprogramming.** Mojo's `__mlir_op` and Triton's `ir_override` replace a compiler stage; D3 of the design
  doc keeps metaprogramming out of the IR. They are declined, with this sentence as the reason.
- **Not link-time constants.** Slang's `extern static const` is a named constant resolved at link. It is a different feature (a `Param`
  that is fixed at link) and is not in I11; filed as a follow-up only if a consumer asks.
- **Not a way around the numeric contract.** The declared result types are `ElementKind`s, so every value that comes back is typed
  and the existing kernel checker types its uses.

## Decisions for the PM

| # | Question | Recommendation |
|---|---|---|
| Q1 | Statement (`Stmt::ExternCall`) or expression (`ScalarExpr::ExternPure`) first | Statement only in v1; the expression form is additive later |
| Q2 | Is a `reference` body required | **Optional**, as above: required would make every libdevice call carry a hand-written reference |
| Q3 | Pointer/descriptor operands in v1 | **No**; they arrive with U8's descriptors, and `reads`/`writes` are reserved until then |
| Q4 | Where do declarations live | In the `Module`, by index, like `Stmt::Call` callees (one mechanism) |
| Q5 | Target key | A string matching the backend's name; a closed enum would make every new backend a break of this type |
| Q6 | Version | One breaking minor (new exhaustive `Stmt` variant, new `Module` field); the PM allocates the number |

## Build order, once approved

1. `ExternDecl`, `Effects`, `TargetBody`, `Stmt::ExternCall` and the `Module` field, with serde and the text form; the checker
   (effects consistent with themselves, arity and types, `sync` as a barrier site). Tests: each refusal named by its text, plus a
   mutant that drops each check.
2. Oracle: run the `reference`; trap when there is none. cpu-c: spell the reference as a call; refuse with a typed error when asked
   to emit a body for its own target that is `Asm`.
3. A PTX leg: one real `Asm` body (a single-instruction one, e.g. `popc` or `fma.rn.f32`) compared with its reference on the self-hosted
   runner. Until that runner leg exists, the doc says in one line that the asm body is unexercised.
