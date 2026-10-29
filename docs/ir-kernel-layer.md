# The kernel layer, v0 (U2): spec

Status: **spec, written before the code; steps 1-4 of section 9 (Elementwise part) have landed** (0.31.0 types and `check_kernel`; 0.31.3 interpreter and `lift_elementwise_opdef`) (PM ruling 2026-10-28: spec doc first, effects model stated, unknowns
marked, scope checker with known-bad mutations). Closes ROADMAP U2 / gap 3.1 of `ir-union-design.md` once built.
This document refines `ir-union-design.md` section 4 (two layers) and decision D1; it does not reopen it.

**Placement ruled by the PM (2026-10-28, `[ASK-ANSWER] U2-shape` option (a))**: the kernel layer is a **separate
artifact** in a new module `kernel.rs`, plus **one** new `ScalarExpr` leaf, `Local(u32)`. It is **not** a field
of `OpDef` or `KernelPlan`, and it touches **no** `Backend` / `Frontend` trait. Consequence: there is nothing for
the plan gate to decline, because nothing in v0 reaches a `KernelPlan`. How a `Backend` receives a kernel-layer
body is a later decision that waits for the Slang requirements (Fuel#514) and for U3's `ProgramModel`.

## 1. What U2 delivers, and what it does not

Delivers: (1) the data types of section 3; (2) `check_kernel`, a scope and type checker (section 4);
(3) a reference **interpreter** on the oracle side (section 5), so that "the layers agree" is a test that runs in
plain Rust with no emitter and no device; (4) `lift_elementwise_opdef` (module `kernel_lift`), which lowers an existing intent-layer `OpDef` into a
`KernelFn` (section 7), in the order Elementwise, Reduction, Scan, Contraction, Im2Col, Window, RowReduce (all landed or in the gate).

Does not deliver: memory spaces, pointers, execution scopes or thread ids (U3); barriers, atomics with order or
scope (U4); collectives (U5); a path from a `KernelFn` to any emitter; any `Frontend` change (ROADMAP I1 waits
for the end of U2). A parser still cannot produce a `KernelFn` after U2; `lift_opdef` is the only producer.

## 2. The machine model in v0: one logical thread

A `KernelFn` is **the program of one logical thread**, run over an iteration space that the lifting makes explicit
as a loop. There is no notion of "other threads". This is deliberate: the sequential reading is the semantics
every later increment must stay consistent with (U3 adds thread ids and scopes *around* it; cpu-c's planned
serialized "thread loop" reference, `ir-union-design.md` I3, is this model repeated per thread).

## 3. Data types

New module `crates/unpopped/src/kernel.rs`. All types derive `Clone, Debug, PartialEq, Serialize, Deserialize`
(the same text carrier as `OpDef`, so a kernel can be authored, diffed and round-tripped). `Stmt` and `KernelFn`
are exhaustive **on purpose**, as `ScalarExpr` is: a new statement must break every consumer at compile time.

```
KernelFn { name, compute: ElementKind, params: Vec<Param>, locals: Vec<LocalDecl>, body: Block }
Param    { name, kind: ParamKind }          // #[non_exhaustive]
ParamKind::Buffer { elem: ElementKind, role: BufferRole }   // Read | Write | ReadWrite
ParamKind::Scalar { ty: ElementKind, local: u32 }   // readable as that local, in scope everywhere, never assigned
LocalDecl{ ty: Ty }                          // Ty = Scalar(ElementKind) | Packed(ElementKind, u8)
Block(Vec<Stmt>)

Stmt::Let    { local: u32, init: ScalarExpr }          // declare + initialize; scope = rest of the block
Stmt::Load   { local: u32, place: Place }              // declare local = the element at place
Stmt::Assign { local: u32, value: ScalarExpr }         // mutation of a declared local
Stmt::Store  { place: Place, value: ScalarExpr }       // write one element
Stmt::If     { cond: ScalarExpr, then: Block, els: Block }
Stmt::For    { local: u32, start, end, step: ScalarExpr, body: Block }   // integer counter, half-open
Stmt::While  { cond: ScalarExpr, body: Block }
Stmt::Break | Stmt::Continue | Stmt::Return
// Stmt::Call { func, args, result }  -- DEFERRED to the PR that introduces a module type (unknown U-2)

Place::Logical { buffer: u32, coords: Vec<ScalarExpr> }   // the element at LOGICAL coordinates
```

Plus the single `ScalarExpr` addition: **`Local(u32)`**, a leaf reading a declared local. It is the only change
to the intent layer's expression type (breaking, because `ScalarExpr` is exhaustive; every match site decides,
and intent-layer consumers that never see a kernel reject it with the same typed decline they use for `Pack`).
A load is a **statement** (`Stmt::Load`), not an expression, so no second leaf is needed in v0. **Unknown U-1:**
whether U3's typed loads need to be expressions (for example a load inside an index expression) is open; if so
that is a second breaking leaf and the PM is asked first. Expressions inside a kernel may use only `Const`,
`ConstInt`, `Local` and operators: the intent-layer leaves `Input`, `Param`, `Reduced`, `Coord`, `CoordInt` have
no meaning in a statement program (`KernelError::IntentLeaf`); inputs arrive through `Load`, scalar arguments
through `ParamKind::Scalar { local }`. `compute` is the dtype of the untyped leaves (`Const`), exactly the
plan dtype's role for an `OpDef`.

Why `Place::Logical` and not a pointer: the intent layer's operands already carry views (strides, broadcast,
reversal) that the oracle applies, so a v0 kernel can say "element at these coordinates" and inherit exactly
those semantics. Pointer arithmetic is U3's `typed load/store; pointers`. See section 8.

## 4. `check_kernel`: scope and type rules

`check_kernel(&KernelFn) -> Result<(), KernelError>` returns the first violation, walking in program order.

1. **Declaration before use, in scope.** A `Local` read, `Assign` target or `For` counter must name a local
   declared by an earlier `Let` in an enclosing block (or the `For` itself). A local declared in a block is out of
   scope after it. `KernelError::UseBeforeLet`, `OutOfScope`.
2. **One declaration per local id.** A `Let` or `For` re-declaring an id that is in scope, or an id outside
   `locals`, is `Redeclared` / `UnknownLocal`.
3. **Types.** Every expression is typed by `infer_ty` (packed values included) with `Local(i)` typed from its
   `LocalDecl` (`types::infer_ty_in`, which takes the local types). `Let`, `Load` and `Assign` require `Ty` equality (a `Load`'s local type equals the buffer's `elem`) (no implicit conversion; use `Cast`). `If` / `While`
   conditions are scalars of any type (tested `!= 0`, as `Select`'s condition is). `For` bounds are one integer type
   shared with the counter. `Store` value type equals the buffer's `elem`. `KernelError::TypeMismatch`.
4. **Control flow.** `Break` / `Continue` only inside `For` / `While`. `Return` anywhere. No statement after
   an unconditional `Break` / `Continue` / `Return` in the same block is an error (`Unreachable`), so a lifted body
   cannot hide a dead store.
5. **Buffers.** `Store` only to a `Write` or `ReadWrite` buffer; `Place::Logical` arity equals the declared rank
   when the rank is known. `KernelError::WriteToReadOnly`, `BadPlace`.
6. **Mutation discipline.** An `Assign` to a loop counter inside its own loop body is rejected (`CounterAssigned`),
   removing the one mutation whose meaning differs between C and the interpreter's half-open loop; an `Assign`
   to a scalar-parameter local is rejected (`ParamAssigned`).

**Known-bad mutations the test suite must exercise** (each is a program that is *valid* except for one change, so
the test fails if the check it targets is deleted, not merely if checking is off): delete the `Let` of a used
local; move a use after its block closes; declare an id twice; `Assign` an `f32` local an `i32`; `Load` into a local of the wrong type; read an intent-layer `Input` leaf; branch on a packed
value; `Break` outside a loop; a statement after `Return`; `Store` into a `Read` buffer; assign the loop counter.
Plus the positive control for each: the unmutated program passes.

## 5. Semantics and the interpreter

The interpreter (`oracle::kernel_eval::run_kernel`, public, no emitter) runs a `KernelFn` with scalar arguments and
buffers and returns the written buffers. Rules:

- Locals are mutable cells initialized by `Let` before first read; there is no uninitialized read (the checker
  forbids it), so there is no undefined value to model.
- `For { local, start, end, step }`: evaluate the three bounds **once**, before the first iteration; iterate
  `start, start+step, ...` while `< end` for positive `step`, `> end` for negative; `step == 0` is a trap
  (`KernelTrap::ZeroStep`), not an infinite loop. Integer arithmetic wraps exactly as `ir-typed-expressions.md`
  section 8 (the loop counter included: a counter that would wrap ends the loop).
- `While` re-evaluates its condition each pass. A trip count cap exists **in the interpreter only**
  (`KernelTrap::StepLimit`, default 10^7) so a test of a bad program terminates; it is not part of the language.
- Expression evaluation reuses the oracle's existing evaluator unchanged (`Local` reads the cell); that is what
  makes the layers comparable bit for bit rather than approximately.
- `Store` writes through the operand's view, as the oracle's writeback does. Two stores to one element: the later
  wins. (Concurrent stores do not exist in v0.)

## 6. The effects model (stated; the unknowns are marked)

An **effect set** is attached to a `KernelFn` by `effects(&KernelFn) -> Effects`, computed, never authored:

```
Effects { reads: set<buffer>, writes: set<buffer>, traps: bool, diverges: bool, calls: set<FuncRef> }
```

- `reads` / `writes`: buffers loaded from or stored to anywhere in the body (including inside branches).
- `traps`: some reachable statement can trap (`step == 0` is statically knowable only for a literal). Integer
  division does not exist in the intent layer (`ir-typed-expressions.md` section 8: "Division is deliberately
  absent"), so v0 has no division trap; when it is added, its zero / `MIN / -1` rule enters here.
- `diverges`: contains a `While` whose condition is not provably false-eventually. Conservative: every `While`
  sets it, every `For` with literal bounds does not.
- `calls`: functions named by `Call`.
- Purity: a body with empty `writes` and `calls` and `traps == false` is pure; the optimizer may reorder pure
  statements. v0 performs **no** statement-level optimization, so this is a stated fact for U3+, not used yet.

**Unknowns, marked so nobody reads them as decisions:**

- **U-2. `Call`.** What a callee is: an intent-layer `OpDef` body, another `KernelFn`, or an opaque extern with a
  declared effect set (U11). v0 implements `Call` only to a `KernelFn` in the same module and rejects the rest;
  recursion is rejected by the checker. Real device-function calls (CUDA `__device__`, Slang functions) need a
  module type that does not exist yet.
- **U-3. Aliasing.** Two `Buffer` params may name the same storage. v0 states that they do **not** alias
  (the lifted ops never alias), and the interpreter checks it. Whether aliasing is a per-param attribute or a
  property of a launch is open.
- **U-4. Atomic, barrier and async effects** are U4's, and the effect set will need a *scope* axis then (a write
  visible to a block, a device, a peer). Nothing in the v0 shape forecloses a `scope` field; nothing implements one.
- **U-5. Nondeterminism** (RNG, `printf`, asserts) is an effect per `ir-union-design.md` D3 and is absent in v0.
- **U-6. Floating-point effect.** Rounding mode, flush-to-zero and `fast-math` already ride `Qualified` on the
  expression. Whether a *region* may carry a numerics mode is open; v0 says no.

## 7. Lifting `OpDef` into the kernel layer

`lift_opdef(&OpDef, &KeyShape) -> Result<KernelFn, LiftDecline>`; the layers must **agree on every existing op**
(`ir-union-design.md` I2). Agree means: for every cell of the existing exhaustive harnesses, interpreting the lifted
`KernelFn` gives the same bits as `oracle::evaluate` on the `OpDef`. The checks are differential and need no
emitter, which is why they can land under the seam hold.

- **Elementwise** (first): one nested `For` per output axis, outermost first (row-major order), so the loop
  counters *are* the logical coordinates and no integer division is needed (the intent layer has none). `Let` each
  input load through `Place::Logical { coords: counters }`, then `Store` the body with `Input(k)` replaced by the
  loaded local. The extents are `Scalar` params (run-time), as Fuel's corpus has them (Fuel#514 R7). A broadcast or
  reversed input is already handled by the view behind `Place::Logical`.
- **Reduction**, **Scan**, **RowReduce**: one accumulator local per `ReduceStage` / scan state, `For` over the
  reduced axis in the order the oracle folds (this order is part of the agreement; a reordering is U5).
- Anything the lifting cannot express is a typed `LiftDecline`, never a partial kernel. The
  multi-output (other than `RowSort`'s) and write-index forms are **out of v0**; each gets its own PR with its own differential. Landed
  after the first four: `Contraction` (identity epilogue), `Im2Col`, `Window` (identity post), `RowReduce`
  (tolerance mode, below), and `RowSort` (below).

- **RowSort** (`lift_row_sort_opdef`, `tests/kernel_row_sort_agrees.rs`): byte-exact, every `SortOut` and `SortLimit`, `F32`, `F64` and `I32`. The kernel is a
  *rank sort*, not the oracle's `sort_by`: each element counts the elements before it under `(key, original index)` with NaN
  greatest, and is stored at that rank. Descending swaps the keys of the comparison and keeps the index tiebreak, so ties keep
  ascending original index in both directions. Values move by `Load`/`Store`, so NaN payloads and the sign of zero are kept.
  The `TopK` guard `rank < k_out` is a loop that runs once or not at all (`for t in rank..min(rank + 1, k_out)`), because an `If`
  condition is a compute-dtype comparison. The interpreter compares integers (`eval_int` has the six comparisons, 1 or 0 at the compute
  type), so `I32` keys sort too; `I64` plans but declines, because the oracle orders keys through `f64`, which collapses large
  64-bit keys. Integer **reductions and scans** (`lift_reduction_opdef`, `lift_scan_opdef`, `tests/kernel_int_reduce_scan_agrees.rs`)
  lift at `I32` and `I64`: an accumulator of the dtype itself (the oracle folds integers exactly and wraps, and wrapping at the width
  commutes with the store's truncation), the identities `0`, `1` and the type's extremes, no NaN arm. An integer `Mean` and a fused post
  at an integer dtype decline.
  Integer **windows** (`Sum`/`Max`/`Min`, `tests/kernel_int_window_agrees.rs`) lift the same way; `RowReduce` is float-only in the plan, so
  it has no integer form.

- **Gather** (`lift_gather_opdef`, `tests/kernel_gather_agrees.rs`): byte-exact, an Elementwise op with one `ReadIndex::Indexed` input
  (torch `gather`, `index_select`, embedding), index dtype `I32`/`I64`/`U32`, every `OobPolicy`. The kernel loads the index once per cell,
  clamps it into the source's extent with integer `Min`/`Max` (so no load is out of bounds, as the oracle's emitter contract says), then
  guards the **store**: `Skip` leaves the cell, `ZeroFill` writes zero, `Clamp` stores the clamped read. The source extent along the
  gathered axis is one extra scalar parameter, typed as the index so the comparisons are same-typed. A 1-D index is a rank-`r` buffer
  with stride 0 on the other axes, as the oracle reads it. The oracle zeroes its output, so `Skip` and `ZeroFill` agree with it on a
  zeroed kernel output; they differ only on a pre-filled one, which has its own test. Open: scatter (`write_index`) and non-identity
  views, each still declined by `check_plain`.
- **Runtime base offsets** (`tests/kernel_base_offset_agrees.rs`): an Elementwise op carrying `BaseOffset::Runtime` on some inputs and/or the
  output (the plan admits them on Elementwise only). The IR defines the offset as a launch argument added to the operand's base before
  any address math; the kernel layer's argument buffer already has that field (`TypedBuffer::base_offset`, applied by the interpreter to
  every `Place` access), so **the lifted kernel is the offset-free one and the caller binds the offset in the buffer**. The lifts
  therefore only stop declining it. Each test buffer sits in junk padding on both sides, and a control reruns the kernel with an
  input's offset dropped to show the agreement depends on it. Open (U3): a backend that emits source from a `KernelFn` has no
  parameter that carries the offset (the spec's "no argument ABI" finding), so how it reaches a launch is the seam's question.

- **Input views** (`tests/kernel_view_agrees.rs`): an Elementwise op whose inputs carry `View::Permute`/`Layout`/`Broadcast`/`Reshape` (the plan
  admits views on Elementwise and Contraction; only Elementwise lifts them). A `Place` goes through the bound buffer's own shape
  and strides, so `Permute` is lift code (iteration axis `d` indexes producer axis `perm[d]`, so the producer coordinate on axis
  `perm[d]` is the counter of axis `d`; the producer buffer is bound as it is), while `Layout`, `Broadcast` (stride 0), `Reshape` and a
  flipped operand are **binding contracts** like a base offset: the caller binds the view's extents and strides on the same storage and
  the lifted kernel is the identity one. The tests use a rank-3 permutation that is not its own inverse (a transposed rank-2 is, and
  would not tell the two directions apart), and a control that the view-free op on the same buffers traps or differs. Open: a view on a
  Contraction (a transposed matmul operand); every other access still declines a view, since it would ignore it.

### 7.1 Two agreement modes, and the oracle as a tolerance referee

**Byte-exact is the default**, and the only mode for a post-expression or epilogue that is the identity
(`Reduced(0)`): the oracle accumulates in `f64` and rounds once at the store, and so does the lifted kernel.

A non-identity post or epilogue cannot agree byte for byte. The oracle hands the finalised stage value to it as an
**unrounded `f64`** (`Reduced(s)`), and the kernel language has no way to put an `f64` leaf inside a compute-dtype
expression. The lift therefore sets `Reduced(s) := Cast(f64 accumulator -> compute dtype)` and runs the later stages
and the epilogue on that. Consequences, stated so nobody has to rediscover them:

- At `F64` compute the cast is the identity, so the kernel is byte-exact for every epilogue.
- At `F32` compute the kernel differs from the oracle by the rounding of the stage values. A case in this mode is
  **opt-in per case**, compares with a bound of **at most 2 ulp** (NaN equal to NaN, infinities exact), and writes
  that bound in the test. Today that is `tests/kernel_row_reduce_agrees.rs` (`MAX_ULPS = 2`; PM ruling 2026-10-29).
- In this mode the oracle is a **tolerance referee**, as it already is for the fused epilogue of
  `eval_contraction`: it says which answers are acceptable, not which bit pattern is the only correct one.
- The tolerance must be non-vacuous: each tolerance case needs a control showing that the rounded and unrounded
  stage values really differ on its corpus (`the_tolerance_is_not_vacuous`), and a mutation that moves the rounding
  point (an `F32` accumulator, a wrong stage register, a mis-divided `Mean`) must turn a test red.
- The stage folds themselves (accumulator width, ascending order, NaN-keeping `Max`/`Min`) stay exactly the oracle's;
  only the hand-off between stages and into the epilogue is rounded.

`Reduction` and `Scan` posts and the `Contraction` epilogue are lifted the same way
(`tests/kernel_reduction_post_agrees.rs`, `tests/kernel_scan_post_agrees.rs`, `tests/kernel_contraction_agrees.rs`;
same bound; a scan post may also read the element at its position).

**The bound is for well-conditioned posts.** Rounding the stage value once moves it by half an ulp, and a post with
condition number `k = |s f'(s) / f(s)|` turns that into about `k / 2` ulp of its result. Measured when the
contraction epilogues were added: `0.37 s + 1` cancels near `s = -2.7` and moved by 71 ulp; `silu` at `s = -5`
(`k` about 5) moved by 3. Neither is a lift defect, and neither fits a 2 ulp bound. A case whose post is that
ill-conditioned must keep clear of the ill-conditioned region in its corpus, or state its own larger bound with this
calculation beside it; it must not loosen the default. (`tanh`, `sin`, `exp` and the like are evaluated at the
compute dtype by the oracle too, so they agree exactly and prove nothing about the tolerance; a tolerance case needs
a post with an arithmetic step, and its non-vacuous control.)

## 8. Which U2 choices constrain U3's `ProgramModel` and resource manifest

Said now so U3 does not discover them by breaking U2. (Fuel#514 section 4 is the input for the `Backend` side;
nothing here answers its five questions.)

1. **`Place::Logical` is the only place in v0.** U3's pointers and memory spaces add `Place` variants; they must
   keep `Logical` meaning "through the operand's view". If U3 wants byte offsets to *replace* it, the lifting and
   the differential in section 7 break, so this is the choice most expensive to reverse.
2. **`Param` is `#[non_exhaustive]` and carries no binding data.** Fuel's corpus needs, per buffer, a descriptor
   binding index, role, storage-vs-uniform and packing, plus a `Params` byte layout (R1, R6). v0 `Param` has name,
   kind and role only. U3 adds the binding manifest *beside* the kernel, not inside `ParamKind`, unless U3 decides
   the opposite; the field set is left open on purpose.
3. **One logical thread.** U3's execution scopes and ids are *new leaves and statements* around this model. If U3
   chooses `BlockSpmd` (Triton: tile-valued programs, `program_id`) as the primary model, the "locals are scalar or
   packed" restriction (3) is too narrow and `Ty` gains a tile type; the checker's equality rule survives.
4. **Locals are untyped-by-space.** All v0 locals are private to the thread. U3's memory spaces (shared, local
   fragments) become a `space` on a declaration; v0's `LocalDecl` is `#[non_exhaustive]` for that.
5. **Subgroup size is not representable** (Fuel#514 R4: 27 files use `Wave*`). Nothing in v0 hard-codes 32; the
   target-property question belongs to the `Backend` capability model, not to `KernelFn`.

## 9. Order of work (each its own PR, red-first)

1. This spec (docs only).
2. (landed, 0.31.0) `ScalarExpr::Local` (breaking) with the match sites, text and DAG carriers, and its own refusals; `kernel.rs`
   types; `check_kernel` with the nine known-bad mutations and their positive controls.
3. (landed, 0.31.3) The interpreter and its direct tests (loops, `Break`, `Continue`, `Return`, `ZeroStep`, wrap of the counter).
4. `lift_elementwise_opdef` + the differential against `oracle::evaluate` (landed, 0.31.3); then Reduction, Scan, Contraction, Im2Col, Window and RowReduce (landed or in the gate; fused posts of the first three are open).
5. Effects (`effects(&KernelFn)`), once a consumer asks; it is stated here so the shape is agreed, not built.
