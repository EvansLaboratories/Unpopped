# Typed expressions: the spec behind `Cast`, `Bitcast`, integer `Coord`/`Select`, packed values and index dtypes

Status: **design, PM ruling A (2026-10-25)**. Part of #112 and #113 (I1, I2 of
`docs/ir-union-design.md`). No code in this document's PR; the code PR carries the red-first tests.

## 1. The constraint being removed

Today an op computes in ONE dtype. The oracle evaluates every node against `ev.dtype`
(`oracle.rs` `arith` branches on `is_complex(ev.dtype)` / `is_int(ev.dtype)`), the plan carries one
`plan.dtype`, and every emitter seam receives it (`cpu_select(c, a, b, plan.dtype)`,
`cpu_fma(.., plan.dtype)`). `Input(i)`, `Const`, `Param`, `Coord` and every operator are implicitly
that dtype. That is why `Cast` cannot be added as one more node: `Cast(x, f16)` inside an f32 body
has no type to be, and `Add(f32, f16)` has no rule. The union directive needs conversions, bit
reinterpretation, integer index arithmetic and packed lanes inside one expression, so the
expression language must carry types.

## 2. The model

**Every `ScalarExpr` node has a type**, an `ElementKind` (scalars) or, for packed values (section 6), a
`(ElementKind, lanes)` pair. A pure function `infer(expr, ctx) -> Result<Ty, TypeError>` computes it.
`ctx` holds the op's compute dtype `C`, its per-input dtypes and the output dtype.

| Node | Type | Rule |
|---|---|---|
| `Input(i)` | `C` | Unchanged: loads convert to `C` today. A later increment may let an op declare per-input dtypes, which `Input(i)` then returns, with an explicit `Cast` required to use it at another type. |
| `Const`, `Param`, `Coord` | `C` | Unchanged. |
| `CoordInt(axis, T)` (later slice) | `T` in `{I32, I64, U32}` | Exact integer coordinate, no float round-trip. Ships with `ConstInt` and integer arithmetic, because alone it is only `Cast(Coord)`. |
| `Add/Sub/Mul/Div/Unary/Binary/Fma` | operand type | **Operands must have identical types**; a mismatch is `TypeError::Mismatch`, never an implicit promotion. |
| `Select(c, a, b)` | type of `a` (= type of `b`) | `c` may be any scalar type (tested `!= 0`); arms must match. Integer arms become legal once the integer cond-observer question (U8/I8) is closed by this rule: the cond is the typed expression, so its type is explicit. |
| `Cast{x, to, round, sat}` (new) | `to` | Section 3. |
| `Bitcast{x, to}` (new) | `to` | Section 4. |
| comparisons | `C` (0/1) | Unchanged in this increment; a `Bool`-typed result is a follow-up so existing `Cmp*` masks keep their bits. |

The body's type must equal the op's store type (`out_dtype` or `C`); otherwise `TypeError::StoreMismatch`.

**Backward compatibility is a proof obligation, not a hope.** An expression containing no `Cast`,
`Bitcast` infers to `C` at every node, so every existing `OpDef` types exactly as it
does today. The code PR proves it: (a) `infer` over every op the repo's tests and goldens construct
returns `C` everywhere (a test enumerating them), (b) the text-format round trip and every golden
are byte-identical, (c) the oracle's existing differential suites stay green with the oracle
re-plumbed from `ev.dtype` to the inferred node type.

## 3. `Cast`

`Cast { x, to, round, sat }`: convert the VALUE of `x` to type `to`.

- `round: RoundMode` in `{NearestEven, TowardZero, TowardNegInf, TowardPosInf}`; consulted only when
  the conversion loses precision (float to narrower float, float to int, int to float). Stochastic
  rounding (needs a random-bits operand) is its own follow-up; the enum is `#[non_exhaustive]`.
- `sat: Saturation` in `{Wrap, Clamp, ClampFinite}`.
  - integer to narrower integer: `Wrap` = two's-complement truncation, `Clamp` = saturate to the
    destination range.
  - float to integer: always saturating with NaN to 0 (this IR defines the semantics so no emitter
    reaches C's undefined behaviour); `sat` must be `Clamp`, anything else is a typed rejection.
  - float to narrower float: `Wrap` = overflow to infinity (per `round`), `ClampFinite` = clamp to
    the largest finite value, NaN stays NaN. `Clamp` (to [0,1]) is reserved and rejected until a
    kernel language needs it.
  - to a type with no infinity (fp8 `FN`/`FNUZ` variants): overflow is defined by the type's
    saturation convention, and the legality table below says which `sat` each pair accepts.
- **A legality table** (source type x destination type, which `round`/`sat` are meaningful) lives
  in `unpopped-vocab` next to `ElementKind`, as data, and is the single source for the plan gate,
  the oracle and emitter declines. Pairs not in the table are `PlanError::InadmissibleOpAtDtype`
  with the pair named, never a silent lower.
- Oracle: the existing independent codecs in `unpopped-vocab` (fp8, half, sub-byte) are the
  reference; `Cast` is added to the oracle by calling them, and the differential checks the
  emitted C against that, exhaustively over every source bit pattern for 8- and 16-bit sources (as
  the fp8 and half suites already do), and by fixed edge vectors for 32/64-bit sources.

## 4. `Bitcast`

`Bitcast { x, to }`: reinterpret the bits; legal only when `bit_width(type(x)) == bit_width(to)`
(complex counts as its full width). It moves bits and computes nothing: NaN payloads and signed
zero survive, exactly as `Select` arms do. Oracle: the `Val::Raw` representation already carries bit
patterns, so it is a retag. Emitter seam: a memcpy-style spelling per target (`memcpy`/`as`
reinterpret in C, `asuint`/`asfloat` in Slang), not a pointer cast (strict aliasing).

## 5. The seam changes (what the ~40 match sites become)

- `Lowering` / `LoweringBuilder` gain `cast` and `bitcast`, defaulting to a typed decline exactly as
  `fma` does, so existing emitters compile and decline. The seam receives source and destination
  types and the qualifiers; it no longer receives only `plan.dtype`.
- `ScalarExpr` gains `Cast`, `Bitcast`; `DagNode`, `ENode`, the optimizer weights,
  the pattern/recipe walkers, jit, contract, plan walkers and the cfamily backstop each gain an arm
  (the compiler lists them; `ScalarExpr` is deliberately not `#[non_exhaustive]`). The optimizer gets
  ZERO rules for the new nodes in the first PR, as for `Select` and `Fma`; folding `Cast(Cast(x))`
  is lossy in general and is added only with a proof per pair.
- The oracle's `Val` already distinguishes Float/Int/Complex/Raw; the change is that dispatch is on
  the inferred node type rather than `ev.dtype`.

## 6. Packed and vector values (second slice)

`(ElementKind, lanes)` with lanes in `{2, 4}` for `half2`, `bf16x2`, `float4`, `int4`-style
vectors; lane-wise operators, `Extract(x, i)` / `Pack(a, b, ..)`, and `Cast` between packed forms
(`cvt.rn.f16x2.f32`). Typed this way, a `float4` is not a special case of the scalar rules but a
consequence of them. Deferred to the slice after scalar `Cast` so the first breaking PR stays
reviewable. fp4/fp6/MX element kinds are D5 (CireSnave, board 290) and stay out.

## 7. Increments

1. `infer` + `TypeError` (crate-private), wired into `try_build_plan` as a validator that every
   existing op passes; oracle re-plumbed from `ev.dtype` to node types; **no new nodes**. Behaviour
   unchanged; this is the safe half and is not breaking.
2. `Cast`, `Bitcast` (first typed nodes; arithmetic stays `C`-typed in v1, gated at plan time) with the legality table, oracle, cpu-c spellings and decline-by-default
   seams. Breaking (new `ScalarExpr` variants): one pre-1.0 minor.
3. Rounding / ftz / approx qualifiers on `Unary`/`Binary`.
4. Packed values (section 6).
5. `CoordInt`, `ConstInt` and integer arithmetic inside a float op (the full integer-index story), after slice 2. **Landed**, section 8.

## 8. Integer index expressions (slice 5, landed)

`CoordInt(axis, T)`, `ConstInt(v, T)` and integer arithmetic inside a float op. `T` is `I32`, `I64` or
`U32` for the leaves (`check_int_leaf`; a constant must be in range); the arithmetic also accepts the
other integer types a `Cast` produces (`I8`, `I16`, `U8`, `U16`).

**What exists at an integer type** (every other operator refuses an integer operand with
`TypeError::IntegerOperand { node, ty }`, and an integer compute dtype is unaffected, because the
refusal applies only to a type that differs from the compute dtype):

| Node | Result | Semantics |
|---|---|---|
| `CoordInt(axis, T)` | `T` | The exact output coordinate: no float round-trip, so no 2^24 bound. Elementwise-only, `axis < rank`, lowers via `Schedule::Strided`, like `Coord`. |
| `ConstInt(v, T)` | `T` | An exact constant. |
| `Add`, `Sub`, `Mul` | `T` | **Two's-complement wrapping** at `T`'s width. Defined, never C's undefined signed overflow: an emitter widens to an unsigned type, operates, converts back. |
| `Binary(Min \| Max)` | `T` | The smaller or larger value (NaN rules do not arise). |
| `Binary(CmpEq .. CmpGe)` | `C` | The compute type's `1`/`0`, exactly as a float comparison: an integer mask feeds a float `Select` or a `Cast`. |
| `Select(c, a, b)` | `T` of the arms | `c` is any scalar, tested `!= 0` in its own type (an integer by value, a float rounded to the compute dtype first); the arm moves unconverted. |

Division is deliberately absent: integer division by zero and `MIN / -1` need a defined result per
target, which is its own decision. Bitwise and shift operators, and `Cast`-free mixed-width
arithmetic, are later slices.

**Gate.** An integer value is admitted anywhere `infer` accepts it, not only at the root or under a
`Cast`: that restriction (section 2, increment 2) existed because float arithmetic on a non-compute
value had no spelling, and integer arithmetic now has one. A non-integer non-compute value (an `f64`
under `f32` compute) keeps the old rule. The body's type must still equal the store type.

**Seams.** `Lowering` gains `coord_int`, `const_int` and one `int_op(IntOp, T, operands)`; all default
to a typed decline (`DeclinedOp::Integer`). A walker routes `Add`/`Sub`/`Mul`/`Min`/`Max`/comparisons/
`Select` to `int_op` when the first value operand is integer-typed, and a hoisted integer temporary is
declared with `type_name(T)`. cpu_c spells the constants and every operation; it has no strided
schedule, so it declines `CoordInt` through the default and the leaf is exercised against the oracle
and by seam-level tests. **Baracuda's CUDA emitter owns `CoordInt` there** (`coord_int`, `int_op`,
`const_int`), as it owns `Coord`.

**Optimizer.** `is_typed` is true of both leaves, so a body containing one is returned as authored
(zero rules), the same bypass as `Cast`.

**Evidence.** `tests/integer_index_expressions.rs` (type rules with controls, gate, oracle at the
types' extremes) and `unpopped-cpu-c/tests/cpu_end_to_end.rs`
`integer_index_arithmetic_wraps_like_the_oracle_in_a_real_compiler`, which compiles each operation at
`I32`/`U32`/`I64` over operands that include both extremes and compares with the oracle's exact `i128`
evaluation wrapped to the width.
