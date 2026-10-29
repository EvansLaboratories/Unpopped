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
reviewable. fp4/fp6/MX element kinds (`F4E2M1`, `F6E2M3`, `F6E3M2`) became `ElementKind` rows in 0.29.0 as storage and key identity only (D5; PM go 2026-10-28); no `Cast` row or codec exists for them yet.

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

## 9. Packed values (slice 4, steps 1, 2, 3a and 3b landed)

Status: **steps 1 and 2 landed**: `Pack`, `Extract`, `Ty`, `infer_ty`, the DAG/text carriers, and lane-wise `Add`/`Sub`/`Mul`/`Fma`/`Min`/`Max`, whole-value `Select` (scalar condition) and per-lane `Cast`, with `types::lane_of` as the reference meaning of `Extract` (the oracle evaluates it). step 3a adds `Bitcast` of a packed value to a scalar of equal total bits (`Packed(I8, 4)` -> `U32`; lane 0 is the least significant; `types::check_packed_bitcast`, evaluated by the oracle). Step 3b adds the inverse as a new node `BitcastPacked(x, elem, lanes)` (PM ruling (a), 2026-10-28; breaking, 0.28.0): the same width rule, lanes 2 or 4, scalar operand; the oracle slices `Extract(BitcastPacked(..), i)` from the operand's bits and `lane_of` declines it (no scalar shift node). `Packed(F16, 2)` <-> `U32` is typeable since 0.30.1, when `F16`/`Bf16` got `Cast` rows (the plan gate still declines every packed body). A packed `Select` condition and every emitter seam are later steps; until they land the plan gate declines any body containing a packed node. `infer` refuses a packed result (`PackedResult`); `infer_ty` is the packed-aware form (the PM ruled option (b), 2026-10-25). This section refines section 6.

**Type.** A packed value is `Packed(elem, lanes)` with `elem` a scalar float or integer `Cast` accepts
(`is_cast_float` / `is_cast_int`, plus `F16` and `Bf16` once their `Cast` rows exist) and `lanes` in
`{2, 4}`. `infer_ty` returns `Ty::Scalar(ElementKind)` or `Ty::Packed(ElementKind, u8)`; `infer` keeps
returning `ElementKind` and refuses a packed result with `TypeError::PackedResult` (so no existing
caller changes meaning). Which of the two the PM rules is `[ASK] packed-values-Ty`.

**Nodes** (new `ScalarExpr` variants, so breaking; every match site chooses to handle or decline):

| Node | Type | Rule |
|---|---|---|
| `Pack(Vec<ScalarExpr>)` | `Packed(t, n)` | 2 or 4 children, all scalar of one type `t`. Lane `i` is child `i`. |
| `Extract(x, i)` | `t` | `x` is `Packed(t, n)`, `i < n`, else `TypeError::BadLane`. |
| `Add/Sub/Mul/Fma/Min/Max` | operand type | Lane-wise. Operands must have identical `Ty`; integer lanes wrap exactly as section 8. |
| `Cast` / `Bitcast` | per lane / by width | `Cast` converts each lane (`cvt.rn.f16x2.f32` is `Cast` of `Pack(a, b)`); `Bitcast` needs equal total bits (`Packed(F16, 2)` <-> `U32`). |
| `Select(c, a, b)` | type of `a` | `c` scalar: whole-value select. `c` packed with the arms' lane count: lane-wise mask, lane `i` of the result is `a`'s lane `i` when `c`'s lane `i` is non-zero, else `b`'s (each mask lane tested `!= 0` in its own type, like a scalar condition; `c`'s lanes may be any scalar type). A packed `c` over scalar arms, or with a different lane count, is `PackedOperand`. |

Everything else (Div, Unary, comparisons, `Bitcast`, `Qualified` over a packed operand) declines with
`TypeError::PackedOperand` until a backend needs it; declining is the safe direction.

**Lowering.** An emitter that has a native spelling (`half2`, `__half2`, `float4`) uses it; the
reference spelling is lane-by-lane scalar code, which is what the oracle and cpu-c do, so a packed
expression always has a correct fallback and the differential harness compares native against it.

**Order of work (red-first).** (1) `Ty`, `infer_ty` and the `TypeError` arms with tests that fail on
the empty enum; (2) `Pack`/`Extract` in `ScalarExpr`, text format, optimizer (opaque, like `Qualified`),
oracle; (3) cpu-c and Slang lowering or typed decline; (4) lane-wise arithmetic. Each step is its own PR.

## 10. Casts into and out of the OCP MX element formats (`F4E2M1`, `F6E2M3`, `F6E3M2`)

Status: **PM ruling A, 2026-10-28** ("saturate-always"). These formats have IEEE-shaped fields but **no
infinity and no NaN**: the all-ones exponent is an ordinary exponent and the largest code is the largest
finite value (e2m1 6, e2m3 7.5, e3m2 28; values checked against the OCP MX v1.0 tables, KISS R1.30). So an
overflow or a NaN has nowhere to go, and the IR defines where, exactly as it does for float to integer.

| Conversion | Rule |
|---|---|
| any float or integer **into** an MX element | `Saturation::ClampFinite` is the only legal `sat`; `Wrap` and `Clamp` are `TypeError::IllegalCast` naming the fix. One rounding from the exact value under the stated `RoundMode` (all four). Every overflow, an infinity included, lands on the largest finite value of its sign. **NaN becomes `+0`.** The sign of a zero, and of a value that rounds to zero, is kept (`-0` stays `-0`). |
| an MX element **out of** to a float or integer | The ordinary rules (exact widening; float to integer takes `Clamp`). `rounding_matters` compares exponent range AND significand. |
| `Bitcast` of an MX element | No row: the typed bit widths are byte-multiples only. |

**NaN to `+0` is the IR's rule, not a hardware fact.** What the vendors' conversion instructions do with a
NaN here is UNVERIFIED (no PTX or Vulkan check has been made; that waits for a consumer). An emitter that
cannot reproduce the rule exactly MUST DECLINE, never approximate. No emitter spells these casts today (CpuC has no
C type for them and declines; Slang is unchanged).

**Where an MX value may live.** Only as an intermediate directly under a `Cast` (the fake-quantise round trip
`Cast(Cast(x, F4E2M1, ClampFinite), F32)` is the use). It may not be the stored result or a compute dtype: there
is no buffer layout for sub-byte floats (the 6-bit packing is its own slice), and the plan gate declines a root or `out_dtype` of an MX
type with a typed reason rather than letting the oracle's buffer sizing panic.

**Evidence.** `cast_ref` tests compare the encoder with a brute-force search of the explicit value grid for
every mode over a few thousand probes, plus the saturation, NaN, `-0` and clamp-edge cases; `typed_expressions` tests the
legality table and the oracle end to end. Seeded mutations (accept `Wrap`, NaN to max, no clamp on carry,
drop the sign) each turn a test red.
