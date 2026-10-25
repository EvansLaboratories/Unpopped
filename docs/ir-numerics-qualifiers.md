# Numerics qualifiers: rounding mode, flush-to-zero and approximate forms

Status: **design, PM-approved representation (2026-10-25)**. Part of #112 (I1 of
`docs/ir-union-design.md`, row 3.12). This document ships before the code; the code PR carries the
red-first tests named in section 7. It follows `docs/ir-typed-expressions.md` (slice 2 of I1, shipped
in 0.24.0) and is its slice 3.

## 1. What is missing

CUDA, PTX, Triton, CuTe and ThunderKittens all let an author say HOW an arithmetic operation
rounds, not only WHAT it computes: `__fadd_rd(a, b)` (round toward minus infinity), `add.rz.f32`,
`mul.ftz.f32` (denormals flushed), `div.approx.f32`, `__frcp_rn`, `ex2.approx.f32`,
`rsqrt.approx.f32`. Today `Add(a, b)` means "the compute type's default": round to nearest even,
denormals honoured, the exact function. An author cannot ask for anything else, so a kernel that
depends on `__fmul_ru` (interval arithmetic, error-free transforms) or on the fast `ex2` (every
softmax) cannot be expressed. This slice adds exactly those controls, as data.

## 2. The representation

```rust
pub struct Numerics { pub round: RoundMode, pub ftz: bool, pub approx: bool }  // Default = nearest, false, false
ScalarExpr::Qualified(Box<ScalarExpr>, Numerics)
```

`Qualified(child, n)` is the child operation evaluated under `n`. The node's type is the child's type
(`infer` is unchanged: a qualifier does not change a value's type, which is why this is a wrapper and
not a `Cast`-style retyping). The child must be one operation from the table below; anything else is
refused at the plan gate with a typed error naming the child. `Qualified` is not
`#[non_exhaustive]` at the `ScalarExpr` level (as `Fma` and `Cast` are not): backends must update.

| child op | `round != NearestEven` | `ftz` | `approx` |
|---|---|---|---|
| `Add`, `Sub`, `Mul`, `Fma` | yes | yes | no |
| `Div`, `Sqrt` | yes | yes | yes |
| `Rsqrt`, `Recip`, `Exp2`, `Log2` | no | yes | yes |

`approx` and a directed `round` never combine (an approximation has no rounding direction to honour);
that pair is `PlanError::InadmissibleOpAtDtype` with both named. v1 is **f32 / f32-strict compute
only** and **`OpKind::Elementwise` only**, the same envelope as `Cast`; f64 has no `ftz` form on the
targets that have the flag (PTX denormal control for f64 does not exist), and directed f64 rounding
needs a wider reference than this slice builds. Both are listed in section 8.

## 3. Rounding mode

`round` is the IEEE 754 attribute of the single operation: the result is the exact real result of the
operation on its (already-rounded) operands, rounded once in the stated direction.
`NearestEven` is today's behaviour and the default. For the directed modes:

- `TowardZero`, `TowardNegInf`, `TowardPosInf` round the exact result; an exact result is returned
  unchanged in every mode.
- **Overflow** follows IEEE 754 section 7.4: `TowardZero` returns the largest finite value of the
  sign; `TowardNegInf` returns the largest finite value for a positive result and `-inf` for a
  negative one; `TowardPosInf` the mirror image.
- **Zero sign**: an exact zero sum of opposite-signed operands is `+0` in every mode except
  `TowardNegInf`, where it is `-0` (IEEE 754 section 6.3).
- `Fma` is one rounding of `a*b + c` in the stated mode; `Sqrt` is the correctly rounded root in the
  stated mode.

The oracle computes these exactly (section 6): an independent reference built from the exact
residual of the operation, not from the host's rounding mode.

## 4. Flush-to-zero

`ftz` follows PTX's `.ftz` definition so the one flag means one thing on every target:

1. each operand that is subnormal is replaced by a zero of the same sign **before** the operation;
2. the operation is performed (in `round`);
3. a result that is subnormal **after rounding** is replaced by a zero of the same sign.

NaN and infinity are untouched; `ftz` has no effect on a value that is zero or normal. It does not
change `Neg`/`Abs`/`Select`/`Cast` (they are not qualified nodes). Example, exact: with `ftz`,
`Mul(2^-130, 2^100)` is `+0` because the first operand is subnormal (input flush), and
`Mul(2^-100, 2^-30)` is `+0` because the exact result `2^-130` is subnormal (output flush). Without
`ftz` the first is the normal `2^-30` and the second the subnormal `2^-130`. An `ftz` kernel on a host whose FPU already
flushes agrees trivially; the cpu-c spelling flushes explicitly in C (section 7, test 2) so it does not depend
on the host's control register.

## 5. Approximate forms, and what `approx` promises

`approx` is a licence, not a requirement: a backend MAY spell the fast instruction, and MAY spell the
exact function, because the exact function is inside the bound. What it promises is a **bound on the
error against the exact (correctly rounded, nearest-even) result**, so a backend that spells an
approximation can be checked against the oracle.

**The bound (the IR's contract, deliberately looser than any current hardware so that it can be met
and is not a moving target):**

- `Div`, `Recip`, `Sqrt`, `Rsqrt`, `Exp2`: `|got - exact| <= 4 ulp(exact)` for f32, ulp measured at
  the exact result.
- `Log2`: `|got - exact| <= max(4 ulp(exact), 2^-20)`: near `log2(1) = 0` an ulp bound is
  unmeetable by any hardware approximation, so an absolute floor is stated.
- **Domain**: finite, normal operands whose exact result is finite and normal. Outside it (NaN,
  infinities, zeros, subnormals in or out, overflow) `approx` promises nothing, exactly as fast-math
  promises nothing; an author who needs those cases must not set `approx`, and the oracle returns the
  exact value there.

**Stated assumption.** 4 ulp (and the `Log2` floor) is chosen, not measured: it is looser than the
documented error of the PTX fast forms as the author recalls them, and no GPU measurement backs it
yet. If a backend measures a fast form WORSE than the bound, that is a `[FINDING]` routed to the PM,
never a quiet loosening here or a silent exemption in the backend's test; the bound changes only by
a ruling that also bumps this document.

A target whose fast instruction cannot meet this bound on this domain **declines `approx`** for that
op; it does not weaken the promise. baracuda owns that measurement for its PTX forms. The helper
`unpopped::numerics::approx_within(op, got, exact)` encodes the bound once, with a control proving it
can fail (a result off by `2^-18` relative is rejected), and is the checker every backend test uses.

The `cpu-c` emitter spells `approx` as the exact function (inside the bound), says so in its docs,
and the end-to-end test asserts the result is within the bound AND is non-vacuous (the same test
asserts `approx_within` rejects a deliberately sloppy value).

## 6. Seams, plan gate, oracle, contract

- **`Lowering::qualified`** (+ builder method), default a typed decline
  (`DeclinedOp::Qualified`, naming the child op and the qualifier). It receives the child's op, the
  already-lowered operand strings, the `Numerics` and the compute dtype. **A backend that cannot
  honour a qualifier declines; it never lowers the unqualified op** (silently dropping `ftz` or a
  rounding mode would make `Qualified` mean nothing). Test: the default seam declines, cpu-c declines
  directed rounding, Slang declines everything, and in each case the decline names the qualifier.
- **Plan gate** `check_qualified_admissibility` (after `check_typed_admissibility`): the table in
  section 2, f32/f32s, Elementwise; violations are typed `PlanError`s, never panics.
- **Optimizer**: a body containing `Qualified` is returned unchanged, exactly as typed expressions
  are, and for the same reason (a rewrite across the node would change which operations are rounded
  how). Pattern derivation returns `PatternError::TypedExprUnsupported`'s sibling
  `PatternError::QualifiedUnsupported` (FKC 4.1 has no qualified op); recipes return `None`.
- **Oracle**: `Qualified` evaluates the child's operands, applies section 4 (input flush), computes
  the operation in the stated mode, applies output flush. The directed result is built from the
  operation's exact residual: for `Add`/`Sub`/`Mul`/`Fma` an error-free transform in f64 (the f32
  operands make the f64 product exact, and the f64 sum's error term is exact), for `Div`/`Sqrt` the
  sign of the f64 remainder computed with a fused multiply-add. The test for this reference does not
  use it to check itself: it compares `Add`/`Sub`/`Mul` against exact `i128` big-integer
  arithmetic over a grid of operands, and checks `Div`/`Sqrt`/`Fma` by the order law
  `RD <= RN <= RU`, `RU - RD <= 1 ulp`, exactness (`RD == RU` iff the result is exact), and
  `RZ == RD` for positive and `RU` for negative results.
- **Contract (`ulp_sum`)**: a `Qualified` node contributes its child's budget; a directed round adds
  one ulp (the error is below one ulp, not half); `approx` adds 4 ulp (the section 5 bound);
  `ftz` adds nothing for normal results (its effect is a flush the budget cannot express and the
  contract's input domain excludes subnormals, as it already must for the unqualified op).

## 7. Tests (each can fail, each has the control that proves it)

1. **`Qualified` unchanged is the child**: with `Numerics::default()`, `Qualified(Add(a, b))` and
   `Add(a, b)` give bit-identical results over the edge-value grid in the oracle AND through cpu-c
   (the back-compat obligation, with the control that a directed mode differs on the same grid).
2. **ftz through the real compiler** (PM condition c): cpu-c compiles and runs `Mul` with `ftz`,
   operands/results chosen so each of the three steps of section 4 is individually exercised: a
   subnormal input, a subnormal result from normal inputs, a negative subnormal (sign kept), and
   the control that the same kernel WITHOUT `ftz` returns the nonzero subnormal. Seeded mutations:
   drop the input flush, drop the output flush, flush to `+0` unconditionally.
3. **Decline, never ignore** (PM condition b): default `Lowering` declines `Qualified`; cpu-c
   declines each of the three directed modes with a decline naming the mode, and spells
   `NearestEven`/`ftz`/`approx`; Slang declines `Qualified` and still generates the unqualified
   form. The test asserts the declined output is NOT the unqualified spelling (it fails against a
   mutation that drops the qualifier).
4. **Directed-rounding reference** (section 6), plus a directed overflow/zero-sign table from
   IEEE 754 section 7.4/6.3 written out as literals.
5. **`approx` bound** (PM condition a): `approx_within` accepts the exact value, accepts a value
   `3 ulp` off, rejects `5 ulp` off, rejects `2^-18` relative error; the cpu-c `approx` kernel is
   within the bound over a 200-value grid per op; a target that is out of bound is the case the
   checker exists to catch.
6. **Plan gate**: each illegal cell of the section 2 table is a typed `PlanError` naming it;
   f64 compute, non-elementwise and `approx`+directed are refused; the legal cells plan.
7. **Optimizer unchanged / pattern and recipe declines**, as for `Cast`.

## 8. Not in this slice (and why)

- f64 `ftz` (no target form) and f64 directed rounding (needs an exact-residual reference wider than
  an f64 error-free transform gives for `Div`/`Sqrt`); f16/bf16 compute.
- Qualifiers on `Unary` ops other than the table (`Exp`, `Sin`, `Tanh`...): `Exp`/`Log`/`Sin`/`Cos`
  have `approx` forms (`__expf`, `__sinf`) whose error is absolute near zeros and large arguments
  and cannot be stated in the section 5 shape; they wait for a bound each, with the domain, from the
  target that has them.
- Directed rounding spelled in cpu-c (needs `fesetround`, global state) or by emulation; the CUDA
  spellings (`__fadd_rd`, `add.rz.f32`, `mul.ftz.f32`, `div.approx.f32`) are baracuda's, via the
  seam.
- Qualifiers inside reductions: arrive with the reduction type story (`Cast` has the same limit).
- Stochastic rounding (needs a random-bits operand); `RoundMode` stays `#[non_exhaustive]`.
