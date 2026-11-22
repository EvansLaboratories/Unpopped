# Integer division, remainder and shift: one IR rule, measured per target

**Status: RULED 2026-10-31 as recommended; implemented at the kernel layer in 0.38.3.** PM task 2026-10-31: per target, what
division by zero, `MIN / -1` and a shift count `>= width` do, measured from each spec with the section cited; then the IR
rule and which targets must decline. The ruling: rules 1-3 stand; open point 1, keep `N = 32` for `S8`/`U8`
(promote-then-truncate, unchanged behaviour, stated in the `Shl`/`Shr` docs); point 2, WGSL's zero divisor giving the
dividend, accepted; point 3, integer `Rem` is this lane's call but must give `0` for `b == 0` and for `MIN / -1`, which
`a - (a / b) * b` alone does not for a zero divisor (it gives `a`).

**What is done (0.38.3).** The kernel interpreter evaluates `Div`, `Shl` and `Shr` on `I32`/`I64`/`U32`/`U64` by these rules;
the cpu-c *kernel* emitter spells them through total helpers (`unp_div_*`, `unp_shl_*`, `unp_shr_*`); boundary vectors run
the interpreter against host-native integers and the compiled C against the interpreter, with a seeded mutant per guard.
**Integer remainder** is not a new operation: it is `Select(b == 0, 0, a - (a / b) * b)` over the total `Div` (`MIN % -1`
already comes out `0` because `MIN / -1` wraps to `MIN`); the test pins it against `checked_rem(..).unwrap_or(0)`.

**Plan-level oracle (0.39.1).** `oracle::evaluate` follows rule 3 for `Shl`/`Shr` too (openclownfish#26).

**What is not done.** The intent-layer plan gate still refuses integer `Div` (an `OpDef` an emitter will lower through
`Backend::lower`), and `Shl`/`Shr` there keep the architecture-inherited caller contract, because the emitters outside this
repository (Baracuda CUDA, Vulkane Slang/SPIR-V, openclownfish OpenCL C, Rockfish HIP) would otherwise receive bare `/` and
`<<`. Each adopts the guards on its own schedule with the same boundary vectors (GPU targets on the self-hosted runner, never
`#[ignore]`); the gate opens when they have. ROADMAP E16.

**The gate opens per backend (0.40.7).** `Backend::total_integer_ops()` defaults to `false`; a backend that returns `true` is
saying its `arith`/`binary`/`const_int` seams spell the rules above, and only then does `try_generate` admit an Elementwise
integer `Div` at `I32`/`I64`/`U32`/`U64`, a `ConstInt` leaf at an integer compute dtype and (0.40.10) an Elementwise `Select` and comparison at those dtypes, so the integer `Rem` composition below plans. A backend that does not override it is
refused exactly as before (`tests/integer_div_gate_is_opt_in.rs`). This document stays the single source of the semantics; the
method's docs and the CHANGELOG point here rather than restate them. `Shl`/`Shr` were already legal at `I32`/`I64` (8- and
16-bit operands stay leaf-only) and are unchanged. Opted in so far: Rockfish (first) and Vulkane (2026-11-11: `vulkane-spirv-emit`
returns `total_integer_ops() = true`, with guarded `Div`/`Rem`/`Select`/comparison/`ConstInt`, vulkane#152; its 288 device
results match the host rule, as reported by the PM); Baracuda and openclownfish are unaffected and opt in separately.

_The note below is the measured design as written before the ruling._

## Where the IR stands today

- Integer `Div` is **rejected** at the plan gate for `I32`/`I64`/`S8`/`U8` (`ir.rs` op x dtype table: "C `/` div-by-zero is
  device-UB"). `docs/ir-typed-expressions.md` section 8 leaves it out on purpose: "integer division by zero and `MIN / -1`
  need a defined result per target".
- `Shl`/`Shr` exist for `I32`/`I64`/`S8`/`U8` but an out-of-range count "inherits the architecture's behavior, same caller
  contract" (`BinaryOp::Shl`/`Shr` docs). That is a contract on the caller, not a defined result, so two conforming
  backends may disagree and nothing can catch it.
- `Add`/`Sub`/`Mul` already wrap (two's complement; the emitter widens to unsigned and converts back). Division and shifts
  are the integer operations still without a total definition.

## What each target says (read 2026-10-30 from the spec text itself)

| target | `x / 0`, `x % 0` | `MIN / -1` | shift count `>= N` (or negative) |
|---|---|---|---|
| **C11** (cpu-c), N1570 | UB: 6.5.5p5 "if the value of the second operand is zero, the behavior is undefined" | UB: 6.5.5p6 "otherwise, the behavior of both a/b and a%b is undefined" | UB: 6.5.7p3 "negative or ... greater than or equal to the width of the promoted left operand"; `<<` of a negative or overflowing signed value is also UB (6.5.7p4); signed `>>` of a negative is implementation-defined (6.5.7p5) |
| **C++ draft** (CUDA, HIP, SYCL source), `eel.is/c++draft` | UB: [expr.mul]/4 "If the second operand of / or % is zero, the behavior is undefined" | UB: [expr.mul]/4 "otherwise, the behavior of both a/b and a%b is undefined" | UB: [expr.shift]/1 "negative, or greater than or equal to the width of the promoted left operand" |
| **PTX ISA 9.4** (CUDA's ISA) | `div` 9.7.1.9: "Division by zero yields an unspecified, machine-specific value" (`rem` 9.7.1.10 not quoted here) | not stated in `div` | **clamped, not masked**: `shl` 9.7.9.8 and `shr` 9.7.9.9: "Shift amounts greater than the register width N are clamped to N" |
| **LLVM LangRef** (what HIP and SYCL compile through) | `udiv`/`sdiv`: "Division by zero is undefined behavior" | `sdiv`: "Overflow also leads to undefined behavior ... -2147483648 by -1" | `shl`: "equal to or larger than the number of bits in op1 ... returns a poison value" |
| **OpenCL C 3.0** | 6.5.1: "A divide by zero with integer types does not cause an exception but will result in an unspecified value" | 6.5.1: out-of-range division "will result in an unspecified value" | **masked**: 6.5.10 "left-shifted by log2(N) least significant bits in E2" |
| **GLSL 4.60.8** | `%`: 5.9 "undefined for any component computed with a second operand that is zero"; for `/` the text I read states only overflow (next column), not a zero divisor | 4.1.3: "Division resulting in overflow will result in an undefined value" | 5.9: "undefined if the right operand is negative, or greater than or equal to the number of bits in the left expression's base type" |
| **SPIR-V** (unified1) | 3.3.13 `OpUDiv`/`OpSDiv`/`OpSRem`/`OpSMod`: "Behavior is undefined if any component of Operand 2 is 0" | `OpSDiv`/`OpSRem`/`OpSMod`: undefined when Operand 2 is -1 and Operand 1 is the minimum value | 3.3.14 `OpShiftLeftLogical`/`OpShiftRightLogical`/`OpShiftRightArithmetic`: "The resulting value is poison if Shift is greater than or equal to the bit width" |
| **WGSL** (W3C candidate recommendation draft, 8.8 and 8.10) | **defined**: `/` gives `e1`, `%` gives `0`. A const-expression zero divisor is a shader-creation error, an override-expression a pipeline-creation error | **defined**: `/` gives `e1`, `%` gives `0`; the same creation errors when both operands are const or override | **masked**: "The number of bits to shift is the value of e2, modulo the bit width of e1"; a const-expression count `>= width` is a shader-creation error. WGSL has no 64-bit integer (no `i64`/`u64` in the spec text) |

**Not measured:** I did not read the CUDA Programming Guide, HIP or SYCL language pages; "CUDA, HIP and SYCL inherit C++"
is their documented basis, not something I re-checked. PTX `rem` and the zero-divisor wording of `OpSRem`/`OpSMod` were
read only in part. Treat those cells as lower confidence than the rest.

Two facts shape the rule. **Only WGSL defines these cases**: every other target leaves division by zero and `MIN / -1`
undefined or unspecified, and in the C family that is real undefined behavior an optimizer may exploit. And **the targets
that define the shift count disagree**: OpenCL C and WGSL mask (modulo the width); PTX clamps.

## Recommended IR rule (direction (a), accepted in principle)

A total definition, chosen to be bare (no extra instruction) where a target already defines it and a short guard elsewhere.
Applies to every integer width the IR admits; `N` is the operand's bit width (for `S8`/`U8` see open point 1).

1. **Unsigned `a / b`**: `b == 0` gives `a`; otherwise the quotient. **`a % b`**: `b == 0` gives `0`.
2. **Signed `a / b`**: `b == 0` gives `a`; `a == MIN` and `b == -1` gives `a` (that is `MIN`, the wrapping result, consistent
   with `Add`/`Sub`/`Mul`); otherwise the quotient truncated toward zero. **`a % b`**: `b == 0` gives `0`; `a == MIN` and
   `b == -1` gives `0`; otherwise `a - (a / b) * b`, with the sign of the dividend.
3. **`Shl`, `Shr`**: the count's bits are read as unsigned and taken **modulo `N`** (`count & (N - 1)`, which also defines a
   negative count). `Shl` discards the bits shifted out and is defined for signed values (no UB for a negative or
   overflowing `a`). `Shr` is arithmetic on signed types and logical on unsigned, as `Shr`'s docs already pin.

Why: (1) and (2) are WGSL's text, so the one target that defines the cases needs no guard, and `MIN / -1 = MIN` is the
wrapping answer the IR already uses for `Mul`. Modulo is what OpenCL C and WGSL already do and defines a negative count in
one expression. PTX's clamp is the odd one out; the cost lands on PTX (one `and` before `shl`/`shr`), not on the targets
that already mask.

## What each target emits (none has to decline for these operations)

| target | division and remainder | shift |
|---|---|---|
| WGSL | bare `/` and `%`, **but** never a const-expression zero divisor or const/const `MIN, -1` operands (creation error): emit a guard or a non-const value | bare `<<` and `>>`; a const-expression count `>= N` is a creation error, so mask it at emit time |
| OpenCL C | guard (`b == 0 ? a : ...`); a bare `/` is unspecified, not trapping | bare (it masks) |
| C (cpu-c), CUDA, HIP, SYCL | guard in a conditional that does **not evaluate the division when the guard fails** (a select over both arms is UB) | `a << (b & (N-1))`, widened to unsigned for `<<` so a negative or overflowing `a` is defined |
| GLSL, SPIR-V | guard (undefined, not trapping, so an unconditional select is fine) | mask the count |

A guarded form costs two compares and a select (signed) or one (unsigned) per division. Where the divisor is known nonzero
(a `Const`) an emitter may drop the guard; that is an emitter detail and must not change a result. A target that cannot
spell the width declines as it does today: WGSL has no 64-bit integer, so `I64` division and shifts decline there for that
reason, not for this rule.

## Open points for the ruling

1. **8-bit operands.** `S8`/`U8` promote to `int` and the shift is taken at 32 bits today (promote-then-truncate, in
   `Shl`/`Shr`'s docs). Recommend keeping `N = 32`: modulo 8 would silently change existing results and is a breaking
   behavior change. Say if you want modulo 8.
2. **Division by zero giving the dividend** is WGSL's choice, not a mathematical one. The alternatives (0, or all-ones as
   RISC-V does) are equally arbitrary and would put the guard on WGSL too. Recommend WGSL's.
3. **Remainder.** The IR's `Rem` is a float op today. This rule defines the integer meaning; whether integer `Rem` becomes a
   `BinaryOp` or is spelled from `Div` is a separate small decision.
4. **KISS.** Whether KISS wants to state these cases (C7 is open) does not block this: the rule lives in the IR and the
   oracle, and a KISS clause, if one appears, would only need to agree.

## Proof obligations if ruled

- The oracle gets the definition (the explicit cases above); the kernel interpreter reuses it.
- A boundary-vector test per emitter, the same vectors everywhere: divisors `0, 1, -1, MIN`, dividends `0, 1, -1, MIN, MAX`,
  counts `0, 1, N-1, N, N+1, 255, -1`, for each integer dtype the emitter spells. cpu-c runs on GitHub against the oracle;
  the GPU targets run on the self-hosted runner (CireSnave 5.1g: never `#[ignore]`).
- A mutation check per guard (drop the zero guard, drop the `MIN, -1` guard, drop the mask): each must turn a vector red.

## Provenance

Every quote was read from the spec text on 2026-10-30: N1570 (port70.net mirror), the C++ draft (eel.is), PTX ISA 9.4
HTML, LLVM LangRef (llvm.org), OpenCL C 3.0 unified, GLSL 4.60.8, SPIR-V unified1, WGSL candidate recommendation draft.
Section numbers are the specs' own headings. Quotes were located by text search of the downloaded pages, not recalled.

## The kernel layer admits these at any integer type (2026-11-22; pm-kernels option A, found lifting Baracuda's decode kernel)

**What changed.** `types::infer_ty_in` (the kernel layer's typing, used by `check_kernel` and the kernel oracle) admits integer `Div`
and the shifts and bitwise operators (`Shl`, `Shr`, `BitAnd`, `BitOr`, `BitXor`) at **any** integer type, not only the kernel's compute
type. Before, a float kernel could not shift or mask an index and a `U32` kernel could not shift a `U64` (the real
`uint_fastdiv::divmod`: `uint32_t((uint64_t(m_) * uint64_t(n)) >> 32)`), though the oracle already evaluated them at every integer type.
`types::infer_ty` (the plan layer) is unchanged: the plan gate still opts integer operators in per backend
(`Backend::total_integer_ops`). The float-only operators (`Pow`, `Rem`, `RemTrunc`, `Atan2`, `Copysign`, `Nextafter`, `FmaxIeee`,
`FminIeee`) and the logical operators stay refused on integers.

**Semantics are the ruled ones above**, at every width `I8` to `U64`, signed and unsigned: `Div` is total; a shift count is read as
unsigned bits modulo the promoted width (64 for a 64-bit type, 32 for every narrower one); `<<` discards what leaves the type; `>>` is
arithmetic on a signed type and logical on an unsigned one. Checked in the oracle by `crates/unpopped/tests/kernel_int_ops_any_width.rs`
(six operators, eight widths, asymmetric operands, counts past the width, an independent reference written from this note).

**Emitters.** `IntOp` (the shared lowering's typed integer seam, `#[non_exhaustive]`) gained `Div`, `Shl`, `Shr`, `BitAnd`, `BitOr`,
`BitXor`, and the lowering routes these nodes to it when the operand type is an integer other than the compute type, so an emitter
that does not spell them gets a typed `UnsupportedOp` decline from the seam instead of the compute-typed spelling (which, before this
change, was a bare C `/` that crashed on a zero divisor in a float kernel: found by the cpu-c differential).
- **cpu-c** spells `Div`, `Shl`, `Shr` at `I32`/`I64`/`U32`/`U64` through the total helpers and the bitwise operators at every width as C
  operators, and declines `Div`/`Shl`/`Shr` at `I8`/`U8`/`I16`/`U16` by name (`cpu_c has no integer div at I8`). 36 of the 48
  (operator, width) pairs lower and every one matches the interpreter byte for byte
  (`crates/unpopped-cpu-c/tests/kernel_emit_int_ops_off_compute_agrees.rs`, with seeded mutants of the emitted C).
- **Read at `origin/main` on 2026-11-22, not run:** `baracuda-cuda-emit/src/kernel.rs` refuses an integer `Div` by name ("integer
  `Div` is not lowered yet") and reaches a shift or bitwise `Binary` only through its `other =>` arm ("expression node `Binary` is not
  lowered by the CUDA backend yet"); its locals are limited to `f32`/`f64`/`i32`/`u32`/`i64`. `vulkane-kernel-text` documents integer
  kernels (`I32`/`U32`) with `BitAnd`/`BitOr`/`BitXor` and `Shl`/`Shr` (the count masked to the width) and "anything else a typed
  `UnsupportedKernel`". `fishstringer-slang-emit` `slang_binary_int` spells the same operators with the mask `& 31` or `& 63`, choosing
  the 63 only for `I64`, so a `U64` operand would get 31 *if* it reached that speller (not verified; its dtype gate was not read).
  **Reported by their lanes, read not run** (pm-kernels `[FINDING]`, 2026-11-22, `rockfish-hip-emit/src/lib.rs:1745-1943` and `openclownfish-clc/kernel.rs@d7ed879`):
  `rockfish-hip-emit` **emits** integer `Div`/`Shl`/`Shr` at `i32`/`i64`/`u32`/`u64` through guarded helpers (`b == 0` gives `a`, signed `MIN / -1` gives `MIN`, shift count
  `& (N-1)`, left shift through unsigned; `And`/`Or`/`Xor` bare), which matches the ruling by reading. **Update, rockfish lane 2026-11-22 (`rockfish-wt-solver@4ff4377`, reported by pm-kernels):
  host-run, mask read-only.** The 12 integer `Div`/`Shl`/`Shr` kernels (`i32`/`i64`/`u32`/`u64`) were compiled with host `clang++` and **run** on boundary vectors
  (`b == 0`, signed `MIN / -1`, shift counts at and past the width; 190 to 286 pairs each): every result equals the Rust wrapping operation, which is the ruled rule, and two
  hand mutants (no `MIN / -1` guard, a logical signed `Shr`) were caught. **Not observable on x86:** the count mask (the hardware masks), so that part stays read-only. There is no
  `hipcc` and no device in that run: **not device-verified.** `openclownfish-clc` **declines** them by name in `lower_kernel` (`Div`: "a Div other than of two floats"; the others: "the binary op {op:?}"); its older fused
  path masks shifts and uses `ocl_idiv` at the compute type only. Neither was read by this lane.
