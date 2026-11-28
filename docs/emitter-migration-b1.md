<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
# B1 emitter migration note: what an emitter (Baracuda, Vulkane, Fuel, cpu-c) must handle

For: `baracuda-cuda-emit`/`-parse`/`-seam`, `vulkane-*`, Fuel's consumers, and our own `unpopped-convert` and `unpopped-cpu-c`.
Scope: the B1 breaking batch, one section per item as each lands. Authority for every behaviour is the doc comment on the item in `crates/unpopped/src/kernel.rs` and `exec.rs`.

## T2: opaque tiles (this PR)

| Item | Where | Kind |
|---|---|---|
| `Stmt::TileLoad { d, src, elements_per_row, origin }`, `TileMma { d, a, b, c }`, `TileFill { d, value }`, `TileStore { a, dst, elements_per_row, origin }` | `Stmt` | four new variants: an exhaustive `match` stops compiling |
| `MemorySpace::Tile` | `MemorySpace` (`non_exhaustive`) | new case: **a wildcard arm hides it**, so search your `MemorySpace` matches and state the decline |
| `AllocDecl::tile: Option<TileShape>` | `AllocDecl` (`non_exhaustive`) | new field, `serde(default)`; build with `AllocDecl::tile(..)` |
| `KernelError::Tile { alloc, why }` | `KernelError` (`non_exhaustive`) | new case |

An emitter that does not spell tiles **declines by name**: refuse a `MemorySpace::Tile` allocation and each of the four statements with an error that says "Tile" (cpu-c: `kernel_tile_declines.rs`). Never lower a tile statement to a per-lane fragment `Mma` or drop it. Meaning: a tile is held by the whole subgroup, no element is addressable, `TileMma` is `D = A.B + C` (exact products, ascending `k`, one rounding to the accumulator), `TileStore` is one writer, every tile statement is reached in uniform control. **No `transpose` option exists and none is to be added** (patent hold, US12405770B2). An emitter for Metal or WMMA that lowers tiles lowers `TileLoad`/`TileStore` to `simdgroup_load`/`store` or `wmma::load_matrix_sync`/`store_matrix_sync` and `TileMma` to `simdgroup_multiply_accumulate` or `mma_sync`, and compares with the oracle within a tolerance, never by bytes.

## 0.62 note: `check_kernel` now accepts the wide integers (E2, in 0.61.3)

Since E2 (0.61.3) `check_kernel` **computes** `I128`, `U128`, `I256` and `U256`: a kernel may declare them, the oracle evaluates them (add, sub, mul, total div, shifts, compares, select, integer casts; `crates/unpopped/src/oracle/wide_ops.rs`), and the old `DtypeNotComputed` refusal for them is gone. `F128`, `F8E3M4` and `F8UE4M3` are still declined by `check_kernel`.

**Every emitter's own gate therefore carries the wide-integer refusal alone.** An emitter that does not lower a wide integer **MUST refuse it by name before emitting** (an error that says which kind, never a truncation to 64 bits, never silent). The **emit hold stands**: no emitter lowers a wide `Add` or `Sub` to a carry-flag chain (Intel US11531542B2, the "overflow flag" limb; findings, not a clearance), and an emitter that ever lowers one says which software carry it uses. The IR and the oracle are not held; the emit is. `unpopped-cpu-c` declines the wide kinds by name (`unpopped-cpu-c/tests/kernel_wide_int_declines.rs`). Each other emitter should carry one positive-control test: a kernel with a wide-integer buffer or local goes through the emitter and is refused by name.

## U37: `Numerics::spelling` (0.63.0, BREAKING for a struct literal)

`Numerics` is a plain struct (`Copy`, `Eq`, `Hash`, public fields, not `non_exhaustive`) and gains `spelling: Option<ApproxSpelling>` (serde default, omitted when `None`). **How an emitter or a lifter that builds a `Numerics` literal moves:** end the literal with `..Numerics::default()`, e.g. `Numerics { ftz: true, approx: true, ..Numerics::default() }`; the new field is `None`. Code that only reads `round`, `ftz` and `approx`, calls `Numerics::default()`, or uses `Expr::rounded/ftz/approx` needs no change. A pattern `Numerics { round, ftz, approx }` without `..` stops compiling too: add `..`. **Find your literals:** `git grep -n 'Numerics {'` in your crate (and `git grep -n 'unpopped::ir::Numerics'`); a crate that never builds a `Numerics` by literal (it takes one from the IR, or uses `Numerics::default()` / `Expr::rounded/ftz/approx`) is not affected.

Meaning for a CUDA emitter: `spelling: Some(ApproxSpelling::Expf)` (and `Exp10f`, `Log2f`, `Logf`, `Log10f`, `Fdividef`) on an `approx` node says the node was lifted from that fast-math intrinsic. You MAY print the token (`ApproxSpelling::token()`) when the node's operand and any surrounding scale are exactly the lifted sequence the walker builds (`__expf(x)` is `Exp2` over `x * log2 e`; `__logf` and `__log10f` are `Log2` scaled by `ln 2` and `log10 2` outside the node); otherwise print the exact function or the `ex2.approx` sequence, never the token. It is a hint: ignoring it is always correct. `check_kernel` refuses a spelling on a node that is not `approx` or is of another operation.

## E3: `check_kernel` computes `F128` (binary128), 0.63.x

After E3 slice 2b `check_kernel` accepts a kernel that declares `F128` (a buffer, a local, a scalar parameter, an allocation, the compute type) and the oracle runs it with a software binary128 (`oracle/kernel_f128.rs`, integer words, one rounding to nearest even per operation; **not** double-double, which stays NOT CLEARED). `DtypeNotComputed` now names only `F8E3M4` and `F8UE4M3` (the two 6-bit floats are `F6E2M3`/`F6E3M2`, see the F6 note when it lands).

What the oracle computes over an `F128`: `Add`, `Sub`, `Mul`, `Div`, `Unary(Neg)`, `Unary(Abs)`, the six comparisons (a result of `1` or `0` in `F128`, a NaN unordered, `+0 == -0`), `Select` (an `F128` condition is taken when not zero), and `Cast` to and from `F32`, `F32Strict`, `F64` and the integers (to a float: nearest-even only, `Wrap` or `ClampFinite`; to an integer: any rounding mode, `Clamp`, a NaN is 0; from a wide integer: rounded to nearest even). **Every other node over an `F128` operand is refused by name** (`TypeError::F128Operation`: `Fma`, `Sqrt`, the other unary and binary operators, `Qualified`, `Bitcast`, packing), and so is a collective, an atomic or a raw access over one.

**An emitter that does not lower binary128 MUST refuse a kernel that declares it by name before emitting** (the wide-integer pattern): `unpopped-cpu-c` does (`kernel_f128_declines.rs`); each other emitter should carry the one positive-control test: an `F128` kernel through the emitter is refused, the same kernel over `F64` is not. A NaN result is the canonical quiet NaN (`0x7fff8000...0`), its payload and sign not preserved; a copy (`Load` then `Store`) moves the bytes verbatim. The test vectors come from an independent Python reference and are **UNVERIFIED against a third-party binary128 library**.
## U43: `check_kernel` takes a directed rounding at f64

A `Qualified` node over **f64** operands now passes `check_kernel` and runs in the oracle when its qualifier is a rounding mode (`TowardZero`, `TowardNegInf`, `TowardPosInf` or `NearestEven`) and it wraps `Add`, `Sub`, `Mul`, `Div`, `Fma` or `Sqrt` (the reference is `numerics_f64::eval_qualified_f64`, exact integer arithmetic and one rounding). `ftz` and `approx` at f64 and a half-type operand stay a named `TypeError::IllegalQualifier`; a **plan** is still refused at the plan gate. **An emitter that has no f64 directed spelling MUST decline such a kernel by name**: `unpopped-cpu-c` does (`kernel_qualified_f64_declines.rs`; C99 reaches a directed mode only through `fesetround`). The spellings (`__dadd_rn` and the family) are ROADMAP U43b.
