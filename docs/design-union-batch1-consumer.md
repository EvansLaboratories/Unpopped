<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
# Union batch 1: what Unpopped does with KISS 0.24.0's seven dtype rows and the K role set (board 501 and 515; 2026-11-26)

Status: **E1a built** (this note's first PR); E1b, E2, E3, E4 planned. KISS source: `a9ad8cf` (0.24.0, KISS#848): `spec/classify.md` §6.1 rows 213 to 219 and its `CHANGELOG.md` 0.24.0 entry; the corpus `conformance/corpus/dtype_manifest.json` (34 rows). Every constant below is KISS's, read there; nothing is invented here, and a layout KISS does not state is not stated here.

## 1. The seven rows and the mapping to KISS's spellings

| `ElementKind` | KISS token | kind | storage bits | KISS clause | What KISS pins (verbatim in `spec/classify.md`) |
|---|---|---|---|---|---|
| `F128` | `f128` | float | 128 | §6.1-0018 | IEEE-754 binary128: sign 1, exponent 15, fraction 112 |
| `I128` | `i128` | int | 128 | §6.1-0017 | signed 128-bit two's-complement |
| `U128` | `u128` | uint | 128 | §6.1-0017 | unsigned 128-bit |
| `I256` | `i256` | int | 256 | §6.1-0017 | signed 256-bit two's-complement |
| `U256` | `u256` | uint | 256 | §6.1-0017 | unsigned 256-bit |
| `F8E3M4` | `f8e3m4` | float | 8 | §6.1-0019 | FP8 E3M4 IEEE-style: sign 1, exp 3, mantissa 4, bias 3, with infinity and NaN. Taken from LLVM `APFloat` `semFloat8E3M4` and `ml_dtypes`, which agree; Mojo's docstring reads it finite-only and KISS did not adopt that (byte-incompatible) |
| `F8UE4M3` | `f8ue4m3` | float | 8 | §6.1-0020 | unsigned FP8 E4M3 **scale** (NVFP4): no sign bit, exp 4, mantissa 3, bias 7, no infinity, NaN `0x7F`, range `[0, 448]`; sourced by KISS to NVIDIA/cutlass `5c179ede` `float8.h` and `exmy_base.h`. A scale type like `F8E8M0` (`f8e8m0`) |

The names are Unpopped's (the `F8E8M0` pattern; pm-kernels accepted them 2026-11-26). The tokens are appended after `f6e3m2`, so the §6.1 ordinals 28 to 34 are new and 1 to 27 are unchanged; no existing key byte moves and `sk8` stays. `UE5M3` (PTX Table 53, not in CUTLASS) is NOT in the batch.

## 2. E1a: what was built, and what "storage and key identity only" means

* `ElementKind` gains the seven rows; `ElementKind::ALL` is 35 entries for 34 tokens (`F32Strict` still folds to `f32`).
* The key codec (`structure_key.rs`): token encode and parse, `dtype_storage_bits`, `dtype_numeric_kind`; `F8UE4M3` joins the scale types that decline at operand 0 (`ScaleDtypeAtOperand0`).
* The vendored KISS corpus is re-vendored at `a9ad8cf` (`crates/unpopped-vocab/kiss/README.md`); `tests/kiss_dtype_manifest.rs` pins all 34 tokens, widths and kinds; `tests/kiss_derivation_vectors.rs` runs 88 vectors, **four** of which (the K role set, section 6) are named in `KNOWN_GAP` for E4.
* `check_kernel` **refuses by name** a kernel that declares one of `F128`, `F8E3M4`, `F8UE4M3` as its compute type, a parameter, a local or an allocation (`KernelError::DtypeNotComputed`); **updated 2026-11-27: the four wide integers have been computed since 0.61.3 (E2) and pass**, the text below describes the original seven-row guard, and a `Cast` into one stays the typed `IllegalCast`. **Deviation from the plan as accepted ("load/store/bitcast only"), and why:** probed 2026-11-26 before the guard, the oracle ran a `Load` from an `F128`/`I128`/`U128`/`F8E3M4`/`F8UE4M3` buffer without holding a value of the type, and **panicked** on `I256`/`U256` (`oracle/float_codec.rs:275`); load and store therefore wait for E1b (the 8-bit formats), E2 (the wide integers) and E3 (`F128`), because a typed refusal is the only honest answer until the oracle can hold the value.
* `unpopped-cpu-c` declines the seven in `supports_dtype`; the FKC contract emitter has no §5 spelling for them (`None`, a decline); the text codec round-trips them.

## 3. Planned (not built)

* **E1b (additive, BUILT):** `Cast` between `F32` (and the integers) and `F8E3M4` / `F8UE4M3` on the oracle, as an **intermediate** value (`Cast(Cast(x, fp8), F32)`; a kernel still cannot declare a local, parameter or allocation of the two, E1a's `DtypeNotComputed`, so load, store and arithmetic wait). Round to nearest even; `F8E3M4` follows IEEE overflow (infinity under `Wrap`, the largest finite 15.5 under `ClampFinite`); `F8UE4M3` takes only `ClampFinite`, saturates at 448, sends NaN to its code `0x7F` and a value at or below zero (and `-0`) to `0x00` (**the IR's rule, UNVERIFIED against a vendor conversion**). The oracle's decode is checked against KISS's own vectors (`test_classify_f8e3m4_format`, `test_classify_ue4m3_format`, KISS `a9ad8cf`) and every `F8E3M4` code and every finite `F8UE4M3` code round trips. `unpopped-cpu-c` declines the casts by name (`UnsupportedOp`).
* **E2 (additive):** a wide oracle integer (a 256-bit two's-complement limb type). Today's oracle integer is `i128` (`types::int_range` returns `i128` pairs), which cannot hold a `u128` or an `I256`, so `int_range` must widen. **Wrap rules (the existing integer contract, `ir.rs` on `BinaryOp`: "exact WRAPPING semantics, not infinite-precision arithmetic"), per row:** `I128` and `I256` add, subtract and multiply modulo 2^128 and 2^256 in two's complement, the result read back as signed; `U128` and `U256` modulo 2^128 and 2^256, never negative; shifts mask the count to the width as the existing integers do (UNVERIFIED against the 64-bit rule until E2 reads it); division and remainder by zero follow the existing trap. A `Cast` between widths wraps (`Saturation::Wrap`) or clamps (`Saturation::Clamp`) to the target range.
* **E3 (additive):** a software binary128 for `F128` (add, subtract, multiply, divide, fma, cast), round to nearest even.
* **E4 (additive):** the K role set, section 6.

## 4. What breaks, for whom

`ElementKind` is **not** `non_exhaustive` (cluster ruling 2026-11-26: no silent default), so every external exhaustive match breaks: baracuda (`-emit`, `-parse`), vulkane, bonefish, fishstringer / slang-emit, rockfish, and fuel's `element_kind_to_dtype`. Each declines the seven by name until it lowers them. In this workspace the sites were `unpopped-vocab` (`element.rs`, `structure_key.rs`), `unpopped` (`contract.rs`, `text.rs`, `types.rs`, tests) and `unpopped-cpu-c`.

## 5. UE5M3 and the mxf4 / nvf4 kind field

`UE5M3` is a later batch (single-source, PTX Table 53). `F8UE4M3` unblocks the NVFP4 scale **type** only. The `Tcgen05Mma` kind field (design note 5k) is a struct-literal break and rides the tile/schedule release B1; the scale writes need `tcgen05.cp` / `st` (ROADMAP U61, blocked on the descriptor walk).

## 6. The K role set (E4: key level BUILT, op layer pinned)

KISS 0.24.0 §6.6-0023a: K is a role set (`ContractionRoles::k_more`, further K axes); the K size class and divisibility come from the product of every declared K extent (each read from its own operand); a further K axis that repeats K, another further axis, M or N, or lies outside its operand declines `ContractionAxisInvalid`. **Built (E4):** `unpopped-vocab` `ContractionRoles::with_k_more_axes` / `k_more_axes` (the struct is `non_exhaustive` with private fields, so additive) and the derivation (`contraction_from_axes`); the four KISS vectors that were `KNOWN_GAP` in E1a (`ctr_two_k_axes_product_8x16`, `ctr_three_k_axes_product_large`, `ctr_extra_k_axis_repeats_k_declines`, `ctr_extra_k_axis_is_m_declines`) now pass and `KNOWN_GAP` is empty. **The red-first question (pm-kernels: must a two-`ContractedK` contraction check and run on the oracle?), measured:** the KEY derives, but the op-layer plan gate REFUSES two `ContractedK` axes by name, `UnservableContraction(RolesInvalid)`, "v1: exactly one ContractedK per operand (multi-group deferred)" (`plan/contraction.rs:84`), so nothing runs; `tests/contraction_two_k.rs` pins that refusal beside its one-K control. Plan-layer multi-K is ROADMAP D2 (CUDA rows 44, 55, 81); the emitters index K as one axis and would miscompile it silently, so it needs its own row (an emitter decline or lowering each) and is now unblocked (KISS asked). A kernel-layer two-K contraction (two nested loops) needs no plan gate.

— lane: unpopped
