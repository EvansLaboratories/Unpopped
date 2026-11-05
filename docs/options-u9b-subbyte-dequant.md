# Options note: sub-byte loads and the dequantizing GEMM (ROADMAP U9b, issue #120)

Status: **Q1 to Q4 ruled by the PM 2026-11-05 as recommended (cast endpoints `I4`/`U4`, sub-byte buffers read-only in kernels for v1, emitter reuses the plan-path codecs, `B1` not now; revisit a sub-byte `Store` only when a consumer needs it); S1 (checker and oracle) is built in this PR.** U9b is the row "mixed-input dequant GEMM at the kernel
layer (int4 weights x wider activations)" whose trigger was "the kernel layer's sub-byte `Load` is read at source and found to work".
It was read at `origin/main` 23671f0 (`kernel_subbyte_load.rs`, `kernel_subbyte_load_refused.rs`). The answer is **half**.

## What was found

| Piece | Today | Evidence |
|---|---|---|
| `Load` and `Store` of an `I4` buffer element, checker and oracle | **Works.** Packed two per byte, low nibble first; eight nibbles round-trip exactly (`8 F 0 1 7 3 C 5` gives the bytes `F8 10 37 5C`) | `a_sub_byte_load_and_store_round_trip_in_the_oracle` |
| `Cast` from `I4` (the widening a dequant loop needs) | **Typed refusal**: `IllegalCast`, "a Cast endpoint outside {f4e2m1, f6e2m3, f6e3m2, f16, bf16, f32, f32s, f64, i8, i16, i32, i64, u8, u16, u32, u64} is not defined yet". `I4`, `U4` and `B1` are not Cast endpoints | `a_widening_cast_from_a_sub_byte_type_is_a_typed_refusal_for_now` |
| The C emitter on an `I4` kernel buffer | **Typed refusal**: "buffer `w` has dtype I4, which this emitter does not spell", although the plan path has software codecs for `i4`/`u4`/`b1` | `an_i4_buffer_is_refused_with_a_typed_error` |

So the trigger half-fired: the memory access is fine, the arithmetic is not. A dequant GEMM cannot be written until a sub-byte element
can be widened.

## Slices

**S1, `Cast` endpoints `I4` and `U4` (checker, oracle, `docs/ir-typed-expressions.md`).** Widening is exact: `I4` sign-extends, `U4`
zero-extends, to every integer and float endpoint already allowed. Narrowing to `I4`/`U4` follows the existing integer rule (`Wrap`,
or `Sat` with the target's range). *Cost:* the cast table and the oracle's integer cast path, plus tests naming each pair. *Risk:* none
for kernels that exist (a refusal becomes an acceptance). `B1` is left out (a bit is not an integer value here).

**S2, the C emitter spells `I4`/`U4` buffer elements.** A load extracts a nibble from the byte; reuse the plan path's software codecs
instead of writing a second pair. *Question below:* what a `Store` is.

**S3, the dequant kernel (the U9b deliverable):** `out[r][c] = sum_k (w4[r][k] - zero) * scale[r][k / group] * x[k][c]`, `w4` an `I4`/`U4`
buffer, group scales in a second buffer, integer or exactly representable data, an oracle test and a C differential against a host
loop, with the seeded-mutation pass (nibble order swapped, sign extension dropped, scale group off by one, zero point ignored).

## Questions for the PM (all ruled yes, 2026-11-05)

| # | Question | Recommendation |
|---|---|---|
| Q1 | Extend the `Cast` endpoints to `I4` and `U4` (S1)? Additive (a refusal becomes an acceptance); the PM allocates the number | **Yes** |
| Q2 | A kernel `Store` of a sub-byte element is a read-modify-write of a shared byte, so two threads writing the two nibbles of one byte race at the byte even though the elements differ. Allow sub-byte stores in v1? | **No: sub-byte buffers are `Read` in kernels in v1** (weights are inputs; outputs stay 8 bits or wider); a later row can add stores with an atomic RMW definition |
| Q3 | `B1` (bit) as a `Cast` endpoint | Not now |
| Q4 | S2 reuses the plan path's codecs rather than new C | Yes |

## Build order, once approved

S1 with its tests (one PR), S2 (one PR, turning `kernel_subbyte_load_refused.rs` into an agreement test), then S3.
