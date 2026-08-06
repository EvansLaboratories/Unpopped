# `ElementKind` spelling delta — Unpopped's position for KISS item #2

Unpopped's action item for the cross-project dtype-vocabulary standardization.
Measured against the published `unpopped-vocab` 0.1.0 enum (18 variants).

Working assumption is the KISS lean: de-vendored `i`-prefix for signed integers
matching `u` for unsigned, `Fp8`-prefixed f8 preserving the fn/fnuz distinction,
sub-byte *values* as dtypes with packing and scale as separate coordinates.

## 1. Mechanical renames — settled

| current | proposed | why |
|---|---|---|
| `S8` | `I8` | `.s8` is NVIDIA PTX spelling; `i8` is the neutral one |
| `S4` | `I4` | same |
| `Bin` | `B1` | KISS spells 1-bit `b1`; `Bin` names an encoding, not a width |

`I32`, `I64`, `U8`, `U4`, `U32`, `F16`, `Bf16`, `F64`, `Bool` are already neutral
and unchanged. `Bool` is a logical type, not a numeric width, and stays distinct
from `B1`.

## 2. `Fp8E4M3` does not pin fn vs fnuz — needs a ruling

The token is `Fp8E4M3`, but the doc describes **fn** semantics: *"max-finite 448,
no infinities."* That is OCP `E4M3FN`. KISS wants the distinction preserved, and
the same argument that makes bare `E4M3` a correctness hazard applies to an
`Fp8E4M3` that silently means one of two encodings.

Proposal: `Fp8E4M3FN`, with `Fp8E4M3FNUZ` reserved. `Fp8E5M2` needs the same
ruling — if only one variant is representable, saying so explicitly is still
better than an unqualified token.

## 3. `Complex32` collides with every other ecosystem's convention — high risk

The doc is explicit: *"interleaved real/imag pair of `f32`… ABI-compatible with
cuFFT's `cufftComplex`, NumPy's `complex64`, and PyTorch's `torch.complex64`."*

So `unpopped_vocab::ElementKind::Complex32` **is** NumPy's `complex64`. The
vocabulary names complex types by **component** width; NumPy, PyTorch, and C all
name them by **total** width. Any consumer reading `Complex32` will reasonably
conclude 32 bits total and be wrong by a factor of two.

This is the same class of hazard as bare `E4M3` — a token that reads as
unambiguous and is not. Proposal: rename to `Complex64` / `Complex128` (total
width, matching the rest of the world). Cost: the new `Complex64` means what the
old `Complex64` did not, so this rename is **not** mechanically safe for
consumers and must land in a schema-version event with a migration note, never
silently.

## 4. `F32` and `F32Strict` are not dtypes — escalating

Their docs:

- `F32` — *"IEEE 754 binary32 inputs reduced through **TF32 tensor cores**
  (10-bit mantissa)."*
- `F32Strict` — *"IEEE 754 binary32 inputs reduced through **SIMT CUDA cores** at
  full f32 precision."*

Both are IEEE 754 binary32 in storage. They are the same data type. They differ
only in **which NVIDIA execution unit performs the reduction** — TF32 tensor
cores versus SIMT CUDA cores.

That is a compute-precision mode, not a data type, and it is a vendor-specific
one. KISS already models math precision as its own coordinate (`<mp>`), which is
where this distinction belongs. A SPIR-V or Metal backend has no TF32 and no
"CUDA cores", so it cannot honour the distinction as a dtype — it would have to
map both to `f32` and silently lose the caller's intent, or decline a dtype it
demonstrably supports.

This is a larger vendor-lineage leak than the `s`-prefix one: the prefix is
cosmetic, but `F32`/`F32Strict` encode **one vendor's hardware topology into the
data vocabulary**, and two of eighteen tokens are affected.

Unpopped is not proposing the fix unilaterally — this touches `<mp>`, the
contraction key's `mp` field, and every impl's derivation. Raising it as an
item-#2 escalation for the four-way. The likely shape is: one `f32` dtype, with
the TF32-vs-strict choice expressed as math precision, which is where the
contraction key already carries it.

## Sequencing

Items 2, 3 and 4 are all `structure_key`-affecting. Per the batching agreed with
KISS, they should ride **one** coordinated schema bump together with the sk4
accumulator coordinate and the MX additions — one five-way re-derivation and
byte-match reverify across KISS, Fuel, Baracuda, kiss-ref and Unpopped, rather
than three.

The renames in §1 are cheap **only while `unpopped-vocab` has no dependents**.
That window is open now and closes as Fuel, Vulkane and Baracuda migrate.
