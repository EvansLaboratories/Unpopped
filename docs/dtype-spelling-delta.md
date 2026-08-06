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

## 4. `F32Strict` — the key codec is already right; the loss is non-contraction

**An earlier revision of this document overstated this item.** It claimed
`F32`/`F32Strict`-as-dtypes was a vendor leak in the key. It is not: sk3 D4
already retired the `f32s` token per KISS §6.1-0005, and `canonical_dtype` folds
`F32Strict → F32` before the token is spelled. Unpopped already implements
KISS's one-`f32` model. The correction matters, so it is recorded rather than
quietly edited.

`F32Strict` surviving as an `ElementKind` variant is also deliberate and correct
— it is the derivation *input* that says "strict SIMT math" on the operand
channel, not a key dtype.

**Two things do survive.**

*Cosmetic:* `F32` and `F32Strict` document themselves in one vendor's terms —
"TF32 tensor cores", "SIMT CUDA cores". Worth rewording in a neutral vocabulary's
public docs.

*Substantive:* **the fold is lossless only for contractions.** `mp` occurs
exactly once in the key, inside `ContractionKey`. A non-contraction op has no
`contraction`, therefore no `<mp>` — while `canonical_dtype` folds `F32Strict`
to `F32` regardless. So a strict-SIMT f32 reduction and a TF32 f32 reduction
produce the **identical token**, with the distinction dropped: the fold is
documented as "deliberately lossy", and for non-gem cells the coordinate it is
supposed to ride does not exist.

That is the accumulator gap's shape on a second axis, and it settles an open
question in KISS's sk4 scoping: sk4's non-contraction coordinate set is
**`(acc + mp)`** — the full analogue of gem's pair — not the accumulator alone.

## 5. Codec absorption — the token is self-describing

Tokens are version-prefixed (`sk3|bin|f32|cuda:sm89|…`, with
`STRUCTURE_KEY_VERSION = 3` matching the literal `sk3`). That prefix is what
makes a spelling change tractable:

- **Pure renames** (`s8`→`i8`, `s4`→`i4`, `bin`→`b1`) can stay parseable across
  the bump: a decoder keeps an `sk3` arm mapping the old spelling to the same
  variant. Meaning is unchanged, so an old token still decodes to the identical
  cell.
- **The `c64` meaning-flip is safe too, because of the prefix.** `sk3|…|c64|…` is
  a pair of `f64`; `sk4|…|c64|…` would be a pair of `f32`. A decoder never
  guesses, and cross-version string equality cannot produce a false match because
  the prefixes differ.
- **The real hazard is detachment, not decoding.** A consumer that persists,
  indexes, or compares a dtype sub-token *separately from its version prefix*
  loses exactly the protection the prefix provides. A cache key holding `"c64"`
  without `"sk4"` is where a silent meaning-flip would still bite.

Recommended clause for the #2 RFC: **the version prefix is normative and
inseparable — a dtype token has no meaning detached from its schema version, and
implementations MUST NOT persist, index, or compare dtype sub-tokens
independently of it.**

## Sequencing

Items 2, 3 and 4 are all `structure_key`-affecting. Per the batching agreed with
KISS, they should ride **one** coordinated schema bump together with the sk4
accumulator coordinate and the MX additions — one five-way re-derivation and
byte-match reverify across KISS, Fuel, Baracuda, kiss-ref and Unpopped, rather
than three.

The renames in §1 are cheap **only while `unpopped-vocab` has no dependents**.
That window is open now and closes as Fuel, Vulkane and Baracuda migrate.
