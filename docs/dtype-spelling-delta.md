# `ElementKind` spelling delta — finalized against the ratified sk4 scheme

Unpopped's deliverable for the KISS dtype standardization, updated after Eric
ratified the item-#2 scheme and greenlit a single coordinated sk4 event.

sk4's committed scope: dtype rename + MX additions + non-contraction
`(accumulator + math-precision)` + the version-prefix codec rule. Per-operand
dtype is documented as a follow-up, gated on a separate scope call about
indexed-region synthesis (adopting it would reverse KISS-Classify §6.6-0015's
deliberate caller-precondition).

## The delta

Measured against published `unpopped-vocab` 0.1.0. **Variant** is the Rust
identifier; **token** is what the `structure_key` codec emits. They move
independently, which matters: some changes are variant-only and therefore not
schema-visible at all.

| variant now | token now | variant at sk4 | token at sk4 | schema-visible |
|---|---|---|---|---|
| `S8` | `s8` | `I8` | `i8` | yes |
| `S4` | `s4` | `I4` | `i4` | yes |
| `Bin` | `b1` | `B1` | `b1` | **no — variant only** |
| `Fp8E4M3` | `e4m3fn` | `Fp8E4M3FN` | `f8e4m3fn` | yes (prefix only) |
| `Fp8E5M2` | `e5m2` | `Fp8E5M2` | `f8e5m2` | yes (prefix only) |
| `Complex32` | `c32` | `Complex64` | `c64` | **yes — meaning flip** |
| `Complex64` | `c64` | `Complex128` | `c128` | **yes — meaning flip** |

Unchanged: `F16`, `Bf16`, `F32`, `F64`, `U8`, `I32`, `I64`, `U32`, `Bool`, `U4`,
`F32Strict` (a pre-fold derivation input, never a key dtype).

Two corrections to the earlier revision of this document, both from reading the
codec rather than the enum:

- **`Bin` already tokenizes as `b1`.** The rename is cosmetic at the Rust level
  and invisible on the wire.
- **`Fp8E4M3` already tokenizes as `e4m3fn`.** The `fn` explicitness the scheme
  requires is *already present in the token*; only the Rust variant name omits
  it. The token change here is the `f8` prefix, not the suffix.

## The one real decodability hazard: `c64`

Complex moves from component-width to total-width naming, so `c64` is a valid
token at both schema versions with **different meanings**:

- `sk3|…|c64|…` — interleaved pair of `f64`, 128 bits
- `sk4|…|c64|…` — interleaved pair of `f32`, 64 bits

Full token-set comparison, which localizes the risk to that single token:

- **Stable in both, same meaning:** `f16`, `bf16`, `f32`, `f64`, `u8`, `i32`,
  `i64`, `u32`, `bool`, `b1`, `u4`
- **sk3 only:** `s8`, `s4`, `e4m3fn`, `e5m2`, `c32`
- **sk4 only:** `i8`, `i4`, `f8e4m3fn`, `f8e5m2`, `c128`
- **Both, different meaning:** `c64` ← the entire hazard

Every retired token is version-exclusive, so an sk4 decoder meeting `s8` or `c32`
rejects it as unknown-at-this-version, and an sk3 decoder meeting `c128` does the
same. Those fail loudly and correctly. Only `c64` decodes successfully under both
and means different things, which is precisely why the codec rule below is
load-bearing rather than hygiene.

## Codec absorption

Tokens are version-prefixed (`sk3|bin|f32|cuda:sm89|…`, with
`STRUCTURE_KEY_VERSION = 3` matching the literal `sk3`).

- **Pure renames** (`s8`→`i8`, `s4`→`i4`, the `f8` prefix) can stay parseable
  across the bump: a decoder keeps an `sk3` arm mapping the retired spelling to
  the same variant. Meaning is unchanged, so an old token still decodes to the
  identical cell, and the migration window can be as long as KISS chooses.
- **The `c64` flip is safe *because of* the prefix**, and only because of it. A
  decoder never guesses: the version is in the token. Cross-version string
  equality cannot false-match, because the prefixes differ.
- **The hazard is detachment, not decoding.** A consumer that persists, indexes,
  or compares a dtype sub-token *separately from its version prefix* loses
  exactly the protection the prefix provides. A cache key holding `"c64"` without
  `"sk4"` is where a silent meaning-flip still bites.

### Proposed §6.7 codec clause

> The schema version prefix is normative and inseparable. A dtype token has no
> meaning detached from the `structure_key` schema version that produced it.
> Implementations MUST NOT persist, index, cache, or compare dtype sub-tokens
> independently of their schema version, and MUST NOT compare tokens across
> schema versions for equality of meaning. A decoder MUST reject a token whose
> spelling is not valid at the declared schema version rather than interpreting
> it under a different version's vocabulary.

The final sentence is what makes the retired-token cases fail loudly instead of
silently, and the clause protects every future schema bump, not only this one.

## Migration note required at the event

Because `c64` changes meaning, the sk4 realization MUST ship an explicit
migration note. It cannot be presented as a mechanical rename: consumers holding
`Complex64` must know it now denotes what `Complex32` denoted, and any persisted
`c64` written under sk3 means what `c128` means under sk4.
