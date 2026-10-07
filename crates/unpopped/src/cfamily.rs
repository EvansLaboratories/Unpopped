//! Shared **C-family scalar-op speller vocabulary** — the neutral scalar
//! syntax (f32/f64/int math, the C ternary `select`, ctype names, and
//! runtime-param spelling) that every C-family backend emits verbatim. These
//! spellers are the single source of truth consumed by the CUDA backend
//! (`crate::cuda`), the portable-C CpuC reference backend (`crate::cpu_c`), and
//! the Slang backend (`crate::slang`), so a per-op spelling can never drift
//! between the three.
//!
//! The vocabulary is deliberately backend-neutral: it depends only on the
//! neutral IR (`crate::ir`), the plan (`crate::plan::KernelPlan`), and the
//! op/dtype vocab (`unpopped_vocab`) — never on any CUDA launch
//! harness. It is the module the standalone kernel generator (Unpopped) keeps
//! when the CUDA-specific emitter is later carved into its own crate.
//!
//! # f16/bf16 are neutral since 0.15.0
//!
//! Until 0.15.0 `scalar_ctype` spelled `F16`/`Bf16` as NVIDIA's `__half` /
//! `__nv_bfloat16`, and the half load/store tail emitted `__half2float`-class
//! CUDA intrinsics: public API, in a module documented as neutral.
//!
//! The halves now have FP8's shape. [`scalar_ctype`] answers the STORAGE
//! question only (`unsigned short`, the 16-bit carrier), and the conversion is a
//! software codec emitted into the kernel ([`half_helpers`]), reached through the
//! same [`narrow_load_fn`] / [`narrow_store_fn`] names FP8 uses. A backend with
//! native halves (CUDA's `__half`) spells them itself; baracuda keeps a closed
//! local shadow of every function here that used to reach the half arms.
//!
//! `tests/neutral_spelling.rs` holds the tripwires: no dtype spells a vendor
//! name, and the codec names the halves use are names the helpers define.

use crate::ir::{ArithOp, BinaryOp, ScalarExpr, UnaryOp, is_admissible_int_reduction_operand};
use crate::plan::KernelPlan;
use unpopped_vocab::ElementKind;

/// The C **storage** type for a dtype, or `None` if this module can't spell it.
///
/// For a narrow float (FP8, f16, bf16) this is the carrier, not a compute type:
/// the value is computed at `float` through the dtype's emitted codec
/// ([`narrow_load_fn`] / [`narrow_store_fn`]).
/// `U8` (increment 0b) is the comparison-predicate mask dtype — `unsigned char`
/// per the FKC §5 Bool→U8 pinning — and, since increment 0c, an audited
/// COMPUTE dtype (wrapping mod-256 C semantics), same class as the i32/i64
/// arms. `S8` (FKC `I8`, increment 0c) is `signed char` — two's-complement
/// wrapping via integer promotion + store truncation (see the ir.rs table).
pub fn scalar_ctype(dt: ElementKind) -> Option<&'static str> {
    Some(match dt {
        ElementKind::F32 | ElementKind::F32Strict => "float",
        ElementKind::F64 => "double",
        // The halves are STORED as 16 bits and COMPUTED as a float, exactly
        // like FP8 one width down. The codec is `half_helpers`, emitted with the
        // kernel. (`__half` / `__nv_bfloat16` until 0.15.0: CUDA names in a
        // module that calls itself neutral.)
        ElementKind::F16 | ElementKind::Bf16 => "unsigned short",
        ElementKind::I32 => "int",
        ElementKind::I64 => "long long",
        ElementKind::I8 => "signed char",
        ElementKind::U8 => "unsigned char",
        // KISS-Classify §6.1 widths that have exact, portable C spellings. No
        // vendor intrinsic and no packing is involved.
        ElementKind::I16 => "short",
        ElementKind::U16 => "unsigned short",
        // U32 is the gather/scatter INDEX-operand ctype (`unsigned int`) — a
        // 4-byte address dtype used ONLY for the index-load pointer type (the
        // Model-A u32-index path), never a compute operand. It has no `Element`
        // impl and no vector/packed path; a compute op never keys `plan.dtype =
        // U32` (no constructor builds one), so this arm serves the index load.
        // FP8 is STORED as a byte and COMPUTED as a float. C has no FP8 type,
        // and does not need one: the codec is a pair of emitted helpers
        // (`fp8_helpers`), not a language feature.
        ElementKind::Fp8E4M3FN | ElementKind::Fp8E5M2 => "unsigned char",
        // `bool` is a 1-byte truth value (§6.1) with the same storage width as
        // `u8` and different semantics: its ops normalize to 0/1. The C spelling
        // is the storage type; the normalization lives in the logical spellers,
        // which already emit `... ? 1 : 0`.
        ElementKind::Bool => "unsigned char",
        // Sub-byte dtypes spell their CONTAINER: several elements share a byte,
        // and the packing lives in `sub_byte_helpers`, not in the type name.
        ElementKind::I4 | ElementKind::U4 | ElementKind::B1 => "unsigned char",
        // Complex is a STRUCT, emitted with the kernel (`complex_helpers`).
        // C99 `_Complex` is not an option: MSVC does not implement it.
        ElementKind::Complex64 => "unpopped_c64",
        ElementKind::Complex128 => "unpopped_c128",
        ElementKind::U32 => "unsigned int",
        ElementKind::U64 => "unsigned long long",
        _ => return None,
    })
}

/// Short dtype tag for generated symbol names. Only called for dtypes that pass
/// [`scalar_ctype`].
pub fn dtype_tag(dt: ElementKind) -> &'static str {
    match dt {
        ElementKind::F32 => "f32",
        ElementKind::F32Strict => "f32s",
        ElementKind::F64 => "f64",
        ElementKind::F16 => "f16",
        ElementKind::Bf16 => "bf16",
        ElementKind::I32 => "i32",
        ElementKind::I64 => "i64",
        ElementKind::I8 => "i8",
        ElementKind::I16 => "i16",
        ElementKind::U8 => "u8",
        ElementKind::U16 => "u16",
        ElementKind::U64 => "u64",
        ElementKind::Fp8E4M3FN => "f8e4m3fn",
        ElementKind::Fp8E5M2 => "f8e5m2",
        ElementKind::Bool => "bool",
        ElementKind::I4 => "i4",
        ElementKind::U4 => "u4",
        ElementKind::B1 => "b1",
        ElementKind::Complex64 => "c64",
        ElementKind::Complex128 => "c128",
        // U32 index-dtype infix: `gather_f32_u32` (the Fuel-facing u32-index
        // variant's entry_point symbol).
        ElementKind::U32 => "u32",
        _ => "x",
    }
}

/// Output `j`'s pointer scalar C type — the input `ctype` for a uniform output
/// (`out_dtype_of(j) == plan.dtype`), `unsigned char` for a u8-predicate/keep-mask
/// output (increment 0b single-output; the hetero multi-output / dropout-class
/// increment per-output). `out_ctype_of(plan, 0, ctype)` is byte-identical to the
/// pre-generalization `out_ctype` (output 0's dtype is `plan.out_dtype`), so every
/// single-output + uniform-multi emitter that passes `j = 0`/a uniform `j` is
/// unchanged.
pub fn out_ctype_of<'c>(plan: &KernelPlan<'_>, j: usize, ctype: &'c str) -> &'c str {
    let d = plan.out_dtype_of(j);
    if d == plan.dtype {
        ctype
    } else {
        scalar_ctype(d).expect("validated out dtype has a scalar ctype")
    }
}

/// The store expression for output `j`'s lowered body root.
///
/// A uniform output (`out_dtype_of(j) == plan.dtype`) stores the root unchanged,
/// **except for a narrow float** (FP8, f16, bf16), whose body was computed at
/// `float`: its root is an f32 value and the store applies the dtype's codec.
///
/// A u8 keep-mask output converts the exact 0.0/1.0 predicate to
/// `unsigned char` — exact by construction (the G1 plan gate + G5 backstop pin
/// the body root to a `Cmp*`). The conversion is applied HERE at the store site,
/// per output, **never baked into the shared DAG node**: for dropout the same
/// compute-dtype `Cmp*` temp is consumed by output 0 inside a `Select` (tested
/// `!= 0.0f`) AND stored as `(unsigned char)` by output 1, so a cast on the shared
/// node would corrupt output 0's value (mutation M9).
pub fn store_expr_of(plan: &KernelPlan<'_>, j: usize, root: String) -> String {
    let d = plan.out_dtype_of(j);
    if d == plan.dtype {
        // A UNIFORM cell still needs a store conversion when the dtype is a
        // NARROW FLOAT: its body was lowered at `float` (the leaf decoded), so
        // the root is an f32 expression and the destination is the integer
        // carrier. Returning it unchanged would let C's implicit conversion
        // TRUNCATE the value instead of encoding it — `1.5f` stored as `1`,
        // silently, with no diagnostic.
        //
        // Exactly ONE encode. `313a798` once applied this to roots that were
        // already encoded (the f16 house convention of the time, `__float2half`
        // at the body root) and shipped `__float2half(__float2half(x))` in 0.2.0:
        // numerically the identity, so only byte goldens saw it.
        // `tests/store_demotes_once.rs` counts the codec for all four.
        //
        // Written as an explicit match, so a new narrow dtype is a decision at
        // this site instead of inheriting whatever `narrow_store_fn` does.
        return match d {
            ElementKind::Fp8E4M3FN
            | ElementKind::Fp8E5M2
            | ElementKind::F16
            | ElementKind::Bf16 => demote_store_f32(d, &root),
            _ => root,
        };
    }
    // The hetero elementwise store is exactly the U8 keep-mask (a `Cmp*`
    // predicate, pinned by `assert_valid_out_dtype`; the bincount-I32 scatter
    // narrows itself via `scatter_combine_store`, never through here). The
    // per-element conversion is the shared [`cast_scalar`] routine — one source of
    // truth with the generated cast helper ([`emit_cast_helper`]).
    //
    // ⚠️ Convert FROM what the root IS, not from `plan.dtype`. A narrow float's
    // body was computed at `float`, so its root is the f32 `0.0`/`1.0`, not a
    // storage pattern. Converting from `plan.dtype` decodes it first and feeds
    // `1.0f` to the codec as the pattern `1` — a tiny subnormal — so every true
    // lane stored `0`. Live for FP8 until 0.15.0
    // (`a_narrow_float_predicate_stores_the_right_mask`).
    let from = if narrow_load_fn(plan.dtype).is_some() {
        ElementKind::F32
    } else {
        plan.dtype
    };
    cast_scalar(from, d, &root)
}

/// Portable-C source for the f16/bf16 codec helpers a kernel needs, or `None`
/// for a dtype that needs none.
///
/// The halves have FP8's shape ([`fp8_helpers`]): STORED in a 16-bit carrier
/// (`unsigned short`), COMPUTED as `float`, converted by a decode on load and an
/// encode on store emitted with the kernel. Portable C99, no vendor intrinsic
/// anywhere. A backend with native halves spells its own; this is the default
/// for one without.
///
/// # Correctness over cleverness, as for FP8
///
/// FP8's encoder searches every pattern for the nearest. A 16-bit format has too
/// many patterns for that, so these encoders round arithmetically, but in a form
/// a reader can check: scaling by a power of two (`ldexpf`, `frexpf`) is exact,
/// so the ONLY rounding is the one written out — `floorf`, then
/// round-half-to-even on the remainder. No bit-twiddling.
///
/// Checked bit for bit through a real C compiler against the oracle's
/// independently written codec (`half_kernels_round_trip_against_the_oracles_
/// independent_codec` in `unpopped-cpu-c`), which is itself checked against the
/// `half` crate. NaN encodes as the canonical quiet NaN; its payload is not kept.
pub fn half_helpers(kind: ElementKind) -> Option<&'static str> {
    match kind {
        ElementKind::F16 => Some(
            r"
/* IEEE-754 binary16 (KISS-CLASSIFY 6.1): 1 sign, 5 exp (bias 15), 10 mantissa.
   Max finite 65504; subnormals down to 2^-24; IEEE infinities and NaN. */
static float unpopped_f16_load(unsigned short h) {
    int   sign = (h >> 15) & 1;
    int   exp  = (h >> 10) & 0x1F;
    int   mant = h & 0x3FF;
    float mag;
    if (exp == 0) {
        mag = ldexpf((float)mant, -24);
    } else if (exp == 0x1F) {
        mag = mant == 0 ? INFINITY : NAN;
    } else {
        mag = ldexpf((float)(mant | 0x400), exp - 25);
    }
    return sign ? -mag : mag;
}
static unsigned short unpopped_f16_store(float x) {
    unsigned short sign;
    float a, q, f;
    int e = 0, sub;
    if (x != x) { return 0x7E00; }                    /* NaN */
    sign = (x < 0.0f || (x == 0.0f && 1.0f / x < 0.0f)) ? 0x8000 : 0x0000;
    a = x < 0.0f ? -x : x;
    if (a >= 65520.0f) { return (unsigned short)(sign | 0x7C00); } /* rounds past 65504 */
    /* `sub`, not `e == 0`: frexpf gives e == 0 for every a in [0.5, 1). */
    sub = a < 6.103515625e-05f;                       /* below 2^-14 */
    if (sub) {
        q = ldexpf(a, 24);                            /* in units of 2^-24 */
    } else {
        frexpf(a, &e);                                /* a in [2^(e-1), 2^e) */
        q = ldexpf(a, 11 - e);                        /* in [1024, 2048) */
    }
    f = floorf(q);
    if (q - f > 0.5f || (q - f == 0.5f && ((long)f & 1))) { f += 1.0f; }
    if (sub) { return (unsigned short)(sign | (unsigned short)f); } /* 1024 = 2^-14 */
    /* f == 2048 carries into the exponent field, which is the right encoding. */
    return (unsigned short)(sign | (((e + 14) << 10) + ((int)f - 1024)));
}
",
        ),
        ElementKind::Bf16 => Some(
            r"
/* bfloat16 (KISS-CLASSIFY 6.1): the top half of binary32 — 1 sign, 8 exp
   (bias 127), 7 mantissa. Same range as float; IEEE infinities and NaN. */
static float unpopped_bf16_load(unsigned short h) {
    int   sign = (h >> 15) & 1;
    int   exp  = (h >> 7) & 0xFF;
    int   mant = h & 0x7F;
    float mag;
    if (exp == 0) {
        mag = ldexpf((float)mant, -133);
    } else if (exp == 0xFF) {
        mag = mant == 0 ? INFINITY : NAN;
    } else {
        mag = ldexpf((float)(mant | 0x80), exp - 134);
    }
    return sign ? -mag : mag;
}
static unsigned short unpopped_bf16_store(float x) {
    unsigned short sign;
    float a, q, f;
    int e = 0, sub;
    if (x != x) { return 0x7FC0; }                    /* NaN */
    sign = (x < 0.0f || (x == 0.0f && 1.0f / x < 0.0f)) ? 0x8000 : 0x0000;
    a = x < 0.0f ? -x : x;
    if (a >= ldexpf(511.0f, 119)) { return (unsigned short)(sign | 0x7F80); } /* (2-2^-8)*2^127 */
    /* `sub`, not `e == 0`: frexpf gives e == 0 for every a in [0.5, 1). */
    sub = a < ldexpf(1.0f, -126);
    if (sub) {
        q = ldexpf(a, 133);                           /* in units of 2^-133 */
    } else {
        frexpf(a, &e);                                /* a in [2^(e-1), 2^e) */
        q = ldexpf(a, 8 - e);                         /* in [128, 256) */
    }
    f = floorf(q);
    if (q - f > 0.5f || (q - f == 0.5f && ((long)f & 1))) { f += 1.0f; }
    if (sub) { return (unsigned short)(sign | (unsigned short)f); } /* 128 = 2^-126 */
    /* f == 256 carries into the exponent field, which is the right encoding. */
    return (unsigned short)(sign | (((e + 126) << 7) + ((int)f - 128)));
}
",
        ),
        _ => None,
    }
}

/// Portable-C source for the FP8 codec helpers a kernel needs, or `None` for a
/// dtype that needs none.
///
/// # Why this exists rather than a scalar type
///
/// C has no FP8 type, and that is not the obstacle it sounds like. An FP8 value
/// is **stored** as a byte and **computed** as a float; what a kernel needs is
/// not a native type but a decode on load and an encode on store. These helpers
/// are that, in portable C99 with no vendor intrinsic anywhere.
///
/// # The neutral seam, first proven here
///
/// FP8 got this shape first because it had **no existing goldens to break**. The
/// halves adopted it in 0.15.0 ([`half_helpers`]), once baracuda had shadowed the
/// functions that reached the old `__half2float` spelling.
///
/// The general form is broader than either: *a backend may substitute the whole
/// spelling family, and its arity, for any dtype whose target has a native one,
/// with this module supplying the portable default.* An intrinsic-carrying backend
/// substitutes four things together — scalar ctype, load, store, and a packed pair
/// type (`__half2`) — and the fourth reshapes the emit loop rather than renaming
/// anything in it.
///
/// # Correctness over cleverness in the encoder
///
/// The encoder searches the 127 non-NaN magnitude patterns for the nearest,
/// ties-to-even, rather than manipulating exponent bits. Bit-twiddling rounders
/// are where FP8 conversion bugs live, and this is the *reference* backend — the
/// one an independent implementation is checked against. A reader can verify a
/// search by inspection; they cannot verify a shift-and-mask rounder that way.
/// A performance-shaped emitter should do better and prove it against this.
pub fn fp8_helpers(kind: ElementKind) -> Option<&'static str> {
    match kind {
        ElementKind::Fp8E4M3FN => Some(
            r"
/* OCP FP8 E4M3 (KISS-CLASSIFY 6.1-0010): 1 sign, 4 exp (bias 7), 3 mantissa.
   Max finite 448. NO infinities. A SINGLE NaN encoding, S.1111.111 — every
   other all-ones-exponent pattern is an ordinary finite value, which is where a
   reader assuming IEEE shape gets a wrong number rather than a wrong class. */
static float unpopped_f8e4m3fn_load(unsigned char b) {
    int   sign = (b >> 7) & 1;
    int   exp  = (b >> 3) & 0xF;
    int   mant = b & 0x7;
    float mag;
    if (exp == 0xF && mant == 0x7) {
        mag = NAN;
    } else if (exp == 0) {
        mag = (float)mant * 0.001953125f;  /* 2^-9 */
    } else {
        mag = (1.0f + (float)mant / 8.0f) * ldexpf(1.0f, exp - 7);
    }
    return sign ? -mag : mag;
}

static unsigned char unpopped_f8e4m3fn_store(float x) {
    unsigned char sign, best;
    float a, best_err;
    int p;
    if (x != x) { return 0x7F; }                 /* NaN */
    sign = (x < 0.0f || (x == 0.0f && 1.0f / x < 0.0f)) ? 0x80 : 0x00;
    a = x < 0.0f ? -x : x;
    if (a > 448.0f) { return sign | 0x7E; }      /* saturate: E4M3 has no inf */
    best = 0; best_err = INFINITY;
    for (p = 0; p <= 0x7E; ++p) {
        float v = unpopped_f8e4m3fn_load((unsigned char)p);
        float e = v - a; if (e < 0.0f) e = -e;
        if (e < best_err || (e == best_err && (p % 2) == 0)) { best = (unsigned char)p; best_err = e; }
    }
    return sign | best;
}
",
        ),
        ElementKind::Fp8E5M2 => Some(
            r"
/* OCP FP8 E5M2 (KISS-CLASSIFY 6.1-0011): 1 sign, 5 exp (bias 15), 2 mantissa.
   Max finite 57344. IEEE-style infinities and NaN — unlike E4M3, the all-ones
   exponent behaves the way an IEEE reader expects. */
static float unpopped_f8e5m2_load(unsigned char b) {
    int   sign = (b >> 7) & 1;
    int   exp  = (b >> 2) & 0x1F;
    int   mant = b & 0x3;
    float mag;
    if (exp == 0) {
        mag = (float)mant * 0.0000152587890625f;  /* 2^-16 */
    } else if (exp == 0x1F) {
        mag = mant == 0 ? INFINITY : NAN;
    } else {
        mag = (1.0f + (float)mant / 4.0f) * ldexpf(1.0f, exp - 15);
    }
    return sign ? -mag : mag;
}

static unsigned char unpopped_f8e5m2_store(float x) {
    unsigned char sign, best;
    float a, best_err;
    int p;
    if (x != x) { return 0x7F; }
    sign = (x < 0.0f || (x == 0.0f && 1.0f / x < 0.0f)) ? 0x80 : 0x00;
    a = x < 0.0f ? -x : x;
    if (a > 61440.0f) { return sign | 0x7C; }    /* overflow to inf (E5M2 has one) */
    best = 0; best_err = INFINITY;
    for (p = 0; p <= 0x7B; ++p) {
        float v = unpopped_f8e5m2_load((unsigned char)p);
        float e = v - a; if (e < 0.0f) e = -e;
        if (e < best_err || (e == best_err && (p % 2) == 0)) { best = (unsigned char)p; best_err = e; }
    }
    return sign | best;
}
",
        ),
        _ => None,
    }
}

/// Portable-C helpers for a **sub-byte** dtype's packed load and store, or
/// `None` for a dtype stored one-per-byte or wider.
///
/// The packing is KISS's, not a choice made here (§6.1, normative per §6.0-0001):
/// `i4`/`u4` pack two per byte with the LOW nibble at the even index, `b1` packs
/// eight with the LSB at the lowest logical index. `i4` sign-extends on read;
/// `u4` and `b1` zero-extend. Both halves of that — the order and the extension —
/// produce a plausible wrong answer rather than a crash when guessed.
///
/// # The store is a read-modify-write, and that bounds which backends may use it
///
/// Two logical elements share a byte, so writing one must preserve its
/// neighbour. In this backend's **serial** `for` loop that is simply correct. In
/// a parallel kernel two threads would read-modify-write the same byte and race.
/// So this helper is safe for the scalar reference emitter and **must not be
/// copied into a threaded backend unchanged** — a GPU emitter needs either a
/// byte-per-thread decomposition or an atomic, and should decline until it has
/// one. Recorded here rather than in a follow-up note because the code that
/// looks copyable is exactly the code that gets copied.
pub fn sub_byte_helpers(kind: ElementKind) -> Option<&'static str> {
    match kind {
        ElementKind::I4 => Some(
            r"
/* i4: two per byte, LOW nibble = even index (KISS-CLASSIFY 6.1); SIGN-extended. */
static int unpopped_i4_load(const unsigned char* p, long long i) {
    unsigned char nib = (i & 1) ? (p[i >> 1] >> 4) : (p[i >> 1] & 0x0F);
    return (nib & 0x08) ? (int)nib - 16 : (int)nib;
}

static void unpopped_i4_store(unsigned char* p, long long i, int v) {
    unsigned char nib = (unsigned char)(v & 0x0F);
    unsigned char* b = &p[i >> 1];
    /* Read-modify-write: the other nibble of this byte belongs to a neighbour. */
    *b = (i & 1) ? (unsigned char)((*b & 0x0F) | (nib << 4))
                 : (unsigned char)((*b & 0xF0) | nib);
}
",
        ),
        ElementKind::U4 => Some(
            r"
/* u4: packing identical to i4; ZERO-extended on read. */
static int unpopped_u4_load(const unsigned char* p, long long i) {
    return (int)((i & 1) ? (p[i >> 1] >> 4) : (p[i >> 1] & 0x0F));
}

static void unpopped_u4_store(unsigned char* p, long long i, int v) {
    unsigned char nib = (unsigned char)(v & 0x0F);
    unsigned char* b = &p[i >> 1];
    *b = (i & 1) ? (unsigned char)((*b & 0x0F) | (nib << 4))
                 : (unsigned char)((*b & 0xF0) | nib);
}
",
        ),
        ElementKind::B1 => Some(
            r"
/* b1: eight per byte, LSB = LOWEST logical index (KISS-CLASSIFY 6.1). */
static int unpopped_b1_load(const unsigned char* p, long long i) {
    return (int)((p[i >> 3] >> (i & 7)) & 1);
}

static void unpopped_b1_store(unsigned char* p, long long i, int v) {
    unsigned char mask = (unsigned char)(1u << (i & 7));
    unsigned char* b = &p[i >> 3];
    *b = (v & 1) ? (unsigned char)(*b | mask) : (unsigned char)(*b & ~mask);
}
",
        ),
        _ => None,
    }
}

/// Portable-C helpers for a **complex** dtype: the struct type and its
/// arithmetic, or `None` for a non-complex dtype.
///
/// # Why a struct rather than C99 `_Complex`
///
/// **MSVC does not implement C99 complex.** Its `<complex.h>` ships `_Fcomplex`
/// / `_Dcomplex` — opaque structs constructed with `_FCbuild` and multiplied with
/// `_FCmulcc` — and rejects `float _Complex` outright (measured: `error C2440:
/// cannot convert from 'int' to '_Fcomplex'`). Clang and GCC do implement it. So
/// `_Complex` is not portable C, it is *portable-except-MSVC* C, and using it
/// would put a `#if defined(_MSC_VER)` fork in a module whose whole claim is
/// neutrality.
///
/// A plain struct with our own arithmetic compiles identically everywhere, needs
/// no conditional compilation, and carries no vendor's spelling. Same answer FP8
/// and the sub-byte dtypes reached: emit the operation rather than name someone
/// else's.
///
/// The multiply is the one to read carefully — `(ac - bd, ad + bc)`, not
/// component-wise. The oracle's independently-written Rust rule is the check.
pub fn complex_helpers(kind: ElementKind) -> Option<&'static str> {
    match kind {
        ElementKind::Complex64 => Some(
            r"
/* c64: a pair of f32, real component first (KISS-CLASSIFY 6.1: named by TOTAL
   width). A struct rather than C99 `float _Complex`, which MSVC does not
   implement — see `complex_helpers` for the measurement. */
typedef struct { float re, im; } unpopped_c64;

static unpopped_c64 unpopped_c64_add(unpopped_c64 a, unpopped_c64 b) {
    unpopped_c64 r; r.re = a.re + b.re; r.im = a.im + b.im; return r;
}
static unpopped_c64 unpopped_c64_sub(unpopped_c64 a, unpopped_c64 b) {
    unpopped_c64 r; r.re = a.re - b.re; r.im = a.im - b.im; return r;
}
static unpopped_c64 unpopped_c64_mul(unpopped_c64 a, unpopped_c64 b) {
    /* (ac - bd, ad + bc) — NOT component-wise. */
    unpopped_c64 r;
    r.re = a.re * b.re - a.im * b.im;
    r.im = a.re * b.im + a.im * b.re;
    return r;
}
static unpopped_c64 unpopped_c64_div(unpopped_c64 a, unpopped_c64 b) {
    /* Smith's algorithm. The textbook form ((ac+bd)/(cc+dd), (bc-ad)/(cc+dd))
       squares the denominator components, so it overflows to inf — or flushes to
       zero — for operands well inside the type's range, and then divides by it.
       Scaling by the LARGER component keeps every intermediate near unity.
       Robert L. Smith, CACM 5(8):435, 1962. */
    unpopped_c64 r;
    float ar = a.re, ai = a.im, br = b.re, bi = b.im;
    if ((br < 0 ? -br : br) >= (bi < 0 ? -bi : bi)) {
        float q = bi / br, den = br + bi * q;
        r.re = (ar + ai * q) / den;
        r.im = (ai - ar * q) / den;
    } else {
        float q = br / bi, den = br * q + bi;
        r.re = (ar * q + ai) / den;
        r.im = (ai * q - ar) / den;
    }
    return r;
}
",
        ),
        ElementKind::Complex128 => Some(
            r"
/* c128: a pair of f64, real component first. */
typedef struct { double re, im; } unpopped_c128;

static unpopped_c128 unpopped_c128_add(unpopped_c128 a, unpopped_c128 b) {
    unpopped_c128 r; r.re = a.re + b.re; r.im = a.im + b.im; return r;
}
static unpopped_c128 unpopped_c128_sub(unpopped_c128 a, unpopped_c128 b) {
    unpopped_c128 r; r.re = a.re - b.re; r.im = a.im - b.im; return r;
}
static unpopped_c128 unpopped_c128_mul(unpopped_c128 a, unpopped_c128 b) {
    /* (ac - bd, ad + bc) — NOT component-wise. */
    unpopped_c128 r;
    r.re = a.re * b.re - a.im * b.im;
    r.im = a.re * b.im + a.im * b.re;
    return r;
}
static unpopped_c128 unpopped_c128_div(unpopped_c128 a, unpopped_c128 b) {
    /* Smith's algorithm. The textbook form ((ac+bd)/(cc+dd), (bc-ad)/(cc+dd))
       squares the denominator components, so it overflows to inf — or flushes to
       zero — for operands well inside the type's range, and then divides by it.
       Scaling by the LARGER component keeps every intermediate near unity.
       Robert L. Smith, CACM 5(8):435, 1962. */
    unpopped_c128 r;
    double ar = a.re, ai = a.im, br = b.re, bi = b.im;
    if ((br < 0 ? -br : br) >= (bi < 0 ? -bi : bi)) {
        double q = bi / br, den = br + bi * q;
        r.re = (ar + ai * q) / den;
        r.im = (ai - ar * q) / den;
    } else {
        double q = br / bi, den = br * q + bi;
        r.re = (ar * q + ai) / den;
        r.im = (ai * q - ar) / den;
    }
    return r;
}
",
        ),
        _ => None,
    }
}

/// The complex arithmetic spelling for [`crate::backend::Lowering::arith`], or
/// `None` for a dtype whose arithmetic is a C operator.
///
/// `Div` is absent deliberately: the plan gate refuses it at a complex dtype as
/// DEFINED-but-unimplemented, so reaching here would mean the gate was bypassed.
pub fn complex_arith(kind: ElementKind, op: ArithOp, a: &str, b: &str) -> Option<String> {
    let ty = match kind {
        ElementKind::Complex64 => "unpopped_c64",
        ElementKind::Complex128 => "unpopped_c128",
        _ => return None,
    };
    let name = match op {
        ArithOp::Add => "add",
        ArithOp::Sub => "sub",
        ArithOp::Mul => "mul",
        ArithOp::Div => "div",
        // No `_` arm ON PURPOSE. A catch-all here returns `None`, and `None`
        // means "fall back to the C operator" — which for a struct is
        // `error C2088`, invalid C emitted silently. A fifth `ArithOp` must
        // break this build and be answered, not default into broken output.
    };
    Some(format!("{ty}_{name}({a}, {b})"))
}

/// The packed-load function name for a sub-byte dtype.
pub fn sub_byte_load_fn(kind: ElementKind) -> Option<&'static str> {
    match kind {
        ElementKind::I4 => Some("unpopped_i4_load"),
        ElementKind::U4 => Some("unpopped_u4_load"),
        ElementKind::B1 => Some("unpopped_b1_load"),
        _ => None,
    }
}

/// The packed-store function name for a sub-byte dtype. Counterpart of
/// [`sub_byte_load_fn`].
pub fn sub_byte_store_fn(kind: ElementKind) -> Option<&'static str> {
    match kind {
        ElementKind::I4 => Some("unpopped_i4_store"),
        ElementKind::U4 => Some("unpopped_u4_store"),
        ElementKind::B1 => Some("unpopped_b1_store"),
        _ => None,
    }
}

/// The load-side widening function for a NARROW FLOAT dtype — the emitted
/// codec's decode ([`fp8_helpers`], [`half_helpers`]) — or `None` for everything
/// else.
///
/// One name for every narrow float, deliberately: stored small, computed at f32,
/// and how the conversion is spelled is a property of the dtype, not of the call
/// site. That is what let the halves move from CUDA intrinsics to an emitted
/// codec in 0.15.0 without any caller changing.
pub fn narrow_load_fn(kind: ElementKind) -> Option<&'static str> {
    match kind {
        ElementKind::Fp8E4M3FN => Some("unpopped_f8e4m3fn_load"),
        ElementKind::Fp8E5M2 => Some("unpopped_f8e5m2_load"),
        ElementKind::F16 => Some("unpopped_f16_load"),
        ElementKind::Bf16 => Some("unpopped_bf16_load"),
        _ => None,
    }
}

/// The store-side narrowing function. Counterpart of [`narrow_load_fn`].
pub fn narrow_store_fn(kind: ElementKind) -> Option<&'static str> {
    match kind {
        ElementKind::Fp8E4M3FN => Some("unpopped_f8e4m3fn_store"),
        ElementKind::Fp8E5M2 => Some("unpopped_f8e5m2_store"),
        ElementKind::F16 => Some("unpopped_f16_store"),
        ElementKind::Bf16 => Some("unpopped_bf16_store"),
        _ => None,
    }
}

/// Widen a loaded `inner` expression to `float`: the narrow float's decode, else
/// the value unchanged (already ≥ f32, or an integer loaded natively).
pub fn promote_load_f32(kind: ElementKind, inner: &str) -> String {
    match narrow_load_fn(kind) {
        Some(f) => format!("{f}({inner})"),
        None => inner.to_string(),
    }
}

/// Narrow a `float`-valued `inner` expression to the storage dtype: the narrow
/// float's encode, else the value unchanged (the caller adds any cast).
pub fn demote_store_f32(kind: ElementKind, inner: &str) -> String {
    match narrow_store_fn(kind) {
        Some(f) => format!("{f}({inner})"),
        None => inner.to_string(),
    }
}

/// A single element-wise dtype-cast expression — the value of `expr` (of dtype
/// `from`) converted to dtype `to`, with the narrow-float detour through `float`. The ONE place the generator spells a per-element conversion
/// between two scalar dtypes, shared by the inline hetero store
/// ([`store_expr_of`]) and the generated cast helper (`emit_cast_helper`), so
/// the two can never drift.
///
/// Mirrors `baracuda_cast.cuh`'s `cast_value<TIn, TOut>` (value-identical, not
/// necessarily text-identical — the generated form uses C-style casts and the
/// shared [`promote_load_f32`] / [`demote_store_f32`] codec picks):
///   * `from == to` → identity (no cast).
///   * f16/bf16 → f16/bf16 (cross) → widen to `float`, then narrow.
///   * f16/bf16 → arithmetic → widen to `float`, then a C-style cast to the target.
///   * arithmetic → f16/bf16 → cast to `float` (unless already `float`), then narrow.
///   * arithmetic → arithmetic → a plain C-style cast.
pub fn cast_scalar(from: ElementKind, to: ElementKind, expr: &str) -> String {
    if from == to {
        return expr.to_string();
    }
    match (
        narrow_load_fn(from).is_some(),
        narrow_store_fn(to).is_some(),
    ) {
        // f16/bf16 -> f16/bf16 (cross): widen to f32, then narrow.
        (true, true) => demote_store_f32(to, &promote_load_f32(from, expr)),
        // f16/bf16 -> arithmetic: widen to f32, C-cast to the target.
        (true, false) => {
            let oct = scalar_ctype(to).expect("cast target dtype has a scalar ctype");
            format!("({oct}){}", promote_load_f32(from, expr))
        }
        // arithmetic -> f16/bf16: cast to float (unless already float), then narrow.
        (false, true) => {
            let widened = if matches!(from, ElementKind::F32 | ElementKind::F32Strict) {
                expr.to_string()
            } else {
                format!("(float){expr}")
            };
            demote_store_f32(to, &widened)
        }
        // arithmetic -> arithmetic: plain C-style cast.
        (false, false) => {
            let oct = scalar_ctype(to).expect("cast target dtype has a scalar ctype");
            format!("({oct}){expr}")
        }
    }
}

/// Spell a [`UnaryOp`] applied to an already-lowered f32 inner expression.
/// Inner strings are atomic or parenthesized, so the function-call forms need no
/// extra wrapping; the operator forms wrap themselves.
///
/// # Operand recompute, and why the temp-binding pass is deferred rather than
/// forgotten
///
/// Several spellings reference their operand more than once — `Sqr`/`Relu` twice,
/// `Gelu`/`Silu`/`Sign` three times, and `Max`/`Min` in [`binary_f32`] four times
/// each. On an atomic load that is free; on a compound inner it is a recompute.
/// Binding the inner to a temp first would remove it.
///
/// **It is a pure optimization here, and that is a property of the op set rather
/// than of temp-binding.** Every op above is float-only: the plan gate
/// (`check_int_op_admissibility` rule 2) rejects *every* `UnaryOp`, the float
/// binary fns, and `Cmp*` at an integer dtype. At a float compute dtype a temp
/// has the same type as the expression it holds, so the round-trip is exact and
/// the emitted values cannot move.
///
/// **That safety argument does not survive the op set changing**, which is the
/// part worth writing down. At a sub-`int` dtype, hoisting is *not*
/// value-preserving: C promotes `char`/`short` to `int`, so an inlined compound
/// operand is observed un-truncated while a hoisted one is truncated by the
/// store to its temp. `(in0+in1)>>in2` at `u8` with `(200,100,1)` is `150`
/// inlined and `22` hoisted. That is precisely why the composition pin exists —
/// see rule 3 in `plan::check_int_op_admissibility`. So anyone extending these
/// ops to 8- or 16-bit dtypes must settle truncation *before* adding the
/// temp-binding pass, or the pass silently changes results.
///
/// Deferred rather than done because it rewrites emitted text, which moves every
/// byte-identity golden — including Baracuda's physical CUDA corpus. It belongs
/// with a coordinated golden regen, not with a quiet cleanup.
pub fn unary_f32(op: UnaryOp, x: String) -> String {
    match op {
        UnaryOp::Neg => format!("(-{x})"),
        UnaryOp::Abs => format!("fabsf({x})"),
        UnaryOp::Sqr => format!("({x}*{x})"),
        UnaryOp::Sqrt => format!("sqrtf({x})"),
        // PORTABLE C99, not the CUDA `rsqrtf` intrinsic this used to spell.
        // `rsqrtf`/`rsqrt` are CUDA math-API names absent from C99 <math.h>, and
        // this module advertises itself as backend-neutral on a crates.io crate.
        // Both in-tree backends already overrode this entry independently, which
        // is the tell: when every consumer overrides a shared default, the
        // default is the bug. A CUDA emitter that wants the intrinsic supplies
        // it through its own `unary` seam.
        UnaryOp::Rsqrt => format!("(1.0f/sqrtf({x}))"),
        UnaryOp::Recip => format!("(1.0f/{x})"),
        UnaryOp::Exp => format!("expf({x})"),
        UnaryOp::Log => format!("logf({x})"),
        UnaryOp::Tanh => format!("tanhf({x})"),
        UnaryOp::Sigmoid => format!("(1.0f/(1.0f+expf(-{x})))"),
        // NaN-propagating: `NaN < 0` is false, so NaN passes through (matches
        // PyTorch). `fmaxf(x,0)` would scrub NaN to 0. (Inner duplicated — the
        // temp-binding pass that fixes recompute is a follow-up.)
        UnaryOp::Relu => format!("({x} < 0.0f ? 0.0f : {x})"),
        UnaryOp::Erf => format!("erff({x})"),
        UnaryOp::Gelu => format!("(0.5f*{x}*(1.0f+erff({x}*0.70710678f)))"),
        UnaryOp::Silu => format!("({x}*(1.0f/(1.0f+expf(-{x}))))"),
        UnaryOp::Sin => format!("sinf({x})"),
        UnaryOp::Cos => format!("cosf({x})"),
        UnaryOp::Floor => format!("floorf({x})"),
        UnaryOp::Ceil => format!("ceilf({x})"),
        UnaryOp::Round => format!("rintf({x})"), // ties to even
        UnaryOp::Sign => format!("({x} > 0.0f ? 1.0f : ({x} < 0.0f ? -1.0f : 0.0f))"),
        UnaryOp::Step => format!("({x} > 0.0f ? 1.0f : 0.0f)"), // heaviside(x, 0): step(0)=0
        // increment-0a scalar fns — all implicit CUDA device math (headerless
        // under nvrtc, same class as expf; no includes).
        UnaryOp::Erfc => format!("erfcf({x})"),
        UnaryOp::Trunc => format!("truncf({x})"),
        UnaryOp::Exp2 => format!("exp2f({x})"),
        UnaryOp::Expm1 => format!("expm1f({x})"),
        UnaryOp::Log2 => format!("log2f({x})"),
        UnaryOp::Log10 => format!("log10f({x})"),
        UnaryOp::Log1p => format!("log1pf({x})"),
        UnaryOp::Sinh => format!("sinhf({x})"),
        UnaryOp::Cosh => format!("coshf({x})"),
        UnaryOp::Tan => format!("tanf({x})"),
        UnaryOp::Asin => format!("asinf({x})"),
        UnaryOp::Acos => format!("acosf({x})"),
        UnaryOp::Atan => format!("atanf({x})"),
        UnaryOp::Asinh => format!("asinhf({x})"),
        UnaryOp::Acosh => format!("acoshf({x})"),
        UnaryOp::Atanh => format!("atanhf({x})"),
        UnaryOp::Cbrt => format!("cbrtf({x})"),
        UnaryOp::Lgamma => format!("lgammaf({x})"),
    }
}

/// Same as [`unary_f32`] but with f64 math-function names and double literals.
pub fn unary_f64(op: UnaryOp, x: String) -> String {
    match op {
        UnaryOp::Neg => format!("(-{x})"),
        UnaryOp::Abs => format!("fabs({x})"),
        UnaryOp::Sqr => format!("({x}*{x})"),
        UnaryOp::Sqrt => format!("sqrt({x})"),
        // Portable C99 — see the f32 twin above.
        UnaryOp::Rsqrt => format!("(1.0/sqrt({x}))"),
        UnaryOp::Recip => format!("(1.0/{x})"),
        UnaryOp::Exp => format!("exp({x})"),
        UnaryOp::Log => format!("log({x})"),
        UnaryOp::Tanh => format!("tanh({x})"),
        UnaryOp::Sigmoid => format!("(1.0/(1.0+exp(-{x})))"),
        UnaryOp::Relu => format!("({x} < 0.0 ? 0.0 : {x})"),
        UnaryOp::Erf => format!("erf({x})"),
        UnaryOp::Gelu => format!("(0.5*{x}*(1.0+erf({x}*0.7071067811865476)))"),
        UnaryOp::Silu => format!("({x}*(1.0/(1.0+exp(-{x}))))"),
        UnaryOp::Sin => format!("sin({x})"),
        UnaryOp::Cos => format!("cos({x})"),
        UnaryOp::Floor => format!("floor({x})"),
        UnaryOp::Ceil => format!("ceil({x})"),
        UnaryOp::Round => format!("rint({x})"), // ties to even
        UnaryOp::Sign => format!("({x} > 0.0 ? 1.0 : ({x} < 0.0 ? -1.0 : 0.0))"),
        UnaryOp::Step => format!("({x} > 0.0 ? 1.0 : 0.0)"), // heaviside(x, 0): step(0)=0
        // increment-0a scalar fns — the double variants of the f32 spellings.
        UnaryOp::Erfc => format!("erfc({x})"),
        UnaryOp::Trunc => format!("trunc({x})"),
        UnaryOp::Exp2 => format!("exp2({x})"),
        UnaryOp::Expm1 => format!("expm1({x})"),
        UnaryOp::Log2 => format!("log2({x})"),
        UnaryOp::Log10 => format!("log10({x})"),
        UnaryOp::Log1p => format!("log1p({x})"),
        UnaryOp::Sinh => format!("sinh({x})"),
        UnaryOp::Cosh => format!("cosh({x})"),
        UnaryOp::Tan => format!("tan({x})"),
        UnaryOp::Asin => format!("asin({x})"),
        UnaryOp::Acos => format!("acos({x})"),
        UnaryOp::Atan => format!("atan({x})"),
        UnaryOp::Asinh => format!("asinh({x})"),
        UnaryOp::Acosh => format!("acosh({x})"),
        UnaryOp::Atanh => format!("atanh({x})"),
        UnaryOp::Cbrt => format!("cbrt({x})"),
        UnaryOp::Lgamma => format!("lgamma({x})"),
    }
}

/// Non-infix binary op in f32 math.
///
/// `Maximum`/`Minimum` are **NaN-propagating** (a NaN operand ⇒ NaN out) —
/// matching `torch.maximum`/`minimum` and the house reference kernel
/// `binary_maximum_fp.cu`, which deliberately reserves `fmaxf`/`fminf` (IEEE
/// `maxNum`, NaN-*suppressing*) for a *separate* op. That separate op now exists
/// as [`BinaryOp::FmaxIeee`]/[`BinaryOp::FminIeee`] below — so `Max`/`Min` emit
/// the compare-select, never `fmaxf`. (Operands appear 3× — the deferred
/// temp-binding pass, cf. relu/sigmoid, removes the recompute on compound inners.)
pub fn binary_f32(op: BinaryOp, a: String, b: String) -> String {
    match op {
        // A ON TIES (`>=`/`<=`): the KISS-Ops `max_prop`/`min_prop` normative
        // decomposition (`cmp_ge`/`cmp_le` select a) and numpy/torch
        // `where(a >= b, a, b)`. Bit-visible only on signed-zero ties
        // (`max_prop(-0.0, +0.0) = -0.0`); a `>`-spelled tie would return b.
        BinaryOp::Max => {
            format!("({a} != {a} ? {a} : ({b} != {b} ? {b} : ({a} >= {b} ? {a} : {b})))")
        }
        BinaryOp::Min => {
            format!("({a} != {a} ? {a} : ({b} != {b} ? {b} : ({a} <= {b} ? {a} : {b})))")
        }
        BinaryOp::Pow => format!("powf({a}, {b})"),
        // Floored remainder (torch.remainder, sign-of-divisor — Fuel's Op::Rem),
        // not C fmodf (sign-of-dividend). Operands appear twice — see the
        // operand-recompute note on `unary_f32` for why the temp-binding pass is
        // deferred and what must be settled before anyone writes it.
        BinaryOp::Rem => format!("({a} - floorf({a} / {b}) * {b})"),
        // increment-0a scalar fns. FmaxIeee/FminIeee are the deliberate
        // NaN-SUPPRESSING fmaxf/fminf — the separate op the house reserves them
        // for; Max/Min above stay the NaN-propagating compare-selects. RemTrunc
        // is C fmodf (sign-of-dividend) — the truncated sibling of Rem above.
        BinaryOp::Atan2 => format!("atan2f({a}, {b})"),
        BinaryOp::Copysign => format!("copysignf({a}, {b})"),
        BinaryOp::Nextafter => format!("nextafterf({a}, {b})"),
        BinaryOp::FmaxIeee => format!("fmaxf({a}, {b})"),
        BinaryOp::FminIeee => format!("fminf({a}, {b})"),
        BinaryOp::RemTrunc => format!("fmodf({a}, {b})"),
        // increment-0b comparison predicates: the C operators, with BOTH
        // operands cast to float so the compare is decided IN THE COMPUTE
        // DTYPE. Without the casts, a `Const` operand (spelled as a
        // suffix-less double literal) promotes the float side to double and
        // the compare is decided against the UNROUNDED constant — e.g.
        // `in0[i] == 0.1` is false at every x including 0.1f, while the
        // compute-dtype compare (and torch scalar promotion, and this
        // emitter's own f16 path, which rounds the constant to half first)
        // says true. The cast is a no-op for already-float operands;
        // arithmetic ops keep the double-then-round-once convention
        // (correctly rounded THROUGH the store — compares have no rounding
        // step, so they must round operands first instead). NaN semantics
        // are the C operators' (any comparison with NaN is false EXCEPT
        // `!=`, which is true); the value is EXACTLY 1.0f or 0.0f.
        BinaryOp::CmpEq => format!("((float){a} == (float){b} ? 1.0f : 0.0f)"),
        BinaryOp::CmpNe => format!("((float){a} != (float){b} ? 1.0f : 0.0f)"),
        BinaryOp::CmpLt => format!("((float){a} < (float){b} ? 1.0f : 0.0f)"),
        BinaryOp::CmpLe => format!("((float){a} <= (float){b} ? 1.0f : 0.0f)"),
        BinaryOp::CmpGt => format!("((float){a} > (float){b} ? 1.0f : 0.0f)"),
        BinaryOp::CmpGe => format!("((float){a} >= (float){b} ? 1.0f : 0.0f)"),
        // increment-0c INT-ONLY ops: an independent emitter backstop behind
        // the plan gate (check_int_op_admissibility) — a bitwise/logical op
        // must never reach a float speller, including the f16/bf16 promote
        // path and the reduction-class accumulator lowerings, which all route
        // through here.
        BinaryOp::BitAnd
        | BinaryOp::BitOr
        | BinaryOp::BitXor
        | BinaryOp::Shl
        | BinaryOp::Shr
        | BinaryOp::LogicalAnd
        | BinaryOp::LogicalOr
        | BinaryOp::LogicalXor => {
            panic!("c-family lowering: {op:?} is int-only (I32/I64/S8/U8) — it has no f32 lowering")
        }
    }
}

/// Same as [`binary_f32`] but with f64 math-function names.
pub fn binary_f64(op: BinaryOp, a: String, b: String) -> String {
    match op {
        // A ON TIES (`>=`/`<=`) — see [`binary_f32`]'s Max/Min note.
        BinaryOp::Max => {
            format!("({a} != {a} ? {a} : ({b} != {b} ? {b} : ({a} >= {b} ? {a} : {b})))")
        }
        BinaryOp::Min => {
            format!("({a} != {a} ? {a} : ({b} != {b} ? {b} : ({a} <= {b} ? {a} : {b})))")
        }
        BinaryOp::Pow => format!("pow({a}, {b})"),
        BinaryOp::Rem => format!("({a} - floor({a} / {b}) * {b})"),
        BinaryOp::Atan2 => format!("atan2({a}, {b})"),
        BinaryOp::Copysign => format!("copysign({a}, {b})"),
        BinaryOp::Nextafter => format!("nextafter({a}, {b})"),
        BinaryOp::FmaxIeee => format!("fmax({a}, {b})"),
        BinaryOp::FminIeee => format!("fmin({a}, {b})"),
        BinaryOp::RemTrunc => format!("fmod({a}, {b})"),
        // increment-0b comparison predicates — double literals, same C-operator
        // NaN semantics as the f32 arms.
        BinaryOp::CmpEq => format!("({a} == {b} ? 1.0 : 0.0)"),
        BinaryOp::CmpNe => format!("({a} != {b} ? 1.0 : 0.0)"),
        BinaryOp::CmpLt => format!("({a} < {b} ? 1.0 : 0.0)"),
        BinaryOp::CmpLe => format!("({a} <= {b} ? 1.0 : 0.0)"),
        BinaryOp::CmpGt => format!("({a} > {b} ? 1.0 : 0.0)"),
        BinaryOp::CmpGe => format!("({a} >= {b} ? 1.0 : 0.0)"),
        // increment-0c INT-ONLY ops — same backstop as the f32 speller.
        BinaryOp::BitAnd
        | BinaryOp::BitOr
        | BinaryOp::BitXor
        | BinaryOp::Shl
        | BinaryOp::Shr
        | BinaryOp::LogicalAnd
        | BinaryOp::LogicalOr
        | BinaryOp::LogicalXor => {
            panic!("c-family lowering: {op:?} is int-only (I32/I64/S8/U8) — it has no f64 lowering")
        }
    }
}

/// Spell an increment-0c INT-ONLY binary op (bitwise/shift/logical) over two
/// already-lowered integer operand strings — the RAW C operators, matching the
/// bespoke functors **exactly** (the 0c charter: express the bespoke
/// functionality, never "improve" it):
///
/// - `BitAnd`/`BitOr`/`BitXor`: `binary_bitwise_{and,or,xor}_int.cu`'s
///   `return a OP b;` — no rounding, no overflow concerns.
/// - `Shl`/`Shr`: `binary_bitwise_{left,right}_shift_int.cu`'s `return a << b;`
///   / `return a >> b;` — NO masking or clamping. Out-of-range amounts
///   (`b < 0` or `b >= 8*sizeof(promoted T)`) inherit the architecture's
///   behavior (the bespoke caller contract, carried verbatim); signed `>>` is
///   arithmetic on every CUDA compiler (PTX `shr.s32`/`shr.s64` — the bespoke
///   kernel's documented reliance), unsigned is logical.
/// - `LogicalAnd`/`LogicalOr`/`LogicalXor`: `binary_logical_*_bool.cu`'s
///   normalize-then-op — `(a != 0 OP b != 0) ? 1 : 0`, so the output is
///   strictly 0/1 even for unnormalized bytes (`2 && 4 == 1`). U8-only (the
///   bespoke Bool surface); the plan gate enforces it and the assert here is
///   the independent emitter backstop.
///
/// **Integer-promotion note (S8/U8):** the operand strings are `signed char`/
/// `unsigned char` loads — GUARANTEED, not assumed: the plan gate's 8-bit
/// composition pin (`plan::check_int_op_admissibility` rule 3) requires every
/// int-op operand at `S8`/`U8` to be a leaf `Input`, so a composed operand
/// (whose inlined un-truncated value would diverge from its hoisted 8-bit-tmp
/// value under DAG sharing) can never reach this speller. The loads promote to
/// `int` (sign-/zero-extended) before any operator. NO defeating casts are
/// emitted, deliberately:
/// - and/or/xor: promote → op → store-truncate is bit-identical to a native
///   8-bit op (extension bits AND/OR/XOR among themselves and truncate away);
/// - `Shl`: the 32-bit shift result store-truncates mod 2⁸ — equal to a
///   native wrapping 8-bit shift for in-range amounts, and amounts 8..31 take
///   the promoted (well-defined-in-practice) semantics rather than native-8-bit
///   UB. This matches how the bespoke i32/i64 kernels compose with C — there
///   is no bespoke 8-bit shift to defer to, so the promotion semantics ARE the
///   documented contract (see `BinaryOp::Shl`);
/// - `Shr`: the promoted value's high bits are the extension of the 8-bit
///   value, so the shifted result always fits 8 bits — truncation is exact,
///   arithmetic for `signed char`, logical for `unsigned char`;
/// - logical ops: the `!= 0` tests and the 0/1 result are promotion-invariant.
///
/// The final `other` arm is the second half of the emitter backstop: a float
/// fn / cmp op that reaches the int speller (i.e. bypassed the plan gate at an
/// int dtype) panics rather than emitting C that happens to compile.
pub fn binary_int(op: BinaryOp, a: String, b: String, dtype: ElementKind) -> String {
    if op.is_logical() {
        // `Bool` itself, or the `U8` that represents it. Both spell `unsigned
        // char` and both normalize to 0/1 through the spellers below — the
        // distinction is which dtype the CELL is keyed by, not what the C says.
        assert!(
            matches!(dtype, ElementKind::U8 | ElementKind::Bool),
            "c-family lowering: {op:?} is the bespoke BOOL surface — `Bool` or its `U8` \
             representation, both instantiating uint8_t; got {dtype:?}"
        );
    }
    match op {
        BinaryOp::BitAnd => format!("({a} & {b})"),
        BinaryOp::BitOr => format!("({a} | {b})"),
        BinaryOp::BitXor => format!("({a} ^ {b})"),
        BinaryOp::Shl => format!("({a} << {b})"),
        BinaryOp::Shr => format!("({a} >> {b})"),
        BinaryOp::LogicalAnd => format!("(({a} != 0 && {b} != 0) ? 1 : 0)"),
        BinaryOp::LogicalOr => format!("(({a} != 0 || {b} != 0) ? 1 : 0)"),
        BinaryOp::LogicalXor => format!("((({a} != 0) != ({b} != 0)) ? 1 : 0)"),
        other => panic!(
            "c-family lowering: {other:?} has no integer lowering — the bespoke \
             elementwise surface instantiates it for float dtypes only \
             (int dtype {dtype:?} must miss honestly at the plan gate)"
        ),
    }
}

/// Ternary select in f32 math: `cond != 0.0f` picks arm `a`, else `b`
/// ([`ScalarExpr::Select`] — nonzero-true, `-0.0` false, NaN true).
///
/// The `(float)` casts are MANDATORY, not cosmetic (the 0b double-promotion
/// lesson): they are identity no-ops on already-float operands but pin the
/// compare AND the ternary's type against a suffix-less double `Const`
/// literal from `const_lit` — without the arm casts, a double-literal arm
/// (`select(c, x, 0.0)`) promotes the whole ternary to double and the
/// double→float round-trip at the store QUIETS an f32 sNaN arm payload (a
/// bit diff vs the bespoke select). The cond cast makes the `!= 0` decision
/// happen in the compute dtype (the cmp-operand precedent, `binary_f32`).
/// No arithmetic ever touches an arm: the ternary is data movement
/// (setp+selp), so ±0 signs and NaN payloads (quiet and signaling) move
/// intact — byte-for-byte the bespoke `keep ? input[k] : zero_of<T>()`.
pub fn select_f32(c: String, a: String, b: String) -> String {
    format!("(((float)({c})) != 0.0f ? (float)({a}) : (float)({b}))")
}

/// [`select_f32`] with double literals/casts (the `binary_f64` cmp precedent).
pub fn select_f64(c: String, a: String, b: String) -> String {
    format!("(((double)({c})) != 0.0 ? (double)({a}) : (double)({b}))")
}

/// Runtime scalar-param indices used by `e`, ascending + unique.
pub fn params_used(e: &ScalarExpr) -> Vec<u8> {
    fn rec(e: &ScalarExpr, out: &mut std::collections::BTreeSet<u8>) {
        match e {
            ScalarExpr::Param(i) => {
                out.insert(*i);
            }
            ScalarExpr::Unary(_, x) => rec(x, out),
            ScalarExpr::Add(a, b)
            | ScalarExpr::Sub(a, b)
            | ScalarExpr::Mul(a, b)
            | ScalarExpr::Div(a, b)
            | ScalarExpr::Binary(_, a, b) => {
                rec(a, out);
                rec(b, out);
            }
            ScalarExpr::Select(c, a, b) => {
                rec(c, out);
                rec(a, out);
                rec(b, out);
            }
            ScalarExpr::Input(_)
            | ScalarExpr::Const(_)
            | ScalarExpr::Reduced(_)
            | ScalarExpr::Coord(_) => {}
        }
    }
    let mut set = std::collections::BTreeSet::new();
    rec(e, &mut set);
    set.into_iter().collect()
}

/// Emitter backstop for the two dtype-blind spellings (increment 0c): panic if
/// `e` contains an infix [`ScalarExpr::Div`] node or a [`ScalarExpr::Const`]
/// leaf while `Backend::lower` is lowering an INTEGER dtype. Both are spelled by
/// shared backend code with no dtype context (`lower_expr` emits C `/` and an
/// f64 C literal for every dtype), so unlike the unary/binary-fn/int-only ops
/// they have no per-op speller panic to catch a plan-gate bypass — and they are
/// exactly the device-dangerous pair: integer `/0` is device-UB, and an
/// f64-spelled Const injects double math into an int kernel (f64 cannot even
/// represent all i64). Called from `Backend::lower` over the body and every
/// reduction-class stage/epilogue, independent of `check_int_op_admissibility`.
///
/// `in_reduction` mirrors `plan::check_int_op_admissibility`'s rule 4 (the
/// any/all/count fused-predicate lift, ba325509/Task 3b): `true` only when the
/// expression is this plan's `Access::Reduction` body/post — CpuC/Slang (v1,
/// Elementwise-only) always pass `false`, so their coverage is unchanged.
/// `at_reduction_root` mirrors `plan::check_int_op_admissibility`'s
/// `at_reduction_root` (whole-branch-review fix, closing the composed-
/// predicate leak): `true` ONLY for the initial call on the reduction
/// body/post root, `false` for every recursive descent — CpuC/Slang pass
/// `false` for both parameters (inert, since `in_reduction` is already
/// `false` there). Within that scope (`in_reduction && at_reduction_root`), a
/// `Cmp*` node's operands (leaf `Input`/`Reduced` or an exact 0/1 `Const`)
/// are exempted from the blanket `Const` panic below — that 0/1 `Const` is
/// safe (it lowers as an INTEGER literal in `cuda::emit_reduction`'s
/// `int_reduction_predicate`, which — like this gate — only inspects the
/// body/post ROOT node, never a nested one), never the f64 C literal this
/// backstop exists to catch); anything else in that position still recurses
/// into the general check and panics as before. A `Cmp*` reached as a
/// sub-node of Add/Sub/Mul (not the root) falls to the general
/// `ScalarExpr::Binary(_, a, b)` arm below like any other binary node, so its
/// own `Const`/`Div` operands are still policed by the blanket rules — it is
/// never itself the leaf-or-{0,1} exemption target.
pub fn assert_no_int_div_or_const(
    e: &ScalarExpr,
    dtype: ElementKind,
    in_reduction: bool,
    at_reduction_root: bool,
) {
    match e {
        ScalarExpr::Input(_) | ScalarExpr::Param(_) | ScalarExpr::Reduced(_) => {}
        // A Coord at an int dtype is the SAME hazard class as Const (its
        // spelling is a float cast) — but it has its own dedicated backstop,
        // `assert_coord_lowerable`, which runs beside this walk in
        // `Backend::lower` for every dtype (not just int) and carries the
        // targeted message; no second assert here (one message per layer).
        ScalarExpr::Coord(_) => {}
        ScalarExpr::Const(_) => panic!(
            "c-family lowering: Const at an integer dtype ({dtype:?}) — a Const is \
             spelled as an f64 C literal, which would silently run double math \
             in an integer kernel; the plan gate rejects this (an int-literal \
             speller is a follow-up)"
        ),
        ScalarExpr::Div(_, _) => panic!(
            "c-family lowering: infix Div has no integer lowering ({dtype:?}) — the \
             bespoke elementwise surface has no int div and C `/` by zero is \
             device-UB; the plan gate rejects this"
        ),
        // Select at an integer dtype is validate-rejected at the plan gate
        // (v1 select is float-only); this walk — which only runs for int
        // dtypes — is its independent emitter backstop (G5), beside the
        // per-speller panic in `cuda_select` (which only the elementwise
        // paths route through; the reduction-class accumulator closures
        // don't, so the walk is the layer that covers them).
        ScalarExpr::Select(_, _, _) => panic!(
            "c-family lowering: Select at an integer dtype ({dtype:?}) — v1 select is \
             float-only (the 0c U8/I8 cond-observer question is unresolved); the \
             plan gate rejects this"
        ),
        ScalarExpr::Unary(_, x) => {
            assert_no_int_div_or_const(x, dtype, in_reduction, false);
        }
        ScalarExpr::Add(a, b) | ScalarExpr::Sub(a, b) | ScalarExpr::Mul(a, b) => {
            assert_no_int_div_or_const(a, dtype, in_reduction, false);
            assert_no_int_div_or_const(b, dtype, in_reduction, false);
        }
        // The exemption test is `ir::is_admissible_int_reduction_operand` —
        // the SAME helper `plan::check_int_op_admissibility` (rule 4) and
        // `cuda::emit_reduction`'s `int_reduction_predicate`/`int_cmp_operand`
        // call, so this shape cannot drift from the gate/emitter again. An
        // operand this helper doesn't admit still recurses into the general
        // walk below (this backstop's job is narrower than the plan gate's —
        // it only polices the Const/Div/Select double-math hazards, not
        // composition — so a non-admitted operand isn't rejected here
        // outright, just walked normally).
        ScalarExpr::Binary(bop, a, b) if in_reduction && at_reduction_root && bop.is_cmp() => {
            for operand in [&**a, &**b] {
                if !is_admissible_int_reduction_operand(operand) {
                    assert_no_int_div_or_const(operand, dtype, in_reduction, false);
                }
            }
        }
        ScalarExpr::Binary(_, a, b) => {
            assert_no_int_div_or_const(a, dtype, in_reduction, false);
            assert_no_int_div_or_const(b, dtype, in_reduction, false);
        }
    }
}

/// The SCALAR COMPUTE ctype for an op's runtime launch params — `scalar_ctype(
/// plan.dtype)`, i.e. `"float"` for F32/F32Strict (byte-identical to the pre-F64
/// hardcode) and `"double"` for F64. A launch param is ALWAYS a scalar arg, never
/// vectorized, so this is the declaration ctype at EVERY emitter (including the
/// vectorized ones, whose OPERANDS are `float4`/`double2` but whose param stays
/// `double p0`) — the F64-param increment's load-bearing distinction: pass THIS,
/// never `vty`/`octype`. The `Backend::lower` param assert (see the `matches!` on
/// `plan.dtype` above) guarantees a param-bearing plan has a spellable scalar
/// ctype, so the `expect` is unreachable for any op that actually declares a param.
pub fn param_ctype(plan: &KernelPlan<'_>) -> &'static str {
    scalar_ctype(plan.dtype).expect("param dtype checked by the Backend::lower param assert")
}

/// The trailing `, <ctype> p0, <ctype> p1, …` kernel-signature suffix for the
/// op's runtime scalar params (empty when the op has none). `param_ctype` is the
/// SCALAR COMPUTE ctype `scalar_ctype(plan.dtype)` — `"float"` for F32/F32Strict
/// (byte-identical to the pre-F64 hardcode by construction), `"double"` for F64.
/// A launch param is ALWAYS scalar: even on the vectorized emitter (operands are
/// `float4`/`double2`) the declaration stays `double p0`, NOT `double2 p0` — so
/// callers pass the SCALAR compute ctype, never `vty`/`octype` (the F64-param
/// increment's load-bearing distinction).
pub fn param_args(e: &ScalarExpr, param_ctype: &str) -> String {
    params_used(e)
        .iter()
        .map(|i| format!(", {param_ctype} p{i}"))
        .collect()
}

#[cfg(test)]
mod int_div_or_const_root_gate_validate {
    //! Direct unit coverage for `assert_no_int_div_or_const`'s
    //! `at_reduction_root` restriction (whole-branch-review fix, mirroring
    //! `plan::int_reduction_predicate_gate_validate`). This backstop normally
    //! only runs AFTER `plan::check_int_op_admissibility` has already
    //! validated the op at `build_plan` time, so a composed-predicate body
    //! never reaches it via the `generate()`/`Backend::lower` path — the plan
    //! gate rejects it first. These tests call the function directly (it is
    //! `pub(crate)`) to exercise it as an independent layer in its own right
    //! (the "gate every layer" principle the surrounding code comments name
    //! throughout this file), the same way the plan-gate tests bypass
    //! `build_plan` to isolate `check_int_op_admissibility`.
    use super::assert_no_int_div_or_const;
    use crate::ir::{BinaryOp, ScalarExpr, input, konst};
    use unpopped_vocab::ElementKind;

    // Root-Cmp positive case (mirrors the shipped `count` shape): a bare
    // `Cmp*` IS the reduction body root — admitted (0/1 Const operand
    // exempted from the blanket Const panic), must not panic.
    #[test]
    fn root_cmp_with_01_const_admitted() {
        let body = input(0).binary(BinaryOp::CmpNe, konst(0.0)).0;
        assert_no_int_div_or_const(&body, ElementKind::I8, true, true);
    }

    // Root-only guard: a COMPOSED predicate — `Add(Cmp*, Cmp*)` — reached as
    // the reduction body root. The root itself is `Add`, not `Cmp`, so the
    // exemption arm never matches at this level; the walk recurses into each
    // `Cmp*` child with `at_reduction_root: false` (this fix), lands the
    // nested Cmp on the general `Binary(_, a, b)` arm instead of the
    // exemption arm, and its `Const(0.0)` operand then hits the ordinary
    // blanket `Const` panic — closing the same composed-predicate leak this
    // gate mirrors from `plan::check_int_op_admissibility`. Before this fix
    // the nested Cmp still matched `in_reduction && bop.is_cmp()`
    // (unconditionally, no root check) and its 0/1 Const was wrongly
    // exempted, so this call did NOT panic.
    #[test]
    fn composed_predicate_not_at_root_panics() {
        let cmp = || input(0).binary(BinaryOp::CmpNe, konst(0.0)).0;
        let body = ScalarExpr::Add(Box::new(cmp()), Box::new(cmp()));
        let r = std::panic::catch_unwind(|| {
            assert_no_int_div_or_const(&body, ElementKind::I8, true, true)
        });
        assert!(
            r.is_err(),
            "a Cmp* reached as a sub-node of Add (not the reduction body/post \
             root) must still panic on its Const operand — admitting it here \
             would mirror the plan-gate leak this fix closes"
        );
    }
}
