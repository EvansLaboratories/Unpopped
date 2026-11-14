// SPDX-License-Identifier: MIT OR Apache-2.0
//! The oracle's value domain ([`Val`]) and the raw-bits decoders.
//!
//! A section of the oracle (ROADMAP U30): `oracle.rs` glob-imports it, so every path it had is unchanged.

use super::*;

// ===========================================================================
// The value domain.
// ===========================================================================

/// A scalar value in one of three domains: a float compute value, a wide
/// integer compute value, or raw storage bits that must move verbatim (a leaf
/// load / a bit-preserving move — so `-0.0`, NaN payloads, and signaling NaNs
/// survive untouched through an identity body or a `Select` arm pick).
#[derive(Clone, Copy, Debug)]
pub(super) enum Val {
    Float(f64),
    Int(i128),
    /// A complex value, carried as an exact `(re, im)` pair of `f64`.
    ///
    /// First-class rather than modelled as two reals threaded through the float
    /// path. The float path would have had to grow a "which half am I" parameter
    /// at every node, and the operations that make complex complex — `Mul`
    /// mixing both components, the absence of an ordering — would have lived as
    /// special cases in code whose type says it is handling one real number.
    ///
    /// `f64` components are exact for BOTH complex dtypes: `c64` is a pair of
    /// `f32` (every value of which is an exact `f64`) and `c128` a pair of `f64`.
    Complex(f64, f64),
    Raw(u128, ElementKind),
}

/// Decode a raw integer storage pattern to a signed wide integer (S8 sign-,
/// U8 zero-extended — C integer promotion).
pub(super) fn raw_to_i128(bits: u128, dt: ElementKind) -> i128 {
    match dt {
        ElementKind::I32 => i128::from(bits as u32 as i32),
        ElementKind::I64 => i128::from(bits as i64),
        ElementKind::I8 => i128::from(bits as u8 as i8),
        ElementKind::I16 => i128::from(bits as u16 as i16),
        ElementKind::U8 | ElementKind::Bool => i128::from(bits as u8),
        ElementKind::U16 => i128::from(bits as u16),
        // Zero-extended: `u32` does NOT promote to signed `int`.
        ElementKind::U32 => i128::from(bits as u32),
        // Zero-extended like `u32`, and for the same reason: `unsigned long
        // long` has rank >= `int`, so it does not integer-promote to signed.
        ElementKind::U64 => i128::from(bits as u64),
        // §6.1: `i4` is SIGN-extended on read, `u4` and `b1` zero-extended.
        ElementKind::I4 => {
            let v = (bits as u8) & 0x0f;
            if v & 0x08 != 0 {
                i128::from(v) - 16
            } else {
                i128::from(v)
            }
        }
        ElementKind::U4 => i128::from((bits as u8) & 0x0f),
        ElementKind::B1 => i128::from((bits as u8) & 1),
        other => panic!("oracle: raw_to_i128 on non-int dtype {other:?}"),
    }
}

/// Decode a raw storage pattern to a complex `(re, im)` pair.
///
/// Little-endian, real component first — the layout every C and Rust complex
/// type uses, and the one a `[T; 2]` would produce.
pub(super) fn raw_to_complex(bits: u128, dt: ElementKind) -> (f64, f64) {
    match dt {
        ElementKind::Complex64 => (
            f64::from(f32::from_bits(bits as u32)),
            f64::from(f32::from_bits((bits >> 32) as u32)),
        ),
        ElementKind::Complex128 => (
            f64::from_bits(bits as u64),
            f64::from_bits((bits >> 64) as u64),
        ),
        other => panic!("oracle: raw_to_complex on non-complex dtype {other:?}"),
    }
}

/// Widen an integer into a complex storage pattern (`(i, 0)`).
pub(super) fn encode_int_as_complex(i: i128, out: ElementKind) -> u128 {
    encode_complex(i as f64, 0.0, out)
}

/// Encode a complex `(re, im)` pair into its storage pattern.
pub(super) fn encode_complex(re: f64, im: f64, out: ElementKind) -> u128 {
    match out {
        ElementKind::Complex64 => {
            u128::from((re as f32).to_bits()) | (u128::from((im as f32).to_bits()) << 32)
        }
        ElementKind::Complex128 => u128::from(re.to_bits()) | (u128::from(im.to_bits()) << 64),
        other => panic!("oracle: encode_complex to non-complex dtype {other:?}"),
    }
}

/// Decode a raw storage pattern to `f64` (independent half decode for f16/bf16).
/// Sign and magnitude masks over a **narrow float's raw storage**, or `None` for
/// any dtype this bit-level path does not serve.
///
/// # Why the bit-level path exists at all
///
/// A body that only MOVES bits, or edits at most the SIGN bit, computes nothing —
/// so there is nothing to round, and rounding it destroys what the op is defined
/// to do (KISS-OPS-6.16-0009). `Select` already moves its chosen arm verbatim;
/// this is the same guarantee for `Neg`/`Abs`/`Copysign`/`Max`/`Min`.
///
/// **The oracle's `f64` detour is lossy in exactly one direction that matters:**
/// `f64` has ONE NaN class, and `e5m2` has six encodings (`0x7D`/`0x7E`/`0x7F`
/// and negatives). Decoding to `f64` and re-encoding collapses them, so the
/// reference disagreed with a correct emitter on 4–8 bytes of 256 — and did so
/// for `Max` since 2026-08, unnoticed, because nothing compared the two legs.
///
/// Restricted to the narrow floats deliberately: that is where the loss is
/// measured. `f32`/`f64` results already round-trip their own storage exactly,
/// so widening this would change behaviour with nothing to fix.
pub(super) fn raw_sign_masks(dt: ElementKind) -> Option<(u128, u128)> {
    // Delegates. This derived the mask from `elem_size` while `unpopped-cpu-c`
    // derived it from a ctype string — two independent derivations of one
    // normative fact, agreeing by construction until one of them stopped.
    crate::ir::narrow_sign_masks(dt)
}

pub(super) fn raw_to_f64(bits: u128, dt: ElementKind) -> f64 {
    match dt {
        ElementKind::Fp8E4M3FN => fp8_e4m3fn_to_f64(bits as u8),
        ElementKind::Fp8E5M2 => fp8_e5m2_to_f64(bits as u8),
        ElementKind::F16 => f16_to_f64(bits as u16),
        ElementKind::Bf16 => bf16_to_f64(bits as u16),
        // An MX element only ever exists as an intermediate `Cast` result (it has no buffer
        // layout yet), carried as its raw code.
        ElementKind::F4E2M1 => crate::cast_ref::decode_mx(bits, 2, 1),
        ElementKind::F6E2M3 => crate::cast_ref::decode_mx(bits, 2, 3),
        ElementKind::F6E3M2 => crate::cast_ref::decode_mx(bits, 3, 2),
        ElementKind::F32 | ElementKind::F32Strict => f64::from(f32::from_bits(bits as u32)),
        ElementKind::F64 => f64::from_bits(bits as u64),
        ElementKind::I32
        | ElementKind::I64
        | ElementKind::I8
        | ElementKind::I16
        | ElementKind::U8
        | ElementKind::U16
        | ElementKind::U32
        | ElementKind::U64
        | ElementKind::I4
        | ElementKind::U4
        | ElementKind::B1
        | ElementKind::Bool => raw_to_i128(bits, dt) as f64,
        other => panic!("oracle: raw_to_f64 on unsupported dtype {other:?}"),
    }
}

impl Val {
    /// Project into the float compute domain.
    pub(super) fn f64(self) -> f64 {
        match self {
            Val::Float(f) => f,
            Val::Int(i) => i as f64,
            Val::Raw(b, dt) => raw_to_f64(b, dt),
            // Deliberately fatal rather than returning the real part. A complex
            // value silently becoming its real component is a wrong answer that
            // looks like a right one — the imaginary half vanishes with no
            // diagnostic anywhere. Reaching here means an op routed a complex
            // operand into the float evaluator, which is a routing bug to fix,
            // not a value to coerce.
            Val::Complex(re, im) => {
                panic!("oracle: complex value ({re}, {im}) projected into the float domain")
            }
        }
    }

    /// Project into the complex compute domain.
    ///
    /// A real widens to `(x, 0)` — the standard embedding, and the one that makes
    /// a real literal usable in a complex expression. The reverse does not hold,
    /// which is why [`Val::f64`] refuses instead.
    pub(super) fn complex(self) -> (f64, f64) {
        match self {
            Val::Complex(re, im) => (re, im),
            Val::Float(f) => (f, 0.0),
            Val::Int(i) => (i as f64, 0.0),
            Val::Raw(b, dt) if is_complex(dt) => raw_to_complex(b, dt),
            Val::Raw(b, dt) if is_int(dt) => (raw_to_i128(b, dt) as f64, 0.0),
            Val::Raw(b, dt) => (raw_to_f64(b, dt), 0.0),
        }
    }

    /// Project into the wide-integer compute domain.
    pub(super) fn i128(self) -> i128 {
        match self {
            Val::Int(i) => i,
            Val::Raw(b, dt) => raw_to_i128(b, dt),
            Val::Float(f) => f as i128,
            Val::Complex(re, im) => {
                panic!("oracle: complex value ({re}, {im}) projected into the integer domain")
            }
        }
    }
}

/// Round an `f64` compute value through `dt`'s storage lattice and widen back —
/// the independent analog of the emitter's MANDATORY `(float)`/`(half)` operand
/// cast at a DISCONTINUOUS decision (the `Cmp*` family, a `Select` condition,
/// `Sign`, `Step`). Continuous ops don't need it: accurate-`f64` arithmetic plus
/// the tolerance band absorbs the sub-ULP rounding gap, and a flipped near-tie in
/// e.g. `Max` still returns a within-tolerance operand. A DISCONTINUOUS op decided
/// on the un-rounded `f64` instead flips by a full magnitude no tolerance catches
/// — `x == 0.1` where `0.1` is inexact in the compute dtype (kernel true, oracle
/// false), or `sign(a*b)` where the product underflows to `0` in `f32` but not in
/// `f64`. The emitter guards this with explicit casts (`((float){a} == (float){b}
/// ? 1.0f : 0.0f)`); the oracle must round the same way before the decision.
///
/// Identity for `F64` and the integer dtypes: their kernel decisions are already
/// made in-domain, and routing an integer through `encode_float` would truncate
/// it. Reuses the store-side rounding so the convention is byte-for-byte identical.
/// NOTE: this rounds the operand ONCE — exact for a leaf / `Const` / single arith
/// op feeding the decision (the mask/threshold/sign patterns), which is every case
/// the kernel's per-op `float` rounding also collapses to a single rounding. A
/// decision on a ≥2-deep arithmetic sub-expression landing within ~1 ULP of the
/// threshold is the residual (round-once here vs the kernel's per-op rounding) —
/// a measure-zero near-boundary event outside v1's tolerance-referee role.
pub(super) fn round_to_compute(f: f64, dt: ElementKind) -> f64 {
    match dt {
        ElementKind::F16 | ElementKind::Bf16 | ElementKind::F32 | ElementKind::F32Strict => {
            raw_to_f64(encode_float(f, dt), dt)
        }
        _ => f,
    }
}
