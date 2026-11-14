// SPDX-License-Identifier: MIT OR Apache-2.0
//! The independent half and FP8 codecs, and raw little-endian byte access.
//!
//! A section of the oracle (ROADMAP U30): `oracle.rs` glob-imports it, so every path it had is unchanged.

// ===========================================================================
// Half codec (independent bit-level decode/encode — NOT __half2float).
// ===========================================================================

/// Decode an IEEE-754 binary16 bit pattern to `f64`.
pub(super) fn f16_to_f64(bits: u16) -> f64 {
    let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = (bits >> 10) & 0x1f;
    let mant = bits & 0x3ff;
    let mag = if exp == 0 {
        // subnormal / zero: value = mant * 2^-24
        f64::from(mant) * 2f64.powi(-24)
    } else if exp == 0x1f {
        if mant == 0 { f64::INFINITY } else { f64::NAN }
    } else {
        (1.0 + f64::from(mant) / 1024.0) * 2f64.powi(i32::from(exp) - 15)
    };
    // `sign * NaN` is a NaN of unspecified sign; a binary16 NaN carries its sign bit (ROADMAP U25), as hardware keeps it
    if mag.is_nan() {
        return mag.copysign(sign);
    }
    sign * mag
}

/// Decode an OCP FP8 **E4M3** (`f8e4m3fn`) bit pattern to `f64`.
///
/// KISS-CLASSIFY §6.1-0010: 1 sign, 4 exponent, 3 mantissa, bias 7; maximum
/// finite magnitude 448; **no infinity encodings**; a **single** NaN encoding.
///
/// The `fn` in the token is "finite": the pattern that would be infinity in an
/// IEEE-shaped format is a finite value here, and only `S.1111.111` is NaN. That
/// is the one place a reader who assumes IEEE gets a wrong *number* rather than a
/// wrong classification — `0x7E` is 448, not infinity.
///
/// This decoder is written from the format definition rather than shared with any
/// emitter's. A differential oracle that reuses the implementation it checks is
/// not a differential.
pub(super) fn fp8_e4m3fn_to_f64(bits: u8) -> f64 {
    let sign = if bits & 0x80 != 0 { -1.0 } else { 1.0 };
    let exp = (bits >> 3) & 0x0f;
    let mant = bits & 0x07;
    // The sole NaN: exponent all-ones AND mantissa all-ones. Every other
    // all-ones-exponent pattern is an ordinary finite value.
    if exp == 0x0f && mant == 0x07 {
        // `.copysign(sign)`, not a bare `f64::NAN`. e4m3fn has TWO NaN encodings
        // — 0x7F and 0xFF — so the sign here is information the format carries,
        // and `sign` was computed on this function's first line and then dropped.
        // Measured: the emitter's `neg` maps 0xFF -> 0x7F while this returned a
        // positive NaN, so the negation produced 0xFF and the two disagreed on
        // exactly one byte of 256.
        return f64::NAN.copysign(sign);
    }
    let mag = if exp == 0 {
        // Subnormal / zero: value = mant * 2^(1-bias) / 8 = mant * 2^-9.
        f64::from(mant) * 2f64.powi(-9)
    } else {
        (1.0 + f64::from(mant) / 8.0) * 2f64.powi(i32::from(exp) - 7)
    };
    sign * mag
}

/// Decode an OCP FP8 **E5M2** (`f8e5m2`) bit pattern to `f64`.
///
/// KISS-CLASSIFY §6.1-0011: 1 sign, 5 exponent, 2 mantissa, bias 15; maximum
/// finite magnitude 57344; **IEEE-style** infinities and NaN — so unlike E4M3
/// the all-ones exponent behaves the way an IEEE reader expects.
pub(super) fn fp8_e5m2_to_f64(bits: u8) -> f64 {
    let sign = if bits & 0x80 != 0 { -1.0 } else { 1.0 };
    let exp = (bits >> 2) & 0x1f;
    let mant = bits & 0x03;
    let mag = if exp == 0 {
        // Subnormal / zero: value = mant * 2^(1-bias) / 4 = mant * 2^-16.
        f64::from(mant) * 2f64.powi(-16)
    } else if exp == 0x1f {
        if mant == 0 { f64::INFINITY } else { f64::NAN }
    } else {
        (1.0 + f64::from(mant) / 4.0) * 2f64.powi(i32::from(exp) - 15)
    };
    // `.copysign(sign)`, not `sign * mag`. ⚠️ MULTIPLICATION DOES NOT CARRY A
    // NaN's SIGN: measured on this platform, `(-1.0 * f64::NAN).is_sign_negative()`
    // is FALSE while `f64::NAN.copysign(-1.0)` is true. IEEE leaves the sign of a
    // NaN product unspecified and x86 returns the NaN operand unchanged, so the
    // three negative e5m2 NaN encodings all decoded to a POSITIVE NaN.
    //
    // Identical for every other class — magnitude is non-negative here, so
    // copysign and multiplication agree on normals, subnormals, zero (giving -0.0)
    // and infinity. NaN is the only input where they differ, which is exactly why
    // the multiplication read as correct.
    mag.copysign(sign)
}

/// Round an `f64` to an OCP FP8 **E4M3** bit pattern (round-to-nearest, ties-to-even).
///
/// Overflow **saturates to the maximum finite magnitude** rather than producing
/// an infinity, because E4M3 has none — the format's whole point. A codec that
/// returned an "infinity" pattern here would be emitting NaN or 448 by accident
/// depending on which pattern it chose.
pub(super) fn f64_to_fp8_e4m3fn_bits(x: f64) -> u8 {
    // ⚠️ SIGN FIRST, and applied to the NaN return.
    //
    // This read `if x.is_nan() { return 0x7f; }` with `sign` computed on the very
    // next line and never applied to it — so a NaN's sign was discarded while the
    // saturate path below applied it correctly, three lines down.
    //
    // Both fp8 formats represent a NaN's sign: e5m2 has 0x7D/0x7E/0x7F and their
    // negatives, and e4m3fn has exactly 0x7F and 0xFF. So this was a loss of
    // information the format can hold, not a distinction it cannot express.
    //
    // Found 2026-09-05, immediately after the same defect was fixed in the CpuC
    // emitter. Before that fix the two AGREED — both collapsed to 0x7F — which is
    // why no differential in this workspace ever caught either of them. An oracle
    // that is wrong in the same direction as the thing it checks is not a check.
    // `neg`/`abs`/`copysign` are sign edits, so preserving is REQUIRED there; for
    // arithmetic IEEE leaves a NaN's sign unspecified, so preserving is permitted
    // and never worse. One rule serves both.
    let sign: u8 = if x.is_sign_negative() { 0x80 } else { 0 };
    if x.is_nan() {
        return sign | 0x7f;
    }
    let a = x.abs();
    const MAX_FINITE: f64 = 448.0;
    if a.is_infinite() || a > MAX_FINITE {
        return sign | 0x7e; // largest finite: exp 1111, mant 110
    }
    // Search the 127 non-NaN magnitude patterns for the nearest, ties-to-even.
    // Exhaustive rather than clever: 127 candidates is nothing, and a bit-twiddling
    // rounder is exactly where an independent codec would reproduce the bug it is
    // supposed to catch.
    let mut best: u8 = 0;
    let mut best_err = f64::INFINITY;
    for pat in 0u8..=0x7e {
        let v = fp8_e4m3fn_to_f64(pat);
        let err = (v - a).abs();
        if err < best_err || (err == best_err && pat % 2 == 0) {
            best = pat;
            best_err = err;
        }
    }
    sign | best
}

/// Round an `f64` to an OCP FP8 **E5M2** bit pattern (round-to-nearest, ties-to-even).
///
/// E5M2 *does* have infinities, so overflow produces one rather than saturating.
pub(super) fn f64_to_fp8_e5m2_bits(x: f64) -> u8 {
    // ⚠️ SIGN FIRST, and applied to the NaN return.
    //
    // This read `if x.is_nan() { return 0x7f; }` with `sign` computed on the very
    // next line and never applied to it — so a NaN's sign was discarded while the
    // saturate path below applied it correctly, three lines down.
    //
    // Both fp8 formats represent a NaN's sign: e5m2 has 0x7D/0x7E/0x7F and their
    // negatives, and e4m3fn has exactly 0x7F and 0xFF. So this was a loss of
    // information the format can hold, not a distinction it cannot express.
    //
    // Found 2026-09-05, immediately after the same defect was fixed in the CpuC
    // emitter. Before that fix the two AGREED — both collapsed to 0x7F — which is
    // why no differential in this workspace ever caught either of them. An oracle
    // that is wrong in the same direction as the thing it checks is not a check.
    // `neg`/`abs`/`copysign` are sign edits, so preserving is REQUIRED there; for
    // arithmetic IEEE leaves a NaN's sign unspecified, so preserving is permitted
    // and never worse. One rule serves both.
    let sign: u8 = if x.is_sign_negative() { 0x80 } else { 0 };
    if x.is_nan() {
        return sign | 0x7f;
    }
    let a = x.abs();
    const MAX_FINITE: f64 = 57344.0;
    if a.is_infinite() {
        return sign | 0x7c;
    }
    // Halfway between max-finite and the next power up rounds to infinity, as
    // IEEE requires; anything larger overflows to infinity too.
    //
    // `>=`, not `>`: the comment above says the halfway case rounds to infinity
    // and the operator said otherwise, so exactly 61440.0 came back as 0x7B.
    // Ties-to-even picks the candidate with the even trailing significand, and
    // between max-finite (`11`) and the overflow (`00`) that is the overflow.
    if a >= MAX_FINITE + 2f64.powi(12) {
        return sign | 0x7c;
    }
    let mut best: u8 = 0;
    let mut best_err = f64::INFINITY;
    for pat in 0u8..=0x7b {
        let v = fp8_e5m2_to_f64(pat);
        let err = (v - a).abs();
        if err < best_err || (err == best_err && pat % 2 == 0) {
            best = pat;
            best_err = err;
        }
    }
    sign | best
}

/// Decode a brain-float16 bit pattern to `f64` (bf16 = the top 16 bits of f32).
pub(super) fn bf16_to_f64(bits: u16) -> f64 {
    f64::from(f32::from_bits(u32::from(bits) << 16))
}

/// Round an `f32` to a binary16 bit pattern (round-to-nearest, ties-to-even).
pub(super) fn f32_to_f16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x007f_ffff;
    if exp == 0xff {
        // inf / nan
        return if mant != 0 {
            sign | 0x7e00
        } else {
            sign | 0x7c00
        };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00; // overflow -> inf
    }
    if e <= 0 {
        if e < -10 {
            return sign; // underflow -> +-0
        }
        // subnormal: implicit 1, then round
        let m = mant | 0x0080_0000; // 24-bit significand
        let shift = (14 - e) as u32; // 14..=24
        let half = 1u32 << (shift - 1);
        let mask = (1u32 << shift) - 1;
        let rem = m & mask;
        let mut r = m >> shift;
        if rem > half || (rem == half && (r & 1) == 1) {
            r += 1;
        }
        return sign | (r as u16);
    }
    // normal: round the 23-bit mantissa to 10 bits (drop 13)
    let half = 1u32 << 12;
    let mask = (1u32 << 13) - 1;
    let rem = mant & mask;
    let mut m10 = mant >> 13;
    let mut ee = e;
    if rem > half || (rem == half && (m10 & 1) == 1) {
        m10 += 1;
        if m10 == 0x400 {
            m10 = 0;
            ee += 1;
            if ee >= 0x1f {
                return sign | 0x7c00;
            }
        }
    }
    sign | ((ee as u16) << 10) | (m10 as u16)
}

/// Round an `f32` to a bf16 bit pattern (round-to-nearest, ties-to-even).
pub(super) fn f32_to_bf16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    if x.is_nan() {
        return ((bits >> 16) as u16) | 0x0040; // keep it a quiet NaN
    }
    let bias = 0x7fff + ((bits >> 16) & 1);
    ((bits.wrapping_add(bias)) >> 16) as u16
}

// ===========================================================================
// Raw byte access.
// ===========================================================================

/// Read `sz` little-endian bytes from `bytes` at `byte_off`, zero-extended to u64.
pub(super) fn read_le(bytes: &[u8], byte_off: usize, sz: usize) -> u128 {
    let mut b = [0u8; 16];
    b[..sz].copy_from_slice(&bytes[byte_off..byte_off + sz]);
    u128::from_le_bytes(b)
}

/// Write the low `sz` little-endian bytes of `bits` into `bytes` at `byte_off`.
pub(super) fn write_le(bytes: &mut [u8], byte_off: usize, sz: usize, bits: u128) {
    let b = bits.to_le_bytes();
    bytes[byte_off..byte_off + sz].copy_from_slice(&b[..sz]);
}
