// SPDX-License-Identifier: MIT OR Apache-2.0
//! Element-type facts: bit widths, signedness, integer wrapping, raw element access.
//!
//! A section of the oracle (ROADMAP U30): `oracle.rs` glob-imports it, so every path it had is unchanged.

use super::*;

// ===========================================================================
// dtype helpers (independent — not reused from cuda.rs / structure_key.rs).
// ===========================================================================

/// Element byte size for the v1-supported dtypes.
/// Storage width in **bits**, which is the only width that describes every §6.1
/// dtype.
///
/// `elem_size` is bytes and cannot express `i4`/`u4` (4 bits) or `b1` (1 bit).
/// Those are not stored one-per-byte: KISS-CLASSIFY §6.1 pins their packing
/// normatively, and packing is part of what a dtype *is* there (§6.0-0001 makes
/// the packing convention normative alongside the token and the width).
pub(crate) fn elem_bits(dt: ElementKind) -> usize {
    match dt {
        ElementKind::I4 | ElementKind::U4 => 4,
        ElementKind::B1 => 1,
        other => elem_size(other) * 8,
    }
}

/// `true` for the dtypes stored more than one to a byte.
pub(super) fn is_sub_byte(dt: ElementKind) -> bool {
    elem_bits(dt) < 8
}

/// Bytes needed to hold `n` elements of `dt`, rounding a partial trailing byte up.
pub(super) fn packed_bytes(dt: ElementKind, n: usize) -> usize {
    (n * elem_bits(dt)).div_ceil(8)
}

/// Read element `i`'s raw storage bits, sub-byte packing included.
///
/// The packing is KISS's, not a choice made here:
/// * `i4`/`u4` — packed pair, **low nibble = even index, high nibble = odd**.
/// * `b1` — 8 per byte, **LSB = lowest logical index**.
///
/// Getting either backwards produces a buffer that round-trips through itself
/// perfectly and disagrees with every other implementation, which is the failure
/// mode a normative packing convention exists to prevent.
pub(super) fn read_elem(bytes: &[u8], i: usize, dt: ElementKind) -> u128 {
    match elem_bits(dt) {
        4 => {
            let b = bytes[i / 2];
            u128::from(if i % 2 == 0 { b & 0x0f } else { b >> 4 })
        }
        1 => u128::from((bytes[i / 8] >> (i % 8)) & 1),
        _ => {
            let sz = elem_size(dt);
            read_le(bytes, i * sz, sz)
        }
    }
}

/// Write element `i`'s raw storage bits, sub-byte packing included.
///
/// Read-modify-write for the sub-byte cases, since a nibble or a bit shares its
/// byte with neighbours that must not be disturbed.
pub(super) fn write_elem(bytes: &mut [u8], i: usize, dt: ElementKind, bits: u128) {
    match elem_bits(dt) {
        4 => {
            let v = (bits as u8) & 0x0f;
            let b = &mut bytes[i / 2];
            if i % 2 == 0 {
                *b = (*b & 0xf0) | v;
            } else {
                *b = (*b & 0x0f) | (v << 4);
            }
        }
        1 => {
            let v = (bits as u8) & 1;
            let b = &mut bytes[i / 8];
            let mask = 1u8 << (i % 8);
            *b = (*b & !mask) | (v << (i % 8));
        }
        _ => {
            let sz = elem_size(dt);
            write_le(bytes, i * sz, sz, bits);
        }
    }
}

pub(super) fn elem_size(dt: ElementKind) -> usize {
    match dt {
        ElementKind::F16 | ElementKind::Bf16 | ElementKind::I16 | ElementKind::U16 => 2,
        ElementKind::F32 | ElementKind::F32Strict | ElementKind::I32 | ElementKind::U32 => 4,
        ElementKind::F64 | ElementKind::I64 | ElementKind::U64 => 8,
        ElementKind::I8
        | ElementKind::U8
        | ElementKind::Bool
        | ElementKind::Fp8E4M3FN
        | ElementKind::Fp8E5M2 => 1,
        // A complex element is a PAIR: c64 is two f32, c128 is two f64. The
        // §6.1 token names the total width, so the name already says this.
        ElementKind::Complex64 => 8,
        ElementKind::Complex128 => 16,
        // Sub-byte dtypes share a byte. `elem_size` is the CONTAINER width here,
        // not the element width — use `elem_bits` for the latter and
        // `packed_bytes` for allocation.
        ElementKind::I4 | ElementKind::U4 | ElementKind::B1 => 1,
        other => panic!("oracle: unsupported dtype {other:?} (v1)"),
    }
}

/// `true` for the integer compute dtypes (`I32`/`I64`/`S8`/`U8`/`Bool`).
pub(super) fn is_int(dt: ElementKind) -> bool {
    matches!(
        dt,
        ElementKind::I32
            | ElementKind::I64
            | ElementKind::I8
            | ElementKind::U8
            | ElementKind::I16
            | ElementKind::U16
            | ElementKind::U32
            | ElementKind::U64
            | ElementKind::I4
            | ElementKind::U4
            | ElementKind::B1
            | ElementKind::Bool
    )
}

/// `true` for the complex dtypes.
///
/// Complex is a third compute domain alongside int and float, not a flavour of
/// either: it has no ordering (so no `Max`/`Min`/`Cmp*`), and its `Mul` mixes
/// components rather than acting elementwise.
pub(super) fn is_complex(dt: ElementKind) -> bool {
    matches!(dt, ElementKind::Complex64 | ElementKind::Complex128)
}

/// `true` when `dt`'s C arithmetic is **unsigned**.
///
/// This is not "is the storage unsigned" — `u8` and `u16` have unsigned storage
/// but *signed* arithmetic, because C's integer promotions lift any type of rank
/// below `int` to signed `int`. `unsigned int` has the same rank as `int`, so it
/// does not promote and its arithmetic really is unsigned modulo 2^32.
///
/// The distinction is invisible for `+`, `-`, `*` and the bitwise ops (identical
/// bit patterns under two's complement) and shows up in exactly two places: the
/// **value** a wrap produces (zero- vs sign-extension) and `>>` (logical vs
/// arithmetic shift). Both fall out of keeping unsigned values non-negative in
/// the wide accumulator — an `i128` right-shift of a non-negative value *is* a
/// logical shift.
pub(super) fn is_unsigned_arith(dt: ElementKind) -> bool {
    matches!(dt, ElementKind::U32 | ElementKind::U64)
}

/// The bit width the emitter's C arithmetic wraps at for `dt`: `I64` = 64;
/// `I32` and the 8-bit dtypes (C integer promotion widens `S8`/`U8` to `int`
/// during arithmetic, the store truncates back) = 32.
pub(super) fn op_width(dt: ElementKind) -> u32 {
    match dt {
        ElementKind::I64 | ElementKind::U64 => 64,
        _ => 32,
    }
}

/// Wrap `x` to its low `bits` bits, sign- or zero-extending per `dt`.
///
/// The single place the signed/unsigned split is applied. Zero-extension keeps
/// an unsigned value non-negative in the wide accumulator, which is what makes a
/// later `>>` behave as the logical shift C performs on an unsigned operand.
pub(super) fn wrap_for(x: i128, bits: u32, dt: ElementKind) -> i128 {
    if is_unsigned_arith(dt) {
        if bits >= 128 {
            return x;
        }
        return x & ((1i128 << bits) - 1);
    }
    wrap_bits(x, bits)
}

/// Sign-extend / truncate `x` to its low `bits` bits (two's-complement wrap).
pub(super) fn wrap_bits(x: i128, bits: u32) -> i128 {
    if bits >= 128 {
        return x;
    }
    let shift = 128 - bits;
    (x << shift) >> shift
}

/// The dtype's most-negative (`most_negative`) or most-positive integer extreme
/// — the scan/window Max/Min monoid identity.
pub(super) fn int_extreme(dt: ElementKind, most_negative: bool) -> i128 {
    match dt {
        ElementKind::I32 => {
            if most_negative {
                i128::from(i32::MIN)
            } else {
                i128::from(i32::MAX)
            }
        }
        ElementKind::I64 => {
            if most_negative {
                i128::from(i64::MIN)
            } else {
                i128::from(i64::MAX)
            }
        }
        ElementKind::I8 => {
            if most_negative {
                -128
            } else {
                127
            }
        }
        ElementKind::I16 => {
            if most_negative {
                i128::from(i16::MIN)
            } else {
                i128::from(i16::MAX)
            }
        }
        ElementKind::U16 => {
            if most_negative {
                0
            } else {
                i128::from(u16::MAX)
            }
        }
        ElementKind::U32 => {
            if most_negative {
                0
            } else {
                i128::from(u32::MAX)
            }
        }
        ElementKind::U64 => {
            if most_negative {
                0
            } else {
                i128::from(u64::MAX)
            }
        }
        ElementKind::U8 | ElementKind::Bool => {
            if most_negative {
                0
            } else {
                255
            }
        }
        other => panic!("oracle: no integer extreme for {other:?}"),
    }
}
