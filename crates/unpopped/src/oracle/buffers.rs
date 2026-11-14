// SPDX-License-Identifier: MIT OR Apache-2.0
//! The storage image: [`TypedBuffer`], the oracle's I/O unit.
//!
//! A section of the oracle (ROADMAP U30): `oracle.rs` glob-imports it, so every path it had is unchanged.

use super::*;

// ===========================================================================
// TypedBuffer — the storage image (the same byte image the GPU pointer sees).
// ===========================================================================

/// A dtype-tagged raw storage image plus its logical layout — the oracle's I/O
/// unit. `bytes` is the raw element storage (little-endian, native element
/// order per `strides`); feeding any dtype is uniform (byte blob + tag).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TypedBuffer {
    /// Element dtype of the storage.
    pub dtype: ElementKind,
    /// Per-axis logical extents.
    pub shape: Vec<i64>,
    /// Per-axis signed element strides (`0` = broadcast, `< 0` = flipped).
    pub strides: Vec<i64>,
    /// Element offset added to the operand base BEFORE address math — the
    /// [`crate::ir::BaseOffset::Runtime`] slice / a sub-view origin. `0` for a
    /// plain dense buffer.
    pub base_offset: i64,
    /// Raw storage bytes.
    pub bytes: Vec<u8>,
}

impl TypedBuffer {
    /// Build a buffer from an explicit dtype, layout, and raw bytes.
    #[must_use]
    pub fn new(dtype: ElementKind, shape: Vec<i64>, strides: Vec<i64>, bytes: Vec<u8>) -> Self {
        Self {
            dtype,
            shape,
            strides,
            base_offset: 0,
            bytes,
        }
    }

    /// Dense (contiguous, row-major) buffer of `f32` values.
    #[must_use]
    pub fn from_f32(shape: &[i64], data: &[f32]) -> Self {
        let mut bytes = Vec::with_capacity(data.len() * 4);
        for &v in data {
            bytes.extend_from_slice(&v.to_bits().to_le_bytes());
        }
        Self::new(
            ElementKind::F32,
            shape.to_vec(),
            dense_strides(shape),
            bytes,
        )
    }

    /// Dense buffer of `f64` values.
    #[must_use]
    pub fn from_f64(shape: &[i64], data: &[f64]) -> Self {
        let mut bytes = Vec::with_capacity(data.len() * 8);
        for &v in data {
            bytes.extend_from_slice(&v.to_bits().to_le_bytes());
        }
        Self::new(
            ElementKind::F64,
            shape.to_vec(),
            dense_strides(shape),
            bytes,
        )
    }

    /// Dense buffer of `i32` values.
    #[must_use]
    pub fn from_i32(shape: &[i64], data: &[i32]) -> Self {
        let mut bytes = Vec::with_capacity(data.len() * 4);
        for &v in data {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        Self::new(
            ElementKind::I32,
            shape.to_vec(),
            dense_strides(shape),
            bytes,
        )
    }

    /// Dense buffer of `i64` values.
    #[must_use]
    pub fn from_i64(shape: &[i64], data: &[i64]) -> Self {
        let mut bytes = Vec::with_capacity(data.len() * 8);
        for &v in data {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        Self::new(
            ElementKind::I64,
            shape.to_vec(),
            dense_strides(shape),
            bytes,
        )
    }

    /// Dense buffer of `i8` (`S8`) values.
    #[must_use]
    pub fn from_i8(shape: &[i64], data: &[i8]) -> Self {
        let bytes: Vec<u8> = data.iter().map(|&v| v as u8).collect();
        Self::new(ElementKind::I8, shape.to_vec(), dense_strides(shape), bytes)
    }

    /// Dense buffer of `i16` (`S16`) values, little-endian.
    #[must_use]
    pub fn from_i16(shape: &[i64], data: &[i16]) -> Self {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        Self::new(
            ElementKind::I16,
            shape.to_vec(),
            dense_strides(shape),
            bytes,
        )
    }

    /// Dense buffer of `u16` (`U16`) values, little-endian.
    #[must_use]
    pub fn from_u16(shape: &[i64], data: &[u16]) -> Self {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        Self::new(
            ElementKind::U16,
            shape.to_vec(),
            dense_strides(shape),
            bytes,
        )
    }

    /// Dense buffer of `u32` (`U32`) values, little-endian.
    ///
    /// The arithmetic these feed is genuinely **unsigned**: unlike `u8`/`u16`,
    /// `u32` does not integer-promote to signed `int`, so a value above
    /// `i32::MAX` stays positive through the whole evaluation.
    #[must_use]
    pub fn from_u32(shape: &[i64], data: &[u32]) -> Self {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        Self::new(
            ElementKind::U32,
            shape.to_vec(),
            dense_strides(shape),
            bytes,
        )
    }

    /// Dense buffer of `c64` values — pairs of `f32`, real component first.
    #[must_use]
    pub fn from_complex64(shape: &[i64], data: &[(f32, f32)]) -> Self {
        let bytes: Vec<u8> = data
            .iter()
            .flat_map(|(re, im)| {
                let mut v = re.to_le_bytes().to_vec();
                v.extend_from_slice(&im.to_le_bytes());
                v
            })
            .collect();
        Self::new(
            ElementKind::Complex64,
            shape.to_vec(),
            dense_strides(shape),
            bytes,
        )
    }

    /// Dense buffer of `c128` values — pairs of `f64`, real component first.
    #[must_use]
    pub fn from_complex128(shape: &[i64], data: &[(f64, f64)]) -> Self {
        let bytes: Vec<u8> = data
            .iter()
            .flat_map(|(re, im)| {
                let mut v = re.to_le_bytes().to_vec();
                v.extend_from_slice(&im.to_le_bytes());
                v
            })
            .collect();
        Self::new(
            ElementKind::Complex128,
            shape.to_vec(),
            dense_strides(shape),
            bytes,
        )
    }

    /// Every element as an exact `(re, im)` pair.
    ///
    /// # Panics
    ///
    /// If the buffer's dtype is not a complex dtype.
    #[must_use]
    pub fn to_complex_vec(&self) -> Vec<(f64, f64)> {
        assert!(
            is_complex(self.dtype),
            "to_complex_vec on non-complex dtype {:?}",
            self.dtype
        );
        let n = self.elem_count();
        (0..n)
            .map(|i| raw_to_complex(read_elem(&self.bytes, i, self.dtype), self.dtype))
            .collect()
    }

    /// Dense buffer of `u64` (`U64`) values, little-endian.
    ///
    /// Note the projection asymmetry these expose: `to_i128_vec` is exact for
    /// every value, while `to_f64_vec` is not above 2^53. That is why the
    /// tolerant comparator routes integers through `i128` — see
    /// `tests/wide_integer_comparison.rs`.
    #[must_use]
    pub fn from_u64(shape: &[i64], data: &[u64]) -> Self {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        Self::new(
            ElementKind::U64,
            shape.to_vec(),
            dense_strides(shape),
            bytes,
        )
    }

    /// Dense buffer of `u8` values.
    #[must_use]
    pub fn from_u8(shape: &[i64], data: &[u8]) -> Self {
        Self::new(
            ElementKind::U8,
            shape.to_vec(),
            dense_strides(shape),
            data.to_vec(),
        )
    }

    /// Dense buffer holding `data` **encoded into `dt`'s storage format**.
    ///
    /// The counterpart of [`TypedBuffer::to_f64_vec`]: that projects storage into
    /// `f64`, this rounds `f64` into storage. Together they are the round-trip a
    /// codec has to satisfy, and having both as public constructors is what lets
    /// a test walk every representable pattern of a narrow dtype and check it
    /// survives the journey.
    ///
    /// # Panics
    ///
    /// If `dt` is not a float dtype with an encoder.
    #[must_use]
    pub fn from_f64_as(dt: ElementKind, shape: &[i64], data: &[f64]) -> Self {
        let sz = elem_size(dt);
        let mut bytes = vec![0u8; data.len() * sz];
        for (i, &v) in data.iter().enumerate() {
            write_le(&mut bytes, i * sz, sz, encode_float(v, dt));
        }
        Self::new(dt, shape.to_vec(), dense_strides(shape), bytes)
    }

    /// Dense buffer of sub-byte values, packed per KISS-CLASSIFY §6.1.
    ///
    /// Takes the logical element values (not bytes) and packs them:
    /// * `i4`/`u4` — two per byte, **low nibble = even index, high nibble = odd**.
    /// * `b1` — eight per byte, **LSB = lowest logical index**.
    ///
    /// The packing is normative (§6.0-0001 makes it part of what the dtype *is*),
    /// which is why it is spelled here rather than left to a caller: a buffer
    /// packed the other way round would round-trip through this crate perfectly
    /// and disagree with every other implementation.
    ///
    /// # Panics
    ///
    /// If `dt` is not a sub-byte dtype, or a value does not fit its width.
    #[must_use]
    pub fn from_sub_byte(dt: ElementKind, shape: &[i64], data: &[i8]) -> Self {
        assert!(
            is_sub_byte(dt),
            "from_sub_byte on {dt:?}, which is not sub-byte"
        );
        let mut bytes = vec![0u8; packed_bytes(dt, data.len())];
        for (i, &v) in data.iter().enumerate() {
            let bits = match dt {
                ElementKind::I4 => {
                    assert!((-8..=7).contains(&v), "i4 value {v} outside [-8, 7]");
                    u128::from((v as u8) & 0x0f)
                }
                ElementKind::U4 => {
                    assert!((0..=15).contains(&v), "u4 value {v} outside [0, 15]");
                    u128::from(v as u8 & 0x0f)
                }
                _ => {
                    assert!((0..=1).contains(&v), "b1 value {v} outside [0, 1]");
                    u128::from(v as u8 & 1)
                }
            };
            write_elem(&mut bytes, i, dt, bits);
        }
        Self::new(dt, shape.to_vec(), dense_strides(shape), bytes)
    }

    /// A buffer over already-packed sub-byte STORAGE bytes.
    ///
    /// The counterpart of [`TypedBuffer::from_sub_byte`], which packs values.
    /// This wraps bytes that are already packed — the shape a kernel wrote.
    #[must_use]
    pub fn from_packed_bytes(dt: ElementKind, shape: &[i64], bytes: &[u8]) -> Self {
        Self::new(dt, shape.to_vec(), dense_strides(shape), bytes.to_vec())
    }

    /// Dense buffer of raw FP8 bit patterns for `dt`.
    ///
    /// Takes bits rather than values because an 8-bit float's whole domain is
    /// 256 patterns — small enough to enumerate exhaustively, which is the only
    /// dtype family where "supports every possible value" can be *proved*
    /// rather than sampled.
    ///
    /// # Panics
    ///
    /// If `dt` is not one of the two computable FP8 dtypes. The MX scales
    /// (`f8e8m0`/`f8e6m2`) are deliberately excluded: KISS-CLASSIFY §6.1-0013
    /// makes them per-block **scale** types carried as sibling operands, never
    /// element value dtypes.
    #[must_use]
    pub fn from_fp8_bits(dt: ElementKind, shape: &[i64], data: &[u8]) -> Self {
        assert!(
            matches!(dt, ElementKind::Fp8E4M3FN | ElementKind::Fp8E5M2),
            "from_fp8_bits on {dt:?}, which is not a computable FP8 dtype"
        );
        Self::new(dt, shape.to_vec(), dense_strides(shape), data.to_vec())
    }

    /// Dense buffer of raw `f16` bit patterns.
    #[must_use]
    pub fn from_f16_bits(shape: &[i64], data: &[u16]) -> Self {
        let mut bytes = Vec::with_capacity(data.len() * 2);
        for &v in data {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        Self::new(
            ElementKind::F16,
            shape.to_vec(),
            dense_strides(shape),
            bytes,
        )
    }

    /// Number of logical elements (`∏ shape`).
    #[must_use]
    pub fn len(&self) -> usize {
        prod(&self.shape).max(0) as usize
    }

    /// `true` if the buffer has no elements.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Decode every stored element to `f64` in linear byte order.
    #[must_use]
    pub fn to_f64_vec(&self) -> Vec<f64> {
        let n = self.elem_count();
        (0..n)
            .map(|i| raw_to_f64(read_elem(&self.bytes, i, self.dtype), self.dtype))
            .collect()
    }

    /// Every element as an exact wide integer.
    ///
    /// The integer counterpart of [`TypedBuffer::to_f64_vec`], and the reason it
    /// exists: `f64` has a 53-bit significand, so the `f64` projection **cannot
    /// distinguish** two `i64`/`u64` values that differ above 2^53. `2^53` and
    /// `2^53 + 1` both land on `9007199254740992.0`. Every integer dtype in the
    /// §6.1 set fits in `i128` exactly, so this projection is lossless for all of
    /// them.
    ///
    /// # Panics
    ///
    /// If the buffer's dtype is not an integer dtype.
    #[must_use]
    pub fn to_i128_vec(&self) -> Vec<i128> {
        assert!(
            is_int(self.dtype),
            "to_i128_vec on non-integer dtype {:?}",
            self.dtype
        );
        let n = self.elem_count();
        (0..n)
            .map(|i| raw_to_i128(read_elem(&self.bytes, i, self.dtype), self.dtype))
            .collect()
    }

    /// The raw storage bits of element `i` in linear byte order.
    #[must_use]
    pub fn bits_at(&self, i: usize) -> u128 {
        read_elem(&self.bytes, i, self.dtype)
    }

    /// The raw storage bytes, for tests that must assert a PACKING rather than a
    /// round-trip.
    #[must_use]
    pub fn raw_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// How many elements the buffer holds.
    ///
    /// Derived from the SHAPE for a sub-byte dtype rather than from the byte
    /// length: `bytes.len() / elem_size` would report 2x too few `i4` elements
    /// and 8x too few `b1` ones, since several share a byte. For whole-byte
    /// dtypes the two agree.
    #[must_use]
    pub fn elem_count(&self) -> usize {
        if is_sub_byte(self.dtype) {
            self.shape.iter().product::<i64>().max(0) as usize
        } else {
            self.bytes.len() / elem_size(self.dtype)
        }
    }
}

/// Dense (row-major, last-axis-fastest) strides for `shape`.
pub(super) fn dense_strides(shape: &[i64]) -> Vec<i64> {
    let mut s = vec![0i64; shape.len()];
    let mut acc = 1i64;
    for d in (0..shape.len()).rev() {
        s[d] = acc;
        acc *= shape[d];
    }
    s
}
