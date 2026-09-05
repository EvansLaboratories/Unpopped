//! # CPU ORACLE (plan-interpreter) — v1
//!
//! An **independent** CPU interpreter that computes what a generated CUDA kernel
//! computes, from the SAME IR ([`crate::plan::KernelPlan`] + [`OperandDesc`]) but
//! via a **separate code path**. It is therefore a differential test of the
//! emission (`crate::cuda`), a universal numeric reference, and a GPU-free CI base.
//!
//! ## Independence (the review discipline)
//!
//! This module shares **zero** lowering code with the emitter. It does NOT call
//! `lower_expr` / `lower_dag` / the `Lowering` struct / `unary_f32` / `binary_f32`
//! / `binary_int` / `cuda_select` / `const_lit` / `offset_expr` /
//! `gathered_offset_expr` / any `emit_*`. Every `ScalarExpr` evaluation, every
//! scalar op, all index / fold / sort math, and the half codec are re-implemented
//! here from each op's DEFINITION (not its CUDA spelling), so a shared spelling
//! bug cannot hide in both the emitter and the oracle.
//!
//! It MAY reuse the pieces UPSTREAM of both lowerings: the IR types
//! ([`crate::ir`]), [`crate::plan::build_plan`]/[`KernelPlan`], and the operand
//! role classifier [`crate::plan::rr_role`] (a broadcast-mask predicate, not a
//! spelling).
//!
//! **Honest scope:** the oracle catches EMISSION bugs, not IR-construction bugs
//! (it reads the same `build_plan` output the emitter does). It complements — does
//! not replace — the hand/bespoke oracles and the `build_plan`-direct gate tests.
//!
//! ## Which of this module's rules a BACKEND must reproduce
//!
//! Much of what follows is phrased as "the emitter's inner loop order",
//! "mirroring the emitter", "the emitter tests `((float)(c)) != 0.0f`". That was
//! sufficient while CUDA was the only backend — "what it does" and "what is
//! required" were the same sentence. They are not any more, and the two kinds of
//! statement look identical in this source.
//!
//! `docs/conformance.md` separates them: which rules are NORMATIVE (a backend
//! that differs produces output the oracle rejects), which are INCIDENTAL
//! (mirrored from CUDA, absorbed by the tolerant comparator, free to differ), and
//! which are OPEN. Two worth knowing before writing a backend:
//!
//! - **Discontinuities decide in the COMPUTE dtype**, never in wider precision.
//!   A tolerance cannot absorb a discontinuity — deciding `x > 0` in f64 where
//!   the kernel decides in f32 selects the other branch, not a nearby value.
//! - **`Max`/`Min` propagate NaN**, which is *not* C `fmax` / IEEE `maxNum`
//!   semantics and *not* GLSL `max`. Every C-family target offers a convenient
//!   built-in with the opposite behavior, and the difference is invisible on all
//!   NaN-free input.
//!
//! ## Precision posture
//!
//! Correctness/precision over speed: arithmetic accumulates in `f64` (never the
//! emitter's "double-then-round-once f32" convention — that would couple the
//! oracle to the speller). Integer arithmetic/bitwise is exact-wrapping in the
//! two's-complement width the emitter uses. The [`compare`] helper encodes the
//! bit-exact-vs-tolerance dichotomy.
//!
//! ## v1 scope
//!
//! [`Access::Elementwise`], [`Access::Reduction`], [`Access::RowReduce`],
//! [`Access::Scan`], [`Access::Window`], [`Access::Im2Col`] + the full
//! `ScalarExpr` vocabulary + layout math (contiguous / strided / broadcast /
//! flipped / permuted / base-offset) + multi-output / hetero.
//!
//! ### Not covered
//!
//! Stated as scope rather than as a TODO list, because this list had gone stale
//! in the direction that matters: it claimed `Access::Contraction` was deferred
//! long after `eval_contraction` shipped and `evaluate` began dispatching it.
//! A reader trusting it would conclude matmul results were unvalidated and skip
//! writing the differential test that already works. An over-cautious scope note
//! is not harmless — it suppresses testing.
//!
//! - `Access::RowSort` — the NaN-greatest `key_lt`, stable index ties, and the
//!   TopK comparator. Genuinely not implemented.
//! - gather / scatter — `ReadIndex`/`WriteIndex` OOB policies plus FP-`atomicAdd`
//!   nondeterminism. Note this one is not merely unimplemented: under a
//!   nondeterministic accumulation order only an **order-independent invariant**
//!   is checkable at all, so it needs a different notion of "correct" rather
//!   than more of this module.
//!
//! **`Access::Contraction` IS covered** (`eval_contraction`, dispatched by
//! [`evaluate`]) — rank-2 and batched rank-3, transposed / permuted / broadcast
//! operands, the `Reduced(0)` epilogue and the fused per-column bias.
//!
//! **Device-side differential fuzzing is not deferred, it is elsewhere**: the
//! cross-backend IR fuzzer moved to `baracuda-cuda-emit`'s integration tests
//! during the carve, because it drives a device backend. It is not something
//! this crate can host.
//!
//! ### Transcendental accuracy
//!
//! `Erf`/`Erfc`/`Gelu` use libm (~1 ULP f64) — a pure-Rust implementation
//! INDEPENDENT of the emitter's device `erf`/`erff`, so the differential check
//! stays honest while being a genuinely tighter reference than the kernel it
//! validates. `Lgamma` uses an in-house Lanczos series (~1e-13). `exp`/`ln`/
//! `sqrt`/`tanh`/… are accurate std `f64`. None is used in a bit-exact path.
//!
//! ### Runtime base offsets
//!
//! A [`crate::ir::BaseOffset::Runtime`] value is threaded through
//! [`TypedBuffer::base_offset`] (an element offset applied to the operand's base
//! BEFORE all per-element address math), since the plan carries only the PRESENCE
//! mask, not the runtime value.

use crate::ir::{Access, BinaryOp, ReduceOp, ScalarExpr, SortOrder, SortOut, UnaryOp, View};
use crate::plan::{KernelPlan, RrRole, Schedule, rr_role};
use unpopped_vocab::{ElementKind, OperandDesc};

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
fn dense_strides(shape: &[i64]) -> Vec<i64> {
    let mut s = vec![0i64; shape.len()];
    let mut acc = 1i64;
    for d in (0..shape.len()).rev() {
        s[d] = acc;
        acc *= shape[d];
    }
    s
}

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
fn elem_bits(dt: ElementKind) -> usize {
    match dt {
        ElementKind::I4 | ElementKind::U4 => 4,
        ElementKind::B1 => 1,
        other => elem_size(other) * 8,
    }
}

/// `true` for the dtypes stored more than one to a byte.
fn is_sub_byte(dt: ElementKind) -> bool {
    elem_bits(dt) < 8
}

/// Bytes needed to hold `n` elements of `dt`, rounding a partial trailing byte up.
fn packed_bytes(dt: ElementKind, n: usize) -> usize {
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
fn read_elem(bytes: &[u8], i: usize, dt: ElementKind) -> u128 {
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
fn write_elem(bytes: &mut [u8], i: usize, dt: ElementKind, bits: u128) {
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

fn elem_size(dt: ElementKind) -> usize {
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
fn is_int(dt: ElementKind) -> bool {
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
fn is_complex(dt: ElementKind) -> bool {
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
fn is_unsigned_arith(dt: ElementKind) -> bool {
    matches!(dt, ElementKind::U32 | ElementKind::U64)
}

/// The bit width the emitter's C arithmetic wraps at for `dt`: `I64` = 64;
/// `I32` and the 8-bit dtypes (C integer promotion widens `S8`/`U8` to `int`
/// during arithmetic, the store truncates back) = 32.
fn op_width(dt: ElementKind) -> u32 {
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
fn wrap_for(x: i128, bits: u32, dt: ElementKind) -> i128 {
    if is_unsigned_arith(dt) {
        if bits >= 128 {
            return x;
        }
        return x & ((1i128 << bits) - 1);
    }
    wrap_bits(x, bits)
}

/// Sign-extend / truncate `x` to its low `bits` bits (two's-complement wrap).
fn wrap_bits(x: i128, bits: u32) -> i128 {
    if bits >= 128 {
        return x;
    }
    let shift = 128 - bits;
    (x << shift) >> shift
}

/// The dtype's most-negative (`most_negative`) or most-positive integer extreme
/// — the scan/window Max/Min monoid identity.
fn int_extreme(dt: ElementKind, most_negative: bool) -> i128 {
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

// ===========================================================================
// Half codec (independent bit-level decode/encode — NOT __half2float).
// ===========================================================================

/// Decode an IEEE-754 binary16 bit pattern to `f64`.
fn f16_to_f64(bits: u16) -> f64 {
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
fn fp8_e4m3fn_to_f64(bits: u8) -> f64 {
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
fn fp8_e5m2_to_f64(bits: u8) -> f64 {
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
fn f64_to_fp8_e4m3fn_bits(x: f64) -> u8 {
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
fn f64_to_fp8_e5m2_bits(x: f64) -> u8 {
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
fn bf16_to_f64(bits: u16) -> f64 {
    f64::from(f32::from_bits(u32::from(bits) << 16))
}

/// Round an `f32` to a binary16 bit pattern (round-to-nearest, ties-to-even).
fn f32_to_f16_bits(x: f32) -> u16 {
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
fn f32_to_bf16_bits(x: f32) -> u16 {
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
fn read_le(bytes: &[u8], byte_off: usize, sz: usize) -> u128 {
    let mut b = [0u8; 16];
    b[..sz].copy_from_slice(&bytes[byte_off..byte_off + sz]);
    u128::from_le_bytes(b)
}

/// Write the low `sz` little-endian bytes of `bits` into `bytes` at `byte_off`.
fn write_le(bytes: &mut [u8], byte_off: usize, sz: usize, bits: u128) {
    let b = bits.to_le_bytes();
    bytes[byte_off..byte_off + sz].copy_from_slice(&b[..sz]);
}

// ===========================================================================
// The value domain.
// ===========================================================================

/// A scalar value in one of three domains: a float compute value, a wide
/// integer compute value, or raw storage bits that must move verbatim (a leaf
/// load / a bit-preserving move — so `-0.0`, NaN payloads, and signaling NaNs
/// survive untouched through an identity body or a `Select` arm pick).
#[derive(Clone, Copy, Debug)]
enum Val {
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
fn raw_to_i128(bits: u128, dt: ElementKind) -> i128 {
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
fn raw_to_complex(bits: u128, dt: ElementKind) -> (f64, f64) {
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
fn encode_int_as_complex(i: i128, out: ElementKind) -> u128 {
    encode_complex(i as f64, 0.0, out)
}

/// Encode a complex `(re, im)` pair into its storage pattern.
fn encode_complex(re: f64, im: f64, out: ElementKind) -> u128 {
    match out {
        ElementKind::Complex64 => {
            u128::from((re as f32).to_bits()) | (u128::from((im as f32).to_bits()) << 32)
        }
        ElementKind::Complex128 => u128::from(re.to_bits()) | (u128::from(im.to_bits()) << 64),
        other => panic!("oracle: encode_complex to non-complex dtype {other:?}"),
    }
}

/// Decode a raw storage pattern to `f64` (independent half decode for f16/bf16).
fn raw_to_f64(bits: u128, dt: ElementKind) -> f64 {
    match dt {
        ElementKind::Fp8E4M3FN => fp8_e4m3fn_to_f64(bits as u8),
        ElementKind::Fp8E5M2 => fp8_e5m2_to_f64(bits as u8),
        ElementKind::F16 => f16_to_f64(bits as u16),
        ElementKind::Bf16 => bf16_to_f64(bits as u16),
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
    fn f64(self) -> f64 {
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
    fn complex(self) -> (f64, f64) {
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
    fn i128(self) -> i128 {
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
fn round_to_compute(f: f64, dt: ElementKind) -> f64 {
    match dt {
        ElementKind::F16 | ElementKind::Bf16 | ElementKind::F32 | ElementKind::F32Strict => {
            raw_to_f64(encode_float(f, dt), dt)
        }
        _ => f,
    }
}

// ===========================================================================
// Independent scalar op semantics (match DEFINITIONS, not CUDA spellings).
// ===========================================================================

/// Independent `f64` unary-op evaluator (op DEFINITIONS, not the emitter's
/// spelling). Transcendentals via Rust std / the local erf/lgamma helpers.
fn unary_op_f64(op: UnaryOp, x: f64) -> f64 {
    use std::f64::consts::FRAC_1_SQRT_2;
    match op {
        UnaryOp::Neg => -x,
        UnaryOp::Abs => x.abs(),
        UnaryOp::Sqr => x * x,
        UnaryOp::Sqrt => x.sqrt(),
        UnaryOp::Rsqrt => 1.0 / x.sqrt(),
        UnaryOp::Recip => 1.0 / x,
        UnaryOp::Exp => x.exp(),
        UnaryOp::Log => x.ln(),
        UnaryOp::Tanh => x.tanh(),
        UnaryOp::Sigmoid => 1.0 / (1.0 + (-x).exp()),
        // Relu = x<0?0:x — NaN-propagating (NaN<0 is false), -0.0-preserving.
        UnaryOp::Relu => {
            if x < 0.0 {
                0.0
            } else {
                x
            }
        }
        UnaryOp::Erf => erf(x),
        UnaryOp::Gelu => 0.5 * x * (1.0 + erf(x * FRAC_1_SQRT_2)),
        UnaryOp::Silu => x * (1.0 / (1.0 + (-x).exp())),
        UnaryOp::Sin => x.sin(),
        UnaryOp::Cos => x.cos(),
        UnaryOp::Floor => x.floor(),
        UnaryOp::Ceil => x.ceil(),
        UnaryOp::Round => x.round_ties_even(),
        // Sign: NaN -> 0 (both comparisons false).
        UnaryOp::Sign => {
            if x > 0.0 {
                1.0
            } else if x < 0.0 {
                -1.0
            } else {
                0.0
            }
        }
        // Heaviside step(0)=0.
        UnaryOp::Step => {
            if x > 0.0 {
                1.0
            } else {
                0.0
            }
        }
        UnaryOp::Erfc => erfc(x),
        UnaryOp::Trunc => x.trunc(),
        UnaryOp::Exp2 => x.exp2(),
        UnaryOp::Expm1 => x.exp_m1(),
        UnaryOp::Log2 => x.log2(),
        UnaryOp::Log10 => x.log10(),
        UnaryOp::Log1p => x.ln_1p(),
        UnaryOp::Sinh => x.sinh(),
        UnaryOp::Cosh => x.cosh(),
        UnaryOp::Tan => x.tan(),
        UnaryOp::Asin => x.asin(),
        UnaryOp::Acos => x.acos(),
        UnaryOp::Atan => x.atan(),
        UnaryOp::Asinh => x.asinh(),
        UnaryOp::Acosh => x.acosh(),
        UnaryOp::Atanh => x.atanh(),
        UnaryOp::Cbrt => x.cbrt(),
        UnaryOp::Lgamma => lgamma(x),
    }
}

/// Independent `f64` binary-op evaluator for the NON-int-only ops. Max/Min are
/// NaN-PROPAGATING (distinct from FmaxIeee/FminIeee, NaN-suppressing); Rem is
/// FLOORED (sign-of-divisor), RemTrunc is C `fmod` (sign-of-dividend); the Cmp*
/// family yields exactly `1.0`/`0.0` with C NaN semantics. `Nextafter` is handled
/// dtype-aware by the caller (it steps a storage-dtype lattice) and never routes
/// here.
fn binary_op_f64(op: BinaryOp, a: f64, b: f64) -> f64 {
    match op {
        // NaN-propagating, and A ON TIES (`a >= b ? a : b` after the NaN
        // checks) — the KISS-Ops `max_prop` normative decomposition
        // (`select(…, select(cmp_ge(a,b), a, b))`) and the numpy/torch
        // `maximum` = `where(a >= b, a, b)` semantics. Bit-visible ONLY on
        // signed-zero ties: `max_prop(-0.0, +0.0) = -0.0` (keep a), never the
        // b-biased `+0.0`. Caught by the kiss-ref step-2 recipe differential.
        BinaryOp::Max => {
            if a.is_nan() {
                a
            } else if b.is_nan() {
                b
            } else if a >= b {
                a
            } else {
                b
            }
        }
        // Mirror: `min_prop` = `select(…, select(cmp_le(a,b), a, b))` — a on ties.
        BinaryOp::Min => {
            if a.is_nan() {
                a
            } else if b.is_nan() {
                b
            } else if a <= b {
                a
            } else {
                b
            }
        }
        BinaryOp::Pow => a.powf(b),
        // Floored remainder: a - floor(a/b)*b (sign of divisor).
        BinaryOp::Rem => a - (a / b).floor() * b,
        // Truncated remainder: C fmod (Rust `%` on f64 is fmod).
        BinaryOp::RemTrunc => a % b,
        BinaryOp::Atan2 => a.atan2(b),
        BinaryOp::Copysign => a.copysign(b),
        // NaN-SUPPRESSING (std max/min return the non-NaN operand).
        BinaryOp::FmaxIeee => a.max(b),
        BinaryOp::FminIeee => a.min(b),
        BinaryOp::CmpEq => f64::from(a == b),
        BinaryOp::CmpNe => f64::from(a != b),
        BinaryOp::CmpLt => f64::from(a < b),
        BinaryOp::CmpLe => f64::from(a <= b),
        BinaryOp::CmpGt => f64::from(a > b),
        BinaryOp::CmpGe => f64::from(a >= b),
        BinaryOp::Nextafter => panic!("oracle: Nextafter is dtype-aware; handled in eval"),
        BinaryOp::BitAnd
        | BinaryOp::BitOr
        | BinaryOp::BitXor
        | BinaryOp::Shl
        | BinaryOp::Shr
        | BinaryOp::LogicalAnd
        | BinaryOp::LogicalOr
        | BinaryOp::LogicalXor => {
            panic!("oracle: {op:?} is int-only; handled in the integer eval path")
        }
    }
}

/// Independent integer binary-op evaluator (bitwise/shift/logical), wrapping at
/// the emitter's C promotion width. Matches the bespoke functor definitions.
fn binary_op_int(op: BinaryOp, a: i128, b: i128, dt: ElementKind) -> i128 {
    let w = op_width(dt);
    match op {
        BinaryOp::BitAnd => wrap_for(a & b, w, dt),
        BinaryOp::BitOr => wrap_for(a | b, w, dt),
        BinaryOp::BitXor => wrap_for(a ^ b, w, dt),
        BinaryOp::Shl => {
            // Out-of-range shift amounts are architecture-inherited UB; clamp to
            // keep the interpreter total (never exercised by valid inputs).
            if (0..i128::from(w)).contains(&b) {
                wrap_for(a << (b as u32), w, dt)
            } else {
                0
            }
        }
        BinaryOp::Shr => {
            // C's `>>` is ARITHMETIC on a signed operand and LOGICAL on an
            // unsigned one. Both fall out of one expression: `wrap_for` keeps an
            // unsigned dtype non-negative in the wide accumulator, and an `i128`
            // shift of a non-negative value is already logical. The `a < 0` tail
            // is therefore unreachable for an unsigned dtype, which is what makes
            // `u32 >> k` correct rather than sign-propagating.
            if (0..i128::from(w)).contains(&b) {
                wrap_for(a >> (b as u32), w, dt)
            } else if a < 0 {
                -1
            } else {
                0
            }
        }
        BinaryOp::LogicalAnd => i128::from(a != 0 && b != 0),
        BinaryOp::LogicalOr => i128::from(a != 0 || b != 0),
        BinaryOp::LogicalXor => i128::from((a != 0) != (b != 0)),
        other => panic!("oracle: {other:?} is not an integer op"),
    }
}

/// Independent nextafter over the f64 lattice (Rust std has no `nextafter`).
fn next_after_f64(x: f64, y: f64) -> f64 {
    if x.is_nan() || y.is_nan() {
        return f64::NAN;
    }
    if x == y {
        return y;
    }
    if x == 0.0 {
        return f64::from_bits(1).copysign(y);
    }
    let bits = x.to_bits();
    let step_up = (y > x) == (x > 0.0);
    f64::from_bits(if step_up { bits + 1 } else { bits - 1 })
}

/// Independent nextafter over the f32 lattice.
fn next_after_f32(x: f32, y: f32) -> f32 {
    if x.is_nan() || y.is_nan() {
        return f32::NAN;
    }
    if x == y {
        return y;
    }
    if x == 0.0 {
        return f32::from_bits(1).copysign(y);
    }
    let bits = x.to_bits();
    let step_up = (y > x) == (x > 0.0);
    f32::from_bits(if step_up { bits + 1 } else { bits - 1 })
}

/// Accurate `f64` `erfc` (libm, ~1 ULP). An INDEPENDENT implementation (pure-Rust
/// MUSL/fdlibm heritage) from the emitter's device `erfc`/`erfcf`, so it stays a
/// valid differential reference. (v1 used a compact ~1e-7 Numerical-Recipes
/// rational — too loose to reference an f32/f64 `erfc`/`gelu` kernel, whose device
/// `erf` is itself ~1 ULP; adversarial review flagged it, so upgraded.)
fn erfc(x: f64) -> f64 {
    libm::erfc(x)
}

/// Accurate `f64` `erf` (libm, ~1 ULP) — independent of the emitter's device `erf`.
fn erf(x: f64) -> f64 {
    libm::erf(x)
}

/// Lanczos `ln|Γ(x)|` (g=7, ~1e-13).
fn lgamma(x: f64) -> f64 {
    use std::f64::consts::PI;
    // Reference Lanczos coefficients — kept at full published precision.
    #[allow(clippy::excessive_precision)]
    const G: [f64; 9] = [
        0.999_999_999_999_809_93,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        // reflection: ln|Γ(x)| = ln(π / |sin(π x)|) - ln|Γ(1-x)|
        (PI / (PI * x).sin()).abs().ln() - lgamma(1.0 - x)
    } else {
        let x = x - 1.0;
        let mut a = G[0];
        let t = x + 7.5;
        for (i, &g) in G.iter().enumerate().skip(1) {
            a += g / (x + i as f64);
        }
        0.5 * (2.0 * PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
    }
}

// ===========================================================================
// The independent ScalarExpr evaluator.
// ===========================================================================

/// Leaf/reduced/coord accessors for one evaluation context — the oracle's
/// independent analog of the emitter's `Lowering`, producing VALUES not strings
/// and sharing no code with `lower_expr`.
struct Eval<'a> {
    /// Compute dtype (`plan.dtype`) — selects float vs. integer arithmetic.
    dtype: ElementKind,
    /// Runtime scalar params.
    params: &'a [f64],
    /// `Input(i)` loader.
    leaf: &'a dyn Fn(u8) -> Val,
    /// `Reduced(i)` scalar accessor.
    reduced: &'a dyn Fn(u8) -> Val,
    /// `Coord(d)` accessor (Elementwise only).
    coord: &'a dyn Fn(u8) -> Val,
}

/// `Reduced` leaf reached outside a reduction context.
fn panic_reduced(s: u8) -> Val {
    panic!("oracle: Reduced({s}) leaf outside a reduction/scan/window context")
}

/// `Coord` leaf reached outside an Elementwise body.
fn panic_coord(d: u8) -> Val {
    panic!("oracle: Coord({d}) is Elementwise-only")
}

/// Evaluate a `ScalarExpr` TREE (hash-consing / `ExprDag` is an emission concern,
/// ignored). Every node is computed from its DEFINITION.
fn eval(e: &ScalarExpr, ev: &Eval<'_>) -> Val {
    match e {
        ScalarExpr::Input(i) => (ev.leaf)(*i),
        ScalarExpr::Const(v) => Val::Float(*v),
        ScalarExpr::Param(i) => Val::Float(ev.params[*i as usize]),
        ScalarExpr::Reduced(i) => (ev.reduced)(*i),
        ScalarExpr::Coord(d) => (ev.coord)(*d),
        ScalarExpr::Add(a, b) => arith(
            ev,
            a,
            b,
            |x, y| x + y,
            i128::wrapping_add,
            |(ar, ai), (br, bi)| (ar + br, ai + bi),
        ),
        ScalarExpr::Sub(a, b) => arith(
            ev,
            a,
            b,
            |x, y| x - y,
            i128::wrapping_sub,
            |(ar, ai), (br, bi)| (ar - br, ai - bi),
        ),
        // The one that is NOT component-wise.
        ScalarExpr::Mul(a, b) => arith(
            ev,
            a,
            b,
            |x, y| x * y,
            i128::wrapping_mul,
            |(ar, ai), (br, bi)| (ar * br - ai * bi, ar * bi + ai * br),
        ),
        // Div goes through `arith` like the other three. Reading `.f64()`
        // directly — as this did — PANICS on a complex operand, because
        // `Val::f64()` is deliberately fatal there. That was safe only for as
        // long as the plan gate refused complex `Div`, i.e. a correctness
        // property held at a distance by an unrelated file.
        ScalarExpr::Div(a, b) => arith(
            ev,
            a,
            b,
            |x, y| x / y,
            |_, _| unreachable!("integer Div is refused at the plan gate"),
            // Smith's algorithm — scale by the larger denominator component so
            // no intermediate squares. The emitted C runs the same recurrence;
            // the textbook form would agree to rounding on well-conditioned
            // inputs and disagree wildly where either overflows, so matching the
            // algorithm is what makes the differential test meaningful rather
            // than a tolerance negotiation.
            |(ar, ai), (br, bi)| {
                if br.abs() >= bi.abs() {
                    let (q, den) = (bi / br, br + bi * (bi / br));
                    ((ar + ai * q) / den, (ai - ar * q) / den)
                } else {
                    let (q, den) = (br / bi, br * (br / bi) + bi);
                    ((ar * q + ai) / den, (ai * q - ar) / den)
                }
            },
        ),
        ScalarExpr::Unary(op, x) => {
            let xv = eval(x, ev).f64();
            // Sign/Step are DISCONTINUOUS (a full ±1/0 flip): the emitter decides
            // on the compute-dtype float register (`x > 0.0f`), so round the
            // argument to the compute dtype first. Continuous unary ops
            // (exp/log/tanh/…) stay accurate-f64 + tolerance.
            let xv = if matches!(op, UnaryOp::Sign | UnaryOp::Step) {
                round_to_compute(xv, ev.dtype)
            } else {
                xv
            };
            Val::Float(unary_op_f64(*op, xv))
        }
        ScalarExpr::Binary(op, a, b) => eval_binary(*op, a, b, ev),
        ScalarExpr::Select(c, a, b) => {
            // cond tested in the compute dtype: -0.0 is false, NaN is true. Round
            // the condition to the compute dtype first — the emitter tests
            // `((float)(c)) != 0.0f`, so a cond sub-expression that underflows to
            // ±0 there must pick the same arm here. The chosen arm MOVES verbatim
            // (Raw stays Raw so -0.0 / NaN payloads / signaling survive; no
            // arithmetic ever touches an arm).
            if round_to_compute(eval(c, ev).f64(), ev.dtype) != 0.0 {
                eval(a, ev)
            } else {
                eval(b, ev)
            }
        }
    }
}

/// Infix arithmetic (`Add`/`Sub`/`Mul`) in the compute dtype's domain: exact-
/// wrapping integers (each op wraps at the C width) or `f64` floats.
fn arith(
    ev: &Eval<'_>,
    a: &ScalarExpr,
    b: &ScalarExpr,
    ff: impl Fn(f64, f64) -> f64,
    fi: impl Fn(i128, i128) -> i128,
    // The complex rule is passed EXPLICITLY rather than derived from `ff`.
    // Component-wise application of the real rule is correct for `+` and `-`
    // and WRONG for `*`: complex multiplication mixes components
    // (`ac - bd, ad + bc`), and no inspection of a closure can recover which
    // operation it represents. Three closures is the price of not guessing.
    fc: impl Fn((f64, f64), (f64, f64)) -> (f64, f64),
) -> Val {
    if is_complex(ev.dtype) {
        let (ar, ai) = eval(a, ev).complex();
        let (br, bi) = eval(b, ev).complex();
        let (re, im) = fc((ar, ai), (br, bi));
        return Val::Complex(re, im);
    }
    if is_int(ev.dtype) {
        let r = fi(eval(a, ev).i128(), eval(b, ev).i128());
        // `+`/`-`/`*` produce identical BIT patterns signed or unsigned, but the
        // wrapped VALUE differs — `wrap_for` zero-extends an unsigned dtype so
        // `3_000_000_000u32` stays positive instead of reading as negative.
        Val::Int(wrap_for(r, op_width(ev.dtype), ev.dtype))
    } else {
        Val::Float(ff(eval(a, ev).f64(), eval(b, ev).f64()))
    }
}

/// `Binary` node: int-only ops (bitwise/shift/logical) route to the integer
/// speller; `Nextafter` is dtype-aware; everything else is the float speller.
fn eval_binary(op: BinaryOp, a: &ScalarExpr, b: &ScalarExpr, ev: &Eval<'_>) -> Val {
    if op.is_int_only() {
        return Val::Int(binary_op_int(
            op,
            eval(a, ev).i128(),
            eval(b, ev).i128(),
            ev.dtype,
        ));
    }
    if matches!(op, BinaryOp::Nextafter) {
        let (x, y) = (eval(a, ev).f64(), eval(b, ev).f64());
        return Val::Float(match ev.dtype {
            // F32Strict stores/steps the f32 lattice too (the emitter lowers it
            // through binary_f32 → nextafterf); the f64 arm would demote-round
            // back to a silent no-op.
            ElementKind::F32 | ElementKind::F32Strict => {
                f64::from(next_after_f32(x as f32, y as f32))
            }
            _ => next_after_f64(x, y),
        });
    }
    let (x, y) = (eval(a, ev).f64(), eval(b, ev).f64());
    // The Cmp* family is DISCONTINUOUS (yields exactly 1.0/0.0): the emitter rounds
    // BOTH operands to the compute dtype before comparing (`(float)a > (float)b`),
    // so an inexact Const or a compute-dtype-underflowing operand decides the same
    // way here. Continuous ops (Max/Min/Pow/Atan2/Rem/…) stay accurate-f64: a
    // flipped near-tie there returns a within-tolerance operand.
    let (x, y) = if op.is_cmp() {
        (round_to_compute(x, ev.dtype), round_to_compute(y, ev.dtype))
    } else {
        (x, y)
    };
    Val::Float(binary_op_f64(op, x, y))
}

// ===========================================================================
// Store — encode a Val into the output storage dtype.
// ===========================================================================

/// Encode `v` into `buf` at element index `off` for output dtype `out`. A `Raw`
/// value whose source dtype equals `out` moves VERBATIM (bit-preserving); a float
/// rounds to `out` (RNE for the halves), an integer wraps to `out`'s width.
fn store_val(v: Val, out: ElementKind, buf: &mut [u8], off: usize) {
    let bits = match v {
        Val::Raw(b, dt) if dt == out => b,
        Val::Raw(b, dt) => {
            if is_int(dt) {
                encode_int(raw_to_i128(b, dt), out)
            } else {
                encode_float(raw_to_f64(b, dt), out)
            }
        }
        Val::Float(f) if is_complex(out) => encode_complex(f, 0.0, out),
        Val::Float(f) => encode_float(f, out),
        Val::Int(i) if is_complex(out) => encode_int_as_complex(i, out),
        Val::Int(i) => encode_int(i, out),
        Val::Complex(re, im) => encode_complex(re, im, out),
    };
    write_elem(buf, off, out, bits);
}

/// Encode an `f64` compute value to `out` (RNE halves; U8 narrows a 0/1
/// predicate; I64 widens a count).
fn encode_float(f: f64, out: ElementKind) -> u128 {
    match out {
        ElementKind::F64 => u128::from(f.to_bits()),
        ElementKind::F32 | ElementKind::F32Strict => u128::from((f as f32).to_bits()),
        ElementKind::Fp8E4M3FN => u128::from(f64_to_fp8_e4m3fn_bits(f)),
        ElementKind::Fp8E5M2 => u128::from(f64_to_fp8_e5m2_bits(f)),
        ElementKind::F16 => u128::from(f32_to_f16_bits(f as f32)),
        ElementKind::Bf16 => u128::from(f32_to_bf16_bits(f as f32)),
        ElementKind::U8 | ElementKind::Bool => u128::from(f as u8),
        ElementKind::I64 => (f as i64) as u128,
        ElementKind::I32 => u128::from((f as i32) as u32),
        ElementKind::I8 => u128::from((f as i8) as u8),
        ElementKind::I16 => u128::from((f as i16) as u16),
        ElementKind::U16 => u128::from(f as u16),
        ElementKind::U32 => u128::from(f as u32),
        ElementKind::U64 => u128::from(f as u64),
        ElementKind::I4 | ElementKind::U4 => u128::from((f as i8 as u8) & 0x0f),
        ElementKind::B1 => u128::from(u8::from(f != 0.0)),
        other => panic!("oracle: encode_float to unsupported dtype {other:?}"),
    }
}

/// Encode a wide integer to `out`, truncating (two's-complement wrap) to width.
fn encode_int(i: i128, out: ElementKind) -> u128 {
    match out {
        ElementKind::I32 => u128::from((i as i32) as u32),
        ElementKind::I64 => (i as i64) as u128,
        ElementKind::I8 => u128::from((i as i8) as u8),
        ElementKind::I16 => u128::from((i as i16) as u16),
        ElementKind::U8 | ElementKind::Bool => u128::from(i as u8),
        ElementKind::U16 => u128::from(i as u16),
        ElementKind::U32 => u128::from(i as u32),
        ElementKind::U64 => u128::from(i as u64),
        ElementKind::I4 | ElementKind::U4 => u128::from((i as u8) & 0x0f),
        ElementKind::B1 => u128::from((i as u8) & 1),
        other => panic!("oracle: encode_int to unsupported dtype {other:?}"),
    }
}

// ===========================================================================
// Layout / coordinate math.
// ===========================================================================

/// The permutation input operand `k` is read through, or `None` for an identity
/// read. Mirrors the emitter's `input_perm` from `plan.views` (an upstream
/// layout fact), independently.
fn input_perm<'a>(plan: &'a KernelPlan<'_>, k: usize) -> Option<&'a [u8]> {
    match plan.views.get(k) {
        Some(View::Permute { perm }) => Some(perm.as_slice()),
        _ => None,
    }
}

/// Unravel a linear index over `extents`, LAST-AXIS-FASTEST.
fn unravel(mut lin: i64, extents: &[i64]) -> Vec<i64> {
    let mut c = vec![0i64; extents.len()];
    for d in (0..extents.len()).rev() {
        c[d] = lin % extents[d];
        lin /= extents[d];
    }
    c
}

/// Product of a slice of extents.
fn prod(s: &[i64]) -> i64 {
    s.iter().product()
}

/// Read input operand `i` at coordinate vector `coords`, honoring signed strides
/// (broadcast = stride 0; flipped = negative stride), a `Permute` view, and the
/// buffer's runtime base offset. The independent analog of the emitter's
/// `offset_expr`: `off = base + Σ_d coords[d] * strides[perm[d]]`.
fn read_strided(
    inputs: &[TypedBuffer],
    operands: &[OperandDesc],
    i: usize,
    coords: &[i64],
    perm: Option<&[u8]>,
) -> Val {
    let od = &operands[i];
    let tb = &inputs[i];
    let rank = od.rank as usize;
    let mut off = tb.base_offset;
    for (d, &c) in coords.iter().enumerate().take(rank) {
        let si = perm.map_or(d, |p| p[d] as usize);
        off += c * od.strides[si];
    }
    assert!(
        off >= 0,
        "oracle: negative element offset {off} on operand {i}"
    );
    Val::Raw(read_elem(&tb.bytes, off as usize, od.dtype), od.dtype)
}

/// Read input operand `i` at FLAT element index `idx` (the RowReduce/Scan/Window
/// role-based load — `in_i[idx]`), honoring the base offset.
fn read_flat(inputs: &[TypedBuffer], operands: &[OperandDesc], i: usize, idx: i64) -> Val {
    let od = &operands[i];
    let tb = &inputs[i];
    let off = tb.base_offset + idx;
    assert!(
        off >= 0,
        "oracle: negative flat offset {off} on operand {i}"
    );
    Val::Raw(read_elem(&tb.bytes, off as usize, od.dtype), od.dtype)
}

/// The role-based flat index for RowReduce/Scan/Window operand `i` at `(row, j)`,
/// classified by the SAME split as [`rr_role`]: RowStreamed `base+j`,
/// ColBroadcast `j`, RowScalar `row`.
fn role_index(plan: &KernelPlan<'_>, i: usize, last: u8, base: i64, j: i64, row: i64) -> i64 {
    match rr_role(plan.key.operands[i], last) {
        RrRole::RowStreamed => base + j,
        RrRole::ColBroadcast => j,
        RrRole::RowScalar => row,
    }
}

/// Allocate a dense (row-major) zeroed output buffer for output operand `j`.
fn alloc_output(
    plan: &KernelPlan<'_>,
    operands: &[OperandDesc],
    n_inputs: usize,
    j: usize,
) -> TypedBuffer {
    let od = &operands[n_inputs + j];
    let rank = od.rank as usize;
    let shape: Vec<i64> = od.shape[..rank].to_vec();
    let strides: Vec<i64> = od.strides[..rank].to_vec();
    let odt = plan.out_dtype_of(j);
    let size = prod(&shape).max(0) as usize;
    TypedBuffer::new(odt, shape, strides, vec![0u8; packed_bytes(odt, size)])
}

// ===========================================================================
// The public entry point.
// ===========================================================================

/// Interpret `plan` over `operands` (inputs THEN outputs, `len ==
/// key.n_operands`), `inputs` (the input storage images), and `params` (runtime
/// scalar `Param` values), returning the computed output storage images.
///
/// This is a SEPARATE code path from the CUDA emitter — the whole point of the
/// differential test. See the module docs for the v1 access-pattern scope and
/// the v2 deferrals (Contraction / RowSort / gather-scatter panic here).
///
/// # Panics
/// On a v2-deferred access pattern, an unsupported dtype, or a malformed plan.
#[must_use]
pub fn evaluate(
    plan: &KernelPlan<'_>,
    operands: &[OperandDesc],
    inputs: &[TypedBuffer],
    params: &[f64],
) -> Vec<TypedBuffer> {
    match plan.access {
        Access::Elementwise => eval_elementwise(plan, operands, inputs, params),
        Access::Reduction { .. } => vec![eval_reduction(plan, operands, inputs, params)],
        Access::RowReduce { .. } => vec![eval_row_reduce(plan, operands, inputs, params)],
        Access::Scan { .. } => vec![eval_scan(plan, operands, inputs, params)],
        Access::Window { .. } => vec![eval_window(plan, operands, inputs, params)],
        Access::Im2Col { .. } => vec![eval_im2col(plan, operands, inputs)],
        Access::Contraction { .. } => vec![eval_contraction(plan, operands, inputs, params)],
        Access::RowSort { .. } => eval_row_sort(plan, operands, inputs),
    }
}

// ===========================================================================
// H. RowSort (stable pair-sort, NaN-greatest).

/// Total order on the sort key with **NaN as the maximum**.
///
/// `Access::RowSort`'s doc pins it: *NaN orders GREATEST in both* — so
/// ascending puts NaN last and **descending puts NaN FIRST**. That is the arm a
/// reader expects to be wrong, so it is one comparator used twice rather than
/// two comparators that could drift apart.
///
/// `-0.0` and `0.0` compare Equal, matching the emitter's `<` on floats. The
/// sort is stable, so equal keys keep their original index order and the
/// signed-zero pair is not reordered.
fn sort_key_cmp(a: f64, b: f64) -> core::cmp::Ordering {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => core::cmp::Ordering::Equal,
        (true, false) => core::cmp::Ordering::Greater,
        (false, true) => core::cmp::Ordering::Less,
        (false, false) => a.partial_cmp(&b).expect("neither operand is NaN"),
    }
}

/// Per-row stable sort of the last axis.
///
/// The body is pinned to exactly `Input(0)` by `validate_row_sort`, so there is
/// no pre- or post-map to apply — this permutes, it does not compute. Values
/// move as **raw bits** rather than through `f64`, because a values-sort is a
/// permutation: a payload that round-tripped through `f64` would normalise a
/// signalling NaN and silently differ from what the kernel writes.
fn eval_row_sort(
    plan: &KernelPlan<'_>,
    operands: &[OperandDesc],
    inputs: &[TypedBuffer],
) -> Vec<TypedBuffer> {
    let (order, out_kind) = match plan.access {
        Access::RowSort { order, out, .. } => (order, out),
        _ => unreachable!(),
    };
    let n_in = 1usize;
    let rank = plan.key.rank as usize;
    let in_od = &operands[0];
    let k_in = in_od.shape[rank - 1];
    let n_rows = if k_in > 0 {
        prod(&in_od.shape[..rank]).max(0) / k_in
    } else {
        0
    };

    let n_outputs = usize::from(matches!(out_kind, SortOut::Both)) + 1;
    let mut outs: Vec<TypedBuffer> = (0..n_outputs)
        .map(|j| alloc_output(plan, operands, n_in, j))
        .collect();

    for row in 0..n_rows {
        let base = row * k_in;
        let keys: Vec<f64> = (0..k_in)
            .map(|j| read_flat(inputs, operands, 0, base + j).f64())
            .collect();
        // `sort_by` is stable, and reversing the comparator does not change
        // which pairs compare Equal — so ties keep ascending original index in
        // BOTH orders, which is what the emitter's (key, index) pair-sort does.
        let mut perm: Vec<usize> = (0..k_in as usize).collect();
        perm.sort_by(|&a, &b| {
            let c = sort_key_cmp(keys[a], keys[b]);
            match order {
                SortOrder::Asc => c,
                SortOrder::Desc => c.reverse(),
            }
        });

        for (j, ob) in outs.iter_mut().enumerate() {
            // The output width may be SHORTER than the input for a TopK cap;
            // `validate_row_sort` allows exactly that and nothing else, so the
            // cap is read off the operand rather than from `SortLimit`.
            let k_out = operands[n_in + j].shape[rank - 1];
            let take = k_out.min(k_in).max(0) as usize;
            let obase = (row * k_out) as usize;
            let writes_indices =
                matches!((out_kind, j), (SortOut::Indices, 0) | (SortOut::Both, 1));
            for (t, &src) in perm.iter().take(take).enumerate() {
                if writes_indices {
                    store_val(
                        Val::Int(src as i128),
                        ElementKind::I32,
                        &mut ob.bytes,
                        obase + t,
                    );
                } else {
                    let bits = read_elem(
                        &inputs[0].bytes,
                        (inputs[0].base_offset + base + src as i64) as usize,
                        in_od.dtype,
                    );
                    let odt = plan.out_dtype_of(j);
                    write_elem(&mut ob.bytes, obase + t, odt, bits);
                }
            }
        }
    }
    outs
}

// ===========================================================================
// A. Elementwise (+ multi-output / hetero).
// ===========================================================================

fn eval_elementwise(
    plan: &KernelPlan<'_>,
    operands: &[OperandDesc],
    inputs: &[TypedBuffer],
    params: &[f64],
) -> Vec<TypedBuffer> {
    let n_in = plan.n_inputs as usize;
    let rank = plan.key.rank as usize;
    let n_outputs = plan.n_outputs as usize;
    let out0 = &operands[n_in];

    let bodies = plan.output_bodies();
    let mut outs: Vec<TypedBuffer> = (0..n_outputs)
        .map(|j| alloc_output(plan, operands, n_in, j))
        .collect();

    // A GATHER is `Access::Elementwise` with one data-dependent read address —
    // the plan gate admits it nowhere else — so it belongs in this walk rather
    // than beside it. `None` for every ordinary elementwise op, and then every
    // branch below is inert.
    let gather = crate::plan::gather_of(plan.read_index);

    // A SCATTER is the mirror: `Access::Elementwise`, single output, one
    // data-dependent WRITE address. Its OOB policy is pinned to `Skip` by the
    // gate ("bespoke scatter/scatter_add/index_add/bincount all skip an OOB
    // target"), so there is no ZeroFill/Clamp branch to write here.
    //
    // **Two semantics worth stating rather than leaving to the reader:**
    //
    // 1. The destination starts ZEROED (`alloc_output`). That is right for
    //    `AtomicAdd` — scatter-add accumulates into zero — and it is a real
    //    assumption for `AtomicMax`/`Min`, where an all-negative source would
    //    max against the 0 that was already there. A kernel whose caller
    //    pre-initialises the destination differently will differ, and correctly.
    // 2. `Assign` with DUPLICATE indices is last-writer-wins with no defined
    //    order on device. This walks the iteration space in order, so it
    //    produces ONE valid outcome, not THE outcome. Compare against it only on
    //    a unique-index corpus; the other combines are order-independent and do
    //    not carry that caveat.
    let scatter = crate::plan::scatter_of(plan.write_index);

    // ⚠️ THE ITERATION SPACE IS THE SOURCE'S FOR A SCATTER, not the output's.
    //
    // Every other elementwise op produces one output element per iteration, so
    // walking the output shape is the same walk. **A scatter does not**: its
    // destination can be smaller than its source (bincount is the extreme —
    // thousands of elements into a handful of bins), and walking the output
    // would visit the bins and silently drop every source element past the
    // first few. The value operand is the input that is not the index.
    let src_slot = scatter.map(|(idx_op, ..)| usize::from(idx_op == 0));
    let ext: Vec<i64> = match src_slot {
        Some(v) => operands[v].shape[..rank].to_vec(),
        None => out0.shape[..rank].to_vec(),
    };
    let n_out = prod(&ext).max(0);

    for lin in 0..n_out {
        let coords = unravel(lin, &ext);

        // Resolve the index ONCE per output element, before any body runs.
        //
        // **Out-of-range is a STORE predicate, not a load behaviour** — the
        // policy docs are explicit that no OOB load occurs, because the emitter
        // clamps the load address in-bounds and guards the store. So the address
        // is always clamped here, and the OOB decision changes what is written:
        // `Skip` leaves the cell alone, `ZeroFill` writes zero, `Clamp` stores
        // the clamped read as an ordinary value. Reading OOB and then discarding
        // would agree on every output byte and disagree on whether the load
        // happened, which is exactly the difference a sanitizer sees.
        let mut gcoords = coords.clone();
        let mut oob_zero = false;
        if let Some((g, idx_op, axis, oob, _)) = gather {
            let axis = axis as usize;
            let extent = operands[g].shape[axis];
            let raw = i64::try_from(
                read_strided(
                    inputs,
                    operands,
                    idx_op as usize,
                    &coords,
                    input_perm(plan, idx_op as usize),
                )
                .i128(),
            )
            .unwrap_or(i64::MIN);
            gcoords[axis] = raw.clamp(0, (extent - 1).max(0));
            if raw < 0 || raw >= extent {
                match oob {
                    crate::ir::OobPolicy::Skip => continue,
                    crate::ir::OobPolicy::ZeroFill => oob_zero = true,
                    crate::ir::OobPolicy::Clamp => {}
                }
            }
        }

        let leaf = |i: u8| {
            let c = if gather.is_some_and(|(g, ..)| g == i as usize) {
                &gcoords
            } else {
                &coords
            };
            read_strided(
                inputs,
                operands,
                i as usize,
                c,
                input_perm(plan, i as usize),
            )
        };
        let coord = |d: u8| Val::Float(coords[d as usize] as f64);
        let ev = Eval {
            dtype: plan.dtype,
            params,
            leaf: &leaf,
            reduced: &panic_reduced,
            coord: &coord,
        };
        // Evaluate all output bodies first (they share the input reads), then store.
        let vals: Vec<Val> = (0..n_outputs)
            .map(|j| {
                if oob_zero {
                    // The op's zero fill. `Val::Int(0)` rather than
                    // `Val::Float(0.0)` so an integer output stores an exact 0
                    // rather than a float that has to round back.
                    Val::Int(0)
                } else {
                    eval(bodies[j], &ev)
                }
            })
            .collect();
        // Scatter redirects the STORE address along one axis. Resolved after the
        // bodies run, because an OOB target guards the store rather than the
        // read — the same shape as gather's policy, mirrored.
        let mut scoords = coords.clone();
        let mut combine = None;
        if let Some((idx_op, axis, cmb, _oob, _)) = scatter {
            let axis = axis as usize;
            let extent = operands[n_in].shape[axis];
            let raw = i64::try_from(
                read_strided(
                    inputs,
                    operands,
                    idx_op as usize,
                    &coords,
                    input_perm(plan, idx_op as usize),
                )
                .i128(),
            )
            .unwrap_or(i64::MIN);
            // v1 pins the policy to Skip, so an out-of-range target is dropped.
            if raw < 0 || raw >= extent {
                continue;
            }
            scoords[axis] = raw;
            combine = Some(cmb);
        }

        for (j, val) in vals.into_iter().enumerate() {
            let od = &operands[n_in + j];
            let dst = if scatter.is_some() { &scoords } else { &coords };
            let off: i64 = dst
                .iter()
                .enumerate()
                .take(rank)
                .map(|(d, &c)| c * od.strides[d])
                .sum();
            let odt = plan.out_dtype_of(j);
            let val = match combine {
                None | Some(crate::ir::WriteCombine::Assign) => val,
                Some(cmb) => {
                    // Read-modify-write. Serial here, and that is exact for
                    // these three: integer add is associative and commutative,
                    // and max/min are too — the gate admits max/min for integers
                    // only, which is also what keeps `max(-0.0, +0.0)` from ever
                    // arising. Float `AtomicAdd` is order-dependent in general
                    // and is exact only on a corpus of integer-valued floats
                    // inside 2^24 (f32) / 2^53 (f64); see docs/deferred.md.
                    let prev = Val::Raw(read_elem(&outs[j].bytes, off as usize, odt), odt);
                    match cmb {
                        crate::ir::WriteCombine::AtomicAdd => {
                            if is_int(odt) {
                                Val::Int(prev.i128() + val.i128())
                            } else {
                                Val::Float(prev.f64() + val.f64())
                            }
                        }
                        crate::ir::WriteCombine::AtomicMax => Val::Int(prev.i128().max(val.i128())),
                        crate::ir::WriteCombine::AtomicMin => Val::Int(prev.i128().min(val.i128())),
                        crate::ir::WriteCombine::Assign => unreachable!("handled above"),
                    }
                }
            };
            store_val(val, odt, &mut outs[j].bytes, off as usize);
        }
    }
    outs
}

// ===========================================================================
// B. Reduction (fold over the reduced axes; post epilogue).
// ===========================================================================

fn eval_reduction(
    plan: &KernelPlan<'_>,
    operands: &[OperandDesc],
    inputs: &[TypedBuffer],
    params: &[f64],
) -> TypedBuffer {
    let (rop, axes, keepdim, post) = match plan.access {
        Access::Reduction {
            op,
            axes,
            keepdim,
            post,
            accum: _,
        } => (*op, *axes, *keepdim, post),
        _ => unreachable!(),
    };
    let n_in = plan.n_inputs as usize;
    let rank = plan.key.rank as usize;
    let in0 = &operands[0];

    // Reduced set: EMPTY mask ⇒ the legacy last-axis default.
    let reduced: Vec<usize> = if axes.is_empty() {
        vec![rank - 1]
    } else {
        (0..rank).filter(|&d| axes.is_set(d as u8)).collect()
    };
    let kept: Vec<usize> = (0..rank).filter(|&d| !reduced.contains(&d)).collect();
    let kept_ext: Vec<i64> = kept.iter().map(|&a| in0.shape[a]).collect();
    let red_ext: Vec<i64> = reduced.iter().map(|&a| in0.shape[a]).collect();
    let divisor = prod(&red_ext) as f64;
    let n_kept = prod(&kept_ext).max(0);
    let n_red = prod(&red_ext).max(0);

    let mut out = alloc_output(plan, operands, n_in, 0);
    let out_dt = plan.out_dtype_of(0);
    let int_acc = is_int(plan.dtype);
    // Mirror the emitter's reduction scope (the integer-reduction scope contract:
    // I32/I64 only, no integer Mean) instead of silently
    // mis-computing: integer reductions are I32/I64 only, and integer Mean is out
    // of scope (the emitter asserts `!(int_acc && Mean)`). Keep the oracle's
    // out-of-scope-PANIC convention (cf. eval_scan/eval_window/eval_row_reduce)
    // rather than returning the un-divided sum for a buildable I32/S8/U8 Mean plan.
    if int_acc {
        assert!(
            matches!(plan.dtype, ElementKind::I32 | ElementKind::I64),
            "oracle: integer reduction dtype {:?} out of scope (emitter: I32/I64 only)",
            plan.dtype
        );
        assert!(
            !matches!(rop, ReduceOp::Mean),
            "oracle: integer Mean reduction out of scope (emitter rejects int_acc && Mean)"
        );
    }

    for klin in 0..n_kept {
        let kc = unravel(klin, &kept_ext);
        // Fold over the reduced-axis cartesian product.
        let mut acc_f = if matches!(rop, ReduceOp::Prod) {
            1.0
        } else {
            0.0
        };
        let mut acc_i: i128 = if matches!(rop, ReduceOp::Prod) { 1 } else { 0 };
        let mut have = false;
        for rlin in 0..n_red {
            let rc = unravel(rlin, &red_ext);
            let mut coords = vec![0i64; rank];
            for (j, &a) in kept.iter().enumerate() {
                coords[a] = kc[j];
            }
            for (j, &a) in reduced.iter().enumerate() {
                coords[a] = rc[j];
            }
            let leaf = |i: u8| {
                read_strided(
                    inputs,
                    operands,
                    i as usize,
                    &coords,
                    input_perm(plan, i as usize),
                )
            };
            let ev = Eval {
                dtype: plan.dtype,
                params,
                leaf: &leaf,
                reduced: &panic_reduced,
                coord: &panic_coord,
            };
            let e = eval(plan.body, &ev);
            fold_step(rop, e, int_acc, &mut acc_f, &mut acc_i, &mut have);
        }
        // Finalize (Mean divides by the reduced-extent product; k==0 ⇒ 0/0 = NaN).
        let finalized: Val = if int_acc {
            Val::Int(acc_i)
        } else if matches!(rop, ReduceOp::Mean) {
            Val::Float(acc_f / divisor)
        } else {
            Val::Float(acc_f)
        };
        // Apply the post epilogue (default identity Reduced(0)).
        let reduced_leaf = |s: u8| {
            assert_eq!(s, 0, "reduction post reads only Reduced(0)");
            finalized
        };
        let ev = Eval {
            dtype: plan.dtype,
            params,
            leaf: &(|i: u8| panic!("reduction post reads no Input({i})")),
            reduced: &reduced_leaf,
            coord: &panic_coord,
        };
        let posted = eval(post, &ev);
        // Output offset: keepdim ⇒ per-input-axis; collapse ⇒ per-kept-position.
        let off: i64 = if keepdim {
            kept.iter()
                .enumerate()
                .map(|(j, &a)| kc[j] * out.strides[a])
                .sum()
        } else {
            kept.iter()
                .enumerate()
                .map(|(j, _)| kc[j] * out.strides[j])
                .sum()
        };
        store_val(posted, out_dt, &mut out.bytes, off as usize);
    }
    out
}

// ===========================================================================
// B'. Contraction (matmul).
// ===========================================================================

/// Evaluate a rank-2 dense contraction `[m,k]·[k,n] → [m,n]` (the matmul axes the
/// plan gate has already asserted). Independent of the CUDA skinny-SIMT emitter:
/// a plain triple loop with the SAME accumulation ORDER (K ascending) and the
/// SAME `Reduced(0)` epilogue over the per-`(m,n)` K-sum.
///
/// Accumulates the K-sum in `f64` — the accurate reference. Like the reductions,
/// this oracle is a tolerance referee, not a bit-for-bit device mirror: the
/// emitter's `f32`/`double` accumulator rounding is absorbed by the tolerant
/// comparator (`Fidelity::Tolerant`), while an exactly-representable cell (small
/// integers) matches bit-for-bit.
///
/// v1 scope mirrors the emitter and `derive_contraction`: each operand is read
/// by its PHYSICAL strides at the logical (role) coordinate, so transposed/
/// permuted (`lhs_order`/`rhs_order` non-identity), batched, and broadcast-batch
/// operands all index correctly — the layout lives in the strides, not in a
/// second permutation applied on top of them (see [`read_strided`] and the
/// module's independence discipline). The epilogue reads only `Reduced(0)` (no
/// `Input`/`Coord`), plus the optional fused per-column bias.
fn eval_contraction(
    plan: &KernelPlan<'_>,
    operands: &[OperandDesc],
    inputs: &[TypedBuffer],
    params: &[f64],
) -> TypedBuffer {
    let epilogue = match plan.access {
        Access::Contraction { epilogue, .. } => epilogue,
        _ => unreachable!("eval_contraction requires Access::Contraction"),
    };
    let n_in = plan.n_inputs as usize;
    let lhs = &operands[0];
    let rhs = &operands[1];
    // rank-3 lhs ⇒ batched `[B,M,K]·[B,K,N]`; rank-2 ⇒ plain (bdim = 1).
    let batched = lhs.rank == 3;
    let (bdim, m, k, n) = if batched {
        (lhs.shape[0], lhs.shape[1], lhs.shape[2], rhs.shape[2])
    } else {
        (1, lhs.shape[0], lhs.shape[1], rhs.shape[1])
    };
    let rhs_k = if batched { rhs.shape[1] } else { rhs.shape[0] };
    assert_eq!(
        rhs_k, k,
        "oracle: contraction K mismatch (lhs cols {k} vs rhs rows {rhs_k})"
    );

    let mut out = alloc_output(plan, operands, n_in, 0);
    let out_dt = plan.out_dtype_of(0);

    for bi in 0..bdim {
        for mi in 0..m {
            for ni in 0..n {
                // K-sum, `kk` ascending — the emitter's inner loop order.
                let mut acc = 0.0f64;
                for kk in 0..k {
                    let (lc, rc): (Vec<i64>, Vec<i64>) = if batched {
                        (vec![bi, mi, kk], vec![bi, kk, ni])
                    } else {
                        (vec![mi, kk], vec![kk, ni])
                    };
                    let a = read_strided(inputs, operands, 0, &lc, None).f64();
                    let b = read_strided(inputs, operands, 1, &rc, None).f64();
                    acc += a * b;
                }
                // Epilogue over Reduced(0) = the accumulator, plus the optional
                // fused bias: Input(i>=2) is the per-column `[N]` bias, read at
                // column `ni` (broadcast over rows; batched v1 carries no bias). No
                // Coord leaves. lhs/rhs (0/1) never appear.
                let reduced_leaf = |s: u8| {
                    assert_eq!(s, 0, "contraction epilogue reads only Reduced(0)");
                    Val::Float(acc)
                };
                let bias_leaf = |i: u8| {
                    assert!(
                        i as usize >= 2 && (i as usize) < n_in,
                        "contraction epilogue Input({i}) must be a fused bias (2..n_inputs)"
                    );
                    read_strided(
                        inputs,
                        operands,
                        i as usize,
                        &[ni],
                        input_perm(plan, i as usize),
                    )
                };
                let ev = Eval {
                    dtype: plan.dtype,
                    params,
                    leaf: &bias_leaf,
                    reduced: &reduced_leaf,
                    coord: &panic_coord,
                };
                let posted = eval(epilogue, &ev);
                let off = if batched {
                    bi * out.strides[0] + mi * out.strides[1] + ni * out.strides[2]
                } else {
                    mi * out.strides[0] + ni * out.strides[1]
                };
                store_val(posted, out_dt, &mut out.bytes, off as usize);
            }
        }
    }
    out
}

/// One fold step for the have-flag NaN-propagating reduction combines.
fn fold_step(
    rop: ReduceOp,
    e: Val,
    int_acc: bool,
    acc_f: &mut f64,
    acc_i: &mut i128,
    have: &mut bool,
) {
    match rop {
        ReduceOp::Sum | ReduceOp::Mean => {
            if int_acc {
                *acc_i = wrap_bits(acc_i.wrapping_add(e.i128()), 64);
            } else {
                *acc_f += e.f64();
            }
        }
        ReduceOp::Prod => {
            if int_acc {
                *acc_i = wrap_bits(acc_i.wrapping_mul(e.i128()), 64);
            } else {
                *acc_f *= e.f64();
            }
        }
        ReduceOp::Max | ReduceOp::Min => {
            if int_acc {
                let v = e.i128();
                let better = if matches!(rop, ReduceOp::Max) {
                    v > *acc_i
                } else {
                    v < *acc_i
                };
                if !*have || better {
                    *acc_i = v;
                }
            } else {
                let v = e.f64();
                let better = if matches!(rop, ReduceOp::Max) {
                    v > *acc_f
                } else {
                    v < *acc_f
                };
                if !*have || v.is_nan() || better {
                    *acc_f = v;
                }
            }
            *have = true;
        }
    }
}

// ===========================================================================
// C. RowReduce (per-row staged reductions, then a full-width epilogue).
// ===========================================================================

fn eval_row_reduce(
    plan: &KernelPlan<'_>,
    operands: &[OperandDesc],
    inputs: &[TypedBuffer],
    params: &[f64],
) -> TypedBuffer {
    let (stages, epilogue) = match plan.access {
        Access::RowReduce { stages, epilogue } => (stages, epilogue),
        _ => unreachable!(),
    };
    let n_in = plan.n_inputs as usize;
    let rank = plan.key.rank as usize;
    let last = (rank - 1) as u8;
    let out = &operands[n_in];
    let k = out.shape[rank - 1];
    let total = prod(&out.shape[..rank]).max(0);
    let n_rows = if k > 0 { total / k } else { 0 };

    let mut result = alloc_output(plan, operands, n_in, 0);
    let out_dt = plan.out_dtype_of(0);

    for row in 0..n_rows {
        let base = row * k;
        // Run each stage in order → Reduced(0..n_stages).
        let mut reduced_vals: Vec<Val> = Vec::with_capacity(stages.len());
        for st in stages {
            let mut acc = 0.0f64;
            let mut have = false;
            for j in 0..k {
                let leaf = |i: u8| {
                    read_flat(
                        inputs,
                        operands,
                        i as usize,
                        role_index(plan, i as usize, last, base, j, row),
                    )
                };
                let reduced_leaf = |s: u8| reduced_vals[s as usize];
                let ev = Eval {
                    dtype: plan.dtype,
                    params,
                    leaf: &leaf,
                    reduced: &reduced_leaf,
                    coord: &panic_coord,
                };
                let e = eval(&st.pre, &ev).f64();
                match st.op {
                    ReduceOp::Sum | ReduceOp::Mean => acc += e,
                    ReduceOp::Max => {
                        if !have || e.is_nan() || e > acc {
                            acc = e;
                        }
                        have = true;
                    }
                    ReduceOp::Min => {
                        if !have || e.is_nan() || e < acc {
                            acc = e;
                        }
                        have = true;
                    }
                    ReduceOp::Prod => panic!("oracle: RowReduce Prod stage is unsupported"),
                }
            }
            let finalized = if matches!(st.op, ReduceOp::Mean) {
                acc / k as f64
            } else {
                acc
            };
            reduced_vals.push(Val::Float(finalized));
        }
        // Epilogue: full-width output, Reduced(s) = the finalized stage folds.
        for j in 0..k {
            let leaf = |i: u8| {
                read_flat(
                    inputs,
                    operands,
                    i as usize,
                    role_index(plan, i as usize, last, base, j, row),
                )
            };
            let reduced_leaf = |s: u8| reduced_vals[s as usize];
            let ev = Eval {
                dtype: plan.dtype,
                params,
                leaf: &leaf,
                reduced: &reduced_leaf,
                coord: &panic_coord,
            };
            let v = eval(epilogue, &ev);
            store_val(v, out_dt, &mut result.bytes, (base + j) as usize);
        }
    }
    result
}

// ===========================================================================
// D. Scan (serial prefix fold — the bit-reference).
// ===========================================================================

fn eval_scan(
    plan: &KernelPlan<'_>,
    operands: &[OperandDesc],
    inputs: &[TypedBuffer],
    params: &[f64],
) -> TypedBuffer {
    let (pre, post) = match plan.access {
        Access::Scan { pre, post, .. } => (pre, post),
        _ => unreachable!(),
    };
    let (sop, _axis, reverse, exclusive) = match plan.schedule {
        Schedule::Scan {
            op,
            axis,
            reverse,
            exclusive,
            ..
        } => (op, axis, reverse, exclusive),
        _ => unreachable!(),
    };
    let n_in = plan.n_inputs as usize;
    let rank = plan.key.rank as usize;
    let last = (rank - 1) as u8;
    let out = &operands[n_in];
    let k = out.shape[rank - 1];
    let total = prod(&out.shape[..rank]).max(0);
    let n_rows = if k > 0 { total / k } else { 0 };
    let int_acc = is_int(plan.dtype);

    let mut result = alloc_output(plan, operands, n_in, 0);
    let out_dt = plan.out_dtype_of(0);

    // Monoid identity in each domain.
    let (ident_f, ident_i) = scan_identity(sop, plan.dtype);

    for row in 0..n_rows {
        let base = row * k;
        let mut acc_f = ident_f;
        let mut acc_i = ident_i;
        let mut have = false;
        for jj in 0..k {
            let j = if reverse { k - 1 - jj } else { jj };
            let leaf = |i: u8| {
                read_flat(
                    inputs,
                    operands,
                    i as usize,
                    role_index(plan, i as usize, last, base, j, row),
                )
            };
            // pre has no running prefix.
            let ev_pre = Eval {
                dtype: plan.dtype,
                params,
                leaf: &leaf,
                reduced: &panic_reduced,
                coord: &panic_coord,
            };
            let v = eval(pre, &ev_pre);

            // The running prefix bound as Reduced(0) in `post`.
            let write = |prefix: Val, result: &mut TypedBuffer| {
                let reduced_leaf = |s: u8| {
                    assert_eq!(s, 0, "scan post reads only Reduced(0)");
                    prefix
                };
                let ev = Eval {
                    dtype: plan.dtype,
                    params,
                    leaf: &leaf,
                    reduced: &reduced_leaf,
                    coord: &panic_coord,
                };
                let out_val = eval(post, &ev);
                store_val(out_val, out_dt, &mut result.bytes, (base + j) as usize);
            };

            match sop {
                ReduceOp::Sum | ReduceOp::Prod => {
                    let combine = |a: Val, b: Val| -> Val {
                        if int_acc {
                            let r = match sop {
                                ReduceOp::Sum => a.i128().wrapping_add(b.i128()),
                                _ => a.i128().wrapping_mul(b.i128()),
                            };
                            Val::Int(wrap_bits(r, op_width(plan.dtype)))
                        } else {
                            let r = match sop {
                                ReduceOp::Sum => a.f64() + b.f64(),
                                _ => a.f64() * b.f64(),
                            };
                            Val::Float(r)
                        }
                    };
                    let acc = if int_acc {
                        Val::Int(acc_i)
                    } else {
                        Val::Float(acc_f)
                    };
                    if exclusive {
                        write(acc, &mut result);
                        let n = combine(acc, v);
                        set_acc(n, int_acc, &mut acc_f, &mut acc_i);
                    } else {
                        let n = combine(acc, v);
                        set_acc(n, int_acc, &mut acc_f, &mut acc_i);
                        let acc2 = if int_acc {
                            Val::Int(acc_i)
                        } else {
                            Val::Float(acc_f)
                        };
                        write(acc2, &mut result);
                    }
                }
                ReduceOp::Max | ReduceOp::Min => {
                    let ident = if int_acc {
                        Val::Int(ident_i)
                    } else {
                        Val::Float(ident_f)
                    };
                    let cur = if int_acc {
                        Val::Int(acc_i)
                    } else {
                        Val::Float(acc_f)
                    };
                    let update = |acc_f: &mut f64, acc_i: &mut i128, have: &mut bool| {
                        if int_acc {
                            let x = v.i128();
                            let better = if matches!(sop, ReduceOp::Max) {
                                x > *acc_i
                            } else {
                                x < *acc_i
                            };
                            if !*have || better {
                                *acc_i = x;
                            }
                        } else {
                            let x = v.f64();
                            let better = if matches!(sop, ReduceOp::Max) {
                                x > *acc_f
                            } else {
                                x < *acc_f
                            };
                            if !*have || x.is_nan() || better {
                                *acc_f = x;
                            }
                        }
                        *have = true;
                    };
                    if exclusive {
                        let prefix = if have { cur } else { ident };
                        write(prefix, &mut result);
                        update(&mut acc_f, &mut acc_i, &mut have);
                    } else {
                        update(&mut acc_f, &mut acc_i, &mut have);
                        let acc2 = if int_acc {
                            Val::Int(acc_i)
                        } else {
                            Val::Float(acc_f)
                        };
                        write(acc2, &mut result);
                    }
                }
                ReduceOp::Mean => panic!("oracle: Scan rejects Mean"),
            }
        }
    }
    result
}

/// Store a combined Val back into the active accumulator domain.
fn set_acc(v: Val, int_acc: bool, acc_f: &mut f64, acc_i: &mut i128) {
    if int_acc {
        *acc_i = v.i128();
    } else {
        *acc_f = v.f64();
    }
}

/// The scan monoid identity in (float, int) domains. Max/Min identities are the
/// type extremes (the int extreme is only meaningful — and only computed — for an
/// integer dtype; the float slot carries `±inf`).
fn scan_identity(sop: ReduceOp, dt: ElementKind) -> (f64, i128) {
    let ext = |neg: bool| if is_int(dt) { int_extreme(dt, neg) } else { 0 };
    match sop {
        ReduceOp::Sum => (0.0, 0),
        ReduceOp::Prod => (1.0, 1),
        ReduceOp::Max => (f64::NEG_INFINITY, ext(true)),
        ReduceOp::Min => (f64::INFINITY, ext(false)),
        ReduceOp::Mean => panic!("oracle: Scan rejects Mean"),
    }
}

// ===========================================================================
// E. Window (sliding-window pool; count_include_pad divisor).
// ===========================================================================

#[allow(clippy::too_many_lines)]
fn eval_window(
    plan: &KernelPlan<'_>,
    operands: &[OperandDesc],
    inputs: &[TypedBuffer],
    params: &[f64],
) -> TypedBuffer {
    let (pre, post) = match plan.access {
        Access::Window { pre, post, .. } => (pre, post),
        _ => unreachable!(),
    };
    let (wop, _axis, size, stride, dilation, pad_lo, _pad_hi, count_include_pad) =
        match plan.schedule {
            Schedule::Window {
                op,
                axis,
                size,
                stride,
                dilation,
                pad_lo,
                pad_hi,
                count_include_pad,
            } => (
                op,
                axis,
                size,
                stride,
                dilation,
                pad_lo,
                pad_hi,
                count_include_pad,
            ),
            _ => unreachable!(),
        };
    let n_in = plan.n_inputs as usize;
    let rank = plan.key.rank as usize;
    let last = (rank - 1) as u8;
    let in0 = &operands[0];
    let out = &operands[n_in];
    let k_in = in0.shape[rank - 1];
    let k_out = out.shape[rank - 1];
    let total_out = prod(&out.shape[..rank]).max(0);
    let n_rows = if k_out > 0 { total_out / k_out } else { 0 };
    let int_acc = is_int(plan.dtype);

    let (size, stride, dil, plo) = (
        i64::from(size),
        i64::from(stride),
        i64::from(dilation),
        i64::from(pad_lo),
    );

    let mut result = alloc_output(plan, operands, n_in, 0);
    let out_dt = plan.out_dtype_of(0);

    for row in 0..n_rows {
        let base = row * k_in;
        for o in 0..k_out {
            // Fold the valid taps.
            let mut acc_f = 0.0f64;
            let mut acc_i: i128 = 0;
            let mut best_f = 0.0f64;
            let mut best_i: i128 = 0;
            let mut cnt = 0i64;
            let mut have = false;
            for kk in 0..size {
                let p = o * stride - plo + kk * dil;
                if p < 0 || p >= k_in {
                    continue;
                }
                let leaf = |i: u8| {
                    read_flat(
                        inputs,
                        operands,
                        i as usize,
                        role_index(plan, i as usize, last, base, p, row),
                    )
                };
                let ev = Eval {
                    dtype: plan.dtype,
                    params,
                    leaf: &leaf,
                    reduced: &panic_reduced,
                    coord: &panic_coord,
                };
                let v = eval(pre, &ev);
                match wop {
                    ReduceOp::Sum | ReduceOp::Mean => {
                        if int_acc {
                            acc_i = wrap_bits(acc_i.wrapping_add(v.i128()), op_width(plan.dtype));
                        } else {
                            acc_f += v.f64();
                        }
                        cnt += 1;
                    }
                    ReduceOp::Max => {
                        if int_acc {
                            let x = v.i128();
                            if !have || x > best_i {
                                best_i = x;
                            }
                        } else {
                            let x = v.f64();
                            if !have || x.is_nan() || x > best_f {
                                best_f = x;
                            }
                        }
                        have = true;
                    }
                    ReduceOp::Min => {
                        if int_acc {
                            let x = v.i128();
                            if !have || x < best_i {
                                best_i = x;
                            }
                        } else {
                            let x = v.f64();
                            if !have || x.is_nan() || x < best_f {
                                best_f = x;
                            }
                        }
                        have = true;
                    }
                    ReduceOp::Prod => panic!("oracle: Window rejects Prod"),
                }
            }
            // Finalize.
            let prefix: Val = match wop {
                ReduceOp::Sum => {
                    if int_acc {
                        Val::Int(acc_i)
                    } else {
                        Val::Float(acc_f)
                    }
                }
                ReduceOp::Mean => {
                    // Float-only (asserted upstream).
                    let div = if count_include_pad {
                        size as f64
                    } else {
                        cnt as f64
                    };
                    Val::Float(if count_include_pad || cnt > 0 {
                        acc_f / div
                    } else {
                        0.0
                    })
                }
                ReduceOp::Max | ReduceOp::Min => {
                    let (ident_f, ident_i) = scan_identity(wop, plan.dtype);
                    if int_acc {
                        Val::Int(if have { best_i } else { ident_i })
                    } else {
                        Val::Float(if have { best_f } else { ident_f })
                    }
                }
                ReduceOp::Prod => unreachable!(),
            };
            // Epilogue.
            let leaf = |i: u8| {
                read_flat(
                    inputs,
                    operands,
                    i as usize,
                    role_index(plan, i as usize, last, base, o, row),
                )
            };
            let reduced_leaf = |s: u8| {
                assert_eq!(s, 0, "window post reads only Reduced(0)");
                prefix
            };
            let ev = Eval {
                dtype: plan.dtype,
                params,
                leaf: &leaf,
                reduced: &reduced_leaf,
                coord: &panic_coord,
            };
            let v = eval(post, &ev);
            store_val(v, out_dt, &mut result.bytes, (row * k_out + o) as usize);
        }
    }
    result
}

// ===========================================================================
// G. Im2Col (2-D expanding gather — BIT-EXACT raw copy).
// ===========================================================================

fn eval_im2col(
    plan: &KernelPlan<'_>,
    operands: &[OperandDesc],
    inputs: &[TypedBuffer],
) -> TypedBuffer {
    let (kernel, stride, pad, dilation) = match plan.schedule {
        Schedule::Im2Col {
            kernel,
            stride,
            pad,
            dilation,
        } => (kernel, stride, pad, dilation),
        _ => unreachable!(),
    };
    let n_in = plan.n_inputs as usize;
    let in0 = &operands[0];
    let (n, c_in, h_in, w_in) = (in0.shape[0], in0.shape[1], in0.shape[2], in0.shape[3]);
    let (kh, kw) = (i64::from(kernel.0), i64::from(kernel.1));
    let (sh, sw) = (i64::from(stride.0), i64::from(stride.1));
    let (ph, pw) = (i64::from(pad.0), i64::from(pad.1));
    let (dh, dw) = (i64::from(dilation.0), i64::from(dilation.1));
    // Conv output spatial extents (the key carries no extents — derive here).
    let o_h = (h_in + 2 * ph - dh * (kh - 1) - 1) / sh + 1;
    let o_w = (w_in + 2 * pw - dw * (kw - 1) - 1) / sw + 1;

    let dt = in0.dtype;
    let sz = elem_size(dt);
    let total = n * c_in * kh * kw * o_h * o_w;

    let mut out = alloc_output(plan, operands, n_in, 0);
    let in_bytes = &inputs[0].bytes;
    let in_base = inputs[0].base_offset;

    for t in 0..total {
        let col = t % (o_h * o_w);
        let row_full = t / (o_h * o_w);
        let oh = col / o_w;
        let ow = col - oh * o_w;
        let patch = row_full % (c_in * kh * kw);
        let nn = row_full / (c_in * kh * kw);
        let cc = patch / (kh * kw);
        let kij = patch - cc * (kh * kw);
        let ki = kij / kw;
        let kj = kij - ki * kw;
        let in_h = oh * sh - ph + ki * dh;
        let in_w = ow * sw - pw + kj * dw;
        let dst = t as usize * sz;
        if in_h >= 0 && in_h < h_in && in_w >= 0 && in_w < w_in {
            let src = (((nn * c_in + cc) * h_in + in_h) * w_in + in_w) + in_base;
            let src_b = src as usize * sz;
            out.bytes[dst..dst + sz].copy_from_slice(&in_bytes[src_b..src_b + sz]);
        }
        // else: typed zero == zeroed bytes (already zero-initialized).
    }
    out
}

// ===========================================================================
// compare — the bit-exact-vs-tolerance dichotomy.
// ===========================================================================

/// How two output buffers are compared: bit-exact (raw storage memcmp) or f64-
/// accurate within a rel+abs tolerance band.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Fidelity {
    /// Raw-byte equality (identity bodies, Im2Col, permutation/movement, Select
    /// arm moves, integer arithmetic/bitwise/shift — exact-wrapping).
    BitExact,
    /// Float arithmetic within `abs + rel·max(|a|,|b|)`; both-NaN payload-agnostic
    /// equal; ±0 equal.
    Tolerant {
        /// Relative tolerance.
        rel: f64,
        /// Absolute tolerance.
        abs: f64,
    },
}

/// What comparison band a cell is **entitled to** — derived from the plan and
/// its operands, not chosen by hand at the call site.
///
/// # The gap this closes
///
/// Every caller of [`compare`] picked a [`Fidelity`] by hand. Nothing said what
/// a given cell was owed, so the choice was a judgement re-made per test, with
/// no way to tell a tolerance that reflects the kernel's real error from one
/// that was widened until the test passed. Those look identical in a diff.
///
/// # Derived from the SAME number the contract declares
///
/// The approximate-op term comes from [`crate::contract::ulp_bound`] — the
/// function that feeds the emitted contract's `max_ulp`. Two independently
/// chosen accuracy figures would drift, silently and in both directions: a
/// validator looser than the contract passes kernels the contract promises are
/// tighter, and a tighter one fails kernels that honour it.
///
/// # What the band is made of
///
/// `rel = (steps + 2·ulp_bound) · u`, where `u` is the compute dtype's unit
/// roundoff (`2⁻²⁴` for `f32`, `2⁻⁵³` for `f64`).
///
/// * **`steps`** — the roundings the DEVICE performs that the oracle does not.
///   The oracle accumulates in `f64` on purpose ("a tolerance referee, not a
///   bit-for-bit device mirror"), so each arithmetic node in the body, plus the
///   reduction length where the cell reduces, is one rounding the reference
///   skipped. Each contributes at most `u` relative error; summing them is the
///   standard forward bound `n·u/(1−n·u) ≈ n·u`.
/// * **`2·ulp_bound`** — the vendor-approximate ops' declared error. One ULP is
///   `2u` relative at the top of a binade, so a `k`-ULP op contributes `2k·u`.
///
/// # `abs` is zero, and that is a stated limit rather than an omission
///
/// A relative band collapses at zero, so a body whose result **cancels** to near
/// zero from larger intermediates needs an absolute floor — and the size of that
/// floor depends on the intermediate magnitudes, which a plan does not know and
/// cannot know without the input values. Returning a fabricated `abs` would be a
/// number with no derivation behind it, which is worse than none: it would look
/// like a bound. Callers whose cells cancel must widen `abs` themselves and say
/// why.
///
/// # Returns `None` when no bound is derivable
///
/// A body containing floored `Rem` has no finite result-ULP bound — its
/// quotient-boundary flip is a discontinuity, not an error term, and
/// [`crate::contract::precision_of`] correctly declares `approximate` with **no**
/// `max_ulp` rather than an understated number. `None` says the same thing here:
/// the caller must decide, and cannot be handed a bound that does not exist.
#[must_use]
pub fn required_fidelity(plan: &KernelPlan<'_>, operands: &[OperandDesc]) -> Option<Fidelity> {
    // Integer, bool and sub-byte cells are EXACT on both sides: the emitter's
    // arithmetic wraps at the C width and the oracle models that wrapping in
    // `i128`, so there is no rounding for a tolerance to absorb. Bit-exact is
    // not a tight choice here, it is the only correct one — a tolerant compare
    // would accept a genuinely wrong integer.
    if is_int(plan.dtype) && is_int(plan.out_dtype) {
        return Some(Fidelity::BitExact);
    }

    // KNOWN LIMIT, stated at the use site because it is invisible from the
    // signature: `ulp_bound` sums a **CUDA** per-op ULP table, and neither it nor
    // this function takes a target. So the band computed here is CUDA's accuracy
    // for every backend. On a target whose guarantees are looser — Vulkan's `exp`
    // is 3 ULP against CUDA `expf`'s 2 — this band is too tight to be met, and a
    // conforming kernel fails a comparison it should pass. See `ulp_bound`'s
    // "KNOWN LIMIT" section; closing it needs a per-target accuracy seam.
    let ulp = crate::contract::ulp_bound(plan.body);
    if !ulp.is_finite() {
        return None;
    }

    // A body that only MOVES values rounds nothing: an identity passthrough, a
    // Select picking between arms, a gather. The `Fidelity` doc already names
    // this class as bit-exact; this derives it instead of trusting the caller to
    // recognise it.
    let steps = arith_steps(plan.body) + reduction_len(plan, operands);
    if steps == 0 && ulp == 0.0 {
        return Some(Fidelity::BitExact);
    }

    #[allow(clippy::cast_precision_loss)]
    let rel = (steps as f64 + 2.0 * ulp) * unit_roundoff(plan.dtype);
    Some(Fidelity::Tolerant { rel, abs: 0.0 })
}

/// The compute dtype's unit roundoff — the largest relative error a single
/// correctly-rounded operation can introduce.
///
/// Keyed to the precision the DEVICE computes at, which for every narrow float
/// is `f32`: the emitters promote an `f16`/`bf16`/FP8 load to `float` and
/// compute there (`cfamily`'s narrow-float seam). Using the storage precision
/// would produce a band far looser than the kernel's real error.
fn unit_roundoff(dt: ElementKind) -> f64 {
    match dt {
        ElementKind::F64 | ElementKind::Complex128 => f64::EPSILON / 2.0,
        _ => f64::from(f32::EPSILON) / 2.0,
    }
}

/// Count the arithmetic nodes whose rounding the device performs and the
/// `f64` oracle does not.
///
/// Leaves and `Select` contribute nothing: a leaf is a load and a `Select`
/// picks an arm without arithmetic (the same call `contract::ulp_bound` makes,
/// for the same reason).
fn arith_steps(e: &ScalarExpr) -> u32 {
    match e {
        ScalarExpr::Input(_)
        | ScalarExpr::Const(_)
        | ScalarExpr::Param(_)
        | ScalarExpr::Reduced(_)
        | ScalarExpr::Coord(_) => 0,
        ScalarExpr::Add(a, b)
        | ScalarExpr::Sub(a, b)
        | ScalarExpr::Mul(a, b)
        | ScalarExpr::Div(a, b) => 1 + arith_steps(a) + arith_steps(b),
        ScalarExpr::Binary(_, a, b) => 1 + arith_steps(a) + arith_steps(b),
        ScalarExpr::Unary(_, x) => 1 + arith_steps(x),
        // A Select rounds nothing itself; its subexpressions still count.
        ScalarExpr::Select(c, a, b) => arith_steps(c) + arith_steps(a) + arith_steps(b),
    }
}

/// How many accumulation steps a reducing cell performs, or `0` if it does not
/// reduce.
///
/// A K-long sequential sum accumulates one rounding per element, and the oracle
/// sums the same series in `f64`, so the whole length is error the band must
/// cover. This is the term that makes a large reduction's band legitimately wide
/// — and hiding it would be exactly the hand-widening this function replaces.
fn reduction_len(plan: &KernelPlan<'_>, operands: &[OperandDesc]) -> u32 {
    let reduced = matches!(
        plan.schedule,
        Schedule::Reduction { .. } | Schedule::RowReduce { .. } | Schedule::Contraction
    );
    if !reduced {
        return 0;
    }
    // The reduced extent is the largest input extent — an upper bound that does
    // not require knowing which axis the schedule folds. Over-stating widens the
    // band, which is safe; under-stating would reject a correct kernel.
    let n = operands
        .iter()
        .flat_map(|o| o.shape[..usize::from(o.rank)].iter().copied())
        .max()
        .unwrap_or(1);
    u32::try_from(n.max(1)).unwrap_or(u32::MAX)
}

/// Compare `expected` against `actual` under `fidelity`. `Ok(())` if they match;
/// `Err(msg)` naming the first mismatch otherwise.
///
/// # Errors
/// Returns a description of the first mismatching element (or a shape/length
/// mismatch).
pub fn compare(
    expected: &TypedBuffer,
    actual: &TypedBuffer,
    fidelity: Fidelity,
) -> Result<(), String> {
    if expected.shape != actual.shape {
        return Err(format!(
            "shape mismatch: {:?} vs {:?}",
            expected.shape, actual.shape
        ));
    }
    match fidelity {
        Fidelity::BitExact => {
            if expected.bytes != actual.bytes {
                let sz = elem_size(expected.dtype);
                for i in 0..(expected.bytes.len() / sz) {
                    let a = expected.bits_at(i);
                    let b = actual.bits_at(i);
                    if a != b {
                        return Err(format!(
                            "bit-exact mismatch at element {i}: 0x{a:x} vs 0x{b:x}"
                        ));
                    }
                }
                return Err("byte-length mismatch".to_string());
            }
            Ok(())
        }
        // An INTEGER dtype is compared exactly, in `i128`, whatever tolerance was
        // requested. Not a special case for tidiness: routing integers through
        // the `f64` projection means `Tolerant { rel: 0.0, abs: 0.0 }` reports
        // `2^53` and `2^53 + 1` as EQUAL, because both round to the same `f64`.
        // A comparator that cannot distinguish two distinct values of the dtype
        // it was handed is not a comparator, and it fails in the silent
        // direction — agreeing rather than complaining.
        //
        // The tolerance is still honoured, in integer units: integer arithmetic
        // has no rounding, so `abs` here means a genuine allowed distance rather
        // than an accumulated-error band, and `rel` scales against the larger
        // magnitude exactly as it does for floats.
        // Complex compares COMPONENT-WISE. A single magnitude tolerance would
        // pass `(0, 5)` against `(5, 0)` — equal magnitude, orthogonal values —
        // and complex has no ordering to fall back on, so magnitude is the only
        // scalar available and it is not enough.
        Fidelity::Tolerant { rel, abs }
            if is_complex(expected.dtype) && is_complex(actual.dtype) =>
        {
            let ea = expected.to_complex_vec();
            let ac = actual.to_complex_vec();
            if ea.len() != ac.len() {
                return Err(format!("length mismatch: {} vs {}", ea.len(), ac.len()));
            }
            for (i, (&(er, ei), &(ar, ai))) in ea.iter().zip(ac.iter()).enumerate() {
                for (part, e, a) in [("re", er, ar), ("im", ei, ai)] {
                    if e == a || (e.is_nan() && a.is_nan()) {
                        continue;
                    }
                    if e.is_infinite() || a.is_infinite() {
                        return Err(format!(
                            "infinity mismatch at element {i} ({part}): expected {e}, got {a}"
                        ));
                    }
                    let diff = (e - a).abs();
                    if diff <= abs + rel * e.abs().max(a.abs()) {
                        continue;
                    }
                    return Err(format!(
                        "tolerance mismatch at element {i} ({part}):                          expected {e}, got {a} (|Δ|={diff})"
                    ));
                }
            }
            Ok(())
        }
        Fidelity::Tolerant { rel, abs } if is_int(expected.dtype) && is_int(actual.dtype) => {
            let ea = expected.to_i128_vec();
            let ac = actual.to_i128_vec();
            if ea.len() != ac.len() {
                return Err(format!("length mismatch: {} vs {}", ea.len(), ac.len()));
            }
            for (i, (&e, &a)) in ea.iter().zip(ac.iter()).enumerate() {
                if e == a {
                    continue;
                }
                let diff = (e - a).unsigned_abs();
                let band = abs + rel * (e.unsigned_abs().max(a.unsigned_abs()) as f64);
                // `band` is computed in f64 only as a WIDTH, never as a value —
                // a lossy width still rejects a mismatch far outside it, whereas
                // a lossy value silently equates distinct ones.
                if band.is_finite() && (diff as f64) <= band {
                    continue;
                }
                return Err(format!(
                    "tolerance mismatch at element {i}: expected {e}, got {a} (|Δ|={diff})"
                ));
            }
            Ok(())
        }
        Fidelity::Tolerant { rel, abs } => {
            let ea = expected.to_f64_vec();
            let ac = actual.to_f64_vec();
            if ea.len() != ac.len() {
                return Err(format!("length mismatch: {} vs {}", ea.len(), ac.len()));
            }
            for (i, (&e, &a)) in ea.iter().zip(ac.iter()).enumerate() {
                if e.is_nan() && a.is_nan() {
                    continue;
                }
                if e == a {
                    continue; // covers ±0 and equal infinities
                }
                // A non-equal infinite expected/actual makes the relative band
                // `rel * max(|e|,|a|)` infinite, which would swallow ANY mismatch
                // (+inf vs -inf, inf vs finite, finite vs inf). An infinity only
                // matches an exactly-equal one (handled just above) — otherwise
                // reject, so an output overflowing to inf can't silently pass.
                if e.is_infinite() || a.is_infinite() {
                    return Err(format!(
                        "infinity mismatch at element {i}: expected {e}, got {a}"
                    ));
                }
                let diff = (e - a).abs();
                if diff <= abs + rel * e.abs().max(a.abs()) {
                    continue;
                }
                return Err(format!(
                    "tolerance mismatch at element {i}: expected {e}, got {a} (|Δ|={diff})"
                ));
            }
            Ok(())
        }
    }
}

// ===========================================================================
// §6 Bootstrap-trust tests (GPU-free): the oracle must prove ITSELF first.
// ===========================================================================
#[cfg(test)]
mod complex_arithmetic_tests {
    use super::*;
    use crate::ir::ScalarExpr;

    /// Evaluate `expr` at a complex dtype with two complex inputs.
    ///
    /// Goes through the real `eval` path rather than calling the arithmetic
    /// closures directly, so the routing (`is_complex` before the int/float
    /// split) is exercised too.
    fn eval_complex(expr: &ScalarExpr, a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
        let leaf = |i: u8| {
            let (re, im) = if i == 0 { a } else { b };
            Val::Complex(re, im)
        };
        let never = |_: u8| -> Val { panic!("not used") };
        let ev = Eval {
            dtype: ElementKind::Complex128,
            params: &[],
            leaf: &leaf,
            reduced: &never,
            coord: &never,
        };
        eval(expr, &ev).complex()
    }

    /// **`Mul` is not component-wise, and this is the only test that can see it.**
    ///
    /// Complex multiplication is `(ac − bd, ad + bc)`. Applying the real rule
    /// component-wise gives `(ac, bd)` — which agrees with the correct answer
    /// whenever either operand is purely real or purely imaginary, so a
    /// carelessly-chosen input pair cannot tell them apart. `(1+2i)(3+4i)` is
    /// `-5 + 10i` correctly and `3 + 8i` component-wise: different in both
    /// components, so neither half can be right by accident.
    ///
    /// # Why this is a unit test
    ///
    /// Complex is not plan-admissible yet, so complex arithmetic is unreachable
    /// through `build_plan`/`evaluate` — the integration tests in
    /// `tests/complex_domain.rs` exercise buffers, sizes and comparison, and
    /// **cannot reach the arithmetic at all**. Seeding the component-wise rule
    /// left every one of them green. Tested here because here is where it is
    /// reachable; when the plan gate admits complex, this belongs end-to-end too.
    #[test]
    fn complex_mul_mixes_components() {
        let got = eval_complex(
            &ScalarExpr::Mul(
                Box::new(ScalarExpr::Input(0)),
                Box::new(ScalarExpr::Input(1)),
            ),
            (1.0, 2.0),
            (3.0, 4.0),
        );
        assert_eq!(got, (-5.0, 10.0), "(1+2i)(3+4i) = -5+10i");
        assert_ne!(
            got,
            (3.0, 8.0),
            "component-wise multiplication is the classic wrong rule"
        );

        // i * i = -1: the identity that defines the domain.
        let i_squared = eval_complex(
            &ScalarExpr::Mul(
                Box::new(ScalarExpr::Input(0)),
                Box::new(ScalarExpr::Input(1)),
            ),
            (0.0, 1.0),
            (0.0, 1.0),
        );
        assert_eq!(i_squared, (-1.0, 0.0), "i^2 must be -1");
    }

    /// `Add`/`Sub` ARE component-wise — the contrast that makes `Mul` a rule
    /// rather than an accident.
    #[test]
    fn complex_add_and_sub_are_component_wise() {
        assert_eq!(
            eval_complex(
                &ScalarExpr::Add(
                    Box::new(ScalarExpr::Input(0)),
                    Box::new(ScalarExpr::Input(1)),
                ),
                (1.0, 2.0),
                (3.0, 4.0)
            ),
            (4.0, 6.0)
        );
        assert_eq!(
            eval_complex(
                &ScalarExpr::Sub(
                    Box::new(ScalarExpr::Input(0)),
                    Box::new(ScalarExpr::Input(1)),
                ),
                (1.0, 2.0),
                (3.0, 4.0)
            ),
            (-2.0, -2.0)
        );
    }

    /// A real operand embeds as `(x, 0)`, so a real literal is usable in a
    /// complex expression — and `Val::f64` refuses the reverse rather than
    /// silently dropping the imaginary half.
    #[test]
    fn reals_widen_into_the_complex_domain_but_not_back() {
        let leaf = |_: u8| Val::Float(3.0);
        let never = |_: u8| -> Val { panic!("not used") };
        let ev = Eval {
            dtype: ElementKind::Complex128,
            params: &[],
            leaf: &leaf,
            reduced: &never,
            coord: &never,
        };
        assert_eq!(eval(&ScalarExpr::Input(0), &ev).complex(), (3.0, 0.0));

        let r = std::panic::catch_unwind(|| Val::Complex(1.0, 2.0).f64());
        assert!(
            r.is_err(),
            "projecting a complex into the float domain must be fatal, not the              real part — a vanished imaginary half is a wrong answer that looks right"
        );
    }
}

#[cfg(test)]
mod tests {
    /// Pins the module doc's coverage list to the code.
    ///
    /// The doc previously claimed `Access::Contraction` was deferred for a long
    /// time after `evaluate` began dispatching it. A stale scope note is the one
    /// defect an ordinary test cannot catch: every implementation test passed,
    /// because the implementation was fine — it was the *claim about* the
    /// implementation that had rotted, and a reader trusting it would skip
    /// writing a differential test that already worked.
    ///
    /// This closes that by construction. The `match` is exhaustive over
    /// [`Access`], so a new access pattern breaks the build here and whoever adds
    /// it must classify it — next to the doc that has to agree. It needs no
    /// constructed values: exhaustiveness is checked at compile time.
    #[derive(Debug, PartialEq, Eq)]
    enum Coverage {
        /// `evaluate` dispatches it to a real evaluator.
        Evaluated,
        /// `evaluate` panics — genuinely not implemented.
        ///
        /// Unreachable since `eval_row_sort` shipped and every `Access` became
        /// `Evaluated`. Kept rather than deleted so the next deferred access has
        /// a home, and `#[expect]` rather than `#[allow]` deliberately: the
        /// moment something constructs this, the expectation goes unfulfilled and
        /// reds, which is the prompt to remove the attribute. An `#[allow]` would
        /// sit here silently forever.
        ///
        /// (`#[expect]` is safe here where it is not generally safe in this
        /// workspace: this lives inside `#[cfg(test)]`, so there is exactly one
        /// compilation of it and the dead-in-lib/live-in-test trap does not
        /// apply.)
        #[expect(dead_code, reason = "a home for the next deferred Access")]
        Deferred,
    }

    #[allow(dead_code)]
    fn coverage(a: &Access) -> Coverage {
        match a {
            Access::Elementwise
            | Access::Reduction { .. }
            | Access::RowReduce { .. }
            | Access::Scan { .. }
            | Access::Window { .. }
            | Access::Im2Col { .. }
            // Covered since `eval_contraction` shipped. The module doc said
            // otherwise for long enough that it is worth naming here.
            | Access::Contraction { .. }
            // Covered since `eval_row_sort` shipped: NaN-greatest ordering in
            // both directions, stable index ties, and the TopK cap read off the
            // output operand's width.
            | Access::RowSort { .. } => Coverage::Evaluated,
        }
    }

    #[test]
    fn the_documented_coverage_matches_what_evaluate_dispatches() {
        // The real work is the exhaustive `match` above, enforced at compile
        // time. This asserts the one variant that is cheap to name, so the
        // classifier cannot be deleted without a test failing too.
        assert_eq!(coverage(&Access::Elementwise), Coverage::Evaluated);
        // Every variant is now Evaluated, which makes `Deferred` unreachable —
        // kept rather than deleted so the NEXT deferred access has somewhere to
        // go, and asserted here so "nothing is deferred" is a measured claim
        // rather than an empty enum nobody noticed.
        assert_eq!(
            coverage(&Access::RowSort {
                order: SortOrder::Asc,
                stable: true,
                out: SortOut::Values,
                limit: crate::ir::SortLimit::Full,
            }),
            Coverage::Evaluated,
            "RowSort is evaluated since eval_row_sort shipped"
        );
    }

    use super::*;
    use crate::ir::{BinaryOp, OobPolicy, OpDef, ReduceOp, input, konst};
    use crate::plan::build_plan;
    use unpopped_vocab::{ArchSku, OpCategory, StructureKey, structure_key};

    // --- plan/key builders --------------------------------------------------

    fn desc(shape: &[i64], strides: &[i64], dt: ElementKind) -> OperandDesc {
        OperandDesc::new(shape.len(), shape, strides, dt, 256)
    }

    fn key(cat: OpCategory, ops: &[OperandDesc]) -> StructureKey {
        structure_key(cat, ops, ArchSku::Sm89)
    }

    fn f32b(shape: &[i64], data: &[f32]) -> TypedBuffer {
        TypedBuffer::from_f32(shape, data)
    }

    // Decode helpers reused across the suite.
    fn i32s(b: &TypedBuffer) -> Vec<i32> {
        (0..b.len()).map(|i| b.bits_at(i) as u32 as i32).collect()
    }

    fn f32s(b: &TypedBuffer) -> Vec<f32> {
        (0..b.len())
            .map(|i| f32::from_bits(b.bits_at(i) as u32))
            .collect()
    }

    // The raw 32-bit storage pattern of element `i`.
    fn bit32(b: &TypedBuffer, i: usize) -> u32 {
        b.bits_at(i) as u32
    }

    // --- A. Elementwise -----------------------------------------------------

    // NOTE (oracle→kiss-ref consolidation, 2026-07-29): the pure float
    // VALUE-semantics elementwise self-tests that lived here —
    // `elementwise_add_contiguous`, `elementwise_relu_neg_zero_and_nan`,
    // `elementwise_maxmin_prop_signed_zero_ties_keep_a`,
    // `elementwise_affine_with_params` (and `probe_classes_add_bit_exact_zeros`
    // below) — were RETIRED. kiss-ref is now the single float value-semantics
    // reference; each edge is asserted against it via the in-tree converter in
    // `kiss_ref_diff::tests` (add_contiguous / relu signed-zero+NaN /
    // max_prop-min_prop a-on-ties / affine-via-runtime_scalar / signed-zero-add).
    // The `Access::Elementwise` arm and the PLUMBING tests below (raw-bit select,
    // int8 store-truncation, strided/broadcast/flipped/permuted views,
    // compute-dtype cmp/select) STAY — that is Baracuda emitter/layout territory
    // kiss-ref's f32/dense value-DAG model does not cover (the locked boundary).

    // NOTE (oracle→kiss-ref consolidation): `select_raw_bit_neg_zero_survives`
    // was RETIRED. kiss-ref's raw-bit select (resolve.rs:58, byte-identical
    // between 0.1.0 and the strengthened test at 5d0538b) is the equal-or-better
    // reference; Baracuda's select round-trips through the converter and
    // preserves -0.0 bit-for-bit, asserted in
    // `kiss_ref_diff::tests::select_raw_bit_neg_zero_through_kiss_ref` (which
    // also verified the previously-unproven select converter path). The
    // Access::Elementwise arm + `cmp_and_select_decide_in_compute_dtype` stay.

    // KILL-test: i8 (S8) add wraps at 127 (mod-2^8 store truncation).
    #[test]
    fn int8_add_wraps() {
        let op = OpDef::elementwise("add8", 2, &[ElementKind::I8], input(0) + input(1));
        let a = desc(&[2], &[1], ElementKind::I8);
        let k = key(OpCategory::BinaryElementwise, &[a, a, a]);
        let plan = build_plan(&op, &k);
        let ops = [a, a, a];
        let ins = [
            TypedBuffer::from_i8(&[2], &[100, 127]),
            TypedBuffer::from_i8(&[2], &[100, 1]),
        ];
        let out = evaluate(&plan, &ops, &ins, &[]);
        // 100+100=200 -> -56 (i8); 127+1=128 -> -128 (i8).
        assert_eq!(out[0].bytes[0] as i8, -56);
        assert_eq!(out[0].bytes[1] as i8, -128);
    }

    // Layout axes: broadcast (stride 0), flipped (neg stride), permute, base_offset.
    #[test]
    fn elementwise_broadcast_input() {
        let op = OpDef::elementwise("addb", 2, &[ElementKind::F32], input(0) + input(1));
        let a = desc(&[3], &[1], ElementKind::F32);
        let b = desc(&[3], &[0], ElementKind::F32); // broadcast scalar
        let k = key(OpCategory::BinaryElementwise, &[a, b, a]);
        let plan = build_plan(&op, &k);
        let ops = [a, b, a];
        let ins = [
            f32b(&[3], &[1.0, 2.0, 3.0]),
            TypedBuffer::new(
                ElementKind::F32,
                vec![3],
                vec![0],
                10.0f32.to_bits().to_le_bytes().to_vec(),
            ),
        ];
        let out = evaluate(&plan, &ops, &ins, &[]);
        assert_eq!(f32s(&out[0]), vec![11.0, 12.0, 13.0]);
    }

    #[test]
    fn elementwise_flipped_input() {
        // Input 0 read with a negative stride (reversed view); base_offset points
        // at the last element so reads walk backward in-bounds.
        let op = OpDef::elementwise("flip", 1, &[ElementKind::F32], input(0) + konst(0.0));
        let a = desc(&[3], &[-1], ElementKind::F32);
        let out_d = desc(&[3], &[1], ElementKind::F32);
        let k = key(OpCategory::UnaryElementwise, &[a, out_d]);
        let plan = build_plan(&op, &k);
        let ops = [a, out_d];
        let mut buf = f32b(&[3], &[1.0, 2.0, 3.0]);
        buf.strides = vec![-1];
        buf.base_offset = 2; // origin at element 2
        let ins = [buf];
        let out = evaluate(&plan, &ops, &ins, &[]);
        assert_eq!(f32s(&out[0]), vec![3.0, 2.0, 1.0]);
    }

    #[test]
    fn elementwise_base_offset_slice() {
        // A base_offset slice: read a 2-element window starting at element 1.
        let op = OpDef::elementwise("slice", 1, &[ElementKind::F32], input(0) + konst(0.0));
        let a = desc(&[2], &[1], ElementKind::F32);
        let k = key(OpCategory::UnaryElementwise, &[a, a]);
        let plan = build_plan(&op, &k);
        let ops = [a, a];
        let mut buf = f32b(&[4], &[10.0, 20.0, 30.0, 40.0]);
        buf.shape = vec![2];
        buf.strides = vec![1];
        buf.base_offset = 1;
        let ins = [buf];
        let out = evaluate(&plan, &ops, &ins, &[]);
        assert_eq!(f32s(&out[0]), vec![20.0, 30.0]);
    }

    // --- B. Reduction -------------------------------------------------------

    // NOTE (oracle→kiss-ref consolidation, Task 3): the pure float
    // VALUE-semantics reduction self-tests — `reduction_sum_last_axis`,
    // `reduction_max_nan_sticks`, `reduction_mean_divisor`, and
    // `reduction_outer_axis` (below) — were RETIRED. Each fold is asserted
    // against kiss-ref via the in-tree converter in `kiss_ref_diff::tests`
    // (reduce_sum_last / reduce_max_nan_sticks — pins kiss-ref's reduce-Max as
    // NaN-propagating / reduce_mean_divisor / reduce_outer_axis). The
    // `Access::Reduction` arm STAYS — exercised by shape.rs's oracle differential
    // and reserved for physical/int reduction plumbing (kiss-ref's f32/dense
    // value-DAG model does not cover those).

    // --- C. RowReduce (softmax, layernorm) ----------------------------------
    // NOTE (oracle→kiss-ref consolidation, Task 5 — prove-then-retire): the
    // rowreduce VALUE self-tests `rowreduce_softmax` and `rowreduce_layernorm`
    // were RETIRED into the `oracle_and_kiss_ref` DIFFERENTIALS in
    // `kiss_ref_diff::tests` (softmax_oracle_eq_kiss_ref /
    // layernorm_oracle_eq_kiss_ref). Each runs the op through BOTH the oracle
    // AND kiss-ref, asserts they agree within tolerance (exp/div/rsqrt are
    // transcendental, not bit-exact), AND asserts the ported independent
    // invariants on kiss-ref (softmax sums-to-1 + monotone; layernorm zero-mean
    // + hand-computed values). So the oracle RowReduce arm stays EXERCISED (leg 1)
    // and the value stays independently checked — equal-or-better than the
    // retired single-impl test.

    // --- D. Scan (cumsum, cummax; fwd/reverse/exclusive) --------------------

    // NOTE (oracle→kiss-ref consolidation, Task 4): the portable float scan
    // VALUE-semantics self-tests `scan_cumsum_forward` and
    // `scan_cumsum_exclusive_first_pos_identity` were RETIRED — asserted against
    // kiss-ref via the converter in `kiss_ref_diff::tests`
    // (scan_cumsum_forward / scan_cumsum_exclusive). The `Access::Scan` arm STAYS
    // (exercised by shape.rs + the cummax test below). `scan_cummax_*` is KEPT
    // WHOLE: it exercises REVERSE scan, an emission-level honest miss (flip
    // withdrawn — `semantics_dag` returns None) that kiss-ref cannot cover, so it
    // has no equal-or-better replacement to retire against.

    // --- I. Gather / scatter -----------------------------------------------

    fn i32b(shape: &[i64], data: &[i32]) -> TypedBuffer {
        let mut b = TypedBuffer::new(
            ElementKind::I32,
            shape.to_vec(),
            row_major(shape),
            vec![0u8; data.len() * 4],
        );
        for (i, &v) in data.iter().enumerate() {
            store_val(Val::Int(i128::from(v)), ElementKind::I32, &mut b.bytes, i);
        }
        b
    }

    fn row_major(shape: &[i64]) -> Vec<i64> {
        let mut st = vec![1i64; shape.len()];
        for d in (0..shape.len().saturating_sub(1)).rev() {
            st[d] = st[d + 1] * shape[d + 1];
        }
        st
    }

    /// A gather reads at a data-dependent coordinate.
    #[test]
    fn gather_permutes_by_its_index_operand() {
        let d = desc(&[1, 4], &[4, 1], ElementKind::F32);
        let i = desc(&[1, 4], &[4, 1], ElementKind::I32);
        let op = OpDef::gather(
            "g",
            &[ElementKind::F32],
            1,
            OobPolicy::Clamp,
            ElementKind::I32,
        );
        let k = key(OpCategory::UnaryElementwise, &[d, i, d]);
        let out = evaluate(
            &build_plan(&op, &k),
            &[d, i, d],
            &[
                f32b(&[1, 4], &[10.0, 20.0, 30.0, 40.0]),
                i32b(&[1, 4], &[3, 0, 2, 1]),
            ],
            &[],
        );
        assert_eq!(f32s(&out[0]), vec![40.0, 10.0, 30.0, 20.0]);
    }

    /// `Clamp` pins out-of-range indices to the ends, in BOTH directions.
    ///
    /// ⚠️ `Skip` and `ZeroFill` are NOT distinguishable here and that is a
    /// property of the oracle rather than of the semantics: `alloc_output`
    /// zeroes the destination, so "leave the cell alone" and "write zero"
    /// produce the same bytes. On device they differ whenever the caller
    /// pre-filled the buffer. **Recorded rather than tested, because a test
    /// asserting they agree would be asserting the oracle's limitation as if it
    /// were the contract.**
    #[test]
    fn gather_clamp_pins_both_ends_and_negative_counts_as_out_of_range() {
        let d = desc(&[1, 4], &[4, 1], ElementKind::F32);
        let i = desc(&[1, 4], &[4, 1], ElementKind::I32);
        let op = OpDef::gather(
            "gc",
            &[ElementKind::F32],
            1,
            OobPolicy::Clamp,
            ElementKind::I32,
        );
        let k = key(OpCategory::UnaryElementwise, &[d, i, d]);
        let out = evaluate(
            &build_plan(&op, &k),
            &[d, i, d],
            &[
                f32b(&[1, 4], &[10.0, 20.0, 30.0, 40.0]),
                i32b(&[1, 4], &[3, -1, 5, 1]),
            ],
            &[],
        );
        assert_eq!(
            f32s(&out[0]),
            vec![40.0, 10.0, 40.0, 20.0],
            "-1 clamps to index 0 and 5 clamps to index 3"
        );
    }

    /// ⚠️ A scatter iterates its SOURCE, not its destination.
    ///
    /// This is the bincount shape: 6 source elements into 2 bins. An
    /// implementation that walked the output shape would visit 2 positions and
    /// silently drop four fifths of the input — and every same-extent test above
    /// would still pass. That is why this one has unequal extents.
    #[test]
    fn scatter_add_iterates_the_source_not_the_destination() {
        let src = desc(&[1, 6], &[6, 1], ElementKind::I32);
        let idx = desc(&[1, 6], &[6, 1], ElementKind::I32);
        let dst = desc(&[1, 2], &[2, 1], ElementKind::I32);
        let op = OpDef::scatter_add("sa", &[ElementKind::I32], 1, ElementKind::I32);
        let k = key(OpCategory::UnaryElementwise, &[src, idx, dst]);
        let out = evaluate(
            &build_plan(&op, &k),
            &[src, idx, dst],
            &[
                i32b(&[1, 6], &[1, 2, 3, 4, 5, 6]),
                i32b(&[1, 6], &[0, 1, 0, 1, 0, 1]),
            ],
            &[],
        );
        assert_eq!(
            i32s(&out[0]),
            vec![9, 12],
            "1+3+5 into bin 0 and 2+4+6 into bin 1; a destination-shaped walk gives [1,2]"
        );
    }

    /// An out-of-range scatter target is dropped — v1 pins the policy to `Skip`.
    #[test]
    fn scatter_skips_out_of_range_targets() {
        let src = desc(&[1, 4], &[4, 1], ElementKind::I32);
        let idx = desc(&[1, 4], &[4, 1], ElementKind::I32);
        let dst = desc(&[1, 2], &[2, 1], ElementKind::I32);
        let op = OpDef::scatter_add("ss", &[ElementKind::I32], 1, ElementKind::I32);
        let k = key(OpCategory::UnaryElementwise, &[src, idx, dst]);
        let out = evaluate(
            &build_plan(&op, &k),
            &[src, idx, dst],
            &[
                i32b(&[1, 4], &[7, 100, 9, 200]),
                i32b(&[1, 4], &[0, 5, 1, -1]),
            ],
            &[],
        );
        assert_eq!(
            i32s(&out[0]),
            vec![7, 9],
            "the 5 and -1 targets are dropped"
        );
    }

    /// Float `AtomicAdd` is bit-exact on the deterministic corpus.
    ///
    /// The values are integer-valued floats well inside f32's exact range
    /// (2^24), so every partial sum is exact and the result is independent of
    /// accumulation order — which is what makes an order-nondeterministic op
    /// checkable by VALUE rather than by a bound. See `docs/deferred.md`.
    #[test]
    fn scatter_add_f32_is_exact_on_the_integer_valued_corpus() {
        let src = desc(&[1, 6], &[6, 1], ElementKind::F32);
        let idx = desc(&[1, 6], &[6, 1], ElementKind::I32);
        let dst = desc(&[1, 2], &[2, 1], ElementKind::F32);
        let op = OpDef::scatter_add("saf", &[ElementKind::F32], 1, ElementKind::I32);
        let k = key(OpCategory::UnaryElementwise, &[src, idx, dst]);
        let out = evaluate(
            &build_plan(&op, &k),
            &[src, idx, dst],
            &[
                f32b(&[1, 6], &[1.0, 2.0, 4.0, 8.0, 16.0, 32.0]),
                i32b(&[1, 6], &[0, 1, 0, 1, 0, 1]),
            ],
            &[],
        );
        assert_eq!(
            f32s(&out[0]),
            vec![21.0, 42.0],
            "1+4+16 and 2+8+32, exact in f32 at any accumulation order"
        );
    }

    // --- H. RowSort ---------------------------------------------------------

    /// NaN orders GREATEST in BOTH directions — so ascending puts it last and
    /// **descending puts it FIRST**.
    ///
    /// The descending arm is the one a reader expects to be wrong, and a test
    /// that only checked ascending would pass against a comparator that treated
    /// NaN as smallest-under-reversal. Both arms, from one corpus.
    #[test]
    fn row_sort_orders_nan_greatest_in_both_directions() {
        let ind = desc(&[1, 4], &[4, 1], ElementKind::F32);
        let data = [3.0f32, f32::NAN, 1.0, 2.0];
        let k = key(OpCategory::Sorting, &[ind, ind]);

        let asc = OpDef::row_sort("s_asc", ElementKind::F32, SortOrder::Asc);
        let out = evaluate(
            &build_plan(&asc, &k),
            &[ind, ind],
            &[f32b(&[1, 4], &data)],
            &[],
        );
        let v = f32s(&out[0]);
        assert_eq!(&v[..3], &[1.0, 2.0, 3.0], "ascending values");
        assert!(v[3].is_nan(), "ascending puts NaN LAST, got {v:?}");

        let desc_op = OpDef::row_sort("s_desc", ElementKind::F32, SortOrder::Desc);
        let out = evaluate(
            &build_plan(&desc_op, &k),
            &[ind, ind],
            &[f32b(&[1, 4], &data)],
            &[],
        );
        let v = f32s(&out[0]);
        assert!(v[0].is_nan(), "descending puts NaN FIRST, got {v:?}");
        assert_eq!(&v[1..], &[3.0, 2.0, 1.0], "descending values after the NaN");
    }

    /// Ties keep their original index order, in both directions.
    ///
    /// `sort_by` is stable and reversing a comparator does not change which
    /// pairs compare Equal — so descending must NOT reverse ties. An
    /// implementation that sorted ascending then reversed the whole row would
    /// pass the ordering test above and fail this one.
    #[test]
    fn row_sort_ties_keep_their_original_index_in_both_directions() {
        // `row_sort_indices` builds `SortOut::Both`: input, VALUES out, INDICES
        // out. Three operands, and the indices are output 1.
        let ind = desc(&[1, 4], &[4, 1], ElementKind::F32);
        let iout = desc(&[1, 4], &[4, 1], ElementKind::I32);
        let data = [5.0f32, 1.0, 5.0, 1.0];
        let k = key(OpCategory::Sorting, &[ind, ind, iout]);
        let ops = [ind, ind, iout];

        let asc = OpDef::row_sort_indices("i_asc", ElementKind::F32, SortOrder::Asc);
        let out = evaluate(&build_plan(&asc, &k), &ops, &[f32b(&[1, 4], &data)], &[]);
        assert_eq!(
            i32s(&out[1]),
            vec![1, 3, 0, 2],
            "the two 1.0s keep 1<3, the two 5.0s keep 0<2"
        );
        assert_eq!(
            f32s(&out[0]),
            vec![1.0, 1.0, 5.0, 5.0],
            "Both's value output must agree with its own index output"
        );

        let dsc = OpDef::row_sort_indices("i_desc", ElementKind::F32, SortOrder::Desc);
        let out = evaluate(&build_plan(&dsc, &k), &ops, &[f32b(&[1, 4], &data)], &[]);
        assert_eq!(
            i32s(&out[1]),
            vec![0, 2, 1, 3],
            "descending reverses the VALUES but not the ties — a whole-row reverse would give [2,0,3,1]"
        );
    }

    /// A narrower output is a TopK cap, read off the operand rather than a flag.
    #[test]
    fn row_sort_truncates_to_the_output_width() {
        let ind = desc(&[1, 5], &[5, 1], ElementKind::F32);
        let outd = desc(&[1, 2], &[2, 1], ElementKind::F32);
        let data = [3.0f32, 1.0, 4.0, 1.5, 9.0];
        let k = key(OpCategory::Sorting, &[ind, outd]);
        let op = OpDef::row_sort("top2", ElementKind::F32, SortOrder::Desc);
        let out = evaluate(
            &build_plan(&op, &k),
            &[ind, outd],
            &[f32b(&[1, 5], &data)],
            &[],
        );
        assert_eq!(f32s(&out[0]), vec![9.0, 4.0], "top 2 descending");
    }

    /// Multiple rows are sorted independently.
    ///
    /// A row loop that computed `base` once, or shared the permutation, passes
    /// every single-row test above.
    #[test]
    fn row_sort_treats_rows_independently() {
        let ind = desc(&[2, 3], &[3, 1], ElementKind::F32);
        let data = [3.0f32, 1.0, 2.0, 9.0, 7.0, 8.0];
        let k = key(OpCategory::Sorting, &[ind, ind]);
        let op = OpDef::row_sort("rows", ElementKind::F32, SortOrder::Asc);
        let out = evaluate(
            &build_plan(&op, &k),
            &[ind, ind],
            &[f32b(&[2, 3], &data)],
            &[],
        );
        assert_eq!(f32s(&out[0]), vec![1.0, 2.0, 3.0, 7.0, 8.0, 9.0]);
    }

    #[test]
    fn scan_cummax_forward_and_reverse_and_exclusive() {
        let ind = desc(&[1, 4], &[4, 1], ElementKind::F32);
        let data = [3.0f32, 1.0, 4.0, 2.0];

        // forward inclusive
        let op = OpDef::scan_simple(
            "cummax",
            &[ElementKind::F32],
            ReduceOp::Max,
            1,
            false,
            false,
        );
        let k = key(OpCategory::Scan, &[ind, ind]);
        let plan = build_plan(&op, &k);
        let out = evaluate(&plan, &[ind, ind], &[f32b(&[1, 4], &data)], &[]);
        assert_eq!(f32s(&out[0]), vec![3.0, 3.0, 4.0, 4.0]);

        // reverse inclusive
        let op = OpDef::scan_simple(
            "cummax_r",
            &[ElementKind::F32],
            ReduceOp::Max,
            1,
            true,
            false,
        );
        let plan = build_plan(&op, &k);
        let out = evaluate(&plan, &[ind, ind], &[f32b(&[1, 4], &data)], &[]);
        assert_eq!(f32s(&out[0]), vec![4.0, 4.0, 4.0, 2.0]);

        // forward EXCLUSIVE: first pos = -inf (Max monoid identity).
        let op = OpDef::scan_simple(
            "cummax_e",
            &[ElementKind::F32],
            ReduceOp::Max,
            1,
            false,
            true,
        );
        let plan = build_plan(&op, &k);
        let out = evaluate(&plan, &[ind, ind], &[f32b(&[1, 4], &data)], &[]);
        let g = f32s(&out[0]);
        assert_eq!(g[0], f32::NEG_INFINITY, "excl cummax first pos = -inf");
        assert_eq!(&g[1..], &[3.0, 3.0, 4.0]);
    }

    // --- E. Window (maxpool, avgpool) ---------------------------------------

    #[test]
    fn window_maxpool() {
        // k_in=4, size=2, stride=2 -> k_out=2. max over [1,2],[3,4].
        let op = OpDef::window_simple(
            "maxpool",
            &[ElementKind::F32],
            ReduceOp::Max,
            1,
            2,
            2,
            1,
            0,
            0,
            false,
        );
        let ind = desc(&[1, 4], &[4, 1], ElementKind::F32);
        let outd = desc(&[1, 2], &[2, 1], ElementKind::F32);
        let k = key(OpCategory::Pooling, &[ind, outd]);
        let plan = build_plan(&op, &k);
        let ops = [ind, outd];
        let ins = [f32b(&[1, 4], &[1.0, 2.0, 3.0, 4.0])];
        let out = evaluate(&plan, &ops, &ins, &[]);
        assert_eq!(f32s(&out[0]), vec![2.0, 4.0]);
    }

    // KILL-test: avg_pool count_include_pad divisor.
    #[test]
    fn window_avgpool_count_include_pad_divisor() {
        // k_in=4, size=3, stride=2, pad_lo=1, pad_hi=1 -> k_out=2.
        // o=0: taps p=-1(pad),0,1 -> valid {x0,x1}. o=1: p=1,2,3 -> {x1,x2,x3}.
        let ind = desc(&[1, 4], &[4, 1], ElementKind::F32);
        let outd = desc(&[1, 2], &[2, 1], ElementKind::F32);
        let k = key(OpCategory::Pooling, &[ind, outd]);
        let data = [2.0f32, 4.0, 6.0, 8.0];

        // count_include_pad = false: divide by valid count.
        let op = OpDef::window_simple(
            "avg",
            &[ElementKind::F32],
            ReduceOp::Mean,
            1,
            3,
            2,
            1,
            1,
            1,
            false,
        );
        let plan = build_plan(&op, &k);
        let out = evaluate(&plan, &[ind, outd], &[f32b(&[1, 4], &data)], &[]);
        // o0 = (2+4)/2 = 3 ; o1 = (4+6+8)/3 = 6.
        assert_eq!(f32s(&out[0]), vec![3.0, 6.0]);

        // count_include_pad = true: divide by size (3).
        let op = OpDef::window_simple(
            "avg_cip",
            &[ElementKind::F32],
            ReduceOp::Mean,
            1,
            3,
            2,
            1,
            1,
            1,
            true,
        );
        let plan = build_plan(&op, &k);
        let out = evaluate(&plan, &[ind, outd], &[f32b(&[1, 4], &data)], &[]);
        // o0 = (2+4)/3 ; o1 = (4+6+8)/3 = 6.
        let g = f32s(&out[0]);
        assert!((g[0] - 6.0 / 3.0).abs() < 1e-6);
        assert!((g[1] - 6.0).abs() < 1e-6);
    }

    // --- G. Im2Col ----------------------------------------------------------

    #[test]
    fn im2col_3x3() {
        // N=1,C=1,H=3,W=3; 3x3 kernel, stride 1, pad 1, dil 1 -> oH=oW=3.
        // Output [1, 9, 9]. Verify the center tap (ki=1,kj=1) reproduces the input,
        // and a corner tap zero-pads out of bounds.
        let op = OpDef::im2col_2d("im2col", ElementKind::F32, (3, 3), (1, 1), (1, 1), (1, 1));
        let ind = desc(&[1, 1, 3, 3], &[9, 9, 3, 1], ElementKind::F32);
        let outd = desc(&[1, 9, 9], &[81, 9, 1], ElementKind::F32);
        let k = key(OpCategory::Convolution, &[ind, outd]);
        let plan = build_plan(&op, &k);
        let ops = [ind, outd];
        let data: Vec<f32> = (1..=9).map(|v| v as f32).collect();
        let ins = [f32b(&[1, 1, 3, 3], &data)];
        let out = evaluate(&plan, &ops, &ins, &[]);
        let g = f32s(&out[0]);
        // Center tap row = c*9 + ki*3 + kj = 0*9 + 1*3 + 1 = 4; each col=oh*3+ow.
        for oh in 0..3usize {
            for ow in 0..3usize {
                let col = oh * 3 + ow;
                let val = g[4 * 9 + col];
                assert_eq!(val, data[oh * 3 + ow], "center tap = input");
            }
        }
        // Top-left tap (ki=0,kj=0) at output (oh=0,ow=0) is t=0: in_h=-1,in_w=-1
        // is out of bounds -> zero-pad.
        assert_eq!(g[0], 0.0, "OOB tap zero-pads");
    }

    // --- half codec round-trips + probe classes -----------------------------

    #[test]
    fn half_codec_roundtrip() {
        for &v in &[0.0f32, -0.0, 1.0, -2.5, 0.5, 65504.0] {
            let b = f32_to_f16_bits(v);
            let back = f16_to_f64(b);
            assert!(
                (back - f64::from(v)).abs() <= f64::from(v).abs() * 1e-2 + 1e-3,
                "{v} -> {back}"
            );
        }
        // ±inf and -0.0 map exactly.
        assert_eq!(f16_to_f64(f32_to_f16_bits(f32::INFINITY)), f64::INFINITY);
        assert!(f16_to_f64(f32_to_f16_bits(-0.0)).is_sign_negative());
        // bf16 preserves the high mantissa; 1.0 exact.
        assert_eq!(bf16_to_f64(f32_to_bf16_bits(1.0)), 1.0);
        // NaN stays NaN in both codecs.
        assert!(f16_to_f64(f32_to_f16_bits(f32::NAN)).is_nan());
        assert!(bf16_to_f64(f32_to_bf16_bits(f32::NAN)).is_nan());
    }

    // (`probe_classes_add_bit_exact_zeros` retired here — signed-zero add is now
    // referenced against kiss-ref in `kiss_ref_diff::tests::signed_zero_add_through_kiss_ref`.)

    // --- H. Contraction (matmul) --------------------------------------------
    // NOTE (oracle→kiss-ref consolidation, Task 6 — prove-then-retire): the
    // matmul VALUE self-tests (identity / relu-epilogue / bias-relu / batched)
    // and their gemm cell builders were RETIRED into the `oracle_and_kiss_ref`
    // DIFFERENTIALS in `kiss_ref_diff::tests` (matmul_identity /
    // matmul_relu_epilogue / matmul_bias_relu / batched_matmul, all suffixed
    // `_oracle_eq_kiss_ref`). Each runs the op through BOTH the oracle AND
    // kiss-ref over an exactly-representable integer cell, asserts they agree
    // BIT-for-bit, AND asserts kiss-ref equals the hand-computed value. So the
    // oracle Contraction arm stays EXERCISED (leg 1) and the value stays
    // independently checked — equal-or-better than the retired single-impl tests.

    // Task 6: `eval_contraction` reads operand layout by PHYSICAL strides (the
    // ground truth, matching the emitter's `operand_stride_binding`), not by
    // re-applying the op's `View::Permute` on top of already-transposed strides
    // (that would double-transpose). This is a VIEWED contraction — the op
    // carries `View::Permute` on rhs — so it exercises the `input_perm` path.
    #[test]
    fn oracle_matches_reference_transposed_rhs_f32() {
        use crate::ir::{ContractionAxes, View, reduced};
        let (m, k, n) = (2usize, 3usize, 2usize);
        // lhs [M,K] row-major; rhs LOGICAL [K,N] but stored transposed ([N,K], K unit).
        let lhs_d = desc(&[m as i64, k as i64], &[k as i64, 1], ElementKind::F32);
        let rhs_d = desc(&[k as i64, n as i64], &[1, k as i64], ElementKind::F32); // transposed
        let out_d = desc(&[m as i64, n as i64], &[n as i64, 1], ElementKind::F32);
        let op = OpDef::contraction(
            "matmul",
            &[ElementKind::F32],
            ContractionAxes::matmul(),
            reduced(0),
        )
        .with_views(vec![View::Identity, View::Permute { perm: vec![1, 0] }]);
        let kk = key(OpCategory::Gemm, &[lhs_d, rhs_d, out_d]);
        let plan = build_plan(&op, &kk);
        // lhs data: logical [M,K] row-major = [[1,2,3],[4,5,6]].
        let lhs = f32b(&[m as i64, k as i64], &[1., 2., 3., 4., 5., 6.]);
        // rhs LOGICAL [K,N]=[[10,11],[12,13],[14,15]] stored transposed [N,K] row-major
        // = flatten([[10,12,14],[11,13,15]]) so physical strides [1,k] read it back.
        let rhs = f32b(&[k as i64, n as i64], &[10., 12., 14., 11., 13., 15.]);
        let got = f32s(&evaluate(&plan, &[lhs_d, rhs_d, out_d], &[lhs, rhs], &[])[0]);
        // from-scratch [M,K]·[K,N], K ascending: want[m][n] = Σ_k lhsL[m][k]·rhsL[k][n]
        let lhs_l = [[1., 2., 3.], [4., 5., 6.]];
        let rhs_l = [[10., 11.], [12., 13.], [14., 15.]];
        let mut want = vec![0f32; m * n];
        for mi in 0..m {
            for ni in 0..n {
                let mut a = 0f32;
                for ki in 0..k {
                    a += lhs_l[mi][ki] * rhs_l[ki][ni];
                }
                want[mi * n + ni] = a;
            }
        }
        assert_eq!(
            got, want,
            "oracle transposed-rhs must equal the logical matmul reference"
        );
    }

    // Task 6 differential matrix: {lhs_order, rhs_order} x {rank-2, rank-3
    // batched}, each vs a from-scratch `[M,K]·[K,N]` (or batched) reference.
    //
    // Broadcast-batch rhs (stride 0) is deliberately NOT a matrix cell here:
    // at the time this matrix was written, `classify_mat_layout` (Task 1/2,
    // already shipped on this branch) declined ANY zero-stride axis as
    // non-packed regardless of position, so `derive_contraction` returned
    // `None` for a stride-0 batch axis and `build_plan` panicked
    // ("contraction cell must carry ContractionKey facts") before a plan
    // could even be built — verified directly with a throwaway probe during
    // development, not guessed. That gate sat upstream of Task 6 (in the
    // vocab crate) and was out of this task's surgical scope to relax.
    //
    // UPDATE (Task 8, commit 2203bcfb): `classify_mat_layout` now admits a
    // stride-0 axis (skips it in the packedness walk instead of declining),
    // so a broadcast-KV rhs is constructible. Task 9 adds the two
    // now-possible broadcast-KV (GQA) cells in a DEDICATED helper below
    // (`assert_broadcast_kv_matches_reference`) rather than folding them into
    // this matrix/helper, to keep this committed 8-cell matrix untouched.

    /// Realize a (possibly batched) operand's PHYSICAL storage for a chosen
    /// storage order: `logical[b]` is batch `b`'s row-major `rows*cols`
    /// logical matrix (`logical.len()==1` for the unbatched case). Batch, when
    /// present, is the OUTERMOST storage axis (v1 convention, per the Task 5
    /// emitter). `transposed=false` -> canonical row-major (`strides=
    /// [cols,1]`, physical == logical); `transposed=true` -> physically
    /// `[cols,rows]` row-major per batch (`strides=[1,rows]`), so reading
    /// logical `(r,c)` through the returned strides reproduces
    /// `logical[b][r*cols+c]`. Returns `(strides, physical_flat_data)`.
    fn layout_matrix(
        rows: i64,
        cols: i64,
        logical: &[Vec<f32>],
        transposed: bool,
    ) -> (Vec<i64>, Vec<f32>) {
        let (r, c) = (rows as usize, cols as usize);
        let mut strides = Vec::new();
        if logical.len() > 1 {
            strides.push(rows * cols); // batch stride: outermost, packed
        }
        if transposed {
            strides.extend_from_slice(&[1, rows]);
        } else {
            strides.extend_from_slice(&[cols, 1]);
        }
        let mut phys = Vec::with_capacity(logical.len() * r * c);
        for slice in logical {
            if transposed {
                let mut buf = vec![0f32; r * c];
                for ri in 0..r {
                    for ci in 0..c {
                        buf[ci * r + ri] = slice[ri * c + ci];
                    }
                }
                phys.extend(buf);
            } else {
                phys.extend_from_slice(slice);
            }
        }
        (strides, phys)
    }

    // Hand-verification of `layout_matrix`'s transposed branch: the rank-2
    // case reuses the EXACT rhs fixture from
    // `oracle_matches_reference_transposed_rhs_f32` above (hand-verified
    // against the brief: rhs LOGICAL [K,N]=[[10,11],[12,13],[14,15]] stored
    // transposed [N,K] row-major = [10,12,14,11,13,15], strides=[1,k]) as a
    // golden case, plus a from-scratch batched hand trace.
    #[test]
    fn layout_matrix_matches_hand_verified_fixtures() {
        // rank-2 transposed: matches the RED fixture above exactly.
        let (strides, phys) = layout_matrix(3, 2, &[vec![10., 11., 12., 13., 14., 15.]], true);
        assert_eq!(strides, vec![1, 3]);
        assert_eq!(phys, vec![10., 12., 14., 11., 13., 15.]);
        // canonical (identity) is a pass-through.
        let (strides, phys) = layout_matrix(3, 2, &[vec![10., 11., 12., 13., 14., 15.]], false);
        assert_eq!(strides, vec![2, 1]);
        assert_eq!(phys, vec![10., 11., 12., 13., 14., 15.]);
        // batched (B=2, rows=2, cols=2), transposed: batch stays outermost.
        // logical batch0=[[1,2],[3,4]], batch1=[[5,6],[7,8]]; transposed
        // per-batch physical [cols,rows] row-major = [[1,3],[2,4]] /
        // [[5,7],[6,8]] -> flat [1,3,2,4, 5,7,6,8]; strides=[rows*cols,1,rows]
        // = [4,1,2]. Hand-check logical(b=0,r=1,c=0)=3: offset =
        // 0*4 + 1*1 + 0*2 = 1 -> phys[1] = 3. ✓
        let (strides, phys) =
            layout_matrix(2, 2, &[vec![1., 2., 3., 4.], vec![5., 6., 7., 8.]], true);
        assert_eq!(strides, vec![4, 1, 2]);
        assert_eq!(phys, vec![1., 3., 2., 4., 5., 7., 6., 8.]);
    }

    /// One matrix cell: build a (possibly batched, possibly layout-permuted)
    /// contraction through the REAL `build_plan`/`evaluate` pipeline and
    /// assert it equals a from-scratch `[M,K]·[K,N]` reference (K ascending).
    fn assert_contraction_matches_reference(
        batched: bool,
        lhs_transposed: bool,
        rhs_transposed: bool,
    ) {
        use crate::ir::{ContractionAxes, View, reduced};
        let (m, k, n): (i64, i64, i64) = (2, 3, 2);
        let bdim: i64 = if batched { 2 } else { 1 };

        // Small distinct-per-batch integer data (exact in f32).
        let lhs_logical: Vec<Vec<f32>> = (0..bdim)
            .map(|bi| (0..m * k).map(|idx| (bi * 100 + idx + 1) as f32).collect())
            .collect();
        let rhs_logical: Vec<Vec<f32>> = (0..bdim)
            .map(|bi| (0..k * n).map(|idx| (bi * 100 + idx + 1) as f32).collect())
            .collect();

        let (lhs_strides, lhs_phys) = layout_matrix(m, k, &lhs_logical, lhs_transposed);
        let (rhs_strides, rhs_phys) = layout_matrix(k, n, &rhs_logical, rhs_transposed);

        let (lhs_shape, rhs_shape, out_shape): (Vec<i64>, Vec<i64>, Vec<i64>) = if batched {
            (vec![bdim, m, k], vec![bdim, k, n], vec![bdim, m, n])
        } else {
            (vec![m, k], vec![k, n], vec![m, n])
        };
        let out_strides = if batched {
            vec![m * n, n, 1]
        } else {
            vec![n, 1]
        };
        let rank = lhs_shape.len();

        let lhs_d = desc(&lhs_shape, &lhs_strides, ElementKind::F32);
        let rhs_d = desc(&rhs_shape, &rhs_strides, ElementKind::F32);
        let out_d = desc(&out_shape, &out_strides, ElementKind::F32);

        // The permutation `classify_mat_layout` itself derives for "swap the
        // last two axes, batch (if any) stays outermost": [1,0] at rank-2,
        // [0,2,1] at rank-3 (hand-derived and cross-checked against
        // `classify_mat_layout`'s algorithm for these exact strides). Not
        // cross-checked BY `build_plan` (a caller precondition per the
        // module docs — `read_strided` ignores it after Edit 1 regardless),
        // but kept truthful to what a real planner would emit.
        let perm_for = |transposed: bool| -> View {
            if !transposed {
                View::Identity
            } else if rank == 2 {
                View::Permute { perm: vec![1, 0] }
            } else {
                View::Permute {
                    perm: vec![0, 2, 1],
                }
            }
        };
        let views = vec![perm_for(lhs_transposed), perm_for(rhs_transposed)];

        let axes = if batched {
            ContractionAxes::batched_matmul()
        } else {
            ContractionAxes::matmul()
        };
        let op =
            OpDef::contraction("matmul", &[ElementKind::F32], axes, reduced(0)).with_views(views);
        let kk = key(OpCategory::Gemm, &[lhs_d, rhs_d, out_d]);
        let plan = build_plan(&op, &kk);

        let lhs_buf = f32b(&lhs_shape, &lhs_phys);
        let rhs_buf = f32b(&rhs_shape, &rhs_phys);
        let got = f32s(&evaluate(&plan, &[lhs_d, rhs_d, out_d], &[lhs_buf, rhs_buf], &[])[0]);

        // From-scratch reference, K ascending — independent of `eval_contraction`.
        let mut want = vec![0f32; (bdim * m * n) as usize];
        for bi in 0..bdim {
            for mi in 0..m {
                for ni in 0..n {
                    let mut acc = 0f32;
                    for ki in 0..k {
                        let a = lhs_logical[bi as usize][(mi * k + ki) as usize];
                        let b = rhs_logical[bi as usize][(ki * n + ni) as usize];
                        acc += a * b;
                    }
                    want[(bi * m * n + mi * n + ni) as usize] = acc;
                }
            }
        }
        assert_eq!(
            got, want,
            "contraction matrix cell mismatch: batched={batched} \
             lhs_transposed={lhs_transposed} rhs_transposed={rhs_transposed}"
        );
    }

    #[test]
    fn oracle_contraction_layout_batch_differential_matrix() {
        for &batched in &[false, true] {
            for &lhs_t in &[false, true] {
                for &rhs_t in &[false, true] {
                    assert_contraction_matches_reference(batched, lhs_t, rhs_t);
                }
            }
        }
    }

    // --- Task 9: broadcast-KV (GQA) differential cells ----------------------
    //
    // View-requirement probe (empirical, run before writing the fixtures
    // below): a throwaway `build_plan` call with the Cell-1 broadcast rhs
    // key/operands and a VIEW-FREE op (no `.with_views(...)`) succeeded —
    // `assert_valid_views` returns immediately when `op.views.is_empty()`
    // (plan.rs:1066), and `build_plan`'s `Access::Contraction` arm only
    // requires `key.contraction.is_some()` (plan.rs:369), which
    // `derive_contraction` now supplies via `classify_mat_layout`'s stride-0
    // admission (Task 8) — no view of any kind is consulted. Independently,
    // `eval_contraction` calls `read_strided(.., i, &coords, None)` for BOTH
    // lhs (index 0) and rhs (index 1) with a HARD-CODED `None` perm
    // (oracle.rs:1329-1330) — `input_perm` is only ever applied to a fused
    // BIAS operand (index >= 2) inside the epilogue, never to lhs/rhs — so a
    // `View::Permute`/`View::Broadcast` on lhs/rhs would not even be read by
    // the oracle. The same is true for the transposed-broadcast Cell 2: its
    // rhs strides `[0, 1, k]` are read directly off `OperandDesc::strides` by
    // logical coordinate, with no view involved. Conclusion: BOTH cells are
    // VIEW-FREE — no `.with_views(...)` call on either `OpDef`.

    /// One broadcast-KV (GQA) matrix cell: rhs is LOGICAL `[B,K,N]` with a
    /// stride-0 BATCH axis — the physical buffer holds exactly ONE `K·N`
    /// slice, read for every batch index (`bi*0 == 0`). lhs is a REAL batched
    /// `[B,M,K]` dense operand (GQA broadcasts KV=rhs; broadcast-LHS is not a
    /// real use case and is out of scope). Built through the REAL
    /// `build_plan`/`evaluate` pipeline — constructible since Task 8 made
    /// `classify_mat_layout` admit the stride-0 rhs batch axis — and checked
    /// against a from-scratch reference (K ascending) that reads the SAME
    /// rhs slice for every `bi`. `rhs_transposed=false` -> canonical `[K,N]`
    /// rhs slice (Cell 1, strides `[0,n,1]`); `rhs_transposed=true` ->
    /// physically `[N,K]` per slice (Cell 2, the real Kᵀ+GQA case, strides
    /// `[0,1,k]`). Reuses `layout_matrix` (already hand-verified in
    /// `layout_matrix_matches_hand_verified_fixtures` above) for the K,N
    /// slice's physical strides/data, then prepends the broadcast batch
    /// stride `0`.
    fn assert_broadcast_kv_matches_reference(rhs_transposed: bool) {
        use crate::ir::{ContractionAxes, reduced};
        let (b, m, k, n): (i64, i64, i64, i64) = (2, 2, 3, 2);

        // lhs: REAL batched [B,M,K] dense, distinct per-batch small integers.
        let lhs_logical: Vec<Vec<f32>> = (0..b)
            .map(|bi| (0..m * k).map(|idx| (bi * 100 + idx + 1) as f32).collect())
            .collect();
        let lhs_strides = vec![m * k, k, 1];
        let lhs_phys: Vec<f32> = lhs_logical.iter().flatten().copied().collect();

        // rhs: ONE [K,N] logical slice (not one per batch), broadcast via a
        // stride-0 batch axis prepended to the K,N strides `layout_matrix`
        // derives for the chosen orientation.
        let rhs_logical_slice: Vec<f32> = (0..k * n).map(|idx| (idx + 10) as f32).collect();
        let (kn_strides, rhs_phys) = layout_matrix(
            k,
            n,
            std::slice::from_ref(&rhs_logical_slice),
            rhs_transposed,
        );
        let mut rhs_strides = vec![0i64];
        rhs_strides.extend(kn_strides);

        let lhs_shape = vec![b, m, k];
        let rhs_shape = vec![b, k, n];
        let out_shape = vec![b, m, n];
        let out_strides = vec![m * n, n, 1];

        let lhs_d = desc(&lhs_shape, &lhs_strides, ElementKind::F32);
        let rhs_d = desc(&rhs_shape, &rhs_strides, ElementKind::F32);
        let out_d = desc(&out_shape, &out_strides, ElementKind::F32);

        // View-free (see the probe note above): no `.with_views(...)` call.
        let op = OpDef::contraction(
            "matmul",
            &[ElementKind::F32],
            ContractionAxes::batched_matmul(),
            reduced(0),
        );
        let kk_key = key(OpCategory::Gemm, &[lhs_d, rhs_d, out_d]);
        let plan = build_plan(&op, &kk_key);

        let lhs_buf = f32b(&lhs_shape, &lhs_phys);
        // Buffer shape is a storage convenience (only `.bytes`/`.base_offset`
        // are read by `read_strided`; the OperandDesc `rhs_d` above supplies
        // the strides that actually drive address math) — [k, n] holds
        // exactly the one slice's `k*n` elements, matching Task 6's
        // convention of sizing the buffer to the logical (here: per-slice)
        // element count.
        let rhs_buf = f32b(&[k, n], &rhs_phys);
        let got = f32s(&evaluate(&plan, &[lhs_d, rhs_d, out_d], &[lhs_buf, rhs_buf], &[])[0]);

        // From-scratch reference, independent of `eval_contraction`: the SAME
        // rhs slice for every batch, K ascending.
        let mut want = vec![0f32; (b * m * n) as usize];
        for bi in 0..b {
            for mi in 0..m {
                for ni in 0..n {
                    let mut acc = 0f32;
                    for ki in 0..k {
                        let a = lhs_logical[bi as usize][(mi * k + ki) as usize];
                        let rv = rhs_logical_slice[(ki * n + ni) as usize];
                        acc += a * rv;
                    }
                    want[(bi * m * n + mi * n + ni) as usize] = acc;
                }
            }
        }
        assert_eq!(
            got, want,
            "broadcast-KV cell mismatch: rhs_transposed={rhs_transposed} \
             (the SAME rhs slice must be read for every batch)"
        );
    }

    // Cell 1: broadcast-batch rhs, canonical [K,N] storage order (strides
    // [0,n,1]) — the plain GQA broadcast-KV case.
    #[test]
    fn oracle_broadcast_kv_canonical_rhs_matches_reference() {
        assert_broadcast_kv_matches_reference(false);
    }

    // Cell 2: broadcast-batch rhs, transposed [K,N] storage order (physically
    // [N,K] per slice, strides [0,1,k]) — the real Kᵀ+GQA cell.
    #[test]
    fn oracle_broadcast_kv_transposed_rhs_matches_reference() {
        assert_broadcast_kv_matches_reference(true);
    }

    // --- compare helper -----------------------------------------------------

    #[test]
    fn compare_bit_exact_and_tolerant() {
        let a = f32b(&[2], &[1.0, 2.0]);
        let b = f32b(&[2], &[1.0, 2.0]);
        assert!(compare(&a, &b, Fidelity::BitExact).is_ok());
        // A difference above the f32 ULP at 2.0 (~2.4e-7) so the bits truly differ.
        let c = f32b(&[2], &[1.0, 2.0 + 1e-6]);
        assert!(compare(&a, &c, Fidelity::BitExact).is_err());
        assert!(
            compare(
                &a,
                &c,
                Fidelity::Tolerant {
                    rel: 1e-5,
                    abs: 1e-5
                }
            )
            .is_ok()
        );
        // both-NaN equal under tolerance.
        let n1 = f32b(&[1], &[f32::NAN]);
        let n2 = f32b(&[1], &[f32::NAN]);
        assert!(compare(&n1, &n2, Fidelity::Tolerant { rel: 0.0, abs: 0.0 }).is_ok());
    }

    // --- extra layout: permuted (transpose) read ----------------------------

    #[test]
    fn elementwise_permute_transpose() {
        use crate::ir::View;
        // Read input 0 through a transpose view: iteration [2,3] reads producer [3,2].
        let mut op = OpDef::elementwise("t", 1, &[ElementKind::F32], input(0) + konst(0.0));
        op.views = vec![View::Permute { perm: vec![1, 0] }];
        // producer storage is [3,2] row-major, strides [2,1]; iteration shape [2,3].
        let a = desc(&[3, 2], &[2, 1], ElementKind::F32);
        let outd = desc(&[2, 3], &[3, 1], ElementKind::F32);
        let k = key(OpCategory::UnaryElementwise, &[a, outd]);
        let plan = build_plan(&op, &k);
        let ops = [a, outd];
        // producer[3][2] = [[1,2],[3,4],[5,6]] laid out row-major.
        let ins = [f32b(&[3, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])];
        let out = evaluate(&plan, &ops, &ins, &[]);
        // out[i][j] = producer[j][i]: row0 = [1,3,5], row1 = [2,4,6].
        assert_eq!(f32s(&out[0]), vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
    }

    // (`reduction_outer_axis` retired here — outer-axis fold is now referenced
    // against kiss-ref in `kiss_ref_diff::tests::reduce_outer_axis_through_kiss_ref`.)

    // --- Adversarial-review regression kills (5 confirmed defects) -----------

    // NOTE (oracle→kiss-ref consolidation, closeout): the compute-dtype
    // cmp/select VALUE test `cmp_and_select_decide_in_compute_dtype` was RETIRED.
    // kiss-ref is monomorphic on the compute dtype (`eval_op::<f32>`,
    // resolve.rs:67), so it narrows the const into the f32 lane and decides the
    // compare in f32 — matching the kernel (and Baracuda's oracle fix). VERIFIED
    // + asserted against kiss-ref via the converter in
    // `kiss_ref_diff::tests::cmp_select_decide_in_compute_dtype_through_kiss_ref`
    // (the previously-unproven cmp/select-const converter path — no divergence).
    // The compute-dtype `sign` underflow + F32Strict `nextafter` tests below STAY:
    // the former is a compute-dtype-storage nuance kept on the oracle; the latter
    // rides `F32Strict`, a Baracuda compute-precision distinction kiss-ref (single
    // F32) does not model.

    // KILL (major): Sign/Step decide on the compute-dtype-rounded value. sign(a*b)
    // with a=b=1e-30f: the product underflows to 0.0f in f32 → sign 0; the pre-fix
    // oracle kept 1e-60 in f64 → sign 1.
    #[test]
    fn sign_of_underflowing_product_is_zero_in_f32() {
        let op = OpDef::elementwise(
            "sgn",
            2,
            &[ElementKind::F32],
            (input(0) * input(1)).unary(UnaryOp::Sign),
        );
        let a = desc(&[1], &[1], ElementKind::F32);
        let k = key(OpCategory::BinaryElementwise, &[a, a, a]);
        let plan = build_plan(&op, &k);
        let ops = [a, a, a];
        let ins = [f32b(&[1], &[1e-30]), f32b(&[1], &[1e-30])];
        let out = evaluate(&plan, &ops, &ins, &[]);
        assert_eq!(
            f32s(&out[0]),
            vec![0.0],
            "product underflows to 0.0f → sign 0"
        );
    }

    // KILL (major): F32Strict Nextafter steps the f32 lattice (not f64, which would
    // demote-round back to a silent no-op). next(1.0f → 2.0f) = the next f32.
    #[test]
    fn nextafter_f32strict_steps_the_f32_lattice() {
        let op = OpDef::elementwise(
            "naft",
            1,
            &[ElementKind::F32Strict],
            input(0).binary(BinaryOp::Nextafter, konst(2.0)),
        );
        let a = desc(&[1], &[1], ElementKind::F32Strict);
        let k = key(OpCategory::UnaryElementwise, &[a, a]);
        let plan = build_plan(&op, &k);
        let ops = [a, a];
        let ins = [f32b(&[1], &[1.0])];
        let out = evaluate(&plan, &ops, &ins, &[]);
        assert_eq!(
            bit32(&out[0], 0),
            1.0f32.to_bits() + 1,
            "F32Strict nextafter must step one f32 ULP, not no-op"
        );
    }

    // KILL (major): the Tolerant comparator rejects a non-equal infinity instead of
    // letting the infinite relative band swallow it. An output overflowing to +inf
    // must NOT silently match 0.0 / -inf; an equal +inf still passes.
    #[test]
    fn tolerant_compare_rejects_nonequal_infinity() {
        let tol = Fidelity::Tolerant {
            rel: 1e-5,
            abs: 1e-6,
        };
        let inf = f32b(&[1], &[f32::INFINITY]);
        assert!(
            compare(&inf, &f32b(&[1], &[0.0]), tol).is_err(),
            "+inf vs 0"
        );
        assert!(
            compare(&inf, &f32b(&[1], &[f32::NEG_INFINITY]), tol).is_err(),
            "+inf vs -inf (sign bug) must fail"
        );
        assert!(
            compare(&inf, &f32b(&[1], &[1e30]), tol).is_err(),
            "+inf vs finite must fail"
        );
        assert!(compare(&inf, &inf, tol).is_ok(), "equal +inf still passes");
    }

    // KILL (minor): integer Mean is out of scope — panic (mirroring the emitter),
    // not silently return the un-divided sum.
    #[test]
    #[should_panic(expected = "integer Mean")]
    fn integer_mean_reduction_panics() {
        let op = OpDef::reduction("imean", 1, &[ElementKind::I32], input(0), ReduceOp::Mean);
        let ind = desc(&[1, 4], &[4, 1], ElementKind::I32);
        let outd = desc(&[1], &[1], ElementKind::I32);
        let k = key(OpCategory::Reduction, &[ind, outd]);
        let plan = build_plan(&op, &k);
        let ops = [ind, outd];
        let ins = [TypedBuffer::from_i32(&[1, 4], &[2, 4, 6, 8])];
        let _ = evaluate(&plan, &ops, &ins, &[]); // must panic, not return sum=20
    }

    // KILL (minor): erf/erfc/gelu are accurate (libm ~1 ULP), tight enough to be a
    // real reference for an f32/f64 erf/gelu kernel. The v1 ~1e-7 rational fails
    // this ~1e-12 pin. gelu(1) = Φ(1); erf+erfc = 1.
    #[test]
    fn erf_family_is_accurate_f64() {
        assert!((erf(1.0) - 0.842_700_792_949_714_9).abs() < 1e-12, "erf(1)");
        assert!(
            (erfc(1.0) - 0.157_299_207_050_285_13).abs() < 1e-12,
            "erfc(1)"
        );
        assert!((erf(1.0) + erfc(1.0) - 1.0).abs() < 1e-15, "erf+erfc=1");
        assert!(
            (unary_op_f64(UnaryOp::Gelu, 1.0) - 0.841_344_746_068_542_9).abs() < 1e-12,
            "gelu(1) = Phi(1)"
        );
    }
}
