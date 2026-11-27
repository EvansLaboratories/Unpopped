// SPDX-License-Identifier: MIT OR Apache-2.0
//! The XOR [`Swizzle`] address permutation and [`ComposedLayout`], split from
//! the parent module for size.

use super::{Layout, LayoutError};

/// The XOR address permutation `Swizzle<B,M,S>`: `bits` (B) address bits starting
/// at `base`+max(`shift`,0) are XORed into the `bits` bits `|shift|` positions
/// away, leaving the low `base` bits alone. It is a bijection on non-negative
/// integers and its own inverse. Used to spread shared-memory rows across banks.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct Swizzle {
    bits: u32,
    base: u32,
    shift: i32,
}

impl Swizzle {
    /// Build a swizzle. Requires `|shift| >= bits` (so the source and target bit
    /// groups cannot overlap, which is what makes it an involution) and
    /// `bits + base + |shift| <= 62` (so every intermediate stays in `i64`).
    pub fn new(bits: u32, base: u32, shift: i32) -> Result<Self, LayoutError> {
        let a = shift.unsigned_abs();
        if a < bits || u64::from(bits) + u64::from(base) + u64::from(a) > 62 {
            return Err(LayoutError::BadSwizzle);
        }
        Ok(Self { bits, base, shift })
    }

    /// Number of bits XORed (B).
    pub fn bits(&self) -> u32 {
        self.bits
    }

    /// Number of untouched low bits (M).
    pub fn base(&self) -> u32 {
        self.base
    }

    /// Signed distance between source and target bit groups (S).
    pub fn shift(&self) -> i32 {
        self.shift
    }

    /// The same permutation read over **elements** of `width_bytes` bytes, given that `self` acts on the **byte** address
    /// (the PTX form, `Swizzle<B,4,3>`): `self.apply(i * w) == w * self.in_elements(w).apply(i)` for every `i >= 0`.
    /// A byte swizzle with `base = M` leaves the low `M` bits alone, so for `w = 2^k` with `k <= M` the element swizzle is
    /// the one with `base = M - k` and the same `bits` and `shift`. Declined with [`LayoutError::NotDivisible`] when `w`
    /// is not a power of two (a byte address is not a whole number of elements' bits), and with
    /// [`LayoutError::BadSwizzle`] when `k > M` (the swizzle would mix bits inside one element).
    /// CuTe reaches the same form by upcasting a layout atom; that rule was not read (**UNVERIFIED** against
    /// `cute/layout.hpp` and `cute/swizzle.hpp`), so this is pinned by the identity above, not by CuTe.
    pub fn in_elements(&self, width_bytes: u32) -> Result<Swizzle, LayoutError> {
        if width_bytes == 0 || !width_bytes.is_power_of_two() {
            return Err(LayoutError::NotDivisible);
        }
        let k = width_bytes.trailing_zeros();
        if k > self.base {
            return Err(LayoutError::BadSwizzle);
        }
        Swizzle::new(self.bits, self.base - k, self.shift)
    }

    /// Apply the swizzle to a non-negative offset.
    pub fn apply(&self, x: i64) -> Result<i64, LayoutError> {
        if x < 0 {
            return Err(LayoutError::OutOfRange);
        }
        let msk = (1i64 << self.bits) - 1;
        let src = msk << (self.base + self.shift.max(0).unsigned_abs());
        let moved = if self.shift >= 0 {
            (x & src) >> self.shift
        } else {
            (x & src) << self.shift.unsigned_abs()
        };
        Ok(x ^ moved)
    }
}

/// A [`Swizzle`] applied after a [`Layout`]: `i -> swizzle(layout(i))`.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct ComposedLayout {
    swizzle: Swizzle,
    layout: Layout,
}

impl ComposedLayout {
    /// Apply `swizzle` to the offsets of `layout`.
    pub fn new(swizzle: Swizzle, layout: Layout) -> Self {
        Self { swizzle, layout }
    }

    /// The swizzle stage.
    pub fn swizzle(&self) -> Swizzle {
        self.swizzle
    }

    /// The underlying layout.
    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    /// The swizzled offset for linear index `idx`.
    pub fn call(&self, idx: i64) -> Result<i64, LayoutError> {
        self.swizzle.apply(self.layout.call(idx)?)
    }
}
