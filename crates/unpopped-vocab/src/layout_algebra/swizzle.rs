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
