// SPDX-License-Identifier: MIT OR Apache-2.0
//! `make_ordered_layout`, `recast_layout`, `max_common_layout` and
//! `max_common_vector` for [`Layout`].
//!
//! Source: the names and intent are those of CuTe (NVIDIA CUTLASS
//! `include/cute/layout.hpp`; `media/docs/cpp/cute/02_layout_algebra.md`). The
//! headers were not re-read for this module, so **each operation here is
//! UNVERIFIED against `cute/layout.hpp`** and is pinned only by the brute-force
//! reading in `tests/layout_algebra_ordered.rs`. `recast_layout` is a
//! restricted form (see its doc); it declines what CuTe's general rule would
//! accept rather than guess it. `make_identity_layout` is not here: CuTe's is a
//! layout with basis strides (`E<0>`, `E<1>`), which this module's integer
//! strides cannot hold (the module header's "TMA basis strides").

use super::{IntTuple, Layout, LayoutError};

fn scale_strides(t: &IntTuple, k: i64) -> Result<IntTuple, LayoutError> {
    match t {
        IntTuple::Int(d) => Ok(IntTuple::Int(
            d.checked_mul(k).ok_or(LayoutError::Overflow)?,
        )),
        IntTuple::Tuple(v) => Ok(IntTuple::Tuple(
            v.iter()
                .map(|c| scale_strides(c, k))
                .collect::<Result<_, _>>()?,
        )),
    }
}

impl Layout {
    /// The compact layout over the top-level modes of `shape` whose strides are
    /// assigned by `order`: the mode with the smallest `order` value gets
    /// stride 1, the next the product of the extents before it, and so on.
    /// `make_ordered_layout((2,3,4), (1,0,2))` is `(2,3,4):(3,1,6)`. A nested
    /// mode is laid out column-major inside its slot.
    /// [`LayoutError::ProfileMismatch`] when `order` does not have one entry per
    /// top-level mode or repeats a value.
    pub fn make_ordered_layout(shape: &IntTuple, order: &[i64]) -> Result<Layout, LayoutError> {
        let modes: Vec<IntTuple> = match shape {
            IntTuple::Tuple(v) => v.clone(),
            IntTuple::Int(_) => vec![shape.clone()],
        };
        let mut rank: Vec<usize> = (0..modes.len()).collect();
        rank.sort_by_key(|&i| order.get(i).copied());
        let distinct = (0..order.len()).all(|i| (0..i).all(|j| order[i] != order[j]));
        if order.len() != modes.len() || !distinct {
            return Err(LayoutError::ProfileMismatch);
        }
        let mut stride = vec![IntTuple::Int(0); modes.len()];
        let mut run: i64 = 1;
        for i in rank {
            let inner = Layout::col_major(modes[i].clone())?;
            stride[i] = scale_strides(inner.stride(), run)?;
            run = run.checked_mul(inner.size()).ok_or(LayoutError::Overflow)?;
        }
        if matches!(shape, IntTuple::Int(_)) {
            Layout::new(shape.clone(), stride.remove(0))
        } else {
            Layout::new(shape.clone(), IntTuple::Tuple(stride))
        }
    }

    /// Reinterpret a layout of `old_bits`-wide elements as a layout of
    /// `new_bits`-wide elements over the same bytes. Restricted form: needs a
    /// stride-1 mode, and
    /// - **widening** by `n = new/old`: that mode's extent must be divisible by
    ///   `n` and every other non-zero stride by `n`; extent and strides divide;
    /// - **narrowing** by `n = old/new`: that mode's extent and every other
    ///   non-zero stride multiply by `n`.
    ///
    /// A widening that leaves the stride-1 mode at extent 1 cannot be narrowed
    /// back (which mode was it?); the round trip holds while the extent stays above 1.
    ///
    /// Anything else is [`LayoutError::NotDivisible`] (including the cases CuTe's
    /// general rule might accept). Both widths must be positive and one must
    /// divide the other ([`LayoutError::NonPositiveCotarget`] / `NotDivisible`).
    pub fn recast_layout(&self, old_bits: i64, new_bits: i64) -> Result<Layout, LayoutError> {
        for w in [old_bits, new_bits] {
            if w < 1 {
                return Err(LayoutError::NonPositiveCotarget(w));
            }
        }
        let (wide, n) = if new_bits >= old_bits {
            (true, new_bits / old_bits)
        } else {
            (false, old_bits / new_bits)
        };
        if (wide && new_bits % old_bits != 0) || (!wide && old_bits % new_bits != 0) {
            return Err(LayoutError::NotDivisible);
        }
        let modes = self.flat_modes();
        let unit = modes
            .iter()
            .position(|&(s, d)| d == 1 && s > 1)
            .or_else(|| modes.iter().position(|&(_, d)| d == 1))
            .ok_or(LayoutError::NotDivisible)?;
        let mut shape = Vec::with_capacity(modes.len());
        let mut stride = Vec::with_capacity(modes.len());
        for (k, &(s, d)) in modes.iter().enumerate() {
            let (s2, d2) = match (k == unit, wide) {
                (true, true) if s % n == 0 => (s / n, d),
                (true, false) => (s.checked_mul(n).ok_or(LayoutError::Overflow)?, d),
                (false, true) if d == 0 => (s, 0),
                (false, true) if d % n == 0 => (s, d / n),
                (false, false) => (s, d.checked_mul(n).ok_or(LayoutError::Overflow)?),
                _ => return Err(LayoutError::NotDivisible),
            };
            shape.push(s2);
            stride.push(d2);
        }
        // Rebuild with the original profile.
        fn rebuild(t: &IntTuple, flat: &mut std::slice::Iter<'_, i64>) -> IntTuple {
            match t {
                IntTuple::Int(_) => IntTuple::Int(*flat.next().unwrap_or(&1)),
                IntTuple::Tuple(v) => IntTuple::Tuple(v.iter().map(|c| rebuild(c, flat)).collect()),
            }
        }
        let sh = rebuild(self.shape(), &mut shape.iter());
        let st = rebuild(self.stride(), &mut stride.iter());
        Layout::new(sh, st)
    }

    /// The number of leading logical indices `i` for which `self(b⁻¹(i)) == i`,
    /// where `b⁻¹ = other.right_inverse()`: how many consecutive elements the
    /// two layouts agree are contiguous. At least 1. Declines what
    /// [`Layout::composition`] declines.
    pub fn max_common_vector(&self, other: &Layout) -> Result<i64, LayoutError> {
        let common = self.composition(&other.right_inverse())?.coalesce();
        Ok(match common.flat_modes().first() {
            Some(&(s, 1)) => s,
            _ => 1,
        })
    }

    /// The layout `R` of size [`Layout::max_common_vector`] with
    /// `self(R(i)) == i` and `other(R(i)) == i` for every `i < size(R)`: the
    /// prefix of `other`'s right inverse. `1:0` when the common vector is 1.
    pub fn max_common_layout(&self, other: &Layout) -> Result<Layout, LayoutError> {
        let inv = other.right_inverse();
        let n = self.max_common_vector(other)?;
        if n == 1 {
            return Layout::leaf(1, 0);
        }
        inv.composition(&Layout::leaf(n, 1)?)
    }
}
