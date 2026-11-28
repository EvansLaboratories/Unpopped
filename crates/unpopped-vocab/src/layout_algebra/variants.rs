// SPDX-License-Identifier: MIT OR Apache-2.0
//! By-mode `coalesce`, the by-mode `logical_divide` and `logical_product`, and their `zipped`, `tiled`,
//! `flat`, `blocked` and `raked` regroupings, for [`Layout`] (IR-union rows 7, 10, 11 of `cute.md`).
//!
//! Source: the names and the groupings are CuTe's (NVIDIA CUTLASS `include/cute/layout.hpp`;
//! `media/docs/cpp/cute/02_layout_algebra.md`). The headers were not re-read for this module, so **each
//! grouping is UNVERIFIED against `cute/layout.hpp`** (in particular, what a `flat_*` does with a mode that is
//! itself nested: here the modes are kept as they are) and is pinned by `tests/layout_algebra_variants.rs`:
//! hand-computed examples, plus a brute-force reading of every coordinate through the per-mode definition.
//!
//! Every operation applies the existing single-mode [`Layout::logical_divide`] or [`Layout::logical_product`]
//! to each top-level mode and only regroups the results, so none of them can disagree with those two on a mode.

use super::{IntTuple, Layout, LayoutError};

/// Per top-level mode: its `(first, second)` pair, the two modes of the rank-2 result of `f` on that mode.
fn pairs(
    this: &Layout,
    other: &[Layout],
    f: impl Fn(&Layout, &Layout) -> Result<Layout, LayoutError>,
) -> Result<Vec<(Layout, Layout)>, LayoutError> {
    let modes = this.modes();
    if modes.len() != other.len() {
        return Err(LayoutError::ProfileMismatch);
    }
    modes
        .iter()
        .zip(other)
        .map(|(m, o)| {
            let r = f(m, o)?;
            let mut it = r.modes().into_iter();
            match (it.next(), it.next(), it.next()) {
                (Some(a), Some(b), None) => Ok((a, b)),
                _ => Err(LayoutError::ProfileMismatch),
            }
        })
        .collect()
}

fn group(parts: &[Layout]) -> Result<Layout, LayoutError> {
    Layout::from_modes(parts)
}

/// `((first_0, first_1, ..), (second_0, second_1, ..))`.
fn zipped(p: &[(Layout, Layout)]) -> Result<Layout, LayoutError> {
    let (a, b): (Vec<_>, Vec<_>) = p.iter().cloned().unzip();
    group(&[group(&a)?, group(&b)?])
}

/// `((first_0, first_1, ..), second_0, second_1, ..)`.
fn tiled(p: &[(Layout, Layout)]) -> Result<Layout, LayoutError> {
    let a: Vec<_> = p.iter().map(|(a, _)| a.clone()).collect();
    let mut all = vec![group(&a)?];
    all.extend(p.iter().map(|(_, b)| b.clone()));
    group(&all)
}

/// `(first_0, first_1, .., second_0, second_1, ..)`.
fn flat(p: &[(Layout, Layout)]) -> Result<Layout, LayoutError> {
    let mut all: Vec<Layout> = p.iter().map(|(a, _)| a.clone()).collect();
    all.extend(p.iter().map(|(_, b)| b.clone()));
    group(&all)
}

/// `((first_0, second_0), (first_1, second_1), ..)`, or with each pair swapped when `swap`.
fn by_mode(p: &[(Layout, Layout)], swap: bool) -> Result<Layout, LayoutError> {
    let modes: Vec<Layout> = p
        .iter()
        .map(|(a, b)| {
            if swap {
                group(&[b.clone(), a.clone()])
            } else {
                group(&[a.clone(), b.clone()])
            }
        })
        .collect::<Result<_, _>>()?;
    group(&modes)
}

impl Layout {
    /// Coalesce by profile: a leaf of `profile` coalesces the sub-layout at that position, a tuple recurses
    /// mode by mode (so `(2,3,4,5):(1,2,6,24)` with profile `(1,1,1,1)` is unchanged in rank where
    /// `coalesce` would merge them). [`LayoutError::ProfileMismatch`] when a tuple of `profile` does not have
    /// one entry per mode at its level.
    pub fn coalesce_profile(&self, profile: &IntTuple) -> Result<Layout, LayoutError> {
        match profile {
            IntTuple::Int(_) => Ok(self.coalesce()),
            IntTuple::Tuple(ps) => {
                if !matches!(self.shape(), IntTuple::Tuple(_)) || self.modes().len() != ps.len() {
                    return Err(LayoutError::ProfileMismatch);
                }
                let modes: Vec<Layout> = self
                    .modes()
                    .iter()
                    .zip(ps)
                    .map(|(m, p)| m.coalesce_profile(p))
                    .collect::<Result<_, _>>()?;
                group(&modes)
            }
        }
    }

    /// `logical_divide` of each top-level mode by the matching tiler: a layout of the same rank whose mode `i`
    /// is `(tile_i, rest_i)`. [`LayoutError::ProfileMismatch`] unless there is one tiler per mode.
    pub fn logical_divide_by_mode(&self, tilers: &[Layout]) -> Result<Layout, LayoutError> {
        by_mode(&pairs(self, tilers, Layout::logical_divide)?, false)
    }

    /// `((tile_0, tile_1, ..), (rest_0, rest_1, ..))` (CuTe `zipped_divide`).
    ///
    /// **A tile that does not divide its mode is accepted**, as CuTe's primitive does: the number of tiles is rounded up, so the
    /// last tile's image reaches past the layout (4 over 6 reaches offset 7 of 6). A caller that needs an exact tiling uses
    /// [`Layout::zipped_divide_exact`], which refuses it.
    pub fn zipped_divide(&self, tilers: &[Layout]) -> Result<Layout, LayoutError> {
        zipped(&pairs(self, tilers, Layout::logical_divide)?)
    }

    /// `((tile_0, tile_1, ..), rest_0, rest_1, ..)` (CuTe `tiled_divide`).
    pub fn tiled_divide(&self, tilers: &[Layout]) -> Result<Layout, LayoutError> {
        tiled(&pairs(self, tilers, Layout::logical_divide)?)
    }

    /// `(tile_0, tile_1, .., rest_0, rest_1, ..)` (CuTe `flat_divide`).
    pub fn flat_divide(&self, tilers: &[Layout]) -> Result<Layout, LayoutError> {
        flat(&pairs(self, tilers, Layout::logical_divide)?)
    }

    /// `logical_product` of each top-level mode with the matching mode of `rhs`: mode `i` is `(self_i, copies_i)`
    /// (CuTe `blocked_product`). [`LayoutError::ProfileMismatch`] unless the two have the same rank.
    pub fn blocked_product(&self, rhs: &Layout) -> Result<Layout, LayoutError> {
        by_mode(&pairs(self, &rhs.modes(), Layout::logical_product)?, false)
    }

    /// Mode `i` is `(copies_i, self_i)`: the copies interleave inside each block (CuTe `raked_product`).
    pub fn raked_product(&self, rhs: &Layout) -> Result<Layout, LayoutError> {
        by_mode(&pairs(self, &rhs.modes(), Layout::logical_product)?, true)
    }

    /// `((self_0, self_1, ..), (copies_0, copies_1, ..))` (CuTe `zipped_product`).
    pub fn zipped_product(&self, rhs: &Layout) -> Result<Layout, LayoutError> {
        zipped(&pairs(self, &rhs.modes(), Layout::logical_product)?)
    }

    /// `((self_0, self_1, ..), copies_0, copies_1, ..)` (CuTe `tiled_product`).
    pub fn tiled_product(&self, rhs: &Layout) -> Result<Layout, LayoutError> {
        tiled(&pairs(self, &rhs.modes(), Layout::logical_product)?)
    }

    /// `(self_0, self_1, .., copies_0, copies_1, ..)` (CuTe `flat_product`).
    pub fn flat_product(&self, rhs: &Layout) -> Result<Layout, LayoutError> {
        flat(&pairs(self, &rhs.modes(), Layout::logical_product)?)
    }
}
