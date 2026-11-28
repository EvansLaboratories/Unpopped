// SPDX-License-Identifier: MIT OR Apache-2.0
//! Tile and partition selection over [`Layout`]: pure vocabulary (`docs/design-tile-and-schedule-plan.md` T1b), no IR and no
//! kernel statement.
//!
//! The tile and the partition are the two halves of [`Layout::zipped_divide`], `((tile_0, ..), (rest_0, ..))`, with one half
//! pinned at a coordinate: [`Layout::inner_partition`] / [`Layout::local_tile`] pin the REST coordinate (which tile), and
//! [`Layout::outer_partition`] / [`Layout::local_partition`] pin the TILE coordinate (which element of every tile: the
//! strided share of one thread). That is CuTe's `local_tile = zipped_divide(.)(_, coord)` and
//! `outer_partition = zipped_divide(.)(coord, _)`; the groupings are worked from the definitions of `zipped_divide` and
//! `slice` here, UNVERIFIED against `cute/tensor_impl.hpp`. Each returns the kept layout and the offset the pinned
//! coordinate contributes, so that `kept(j) + offset == self(full coordinate)`.

use super::{IntTuple, Layout, LayoutError, SliceCrd};

/// The pattern that pins a mode at `coord`: one integer for one tiler, a tuple of integers otherwise.
fn pin(coord: &[i64]) -> SliceCrd {
    match coord {
        [c] => SliceCrd::At(*c),
        many => SliceCrd::Tuple(many.iter().map(|c| SliceCrd::At(*c)).collect()),
    }
}

impl Layout {
    /// Every tiler's size divides the size of its mode: `zipped_divide` would otherwise round the number of tiles up, and the
    /// last tile would reach past the layout. A tiling with a ragged edge is not expressed here (a mask over the identity
    /// tensor is the recipe). [`LayoutError::ProfileMismatch`] unless there is one tiler per mode.
    fn require_dividing_tiles(&self, tilers: &[Layout]) -> Result<(), LayoutError> {
        let modes = self.modes();
        if modes.len() != tilers.len() {
            return Err(LayoutError::ProfileMismatch);
        }
        for (m, t) in modes.iter().zip(tilers) {
            let tile = t.size();
            if tile <= 0 || m.size() % tile != 0 {
                return Err(LayoutError::NotDivisible);
            }
        }
        Ok(())
    }

    /// Pin the REST coordinate `coord` (one entry per tiler) of `zipped_divide(tilers)`: the layout of one tile and its
    /// offset. [`LayoutError::ProfileMismatch`] unless `coord` has one entry per tiler; [`LayoutError::OutOfRange`] for a
    /// coordinate past the number of tiles; [`LayoutError::NotDivisible`] when a tiler's size does not divide its mode (no tile is
    /// rounded up).
    pub fn inner_partition(
        &self,
        tilers: &[Layout],
        coord: &[i64],
    ) -> Result<(Layout, i64), LayoutError> {
        if coord.len() != tilers.len() {
            return Err(LayoutError::ProfileMismatch);
        }
        self.require_dividing_tiles(tilers)?;
        let z = self.zipped_divide(tilers)?;
        z.slice(&SliceCrd::Tuple(vec![SliceCrd::All, pin(coord)]))
    }

    /// Pin the TILE coordinate `coord` (one entry per tiler) of `zipped_divide(tilers)`: the layout of the elements at
    /// that position of every tile, and its offset. Errors as [`Layout::inner_partition`].
    pub fn outer_partition(
        &self,
        tilers: &[Layout],
        coord: &[i64],
    ) -> Result<(Layout, i64), LayoutError> {
        if coord.len() != tilers.len() {
            return Err(LayoutError::ProfileMismatch);
        }
        self.require_dividing_tiles(tilers)?;
        let z = self.zipped_divide(tilers)?;
        z.slice(&SliceCrd::Tuple(vec![pin(coord), SliceCrd::All]))
    }

    /// Tile number `coord` of the tiling by `tilers` (CuTe `local_tile`): [`Layout::inner_partition`].
    pub fn local_tile(
        &self,
        tilers: &[Layout],
        coord: &[i64],
    ) -> Result<(Layout, i64), LayoutError> {
        self.inner_partition(tilers, coord)
    }

    /// The share of thread `thread` (a linear index, column-major over the thread tile) when `thread_tilers` give the
    /// thread grid, one extent per mode (CuTe `local_partition`): [`Layout::outer_partition`] at the thread's coordinate.
    /// [`LayoutError::OutOfRange`] for a thread past the grid.
    pub fn local_partition(
        &self,
        thread_tilers: &[Layout],
        thread: i64,
    ) -> Result<(Layout, i64), LayoutError> {
        let extents: Vec<i64> = thread_tilers.iter().map(Layout::size).collect();
        let grid = IntTuple::ints(extents.iter().copied());
        let crd = grid.idx2crd(thread)?;
        let coord = crd.leaves();
        self.outer_partition(thread_tilers, &coord)
    }

    /// The same elements in the same order, re-expressed over `target`'s shape (a rank change): `self` composed with the
    /// column-major layout of `target`, so that `retile(target)(crd2idx(target, c)) == self(colex index of c)`.
    /// [`LayoutError::ProfileMismatch`] unless `size(target) == size(self)`; [`LayoutError::NotDivisible`] when the target's
    /// modes do not align with the layout's (no layout expresses that reshape). A pure layout function: it names no view and no
    /// statement.
    pub fn retile(&self, target: &IntTuple) -> Result<Layout, LayoutError> {
        if target.product()? != self.size() {
            return Err(LayoutError::ProfileMismatch);
        }
        self.composition(&Layout::col_major(target.clone())?)
    }
}
