// SPDX-License-Identifier: MIT OR Apache-2.0
//! Mode selection, slicing and tiling for [`Layout`], split from the parent
//! module for size.

use super::{IntTuple, Layout, LayoutError};

/// A coordinate pattern for [`Layout::slice`]: fix some positions, keep others.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub enum SliceCrd {
    /// Keep this mode (CuTe's underscore).
    All,
    /// Fix this mode at a coordinate. Against a nested mode the integer is a
    /// linear index into that mode.
    At(i64),
    /// One pattern per element of a tuple mode.
    Tuple(Vec<SliceCrd>),
}

impl Layout {
    /// The layout of the top-level modes at `indices`, in that order (repeats
    /// allowed). Always a tuple, even for one index.
    pub fn select(&self, indices: &[usize]) -> Result<Layout, LayoutError> {
        let modes = self.modes();
        let picked = indices
            .iter()
            .map(|&i| modes.get(i).cloned().ok_or(LayoutError::OutOfRange))
            .collect::<Result<Vec<_>, _>>()?;
        Layout::from_modes(&picked)
    }

    /// The top-level modes `begin..end`. Fails unless `begin < end <= rank`.
    pub fn take(&self, begin: usize, end: usize) -> Result<Layout, LayoutError> {
        if begin >= end || end > self.modes().len() {
            return Err(LayoutError::OutOfRange);
        }
        self.select(&(begin..end).collect::<Vec<_>>())
    }

    /// Fix the coordinates `pat` pins and keep the rest: returns the layout of
    /// the kept modes (profile preserved, fixed modes dropped; `1:0` when none
    /// are kept, a bare mode when exactly one top-level mode is kept) and the
    /// offset the fixed coordinates contribute, so that
    /// `kept(j) + offset == self(full coordinate)`.
    pub fn slice(&self, pat: &SliceCrd) -> Result<(Layout, i64), LayoutError> {
        let (kept, off) = slice_node(pat, &self.shape, &self.stride)?;
        let layout = match kept {
            Some((s, d)) => Layout::new(s, d)?,
            None => Layout::leaf(1, 0)?,
        };
        Ok((layout, off))
    }

    /// Replicate `self` (a block) to cover `target`, one extent per top-level
    /// mode: mode `i` of the result is `(block_i, tiles_i)` with
    /// `size = target[i]`, and tiles are laid out after the whole block
    /// (`logical_product` with a column-major tile grid). Each `target[i]` must
    /// be a multiple of `size(block_i)` ([`LayoutError::NotDivisible`]).
    pub fn tile_to_shape(&self, target: &[i64]) -> Result<Layout, LayoutError> {
        let blocks = self.modes();
        if blocks.len() != target.len() {
            return Err(LayoutError::ProfileMismatch);
        }
        let mut counts = Vec::with_capacity(target.len());
        for (b, &t) in blocks.iter().zip(target) {
            let sz = b.size();
            if t < 1 || t % sz != 0 {
                return Err(LayoutError::NotDivisible);
            }
            counts.push(t / sz);
        }
        let grid = if counts.len() == 1 {
            Layout::col_major(IntTuple::Int(counts[0]))?
        } else {
            Layout::col_major(IntTuple::ints(counts))?
        };
        let rest = self.logical_product(&grid)?.modes().remove(1);
        if blocks.len() == 1 {
            return Layout::from_modes(&[Layout::from_modes(&[blocks[0].clone(), rest])?]);
        }
        let tiles = rest.modes();
        let per_dim = blocks
            .iter()
            .zip(&tiles)
            .map(|(b, t)| Layout::from_modes(&[b.clone(), t.clone()]))
            .collect::<Result<Vec<_>, _>>()?;
        Layout::from_modes(&per_dim)
    }
}

type Kept = Option<(IntTuple, IntTuple)>;

fn slice_node(
    pat: &SliceCrd,
    shape: &IntTuple,
    stride: &IntTuple,
) -> Result<(Kept, i64), LayoutError> {
    match (pat, shape, stride) {
        (SliceCrd::All, _, _) => Ok((Some((shape.clone(), stride.clone())), 0)),
        (SliceCrd::At(c), _, _) => {
            let sub = Layout {
                shape: shape.clone(),
                stride: stride.clone(),
            };
            if *c < 0 || *c >= sub.size() {
                return Err(LayoutError::OutOfRange);
            }
            Ok((None, sub.call(*c)?))
        }
        (SliceCrd::Tuple(ps), IntTuple::Tuple(ss), IntTuple::Tuple(ds)) if ps.len() == ss.len() => {
            let (mut s_out, mut d_out, mut off) = (Vec::new(), Vec::new(), 0i64);
            for ((p, s), d) in ps.iter().zip(ss).zip(ds) {
                let (kept, o) = slice_node(p, s, d)?;
                off = off.checked_add(o).ok_or(LayoutError::Overflow)?;
                if let Some((ks, kd)) = kept {
                    s_out.push(ks);
                    d_out.push(kd);
                }
            }
            let kept = match s_out.len() {
                0 => None,
                1 => Some((s_out.remove(0), d_out.remove(0))),
                _ => Some((IntTuple::Tuple(s_out), IntTuple::Tuple(d_out))),
            };
            Ok((kept, off))
        }
        _ => Err(LayoutError::ProfileMismatch),
    }
}
