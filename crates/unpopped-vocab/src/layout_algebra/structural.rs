// SPDX-License-Identifier: MIT OR Apache-2.0
//! Structural operations on the modes of a [`Layout`]: `group`, `append`,
//! `prepend`, `replace`, `dice`, `filter_zeros` and `filter`. None of them
//! changes what a layout computes at a coordinate except where stated; they
//! rearrange, add, swap or drop modes.
//!
//! Source: the names and meaning are those of the CuTe layout documentation
//! (NVIDIA CUTLASS `media/docs/cpp/cute/01_layout.md`, "Layout Manipulation:
//! group, append, prepend, replace, ..."). The CUTLASS headers were not
//! re-read for this module; the behaviour of `filter_zeros` and `filter` is
//! taken from that documentation's description ("all size-1 and stride-0 modes
//! removed") and is UNVERIFIED against `cute/layout.hpp`. Every operation is
//! checked against a brute-force reading in `tests/layout_algebra_structural.rs`.

use super::{IntTuple, Layout, LayoutError, SliceCrd};

impl Layout {
    /// Nest the top-level modes `begin..end` into one mode, leaving the others:
    /// `group(1, 3)` of `(a,b,c,d)` is `(a,(b,c),d)`. The index-to-offset map is
    /// unchanged (the flattened mode order is). A one-mode range
    /// (`end == begin + 1`) returns the layout unchanged, since a mode is
    /// already its own group. Fails with [`LayoutError::OutOfRange`] unless
    /// `begin < end <= rank`.
    pub fn group(&self, begin: usize, end: usize) -> Result<Layout, LayoutError> {
        let modes = self.modes();
        if begin >= end || end > modes.len() {
            return Err(LayoutError::OutOfRange);
        }
        if end - begin == 1 {
            return Ok(self.clone());
        }
        let inner = Layout::from_modes(&modes[begin..end])?;
        let mut out: Vec<Layout> = modes[..begin].to_vec();
        out.push(inner);
        out.extend_from_slice(&modes[end..]);
        if out.len() == 1 {
            // The group covers every mode: the layout is the group itself, read
            // as a single mode, so there is nothing left to wrap.
            return Ok(out.remove(0));
        }
        Layout::from_modes(&out)
    }

    /// `self` with `mode` added after its last top-level mode. A single-mode
    /// layout is read as a one-mode tuple, so `append` of `s:d` is `(s,..):(d,..)`.
    pub fn append(&self, mode: &Layout) -> Result<Layout, LayoutError> {
        let mut modes = self.modes();
        modes.push(mode.clone());
        Layout::from_modes(&modes)
    }

    /// `self` with `mode` added before its first top-level mode.
    pub fn prepend(&self, mode: &Layout) -> Result<Layout, LayoutError> {
        let mut modes = vec![mode.clone()];
        modes.extend(self.modes());
        Layout::from_modes(&modes)
    }

    /// `self` with top-level mode `index` replaced by `mode`. A single-mode
    /// layout is its own mode 0, so `replace(0, m)` of it is `m`. Fails with
    /// [`LayoutError::OutOfRange`] when `index >= rank`.
    pub fn replace(&self, index: usize, mode: &Layout) -> Result<Layout, LayoutError> {
        let mut modes = self.modes();
        let slot = modes.get_mut(index).ok_or(LayoutError::OutOfRange)?;
        *slot = mode.clone();
        if modes.len() == 1 {
            return Ok(modes.remove(0));
        }
        Layout::from_modes(&modes)
    }

    /// The modes `pat` *fixes*, the complement of [`Layout::slice`]: `All` drops
    /// a mode, `At(_)` keeps it (the integer is a placeholder; only the pattern's
    /// structure is read). Profile preserved, `1:0` when nothing is kept, a bare
    /// mode when exactly one top-level mode is kept.
    /// [`LayoutError::ProfileMismatch`] when the pattern does not fit the shape.
    pub fn dice(&self, pat: &SliceCrd) -> Result<Layout, LayoutError> {
        match dice_node(pat, &self.shape, &self.stride)? {
            Some((s, d)) => Layout::new(s, d),
            None => Layout::leaf(1, 0),
        }
    }

    /// Every extent whose stride is 0 becomes 1; strides and profile are kept.
    /// The result is the same function over the *reduced* index space: it
    /// agrees with `self` at every index whose stride-0 coordinates are 0.
    pub fn filter_zeros(&self) -> Layout {
        let shape = map_pair(&self.shape, &self.stride, &|s, d| {
            if d == 0 { 1 } else { s }
        });
        // Extents only shrink to 1, so every `Layout::new` check still holds.
        Layout {
            shape,
            stride: self.stride.clone(),
        }
    }

    /// [`Layout::filter_zeros`] then [`Layout::coalesce`]: all stride-0 and
    /// extent-1 modes removed and contiguous modes merged. Its image (the set of
    /// offsets it reaches) equals `self`'s.
    pub fn filter(&self) -> Layout {
        self.filter_zeros().coalesce()
    }
}

fn map_pair(shape: &IntTuple, stride: &IntTuple, f: &dyn Fn(i64, i64) -> i64) -> IntTuple {
    match (shape, stride) {
        (IntTuple::Tuple(ss), IntTuple::Tuple(ds)) => {
            IntTuple::Tuple(ss.iter().zip(ds).map(|(s, d)| map_pair(s, d, f)).collect())
        }
        (IntTuple::Int(s), IntTuple::Int(d)) => IntTuple::Int(f(*s, *d)),
        // `Layout` guarantees congruent profiles.
        _ => shape.clone(),
    }
}

type Kept = Option<(IntTuple, IntTuple)>;

fn dice_node(pat: &SliceCrd, shape: &IntTuple, stride: &IntTuple) -> Result<Kept, LayoutError> {
    match (pat, shape, stride) {
        (SliceCrd::All, _, _) => Ok(None),
        (SliceCrd::At(_), _, _) => Ok(Some((shape.clone(), stride.clone()))),
        (SliceCrd::Tuple(ps), IntTuple::Tuple(ss), IntTuple::Tuple(ds)) if ps.len() == ss.len() => {
            let (mut s_out, mut d_out) = (Vec::new(), Vec::new());
            for ((p, s), d) in ps.iter().zip(ss).zip(ds) {
                if let Some((ks, kd)) = dice_node(p, s, d)? {
                    s_out.push(ks);
                    d_out.push(kd);
                }
            }
            Ok(match s_out.len() {
                0 => None,
                1 => Some((s_out.remove(0), d_out.remove(0))),
                _ => Some((IntTuple::Tuple(s_out), IntTuple::Tuple(d_out))),
            })
        }
        _ => Err(LayoutError::ProfileMismatch),
    }
}
