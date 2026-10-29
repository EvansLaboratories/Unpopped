// SPDX-License-Identifier: MIT OR Apache-2.0
//! [`View`] as a [`Layout`]: the bridge between the four-variant read-through
//! enum and the layout algebra of `unpopped_vocab::layout_algebra`.
//!
//! Additive and one-way-safe: it changes no IR type, key or codec. A `View` is
//! a statement about *which producer address an iteration coordinate reads*;
//! given the iteration extents and the producer's physical strides, that is a
//! [`Layout`], and [`view_to_layout`] builds it. [`layout_to_view`] goes back
//! only when one of the four variants says exactly the same thing, and returns
//! `None` otherwise: it never returns a nearest `View`.
//!
//! **Axis order.** The oracle and emitters iterate LAST-AXIS-FASTEST; a
//! `Layout` indexes FIRST-MODE-FASTEST. So the layout's modes are the
//! iteration axes in *reverse* order, and `layout.call(i)` is the producer
//! element offset of the iteration coordinate whose row-major linear index is
//! `i`. Forgetting the reversal is the characteristic bug here; the oracle
//! differential in `tests/view_layout_oracle.rs` is what pins it.
//!
//! **Strides are the producer's physical strides** (`OperandDesc::strides`),
//! per producer axis, exactly as the oracle's `read_strided` reads them:
//! `offset = sum_d coord[d] * strides[perm[d]]`. A broadcast axis is a
//! producer stride of 0 (the key already encodes it); a negative (flipped)
//! stride is outside `Layout`'s domain and is declined.

use crate::ir::View;
use unpopped_vocab::AxisMask;
use unpopped_vocab::layout_algebra::{IntTuple, Layout, LayoutError};

/// Why a [`View`] could not be turned into a [`Layout`].
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ViewLayoutError {
    /// `iter_shape` and `producer_strides` differ in length, or are empty, or
    /// exceed the rank cap of 8.
    RankMismatch,
    /// A `Permute` that is not a permutation of `0..rank`.
    NotAPermutation,
    /// A `Broadcast` view names an axis whose producer stride is not 0 (the key
    /// drives the address math, so a view-only broadcast would be a lie).
    BroadcastDisagrees {
        /// The offending iteration axis.
        axis: u8,
    },
    /// A `Reshape` whose producer rank differs from the iteration rank:
    /// rank-change reshape has no address math yet (see `View`).
    RankChangingReshape,
    /// The layout algebra declined the strides or extents (negative stride,
    /// non-positive extent, overflow).
    Layout(LayoutError),
}

impl std::fmt::Display for ViewLayoutError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RankMismatch => {
                write!(f, "iteration shape and producer strides disagree in rank")
            }
            Self::NotAPermutation => write!(f, "Permute is not a permutation of 0..rank"),
            Self::BroadcastDisagrees { axis } => {
                write!(
                    f,
                    "Broadcast names axis {axis}, whose producer stride is not 0"
                )
            }
            Self::RankChangingReshape => write!(f, "rank-changing Reshape has no address math"),
            Self::Layout(e) => write!(f, "layout algebra declined: {e}"),
        }
    }
}

impl std::error::Error for ViewLayoutError {}

impl From<LayoutError> for ViewLayoutError {
    fn from(e: LayoutError) -> Self {
        Self::Layout(e)
    }
}

/// The layout of the producer reads `view` performs over `iter_shape`, given the
/// producer's physical `producer_strides` (see the module docs for axis order).
pub fn view_to_layout(
    view: &View,
    iter_shape: &[i64],
    producer_strides: &[i64],
) -> Result<Layout, ViewLayoutError> {
    let rank = iter_shape.len();
    if rank == 0 || rank > 8 || producer_strides.len() != rank {
        return Err(ViewLayoutError::RankMismatch);
    }
    let strides: Vec<i64> = match view {
        View::Identity => producer_strides.to_vec(),
        View::Reshape { producer_rank } if usize::from(*producer_rank) == rank => {
            producer_strides.to_vec()
        }
        View::Reshape { .. } => return Err(ViewLayoutError::RankChangingReshape),
        View::Permute { perm } => {
            if !is_permutation(perm, rank) {
                return Err(ViewLayoutError::NotAPermutation);
            }
            perm.iter()
                .map(|&p| producer_strides[usize::from(p)])
                .collect()
        }
        View::Broadcast { bcast } => {
            for (d, &s) in producer_strides.iter().enumerate() {
                if bcast.is_set(d as u8) && s != 0 {
                    return Err(ViewLayoutError::BroadcastDisagrees { axis: d as u8 });
                }
            }
            producer_strides.to_vec()
        }
    };
    Ok(reversed_layout(iter_shape, &strides)?)
}

/// Modes in reverse axis order; a rank-1 layout is a bare mode.
fn reversed_layout(shape: &[i64], strides: &[i64]) -> Result<Layout, LayoutError> {
    let s = IntTuple::ints(shape.iter().rev().copied());
    let d = IntTuple::ints(strides.iter().rev().copied());
    if shape.len() == 1 {
        return Layout::leaf(shape[0], strides[0]);
    }
    Layout::new(s, d)
}

fn is_permutation(perm: &[u8], rank: usize) -> bool {
    let mut seen = vec![false; rank];
    perm.len() == rank
        && perm.iter().all(|&p| {
            let p = usize::from(p);
            p < rank && !std::mem::replace(&mut seen[p], true)
        })
}

/// The `View` that reads `producer_strides` exactly as `layout` does, or `None`
/// when none of the four variants says that.
///
/// `layout` must be flat (depth at most 1: one mode per iteration axis, in the
/// reverse axis order [`view_to_layout`] produces). Returns:
/// - [`View::Identity`] when every stride equals the producer's;
/// - [`View::Broadcast`] when the strides equal the producer's and some axis of
///   extent above 1 has stride 0 (the key encodes the broadcast already);
/// - [`View::Permute`] when a *unique* assignment of producer axes reproduces
///   every stride, none being a broadcast (Permute and Broadcast stay
///   orthogonal in v1);
/// - `None` otherwise, including an ambiguous permutation.
///
/// [`View::Reshape`] is never returned: a same-rank reshape reads exactly like
/// `Identity`, which is returned instead.
#[must_use]
pub fn layout_to_view(layout: &Layout, producer_strides: &[i64]) -> Option<View> {
    if layout.depth() > 1 || producer_strides.is_empty() {
        return None;
    }
    let modes = layout.flat_modes();
    let rank = producer_strides.len();
    if modes.len() != rank || rank > 8 {
        return None;
    }
    // Iteration axis d is mode rank-1-d.
    let axis = |d: usize| modes[rank - 1 - d];
    if (0..rank).all(|d| axis(d).1 == producer_strides[d]) {
        let mut mask = AxisMask::EMPTY;
        for d in 0..rank {
            if axis(d).1 == 0 && axis(d).0 > 1 {
                mask.set(d as u8);
            }
        }
        return Some(if mask.is_empty() {
            View::Identity
        } else {
            View::Broadcast { bcast: mask }
        });
    }
    if (0..rank).any(|d| axis(d).1 == 0 && axis(d).0 > 1) {
        return None;
    }
    let mut found: Vec<Vec<u8>> = Vec::new();
    assign(
        &(0..rank).map(|d| axis(d).1).collect::<Vec<_>>(),
        producer_strides,
        &mut Vec::new(),
        &mut found,
    );
    (found.len() == 1).then(|| View::Permute {
        perm: found.remove(0),
    })
}

/// All bijections `perm` with `wanted[d] == producer[perm[d]]` (stops at two).
fn assign(wanted: &[i64], producer: &[i64], cur: &mut Vec<u8>, out: &mut Vec<Vec<u8>>) {
    if out.len() > 1 {
        return;
    }
    let d = cur.len();
    if d == wanted.len() {
        out.push(cur.clone());
        return;
    }
    for j in 0..producer.len() {
        if producer[j] == wanted[d] && !cur.contains(&(j as u8)) {
            cur.push(j as u8);
            assign(wanted, producer, cur, out);
            cur.pop();
        }
    }
}
