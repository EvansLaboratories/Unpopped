// SPDX-License-Identifier: MIT OR Apache-2.0
//! Helpers for lifting a fused post-expression or epilogue (`Reduced(s)` leaves).
//!
//! The oracle hands `Reduced(s)` to a post as an unrounded `f64`; the lifts bind it instead to a
//! compute-dtype local holding the stage value rounded once (`docs/ir-kernel-layer.md` 7.1).

use std::convert::Infallible;

use super::{LiftDecline, leaves_ok};
use crate::ir::ScalarExpr;
use crate::types::map_children;

/// [`leaves_ok`] for an expression that may read `Reduced(t)` for `t < available`.
pub(super) fn check_post(e: &ScalarExpr, available: usize) -> Result<(), LiftDecline> {
    leaves_ok(&strip_reduced(e, available)?)
}

/// `e` with every in-range `Reduced(t)` replaced by a constant, so [`leaves_ok`] can judge the rest.
fn strip_reduced(e: &ScalarExpr, available: usize) -> Result<ScalarExpr, LiftDecline> {
    if let ScalarExpr::Reduced(t) = e {
        return if usize::from(*t) < available {
            Ok(ScalarExpr::Const(0.0))
        } else {
            Err(LiftDecline::UnsupportedLeaf("a Reduced of a later stage"))
        };
    }
    map_children(e, &mut |c| strip_reduced(c, available))
}

/// Whether `e` reads any `Input`.
pub(super) fn reads_input(e: &ScalarExpr) -> bool {
    if matches!(e, ScalarExpr::Input(_)) {
        return true;
    }
    let mut found = false;
    let _: Result<ScalarExpr, Infallible> = map_children(e, &mut |c| {
        found |= reads_input(c);
        Ok(c.clone())
    });
    found
}

/// Whether `e` reads an `Input(i)` outside `lo..hi`.
pub(super) fn reads_input_outside(e: &ScalarExpr, lo: u8, hi: u8) -> bool {
    if let ScalarExpr::Input(i) = e {
        return !(lo..hi).contains(i);
    }
    let mut found = false;
    let _: Result<ScalarExpr, Infallible> = map_children(e, &mut |c| {
        found |= reads_input_outside(c, lo, hi);
        Ok(c.clone())
    });
    found
}

/// `e` with `Reduced(t)` replaced by `Local(local_of(t))`; every other leaf is left to `leaf`.
pub(super) fn bind(e: &ScalarExpr, leaf: &dyn Fn(&ScalarExpr) -> Option<ScalarExpr>) -> ScalarExpr {
    if let Some(bound) = leaf(e) {
        return bound;
    }
    let r: Result<ScalarExpr, Infallible> = map_children(e, &mut |c| Ok(bind(c, leaf)));
    match r {
        Ok(e) => e,
        Err(never) => match never {},
    }
}
