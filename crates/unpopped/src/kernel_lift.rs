// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Lifting an intent-layer `OpDef` into the kernel layer** (`docs/ir-kernel-layer.md`
//! section 7). The layers must agree on every existing op: interpreting the lifted
//! [`KernelFn`] must give the bits `oracle::evaluate` gives the `OpDef`.
//!
//! Elementwise only in this increment. Anything the lift cannot express is a typed
//! [`LiftDecline`], never a partial kernel.
//!
//! The shape of an Elementwise kernel: one nested `For` per output axis, outermost first (the
//! counters are the logical coordinates, so no integer division is needed; the intent layer has
//! none), each input loaded through `Place::Logical`, the body stored through `Place::Logical`.
//! Views (strides, broadcast, reversal) live behind `Place::Logical`, exactly where the oracle
//! applies them.

use unpopped_vocab::ElementKind;

use crate::ir::{Access, OpDef, ScalarExpr, WriteIndex};
use crate::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt};
use crate::types::map_children;

/// Why an `OpDef` was not lifted.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum LiftDecline {
    /// The op is not `Access::Elementwise`; Reduction, Scan and RowReduce follow.
    NotElementwise,
    /// More than one output, a per-output dtype, a gather/scatter, a runtime base offset or a
    /// non-identity view: forms this increment does not lift.
    UnsupportedForm(&'static str),
    /// The body reads an intent-layer leaf that has no kernel-layer meaning yet.
    UnsupportedLeaf(&'static str),
    /// The op's input dtypes are not all the compute dtype.
    MixedDtypes,
    /// Rank is zero or more than the 8 the structure key allows.
    BadRank,
}

/// Lift the Elementwise `op` computed at `dtype` over `rank` output axes.
///
/// Parameters, in order: the `n_inputs` input buffers (`Read`), the output buffer (`Write`), then
/// one `I32` extent scalar per axis, readable as locals `0..rank`. Locals `rank..2*rank` are the
/// loop counters; locals `2*rank..` are the loaded inputs.
///
/// # Errors
///
/// A [`LiftDecline`].
pub fn lift_elementwise_opdef(
    op: &OpDef,
    dtype: ElementKind,
    rank: u8,
) -> Result<KernelFn, LiftDecline> {
    check_liftable(op, dtype, rank)?;
    let r = u32::from(rank);
    let n = u32::from(op.n_inputs);
    let counter = |d: u32| r + d;
    let loaded = |k: u32| 2 * r + k;
    let coords = || {
        (0..r)
            .map(|d| ScalarExpr::Local(counter(d)))
            .collect::<Vec<_>>()
    };
    let buffer = |b: u32| Place::Logical {
        buffer: b,
        coords: coords(),
    };

    let mut inner: Vec<Stmt> = (0..n)
        .map(|k| Stmt::Load {
            local: loaded(k),
            place: buffer(k),
        })
        .collect();
    inner.push(Stmt::Store {
        place: buffer(n),
        value: substitute_inputs(&op.body, &loaded),
    });
    // Wrap outward: the last axis is the innermost loop (row-major order).
    let mut body = Block(inner);
    for d in (0..r).rev() {
        body = Block(vec![Stmt::For {
            local: counter(d),
            start: ScalarExpr::ConstInt(0, ElementKind::I32),
            end: ScalarExpr::Local(d),
            step: ScalarExpr::ConstInt(1, ElementKind::I32),
            body,
        }]);
    }

    let mut params: Vec<Param> = (0..n)
        .map(|k| Param::buffer(format!("in{k}"), dtype, BufferRole::Read, Some(rank)))
        .collect();
    params.push(Param::buffer("out", dtype, BufferRole::Write, Some(rank)));
    params.extend((0..r).map(|d| Param::scalar(format!("n{d}"), ElementKind::I32, d)));
    let mut locals = vec![LocalDecl::scalar(ElementKind::I32); 2 * usize::from(rank)];
    locals.extend((0..n).map(|_| LocalDecl::scalar(dtype)));
    Ok(KernelFn::new(op.name.clone(), dtype, params, locals, body))
}

fn check_liftable(op: &OpDef, dtype: ElementKind, rank: u8) -> Result<(), LiftDecline> {
    if !matches!(op.access, Access::Elementwise) {
        return Err(LiftDecline::NotElementwise);
    }
    if rank == 0 || rank > 8 {
        return Err(LiftDecline::BadRank);
    }
    if op.output_bodies().len() != 1 || op.out_dtype.is_some() {
        return Err(LiftDecline::UnsupportedForm(
            "multi-output or a per-output dtype",
        ));
    }
    if !op.read_index.is_empty() || !matches!(op.write_index, WriteIndex::Direct) {
        return Err(LiftDecline::UnsupportedForm("gather or scatter"));
    }
    if !op.base_offsets.is_empty() || !op.out_base_offset.is_zero() {
        return Err(LiftDecline::UnsupportedForm("a runtime base offset"));
    }
    if op
        .views
        .iter()
        .any(|v| !matches!(v, crate::ir::View::Identity))
    {
        return Err(LiftDecline::UnsupportedForm("a non-identity view"));
    }
    if op.dtypes.iter().any(|d| *d != dtype) {
        return Err(LiftDecline::MixedDtypes);
    }
    leaves_ok(&op.body)
}

fn leaves_ok(e: &ScalarExpr) -> Result<(), LiftDecline> {
    match e {
        ScalarExpr::Coord(_) | ScalarExpr::CoordInt(..) => {
            return Err(LiftDecline::UnsupportedLeaf("Coord"));
        }
        ScalarExpr::Param(_) => return Err(LiftDecline::UnsupportedLeaf("Param")),
        ScalarExpr::Reduced(_) => return Err(LiftDecline::UnsupportedLeaf("Reduced")),
        ScalarExpr::Local(_) => return Err(LiftDecline::UnsupportedLeaf("Local")),
        _ => {}
    }
    map_children(e, &mut |c| leaves_ok(c).map(|()| c.clone())).map(|_| ())
}

/// `body` with every `Input(k)` replaced by `Local(loaded(k))`.
fn substitute_inputs(body: &ScalarExpr, loaded: &dyn Fn(u32) -> u32) -> ScalarExpr {
    if let ScalarExpr::Input(k) = body {
        return ScalarExpr::Local(loaded(u32::from(*k)));
    }
    let r: Result<ScalarExpr, std::convert::Infallible> =
        map_children(body, &mut |c| Ok(substitute_inputs(c, loaded)));
    match r {
        Ok(e) => e,
        Err(never) => match never {},
    }
}
