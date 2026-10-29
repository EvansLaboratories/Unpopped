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

use crate::ir::{
    Access, BinaryOp, CastSpec, OpDef, ReduceOp, RoundMode, Saturation, ScalarExpr, WriteIndex,
};
use crate::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt};
use crate::types::map_children;

/// Why an `OpDef` was not lifted.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum LiftDecline {
    /// The op is not `Access::Elementwise` (for [`lift_elementwise_opdef`]); Scan and RowReduce
    /// follow.
    NotElementwise,
    /// The op is not `Access::Reduction` (for [`lift_reduction_opdef`]).
    NotReduction,
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
    check_common(op, dtype, rank)
}

/// The conditions every lifted access shares: rank, one output, plain addressing, one dtype,
/// and a body that reads only inputs and constants.
fn check_common(op: &OpDef, dtype: ElementKind, rank: u8) -> Result<(), LiftDecline> {
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

// ---------------------------------------------------------------------------------------------
// Reduction

/// Lift a float `Access::Reduction` over `rank` input axes, computed at `dtype` (`F32` or `F64`).
///
/// The oracle accumulates a float reduction in `f64` and rounds once at the store, so the kernel
/// does too: an `F64` accumulator local, a `Cast` of each body value up to `F64`, the fold in the
/// oracle's order (kept axes outermost ascending, reduced axes ascending, last fastest), and one
/// `Cast` down at the `Store`. `Max` and `Min` carry a `have` flag and keep a NaN once seen (the
/// oracle's rule: `!have || v.is_nan() || better`); `Mean` divides the `F64` accumulator by the
/// reduced extents' product. Only the identity post-expression (`Reduced(0)`) lifts: a fused post
/// would have to run at the compute dtype over an unrounded `F64` value, which the typed kernel
/// language cannot state.
///
/// Parameters, in order: the inputs (`Read`, rank `rank`), the output (`Write`; rank `rank` with
/// `keepdim`, else the kept-axis count), then one `I32` extent per input axis. Locals:
/// extents `0..rank`, counters `rank..2*rank`, loaded inputs `2*rank..2*rank+n`, then the
/// accumulator, the `have` flag and the current value.
///
/// # Errors
///
/// A [`LiftDecline`].
pub fn lift_reduction_opdef(
    op: &OpDef,
    dtype: ElementKind,
    rank: u8,
) -> Result<KernelFn, LiftDecline> {
    let Access::Reduction {
        op: rop,
        axes,
        keepdim,
        post,
        ..
    } = &op.access
    else {
        return Err(LiftDecline::NotReduction);
    };
    if !matches!(dtype, ElementKind::F32 | ElementKind::F64) {
        return Err(LiftDecline::UnsupportedForm("a non-float reduction"));
    }
    if !matches!(post, ScalarExpr::Reduced(0)) {
        return Err(LiftDecline::UnsupportedForm("a fused post-expression"));
    }
    check_common(op, dtype, rank)?;

    let sh = ReductionShape::new(op.n_inputs, dtype, rank, axes);
    let mut body = Block(vec![
        Stmt::Let {
            local: sh.acc,
            init: sh.up(ScalarExpr::Const(if matches!(rop, ReduceOp::Prod) {
                1.0
            } else {
                0.0
            })),
        },
        Stmt::Let {
            local: sh.have,
            init: ScalarExpr::ConstInt(0, ElementKind::I32),
        },
    ]);
    body.0.extend(sh.reduce_loop(op, *rop).0);
    let out_coords = sh.out_coords(*keepdim);
    let out_rank = u8::try_from(out_coords.len()).expect("rank is at most 8");
    body.0.push(Stmt::Store {
        place: Place::Logical {
            buffer: sh.n,
            coords: out_coords,
        },
        value: sh.stored(*rop),
    });
    for &d in sh.kept.iter().rev() {
        body = for_axis(d, sh.r, body);
    }

    let (params, locals) = sh.signature(out_rank);
    Ok(KernelFn::new(op.name.clone(), dtype, params, locals, body))
}

/// The axis bookkeeping of a lifted reduction: which axes are kept and which are folded, and the
/// local ids of the extents, counters, loaded inputs, accumulator, `have` flag and current value.
struct ReductionShape {
    dtype: ElementKind,
    /// Input axis count.
    r: u32,
    /// Input count.
    n: u32,
    reduced: Vec<u32>,
    kept: Vec<u32>,
    acc: u32,
    have: u32,
    val: u32,
}

impl ReductionShape {
    fn new(n_inputs: u8, dtype: ElementKind, rank: u8, axes: &unpopped_vocab::AxisMask) -> Self {
        let (r, n) = (u32::from(rank), u32::from(n_inputs));
        let reduced: Vec<u32> = if axes.is_empty() {
            vec![r - 1]
        } else {
            (0..r).filter(|&d| axes.is_set(d as u8)).collect()
        };
        let kept = (0..r).filter(|d| !reduced.contains(d)).collect();
        Self {
            dtype,
            r,
            n,
            reduced,
            kept,
            acc: 2 * r + n,
            have: 2 * r + n + 1,
            val: 2 * r + n + 2,
        }
    }

    fn counter(&self, d: u32) -> u32 {
        self.r + d
    }

    fn loaded(&self, k: u32) -> u32 {
        2 * self.r + k
    }

    /// `e` widened to the `F64` accumulator width (a no-op for `F64` compute).
    fn up(&self, e: ScalarExpr) -> ScalarExpr {
        if self.dtype == ElementKind::F64 {
            e
        } else {
            ScalarExpr::Cast(Box::new(e), widen(ElementKind::F64))
        }
    }

    /// The reduced axes' loops, outermost ascending, around one pass of the fold.
    fn reduce_loop(&self, op: &OpDef, rop: ReduceOp) -> Block {
        let mut inner: Vec<Stmt> = (0..self.n)
            .map(|k| Stmt::Load {
                local: self.loaded(k),
                place: Place::Logical {
                    buffer: k,
                    coords: (0..self.r)
                        .map(|d| ScalarExpr::Local(self.counter(d)))
                        .collect(),
                },
            })
            .collect();
        inner.push(Stmt::Let {
            local: self.val,
            init: self.up(substitute_inputs(&op.body, &|k| self.loaded(k))),
        });
        inner.extend(fold_step(rop, self.acc, self.have, self.val));
        let mut nest = Block(inner);
        for &d in self.reduced.iter().rev() {
            nest = for_axis(d, self.r, nest);
        }
        nest
    }

    /// Output coordinates: the kept counters, with `0` for each reduced axis under `keepdim`.
    fn out_coords(&self, keepdim: bool) -> Vec<ScalarExpr> {
        if keepdim {
            (0..self.r)
                .map(|d| {
                    if self.kept.contains(&d) {
                        ScalarExpr::Local(self.counter(d))
                    } else {
                        ScalarExpr::ConstInt(0, ElementKind::I32)
                    }
                })
                .collect()
        } else {
            self.kept
                .iter()
                .map(|&d| ScalarExpr::Local(self.counter(d)))
                .collect()
        }
    }

    /// The stored value: the accumulator (divided by the reduced extents for `Mean`), cast down
    /// to the compute dtype once.
    fn stored(&self, rop: ReduceOp) -> ScalarExpr {
        let acc = ScalarExpr::Local(self.acc);
        let finalized = if matches!(rop, ReduceOp::Mean) {
            let divisor = self
                .reduced
                .iter()
                .map(|&d| ScalarExpr::Cast(Box::new(ScalarExpr::Local(d)), widen(ElementKind::F64)))
                .reduce(|a, b| ScalarExpr::Mul(Box::new(a), Box::new(b)))
                .expect("a reduction has at least one reduced axis");
            ScalarExpr::Div(Box::new(acc), Box::new(divisor))
        } else {
            acc
        };
        if self.dtype == ElementKind::F64 {
            finalized
        } else {
            ScalarExpr::Cast(Box::new(finalized), widen(self.dtype))
        }
    }

    /// Parameters (inputs, output, one `I32` extent per input axis) and local declarations.
    fn signature(&self, out_rank: u8) -> (Vec<Param>, Vec<LocalDecl>) {
        let rank = u8::try_from(self.r).expect("rank is at most 8");
        let mut params: Vec<Param> = (0..self.n)
            .map(|k| Param::buffer(format!("in{k}"), self.dtype, BufferRole::Read, Some(rank)))
            .collect();
        params.push(Param::buffer(
            "out",
            self.dtype,
            BufferRole::Write,
            Some(out_rank),
        ));
        params.extend((0..self.r).map(|d| Param::scalar(format!("n{d}"), ElementKind::I32, d)));
        let mut locals = vec![LocalDecl::scalar(ElementKind::I32); 2 * self.r as usize];
        locals.extend((0..self.n).map(|_| LocalDecl::scalar(self.dtype)));
        locals.push(LocalDecl::scalar(ElementKind::F64)); // acc
        locals.push(LocalDecl::scalar(ElementKind::I32)); // have
        locals.push(LocalDecl::scalar(ElementKind::F64)); // val
        (params, locals)
    }
}

/// A float-to-float (or int-to-float) conversion that rounds to nearest even and never traps.
fn widen(to: ElementKind) -> CastSpec {
    CastSpec {
        to,
        round: RoundMode::NearestEven,
        sat: Saturation::Wrap,
    }
}

/// `for counter(d) in 0..extent(d) { inner }` for input axis `d` of a rank-`r` kernel.
fn for_axis(d: u32, r: u32, inner: Block) -> Block {
    Block(vec![Stmt::For {
        local: r + d,
        start: ScalarExpr::ConstInt(0, ElementKind::I32),
        end: ScalarExpr::Local(d),
        step: ScalarExpr::ConstInt(1, ElementKind::I32),
        body: inner,
    }])
}

/// The oracle's `fold_step` for a float accumulator `acc` (an `F64` local) and value `val`.
fn fold_step(rop: ReduceOp, acc: u32, have: u32, val: u32) -> Vec<Stmt> {
    let l = ScalarExpr::Local;
    let bx = Box::new;
    let set = |v: u32| Stmt::Assign {
        local: acc,
        value: l(v),
    };
    match rop {
        ReduceOp::Sum | ReduceOp::Mean => vec![Stmt::Assign {
            local: acc,
            value: ScalarExpr::Add(bx(l(acc)), bx(l(val))),
        }],
        ReduceOp::Prod => vec![Stmt::Assign {
            local: acc,
            value: ScalarExpr::Mul(bx(l(acc)), bx(l(val))),
        }],
        ReduceOp::Max | ReduceOp::Min => {
            let better = if matches!(rop, ReduceOp::Max) {
                BinaryOp::CmpGt
            } else {
                BinaryOp::CmpLt
            };
            // if !have { acc = v } else if v != v { acc = v } else if better(v, acc) { acc = v }
            let keep_going = Stmt::If {
                cond: ScalarExpr::Binary(better, bx(l(val)), bx(l(acc))),
                then: Block(vec![set(val)]),
                els: Block(vec![]),
            };
            let nan_check = Stmt::If {
                cond: ScalarExpr::Binary(BinaryOp::CmpNe, bx(l(val)), bx(l(val))),
                then: Block(vec![set(val)]),
                els: Block(vec![keep_going]),
            };
            vec![
                Stmt::If {
                    cond: l(have),
                    then: Block(vec![nan_check]),
                    els: Block(vec![set(val)]),
                },
                Stmt::Assign {
                    local: have,
                    value: ScalarExpr::ConstInt(1, ElementKind::I32),
                },
            ]
        }
    }
}
