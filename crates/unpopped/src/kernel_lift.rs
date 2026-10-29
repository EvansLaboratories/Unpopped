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

use unpopped_vocab::{AxisMask, ElementKind};

use crate::ir::{
    Access, BinaryOp, CastSpec, ContractionAxes, OpDef, ReduceOp, RoundMode, Saturation,
    ScalarExpr, WriteIndex,
};
use crate::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt};
use crate::types::map_children;

mod im2col;
mod post;
mod row_reduce;
mod window;
pub use im2col::lift_im2col_opdef;
pub use row_reduce::lift_row_reduce_opdef;
pub use window::lift_window_opdef;

/// Why an `OpDef` was not lifted.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum LiftDecline {
    /// The op is not `Access::Elementwise` (for [`lift_elementwise_opdef`]); Scan and RowReduce
    /// follow.
    NotElementwise,
    /// The op is not `Access::Reduction` (for [`lift_reduction_opdef`]).
    NotReduction,
    /// The op is not `Access::Contraction` (for [`lift_contraction_opdef`]).
    NotContraction,
    /// The op is not `Access::Scan` (for [`lift_scan_opdef`]).
    NotScan,
    /// The op is not `Access::Im2Col` (for [`lift_im2col_opdef`]).
    NotIm2Col,
    /// The op is not `Access::Window` (for [`lift_window_opdef`]).
    NotWindow,
    /// The op is not `Access::RowReduce` (for [`lift_row_reduce_opdef`]).
    NotRowReduce,
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
    check_plain(op, dtype, rank)?;
    leaves_ok(&op.body)
}

/// [`check_common`] without the body-leaf check (a `Scan` keeps its `Reduced(0)` post in
/// `body`, and its value expression in `pre`).
fn check_plain(op: &OpDef, dtype: ElementKind, rank: u8) -> Result<(), LiftDecline> {
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
    Ok(())
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
    let (rop, axes, keepdim, post) = check_reduction(op, dtype, rank)?;
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
    body.0.extend(sh.reduce_loop(op, rop).0);
    let out_coords = sh.out_coords(keepdim);
    let out_rank = u8::try_from(out_coords.len()).expect("rank is at most 8");
    let value = match post {
        None => sh.stored(rop),
        Some(post) => {
            body.0.push(Stmt::Let {
                local: sh.red,
                init: sh.stored(rop),
            });
            post::bind(post, &|leaf| {
                matches!(leaf, ScalarExpr::Reduced(_)).then(|| ScalarExpr::Local(sh.red))
            })
        }
    };
    body.0.push(Stmt::Store {
        place: Place::Logical {
            buffer: sh.n,
            coords: out_coords,
        },
        value,
    });
    for &d in sh.kept.iter().rev() {
        body = for_axis(d, sh.r, body);
    }

    let (params, mut locals) = sh.signature(out_rank);
    if post.is_some() {
        locals.push(LocalDecl::scalar(dtype)); // red: the stage value rounded to the compute dtype
    }
    Ok(KernelFn::new(op.name.clone(), dtype, params, locals, body))
}

/// Lift a float `Access::Scan` over `rank` axes (the scanned axis is the innermost), computed at
/// `dtype` (`F32` or `F64`), reading dense row-major inputs as the oracle's flat addressing does.
///
/// Per row, the oracle keeps an `f64` prefix that starts at the monoid identity (`0`, `1`, `-inf`,
/// `+inf`), folds each pre-mapped element into it (the `Max`/`Min` NaN rule as in a reduction) and
/// writes the prefix after the fold (inclusive) or before it (exclusive), walking the axis
/// descending for a reverse scan. The kernel does the same with an `F64` prefix local and one
/// `Cast` down at each `Store`. Only the identity post-expression (`Reduced(0)`) lifts, for the
/// reason a reduction's fused post does not.
///
/// Parameters and locals are laid out as for [`lift_reduction_opdef`], with the output the same
/// rank as the input.
///
/// # Errors
///
/// A [`LiftDecline`].
pub fn lift_scan_opdef(op: &OpDef, dtype: ElementKind, rank: u8) -> Result<KernelFn, LiftDecline> {
    let (sop, reverse, exclusive, pre) = check_scan(op, dtype, rank)?;
    let sh = ReductionShape::new(op.n_inputs, dtype, rank, AxisMask::EMPTY);
    let pass = scan_pass(&sh, sop, exclusive, pre);
    let mut row = scan_row(&sh, sop, reverse, pass);
    for &d in sh.kept.iter().rev() {
        row = for_axis(d, sh.r, row);
    }
    let (params, locals) = sh.signature(u8::try_from(sh.r).expect("rank is at most 8"));
    Ok(KernelFn::new(op.name.clone(), dtype, params, locals, row))
}

/// One position of the scan: load the inputs, pre-map, fold into the prefix, and write the prefix
/// after the fold (inclusive) or before it (exclusive).
fn scan_pass(sh: &ReductionShape, sop: ReduceOp, exclusive: bool, pre: &ScalarExpr) -> Vec<Stmt> {
    let all_counters = || {
        (0..sh.r)
            .map(|d| ScalarExpr::Local(sh.counter(d)))
            .collect::<Vec<_>>()
    };
    let mut pass: Vec<Stmt> = (0..sh.n)
        .map(|k| Stmt::Load {
            local: sh.loaded(k),
            place: Place::Logical {
                buffer: k,
                coords: all_counters(),
            },
        })
        .collect();
    pass.push(Stmt::Let {
        local: sh.val,
        init: sh.up(substitute_inputs(pre, &|k| sh.loaded(k))),
    });
    let write = Stmt::Store {
        place: Place::Logical {
            buffer: sh.n,
            coords: all_counters(),
        },
        value: sh.stored(sop),
    };
    let fold = fold_step(sop, sh.acc, sh.have, sh.val);
    if exclusive {
        pass.push(write);
        pass.extend(fold);
    } else {
        pass.extend(fold);
        pass.push(write);
    }
    pass
}

/// The scan of one row: the prefix starts at the monoid identity and the scanned axis is walked
/// ascending, or descending for a reverse scan.
fn scan_row(sh: &ReductionShape, sop: ReduceOp, reverse: bool, pass: Vec<Stmt>) -> Block {
    let int = |v: i64| ScalarExpr::ConstInt(v, ElementKind::I32);
    let scanned = sh.reduced[0];
    let extent = ScalarExpr::Local(scanned);
    let (start, end, step) = if reverse {
        (
            ScalarExpr::Sub(Box::new(extent), Box::new(int(1))),
            int(-1),
            int(-1),
        )
    } else {
        (int(0), extent, int(1))
    };
    Block(vec![
        Stmt::Let {
            local: sh.acc,
            init: sh.up(scan_identity(sop, sh.dtype)),
        },
        Stmt::Let {
            local: sh.have,
            init: int(0),
        },
        Stmt::For {
            local: sh.counter(scanned),
            start,
            end,
            step,
            body: Block(pass),
        },
    ])
}

/// The scan's op, direction, exclusivity and pre-map once the op is known to be liftable.
fn check_scan(
    op: &OpDef,
    dtype: ElementKind,
    rank: u8,
) -> Result<(ReduceOp, bool, bool, &ScalarExpr), LiftDecline> {
    let Access::Scan {
        op: sop,
        axis,
        reverse,
        exclusive,
        pre,
        post,
    } = &op.access
    else {
        return Err(LiftDecline::NotScan);
    };
    if !matches!(dtype, ElementKind::F32 | ElementKind::F64) {
        return Err(LiftDecline::UnsupportedForm("a non-float scan"));
    }
    if matches!(sop, ReduceOp::Mean) {
        return Err(LiftDecline::UnsupportedForm("a Mean scan (not a monoid)"));
    }
    if !matches!(post, ScalarExpr::Reduced(0)) {
        return Err(LiftDecline::UnsupportedForm("a fused post-expression"));
    }
    if u32::from(*axis) + 1 != u32::from(rank) {
        return Err(LiftDecline::UnsupportedForm(
            "a scan over a non-innermost axis",
        ));
    }
    check_plain(op, dtype, rank)?;
    leaves_ok(pre)?;
    Ok((*sop, *reverse, *exclusive, pre))
}

/// The scan monoid identity (the oracle's `scan_identity`, float domain) as an expression of the
/// compute dtype. The infinities are built from their bit patterns (`Bitcast` of an integer
/// constant) because a JSON number cannot be infinite and a kernel is authored as JSON.
fn scan_identity(rop: ReduceOp, dtype: ElementKind) -> ScalarExpr {
    let infinity = |neg: bool| {
        let (bits, int_ty) = if dtype == ElementKind::F64 {
            let v = if neg {
                f64::NEG_INFINITY
            } else {
                f64::INFINITY
            };
            (v.to_bits() as i64, ElementKind::I64)
        } else {
            let v = if neg {
                f32::NEG_INFINITY
            } else {
                f32::INFINITY
            };
            (i64::from(v.to_bits() as i32), ElementKind::I32)
        };
        ScalarExpr::Bitcast(Box::new(ScalarExpr::ConstInt(bits, int_ty)), dtype)
    };
    match rop {
        ReduceOp::Prod => ScalarExpr::Const(1.0),
        ReduceOp::Max => infinity(true),
        ReduceOp::Min => infinity(false),
        ReduceOp::Sum | ReduceOp::Mean => ScalarExpr::Const(0.0),
    }
}

/// The reduction's op, axis set and `keepdim` once the op is known to be liftable: a float
/// reduction with the identity post-expression and a plain body.
fn check_reduction(
    op: &OpDef,
    dtype: ElementKind,
    rank: u8,
) -> Result<(ReduceOp, AxisMask, bool, Option<&ScalarExpr>), LiftDecline> {
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
    let post = if matches!(post, ScalarExpr::Reduced(0)) {
        None
    } else {
        post::check_post(post, 1)?;
        if post::reads_input(post) {
            return Err(LiftDecline::UnsupportedLeaf("an Input in a reduction post"));
        }
        Some(post)
    };
    check_common(op, dtype, rank)?;
    Ok((*rop, *axes, *keepdim, post))
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
    /// The stage value rounded to the compute dtype (declared only when a post is lifted).
    red: u32,
}

impl ReductionShape {
    fn new(n_inputs: u8, dtype: ElementKind, rank: u8, axes: AxisMask) -> Self {
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
            red: 2 * r + n + 3,
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
    match rop {
        ReduceOp::Sum | ReduceOp::Mean => vec![Stmt::Assign {
            local: acc,
            value: ScalarExpr::Add(bx(l(acc)), bx(l(val))),
        }],
        ReduceOp::Prod => vec![Stmt::Assign {
            local: acc,
            value: ScalarExpr::Mul(bx(l(acc)), bx(l(val))),
        }],
        ReduceOp::Max | ReduceOp::Min => extremum_step(rop, acc, have, val),
    }
}

/// `Max` / `Min` step: `if !have { acc = v } else if v != v { acc = v } else if better(v, acc)
/// { acc = v }`, then `have = 1` (the oracle's `!have || v.is_nan() || better`).
fn extremum_step(rop: ReduceOp, acc: u32, have: u32, val: u32) -> Vec<Stmt> {
    let l = ScalarExpr::Local;
    let bx = Box::new;
    let set = |v: u32| Stmt::Assign {
        local: acc,
        value: l(v),
    };
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

// ---------------------------------------------------------------------------------------------
// Contraction

/// Lift a float `Access::Contraction`, plain `[M,K]x[K,N]` or batched `[B,M,K]x[B,K,N]`, computed
/// at `dtype` (`F32` or `F64`), with the identity epilogue (`Reduced(0)`) and no fused bias.
///
/// The oracle reads each operand by its physical strides at the logical `(m, k)` / `(k, n)`
/// coordinate (so transposed and broadcast operands need nothing extra: that is what
/// `Place::Logical` does), widens both factors to `f64`, multiplies and accumulates in `f64` with
/// `k` ascending, and rounds once at the store. The kernel does the same: nested `For` over
/// `b`, `m`, `n`, an inner `For` over `k`, an `F64` accumulator, `Cast` up on each factor.
///
/// Parameters: `lhs`, `rhs` (`Read`), `out` (`Write`), then `I32` extents `[B,] M, N, K`.
///
/// # Errors
///
/// A [`LiftDecline`].
pub fn lift_contraction_opdef(op: &OpDef, dtype: ElementKind) -> Result<KernelFn, LiftDecline> {
    let batched = check_contraction(op, dtype)?;
    let sh = ContractionShape::new(dtype, batched);
    let k_loop = for_axis(
        sh.k,
        sh.axes,
        Block(vec![
            sh.load(0, &sh.lhs_coords(), sh.a),
            sh.load(1, &sh.rhs_coords(), sh.b),
            Stmt::Assign {
                local: sh.acc,
                value: ScalarExpr::Add(
                    Box::new(ScalarExpr::Local(sh.acc)),
                    Box::new(ScalarExpr::Mul(
                        Box::new(sh.up(ScalarExpr::Local(sh.a))),
                        Box::new(sh.up(ScalarExpr::Local(sh.b))),
                    )),
                ),
            },
        ]),
    );
    let mut body = Block(vec![Stmt::Let {
        local: sh.acc,
        init: sh.up(ScalarExpr::Const(0.0)),
    }]);
    body.0.extend(k_loop.0);
    body.0.push(Stmt::Store {
        place: Place::Logical {
            buffer: 2,
            coords: sh.out_coords(),
        },
        value: sh.stored(),
    });
    for d in sh.outer_axes().into_iter().rev() {
        body = for_axis(d, sh.axes, body);
    }
    let (params, locals) = sh.signature();
    Ok(KernelFn::new(op.name.clone(), dtype, params, locals, body))
}

/// `true` when batched. The op must be exactly a plain or a batched matmul with the identity
/// epilogue, two inputs (no fused bias) and float compute.
fn check_contraction(op: &OpDef, dtype: ElementKind) -> Result<bool, LiftDecline> {
    let Access::Contraction { axes, epilogue, .. } = &op.access else {
        return Err(LiftDecline::NotContraction);
    };
    if !matches!(dtype, ElementKind::F32 | ElementKind::F64) {
        return Err(LiftDecline::UnsupportedForm("a non-float contraction"));
    }
    if !matches!(epilogue, ScalarExpr::Reduced(0)) {
        return Err(LiftDecline::UnsupportedForm("a fused epilogue or bias"));
    }
    let batched = *axes == ContractionAxes::batched_matmul();
    if !batched && *axes != ContractionAxes::matmul() {
        return Err(LiftDecline::UnsupportedForm(
            "a contraction other than matmul",
        ));
    }
    if op.n_inputs != 2 {
        return Err(LiftDecline::UnsupportedForm("a fused bias input"));
    }
    check_plain(op, dtype, if batched { 3 } else { 2 })?;
    Ok(batched)
}

/// Local ids of a lifted contraction: extents `0..axes` in the order `[B,] M, N, K`, counters
/// `axes..2*axes`, then the two loaded factors and the accumulator.
struct ContractionShape {
    dtype: ElementKind,
    batched: bool,
    /// Loop-axis count: 3 plain (M, N, K), 4 batched (B, M, N, K).
    axes: u32,
    m: u32,
    n: u32,
    k: u32,
    a: u32,
    b: u32,
    acc: u32,
}

impl ContractionShape {
    fn new(dtype: ElementKind, batched: bool) -> Self {
        let axes = if batched { 4 } else { 3 };
        let off = u32::from(batched);
        Self {
            dtype,
            batched,
            axes,
            m: off,
            n: off + 1,
            k: off + 2,
            a: 2 * axes,
            b: 2 * axes + 1,
            acc: 2 * axes + 2,
        }
    }

    fn counter(&self, d: u32) -> ScalarExpr {
        ScalarExpr::Local(self.axes + d)
    }

    /// The loop axes outside the `k` loop, outermost first: `[B,] M, N`.
    fn outer_axes(&self) -> Vec<u32> {
        (0..self.k).collect()
    }

    fn lhs_coords(&self) -> Vec<ScalarExpr> {
        let mut c: Vec<ScalarExpr> = if self.batched {
            vec![self.counter(0)]
        } else {
            vec![]
        };
        c.push(self.counter(self.m));
        c.push(self.counter(self.k));
        c
    }

    fn rhs_coords(&self) -> Vec<ScalarExpr> {
        let mut c: Vec<ScalarExpr> = if self.batched {
            vec![self.counter(0)]
        } else {
            vec![]
        };
        c.push(self.counter(self.k));
        c.push(self.counter(self.n));
        c
    }

    fn out_coords(&self) -> Vec<ScalarExpr> {
        let mut c: Vec<ScalarExpr> = if self.batched {
            vec![self.counter(0)]
        } else {
            vec![]
        };
        c.push(self.counter(self.m));
        c.push(self.counter(self.n));
        c
    }

    fn load(&self, buffer: u32, coords: &[ScalarExpr], local: u32) -> Stmt {
        Stmt::Load {
            local,
            place: Place::Logical {
                buffer,
                coords: coords.to_vec(),
            },
        }
    }

    /// `e` widened to the `F64` accumulator width (a no-op for `F64` compute).
    fn up(&self, e: ScalarExpr) -> ScalarExpr {
        if self.dtype == ElementKind::F64 {
            e
        } else {
            ScalarExpr::Cast(Box::new(e), widen(ElementKind::F64))
        }
    }

    fn stored(&self) -> ScalarExpr {
        let acc = ScalarExpr::Local(self.acc);
        if self.dtype == ElementKind::F64 {
            acc
        } else {
            ScalarExpr::Cast(Box::new(acc), widen(self.dtype))
        }
    }

    fn signature(&self) -> (Vec<Param>, Vec<LocalDecl>) {
        let rank = if self.batched { 3 } else { 2 };
        let buf = |name: &str, role| Param::buffer(name, self.dtype, role, Some(rank));
        let mut params = vec![
            buf("lhs", BufferRole::Read),
            buf("rhs", BufferRole::Read),
            buf("out", BufferRole::Write),
        ];
        let names = if self.batched {
            vec!["b", "m", "n", "k"]
        } else {
            vec!["m", "n", "k"]
        };
        params.extend(
            names
                .into_iter()
                .zip(0u32..)
                .map(|(nm, d)| Param::scalar(nm, ElementKind::I32, d)),
        );
        let mut locals = vec![LocalDecl::scalar(ElementKind::I32); 2 * self.axes as usize];
        locals.push(LocalDecl::scalar(self.dtype)); // a
        locals.push(LocalDecl::scalar(self.dtype)); // b
        locals.push(LocalDecl::scalar(ElementKind::F64)); // acc
        (params, locals)
    }
}
