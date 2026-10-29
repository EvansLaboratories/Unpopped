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
    ScalarExpr, View, WriteIndex,
};
use crate::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt};
use crate::types::{int_range, map_children};

mod gather;
mod im2col;
mod post;
mod row_reduce;
mod row_sort;
mod window;
pub use gather::lift_gather_opdef;
pub use im2col::lift_im2col_opdef;
pub use row_reduce::lift_row_reduce_opdef;
pub use row_sort::lift_row_sort_opdef;
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
    /// The op is not `Access::RowSort` (for [`lift_row_sort_opdef`]).
    NotRowSort,
    /// No input has a `ReadIndex::Indexed` read (for [`lift_gather_opdef`]).
    NotGather,
    /// More than one output, a per-output dtype, a gather/scatter (outside their own lifts), a
    /// runtime base offset outside Elementwise or a non-identity view: forms this increment does
    /// not lift.
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
    let loaded = |k: u32| 2 * r + k;
    let mut inner: Vec<Stmt> = (0..n)
        .map(|k| Stmt::Load {
            local: loaded(k),
            place: counter_place(k, r, op.views.get(k as usize)),
        })
        .collect();
    inner.push(Stmt::Store {
        place: counter_place(n, r, None),
        value: substitute_inputs(&op.body, &loaded),
    });
    // Wrap outward: the last axis is the innermost loop (row-major order).
    let mut body = Block(inner);
    for d in (0..r).rev() {
        body = Block(vec![Stmt::For {
            local: r + d,
            start: ScalarExpr::ConstInt(0, ElementKind::I32),
            end: ScalarExpr::Local(d),
            step: ScalarExpr::ConstInt(1, ElementKind::I32),
            body,
        }]);
    }
    let (params, locals) = elementwise_signature(op.n_inputs, dtype, rank);
    Ok(KernelFn::new(op.name.clone(), dtype, params, locals, body))
}

/// `buffer` at the loop counters of a rank-`r` kernel (counters are locals `r..2r`). An input read
/// through a `Permute` view takes its coordinates in the producer's axis order: iteration axis `d`
/// indexes producer axis `perm[d]`, so the producer coordinate on `perm[d]` is the counter of `d`.
fn counter_place(buffer: u32, r: u32, view: Option<&View>) -> Place {
    let mut coords: Vec<ScalarExpr> = (0..r).map(|d| ScalarExpr::Local(r + d)).collect();
    if let Some(View::Permute { perm }) = view {
        for (d, &a) in perm.iter().enumerate() {
            coords[usize::from(a)] = ScalarExpr::Local(r + d as u32);
        }
    }
    Place::Logical { buffer, coords }
}

/// The Elementwise kernel's parameters (inputs, output, one `I32` extent per axis) and locals
/// (extents, counters, one loaded value per input).
fn elementwise_signature(
    n_inputs: u8,
    dtype: ElementKind,
    rank: u8,
) -> (Vec<Param>, Vec<LocalDecl>) {
    let (r, n) = (u32::from(rank), u32::from(n_inputs));
    let mut params: Vec<Param> = (0..n)
        .map(|k| Param::buffer(format!("in{k}"), dtype, BufferRole::Read, Some(rank)))
        .collect();
    params.push(Param::buffer("out", dtype, BufferRole::Write, Some(rank)));
    params.extend((0..r).map(|d| Param::scalar(format!("n{d}"), ElementKind::I32, d)));
    let mut locals = vec![LocalDecl::scalar(ElementKind::I32); 2 * usize::from(rank)];
    locals.extend((0..n).map(|_| LocalDecl::scalar(dtype)));
    (params, locals)
}

fn check_liftable(op: &OpDef, dtype: ElementKind, rank: u8) -> Result<(), LiftDecline> {
    if !matches!(op.access, Access::Elementwise) {
        return Err(LiftDecline::NotElementwise);
    }
    check_plain_with(op, dtype, rank, false, true)?;
    leaves_ok(&op.body)
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
    check_plain_with(op, dtype, rank, false, false)
}

/// [`check_plain`], optionally admitting a gathered read (only the gather lift builds the
/// address math for one) and per-input views (only the Elementwise lift builds the read-through
/// for them; every other lift would silently ignore one).
fn check_plain_with(
    op: &OpDef,
    dtype: ElementKind,
    rank: u8,
    allow_gather: bool,
    views: bool,
) -> Result<(), LiftDecline> {
    if rank == 0 || rank > 8 {
        return Err(LiftDecline::BadRank);
    }
    if op.output_bodies().len() != 1 || op.out_dtype.is_some() {
        return Err(LiftDecline::UnsupportedForm(
            "multi-output or a per-output dtype",
        ));
    }
    if (!allow_gather && !op.read_index.is_empty()) || !matches!(op.write_index, WriteIndex::Direct)
    {
        return Err(LiftDecline::UnsupportedForm("gather or scatter"));
    }
    // A runtime base offset is a launch argument added to the operand's base before any address
    // math, which is what an argument buffer's `base_offset` is in the kernel layer: the lifted
    // kernel is the offset-free one, and the caller binds the offset in the buffer. The plan admits
    // one on an Elementwise op only.
    if !op.base_offsets.is_empty() && op.base_offsets.len() != usize::from(op.n_inputs) {
        return Err(LiftDecline::UnsupportedForm("an ill-formed base_offsets"));
    }
    let offset = op.base_offsets.iter().any(|b| !b.is_zero()) || !op.out_base_offset.is_zero();
    if offset && !matches!(op.access, Access::Elementwise) {
        return Err(LiftDecline::UnsupportedForm(
            "a runtime base offset outside Elementwise",
        ));
    }
    if op.views.iter().any(|v| !v.is_identity()) {
        if !views {
            return Err(LiftDecline::UnsupportedForm("a non-identity view"));
        }
        // A view is a per-input read-through, `views[i]` for `Input(i)`, well formed for the
        // iteration rank.
        if op.views.len() != usize::from(op.n_inputs) || !op.views.iter().all(|v| v.is_valid(rank))
        {
            return Err(LiftDecline::UnsupportedForm("an ill-formed view"));
        }
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
/// reduced extents' product. The identity post-expression (`Reduced(0)`) is byte-exact. A fused
/// post lifts in tolerance mode: the oracle runs it over an unrounded `F64` value, which the typed
/// kernel language cannot state, so the lift binds `Reduced(0)` to the finalised fold rounded once
/// to the compute dtype (`docs/ir-kernel-layer.md` 7.1); it must not read an `Input`.
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
            init: sh.fold_identity(rop),
        },
        Stmt::Let {
            local: sh.have,
            init: ScalarExpr::ConstInt(0, ElementKind::I32),
        },
    ]);
    body.0.extend(sh.reduce_loop(op, rop).0);
    let out_coords = sh.out_coords(keepdim);
    let out_rank = u8::try_from(out_coords.len()).expect("rank is at most 8");
    let value = sh.stored_value(rop, post, &mut body);
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
/// `Cast` down at each `Store`. The identity post-expression (`Reduced(0)`) is byte-exact; a fused
/// post lifts in tolerance mode, with `Reduced(0)` bound to the prefix rounded once to the compute
/// dtype and its `Input(k)` leaves to the elements loaded at the position (`docs/ir-kernel-layer.md`
/// 7.1), as a reduction's does.
///
/// Parameters and locals are laid out as for [`lift_reduction_opdef`], with the output the same
/// rank as the input.
///
/// # Errors
///
/// A [`LiftDecline`].
pub fn lift_scan_opdef(op: &OpDef, dtype: ElementKind, rank: u8) -> Result<KernelFn, LiftDecline> {
    let (sop, reverse, exclusive, pre, post) = check_scan(op, dtype, rank)?;
    let sh = ReductionShape::new(op.n_inputs, dtype, rank, AxisMask::EMPTY);
    let pass = scan_pass(&sh, sop, exclusive, pre, post);
    let mut row = scan_row(&sh, sop, reverse, pass);
    for &d in sh.kept.iter().rev() {
        row = for_axis(d, sh.r, row);
    }
    let (params, mut locals) = sh.signature(u8::try_from(sh.r).expect("rank is at most 8"));
    if post.is_some() {
        locals.push(LocalDecl::scalar(dtype)); // red: the prefix rounded to the compute dtype
    }
    Ok(KernelFn::new(op.name.clone(), dtype, params, locals, row))
}

/// One position of the scan: load the inputs, pre-map, fold into the prefix, and write the prefix
/// after the fold (inclusive) or before it (exclusive).
fn scan_pass(
    sh: &ReductionShape,
    sop: ReduceOp,
    exclusive: bool,
    pre: &ScalarExpr,
    post: Option<&ScalarExpr>,
) -> Vec<Stmt> {
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
    let write = scan_write(sh, sop, post, all_counters());
    let fold = fold_step(sop, sh.acc, sh.have, sh.val, sh.is_int());
    if exclusive {
        pass.extend(write);
        pass.extend(fold);
    } else {
        pass.extend(fold);
        pass.extend(write);
    }
    pass
}

/// Write the prefix at the current position. With a fused post, the prefix is first rounded once
/// to the compute dtype (`red`) and the post reads it as `Reduced(0)`, its `Input(k)` leaves being
/// the elements loaded for this position (`docs/ir-kernel-layer.md` 7.1).
fn scan_write(
    sh: &ReductionShape,
    sop: ReduceOp,
    post: Option<&ScalarExpr>,
    coords: Vec<ScalarExpr>,
) -> Vec<Stmt> {
    let place = Place::Logical {
        buffer: sh.n,
        coords,
    };
    let Some(post) = post else {
        return vec![Stmt::Store {
            place,
            value: sh.stored(sop),
        }];
    };
    let value = post::bind(post, &|leaf| match leaf {
        ScalarExpr::Reduced(_) => Some(ScalarExpr::Local(sh.red)),
        ScalarExpr::Input(k) => Some(ScalarExpr::Local(sh.loaded(u32::from(*k)))),
        _ => None,
    });
    vec![
        Stmt::Let {
            local: sh.red,
            init: sh.stored(sop),
        },
        Stmt::Store { place, value },
    ]
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
) -> Result<(ReduceOp, bool, bool, &ScalarExpr, Option<&ScalarExpr>), LiftDecline> {
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
    if matches!(sop, ReduceOp::Mean) {
        return Err(LiftDecline::UnsupportedForm("a Mean scan (not a monoid)"));
    }
    check_accumulated_dtype(dtype, *sop, post, "a scan")?;
    let post = if matches!(post, ScalarExpr::Reduced(0)) {
        None
    } else {
        post::check_post(post, 1)?;
        Some(post)
    };
    if u32::from(*axis) + 1 != u32::from(rank) {
        return Err(LiftDecline::UnsupportedForm(
            "a scan over a non-innermost axis",
        ));
    }
    check_plain(op, dtype, rank)?;
    leaves_ok(pre)?;
    Ok((*sop, *reverse, *exclusive, pre, post))
}

/// The scan monoid identity (the oracle's `scan_identity`, float domain) as an expression of the
/// compute dtype. The infinities are built from their bit patterns (`Bitcast` of an integer
/// constant) because a JSON number cannot be infinite and a kernel is authored as JSON.
fn scan_identity(rop: ReduceOp, dtype: ElementKind) -> ScalarExpr {
    if let Some((lo, hi)) = int_range(dtype) {
        // The integer extremes (the oracle's `int_extreme`); only I32 and I64 lift, so they fit.
        let v = match rop {
            ReduceOp::Prod => 1,
            ReduceOp::Max => lo,
            ReduceOp::Min => hi,
            ReduceOp::Sum | ReduceOp::Mean => 0,
        };
        return ScalarExpr::ConstInt(i64::try_from(v).expect("an I32 or I64 extreme"), dtype);
    }
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

/// Compute dtypes whose reductions and scans lift as integers: the plan's integer dtypes that the
/// kernel interpreter folds exactly at their own width.
fn is_int_kernel_dtype(dtype: ElementKind) -> bool {
    matches!(dtype, ElementKind::I32 | ElementKind::I64)
}

/// A reduction or scan lifts at a float dtype, or at an integer one with no `Mean` (an integer
/// mean has no oracle rounding to match) and no fused post (whose tolerance mode is float-only).
fn check_accumulated_dtype(
    dtype: ElementKind,
    rop: ReduceOp,
    post: &ScalarExpr,
    what: &'static str,
) -> Result<(), LiftDecline> {
    if matches!(dtype, ElementKind::F32 | ElementKind::F64) {
        return Ok(());
    }
    if !is_int_kernel_dtype(dtype) {
        return Err(LiftDecline::UnsupportedForm(match what {
            "a scan" => "a scan outside F32, F64, I32 and I64",
            "a window" => "a window outside F32, F64, I32 and I64",
            _ => "a reduction outside F32, F64, I32 and I64",
        }));
    }
    if matches!(rop, ReduceOp::Mean) {
        return Err(LiftDecline::UnsupportedForm("an integer Mean"));
    }
    if !matches!(post, ScalarExpr::Reduced(0)) {
        return Err(LiftDecline::UnsupportedForm(
            "a fused post at an integer dtype",
        ));
    }
    Ok(())
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
    check_accumulated_dtype(dtype, *rop, post, "a reduction")?;
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

    /// An integer compute dtype: the accumulator is the dtype itself (the oracle folds integers
    /// exactly and wraps, and wrapping at the dtype's width commutes with the store's truncation).
    fn is_int(&self) -> bool {
        is_int_kernel_dtype(self.dtype)
    }

    /// The accumulator's and the current value's type.
    fn acc_ty(&self) -> ElementKind {
        if self.is_int() {
            self.dtype
        } else {
            ElementKind::F64
        }
    }

    /// The fold's starting value: the monoid's `0` or `1` (a `Max`/`Min` is guarded by `have`).
    fn fold_identity(&self, rop: ReduceOp) -> ScalarExpr {
        let one = matches!(rop, ReduceOp::Prod);
        if self.is_int() {
            ScalarExpr::ConstInt(i64::from(one), self.dtype)
        } else {
            self.up(ScalarExpr::Const(if one { 1.0 } else { 0.0 }))
        }
    }

    /// `e` widened to the `F64` accumulator width (a no-op for `F64` compute and for integers,
    /// whose accumulator is the dtype).
    fn up(&self, e: ScalarExpr) -> ScalarExpr {
        if self.dtype == ElementKind::F64 || self.is_int() {
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
        inner.extend(fold_step(rop, self.acc, self.have, self.val, self.is_int()));
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
        if self.dtype == ElementKind::F64 || self.is_int() {
            finalized
        } else {
            ScalarExpr::Cast(Box::new(finalized), widen(self.dtype))
        }
    }

    /// The value to store. With a fused post, `red` (the finalised fold rounded once to the
    /// compute dtype) is declared at the end of `body` and the post reads it as `Reduced(0)`.
    fn stored_value(
        &self,
        rop: ReduceOp,
        post: Option<&ScalarExpr>,
        body: &mut Block,
    ) -> ScalarExpr {
        let Some(post) = post else {
            return self.stored(rop);
        };
        body.0.push(Stmt::Let {
            local: self.red,
            init: self.stored(rop),
        });
        post::bind(post, &|leaf| {
            matches!(leaf, ScalarExpr::Reduced(_)).then(|| ScalarExpr::Local(self.red))
        })
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
        locals.push(LocalDecl::scalar(self.acc_ty())); // acc
        locals.push(LocalDecl::scalar(ElementKind::I32)); // have
        locals.push(LocalDecl::scalar(self.acc_ty())); // val
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
fn fold_step(rop: ReduceOp, acc: u32, have: u32, val: u32, int: bool) -> Vec<Stmt> {
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
        ReduceOp::Max | ReduceOp::Min => extremum_step(rop, acc, have, val, int),
    }
}

/// `Max` / `Min` step: `if !have { acc = v } else if v != v { acc = v } else if better(v, acc)
/// { acc = v }`, then `have = 1` (the oracle's `!have || v.is_nan() || better`).
fn extremum_step(rop: ReduceOp, acc: u32, have: u32, val: u32, int: bool) -> Vec<Stmt> {
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
    // An integer is never NaN, so its step has no NaN arm.
    let nan_check = if int {
        keep_going
    } else {
        Stmt::If {
            cond: ScalarExpr::Binary(BinaryOp::CmpNe, bx(l(val)), bx(l(val))),
            then: Block(vec![set(val)]),
            els: Block(vec![keep_going]),
        }
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
/// at `dtype` (`F32` or `F64`). The identity epilogue (`Reduced(0)`) is byte-exact; a fused
/// epilogue lifts in tolerance mode, with `Reduced(0)` bound to the K-sum rounded once to the
/// compute dtype (`docs/ir-kernel-layer.md` 7.1). The epilogue may read only the fused per-column
/// bias inputs `Input(2..n_inputs)` of a plain matmul (a `[N]` buffer broadcast over the rows,
/// the oracle's rule; a batched contraction carries no bias).
///
/// The oracle reads each operand by its physical strides at the logical `(m, k)` / `(k, n)`
/// coordinate (so transposed and broadcast operands need nothing extra: that is what
/// `Place::Logical` does), widens both factors to `f64`, multiplies and accumulates in `f64` with
/// `k` ascending, and rounds once at the store. The kernel does the same: nested `For` over
/// `b`, `m`, `n`, an inner `For` over `k`, an `F64` accumulator, `Cast` up on each factor.
///
/// Parameters: `lhs`, `rhs` (`Read`), `out` (`Write`), one rank-1 `Read` buffer per fused bias
/// (`Input(2)` is buffer 3, and so on), then `I32` extents `[B,] M, N, K`.
///
/// # Errors
///
/// A [`LiftDecline`].
pub fn lift_contraction_opdef(op: &OpDef, dtype: ElementKind) -> Result<KernelFn, LiftDecline> {
    let (batched, epilogue) = check_contraction(op, dtype)?;
    let sh = ContractionShape::new(dtype, batched, op.n_inputs.saturating_sub(2));
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
    let value = sh.stored_value(epilogue, &mut body);
    body.0.push(Stmt::Store {
        place: Place::Logical {
            buffer: 2,
            coords: sh.out_coords(),
        },
        value,
    });
    for d in sh.outer_axes().into_iter().rev() {
        body = for_axis(d, sh.axes, body);
    }
    let (params, mut locals) = sh.signature();
    if epilogue.is_some() {
        locals.push(LocalDecl::scalar(dtype)); // red: the K-sum rounded to the compute dtype
        locals.extend((0..sh.biases).map(|_| LocalDecl::scalar(dtype))); // one loaded bias each
    }
    Ok(KernelFn::new(op.name.clone(), dtype, params, locals, body))
}

/// `true` when batched. The op must be exactly a plain or a batched matmul (a bias only on the
/// plain one) with float compute; the epilogue reads `Reduced(0)` and the bias inputs only.
fn check_contraction(
    op: &OpDef,
    dtype: ElementKind,
) -> Result<(bool, Option<&ScalarExpr>), LiftDecline> {
    let Access::Contraction { axes, epilogue, .. } = &op.access else {
        return Err(LiftDecline::NotContraction);
    };
    if !matches!(dtype, ElementKind::F32 | ElementKind::F64) {
        return Err(LiftDecline::UnsupportedForm("a non-float contraction"));
    }
    let epilogue = if matches!(epilogue, ScalarExpr::Reduced(0)) {
        None
    } else {
        post::check_post(epilogue, 1)?;
        if post::reads_input_outside(epilogue, 2, op.n_inputs) {
            return Err(LiftDecline::UnsupportedLeaf(
                "an Input in a contraction epilogue",
            ));
        }
        Some(epilogue)
    };
    let batched = *axes == ContractionAxes::batched_matmul();
    if !batched && *axes != ContractionAxes::matmul() {
        return Err(LiftDecline::UnsupportedForm(
            "a contraction other than matmul",
        ));
    }
    if op.n_inputs < 2 || (batched && op.n_inputs != 2) {
        return Err(LiftDecline::UnsupportedForm("a batched fused bias input"));
    }
    check_plain_with(op, dtype, if batched { 3 } else { 2 }, false, true)?;
    // A view on the two factors is recognition metadata: the oracle reads them through the bound
    // operand's own strides (a transposed operand is a transposed binding), so the kernel is the
    // view-free one. A fused bias is read through its view, which this lift does not build.
    if op.views.iter().skip(2).any(|v| !v.is_identity()) {
        return Err(LiftDecline::UnsupportedForm("a view on a fused bias"));
    }
    Ok((batched, epilogue))
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
    /// The K-sum rounded to the compute dtype (declared only when an epilogue is lifted).
    red: u32,
    /// Fused bias inputs: bias `j` is buffer `3 + j` and, with an epilogue, local `red + 1 + j`.
    biases: u32,
}

impl ContractionShape {
    fn new(dtype: ElementKind, batched: bool, biases: u8) -> Self {
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
            red: 2 * axes + 3,
            biases: u32::from(biases),
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

    /// The value to store. With a fused epilogue, `red` (the K-sum rounded once to the compute
    /// dtype) is declared at the end of `body` and the epilogue reads it as `Reduced(0)`.
    fn stored_value(&self, epilogue: Option<&ScalarExpr>, body: &mut Block) -> ScalarExpr {
        let Some(epilogue) = epilogue else {
            return self.stored();
        };
        body.0.push(Stmt::Let {
            local: self.red,
            init: self.stored(),
        });
        for j in 0..self.biases {
            body.0
                .push(self.load(3 + j, &[self.counter(self.n)], self.red + 1 + j));
        }
        post::bind(epilogue, &|leaf| match leaf {
            ScalarExpr::Reduced(_) => Some(ScalarExpr::Local(self.red)),
            ScalarExpr::Input(i) => Some(ScalarExpr::Local(self.red + u32::from(*i) - 1)),
            _ => None,
        })
    }

    fn signature(&self) -> (Vec<Param>, Vec<LocalDecl>) {
        let rank = if self.batched { 3 } else { 2 };
        let buf = |name: &str, role| Param::buffer(name, self.dtype, role, Some(rank));
        let mut params = vec![
            buf("lhs", BufferRole::Read),
            buf("rhs", BufferRole::Read),
            buf("out", BufferRole::Write),
        ];
        params.extend(
            (0..self.biases)
                .map(|j| Param::buffer(format!("bias{j}"), self.dtype, BufferRole::Read, Some(1))),
        );
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
