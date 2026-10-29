// SPDX-License-Identifier: MIT OR Apache-2.0
//! The `Access::RowReduce` lift: per-row staged reductions, then a full-width epilogue.
//!
//! **This lift is in tolerance mode, not byte-exact mode.** The oracle hands each stage's finalised
//! value to the later stages and to the epilogue as an unrounded `f64` (`Reduced(s)`), and the
//! kernel language has no way to say "an `f64` leaf inside a compute-dtype expression". The lift
//! therefore rounds each stage once to the compute dtype, `Reduced(s) := Cast(f64 acc -> compute)`,
//! and runs the later stages and the epilogue on that. At `F64` compute the cast is the identity
//! and the kernel is byte-exact; at `F32` it differs from the oracle by the rounding of the stage
//! values, which `tests/kernel_row_reduce_agrees.rs` bounds at 2 ulp. The stage folds themselves
//! (accumulator width, ascending order, NaN-keeping `Max`/`Min`) are exactly the oracle's.

use unpopped_vocab::ElementKind;

use super::im2col::{int, local};
use super::{LiftDecline, check_plain, fold_step, leaves_ok, widen};
use crate::ir::{Access, OpDef, ReduceOp, ReduceStage, ScalarExpr};
use crate::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt};
use crate::types::map_children;

/// Lift a float `Access::RowReduce` over `rank` axes (the reduced axis is the innermost; the output
/// has the input's shape), computed at `dtype` (`F32` or `F64`). See the module docs for the
/// tolerance this lift is held to.
///
/// Parameters: the inputs (`Read`, rank `rank`), the output (`Write`, rank `rank`), then one `I32`
/// extent per axis.
///
/// # Errors
///
/// A [`LiftDecline`].
pub fn lift_row_reduce_opdef(
    op: &OpDef,
    dtype: ElementKind,
    rank: u8,
) -> Result<KernelFn, LiftDecline> {
    let (stages, epilogue) = check_row_reduce(op, dtype, rank)?;
    let w = Row::new(op.n_inputs, dtype, rank, stages.len());
    let mut row: Vec<Stmt> = Vec::new();
    for (s, stage) in stages.iter().enumerate() {
        row.extend(w.stage(s, stage));
    }
    row.push(w.epilogue(epilogue));
    let mut body = Block(row);
    for d in (0..w.r - 1).rev() {
        body = w.for_lead(d, body);
    }
    let (params, locals) = w.signature();
    Ok(KernelFn::new(op.name.clone(), dtype, params, locals, body))
}

fn check_row_reduce(
    op: &OpDef,
    dtype: ElementKind,
    rank: u8,
) -> Result<(&[ReduceStage], &ScalarExpr), LiftDecline> {
    let Access::RowReduce { stages, epilogue } = &op.access else {
        return Err(LiftDecline::NotRowReduce);
    };
    if !matches!(dtype, ElementKind::F32 | ElementKind::F64) {
        return Err(LiftDecline::UnsupportedForm("a non-float row reduction"));
    }
    if stages.is_empty() || stages.iter().any(|s| matches!(s.op, ReduceOp::Prod)) {
        return Err(LiftDecline::UnsupportedForm("a Prod or empty stage list"));
    }
    check_plain(op, dtype, rank)?;
    for (s, stage) in stages.iter().enumerate() {
        leaves_ok_but_reduced(&stage.pre, s)?;
    }
    leaves_ok_but_reduced(epilogue, stages.len())?;
    Ok((stages, epilogue))
}

/// [`leaves_ok`] for an expression that may read `Reduced(t)` for `t < available`.
fn leaves_ok_but_reduced(e: &ScalarExpr, available: usize) -> Result<(), LiftDecline> {
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

/// Local ids. Extents `0..r`; counters `r..2r` (the last is the position along the row); then, per
/// phase (each stage, then the epilogue), one loaded local per input; then, per stage, the `F64`
/// accumulator, the `have` flag, the current `F64` value and the rounded stage result.
struct Row {
    dtype: ElementKind,
    r: u32,
    n: u32,
    stages: u32,
}

impl Row {
    fn new(n_inputs: u8, dtype: ElementKind, rank: u8, stages: usize) -> Self {
        Self {
            dtype,
            r: u32::from(rank),
            n: u32::from(n_inputs),
            stages: u32::try_from(stages).expect("a handful of stages"),
        }
    }

    fn counter(&self, d: u32) -> u32 {
        self.r + d
    }

    fn loaded(&self, phase: u32, k: u32) -> u32 {
        2 * self.r + phase * self.n + k
    }

    /// First local of stage `s`'s four.
    fn stage_base(&self, s: u32) -> u32 {
        2 * self.r + (self.stages + 1) * self.n + 4 * s
    }

    fn up(&self, e: ScalarExpr) -> ScalarExpr {
        if self.dtype == ElementKind::F64 {
            e
        } else {
            ScalarExpr::Cast(Box::new(e), widen(ElementKind::F64))
        }
    }

    fn down(&self, e: ScalarExpr) -> ScalarExpr {
        if self.dtype == ElementKind::F64 {
            e
        } else {
            ScalarExpr::Cast(Box::new(e), widen(self.dtype))
        }
    }

    /// `for counter(d) in 0..extent(d) { inner }`.
    fn for_lead(&self, d: u32, inner: Block) -> Block {
        Block(vec![Stmt::For {
            local: self.counter(d),
            start: int(0),
            end: local(d),
            step: int(1),
            body: inner,
        }])
    }

    /// The expression with `Input(k)` read from phase `phase`'s loaded locals and `Reduced(t)` from
    /// stage `t`'s rounded result.
    fn bind(&self, e: &ScalarExpr, phase: u32) -> ScalarExpr {
        match e {
            ScalarExpr::Input(k) => local(self.loaded(phase, u32::from(*k))),
            ScalarExpr::Reduced(t) => local(self.stage_base(u32::from(*t)) + 3),
            _ => {
                let r: Result<ScalarExpr, std::convert::Infallible> =
                    map_children(e, &mut |c| Ok(self.bind(c, phase)));
                match r {
                    Ok(e) => e,
                    Err(never) => match never {},
                }
            }
        }
    }

    fn loads(&self, phase: u32) -> Vec<Stmt> {
        (0..self.n)
            .map(|k| Stmt::Load {
                local: self.loaded(phase, k),
                place: Place::Logical {
                    buffer: k,
                    coords: (0..self.r).map(|d| local(self.counter(d))).collect(),
                },
            })
            .collect()
    }

    /// One stage: fold the row into an `F64` accumulator, then round the finalised value once.
    fn stage(&self, s: usize, stage: &ReduceStage) -> Vec<Stmt> {
        let s = u32::try_from(s).expect("a handful of stages");
        let base = self.stage_base(s);
        let (acc, have, val, red) = (base, base + 1, base + 2, base + 3);
        let mut inner = self.loads(s);
        inner.push(Stmt::Let {
            local: val,
            init: self.up(self.bind(&stage.pre, s)),
        });
        inner.extend(fold_step(stage.op, acc, have, val));
        let last = self.r - 1;
        let finalised = if matches!(stage.op, ReduceOp::Mean) {
            let k = ScalarExpr::Cast(Box::new(local(last)), widen(ElementKind::F64));
            ScalarExpr::Div(Box::new(local(acc)), Box::new(k))
        } else {
            local(acc)
        };
        vec![
            Stmt::Let {
                local: acc,
                init: self.up(ScalarExpr::Const(0.0)),
            },
            Stmt::Let {
                local: have,
                init: int(0),
            },
            self.for_lead(last, Block(inner)).0.remove(0),
            Stmt::Let {
                local: red,
                init: self.down(finalised),
            },
        ]
    }

    /// The full-width output: the epilogue over the inputs and the rounded stage results.
    fn epilogue(&self, epilogue: &ScalarExpr) -> Stmt {
        let mut inner = self.loads(self.stages);
        inner.push(Stmt::Store {
            place: Place::Logical {
                buffer: self.n,
                coords: (0..self.r).map(|d| local(self.counter(d))).collect(),
            },
            value: self.bind(epilogue, self.stages),
        });
        self.for_lead(self.r - 1, Block(inner)).0.remove(0)
    }

    fn signature(&self) -> (Vec<Param>, Vec<LocalDecl>) {
        let rank = u8::try_from(self.r).expect("rank is at most 8");
        let mut params: Vec<Param> = (0..self.n)
            .map(|k| Param::buffer(format!("in{k}"), self.dtype, BufferRole::Read, Some(rank)))
            .collect();
        params.push(Param::buffer(
            "out",
            self.dtype,
            BufferRole::Write,
            Some(rank),
        ));
        params.extend((0..self.r).map(|d| Param::scalar(format!("n{d}"), ElementKind::I32, d)));
        let mut locals = vec![LocalDecl::scalar(ElementKind::I32); 2 * self.r as usize];
        locals.extend((0..(self.stages + 1) * self.n).map(|_| LocalDecl::scalar(self.dtype)));
        for _ in 0..self.stages {
            locals.push(LocalDecl::scalar(ElementKind::F64)); // acc
            locals.push(LocalDecl::scalar(ElementKind::I32)); // have
            locals.push(LocalDecl::scalar(ElementKind::F64)); // val
            locals.push(LocalDecl::scalar(self.dtype)); // red
        }
        (params, locals)
    }
}
