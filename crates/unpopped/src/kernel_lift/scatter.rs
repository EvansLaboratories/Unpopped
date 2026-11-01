// SPDX-License-Identifier: MIT OR Apache-2.0
//! The scatter lift: an `Access::Elementwise` op whose output has a `WriteIndex::ScatterIndexed`
//! write (torch `scatter`, `scatter_add`, `index_add`).
//!
//! `out[.., index[coords], ..] (combine)= body(inputs)`: the iteration is the **source's** (the
//! destination can be smaller, as in a histogram), the index operand's value replaces the output
//! coordinate on `axis`, and an out-of-range target (`< 0` or `>= extent`) is skipped. `Assign` is a
//! store; `AtomicAdd`, `AtomicMax` and `AtomicMin` read the destination, combine and store it back,
//! which is exact because the kernel walks the cells serially in the oracle's order (so a duplicate
//! target under `Assign` is last-writer-wins in that same order). The kernel does not zero the
//! destination: the caller owns its initial contents, and the oracle's zero start is what makes
//! `AtomicAdd` a count.
//!
//! The destination's extent along `axis` is one more scalar parameter, typed as the index.

use unpopped_vocab::ElementKind;

use super::gather::place_at;
use super::im2col::{int, local};
use super::{Indexing, LiftDecline, check_plain_with, leaves_ok, substitute_inputs};
use crate::ir::{Access, BinaryOp, OobPolicy, OpDef, ReadIndex, ScalarExpr, WriteCombine};
use crate::kernel::{AtomicOp, Block, BufferRole, KernelFn, LocalDecl, Param, Stmt};

/// Lift a scatter: an Elementwise op with a `WriteIndex::ScatterIndexed` output (policy `Skip`,
/// which is the only one the plan admits), a single output of the compute dtype, identity views and
/// no base offsets, iterating `rank` source axes.
///
/// Parameters, in order: the `n_inputs` input buffers (`Read`; the index operand has the index
/// dtype, the others `dtype`), the output (`Write` for `Assign`, `ReadWrite` for the atomics), one
/// `I32` extent per source axis, then the destination extent along the scattered axis (index
/// dtype). A 1-D index (`index_add`) is a rank-`rank` buffer with stride 0 on the other axes.
///
/// # Errors
///
/// A [`LiftDecline`], `NotScatter` when the output is not scattered.
pub fn lift_scatter_opdef(
    op: &OpDef,
    dtype: ElementKind,
    rank: u8,
) -> Result<KernelFn, LiftDecline> {
    build(check_scatter(op, dtype, rank)?, op, dtype, rank)
}

/// [`lift_scatter_opdef`] launched on `grid` blocks of `block` threads (E1d-6): the outermost
/// source axis is a grid-stride loop and each combine is one [`Stmt::AtomicRmw`], so threads that
/// scatter to one destination cell never race. Integer `AtomicAdd`, `AtomicMax` and `AtomicMin`
/// are order-independent, so the destination is **bit-exact** against the serial kernel for every
/// `grid` and `block`. An `F32`/`F64` `AtomicAdd` is **not** bit-exact: the order threads reach a
/// cell is unspecified, so each cell equals the serial sum only inside the reassociation band of
/// `docs/ir-kernel-layer.md` section 5a (the destination's initial value is one summand).
///
/// # Errors
///
/// A [`LiftDecline`]: those of [`lift_scatter_opdef`]; `UnsupportedForm` for an `Assign` (two
/// sources can name one target, and the winner would depend on the schedule) and for a destination
/// that is not an integer, `F32` or `F64`; [`LiftDecline::BadLaunch`] for a zero `block` or
/// `grid`.
pub fn lift_scatter_opdef_launched(
    op: &OpDef,
    dtype: ElementKind,
    rank: u8,
    block: u32,
    grid: u32,
) -> Result<KernelFn, LiftDecline> {
    let mut s = check_scatter(op, dtype, rank)?;
    if matches!(s.combine, WriteCombine::Assign) {
        return Err(LiftDecline::UnsupportedForm(
            "a launched scatter that assigns (duplicate targets would race)",
        ));
    }
    if crate::types::int_range(s.out_dtype).is_none()
        && !matches!(s.out_dtype, ElementKind::F32 | ElementKind::F64)
    {
        return Err(LiftDecline::UnsupportedForm(
            "a launched scatter into a destination that is not an integer, F32 or F64",
        ));
    }
    s.atomic = true;
    super::grid_stride_outermost(build(s, op, dtype, rank)?, block, grid)
}

fn build(s: Scatter, op: &OpDef, dtype: ElementKind, rank: u8) -> Result<KernelFn, LiftDecline> {
    let mut body = s.cell(op);
    for d in (0..s.r).rev() {
        body = Block(vec![Stmt::For {
            local: s.counter(d),
            start: int(0),
            end: local(d),
            step: int(1),
            body,
        }]);
    }
    let (params, locals) = s.signature(rank);
    Ok(KernelFn::new(op.name.clone(), dtype, params, locals, body))
}

/// Local ids and the scatter's fields.
struct Scatter {
    /// Combine with an [`Stmt::AtomicRmw`] instead of a load and a store.
    atomic: bool,
    dtype: ElementKind,
    /// The destination's element type: the compute dtype, or an integer count type (`bincount`).
    out_dtype: ElementKind,
    /// The constant a `bincount` adds per element (its body is `Const(c)`, not a read).
    count: Option<i64>,
    index_dtype: ElementKind,
    r: u32,
    n: u32,
    index: u32,
    axis: u32,
    combine: WriteCombine,
}

fn check_scatter(op: &OpDef, dtype: ElementKind, rank: u8) -> Result<Scatter, LiftDecline> {
    if !matches!(op.access, Access::Elementwise) {
        return Err(LiftDecline::NotElementwise);
    }
    let Some((index, axis, combine, oob, index_dtype)) = op.write_index.scatter() else {
        return Err(LiftDecline::NotScatter);
    };
    if !matches!(
        index_dtype,
        ElementKind::I32 | ElementKind::I64 | ElementKind::U32
    ) {
        return Err(LiftDecline::UnsupportedForm("a non-integer scatter index"));
    }
    if !matches!(oob, OobPolicy::Skip) {
        return Err(LiftDecline::UnsupportedForm("a scatter that is not Skip"));
    }
    if !op.read_index.iter().all(ReadIndex::is_direct) {
        return Err(LiftDecline::UnsupportedForm("gather and scatter together"));
    }
    let is_int = crate::types::int_range(dtype).is_some();
    if matches!(combine, WriteCombine::AtomicMax | WriteCombine::AtomicMin) && !is_int {
        return Err(LiftDecline::UnsupportedForm("a float atomic max or min"));
    }
    // A per-output dtype is admitted for `bincount` only: an integer count destination that takes
    // `+= Const(c)` per source element.
    let (out_dtype, count) = count_output(op, dtype, combine)?;
    let mut plain = op.clone();
    plain.out_dtype = None;
    check_plain_with(&plain, dtype, rank, Indexing::Scatter, false)?;
    leaves_ok(&op.body)?;
    if usize::from(axis) >= usize::from(rank) || index >= op.n_inputs {
        return Err(LiftDecline::UnsupportedForm("an ill-formed scatter"));
    }
    Ok(Scatter {
        atomic: false,
        dtype,
        out_dtype,
        count,
        index_dtype,
        r: u32::from(rank),
        n: u32::from(op.n_inputs),
        index: u32::from(index),
        axis: u32::from(axis),
        combine,
    })
}

/// The destination dtype and, for a counting scatter, its per-element increment.
fn count_output(
    op: &OpDef,
    dtype: ElementKind,
    combine: WriteCombine,
) -> Result<(ElementKind, Option<i64>), LiftDecline> {
    let Some(out) = op.out_dtype else {
        return Ok((dtype, None));
    };
    let ScalarExpr::Const(c) = op.body else {
        return Err(LiftDecline::UnsupportedForm(
            "a per-output dtype that is not a count",
        ));
    };
    let counts = crate::types::int_range(out).is_some()
        && matches!(combine, WriteCombine::AtomicAdd)
        && c.fract() == 0.0
        && c.abs() <= 1.0e9;
    if !counts {
        return Err(LiftDecline::UnsupportedForm(
            "a per-output dtype that is not a count",
        ));
    }
    Ok((out, Some(c as i64)))
}

impl Scatter {
    fn counter(&self, d: u32) -> u32 {
        self.r + d
    }

    fn loaded(&self, k: u32) -> u32 {
        2 * self.r + k
    }

    /// The destination's extent along the scattered axis.
    fn extent(&self) -> u32 {
        2 * self.r + self.n
    }

    /// The destination cell's current value, for the atomics.
    fn prev(&self) -> u32 {
        self.extent() + 1
    }

    fn idx_const(&self, v: i64) -> ScalarExpr {
        ScalarExpr::ConstInt(v, self.index_dtype)
    }

    /// The destination place: the counters, with the scattered axis read from the loaded index.
    fn dest(&self) -> crate::kernel::Place {
        place_at(self.n, self.r, Some((self.axis, self.loaded(self.index))))
    }

    /// The store of one source cell into its destination.
    fn write(&self, op: &OpDef) -> Vec<Stmt> {
        let value = match self.count {
            Some(c) => ScalarExpr::ConstInt(c, self.out_dtype),
            None => substitute_inputs(&op.body, &|k| self.loaded(k)),
        };
        let combined = |f: &dyn Fn(Box<ScalarExpr>, Box<ScalarExpr>) -> ScalarExpr| {
            vec![
                Stmt::Load {
                    local: self.prev(),
                    place: self.dest(),
                },
                Stmt::Store {
                    place: self.dest(),
                    value: f(Box::new(local(self.prev())), Box::new(value.clone())),
                },
            ]
        };
        if self.atomic {
            let atomic = |op| {
                vec![Stmt::AtomicRmw {
                    op,
                    place: self.dest(),
                    value: value.clone(),
                }]
            };
            return match self.combine {
                WriteCombine::AtomicMax => atomic(AtomicOp::Max),
                WriteCombine::AtomicMin => atomic(AtomicOp::Min),
                _ => atomic(AtomicOp::Add),
            };
        }
        let extremum = |op| move |a, b| ScalarExpr::Binary(op, a, b);
        match self.combine {
            WriteCombine::Assign => vec![Stmt::Store {
                place: self.dest(),
                value: value.clone(),
            }],
            WriteCombine::AtomicAdd => combined(&ScalarExpr::Add),
            WriteCombine::AtomicMax => combined(&extremum(BinaryOp::Max)),
            WriteCombine::AtomicMin => combined(&extremum(BinaryOp::Min)),
        }
    }

    /// One source cell: load the inputs, then write unless the target is out of range.
    fn cell(&self, op: &OpDef) -> Block {
        let mut cell: Vec<Stmt> = (0..self.n)
            .map(|k| Stmt::Load {
                local: self.loaded(k),
                place: place_at(k, self.r, None),
            })
            .collect();
        let raw = || local(self.loaded(self.index));
        let cmp = |op, rhs| ScalarExpr::Binary(op, Box::new(raw()), Box::new(rhs));
        let upper = Stmt::If {
            cond: cmp(BinaryOp::CmpGe, local(self.extent())),
            then: Block(vec![]),
            els: Block(self.write(op)),
        };
        cell.push(Stmt::If {
            cond: cmp(BinaryOp::CmpLt, self.idx_const(0)),
            then: Block(vec![]),
            els: Block(vec![upper]),
        });
        Block(cell)
    }

    fn signature(&self, rank: u8) -> (Vec<Param>, Vec<LocalDecl>) {
        let elem = |k: u32| {
            if k == self.index {
                self.index_dtype
            } else {
                self.dtype
            }
        };
        let mut params: Vec<Param> = (0..self.n)
            .map(|k| Param::buffer(format!("in{k}"), elem(k), BufferRole::Read, Some(rank)))
            .collect();
        let role = if matches!(self.combine, WriteCombine::Assign) {
            BufferRole::Write
        } else {
            BufferRole::ReadWrite
        };
        params.push(Param::buffer("out", self.out_dtype, role, Some(rank)));
        params.extend((0..self.r).map(|d| Param::scalar(format!("n{d}"), ElementKind::I32, d)));
        params.push(Param::scalar("dst_extent", self.index_dtype, self.extent()));
        let mut locals = vec![LocalDecl::scalar(ElementKind::I32); 2 * self.r as usize];
        locals.extend((0..self.n).map(|k| LocalDecl::scalar(elem(k))));
        locals.push(LocalDecl::scalar(self.index_dtype)); // dst_extent
        locals.push(LocalDecl::scalar(self.out_dtype)); // prev
        (params, locals)
    }
}
