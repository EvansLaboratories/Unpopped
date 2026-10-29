// SPDX-License-Identifier: MIT OR Apache-2.0
//! The gather lift: an `Access::Elementwise` op with one `ReadIndex::Indexed` input.
//!
//! `out[.., c, ..] = body(data[.., index[coords], ..], other inputs)`: the iteration coordinate on
//! `axis` is replaced, for the data operand only, by a value loaded from the integer index operand.
//! The kernel does what the oracle does: load the index once per cell, clamp it into the source's
//! extent (so no load is ever out of bounds), then apply the out-of-range policy to the **store**:
//! `Skip` leaves the cell alone, `ZeroFill` writes zero, `Clamp` stores the clamped read.
//!
//! The source's extent along `axis` is not among the iteration extents (the output may be shorter
//! or longer), so it is one more scalar parameter, of the index dtype so the comparisons are
//! same-typed.

use unpopped_vocab::ElementKind;

use super::im2col::{int, local};
use super::{LiftDecline, check_plain_with, leaves_ok, substitute_inputs};
use crate::ir::{Access, BinaryOp, OobPolicy, OpDef, ReadIndex, ScalarExpr, WriteIndex};
use crate::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt};

/// Lift a gather: an Elementwise op with exactly one `ReadIndex::Indexed` input, a single output,
/// identity views and no base offsets, computed at `dtype` over `rank` output axes.
///
/// Parameters, in order: the `n_inputs` input buffers (`Read`; the index operand has the index
/// dtype, the others `dtype`), the output (`Write`), one `I32` extent per output axis, then the
/// source extent along the gathered axis (index dtype). Locals: extents `0..rank`, counters
/// `rank..2*rank`, loaded inputs `2*rank..2*rank+n`, then the source extent, the clamped index and
/// the zero-fill predicate's operands share those. The index buffer must carry the stride-0
/// broadcast layout a 1-D index (an embedding's ids) has, as the oracle reads it through its own
/// strides.
///
/// # Errors
///
/// A [`LiftDecline`], `NotGather` when no input is indexed.
pub fn lift_gather_opdef(
    op: &OpDef,
    dtype: ElementKind,
    rank: u8,
) -> Result<KernelFn, LiftDecline> {
    let g = check_gather(op, dtype, rank)?;
    let s = Shape::new(op.n_inputs, dtype, rank, &g);
    let mut body = s.cell(op, &g);
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

/// The gathered input's [`ReadIndex::Indexed`] fields.
struct Gather {
    data: u32,
    index: u32,
    axis: u32,
    oob: OobPolicy,
    index_dtype: ElementKind,
}

fn check_gather(op: &OpDef, dtype: ElementKind, rank: u8) -> Result<Gather, LiftDecline> {
    if !matches!(op.access, Access::Elementwise) {
        return Err(LiftDecline::NotElementwise);
    }
    let Some((data, index, axis, oob, index_dtype)) = crate::plan::gather_of(&op.read_index) else {
        return Err(LiftDecline::NotGather);
    };
    if op.read_index.iter().filter(|r| !r.is_direct()).count() != 1 {
        return Err(LiftDecline::UnsupportedForm("more than one gathered input"));
    }
    if !matches!(
        index_dtype,
        ElementKind::I32 | ElementKind::I64 | ElementKind::U32
    ) {
        return Err(LiftDecline::UnsupportedForm("a non-integer gather index"));
    }
    check_plain_with(op, dtype, rank, true, false)?;
    leaves_ok(&op.body)?;
    if usize::from(axis) >= usize::from(rank)
        || usize::from(index) >= usize::from(op.n_inputs)
        || usize::from(index) == data
        || !matches!(op.read_index[usize::from(index)], ReadIndex::Direct)
    {
        return Err(LiftDecline::UnsupportedForm("an ill-formed gather"));
    }
    if !matches!(op.write_index, WriteIndex::Direct) {
        return Err(LiftDecline::UnsupportedForm("gather and scatter together"));
    }
    Ok(Gather {
        data: u32::try_from(data).expect("an input slot fits in u32"),
        index: u32::from(index),
        axis: u32::from(axis),
        oob,
        index_dtype,
    })
}

/// Local ids of a gather kernel.
struct Shape {
    dtype: ElementKind,
    index_dtype: ElementKind,
    r: u32,
    n: u32,
    index: u32,
}

impl Shape {
    fn new(n_inputs: u8, dtype: ElementKind, rank: u8, g: &Gather) -> Self {
        Self {
            dtype,
            index_dtype: g.index_dtype,
            r: u32::from(rank),
            n: u32::from(n_inputs),
            index: g.index,
        }
    }

    fn counter(&self, d: u32) -> u32 {
        self.r + d
    }

    fn loaded(&self, k: u32) -> u32 {
        2 * self.r + k
    }

    /// The source's extent along the gathered axis.
    fn extent(&self) -> u32 {
        2 * self.r + self.n
    }

    /// The index clamped into `[0, extent - 1]`.
    fn clamped(&self) -> u32 {
        self.extent() + 1
    }

    fn idx_const(&self, v: i64) -> ScalarExpr {
        ScalarExpr::ConstInt(v, self.index_dtype)
    }

    fn at_counters(&self, buffer: u32, axis: Option<(u32, u32)>) -> Place {
        let coords = (0..self.r)
            .map(|d| match axis {
                Some((a, via)) if a == d => local(via),
                _ => local(self.counter(d)),
            })
            .collect();
        Place::Logical { buffer, coords }
    }

    fn load(&self, k: u32, via: Option<(u32, u32)>) -> Stmt {
        Stmt::Load {
            local: self.loaded(k),
            place: self.at_counters(k, via),
        }
    }

    fn zero(&self) -> ScalarExpr {
        if crate::types::int_range(self.dtype).is_some() {
            ScalarExpr::ConstInt(0, self.dtype)
        } else {
            ScalarExpr::Const(0.0)
        }
    }

    fn store(&self, value: ScalarExpr) -> Stmt {
        Stmt::Store {
            place: self.at_counters(self.n, None),
            value,
        }
    }

    /// One output cell: the index, its clamp, the policy-guarded loads and store.
    fn cell(&self, op: &OpDef, g: &Gather) -> Block {
        let raw = local(self.loaded(self.index));
        let last = ScalarExpr::Sub(Box::new(local(self.extent())), Box::new(self.idx_const(1)));
        let clamp = ScalarExpr::Binary(
            BinaryOp::Min,
            Box::new(ScalarExpr::Binary(
                BinaryOp::Max,
                Box::new(raw.clone()),
                Box::new(self.idx_const(0)),
            )),
            Box::new(last),
        );
        let mut compute: Vec<Stmt> = (0..self.n)
            .filter(|&k| k != self.index)
            .map(|k| {
                let via = (k == g.data).then_some((g.axis, self.clamped()));
                self.load(k, via)
            })
            .collect();
        compute.push(self.store(substitute_inputs(&op.body, &|k| self.loaded(k))));
        let compute = Block(compute);
        let cmp = |op, rhs| ScalarExpr::Binary(op, Box::new(raw.clone()), Box::new(rhs));
        let guarded = |oob: Block| {
            let upper = Stmt::If {
                cond: cmp(BinaryOp::CmpGe, local(self.extent())),
                then: oob.clone(),
                els: compute.clone(),
            };
            Stmt::If {
                cond: cmp(BinaryOp::CmpLt, self.idx_const(0)),
                then: oob,
                els: Block(vec![upper]),
            }
        };
        let mut cell = vec![
            self.load(self.index, None),
            Stmt::Let {
                local: self.clamped(),
                init: clamp,
            },
        ];
        match g.oob {
            OobPolicy::Skip => cell.push(guarded(Block(vec![]))),
            OobPolicy::ZeroFill => cell.push(guarded(Block(vec![self.store(self.zero())]))),
            // Every index is clamped into range, so the policy leaves the cell unguarded.
            OobPolicy::Clamp => cell.extend(compute.0.iter().cloned()),
        }
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
        params.push(Param::buffer(
            "out",
            self.dtype,
            BufferRole::Write,
            Some(rank),
        ));
        params.extend((0..self.r).map(|d| Param::scalar(format!("n{d}"), ElementKind::I32, d)));
        params.push(Param::scalar("src_extent", self.index_dtype, self.extent()));
        let mut locals = vec![LocalDecl::scalar(ElementKind::I32); 2 * self.r as usize];
        locals.extend((0..self.n).map(|k| LocalDecl::scalar(elem(k))));
        locals.push(LocalDecl::scalar(self.index_dtype)); // src_extent
        locals.push(LocalDecl::scalar(self.index_dtype)); // clamped index
        (params, locals)
    }
}
