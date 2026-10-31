// SPDX-License-Identifier: MIT OR Apache-2.0
//! The launched `Access::RowReduce` lift (E1d-3): one cooperating block per row, each stage a
//! block-tree reduction whose result is broadcast to every thread, then a strided epilogue.
//!
//! Per row, in this order: for each stage `s`, every thread folds a strided slice of the row
//! (`col = tid; < n; += BlockDim`) into a private `F64` accumulator over the stage's `pre`
//! expression (which may read the earlier stages' broadcast results); the partials go through the
//! same log-step tree as `block_reduce.rs`; thread 0 finalises (`Mean` divides by the row length),
//! rounds **once** to the compute dtype and stores the result in `Shared`; a `Barrier`; every
//! thread loads it as `Reduced(s)`. After the last stage every thread runs the epilogue over the
//! columns `col = tid; < n; += BlockDim`, reading the broadcast stage results. Blocks stride over
//! the outermost leading axis (`row = BlockIdx; += GridDim`).
//!
//! **Numerics.** `Max`/`Min` stages are order-independent (the oracle's NaN-keeping set function),
//! so they equal the serial lift's bytes except for NaN payload and the sign of a zero when both
//! occur. A `Sum`/`Mean` stage is held to the E1d-2 declared bound, `gamma_(n-1) * sum|x_i|` with
//! `u = 2^-53` on its `F64` accumulation (the stage's `pre` values are the ones the serial lift
//! folds), then the same single rounding the serial lift does; its order is fixed by `block`
//! alone, so the result is reproducible and independent of `grid`. A later stage or the epilogue
//! sees the rounded stage value, exactly as in the serial lift, so the whole op inherits the serial
//! lift's tolerance mode (`row_reduce.rs`) plus the stage-value difference. At `block = 1` the
//! kernel's bytes equal the serial lift's. `docs/e1d-parallel-mapping.md` ("E1d-3 as built").
//!
//! Geometry is the caller's and never derived from an extent (E1d rules, `docs/PATENT-SWEEP.md`).

use unpopped_vocab::ElementKind;

use super::LiftDecline;
use super::block_reduce::{Fold, as_i32, barrier, shared, thread, uconst};
use super::im2col::int;
use super::row_reduce::{Row, check_row_reduce};
use crate::exec::{AllocDecl, ExecId, Extent, LaunchShape, MemorySpace};
use crate::ir::{BinaryOp, OpDef, ReduceStage, ScalarExpr};
use crate::kernel::{Block, KernelFn, LocalDecl, Stmt};

/// Lift a float `Access::RowReduce` over `rank` axes (`rank >= 2`: blocks stride over the leading
/// axes) as a block-tree row reduction on `grid` blocks of `block` threads (`block` a power of
/// two). Same parameters as [`super::lift_row_reduce_opdef`], plus one `I32` local for the
/// partner's `have` flag, one `Shared` array of partial accumulators (and `have` flags) and one of
/// the stage results.
///
/// # Errors
///
/// A [`LiftDecline`]: those of [`super::lift_row_reduce_opdef`]; [`LiftDecline::BadLaunch`] for a
/// zero `grid`, or a `block` that is zero or not a power of two; [`LiftDecline::UnsupportedForm`]
/// for a rank-1 row reduction (no leading axis to stride blocks over).
pub fn lift_row_reduce_opdef_launched(
    op: &OpDef,
    dtype: ElementKind,
    rank: u8,
    block: u32,
    grid: u32,
) -> Result<KernelFn, LiftDecline> {
    if grid == 0 || !block.is_power_of_two() {
        return Err(LiftDecline::BadLaunch);
    }
    let (stages, epilogue) = check_row_reduce(op, dtype, rank)?;
    if rank < 2 {
        return Err(LiftDecline::UnsupportedForm(
            "a launched row reduction needs a leading axis to stride blocks over",
        ));
    }
    let w = Row::new(op.n_inputs, dtype, rank, stages.len());
    let mut row: Vec<Stmt> = Vec::new();
    for (s, stage) in stages.iter().enumerate() {
        row.extend(launched_stage(&w, s, stage, block));
    }
    row.push(strided(w.epilogue(epilogue)));
    let mut body = Block(row);
    for d in (0..w.r - 1).rev() {
        body = w.for_lead(d, body);
    }
    block_stride_outermost(&mut body)?;
    let (params, mut locals) = w.signature();
    locals.push(LocalDecl::scalar(ElementKind::I32)); // the partner's `have`
    Ok(KernelFn::new(op.name.clone(), dtype, params, locals, body)
        .with_launch(LaunchShape::new([block, 1, 1], [grid, 1, 1]))
        .with_allocs(allocs(&w, block)))
}

/// One stage: the serial lift's fold made thread-strided, the tree, thread 0's single rounding
/// stored in `Shared` slot `s`, a `Barrier`, and every thread's load of it as `Reduced(s)`.
fn launched_stage(w: &Row, s: usize, stage: &ReduceStage, block: u32) -> Vec<Stmt> {
    let slot = u32::try_from(s).expect("a handful of stages");
    let base = w.stage_base(slot);
    let fold = Fold {
        rop: stage.op,
        block,
        acc: base,
        have: base + 1,
        val: base + 2,
        partner_have: w.stage_base(w.stages),
        int: false,
    };
    let mut v = w.stage(s, stage);
    // The serial stage ends with `red = round(finalised)`; here thread 0 computes it and shares it.
    let Some(Stmt::Let { init: rounded, .. }) = v.pop() else {
        unreachable!("a stage ends with the rounding of its result")
    };
    let fold_loop = v.pop().expect("a stage's fold loop");
    v.push(strided(fold_loop));
    v.extend(fold.publish());
    v.push(barrier());
    v.extend(fold.rounds());
    v.push(Stmt::If {
        cond: ScalarExpr::Binary(BinaryOp::CmpEq, Box::new(thread()), Box::new(uconst(0))),
        then: Block(vec![Stmt::Store {
            place: shared(2, int(i64::from(slot))),
            value: rounded,
        }]),
        els: Block(vec![]),
    });
    v.push(barrier());
    v.push(Stmt::Load {
        local: base + 3,
        place: shared(2, int(i64::from(slot))),
    });
    v
}

/// `for col = tid; col < n; col += BlockDim`: the serial loop with its start and step replaced.
fn strided(mut serial_loop: Stmt) -> Stmt {
    let Stmt::For { start, step, .. } = &mut serial_loop else {
        unreachable!("a stage's fold and the epilogue are loops over the row")
    };
    *start = as_i32(thread());
    *step = as_i32(ScalarExpr::Exec(ExecId::BlockDim(0)));
    serial_loop
}

/// Make the outermost leading loop block-strided: `row = BlockIdx; row < n; row += GridDim`.
fn block_stride_outermost(body: &mut Block) -> Result<(), LiftDecline> {
    let Some(Stmt::For { start, step, .. }) = body.0.first_mut() else {
        return Err(LiftDecline::UnsupportedForm(
            "a row reduction's outermost statement is a loop",
        ));
    };
    *start = as_i32(ScalarExpr::Exec(ExecId::BlockIdx(0)));
    *step = as_i32(ScalarExpr::Exec(ExecId::GridDim(0)));
    Ok(())
}

/// Allocation 0: the `F64` partial accumulators; 1: their `have` flags; 2: one slot per stage for
/// the broadcast result, in the compute dtype.
fn allocs(w: &Row, block: u32) -> Vec<AllocDecl> {
    let per_thread = Extent::Static(block);
    vec![
        AllocDecl::new("partial", MemorySpace::Shared, ElementKind::F64, per_thread),
        AllocDecl::new(
            "partial_have",
            MemorySpace::Shared,
            ElementKind::I32,
            per_thread,
        ),
        AllocDecl::new(
            "stage_result",
            MemorySpace::Shared,
            w.dtype,
            Extent::Static(w.stages),
        ),
    ]
}
