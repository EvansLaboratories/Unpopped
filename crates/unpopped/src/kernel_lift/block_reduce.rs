// SPDX-License-Identifier: MIT OR Apache-2.0
//! The launched `Access::Reduction` lift (E1d-1): one cooperating block reduces one output, and
//! blocks stride over the outermost kept axis.
//!
//! Each thread folds a strided slice of the reduced axis into a private accumulator, writes it to
//! a `Shared` array, and a log-step tree (a `Barrier` between rounds) combines the partials; thread
//! 0 stores. The accumulation **order** is not the serial order, so this lift accepts only
//! reductions whose result does not depend on order: integer `Sum`/`Prod`/`Max`/`Min` (wrapping
//! arithmetic is associative and commutative) and float `Max`/`Min` (a set function, with the
//! oracle's NaN-keeping rule). It is bit-exact against the serial kernel except for NaN payload
//! and the sign of a zero when `+0` and `-0` both occur in a `Max`/`Min` (the serial fold keeps
//! the first, the tree keeps whichever it combines first).
//!
//! A float `Sum`/`Mean` is **not** order-independent and is accepted with a declared bound
//! (PM ruling 2026-10-31, E1d-2), not bit-exactness. The order is fixed by `(grid, block)` and by
//! nothing else, so the result is reproducible run to run: thread `t` folds elements `t`, `t + B`,
//! `t + 2B`, ... in ascending order into an `f64` accumulator (`B` is `block`); the tree then adds
//! `acc[t] + acc[t + s]` for `s = B/2, B/4, ..., 1`, left operand first; thread 0 holds the total,
//! divides it by the reduced extent for `Mean`, and rounds once to the compute dtype. The error
//! against the exact sum is at most `gamma_(n-1) * sum|x_i|` with `u = 2^-53` (the standard
//! summation bound, which holds for every order of the same `n` additions), plus the half ulp of the
//! final rounding; against the serial kernel it is at most twice that. `docs/e1d-parallel-mapping.md`
//! states the bound and `tests/kernel_parallel_reduction_float_sum_bound.rs` checks it. `Prod` of a
//! float still declines.
//!
//! The geometry is the caller's and never derived from an extent; the parallel axes are declared
//! from the op family, not found by inspecting loops (`docs/PATENT-SWEEP.md`, E1d rules).

use super::{LiftDecline, ReductionShape, check_reduction, fold_step, for_axis, widen};
use crate::exec::{AllocDecl, ExecId, Extent, LaunchShape, MemorySpace, Scope};
use crate::ir::{BinaryOp, OpDef, ReduceOp, ScalarExpr};
use crate::kernel::{Block, KernelFn, LocalDecl, Place, Stmt};
use unpopped_vocab::ElementKind;

/// Lift a `Access::Reduction` as a block-tree reduction on `grid` blocks of `block` threads
/// (`block` a power of two). Same parameters and locals as [`super::lift_reduction_opdef`], plus
/// one `I32` local for the partner's `have` flag.
///
/// # Errors
///
/// A [`LiftDecline`]: those of [`super::lift_reduction_opdef`]; [`LiftDecline::BadLaunch`] for a
/// zero `grid`, or a `block` that is zero or not a power of two; and
/// [`LiftDecline::UnsupportedForm`] for a float `Prod` or an integer `Mean`, a fused
/// post, a reduction over more than one axis, or one with no kept axis.
pub fn lift_reduction_opdef_launched(
    op: &OpDef,
    dtype: ElementKind,
    rank: u8,
    block: u32,
    grid: u32,
) -> Result<KernelFn, LiftDecline> {
    if grid == 0 || !block.is_power_of_two() {
        return Err(LiftDecline::BadLaunch);
    }
    let (rop, axes, keepdim, post) = check_reduction(op, dtype, rank)?;
    let sh = ReductionShape::new(op.n_inputs, dtype, rank, axes);
    check_order_independent(&sh, rop, post.is_some())?;
    let t = Tree {
        sh: &sh,
        fold: Fold {
            rop,
            block,
            acc: sh.acc,
            have: sh.have,
            val: sh.val,
            partner_have: sh.red,
            int: sh.is_int(),
        },
    };

    let mut row = vec![
        Stmt::Let {
            local: sh.acc,
            init: sh.fold_identity(rop),
        },
        Stmt::Let {
            local: sh.have,
            init: ScalarExpr::ConstInt(0, ElementKind::I32),
        },
    ];
    row.extend(t.slice(op)?);
    row.extend(t.fold.publish());
    row.push(barrier());
    row.extend(t.fold.rounds());
    let out_coords = sh.out_coords(keepdim);
    let out_rank = u8::try_from(out_coords.len()).expect("rank is at most 8");
    row.push(t.thread0_stores(out_coords));

    let body = block_strided_rows(&sh, Block(row))?;
    let (params, mut locals) = sh.signature(out_rank);
    locals.push(LocalDecl::scalar(ElementKind::I32)); // the partner's `have`
    Ok(KernelFn::new(op.name.clone(), dtype, params, locals, body)
        .with_launch(LaunchShape::new([block, 1, 1], [grid, 1, 1]))
        .with_allocs(t.fold.allocs(sh.acc_ty())))
}

/// The forms this lift takes: an order-independent fold (or a float `Sum`/`Mean` with a declared
/// bound), no post, one reduced axis, one or more kept axes.
fn check_order_independent(
    sh: &ReductionShape,
    rop: ReduceOp,
    has_post: bool,
) -> Result<(), LiftDecline> {
    let independent = match rop {
        ReduceOp::Max | ReduceOp::Min => true,
        // A float Sum or Mean has a declared bound instead (module docs); an integer Mean is not a fold.
        ReduceOp::Sum => true,
        ReduceOp::Prod => sh.is_int(),
        ReduceOp::Mean => !sh.is_int(),
    };
    if !independent {
        return Err(LiftDecline::UnsupportedForm(
            "an integer Mean or a float Prod in a launched reduction (not order-independent, and no declared bound)",
        ));
    }
    if has_post {
        return Err(LiftDecline::UnsupportedForm(
            "a fused post in a launched reduction",
        ));
    }
    if sh.reduced.len() != 1 || sh.kept.is_empty() {
        return Err(LiftDecline::UnsupportedForm(
            "a launched reduction needs one reduced axis and at least one kept axis",
        ));
    }
    Ok(())
}

pub(super) fn barrier() -> Stmt {
    Stmt::Barrier {
        scope: Scope::Block,
    }
}

pub(super) fn thread() -> ScalarExpr {
    ScalarExpr::Exec(ExecId::ThreadIdx(0))
}

pub(super) fn uconst(v: u32) -> ScalarExpr {
    ScalarExpr::ConstInt(i64::from(v), ElementKind::U32)
}

/// An unsigned id as the `I32` the counters and indices are.
pub(super) fn as_i32(e: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Cast(Box::new(e), widen(ElementKind::I32))
}

pub(super) fn shared(alloc: u32, index: ScalarExpr) -> Place {
    Place::Alloc { alloc, index }
}

/// The cooperative fold shared by every block-tree lift: the locals a thread folds into, and the
/// geometry of the log-step tree that combines the threads' partials. Allocation 0 holds the
/// partial accumulators and, for `Max`/`Min`, allocation 1 the partials' `have` flags.
pub(super) struct Fold {
    pub(super) rop: ReduceOp,
    pub(super) block: u32,
    /// The thread's accumulator, `have` flag and loaded partner value.
    pub(super) acc: u32,
    pub(super) have: u32,
    pub(super) val: u32,
    /// Where a round loads the partner's `have` flag (`Max`/`Min` only).
    pub(super) partner_have: u32,
    pub(super) int: bool,
}

impl Fold {
    pub(super) fn has_flag(&self) -> bool {
        matches!(self.rop, ReduceOp::Max | ReduceOp::Min)
    }

    /// Write this thread's accumulator (and `have` flag) to its own slot.
    pub(super) fn publish(&self) -> Vec<Stmt> {
        let me = as_i32(thread());
        let mut v = vec![Stmt::Store {
            place: shared(0, me.clone()),
            value: ScalarExpr::Local(self.acc),
        }];
        if self.has_flag() {
            v.push(Stmt::Store {
                place: shared(1, me),
                value: ScalarExpr::Local(self.have),
            });
        }
        v
    }

    /// `block/2, block/4, ... 1`: threads below `s` fold in the partial `s` slots up, then every
    /// thread meets at a `Barrier`.
    pub(super) fn rounds(&self) -> Vec<Stmt> {
        let mut out = Vec::new();
        let mut s = self.block / 2;
        while s > 0 {
            out.push(Stmt::If {
                cond: ScalarExpr::Binary(BinaryOp::CmpLt, Box::new(thread()), Box::new(uconst(s))),
                then: Block(self.round_body(s)),
                els: Block(vec![]),
            });
            out.push(barrier());
            s /= 2;
        }
        out
    }

    /// One thread's work in the round with stride `s`: load the partner's partial, fold it in,
    /// publish the result.
    fn round_body(&self, s: u32) -> Vec<Stmt> {
        let partner = as_i32(ScalarExpr::Add(Box::new(thread()), Box::new(uconst(s))));
        let mut body = vec![Stmt::Load {
            local: self.val,
            place: shared(0, partner.clone()),
        }];
        let combine = fold_step(self.rop, self.acc, self.have, self.val, self.int);
        if self.has_flag() {
            body.push(Stmt::Load {
                local: self.partner_have,
                place: shared(1, partner),
            });
            // A partner that folded nothing carries no value to combine.
            body.push(Stmt::If {
                cond: ScalarExpr::Local(self.partner_have),
                then: Block(combine),
                els: Block(vec![]),
            });
        } else {
            body.extend(combine);
        }
        body.extend(self.publish());
        body
    }

    /// The shared arrays the tree needs, `acc_ty` being the accumulator's element type.
    pub(super) fn allocs(&self, acc_ty: ElementKind) -> Vec<AllocDecl> {
        let extent = Extent::Static(self.block);
        let mut allocs = vec![AllocDecl::new(
            "partial",
            MemorySpace::Shared,
            acc_ty,
            extent,
        )];
        if self.has_flag() {
            allocs.push(AllocDecl::new(
                "partial_have",
                MemorySpace::Shared,
                ElementKind::I32,
                extent,
            ));
        }
        allocs
    }
}

/// One block's tree over a [`ReductionShape`].
struct Tree<'a> {
    sh: &'a ReductionShape,
    fold: Fold,
}

impl Tree<'_> {
    /// This thread's strided slice of the reduced axis: `reduced = tid; < n; += BlockDim`.
    fn slice(&self, op: &OpDef) -> Result<Vec<Stmt>, LiftDecline> {
        let mut slice = self.sh.reduce_loop(op, self.fold.rop).0;
        let Some(Stmt::For { start, step, .. }) = slice.first_mut() else {
            return Err(LiftDecline::UnsupportedForm(
                "a reduction's fold is a loop over the reduced axis",
            ));
        };
        *start = as_i32(thread());
        *step = as_i32(ScalarExpr::Exec(ExecId::BlockDim(0)));
        Ok(slice)
    }

    /// Thread 0 holds the row's result in its accumulator and stores it.
    fn thread0_stores(&self, coords: Vec<ScalarExpr>) -> Stmt {
        Stmt::If {
            cond: ScalarExpr::Binary(BinaryOp::CmpEq, Box::new(thread()), Box::new(uconst(0))),
            then: Block(vec![Stmt::Store {
                place: Place::Logical {
                    buffer: self.sh.n,
                    coords,
                },
                value: self.sh.stored(self.fold.rop),
            }]),
            els: Block(vec![]),
        }
    }
}

/// Wrap `row` in the kept axes' loops and make the outermost one block-strided:
/// `row = BlockIdx; row < n; row += GridDim`.
fn block_strided_rows(sh: &ReductionShape, row: Block) -> Result<Block, LiftDecline> {
    let mut body = row;
    for &d in sh.kept.iter().rev() {
        body = for_axis(d, sh.r, body);
    }
    let Some(Stmt::For { start, step, .. }) = body.0.first_mut() else {
        return Err(LiftDecline::UnsupportedForm(
            "a reduction's outermost statement is a loop",
        ));
    };
    *start = as_i32(ScalarExpr::Exec(ExecId::BlockIdx(0)));
    *step = as_i32(ScalarExpr::Exec(ExecId::GridDim(0)));
    Ok(body)
}
