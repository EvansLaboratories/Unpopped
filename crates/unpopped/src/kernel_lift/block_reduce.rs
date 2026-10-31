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
//! the first, the tree keeps whichever it combines first). A float `Sum`/`Mean` declines: it needs
//! a documented tolerance, which is its own decision (`docs/e1d-parallel-mapping.md`).
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
/// [`LiftDecline::UnsupportedForm`] for a float `Sum`/`Prod`/`Mean` (order-dependent), a fused
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
        rop,
        block,
        has_flag: matches!(rop, ReduceOp::Max | ReduceOp::Min),
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
    row.extend(t.publish());
    row.push(barrier());
    row.extend(t.rounds());
    let out_coords = sh.out_coords(keepdim);
    let out_rank = u8::try_from(out_coords.len()).expect("rank is at most 8");
    row.push(t.thread0_stores(out_coords));

    let body = block_strided_rows(&sh, Block(row))?;
    let (params, mut locals) = sh.signature(out_rank);
    locals.push(LocalDecl::scalar(ElementKind::I32)); // the partner's `have`
    Ok(KernelFn::new(op.name.clone(), dtype, params, locals, body)
        .with_launch(LaunchShape::new([block, 1, 1], [grid, 1, 1]))
        .with_allocs(t.allocs()))
}

/// The forms this lift takes: an order-independent fold, no post, one reduced axis, one or more
/// kept axes.
fn check_order_independent(
    sh: &ReductionShape,
    rop: ReduceOp,
    has_post: bool,
) -> Result<(), LiftDecline> {
    let independent = match rop {
        ReduceOp::Max | ReduceOp::Min => true,
        ReduceOp::Sum | ReduceOp::Prod => sh.is_int(),
        _ => false,
    };
    if !independent {
        return Err(LiftDecline::UnsupportedForm(
            "a float Sum, Prod or Mean in a launched reduction (its order is observable)",
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

fn barrier() -> Stmt {
    Stmt::Barrier {
        scope: Scope::Block,
    }
}

fn thread() -> ScalarExpr {
    ScalarExpr::Exec(ExecId::ThreadIdx(0))
}

fn uconst(v: u32) -> ScalarExpr {
    ScalarExpr::ConstInt(i64::from(v), ElementKind::U32)
}

/// An unsigned id as the `I32` the counters and indices are.
fn as_i32(e: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Cast(Box::new(e), widen(ElementKind::I32))
}

fn shared(alloc: u32, index: ScalarExpr) -> Place {
    Place::Alloc { alloc, index }
}

/// The pieces of one block's tree: allocation 0 holds the partial accumulators and, for
/// `Max`/`Min`, allocation 1 the partials' `have` flags.
struct Tree<'a> {
    sh: &'a ReductionShape,
    rop: ReduceOp,
    block: u32,
    has_flag: bool,
}

impl Tree<'_> {
    /// This thread's strided slice of the reduced axis: `reduced = tid; < n; += BlockDim`.
    fn slice(&self, op: &OpDef) -> Result<Vec<Stmt>, LiftDecline> {
        let mut slice = self.sh.reduce_loop(op, self.rop).0;
        let Some(Stmt::For { start, step, .. }) = slice.first_mut() else {
            return Err(LiftDecline::UnsupportedForm(
                "a reduction's fold is a loop over the reduced axis",
            ));
        };
        *start = as_i32(thread());
        *step = as_i32(ScalarExpr::Exec(ExecId::BlockDim(0)));
        Ok(slice)
    }

    /// Write this thread's accumulator (and `have` flag) to its own slot.
    fn publish(&self) -> Vec<Stmt> {
        let me = as_i32(thread());
        let mut v = vec![Stmt::Store {
            place: shared(0, me.clone()),
            value: ScalarExpr::Local(self.sh.acc),
        }];
        if self.has_flag {
            v.push(Stmt::Store {
                place: shared(1, me),
                value: ScalarExpr::Local(self.sh.have),
            });
        }
        v
    }

    /// `block/2, block/4, ... 1`: threads below `s` fold in the partial `s` slots up, then every
    /// thread meets at a `Barrier`.
    fn rounds(&self) -> Vec<Stmt> {
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
        let sh = self.sh;
        let partner_have = sh.red;
        let partner = as_i32(ScalarExpr::Add(Box::new(thread()), Box::new(uconst(s))));
        let mut body = vec![Stmt::Load {
            local: sh.val,
            place: shared(0, partner.clone()),
        }];
        let combine = fold_step(self.rop, sh.acc, sh.have, sh.val, sh.is_int());
        if self.has_flag {
            body.push(Stmt::Load {
                local: partner_have,
                place: shared(1, partner),
            });
            // A partner that folded nothing carries no value to combine.
            body.push(Stmt::If {
                cond: ScalarExpr::Local(partner_have),
                then: Block(combine),
                els: Block(vec![]),
            });
        } else {
            body.extend(combine);
        }
        body.extend(self.publish());
        body
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
                value: self.sh.stored(self.rop),
            }]),
            els: Block(vec![]),
        }
    }

    fn allocs(&self) -> Vec<AllocDecl> {
        let extent = Extent::Static(self.block);
        let mut allocs = vec![AllocDecl::new(
            "partial",
            MemorySpace::Shared,
            self.sh.acc_ty(),
            extent,
        )];
        if self.has_flag {
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
