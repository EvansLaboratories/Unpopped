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

    let tid = || ScalarExpr::Exec(ExecId::ThreadIdx(0));
    let at = |e: ScalarExpr| ScalarExpr::Cast(Box::new(e), widen(ElementKind::I32));
    let uconst = |v: u32| ScalarExpr::ConstInt(i64::from(v), ElementKind::U32);
    let has_flag = matches!(rop, ReduceOp::Max | ReduceOp::Min);
    let partner_have = sh.red;

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
    // This thread's strided slice of the reduced axis.
    let mut slice = sh.reduce_loop(op, rop).0;
    let Some(Stmt::For { start, step, .. }) = slice.first_mut() else {
        return Err(LiftDecline::UnsupportedForm(
            "a reduction's fold is a loop over the reduced axis",
        ));
    };
    *start = at(tid());
    *step = at(ScalarExpr::Exec(ExecId::BlockDim(0)));
    row.extend(slice);

    let shared = |alloc: u32, index: ScalarExpr| Place::Alloc { alloc, index };
    let publish = |me: ScalarExpr| {
        let mut v = vec![Stmt::Store {
            place: shared(0, me.clone()),
            value: ScalarExpr::Local(sh.acc),
        }];
        if has_flag {
            v.push(Stmt::Store {
                place: shared(1, me),
                value: ScalarExpr::Local(sh.have),
            });
        }
        v
    };
    row.extend(publish(at(tid())));
    row.push(Stmt::Barrier {
        scope: Scope::Block,
    });
    let mut s = block / 2;
    while s > 0 {
        let partner = at(ScalarExpr::Add(Box::new(tid()), Box::new(uconst(s))));
        let mut then = vec![Stmt::Load {
            local: sh.val,
            place: shared(0, partner.clone()),
        }];
        let combine = fold_step(rop, sh.acc, sh.have, sh.val, sh.is_int());
        if has_flag {
            then.push(Stmt::Load {
                local: partner_have,
                place: shared(1, partner),
            });
            // A partner that folded nothing carries no value to combine.
            then.push(Stmt::If {
                cond: ScalarExpr::Local(partner_have),
                then: Block(combine),
                els: Block(vec![]),
            });
        } else {
            then.extend(combine);
        }
        then.extend(publish(at(tid())));
        row.push(Stmt::If {
            cond: ScalarExpr::Binary(BinaryOp::CmpLt, Box::new(tid()), Box::new(uconst(s))),
            then: Block(then),
            els: Block(vec![]),
        });
        row.push(Stmt::Barrier {
            scope: Scope::Block,
        });
        s /= 2;
    }
    let out_coords = sh.out_coords(keepdim);
    let out_rank = u8::try_from(out_coords.len()).expect("rank is at most 8");
    row.push(Stmt::If {
        cond: ScalarExpr::Binary(BinaryOp::CmpEq, Box::new(tid()), Box::new(uconst(0))),
        then: Block(vec![Stmt::Store {
            place: Place::Logical {
                buffer: sh.n,
                coords: out_coords,
            },
            value: sh.stored(rop),
        }]),
        els: Block(vec![]),
    });

    let mut body = Block(row);
    for &d in sh.kept.iter().rev() {
        body = for_axis(d, sh.r, body);
    }
    // Blocks stride over the outermost kept axis: `row = BlockIdx; row < n; row += GridDim`.
    let Some(Stmt::For { start, step, .. }) = body.0.first_mut() else {
        return Err(LiftDecline::UnsupportedForm(
            "a reduction's outermost statement is a loop",
        ));
    };
    *start = at(ScalarExpr::Exec(ExecId::BlockIdx(0)));
    *step = at(ScalarExpr::Exec(ExecId::GridDim(0)));

    let (params, mut locals) = sh.signature(out_rank);
    locals.push(LocalDecl::scalar(ElementKind::I32)); // partner_have
    let mut allocs = vec![AllocDecl::new(
        "partial",
        MemorySpace::Shared,
        sh.acc_ty(),
        Extent::Static(block),
    )];
    if has_flag {
        allocs.push(AllocDecl::new(
            "partial_have",
            MemorySpace::Shared,
            ElementKind::I32,
            Extent::Static(block),
        ));
    }
    Ok(KernelFn::new(op.name.clone(), dtype, params, locals, body)
        .with_launch(LaunchShape::new([block, 1, 1], [grid, 1, 1]))
        .with_allocs(allocs))
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
