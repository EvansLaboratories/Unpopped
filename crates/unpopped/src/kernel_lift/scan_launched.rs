// SPDX-License-Identifier: MIT OR Apache-2.0
//! The launched `Access::Scan` lift (E1d-5): one cooperating block per row, a chunked three-phase
//! scan along the innermost axis, blocks striding over the leading axes.
//!
//! Let `B` be `block` and `n` the row length. Walk the row in scan order (ascending, or descending
//! for a reverse scan) and give thread `t` the contiguous chunk of scan positions
//! `[t*c, min(t*c + c, n))` with `c = ceil(n / B)`.
//!
//! 1. **Totals.** Thread `t` folds its chunk, in scan order, into a private accumulator that starts
//!    at the monoid identity, and publishes it (with its `have` flag) to `Shared`.
//! 2. **Offsets.** A Kogge-Stone inclusive scan over the `B` totals, `Barrier`s around each round:
//!    for `d = 1, 2, 4, ... < B`, every thread `t >= d` replaces its total by
//!    `total[t] (op) total[t - d]`, own value on the left, reading the round's old values. Thread `t`'s
//!    offset is then the result of thread `t - 1` (the identity, `have = 0`, for thread 0).
//! 3. **Writes.** Thread `t` walks its chunk again, starting from its offset, and writes the prefix
//!    exactly as the serial scan does (before the fold when exclusive, after it when inclusive, a
//!    fused post reading the prefix rounded once to the compute dtype).
//!
//! **Numerics.** `Max`/`Min` and integer `Sum`/`Prod` are order-independent, so the launched kernel
//! is bit-exact against the serial scan (except NaN payload and the sign of a zero in `Max`/`Min`,
//! as in E1d-1). A float `Sum` is held to the E1d-2 bound for every output: each prefix is the sum of
//! at most `n` elements added in the order above, in `f64`, so its error against the exact prefix is
//! at most `gamma_(n-1) * sum|x_i|` over that prefix, plus the half ulp of the single final rounding;
//! the order is fixed by `block` alone, so the result is reproducible and independent of `grid`. At
//! `block = 1` the kernel's bytes equal the serial scan's. A float `Prod` declines, as in E1d-1, and
//! `Mean` is not a monoid. `docs/e1d-parallel-mapping.md` ("E1d-5 as built") states the order.
//!
//! The geometry is the caller's and never derived from an extent; only the in-kernel chunk size
//! `ceil(n / B)` depends on the row length (E1d rules, `docs/PATENT-SWEEP.md`).

use unpopped_vocab::ElementKind;

use super::block_reduce::{Fold, as_i32, barrier, thread, uconst};
use super::im2col::{add, int, local, mul, sub};
use super::{
    LiftDecline, ReductionShape, check_scan, fold_step, for_axis, scan_identity, scan_loads,
    scan_pass,
};
use crate::exec::{ExecId, LaunchShape};
use crate::ir::{BinaryOp, OpDef, ReduceOp, ScalarExpr};
use crate::kernel::{Block, KernelFn, LocalDecl, Place, Stmt};
use unpopped_vocab::AxisMask;

/// Lift a float or integer `Access::Scan` over `rank` axes (the scanned axis is the innermost) as a
/// block-level chunked scan on `grid` blocks of `block` threads (`block` a power of two). Same
/// parameters as [`super::lift_scan_opdef`]; the locals gain the `red`, partner-`have`, `chunk`,
/// `lo` and `hi` slots, and the kernel declares the `partial` (and `partial_have`) `Shared` arrays.
///
/// # Errors
///
/// A [`LiftDecline`]: those of [`super::lift_scan_opdef`]; [`LiftDecline::BadLaunch`] for a zero
/// `grid` or a `block` that is zero or not a power of two; [`LiftDecline::UnsupportedForm`] for a
/// float `Prod` (not order-independent, and no declared bound).
pub fn lift_scan_opdef_launched(
    op: &OpDef,
    dtype: ElementKind,
    rank: u8,
    block: u32,
    grid: u32,
) -> Result<KernelFn, LiftDecline> {
    if grid == 0 || !block.is_power_of_two() {
        return Err(LiftDecline::BadLaunch);
    }
    let (sop, reverse, exclusive, pre, post) = check_scan(op, dtype, rank)?;
    let sh = ReductionShape::new(op.n_inputs, dtype, rank, AxisMask::EMPTY);
    if matches!(sop, ReduceOp::Prod) && !sh.is_int() {
        return Err(LiftDecline::UnsupportedForm(
            "a float Prod scan in a launched scan (not order-independent, and no declared bound)",
        ));
    }
    let ids = Ids::new(&sh);
    let fold = Fold {
        rop: sop,
        block,
        acc: sh.acc,
        have: sh.have,
        val: sh.val,
        partner_have: ids.flag_tmp,
        int: sh.is_int(),
    };
    let scan = Scan {
        sh: &sh,
        fold: &fold,
        ids: &ids,
        reverse,
    };
    let mut body = scan.over_rows(Block(scan.row(exclusive, pre, post)))?;
    body.0.splice(0..0, scan.bounds());
    let (params, mut locals) = sh.signature(u8::try_from(sh.r).expect("rank is at most 8"));
    locals.push(LocalDecl::scalar(dtype)); // red: the prefix rounded once (a fused post reads it)
    locals.extend([LocalDecl::scalar(ElementKind::I32); 3]); // chunk, lo, hi
    locals.push(LocalDecl::scalar(sh.acc_ty())); // the partner's total
    locals.push(LocalDecl::scalar(ElementKind::I32)); // the load target for a `have` flag
    Ok(KernelFn::new(op.name.clone(), dtype, params, locals, body)
        .with_launch(LaunchShape::new([block, 1, 1], [grid, 1, 1]))
        .with_allocs(fold.allocs(sh.acc_ty())))
}

/// The extra local ids, after the shape's own `red`.
struct Ids {
    chunk: u32,
    lo: u32,
    hi: u32,
    /// The partner's total, carried across the round's `Barrier` (declared once per row).
    partner: u32,
    /// The target of a `Load` of a `have` flag: a `Load` declares its local, so the value is
    /// assigned on to the long-lived local.
    flag_tmp: u32,
}

impl Ids {
    fn new(sh: &ReductionShape) -> Self {
        let base = sh.red + 1;
        Self {
            chunk: base,
            lo: base + 1,
            hi: base + 2,
            partner: base + 3,
            flag_tmp: base + 4,
        }
    }
}

struct Scan<'a> {
    sh: &'a ReductionShape,
    fold: &'a Fold,
    ids: &'a Ids,
    reverse: bool,
}

impl Scan<'_> {
    /// One row: the three phases, ending in a `Barrier` so the next row's publish cannot overwrite
    /// a total another thread is still reading.
    fn row(&self, exclusive: bool, pre: &ScalarExpr, post: Option<&ScalarExpr>) -> Vec<Stmt> {
        let (sh, rop) = (self.sh, self.fold.rop);
        let identity = || sh.up(scan_identity(rop, sh.dtype));
        let mut row = vec![
            Stmt::Let {
                local: sh.acc,
                init: identity(),
            },
            Stmt::Let {
                local: sh.have,
                init: int(0),
            },
            Stmt::Let {
                local: self.ids.partner,
                init: identity(),
            },
        ];
        row.push(self.walk(self.fold_only(pre)));
        row.extend(self.fold.publish());
        row.push(barrier());
        row.extend(self.totals_scan());
        row.push(self.take_offset());
        row.push(self.walk(scan_pass(sh, rop, exclusive, pre, post)));
        row.push(barrier());
        row
    }

    fn tid(&self) -> ScalarExpr {
        as_i32(thread())
    }

    /// `chunk = ceil(n / block)`, `lo = tid * chunk`, `hi = min(lo + chunk, n)`. The kernel
    /// integer subset has no division or shift, so the ceiling is found by counting up
    /// (`n / block` iterations of an `I32` loop, small beside the walk).
    fn bounds(&self) -> Vec<Stmt> {
        let n = local(self.sh.reduced[0]);
        let ids = self.ids;
        let past = ScalarExpr::Binary(
            BinaryOp::CmpLt,
            Box::new(mul(local(ids.chunk), int(i64::from(self.fold.block)))),
            Box::new(n.clone()),
        );
        let hi = ScalarExpr::Binary(
            BinaryOp::Min,
            Box::new(add(local(ids.lo), local(ids.chunk))),
            Box::new(n),
        );
        vec![
            Stmt::Let {
                local: ids.chunk,
                init: int(0),
            },
            Stmt::While {
                cond: past,
                body: Block(vec![Stmt::Assign {
                    local: ids.chunk,
                    value: add(local(ids.chunk), int(1)),
                }]),
            },
            Stmt::Let {
                local: ids.lo,
                init: mul(self.tid(), local(ids.chunk)),
            },
            Stmt::Let {
                local: ids.hi,
                init: hi,
            },
        ]
    }

    /// This thread's chunk in scan order, with `body` run at each position.
    fn walk(&self, body: Vec<Stmt>) -> Stmt {
        let n = local(self.sh.reduced[0]);
        let (lo, hi) = (local(self.ids.lo), local(self.ids.hi));
        let (start, end, step) = if self.reverse {
            (
                sub(sub(n.clone(), int(1)), lo),
                sub(sub(n, int(1)), hi),
                int(-1),
            )
        } else {
            (lo, hi, int(1))
        };
        Stmt::For {
            local: self.sh.counter(self.sh.reduced[0]),
            start,
            end,
            step,
            body: Block(body),
        }
    }

    /// One position of phase 1: load, pre-map, fold into the chunk total.
    fn fold_only(&self, pre: &ScalarExpr) -> Vec<Stmt> {
        let mut pass = scan_loads(self.sh, pre);
        pass.extend(fold_step(
            self.fold.rop,
            self.sh.acc,
            self.sh.have,
            self.sh.val,
            self.sh.is_int(),
        ));
        pass
    }

    /// Phase 2: the Kogge-Stone rounds over the published totals.
    fn totals_scan(&self) -> Vec<Stmt> {
        let mut out = Vec::new();
        let mut d = 1;
        while d < self.fold.block {
            let active =
                ScalarExpr::Binary(BinaryOp::CmpGe, Box::new(thread()), Box::new(uconst(d)));
            out.push(Stmt::If {
                cond: active.clone(),
                then: Block(self.read_partner(d)),
                els: Block(vec![]),
            });
            out.push(barrier());
            out.push(Stmt::If {
                cond: active,
                then: Block(self.combine_and_publish()),
                els: Block(vec![]),
            });
            out.push(barrier());
            d *= 2;
        }
        out
    }

    fn read_partner(&self, d: u32) -> Vec<Stmt> {
        let from = as_i32(ScalarExpr::Sub(Box::new(thread()), Box::new(uconst(d))));
        // The values are carried to the combine (after a `Barrier`) in row-level locals.
        let v = vec![
            Stmt::Load {
                local: self.sh.val,
                place: Place::Alloc {
                    alloc: 0,
                    index: from.clone(),
                },
            },
            Stmt::Assign {
                local: self.ids.partner,
                value: local(self.sh.val),
            },
        ];
        v
    }

    /// `total[t] = total[t] (op) total[t - d]`, own value on the left; a `Max`/`Min` partner that
    /// folded nothing is skipped.
    fn combine_and_publish(&self) -> Vec<Stmt> {
        let sh = self.sh;
        let combine = fold_step(
            self.fold.rop,
            sh.acc,
            sh.have,
            self.ids.partner,
            sh.is_int(),
        );
        // A partner thread that folded nothing holds the identity, which a fold leaves unchanged;
        // and a thread with elements never has an empty lower partner (chunks are contiguous from
        // the front), so no `have` guard is needed here.
        let mut v = combine;
        v.extend(self.fold.publish());
        v
    }

    /// Thread `t >= 1` starts phase 3 from thread `t - 1`'s inclusive total; thread 0 from the
    /// identity.
    fn take_offset(&self) -> Stmt {
        let sh = self.sh;
        let prev = as_i32(ScalarExpr::Sub(Box::new(thread()), Box::new(uconst(1))));
        // A `Load` declares its local, so the offset goes through the loaded-partner temporaries.
        let mut load = vec![
            Stmt::Load {
                local: sh.val,
                place: Place::Alloc {
                    alloc: 0,
                    index: prev.clone(),
                },
            },
            Stmt::Assign {
                local: sh.acc,
                value: local(sh.val),
            },
        ];
        if self.fold.has_flag() {
            load.push(Stmt::Load {
                local: self.ids.flag_tmp,
                place: Place::Alloc {
                    alloc: 1,
                    index: prev,
                },
            });
            load.push(Stmt::Assign {
                local: sh.have,
                value: local(self.ids.flag_tmp),
            });
        }
        Stmt::If {
            cond: ScalarExpr::Binary(BinaryOp::CmpEq, Box::new(thread()), Box::new(uconst(0))),
            then: Block(vec![
                Stmt::Assign {
                    local: sh.acc,
                    value: sh.up(scan_identity(self.fold.rop, sh.dtype)),
                },
                Stmt::Assign {
                    local: sh.have,
                    value: int(0),
                },
            ]),
            els: Block(load),
        }
    }

    /// Wrap `row` in the leading axes' loops, the outermost block-strided; with no leading axis
    /// block 0 does the one row.
    fn over_rows(&self, row: Block) -> Result<Block, LiftDecline> {
        let sh = self.sh;
        if sh.kept.is_empty() {
            let block_zero = ScalarExpr::Binary(
                BinaryOp::CmpEq,
                Box::new(ScalarExpr::Exec(ExecId::BlockIdx(0))),
                Box::new(uconst(0)),
            );
            return Ok(Block(vec![Stmt::If {
                cond: block_zero,
                then: row,
                els: Block(vec![]),
            }]));
        }
        let mut body = row;
        for &d in sh.kept.iter().rev() {
            body = for_axis(d, sh.r, body);
        }
        let Some(Stmt::For { start, step, .. }) = body.0.first_mut() else {
            return Err(LiftDecline::UnsupportedForm(
                "a scan's outermost statement is a loop",
            ));
        };
        *start = as_i32(ScalarExpr::Exec(ExecId::BlockIdx(0)));
        *step = as_i32(ScalarExpr::Exec(ExecId::GridDim(0)));
        Ok(body)
    }
}
