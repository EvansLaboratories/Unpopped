// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The barrier-uniformity rule** (decision D-R4-5, `docs/u3-row4-scopes-and-memory-spaces.md`
//! section 4): a `Barrier` is defined only if every thread of its scope reaches it, so the checker
//! rejects one that sits under control that may differ between threads.
//!
//! A conservative taint analysis. A value is *divergent* if it reads an `Exec` leaf that differs
//! between the threads of a barrier's scope (everything but `BlockIdx`, `BlockDim` and `GridDim`,
//! which every thread of a block, and so of a subgroup, reads alike) or a `Load`
//! from global, shared or private memory, or flows from one. The analysis is flow-insensitive per
//! local (a local assigned a divergent value anywhere is divergent everywhere) and iterated to a
//! fixed point, so a value carried round a loop is still seen. A `Barrier` is rejected under an
//! `If` or `While` with a divergent condition, a `For` with a divergent bound, or after a `Break`
//! or `Continue` that sat under such control. It rejects some correct programs and never accepts
//! a divergent barrier; the four known over-rejections are pinned in `tests/kernel_barrier.rs`.

use crate::exec::{ExecId, MemorySpace};
use crate::ir::ScalarExpr;
use crate::kernel::{Block, KernelError, KernelFn, Place, Stmt};
use crate::types::operands;

/// Reject a `Barrier` that may not be reached by every thread of its scope.
pub(crate) fn check_barriers(k: &KernelFn) -> Result<(), KernelError> {
    let mut w = Walk {
        k,
        divergent: vec![false; k.locals.len()],
        changed: false,
        check: false,
    };
    loop {
        w.changed = false;
        w.block(&k.body, false, &mut false)?;
        if !w.changed {
            break;
        }
    }
    w.check = true;
    w.block(&k.body, false, &mut false)
}

struct Walk<'a> {
    k: &'a KernelFn,
    divergent: Vec<bool>,
    changed: bool,
    check: bool,
}

impl Walk<'_> {
    fn mark(&mut self, local: u32) {
        let d = &mut self.divergent[local as usize];
        if !*d {
            *d = true;
            self.changed = true;
        }
    }

    fn expr(&self, e: &ScalarExpr) -> bool {
        match e {
            ScalarExpr::Exec(id) => !matches!(
                id,
                ExecId::BlockIdx(_) | ExecId::BlockDim(_) | ExecId::GridDim(_)
            ),
            ScalarExpr::Local(i) => self.divergent[*i as usize],
            _ => operands(e).into_iter().any(|x| self.expr(x)),
        }
    }

    /// Whether a value loaded from `place` may differ between threads.
    fn load_divergent(&self, place: &Place) -> bool {
        match place {
            Place::Logical { .. } => true,
            Place::Alloc { alloc, index } => {
                let constant = self
                    .k
                    .allocs
                    .get(*alloc as usize)
                    .is_some_and(|a| a.space == MemorySpace::Constant);
                !constant || self.expr(index)
            }
        }
    }

    /// Walk `b`. `div`: control reaching `b` may differ between threads. `poisoned`: a `Break` or
    /// `Continue` under such control has been seen, so what follows may be skipped by some threads.
    fn block(&mut self, b: &Block, div: bool, poisoned: &mut bool) -> Result<(), KernelError> {
        for st in &b.0 {
            self.stmt(st, div, poisoned)?;
        }
        Ok(())
    }

    /// A loop body. A `Break` or `Continue` under divergent control late in the body also skips
    /// the start of the next pass, so a body that poisons is walked again as poisoned.
    fn loop_body(&mut self, b: &Block, div: bool, poisoned: &mut bool) -> Result<(), KernelError> {
        let before = *poisoned;
        self.block(b, div, poisoned)?;
        if !before && *poisoned {
            self.block(b, div, poisoned)?;
        }
        Ok(())
    }

    fn stmt(&mut self, st: &Stmt, div: bool, poisoned: &mut bool) -> Result<(), KernelError> {
        match st {
            Stmt::Let { local, init } => {
                if self.expr(init) {
                    self.mark(*local);
                }
            }
            Stmt::Load { local, place } => {
                if self.load_divergent(place) {
                    self.mark(*local);
                }
            }
            Stmt::Assign { local, value } => {
                if div || *poisoned || self.expr(value) {
                    self.mark(*local);
                }
            }
            Stmt::Store { .. } | Stmt::AtomicRmw { .. } | Stmt::Return => {}
            Stmt::If { cond, then, els } => {
                let d = div || self.expr(cond);
                self.block(then, d, poisoned)?;
                self.block(els, d, poisoned)?;
            }
            Stmt::For {
                local,
                start,
                end,
                step,
                body,
            } => {
                let bound = [start, end, step].into_iter().any(|e| self.expr(e));
                if bound {
                    self.mark(*local);
                }
                self.loop_body(body, div || bound, poisoned)?;
            }
            Stmt::While { cond, body } => {
                let d = div || self.expr(cond);
                self.loop_body(body, d, poisoned)?;
            }
            Stmt::Break | Stmt::Continue => *poisoned |= div,
            Stmt::Barrier { .. } => {
                if self.check && (div || *poisoned) {
                    return Err(KernelError::BarrierDivergent);
                }
            }
        }
        Ok(())
    }
}
