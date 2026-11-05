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

use crate::exec::{ExecId, MemorySpace, Scope};
use crate::ir::ScalarExpr;
use crate::kernel::{Block, CallArg, KernelError, KernelFn, Place, Stmt};
use crate::kernel_check::ExternSig;
use crate::types::operands;

/// Reject a `Barrier` that may not be reached by every thread of its scope.
///
/// `barrier_callees[i]` is whether function `i` of the module reaches a barrier: a `Call` of it is a barrier
/// site (it must sit in uniform control and pass uniform scalar arguments, since the callee was checked
/// with its scalar parameters uniform).
pub(crate) fn check_barriers(
    k: &KernelFn,
    barrier_callees: &[bool],
    externs: &[ExternSig],
) -> Result<(), KernelError> {
    let mut w = Walk {
        k,
        barrier_callees,
        externs,
        device: holds_device_barrier(&k.body),
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

/// Whether `b` holds a `Barrier` of `Device` or `Cluster` scope: its threads span blocks, so a value read from `BlockIdx`
/// differs between them (the conservative rule for a `Cluster` barrier, whose blocks differ within the cluster).
fn holds_device_barrier(b: &Block) -> bool {
    b.0.iter().any(|st| match st {
        Stmt::Barrier { scope, .. } => matches!(scope, Scope::Device | Scope::Cluster),
        Stmt::If { then, els, .. } => holds_device_barrier(then) || holds_device_barrier(els),
        Stmt::For { body, .. } | Stmt::While { body, .. } => holds_device_barrier(body),
        _ => false,
    })
}

struct Walk<'a> {
    k: &'a KernelFn,
    barrier_callees: &'a [bool],
    externs: &'a [ExternSig],
    /// The kernel holds a `Device` barrier, so `BlockIdx` is divergent.
    device: bool,
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
            ScalarExpr::Exec(ExecId::BlockIdx(_)) => self.device,
            ScalarExpr::Exec(id) => !matches!(id, ExecId::BlockDim(_) | ExecId::GridDim(_)),
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

    /// An `ExternCall`. A pure one's results differ between threads only if an argument does; any other effect may
    /// (a clock, a lane id), so its results are marked. A `sync` call is a barrier site: `div` is whether control
    /// reaching it may differ between threads, and its arguments must be uniform like a call's.
    fn extern_call(
        &mut self,
        decl: u32,
        args: &[ScalarExpr],
        results: &[u32],
        div: bool,
    ) -> Result<(), KernelError> {
        let sig = self.externs.get(decl as usize);
        let divergent_arg = args.iter().any(|a| self.expr(a));
        if sig.is_none_or(|s| !s.pure || divergent_arg) {
            for r in results {
                self.mark(*r);
            }
        }
        if sig.is_some_and(|s| s.sync.is_some()) && self.check && (div || divergent_arg) {
            return Err(KernelError::BarrierDivergent);
        }
        Ok(())
    }

    /// A `Call`. Its result may differ between threads (the conservative mark). A call that reaches a barrier
    /// is a barrier site: `div` is whether control reaching it may differ between threads.
    fn call(
        &mut self,
        callee: u32,
        args: &[CallArg],
        result: Option<u32>,
        div: bool,
    ) -> Result<(), KernelError> {
        if let Some(local) = result {
            self.mark(local);
        }
        let reaches = self
            .barrier_callees
            .get(callee as usize)
            .copied()
            .unwrap_or(false);
        let divergent_arg = args
            .iter()
            .any(|a| matches!(a, CallArg::Scalar(e) if self.expr(e)));
        if reaches && self.check && (div || divergent_arg) {
            return Err(KernelError::BarrierDivergent);
        }
        Ok(())
    }

    fn stmt(&mut self, st: &Stmt, div: bool, poisoned: &mut bool) -> Result<(), KernelError> {
        match st {
            Stmt::Let { .. }
            | Stmt::Load { .. }
            | Stmt::Assign { .. }
            | Stmt::AtomicRmw { .. }
            | Stmt::AtomicCas { .. }
            | Stmt::Collective { .. }
            | Stmt::Call { .. }
            | Stmt::ExternCall { .. } => self.defines(st, div || *poisoned)?,
            Stmt::If { .. } | Stmt::For { .. } | Stmt::While { .. } => {
                self.control(st, div, poisoned)?;
            }
            Stmt::Store { .. }
            | Stmt::Fence { .. }
            | Stmt::CopyAsync { .. }
            | Stmt::AsyncCommit
            | Stmt::AsyncWait { .. }
            | Stmt::Return => {}
            // an MMA is a rendezvous of its subgroup, like a barrier; it defines no local
            Stmt::Mma { .. } => {
                if self.check && (div || *poisoned) {
                    return Err(KernelError::BarrierDivergent);
                }
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

    /// The statements that define a local. `skipped`: control reaching it, or what precedes it, may
    /// differ between threads.
    fn defines(&mut self, st: &Stmt, skipped: bool) -> Result<(), KernelError> {
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
                if skipped || self.expr(value) {
                    self.mark(*local);
                }
            }
            Stmt::Call {
                callee,
                args,
                result,
            } => self.call(*callee, args, *result, skipped)?,
            Stmt::ExternCall {
                decl,
                args,
                results,
            } => self.extern_call(*decl, args, results, skipped)?,
            // A collective is a rendezvous like a barrier: every lane of its scope must reach it, and a lane's
            // result differs from another's.
            Stmt::Collective { result, .. } => {
                self.mark(*result);
                if self.check && skipped {
                    return Err(KernelError::BarrierDivergent);
                }
            }
            // The old value depends on the order the threads reach the element, so it differs
            // between threads whatever the operands are: a barrier under `if old == 0` is divergent.
            Stmt::AtomicRmw { old, .. } | Stmt::AtomicCas { old, .. } => {
                if let Some(local) = old {
                    self.mark(*local);
                }
            }
            _ => unreachable!("defines is called for local-defining statements only"),
        }
        Ok(())
    }

    /// `If`, `For` and `While`: the body runs under control that may differ between threads when the
    /// condition or a bound does.
    fn control(&mut self, st: &Stmt, div: bool, poisoned: &mut bool) -> Result<(), KernelError> {
        match st {
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
            _ => unreachable!("control is called for If, For and While only"),
        }
        Ok(())
    }
}
