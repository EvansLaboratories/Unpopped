// SPDX-License-Identifier: MIT OR Apache-2.0
//! **What the `mbarrier` statements do** (ROADMAP U8c, `docs/options-u8-copy-atoms.md` S3). The object's state (arrival
//! count, pending arrivals, transaction count, phase) lives in the block's `mbar` table beside the `Shared` element that
//! names it; every statement also logs an access to that element, so the race rule and the release/acquire edges of
//! `kernel_sync` apply: an arrival is a release-class atomic at `Block` scope, a successful wait an acquire-class one, an
//! initialisation a plain write. A wait that finds its phase incomplete blocks the thread (`Stop::MbarWait`, the
//! statement is re-run when the scheduler wakes it); the scheduler wakes it when the phase has completed.

use super::kernel_eval::{KernelTrap, Machine};
use super::kernel_mem::AccessKind;
use super::kernel_resume::{Frame, Stop};
use super::{Val, elem_bits};
use crate::exec::{MemorySpace, Scope};
use crate::ir::ScalarExpr;
use crate::kernel::{MemoryOrder, Place, Stmt};
use crate::types::{self, Ty};
use unpopped_vocab::ElementKind;

/// One `mbarrier` object.
#[derive(Clone, Copy, Debug)]
pub(super) struct MbarState {
    expected: i128,
    pending: i128,
    tx: i128,
    phase: u32,
}

impl MbarState {
    /// Whether the phase whose parity is `parity` has completed: the current phase has the other parity.
    pub(super) fn complete(&self, parity: u32) -> bool {
        self.phase % 2 != parity
    }

    /// Complete the phase when no arrival is pending and no transaction bytes are outstanding.
    fn settle(&mut self) {
        if self.pending == 0 && self.tx == 0 {
            self.phase += 1;
            self.pending = self.expected;
        }
    }
}

fn misuse(what: &'static str) -> KernelTrap {
    KernelTrap::MbarrierMisuse { what }
}

impl Machine<'_> {
    /// Run an `mbarrier` statement; `Some` when the thread must block (the statement is left to run again).
    pub(super) fn mbar_stmt<'k>(
        &mut self,
        st: &Stmt,
        stack: &mut [Frame<'k>],
    ) -> Result<Option<Stop>, KernelTrap> {
        match st {
            Stmt::MbarInit { bar, count } => {
                let count = self.mbar_operand(count)?;
                if count < 1 {
                    return Err(misuse(
                        "an mbarrier initialised with an arrival count below 1",
                    ));
                }
                let key = self.mbar_key(bar, AccessKind::Write, None)?;
                self.mbar_table().insert(
                    key,
                    MbarState {
                        expected: count,
                        pending: count,
                        tx: 0,
                        phase: 0,
                    },
                );
            }
            Stmt::MbarArrive { bar, tx } => {
                let tx = tx.as_ref().map(|e| self.mbar_operand(e)).transpose()?;
                self.arrive(bar, tx.unwrap_or(0))?;
            }
            Stmt::Tcgen05Commit { bar } => self.arrive(bar, 0)?,
            Stmt::MbarCompleteTx { bar, bytes } => {
                let bytes = self.mbar_operand(bytes)?;
                self.complete_tx(bar, bytes)?;
            }
            Stmt::MbarWait { bar, parity } => return self.mbar_wait(bar, parity, stack),
            _ => unreachable!("mbar_stmt is called for an mbarrier statement"),
        }
        Ok(None)
    }

    /// A release-class arrival at `bar`, first adding `tx` bytes to its transaction count.
    fn arrive(&mut self, bar: &Place, tx: i128) -> Result<(), KernelTrap> {
        let key = self.mbar_key(bar, AccessKind::Atomic, Some(MemoryOrder::Release))?;
        let s = self.mbar_state(key)?;
        if s.pending == 0 {
            return Err(misuse("an arrival with none pending"));
        }
        s.tx += tx;
        s.pending -= 1;
        s.settle();
        Ok(())
    }

    /// Subtract `bytes` from the transaction count of the barrier at `bar` (a release-class access) and settle it.
    fn complete_tx(&mut self, bar: &Place, bytes: i128) -> Result<(), KernelTrap> {
        let key = self.mbar_key(bar, AccessKind::Atomic, Some(MemoryOrder::Release))?;
        let s = self.mbar_state(key)?;
        s.tx -= bytes;
        s.settle();
        Ok(())
    }

    /// Run a `CopyBulk`: land the data at once (a logged write per element by this thread), then complete its bytes on
    /// the barrier, so a reader needs an acquire on the barrier to be ordered after the data.
    pub(super) fn copy_bulk(
        &mut self,
        dst: &Place,
        src: &Place,
        elems: u32,
        bar: &Place,
    ) -> Result<(), KernelTrap> {
        let Place::Alloc { alloc, index } = dst else {
            return Err(KernelTrap::Unsupported(
                "a bulk copy into something other than an allocation",
            ));
        };
        let Place::Alloc {
            alloc: ba,
            index: bi,
        } = bar
        else {
            return Err(KernelTrap::Unsupported(
                "an mbarrier that is not an allocation place",
            ));
        };
        let bi = self.alloc_index(*ba, bi)?;
        if !self.mbar_table().contains_key(&(*ba, bi)) {
            return Err(misuse("a bulk copy on an mbarrier never initialised"));
        }
        let at = self.alloc_index(*alloc, index)?;
        let n = elems as usize;
        let len = self
            .ctx
            .as_ref()
            .and_then(|c| c.mem(*alloc, MemorySpace::Shared))
            .map_or(0, |m| m.init.len());
        if at + n > len {
            return Err(KernelTrap::AllocOutOfBounds { alloc: *alloc });
        }
        let vals = self.read_source(src, n)?;
        let bits = vals.first().map_or(0, |v| match v {
            Val::Raw(_, dt) => elem_bits(*dt),
            _ => 0,
        });
        for (j, v) in vals.into_iter().enumerate() {
            let index = ScalarExpr::ConstInt((at + j) as i64, ElementKind::I64);
            self.alloc_store(*alloc, &index, v)?;
        }
        self.complete_tx(bar, (bits * n / 8) as i128)
    }

    fn mbar_wait<'k>(
        &mut self,
        bar: &Place,
        parity: &ScalarExpr,
        stack: &mut [Frame<'k>],
    ) -> Result<Option<Stop>, KernelTrap> {
        let parity = self.mbar_operand(parity)?;
        let parity = u32::try_from(parity)
            .ok()
            .filter(|p| *p < 2)
            .ok_or_else(|| misuse("an mbarrier wait whose parity is not 0 or 1"))?;
        let Place::Alloc { alloc, index } = bar else {
            return Err(KernelTrap::Unsupported(
                "an mbarrier that is not an allocation place",
            ));
        };
        let i = self.alloc_index(*alloc, index)?;
        let state = *self
            .mbar_table()
            .get(&(*alloc, i))
            .ok_or_else(|| misuse("a wait on an mbarrier that was never initialised"))?;
        if !state.complete(parity) {
            // stay at this statement: the scheduler runs it again once the phase has completed
            if let Some(Frame::Seq { pc, .. }) = stack.last_mut() {
                *pc -= 1;
            }
            return Ok(Some(Stop::MbarWait {
                alloc: *alloc,
                index: i,
                parity,
            }));
        }
        self.mbar_key(bar, AccessKind::Atomic, Some(MemoryOrder::Acquire))?;
        Ok(None)
    }

    /// An integer operand of an `mbarrier` statement.
    fn mbar_operand(&mut self, e: &ScalarExpr) -> Result<i128, KernelTrap> {
        let Ty::Scalar(t) = types::infer_ty_in(e, self.k.compute, &self.local_tys)
            .map_err(|_| KernelTrap::Unsupported("an mbarrier operand type"))?
        else {
            return Err(KernelTrap::Unsupported("a packed mbarrier operand"));
        };
        self.eval_int(e, t)
    }

    /// The table key of the barrier at `bar`, after logging the access (an atomic one at `order`, `Block` scope).
    fn mbar_key(
        &mut self,
        bar: &Place,
        kind: AccessKind,
        order: Option<MemoryOrder>,
    ) -> Result<(u32, usize), KernelTrap> {
        let Place::Alloc { alloc, index } = bar else {
            return Err(KernelTrap::Unsupported(
                "an mbarrier that is not an allocation place",
            ));
        };
        let i = self.alloc_index(*alloc, index)?;
        if let Some(o) = order {
            self.set_site(o, Scope::Block);
        }
        let c = self
            .ctx
            .as_mut()
            .expect("alloc_index found a launch context");
        c.log_shared(*alloc, i, MemorySpace::Shared, kind)?;
        Ok((*alloc, i))
    }

    fn mbar_table(&mut self) -> &mut std::collections::HashMap<(u32, usize), MbarState> {
        &mut self
            .ctx
            .as_mut()
            .expect("an mbarrier statement runs in a launch")
            .block
            .mbar
    }

    fn mbar_state(&mut self, key: (u32, usize)) -> Result<&mut MbarState, KernelTrap> {
        self.mbar_table()
            .get_mut(&key)
            .ok_or_else(|| misuse("an mbarrier used before it was initialised"))
    }
}
