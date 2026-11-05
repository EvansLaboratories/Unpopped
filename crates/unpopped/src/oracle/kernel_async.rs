// SPDX-License-Identifier: MIT OR Apache-2.0
//! The oracle's `cp.async` model (ROADMAP U8 S1, `docs/options-u8-copy-atoms.md`): `CopyAsync` reads its source at
//! issue and records the values, `AsyncCommit` closes the open copies into a group, and `AsyncWait` lands every group
//! older than the newest `pending` ones by storing the recorded values (a logged `Shared` write at the wait, so the
//! existing race rules order it against other threads). Touching a destination element before it lands is
//! `KernelTrap::AsyncRace` (checked in `alloc_index`).

use super::kernel_eval::{KernelArg, KernelTrap, Machine};
use super::kernel_mem::{AccessKind, PendingCopy};
use super::{Val, elem_bits, read_elem};
use crate::exec::MemorySpace;
use crate::ir::ScalarExpr;
use crate::kernel::Place;
use unpopped_vocab::ElementKind;

impl Machine<'_> {
    /// Issue a `CopyAsync`: read `elems` source elements now, remember them for the destination.
    pub(super) fn copy_async(
        &mut self,
        dst: &Place,
        src: &Place,
        elems: u32,
    ) -> Result<(), KernelTrap> {
        let Place::Alloc { alloc, index } = dst else {
            return Err(KernelTrap::Unsupported(
                "an async copy into something other than an allocation",
            ));
        };
        if self.alloc_bound(src).is_some() {
            return Err(KernelTrap::Unsupported(
                "an async copy from an allocation-bound parameter",
            ));
        }
        let at = self.alloc_index(*alloc, index)?;
        let n = elems as usize;
        let c = self
            .ctx
            .as_ref()
            .ok_or(KernelTrap::Unsupported("an async copy outside run_launch"))?;
        let len = c
            .mem(*alloc, MemorySpace::Shared)
            .map_or(0, |m| m.init.len());
        if at + n > len {
            return Err(KernelTrap::AllocOutOfBounds { alloc: *alloc });
        }
        if (at..at + n).any(|i| c.asyncq.covers(*alloc, i)) {
            return Err(KernelTrap::AsyncRace { alloc: *alloc });
        }
        let (b, off) = self.offset(src)?;
        let (count, dt) = match &self.args[b] {
            KernelArg::Buffer(tb) => (tb.bytes.len() * 8 / elem_bits(tb.dtype), tb.dtype),
            _ => unreachable!("offset() returned a buffer"),
        };
        if off + n > count {
            return Err(KernelTrap::OutOfBounds { buffer: b as u32 });
        }
        let mut vals = Vec::with_capacity(n);
        for j in 0..n {
            self.note_global(b, off + j, AccessKind::Read)?;
            let KernelArg::Buffer(tb) = &self.args[b] else {
                unreachable!("offset() returned a buffer");
            };
            vals.push(Val::Raw(read_elem(&tb.bytes, off + j, dt), dt));
        }
        let c = self.ctx.as_mut().expect("checked above");
        c.asyncq.open.push(PendingCopy {
            alloc: *alloc,
            at,
            vals,
        });
        Ok(())
    }

    /// Close the open copies into a group (an empty group too, as `cp.async.commit_group` does).
    pub(super) fn async_commit(&mut self) -> Result<(), KernelTrap> {
        let c = self.ctx.as_mut().ok_or(KernelTrap::Unsupported(
            "an async commit outside run_launch",
        ))?;
        let open = std::mem::take(&mut c.asyncq.open);
        c.asyncq.groups.push_back(open);
        Ok(())
    }

    /// Land the oldest groups until at most `pending` remain.
    pub(super) fn async_wait(&mut self, pending: u32) -> Result<(), KernelTrap> {
        loop {
            let c = self
                .ctx
                .as_mut()
                .ok_or(KernelTrap::Unsupported("an async wait outside run_launch"))?;
            if c.asyncq.groups.len() <= pending as usize {
                return Ok(());
            }
            let group = c.asyncq.groups.pop_front().expect("longer than pending");
            for copy in group {
                for (j, v) in copy.vals.into_iter().enumerate() {
                    let index = ScalarExpr::ConstInt((copy.at + j) as i64, ElementKind::I64);
                    self.alloc_store(copy.alloc, &index, v)?;
                }
            }
        }
    }
}
