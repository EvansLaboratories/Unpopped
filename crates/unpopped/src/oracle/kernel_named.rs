// SPDX-License-Identifier: MIT OR Apache-2.0
//! **What a named barrier does** (ROADMAP U4 S6b-2, `docs/options-u4-sync-and-atomics.md`): `bar.sync id, count` and
//! `bar.arrive id, count`. The block keeps a table of barriers by id. Each arrival counts one thread, and is a
//! release-class access on a key of its own, so what the thread did before it is ordered before whoever waited on the
//! barrier; when `count` arrivals have been made the barrier opens (its phase advances), and every thread that waited
//! (`bar.sync`) in that phase runs again, taking an acquire on the same key first. A `bar.arrive` thread does not wait.
//! Arrivals that disagree on `count` are [`KernelTrap::BarrierMismatch`]; threads left waiting on a barrier that cannot
//! open are [`KernelTrap::MbarrierDeadlock`]. Not modelled: a count that splits a warp (the checker needs whole warps, but
//! the oracle counts threads, not warps), and ordering a named barrier against a `Block` barrier.

use super::kernel_eval::{KernelTrap, Machine};
use super::kernel_mem::AccessKind;
use super::kernel_resume::Stop;
use crate::exec::Scope;
use crate::kernel::MemoryOrder;

/// One named barrier of a block.
#[derive(Clone, Copy, Debug)]
pub(super) struct NamedState {
    count: u32,
    arrived: u32,
    phase: u32,
}

impl NamedState {
    /// Whether the phase a thread arrived in (`phase`) has opened.
    pub(super) fn opened_after(&self, phase: u32) -> bool {
        self.phase > phase
    }
}

impl Machine<'_> {
    /// Arrive at the named barrier `id` of `count` threads; with `wait`, block until it opens.
    pub(super) fn named_barrier(
        &mut self,
        id: u32,
        count: u32,
        wait: bool,
    ) -> Result<Option<Stop>, KernelTrap> {
        self.named_edge(id, MemoryOrder::Release);
        let c = self.ctx.as_mut().ok_or(KernelTrap::Unsupported(
            "a named barrier outside run_launch",
        ))?;
        let s = c.block.named.entry(id).or_insert(NamedState {
            count,
            arrived: 0,
            phase: 0,
        });
        if s.count != count {
            return Err(KernelTrap::BarrierMismatch);
        }
        let phase = s.phase;
        s.arrived += 1;
        let opened = s.arrived == s.count;
        if opened {
            s.arrived = 0;
            s.phase += 1;
        }
        if !wait {
            return Ok(None);
        }
        if opened {
            self.named_acquire(id);
            return Ok(None);
        }
        Ok(Some(Stop::NamedWait { id, phase }))
    }

    /// Take in what the other participants released on the barrier `id`.
    pub(super) fn named_acquire(&mut self, id: u32) {
        self.named_edge(id, MemoryOrder::Acquire);
    }

    fn named_edge(&mut self, id: u32, order: MemoryOrder) {
        self.set_site(order, Scope::Block);
        if let Some(c) = self.ctx.as_mut() {
            let now = c.stamp(AccessKind::Atomic);
            let key = (2, c.block_lin, id, 0);
            c.after(now, AccessKind::Atomic, key);
        }
    }
}
