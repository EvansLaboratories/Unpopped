// SPDX-License-Identifier: MIT OR Apache-2.0
//! The [`Machine`] operations over the memory rules of `kernel_mem` (ROADMAP U35): the machine's accesses, barriers and
//! asynchronous copies. Split from `kernel_mem` so that module holds data and rules only and does not name the machine.

use super::kernel_eval::Machine;
use super::kernel_mem::*;
use super::kernel_trap::KernelTrap;
use super::{Val, read_elem, store_val};
use crate::exec::{ExecId, MemorySpace, Scope};
use crate::ir::ScalarExpr;
use crate::kernel::MemoryOrder;
use crate::types::{self, Ty};
use unpopped_vocab::ElementKind;

impl Machine<'_> {
    /// The value of an execution id for this thread.
    pub(super) fn exec_val(&self, id: ExecId) -> Result<i128, KernelTrap> {
        let c = self.ctx.as_ref().ok_or(KernelTrap::Unsupported(
            "an execution id outside run_launch",
        ))?;
        let axis = |a: u8| usize::from(a);
        Ok(i128::from(match id {
            ExecId::ThreadIdx(a) => c.thread_idx[axis(a)],
            ExecId::BlockIdx(a) => c.block_idx[axis(a)],
            ExecId::BlockDim(a) => c.launch.block[axis(a)],
            ExecId::GridDim(a) => c.launch.grid[axis(a)],
            ExecId::LaneId => c.thread_lin % c.subgroup,
            ExecId::SubgroupId => c.thread_lin / c.subgroup,
            ExecId::SubgroupSize => c.subgroup,
            ExecId::DeviceRank => c.device,
            ExecId::DeviceCount => c.launch.devices,
        }))
    }

    /// State the order and scope of the atomic statement about to run (a no-op outside `run_launch`).
    pub(super) fn set_site(&mut self, order: MemoryOrder, scope: Scope) {
        if let Some(c) = &mut self.ctx {
            c.site = (order, scope);
        }
    }

    /// A fence by this thread (a no-op outside `run_launch`): it orders through `kernel_sync`, touching no memory.
    pub(super) fn fence(&mut self, order: MemoryOrder, scope: Scope) {
        if let Some(c) = &mut self.ctx {
            let now = c.stamp(AccessKind::Read);
            c.sync.fence(now, (order, scope));
        }
    }

    /// Log a global access by this thread (a no-op outside `run_launch`).
    pub(super) fn note_global(
        &mut self,
        buffer: usize,
        off: usize,
        kind: AccessKind,
    ) -> Result<(), KernelTrap> {
        self.note_global_write(buffer, off, kind, false)
    }

    /// [`Self::note_global`] for a store that may write the value the element already holds (`same_value`).
    pub(super) fn note_global_write(
        &mut self,
        buffer: usize,
        off: usize,
        kind: AccessKind,
        same_value: bool,
    ) -> Result<(), KernelTrap> {
        match &mut self.ctx {
            Some(c) => {
                let now = c.stamp(kind);
                let same_value = same_value && c.block.same_value_writes;
                touch(
                    c.global,
                    (buffer as u32, off),
                    now,
                    kind,
                    c.sync,
                    same_value,
                )?;
                c.after(now, kind, (0, 0, buffer as u32, off));
                Ok(())
            }
            None => Ok(()),
        }
    }

    /// The checked element index of `index` into allocation `alloc`.
    pub(super) fn alloc_index(&self, alloc: u32, index: &ScalarExpr) -> Result<usize, KernelTrap> {
        let space = self.alloc_k.allocs[alloc as usize].space;
        if space == MemorySpace::Constant {
            return Err(KernelTrap::Unsupported(
                "a Constant allocation (AllocDecl has no initializer, so none can be read here)",
            ));
        }
        let Ty::Scalar(t) = types::infer_ty_in(index, self.k.compute, &self.local_tys)
            .map_err(|_| KernelTrap::Unsupported("allocation index type"))?
        else {
            return Err(KernelTrap::Unsupported("packed allocation index"));
        };
        let i = self.eval_int(index, t)?;
        let c = self
            .ctx
            .as_ref()
            .ok_or(KernelTrap::Unsupported("an allocation outside run_launch"))?;
        let len = c.mem(alloc, space).map_or(0, |m| m.init.len());
        let i = usize::try_from(i)
            .ok()
            .filter(|i| *i < len)
            .ok_or(KernelTrap::AllocOutOfBounds { alloc })?;
        if c.peer.is_none() && c.asyncq.covers(alloc, i) {
            return Err(KernelTrap::AsyncRace { alloc });
        }
        Ok(i)
    }

    /// Run `access` on the `Shared` memory of the block of this cluster with rank `block` (a
    /// [`crate::kernel::Place::ClusterPeer`]): the rank is evaluated and checked, the access then reaches that block's
    /// array and log, and the running block's own memory is used again afterwards.
    pub(super) fn on_peer<T>(
        &mut self,
        block: &ScalarExpr,
        access: impl FnOnce(&mut Self) -> Result<T, KernelTrap>,
    ) -> Result<T, KernelTrap> {
        let Ty::Scalar(t) = types::infer_ty_in(block, self.k.compute, &self.local_tys)
            .map_err(|_| KernelTrap::Unsupported("cluster rank type"))?
        else {
            return Err(KernelTrap::Unsupported("packed cluster rank"));
        };
        let rank = self.eval_int(block, t)?;
        let c = self.ctx.as_mut().ok_or(KernelTrap::Unsupported(
            "a cluster-peer access outside run_launch",
        ))?;
        let owner = peer_block(&c.launch, c.block_idx, rank)
            .ok_or(KernelTrap::ClusterRankOutOfRange { rank })?
            + c.block_base;
        // the rank of the running block names its own array
        c.peer = (owner != c.block_lin).then_some(owner);
        let r = access(self);
        if let Some(c) = self.ctx.as_mut() {
            c.peer = None;
        }
        r
    }

    /// Every element of this thread's `Private` allocation `alloc`, in order. A trap if any was never stored.
    pub(super) fn private_all(&self, alloc: u32) -> Result<Vec<Val>, KernelTrap> {
        let c = self
            .ctx
            .as_ref()
            .ok_or(KernelTrap::Unsupported("a fragment outside run_launch"))?;
        let m = c
            .mem(alloc, MemorySpace::Private)
            .ok_or(KernelTrap::Unsupported("a fragment that is not Private"))?;
        if m.init.iter().any(|i| !i) {
            return Err(KernelTrap::UninitializedPrivate { alloc });
        }
        Ok((0..m.init.len())
            .map(|i| Val::Raw(read_elem(&m.bytes, i, m.dtype), m.dtype))
            .collect())
    }

    pub(super) fn alloc_load(&mut self, alloc: u32, index: &ScalarExpr) -> Result<Val, KernelTrap> {
        let i = self.alloc_index(alloc, index)?;
        let space = self.alloc_k.allocs[alloc as usize].space;
        let c = self
            .ctx
            .as_mut()
            .expect("alloc_index found a launch context");
        c.log_shared(alloc, i, space, AccessKind::Read)?;
        let trap = if space != MemorySpace::Private {
            KernelTrap::UninitializedShared { alloc }
        } else {
            KernelTrap::UninitializedPrivate { alloc }
        };
        let m = c.mem(alloc, space).expect("the space holds an instance");
        if !m.init[i] {
            return Err(trap);
        }
        Ok(Val::Raw(read_elem(&m.bytes, i, m.dtype), m.dtype))
    }

    /// An atomic step on a `Shared` element: logged as an atomic access (it races only with a plain
    /// access), the element must already hold a value, and `step` maps the value before to the value
    /// after. Returns the value before.
    pub(super) fn alloc_atomic(
        &mut self,
        alloc: u32,
        index: &ScalarExpr,
        step: &dyn Fn(u128, ElementKind) -> Result<Val, KernelTrap>,
    ) -> Result<Val, KernelTrap> {
        let i = self.alloc_index(alloc, index)?;
        let space = self.alloc_k.allocs[alloc as usize].space;
        if space != MemorySpace::Shared {
            return Err(KernelTrap::Unsupported("an atomic outside Shared memory"));
        }
        let c = self
            .ctx
            .as_mut()
            .expect("alloc_index found a launch context");
        c.log_shared(alloc, i, space, AccessKind::Atomic)?;
        let m = c.mem_mut(alloc, space).expect("Shared holds an instance");
        if !m.init[i] {
            return Err(KernelTrap::UninitializedShared { alloc });
        }
        let before = read_elem(&m.bytes, i, m.dtype);
        let r = step(before, m.dtype)?;
        store_val(r, m.dtype, &mut m.bytes, i);
        Ok(Val::Raw(before, m.dtype))
    }

    pub(super) fn alloc_store(
        &mut self,
        alloc: u32,
        index: &ScalarExpr,
        v: Val,
    ) -> Result<(), KernelTrap> {
        let i = self.alloc_index(alloc, index)?;
        let space = self.alloc_k.allocs[alloc as usize].space;
        let c = self
            .ctx
            .as_mut()
            .expect("alloc_index found a launch context");
        // whether this store writes the value the element already holds (a benign race when `Leniency` allows it)
        let same = c.mem(alloc, space).is_some_and(|m| {
            let (esz, bits) = (types::elem_size(m.dtype), types::elem_bits(m.dtype));
            let mut tmp = [0u8; 16];
            store_val(v, m.dtype, &mut tmp, 0);
            bits % 8 == 0 && m.init[i] && m.bytes[i * esz..(i + 1) * esz] == tmp[..esz]
        });
        c.log_shared_write(alloc, i, space, AccessKind::Write, same)?;
        let m = c
            .mem_mut(alloc, space)
            .expect("the space holds an instance");
        store_val(v, m.dtype, &mut m.bytes, i);
        m.init[i] = true;
        Ok(())
    }
}
