// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Memory for a launched kernel**: allocation storage with a stored-bit per element, the access
//! log that detects data races, and the [`Machine`] operations that use them
//! (`docs/u3-row4-scopes-and-memory-spaces.md` section 5).
//!
//! Every access is stamped with who made it and how many barriers had released around it. Two
//! accesses to one element conflict (a data race) when at least one writes and **no barrier orders
//! them**: same thread (program order), or the same block with a block barrier released between
//! them, or the same subgroup with a subgroup or block barrier released between them. Accesses by
//! different blocks are never ordered.

use std::collections::HashMap;

use unpopped_vocab::ElementKind;

use super::kernel_eval::{KernelTrap, Machine, atomic_apply};
use super::{Val, elem_bits, read_elem, store_val};
use crate::exec::{ExecId, LaunchShape, MemorySpace};
use crate::ir::ScalarExpr;
use crate::kernel::AtomicOp;
use crate::types::{self, Ty};

/// One access: the thread, and the barrier counts it saw.
#[derive(Clone, Copy)]
pub(super) struct Stamp {
    pub(super) block: u32,
    pub(super) thread: u32,
    pub(super) sub: u32,
    pub(super) block_epoch: u32,
    pub(super) sub_epoch: u32,
}

/// Whether `prev` and `now` are unordered, so a conflict between them is a race.
fn unordered(prev: Stamp, now: Stamp) -> bool {
    if prev.block != now.block {
        return true;
    }
    if prev.thread == now.thread {
        return false;
    }
    let by_block = prev.block_epoch < now.block_epoch;
    let by_subgroup = prev.sub == now.sub && prev.sub_epoch < now.sub_epoch;
    !(by_block || by_subgroup)
}

/// How an access touches an element.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum AccessKind {
    Read,
    Write,
    /// An atomic read-modify-write: it conflicts with a plain access, never with another atomic.
    Atomic,
}

/// What has touched one element: the last plain writer, the atomics since (one per thread) and
/// the plain readers since.
#[derive(Default)]
pub(super) struct Access {
    writer: Option<Stamp>,
    atomics: Vec<Stamp>,
    readers: Vec<Stamp>,
}

/// Element accesses keyed by `(buffer or allocation index, element offset)`.
pub(super) type Log = HashMap<(u32, usize), Access>;

/// Record an access; [`KernelTrap::DataRace`] if it is unordered with a conflicting earlier one.
pub(super) fn touch(
    log: &mut Log,
    key: (u32, usize),
    now: Stamp,
    kind: AccessKind,
) -> Result<(), KernelTrap> {
    let a = log.entry(key).or_default();
    if a.writer.is_some_and(|w| unordered(w, now)) {
        return Err(KernelTrap::DataRace);
    }
    if kind != AccessKind::Atomic && a.atomics.iter().any(|t| unordered(*t, now)) {
        return Err(KernelTrap::DataRace);
    }
    if kind != AccessKind::Read && a.readers.iter().any(|r| unordered(*r, now)) {
        return Err(KernelTrap::DataRace);
    }
    match kind {
        AccessKind::Write => {
            a.writer = Some(now);
            a.atomics.clear();
            a.readers.clear();
        }
        AccessKind::Atomic => match a
            .atomics
            .iter_mut()
            .find(|t| t.block == now.block && t.thread == now.thread)
        {
            Some(t) => *t = now,
            None => a.atomics.push(now),
        },
        AccessKind::Read => note_reader(a, now),
    }
    Ok(())
}

fn note_reader(a: &mut Access, now: Stamp) {
    if let Some(r) = a
        .readers
        .iter_mut()
        .find(|r| r.block == now.block && r.thread == now.thread)
    {
        *r = now;
    } else {
        a.readers.push(now);
    }
}

/// The storage of one allocation instance, with a per-element "has been stored" bit.
pub(super) struct AllocMem {
    dtype: ElementKind,
    bytes: Vec<u8>,
    init: Vec<bool>,
}

impl AllocMem {
    pub(super) fn new(dtype: ElementKind, len: usize) -> Self {
        Self {
            dtype,
            bytes: vec![0; (len * elem_bits(dtype)).div_ceil(8)],
            init: vec![false; len],
        }
    }
}

/// One block's `Shared` instances (indexed by allocation id), their access log, and the barrier
/// counts that order accesses.
pub(super) struct BlockMem {
    pub(super) shared: Vec<Option<AllocMem>>,
    pub(super) log: Log,
    /// Block barriers released so far.
    pub(super) block_epoch: u32,
    /// Subgroup barriers released so far, per subgroup.
    pub(super) sub_epoch: Vec<u32>,
}

/// Everything a running thread needs to know about the launch.
pub(super) struct ThreadCtx<'a> {
    pub(super) launch: LaunchShape,
    pub(super) block_idx: [u32; 3],
    pub(super) thread_idx: [u32; 3],
    pub(super) block_lin: u32,
    pub(super) thread_lin: u32,
    pub(super) subgroup: u32,
    pub(super) private: Vec<Option<AllocMem>>,
    pub(super) block: &'a mut BlockMem,
    pub(super) global: &'a mut Log,
}

impl ThreadCtx<'_> {
    fn stamp(&self) -> Stamp {
        let sub = self.thread_lin / self.subgroup;
        Stamp {
            block: self.block_lin,
            thread: self.thread_lin,
            sub,
            block_epoch: self.block.block_epoch,
            sub_epoch: self.block.sub_epoch[sub as usize],
        }
    }

    fn mem(&self, alloc: u32, space: MemorySpace) -> Option<&AllocMem> {
        match space {
            MemorySpace::Shared => self.block.shared[alloc as usize].as_ref(),
            MemorySpace::Private => self.private[alloc as usize].as_ref(),
            _ => None,
        }
    }

    fn mem_mut(&mut self, alloc: u32, space: MemorySpace) -> Option<&mut AllocMem> {
        match space {
            MemorySpace::Shared => self.block.shared[alloc as usize].as_mut(),
            MemorySpace::Private => self.private[alloc as usize].as_mut(),
            _ => None,
        }
    }

    /// Log an access to a `Shared` element (a no-op for the other spaces).
    fn log_shared(
        &mut self,
        alloc: u32,
        i: usize,
        space: MemorySpace,
        kind: AccessKind,
    ) -> Result<(), KernelTrap> {
        if space == MemorySpace::Shared {
            let now = self.stamp();
            touch(&mut self.block.log, (alloc, i), now, kind)?;
        }
        Ok(())
    }
}

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
        }))
    }

    /// Log a global access by this thread (a no-op outside `run_launch`).
    pub(super) fn note_global(
        &mut self,
        buffer: usize,
        off: usize,
        kind: AccessKind,
    ) -> Result<(), KernelTrap> {
        match &mut self.ctx {
            Some(c) => {
                let now = c.stamp();
                touch(c.global, (buffer as u32, off), now, kind)
            }
            None => Ok(()),
        }
    }

    /// The checked element index of `index` into allocation `alloc`.
    fn alloc_index(&self, alloc: u32, index: &ScalarExpr) -> Result<usize, KernelTrap> {
        let space = self.k.allocs[alloc as usize].space;
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
        usize::try_from(i)
            .ok()
            .filter(|i| *i < len)
            .ok_or(KernelTrap::AllocOutOfBounds { alloc })
    }

    pub(super) fn alloc_load(&mut self, alloc: u32, index: &ScalarExpr) -> Result<Val, KernelTrap> {
        let i = self.alloc_index(alloc, index)?;
        let space = self.k.allocs[alloc as usize].space;
        let c = self
            .ctx
            .as_mut()
            .expect("alloc_index found a launch context");
        c.log_shared(alloc, i, space, AccessKind::Read)?;
        let trap = if space == MemorySpace::Shared {
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

    /// An atomic read-modify-write of a `Shared` element: logged as an atomic access (it races only
    /// with a plain access), and the element must already hold a value.
    pub(super) fn alloc_atomic(
        &mut self,
        op: AtomicOp,
        alloc: u32,
        index: &ScalarExpr,
        v: Val,
    ) -> Result<(), KernelTrap> {
        let i = self.alloc_index(alloc, index)?;
        let space = self.k.allocs[alloc as usize].space;
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
        let Val::Raw(bits, vdt) = v else {
            return Err(KernelTrap::Unsupported("a non-integer atomic operand"));
        };
        let r = atomic_apply(op, m.dtype, read_elem(&m.bytes, i, m.dtype), bits, vdt)?;
        store_val(r, m.dtype, &mut m.bytes, i);
        Ok(())
    }

    pub(super) fn alloc_store(
        &mut self,
        alloc: u32,
        index: &ScalarExpr,
        v: Val,
    ) -> Result<(), KernelTrap> {
        let i = self.alloc_index(alloc, index)?;
        let space = self.k.allocs[alloc as usize].space;
        let c = self
            .ctx
            .as_mut()
            .expect("alloc_index found a launch context");
        c.log_shared(alloc, i, space, AccessKind::Write)?;
        let m = c
            .mem_mut(alloc, space)
            .expect("the space holds an instance");
        store_val(v, m.dtype, &mut m.bytes, i);
        m.init[i] = true;
        Ok(())
    }
}
