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

use super::kernel_sync::{Edges, ElemKey, Stamp, contains, tid};
use super::kernel_trap::KernelTrap;
use super::{Val, elem_bits, store_val};
use crate::exec::{LaunchShape, MemorySpace, Scope};
use crate::kernel::MemoryOrder;

/// Whether `prev` and `now` are unordered, so a conflict between them is a race: no barrier orders them and `now`
/// has not acquired from `prev` through a release/acquire atomic.
fn unordered(prev: Stamp, now: Stamp, sync: &Edges) -> bool {
    if prev.sys_epoch < now.sys_epoch {
        return false;
    }
    if prev.device == now.device && prev.dev_epoch < now.dev_epoch {
        return false;
    }
    if prev.cluster == now.cluster && prev.cluster_epoch < now.cluster_epoch {
        return false;
    }
    if prev.block == now.block && prev.thread == now.thread {
        return false;
    }
    if prev.block == now.block {
        let by_block = prev.block_epoch < now.block_epoch;
        let by_subgroup = prev.sub == now.sub && prev.sub_epoch < now.sub_epoch;
        if by_block || by_subgroup {
            return false;
        }
    }
    !sync.knows(tid(now), prev)
}

/// Two atomics on one element conflict when the narrower of their scopes does not contain both threads.
fn out_of_scope(prev: Stamp, now: Stamp) -> bool {
    let (a, b) = (prev.scope, now.scope);
    let narrow = if super::kernel_sync::width(a) <= super::kernel_sync::width(b) {
        a
    } else {
        b
    };
    !contains(narrow, tid(prev), tid(now))
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
    sync: &Edges,
) -> Result<(), KernelTrap> {
    let a = log.entry(key).or_default();
    if a.writer.is_some_and(|w| unordered(w, now, sync)) {
        return Err(KernelTrap::DataRace);
    }
    if kind != AccessKind::Atomic && a.atomics.iter().any(|t| unordered(*t, now, sync)) {
        return Err(KernelTrap::DataRace);
    }
    if kind == AccessKind::Atomic && a.atomics.iter().any(|t| out_of_scope(*t, now)) {
        return Err(KernelTrap::DataRace);
    }
    if kind != AccessKind::Read && a.readers.iter().any(|r| unordered(*r, now, sync)) {
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
    pub(super) dtype: ElementKind,
    pub(super) bytes: Vec<u8>,
    pub(super) init: Vec<bool>,
}

impl AllocMem {
    /// Store `v` at element `i` (the element becomes initialized).
    pub(super) fn set(&mut self, i: usize, v: Val) {
        store_val(v, self.dtype, &mut self.bytes, i);
        self.init[i] = true;
    }

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
    /// The `mbarrier` objects of the block, keyed by the allocation and element that name them.
    pub(super) mbar: std::collections::HashMap<(u32, usize), super::kernel_mbar::MbarState>,
    /// The named barriers of the block, by id.
    pub(super) named: std::collections::HashMap<u32, super::kernel_named::NamedState>,
}

/// The linear id of the cluster the block at `idx` belongs to (`0` when the launch has no clusters).
pub(super) fn cluster_of(launch: &LaunchShape, idx: [u32; 3]) -> u32 {
    let Some(c) = launch.cluster else { return 0 };
    let across = |a: usize| launch.grid[a] / c[a];
    ((idx[2] / c[2]) * across(1) + idx[1] / c[1]) * across(0) + idx[0] / c[0]
}

/// The linear index (`x` fastest over the grid) of the block with rank `rank` in the cluster that holds the block at
/// `idx`, or `None` when the rank is outside the cluster (`rank` counts `x` fastest, as `%cluster_ctarank` does).
pub(super) fn peer_block(launch: &LaunchShape, idx: [u32; 3], rank: i128) -> Option<u32> {
    let c = launch.cluster?;
    let size = i128::from(c[0]) * i128::from(c[1]) * i128::from(c[2]);
    if rank < 0 || rank >= size {
        return None;
    }
    let r = u32::try_from(rank).ok()?;
    let at = [r % c[0], (r / c[0]) % c[1], r / (c[0] * c[1])];
    let b = |a: usize| (idx[a] / c[a]) * c[a] + at[a];
    Some(b(0) + launch.grid[0] * (b(1) + launch.grid[1] * b(2)))
}

/// The memory of every block but the running one: those before it in linear order, then those after it. A peer access
/// reaches another block's `Shared` arrays and their access log through it.
pub(super) struct Peers<'a> {
    pub(super) before: &'a mut [BlockMem],
    pub(super) after: &'a mut [BlockMem],
}

impl Peers<'_> {
    /// The memory of block `lin` (never the running block itself).
    fn get(&mut self, lin: u32) -> &mut BlockMem {
        let me = self.before.len();
        let lin = lin as usize;
        if lin < me {
            &mut self.before[lin]
        } else {
            &mut self.after[lin - me - 1]
        }
    }

    fn get_ref(&self, lin: u32) -> &BlockMem {
        let me = self.before.len();
        let lin = lin as usize;
        if lin < me {
            &self.before[lin]
        } else {
            &self.after[lin - me - 1]
        }
    }
}

/// Everything a running thread needs to know about the launch.
pub(super) struct ThreadCtx<'a> {
    pub(super) launch: LaunchShape,
    pub(super) block_idx: [u32; 3],
    pub(super) thread_idx: [u32; 3],
    /// The block's index in the launch, devices laid end to end (`device * blocks per device + linear grid index`).
    pub(super) block_lin: u32,
    pub(super) thread_lin: u32,
    pub(super) subgroup: u32,
    /// The device this thread runs on, the launch-wide id of its cluster, the index of its device's first block, and
    /// the arguments each device owns (the stride of `args`).
    pub(super) device: u32,
    pub(super) cluster: u32,
    pub(super) block_base: u32,
    pub(super) stride: usize,
    pub(super) private: Vec<Option<AllocMem>>,
    /// This thread's `CopyAsync` copies that have not landed.
    pub(super) asyncq: AsyncQueue,
    pub(super) block: &'a mut BlockMem,
    /// The other blocks' memory, for a [`crate::kernel::Place::ClusterPeer`] access.
    pub(super) peers: Peers<'a>,
    /// The block whose `Shared` memory the access now running reaches, when it is another block's.
    pub(super) peer: Option<u32>,
    pub(super) global: &'a mut Log,
    /// The launch-wide release/acquire state, and the order and scope of the atomic statement now running.
    pub(super) sync: &'a mut Edges,
    pub(super) site: (MemoryOrder, Scope),
}

/// One `CopyAsync` copy waiting to land: where, and the values read from the source at issue.
pub(super) struct PendingCopy {
    pub(super) alloc: u32,
    pub(super) at: usize,
    pub(super) vals: Vec<Val>,
}

/// A thread's asynchronous copies: those issued since its last commit, and the committed groups, oldest first.
#[derive(Default)]
pub(super) struct AsyncQueue {
    pub(super) open: Vec<PendingCopy>,
    pub(super) groups: std::collections::VecDeque<Vec<PendingCopy>>,
}

impl AsyncQueue {
    /// Nothing issued is still waiting to land.
    pub(super) fn is_empty(&self) -> bool {
        self.open.is_empty() && self.groups.iter().all(Vec::is_empty)
    }

    /// Whether element `i` of `alloc` is the destination of a copy that has not landed.
    pub(super) fn covers(&self, alloc: u32, i: usize) -> bool {
        self.open
            .iter()
            .chain(self.groups.iter().flatten())
            .any(|c| c.alloc == alloc && (c.at..c.at + c.vals.len()).contains(&i))
    }
}

impl ThreadCtx<'_> {
    /// This thread's stamp for an access of `kind`: its barrier counts, the next sequence number, and (for an
    /// atomic) the scope it is issued at.
    pub(super) fn stamp(&mut self, kind: AccessKind) -> Stamp {
        let sub = self.thread_lin / self.subgroup;
        let cluster = self.cluster;
        Stamp {
            block: self.block_lin,
            thread: self.thread_lin,
            sub,
            block_epoch: self.block.block_epoch,
            sub_epoch: self.block.sub_epoch[sub as usize],
            seq: self.sync.next(),
            dev_epoch: self
                .sync
                .dev_epoch
                .get(self.device as usize)
                .copied()
                .unwrap_or(0),
            device: self.device,
            sys_epoch: self.sync.sys_epoch,
            cluster,
            cluster_epoch: self
                .sync
                .cluster_epoch
                .get(cluster as usize)
                .copied()
                .unwrap_or(0),
            scope: if kind == AccessKind::Atomic {
                self.site.1
            } else {
                Scope::Device
            },
        }
    }

    /// What an access does to the release/acquire state once it is logged: only an atomic does anything, acquiring
    /// and releasing by its order.
    pub(super) fn after(&mut self, now: Stamp, kind: AccessKind, key: ElemKey) {
        if kind == AccessKind::Atomic {
            self.sync.atomic(now, key, self.site);
        }
    }

    pub(super) fn mem(&self, alloc: u32, space: MemorySpace) -> Option<&AllocMem> {
        match space {
            MemorySpace::Shared => match self.peer {
                Some(o) => self.peers.get_ref(o).shared[alloc as usize].as_ref(),
                None => self.block.shared[alloc as usize].as_ref(),
            },
            MemorySpace::Private => self.private[alloc as usize].as_ref(),
            _ => None,
        }
    }

    pub(super) fn mem_mut(&mut self, alloc: u32, space: MemorySpace) -> Option<&mut AllocMem> {
        match space {
            MemorySpace::Shared => match self.peer {
                Some(o) => self.peers.get(o).shared[alloc as usize].as_mut(),
                None => self.block.shared[alloc as usize].as_mut(),
            },
            MemorySpace::Private => self.private[alloc as usize].as_mut(),
            _ => None,
        }
    }

    /// Log an access to a `Shared` element (a no-op for the other spaces).
    pub(super) fn log_shared(
        &mut self,
        alloc: u32,
        i: usize,
        space: MemorySpace,
        kind: AccessKind,
    ) -> Result<(), KernelTrap> {
        if space == MemorySpace::Shared {
            let now = self.stamp(kind);
            let owner = self.peer.unwrap_or(self.block_lin);
            let log = match self.peer {
                Some(o) => &mut self.peers.get(o).log,
                None => &mut self.block.log,
            };
            touch(log, (alloc, i), now, kind, self.sync)?;
            self.after(now, kind, (1, owner, alloc, i));
        }
        Ok(())
    }
}
