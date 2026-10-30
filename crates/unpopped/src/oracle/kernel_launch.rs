// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The per-thread driver** (`docs/u3-row4-scopes-and-memory-spaces.md` section 5, steps 1, 3
//! and 4): [`run_launch`] runs every thread of a launched [`KernelFn`], block by block, in plain
//! Rust.
//!
//! Barrier-free only. A thread runs to completion before the next starts, which is sound only if
//! no thread depends on another's work, so the driver *checks* that instead of assuming it:
//!
//! * **Shared memory is not zero-filled.** A read of an element no thread stored is
//!   [`KernelTrap::UninitializedShared`]; a zero would hide the missing-barrier bug the memory
//!   space exists to catch. (`Private` is the same per thread: `UninitializedPrivate`.)
//! * **Any two threads touching one element, at least one of them writing, is
//!   [`KernelTrap::DataRace`]**, for global buffers (across the whole launch) and for shared
//!   memory (within a block). That covers a store racing a store (v0's "later store wins" was
//!   sound for one thread only) and a read of another thread's store, which without a barrier is
//!   the same missing-barrier bug seen from the reader. With no `Barrier` statement every access
//!   is in one phase; PR 3 of the row adds the barrier and resets the logs at it.
//! * An index outside an allocation is [`KernelTrap::AllocOutOfBounds`].
//!
//! `Constant` allocations have no initializer in [`AllocDecl`], so reading one is
//! [`KernelTrap::Unsupported`] here, by name.

use std::collections::HashMap;

use unpopped_vocab::ElementKind;

use super::kernel_eval::{KernelArg, KernelTrap, Machine};
use super::{Val, elem_bits, read_elem, store_val};
use crate::exec::{AllocDecl, ExecId, Extent, LaunchShape, MemorySpace};
use crate::ir::ScalarExpr;
use crate::kernel::{KernelFn, ParamKind, check_kernel};
use crate::types::{self, Ty};

/// `(block linear index, thread linear index in its block)`.
type Who = (u32, u32);

/// What has touched one element: the writer, and the first reader.
///
/// Threads run one after another, so the first reader is the earliest thread to read; any later
/// writer is a different thread and conflicts. PR 3 (barriers, interleaved phases) needs a
/// "several readers" bit here, because there the writer can be the first reader.
#[derive(Default)]
struct Access {
    writer: Option<Who>,
    reader: Option<Who>,
}

/// Element accesses keyed by `(buffer or allocation index, element offset)`.
type Log = HashMap<(u32, usize), Access>;

/// Record an access by `who`; [`KernelTrap::DataRace`] if it conflicts with an earlier one.
fn touch(log: &mut Log, key: (u32, usize), who: Who, write: bool) -> Result<(), KernelTrap> {
    let a = log.entry(key).or_default();
    if a.writer.is_some_and(|w| w != who) {
        return Err(KernelTrap::DataRace);
    }
    if write {
        if a.reader.is_some_and(|r| r != who) {
            return Err(KernelTrap::DataRace);
        }
        a.writer = Some(who);
    } else {
        a.reader.get_or_insert(who);
    }
    Ok(())
}

/// The storage of one allocation instance, with a per-element "has been stored" bit.
pub(super) struct AllocMem {
    dtype: ElementKind,
    bytes: Vec<u8>,
    init: Vec<bool>,
}

impl AllocMem {
    fn new(dtype: ElementKind, len: usize) -> Self {
        Self {
            dtype,
            bytes: vec![0; (len * usize::from(elem_bits(dtype))).div_ceil(8)],
            init: vec![false; len],
        }
    }
}

/// One block's `Shared` instances (indexed by allocation id) and their access log.
pub(super) struct BlockMem {
    shared: Vec<Option<AllocMem>>,
    log: Log,
}

/// Everything a running thread needs to know about the launch.
pub(super) struct ThreadCtx<'a> {
    launch: LaunchShape,
    block_idx: [u32; 3],
    thread_idx: [u32; 3],
    thread_lin: u32,
    subgroup: u32,
    who: Who,
    private: Vec<Option<AllocMem>>,
    block: &'a mut BlockMem,
    global: &'a mut Log,
}

/// Run every thread of `k`'s launch on `args` (one per parameter), choosing the smallest subgroup
/// size the launch allows.
///
/// # Errors
///
/// A [`KernelTrap`].
pub fn run_launch(k: &KernelFn, args: &mut [KernelArg]) -> Result<(), KernelTrap> {
    run_launch_with_subgroup(k, args, None)
}

/// [`run_launch`] with an explicit subgroup size, which must lie in the launch's range.
///
/// # Errors
///
/// A [`KernelTrap`]; [`KernelTrap::ArgMismatch`] when `subgroup` is outside the launch's range.
pub fn run_launch_with_subgroup(
    k: &KernelFn,
    args: &mut [KernelArg],
    subgroup: Option<u32>,
) -> Result<(), KernelTrap> {
    check_kernel(k).map_err(KernelTrap::Malformed)?;
    let Some(launch) = k.launch else {
        return Err(KernelTrap::ArgMismatch(
            "run_launch needs a kernel with a launch shape",
        ));
    };
    if args.len() != k.params.len() {
        return Err(KernelTrap::ArgMismatch("one argument per parameter"));
    }
    let sg = pick_subgroup(&launch, subgroup)?;
    let lens = k
        .allocs
        .iter()
        .map(|a| alloc_len(k, args, a))
        .collect::<Result<Vec<_>, _>>()?;
    let mut global = Log::new();
    let [bx, by, bz] = launch.block;
    for gz in 0..launch.grid[2] {
        for gy in 0..launch.grid[1] {
            for gx in 0..launch.grid[0] {
                let block_lin = gx + launch.grid[0] * (gy + launch.grid[1] * gz);
                let mut block = BlockMem {
                    shared: instances(k, &lens, MemorySpace::Shared),
                    log: Log::new(),
                };
                for tz in 0..bz {
                    for ty in 0..by {
                        for tx in 0..bx {
                            let thread_lin = tx + bx * (ty + by * tz);
                            let ctx = ThreadCtx {
                                launch,
                                block_idx: [gx, gy, gz],
                                thread_idx: [tx, ty, tz],
                                thread_lin,
                                subgroup: sg,
                                who: (block_lin, thread_lin),
                                private: instances(k, &lens, MemorySpace::Private),
                                block: &mut block,
                                global: &mut global,
                            };
                            Machine::new(k, &mut *args, Some(ctx)).run_body()?;
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

fn pick_subgroup(launch: &LaunchShape, want: Option<u32>) -> Result<u32, KernelTrap> {
    let Some(range) = launch.subgroup else {
        return Ok(want.unwrap_or(1).max(1));
    };
    let size = want.unwrap_or(range.min);
    if size < range.min || size > range.max {
        return Err(KernelTrap::ArgMismatch(
            "subgroup size outside the launch's range",
        ));
    }
    Ok(size)
}

/// The element count of `a`, resolving a dynamic extent from its scalar argument.
fn alloc_len(k: &KernelFn, args: &[KernelArg], a: &AllocDecl) -> Result<usize, KernelTrap> {
    let bad = KernelTrap::ArgMismatch("a dynamic extent needs a positive integer argument");
    let n = match a.extent {
        Extent::Static(n) => i128::from(n),
        Extent::Param(local) => {
            let at = k
                .params
                .iter()
                .position(|p| matches!(p.kind, ParamKind::Scalar { local: l, .. } if l == local));
            match at.map(|i| &args[i]) {
                Some(KernelArg::Int(v)) => *v,
                _ => return Err(bad),
            }
        }
    };
    usize::try_from(n).ok().filter(|n| *n > 0).ok_or(bad)
}

/// Fresh instances of every allocation in `space` (and `None` for the others).
fn instances(k: &KernelFn, lens: &[usize], space: MemorySpace) -> Vec<Option<AllocMem>> {
    k.allocs
        .iter()
        .zip(lens)
        .map(|(a, &n)| (a.space == space).then(|| AllocMem::new(a.elem, n)))
        .collect()
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
        write: bool,
    ) -> Result<(), KernelTrap> {
        match &mut self.ctx {
            Some(c) => touch(c.global, (buffer as u32, off), c.who, write),
            None => Ok(()),
        }
    }

    /// The checked element index of `index` into allocation `alloc`.
    fn alloc_index(&self, alloc: u32, index: &ScalarExpr) -> Result<usize, KernelTrap> {
        if self.k.allocs[alloc as usize].space == MemorySpace::Constant {
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
        let len =
            alloc_mem(c, alloc, self.k.allocs[alloc as usize].space).map_or(0, |m| m.init.len());
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
        match space {
            MemorySpace::Shared => touch(&mut c.block.log, (alloc, i), c.who, false)?,
            _ => {}
        }
        let trap = if space == MemorySpace::Shared {
            KernelTrap::UninitializedShared { alloc }
        } else {
            KernelTrap::UninitializedPrivate { alloc }
        };
        let m = alloc_mem(c, alloc, space).expect("the space holds an instance");
        if !m.init[i] {
            return Err(trap);
        }
        Ok(Val::Raw(read_elem(&m.bytes, i, m.dtype), m.dtype))
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
        match space {
            MemorySpace::Shared => touch(&mut c.block.log, (alloc, i), c.who, true)?,
            _ => {}
        }
        let m = alloc_mem_mut(c, alloc, space).expect("the space holds an instance");
        store_val(v, m.dtype, &mut m.bytes, i);
        m.init[i] = true;
        Ok(())
    }
}

fn alloc_mem<'c>(c: &'c ThreadCtx<'_>, alloc: u32, space: MemorySpace) -> Option<&'c AllocMem> {
    match space {
        MemorySpace::Shared => c.block.shared[alloc as usize].as_ref(),
        MemorySpace::Private => c.private[alloc as usize].as_ref(),
        _ => None,
    }
}

fn alloc_mem_mut<'c>(
    c: &'c mut ThreadCtx<'_>,
    alloc: u32,
    space: MemorySpace,
) -> Option<&'c mut AllocMem> {
    match space {
        MemorySpace::Shared => c.block.shared[alloc as usize].as_mut(),
        MemorySpace::Private => c.private[alloc as usize].as_mut(),
        _ => None,
    }
}
