// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The per-thread driver** (`docs/u3-row4-scopes-and-memory-spaces.md` section 5): [`run_launch`]
//! runs every thread of a launched [`KernelFn`], block by block, in plain Rust.
//!
//! Within a block the threads run in **lockstep phases between barriers**. Each thread is a
//! resumable evaluation (`kernel_resume`, private); a phase runs every runnable thread until it
//! reaches a `Barrier` or finishes. When every thread of the barrier's scope (the block, or the
//! subgroup) is waiting at the same barrier statement, they all advance. A thread that finishes
//! while peers wait for it, or threads waiting at different barriers with none releasable, is
//! [`KernelTrap::BarrierMismatch`].
//!
//! The memory rules (`kernel_mem`, private):
//!
//! * **Shared memory is not zero-filled.** A read of an element no thread stored is
//!   [`KernelTrap::UninitializedShared`]; a zero would hide the missing-barrier bug the memory
//!   space exists to catch. (`Private` is the same per thread: `UninitializedPrivate`.)
//! * **Two accesses to one element, at least one a write, with no barrier ordering them, are
//!   [`KernelTrap::DataRace`]**, for global buffers (accesses by different blocks are never
//!   ordered) and for shared memory. A barrier orders what came before it against what comes
//!   after, for the threads of its scope.
//! * An index outside an allocation is [`KernelTrap::AllocOutOfBounds`].
//!
//! `Constant` allocations have no initializer in [`AllocDecl`], so reading one is
//! [`KernelTrap::Unsupported`] here, by name.

use std::ops::Range;

use super::Val;
use super::kernel_collective::{Contribution, results};
use super::kernel_eval::{KernelArg, KernelTrap, Machine, STEP_LIMIT};
use super::kernel_mem::{AllocMem, BlockMem, Log, ThreadCtx};
use super::kernel_resume::{Frame, Stop, initial_stack};
use super::kernel_sync::Edges;
use crate::exec::{AllocDecl, Extent, LaunchShape, MemorySpace, Scope};
use crate::kernel::{KernelFn, MemoryOrder, Module, ParamKind, check_kernel, check_module};

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
    launch_checked(k, None, args, (subgroup, STEP_LIMIT))
}

/// [`run_launch`] with a lower statement limit per thread than [`STEP_LIMIT`]: a kernel that waits on a block that has
/// not run yet reaches [`KernelTrap::StepLimit`] (or [`KernelTrap::CrossBlockWait`]) after `max_steps` statements
/// instead of ten million. The limit bounds the interpreter, not the language.
///
/// # Errors
///
/// A [`KernelTrap`].
pub fn run_launch_with_limit(
    k: &KernelFn,
    args: &mut [KernelArg],
    max_steps: u64,
) -> Result<(), KernelTrap> {
    check_kernel(k).map_err(KernelTrap::Malformed)?;
    launch_checked(k, None, args, (None, max_steps))
}

/// [`run_launch`] for function `entry` of a [`Module`]: a thread may `Call` the module's plain functions
/// (`docs/options-u3-calls.md`, slice M4a). The module is checked by `check_module` first.
///
/// # Errors
///
/// A [`KernelTrap`]; `Malformed` if the module does not check.
pub fn run_launch_module(m: &Module, entry: u32, args: &mut [KernelArg]) -> Result<(), KernelTrap> {
    check_module(m, entry).map_err(KernelTrap::Malformed)?;
    launch_checked(
        &m.fns[entry as usize],
        Some((m, entry)),
        args,
        (None, STEP_LIMIT),
    )
}

/// Run a checked launched kernel; `module` is the module its calls resolve in and the entry's index.
fn launch_checked(
    k: &KernelFn,
    module: Option<(&Module, u32)>,
    args: &mut [KernelArg],
    (subgroup, limit): (Option<u32>, u64),
) -> Result<(), KernelTrap> {
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
    let mut edges = Edges::default();
    let blocks = launch.grid[0] * launch.grid[1] * launch.grid[2];
    for gz in 0..launch.grid[2] {
        for gy in 0..launch.grid[1] {
            for gx in 0..launch.grid[0] {
                let b = BlockRun {
                    k,
                    module,
                    launch,
                    lens: &lens,
                    sg,
                    idx: [gx, gy, gz],
                    lin: gx + launch.grid[0] * (gy + launch.grid[1] * gz),
                    limit,
                };
                match run_block(&b, args, &mut global, &mut edges) {
                    // blocks run in order: a runaway after an acquire, with blocks still to run, is a wait on one of them
                    Err(KernelTrap::StepLimit) if edges.acquired && b.lin + 1 < blocks => {
                        return Err(KernelTrap::CrossBlockWait);
                    }
                    r => r?,
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

/// What one block run needs besides memory.
struct BlockRun<'a> {
    k: &'a KernelFn,
    /// The module a `Call` resolves in, and the entry's index in it.
    module: Option<(&'a Module, u32)>,
    launch: LaunchShape,
    lens: &'a [usize],
    sg: u32,
    idx: [u32; 3],
    lin: u32,
    limit: u64,
}

/// Where a thread is.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Status {
    Runnable,
    Waiting { scope: Scope, at: usize },
    Done,
}

/// One thread between phases.
struct ThreadRun<'a> {
    idx: [u32; 3],
    lin: u32,
    stack: Vec<Frame<'a>>,
    locals: Vec<Option<Val>>,
    private: Vec<Option<AllocMem>>,
    steps: u64,
    started: bool,
    status: Status,
    /// The function the thread is in (the entry unless it is inside a call) and its buffer map.
    fn_idx: u32,
    bufmap: Vec<usize>,
    /// What the thread contributed to the collective it waits at, until the rendezvous completes.
    contrib: Option<Contribution>,
}

fn run_block(
    b: &BlockRun<'_>,
    args: &mut [KernelArg],
    global: &mut Log,
    edges: &mut Edges,
) -> Result<(), KernelTrap> {
    let [bx, by, bz] = b.launch.block;
    let n = bx * by * bz;
    let mut mem = BlockMem {
        shared: instances(b.k, b.lens, MemorySpace::Shared),
        log: Log::new(),
        block_epoch: 0,
        sub_epoch: vec![0; n.div_ceil(b.sg) as usize],
    };
    let mut threads: Vec<ThreadRun<'_>> = (0..n)
        .map(|lin| ThreadRun {
            idx: [lin % bx, (lin / bx) % by, lin / (bx * by)],
            lin,
            stack: Vec::new(),
            locals: vec![None; b.k.locals.len()],
            private: instances(b.k, b.lens, MemorySpace::Private),
            steps: 0,
            started: false,
            status: Status::Runnable,
            fn_idx: b.module.map_or(0, |(_, e)| e),
            bufmap: Vec::new(),
            contrib: None,
        })
        .collect();
    loop {
        for t in threads.iter_mut().filter(|t| t.status == Status::Runnable) {
            run_phase(b, t, (&mut mem, global, &mut *edges), args)?;
        }
        if threads.iter().all(|t| t.status == Status::Done) {
            return Ok(());
        }
        release(b, &mut threads, &mut mem)?;
    }
}

/// Run one thread until it finishes or reaches a barrier.
fn run_phase<'a>(
    b: &BlockRun<'a>,
    t: &mut ThreadRun<'a>,
    (mem, global, edges): (&mut BlockMem, &mut Log, &mut Edges),
    args: &mut [KernelArg],
) -> Result<(), KernelTrap> {
    let ctx = ThreadCtx {
        launch: b.launch,
        block_idx: b.idx,
        thread_idx: t.idx,
        block_lin: b.lin,
        thread_lin: t.lin,
        subgroup: b.sg,
        private: std::mem::take(&mut t.private),
        block: mem,
        global,
        sync: edges,
        site: (MemoryOrder::Relaxed, Scope::Device),
    };
    let mut m = Machine::new(b.k, args, Some(ctx));
    let module = b.module.map(|(m, _)| m);
    if let Some((module, _)) = b.module {
        // the thread may be inside a callee: put its function, locals' types and buffer map back
        m = m.with_module(module, t.fn_idx);
        m.k = &module.fns[t.fn_idx as usize];
        m.local_tys = m.k.locals.iter().map(|d| d.ty()).collect();
        m.bufmap = std::mem::take(&mut t.bufmap);
    }
    m.locals = std::mem::take(&mut t.locals);
    m.steps = t.steps;
    m.limit = b.limit;
    let stop = if t.started {
        m.resume(&mut t.stack, module)
    } else {
        t.started = true;
        t.stack = initial_stack(b.k);
        m.bind_args().and_then(|()| m.resume(&mut t.stack, module))
    };
    t.locals = std::mem::take(&mut m.locals);
    t.steps = m.steps;
    t.fn_idx = m.fn_idx;
    t.bufmap = std::mem::take(&mut m.bufmap);
    if let Some(c) = m.ctx.take() {
        t.private = c.private;
    }
    t.status = status_after(stop?, t);
    Ok(())
}

/// Where a thread is once it stopped: done, or waiting at a barrier or a collective (holding its contribution).
fn status_after(stop: Stop, t: &mut ThreadRun<'_>) -> Status {
    match stop {
        Stop::Done => Status::Done,
        Stop::Barrier { scope, at } => Status::Waiting { scope, at },
        Stop::Collective { scope, at, c } => {
            t.contrib = Some(c);
            Status::Waiting { scope, at }
        }
    }
}

/// A collective whose every lane has arrived: compute the results and put each in its lane's local. It exchanges
/// registers and orders no memory, so no epoch is counted.
fn deliver(members: &mut [ThreadRun<'_>]) -> Result<(), KernelTrap> {
    let lanes: Vec<Contribution> = members
        .iter()
        .map(|t| t.contrib.expect("every lane contributed"))
        .collect();
    let out = results(&lanes)?;
    for ((t, c), v) in members.iter_mut().zip(&lanes).zip(out) {
        t.locals[c.result as usize] = Some(v);
        t.contrib = None;
    }
    Ok(())
}

/// The threads a barrier of `scope` reached by thread `lin` synchronizes.
fn group_of(b: &BlockRun<'_>, scope: Scope, lin: u32, total: u32) -> Range<u32> {
    match scope {
        Scope::Subgroup => {
            let s = lin / b.sg;
            (s * b.sg)..((s + 1) * b.sg).min(total)
        }
        _ => 0..total,
    }
}

/// Release every group whose threads are all waiting at one barrier, and count the release so
/// the accesses before it are ordered against the ones after. Traps when a group can never be
/// released (a member finished) or when nothing at all can be.
fn release(
    b: &BlockRun<'_>,
    threads: &mut [ThreadRun<'_>],
    mem: &mut BlockMem,
) -> Result<(), KernelTrap> {
    let total = threads.len() as u32;
    let mut groups: Vec<(Scope, Range<u32>)> = Vec::new();
    for t in threads.iter() {
        let Status::Waiting { scope, at } = t.status else {
            continue;
        };
        let group = group_of(b, scope, t.lin, total);
        let members = &threads[group.start as usize..group.end as usize];
        if members.iter().any(|m| m.status == Status::Done) {
            return Err(KernelTrap::BarrierMismatch);
        }
        let all_here = members
            .iter()
            .all(|m| m.status == (Status::Waiting { scope, at }));
        if all_here
            && !groups
                .iter()
                .any(|(_, g)| g.start == group.start && g.end == group.end)
        {
            groups.push((scope, group));
        }
    }
    if groups.is_empty() {
        return Err(KernelTrap::BarrierMismatch);
    }
    for (scope, group) in groups {
        let members = &mut threads[group.start as usize..group.end as usize];
        if members.iter().all(|t| t.contrib.is_some()) {
            deliver(members)?;
        } else if scope == Scope::Subgroup {
            mem.sub_epoch[(group.start / b.sg) as usize] += 1;
        } else {
            mem.block_epoch += 1;
        }
        for t in members {
            t.status = Status::Runnable;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    //! `release` on hand-built thread states. A checked kernel cannot reach most of these (the
    //! checker rejects the divergent control that would strand a barrier), so the guards are
    //! pinned here directly instead of being left as code no test can fire.

    use super::*;
    use crate::kernel::Block as Body;
    use unpopped_vocab::ElementKind;

    const BLOCK: Scope = Scope::Block;
    const SUB: Scope = Scope::Subgroup;

    fn wait(scope: Scope, at: usize) -> Status {
        Status::Waiting { scope, at }
    }

    /// Release `statuses` (one thread each, subgroup size `sg`); the new statuses and epochs.
    fn released(
        statuses: &[Status],
        sg: u32,
    ) -> (Result<(), KernelTrap>, Vec<Status>, u32, Vec<u32>) {
        let n = statuses.len() as u32;
        let k = KernelFn::new("t", ElementKind::F32, vec![], vec![], Body::default());
        let b = BlockRun {
            k: &k,
            module: None,
            launch: LaunchShape::new([n, 1, 1], [1, 1, 1]),
            lens: &[],
            sg,
            idx: [0; 3],
            lin: 0,
            limit: STEP_LIMIT,
        };
        let mut threads: Vec<ThreadRun<'_>> = statuses
            .iter()
            .enumerate()
            .map(|(i, s)| ThreadRun {
                idx: [i as u32, 0, 0],
                lin: i as u32,
                stack: Vec::new(),
                locals: Vec::new(),
                private: Vec::new(),
                steps: 0,
                started: true,
                status: *s,
                fn_idx: 0,
                bufmap: Vec::new(),
                contrib: None,
            })
            .collect();
        let mut mem = BlockMem {
            shared: Vec::new(),
            log: Log::new(),
            block_epoch: 0,
            sub_epoch: vec![0; n.div_ceil(sg) as usize],
        };
        let r = release(&b, &mut threads, &mut mem);
        let now = threads.iter().map(|t| t.status).collect();
        (r, now, mem.block_epoch, mem.sub_epoch)
    }

    #[test]
    fn a_block_barrier_releases_when_every_thread_waits_at_it() {
        let (r, now, block, sub) = released(&[wait(BLOCK, 1); 4], 2);
        assert_eq!(r, Ok(()));
        assert_eq!(now, vec![Status::Runnable; 4]);
        assert_eq!((block, sub), (1, vec![0, 0]));
    }

    #[test]
    fn a_subgroup_barrier_releases_only_its_subgroup() {
        let s = [
            wait(SUB, 1),
            wait(SUB, 1),
            Status::Runnable,
            Status::Runnable,
        ];
        let (r, now, block, sub) = released(&s, 2);
        assert_eq!(r, Ok(()));
        assert_eq!(
            now,
            vec![Status::Runnable; 4],
            "threads 2 and 3 were never waiting"
        );
        assert_eq!((block, sub), (0, vec![1, 0]));
        let s = [wait(SUB, 1), wait(SUB, 1), wait(SUB, 2), Status::Runnable];
        let (_, now, ..) = released(&s, 2);
        assert_eq!(now[2], wait(SUB, 2), "its subgroup is not complete");
    }

    #[test]
    fn threads_waiting_at_different_barriers_are_never_released() {
        let (r, ..) = released(&[wait(BLOCK, 1), wait(BLOCK, 2)], 2);
        assert_eq!(r, Err(KernelTrap::BarrierMismatch));
        // The same statement at a different scope is a different barrier too.
        let (r, ..) = released(&[wait(BLOCK, 1), wait(SUB, 1)], 2);
        assert_eq!(r, Err(KernelTrap::BarrierMismatch));
    }

    #[test]
    fn a_finished_member_strands_its_group_even_when_another_group_could_go() {
        // Subgroup 0 has a finished lane; subgroup 1 is complete. Releasing subgroup 1 and
        // reporting later would hide which group was stranded.
        let s = [wait(SUB, 1), Status::Done, wait(SUB, 2), wait(SUB, 2)];
        let (r, ..) = released(&s, 2);
        assert_eq!(r, Err(KernelTrap::BarrierMismatch));
    }
}
