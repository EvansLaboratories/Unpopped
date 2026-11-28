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
//!
//! # The oracle limit on forward progress between blocks
//!
//! The oracle is a deterministic sequential interpreter: **blocks run one after another in index order, each to its end or to a
//! `Device` barrier** (with a `Device` barrier every block is resident and the barrier is a rendezvous of the whole grid).
//! Hardware runs resident blocks concurrently, so a block that spins on a flag a LATER block sets completes there and cannot
//! complete here. What the oracle reports for it depends on how the spin reads the flag:
//!
//! * a spin through an acquire-class atomic, with blocks still to run, is [`KernelTrap::CrossBlockWait`] (the wait detector keys
//!   on the launch's `acquired` edge);
//! * a **`Relaxed`** spin takes no acquire edge, so the same kernel shows [`KernelTrap::StepLimit`] after the step limit instead;
//! * a spin on an EARLIER block's write is fine (that block already ran), as is any wait through a `Device` barrier.
//!
//! Neither trap says the kernel is wrong: it says this interpreter cannot interleave blocks. A kernel that needs forward progress
//! between blocks without a `Device` barrier (a decoupled look-back scan, a persistent-block work queue, a stream-K fix-up wait)
//! is outside what the oracle can run today; ROADMAP U76 is the lift (a resident-block scheduler that interleaves blocks at
//! atomic and fence statements, deterministically seeded), and until it lands such a kernel is checked by `check_kernel` and by
//! the emitters' own tests, not by an oracle run.

use std::ops::Range;

use super::Val;
use super::kernel_collective::{Contribution, TupleContribution, results, tuple_results};
use super::kernel_eval::{KernelArg, KernelTrap, Machine, STEP_LIMIT};
use super::kernel_matrix::{
    MatrixContribution, PendingStore, StoreContribution, assemble, distribute,
};
use super::kernel_mem::{AllocMem, AsyncQueue, BlockMem, Log, Peers, ThreadCtx, cluster_of};
use super::kernel_mma::{MmaContribution, compute};
use super::kernel_resume::{Frame, Stop, initial_stack};
use super::kernel_sync::Edges;
use super::kernel_wgmma::{WgmmaContribution, compute as wgmma_compute};
use crate::exec::{AllocDecl, Extent, LaunchShape, MemorySpace, Scope};
use crate::ir::ScalarExpr;
use crate::kernel::{KernelFn, MemoryOrder, Module, check_kernel, check_module};

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
    launch_checked(k, None, args, (subgroup, STEP_LIMIT, Leniency::default()))
}

/// What [`run_launch_lenient`] tolerates that [`run_launch`] refuses. Both are things real kernels do on purpose and hardware
/// shrugs off; the strict default stays the default because each one could also hide a missing barrier.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct Leniency {
    /// Every `Shared` element starts as a stored zero, so a read of one no thread stored is `0` rather than
    /// [`KernelTrap::UninitializedShared`]. For a kernel that reads unwritten shared slots and masks the result afterwards, as
    /// FlashInfer's paged decode does for the tokens past a request's length (a predicated `cp.async` that does not fill, then
    /// `s = in_range ? s : -INF`).
    pub zero_shared: bool,
    /// A `Shared` store of the value the element already holds is not a [`KernelTrap::DataRace`] with another thread's unordered
    /// store of it. For the idiom where every thread of a row stores the same replicated value (FlashInfer's `sync_state`
    /// writes `smem_md[(tz * BDY + ty) * 2] = st.m` from all 16 `tx` threads). A store of a *different* value still races, and
    /// so does a read racing a store.
    pub same_value_writes: bool,
    /// Every `Shared` element starts as a stored **non-zero** byte pattern derived from this seed (byte `i` is
    /// `1 + (37 * i + seed) % 58`, so every byte is in `1..=58` and any half, `f32` or `u64` read from it is finite). Takes
    /// precedence over `zero_shared`. Run a kernel once with zeros and once with a pattern and compare: if the output differs,
    /// it depends on slots the kernel never filled, and `zero_shared` was hiding that.
    pub shared_pattern: Option<u8>,
}

impl Leniency {
    /// Nothing tolerated: the same as [`run_launch`].
    #[must_use]
    pub const fn strict() -> Self {
        Self {
            zero_shared: false,
            same_value_writes: false,
            shared_pattern: None,
        }
    }

    /// With `zero_shared` on.
    #[must_use]
    pub const fn with_zero_shared(mut self) -> Self {
        self.zero_shared = true;
        self
    }

    /// With `same_value_writes` on.
    #[must_use]
    pub const fn with_same_value_writes(mut self) -> Self {
        self.same_value_writes = true;
        self
    }

    /// With the non-zero shared pattern of `seed` (see [`Leniency::shared_pattern`]).
    #[must_use]
    pub const fn with_shared_pattern(mut self, seed: u8) -> Self {
        self.shared_pattern = Some(seed);
        self
    }

    /// Both allowances on (zeroed shared, same-value writes).
    #[must_use]
    pub const fn all() -> Self {
        Self {
            zero_shared: true,
            same_value_writes: true,
            shared_pattern: None,
        }
    }
}

/// [`run_launch`] with the [`Leniency`] given. Every other check and trap is unchanged.
///
/// # Errors
///
/// A [`KernelTrap`].
pub fn run_launch_lenient(
    k: &KernelFn,
    args: &mut [KernelArg],
    leniency: Leniency,
) -> Result<(), KernelTrap> {
    check_kernel(k).map_err(KernelTrap::Malformed)?;
    launch_checked(k, None, args, (None, STEP_LIMIT, leniency))
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
    launch_checked(k, None, args, (None, max_steps, Leniency::default()))
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
        (None, STEP_LIMIT, Leniency::default()),
    )
}

/// Run a launch that spans several devices (`LaunchShape::devices`, ROADMAP U12 slice 2): `devices[d]` holds device `d`'s
/// arguments, one per parameter, so every device has its own copy of each buffer. The grids of all devices run under one
/// schedule (devices in order, the blocks of each in order), so a cross-device access is checked like a cross-block one:
/// ordered by a `System`-scope barrier or release/acquire pair, or a data race. The oracle runs one block to its end
/// before the next, so a block that waits on a flag a later device sets is a [`KernelTrap::CrossBlockWait`].
///
/// # Errors
///
/// A [`KernelTrap`]; [`KernelTrap::ArgMismatch`] when `devices` is not one argument vector per device of the launch.
pub fn run_launch_devices(k: &KernelFn, devices: &mut [Vec<KernelArg>]) -> Result<(), KernelTrap> {
    check_kernel(k).map_err(KernelTrap::Malformed)?;
    let Some(launch) = k.launch else {
        return Err(KernelTrap::ArgMismatch(
            "run_launch_devices needs a kernel with a launch shape",
        ));
    };
    let np = k.params.len();
    if devices.len() != launch.devices as usize || devices.iter().any(|d| d.len() != np) {
        return Err(KernelTrap::ArgMismatch(
            "one argument vector per device, one argument per parameter",
        ));
    }
    let mut flat: Vec<KernelArg> = devices.iter_mut().flat_map(|d| d.drain(..)).collect();
    let r = launch_checked(k, None, &mut flat, (None, STEP_LIMIT, Leniency::default()));
    let mut rest = flat.into_iter();
    for d in devices.iter_mut() {
        d.extend(rest.by_ref().take(np));
    }
    r
}

/// Run a checked launched kernel; `module` is the module its calls resolve in and the entry's index.
fn launch_checked(
    k: &KernelFn,
    module: Option<(&Module, u32)>,
    args: &mut [KernelArg],
    (subgroup, limit, lenient): (Option<u32>, u64, Leniency),
) -> Result<(), KernelTrap> {
    let Some(launch) = k.launch else {
        return Err(KernelTrap::ArgMismatch(
            "run_launch needs a kernel with a launch shape",
        ));
    };
    // a multi-device launch lays the devices' arguments end to end: device `d` owns `args[d * np..][..np]`
    let np = k.params.len();
    if args.len() != np * launch.devices as usize {
        return Err(KernelTrap::ArgMismatch(
            "one argument per parameter, for every device of the launch",
        ));
    }
    let launch = resolve_launch(k, &args[..np], launch)?;
    let sg = pick_subgroup(&launch, subgroup)?;
    let lens = k
        .allocs
        .iter()
        .map(|a| alloc_len(k, &args[..np], a))
        .collect::<Result<Vec<_>, _>>()?;
    let mut global = Log::new();
    let mut edges = Edges::default();
    let mut runs = Vec::new();
    let per_device = launch.grid[0] * launch.grid[1] * launch.grid[2];
    let clusters_per = launch.cluster.map_or(1, |c| {
        (0..3).map(|a| launch.grid[a] / c[a]).product::<u32>()
    });
    for dev in 0..launch.devices {
        for gz in 0..launch.grid[2] {
            for gy in 0..launch.grid[1] {
                for gx in 0..launch.grid[0] {
                    runs.push(BlockRun {
                        k,
                        module,
                        launch,
                        lens: &lens,
                        sg,
                        idx: [gx, gy, gz],
                        lin: dev * per_device + gx + launch.grid[0] * (gy + launch.grid[1] * gz),
                        limit,
                        lenient,
                        place: DevicePlace {
                            device: dev,
                            block_base: dev * per_device,
                            cluster: dev * clusters_per + cluster_of(&launch, [gx, gy, gz]),
                            stride: np,
                        },
                    });
                }
            }
        }
    }
    let mut states: Vec<BlockState<'_>> = runs.iter().map(start_block).collect();
    // every block's memory, apart from its threads, so a block can reach a peer's `Shared` arrays
    let mut mems: Vec<BlockMem> = runs.iter().map(start_mem).collect();
    let mut stops: Vec<Option<BlockStop>> = vec![None; runs.len()];
    let clusters: Vec<u32> = runs.iter().map(|b| b.place.cluster).collect();
    edges.cluster_epoch = vec![0; (clusters_per * launch.devices) as usize];
    edges.dev_epoch = vec![0; launch.devices as usize];
    loop {
        // Blocks run in order, each to its end or to a `Device` barrier. With no device barrier this is one block
        // after another; with one, every block is resident and the barrier is a rendezvous of the whole grid.
        for (i, (b, st)) in runs.iter().zip(states.iter_mut()).enumerate() {
            if stops[i].is_some() {
                continue;
            }
            let (before, rest) = mems.split_at_mut(i);
            let (own, after) = rest.split_first_mut().expect("block i has memory");
            let peers = Peers { before, after };
            stops[i] = Some(
                match advance(b, (st, own, peers), args, &mut global, &mut edges) {
                    // a runaway after an acquire, with blocks still to run, is a wait on one of them
                    Err(KernelTrap::StepLimit) if edges.acquired && i + 1 < runs.len() => {
                        return Err(KernelTrap::CrossBlockWait);
                    }
                    r => r?,
                },
            );
        }
        let Some(go) = release_devices(&stops, &clusters, per_device as usize)? else {
            return Ok(());
        };
        edges.sys_epoch += u32::from(go.system);
        for d in &go.devices {
            edges.dev_epoch[*d as usize] += 1;
        }
        for id in &go.clusters {
            edges.cluster_epoch[*id as usize] += 1;
        }
        for (i, ((st, stop), cluster)) in states
            .iter_mut()
            .zip(stops.iter_mut())
            .zip(&clusters)
            .enumerate()
        {
            let device = (i / per_device as usize) as u32;
            if !(go.system || go.devices.contains(&device) || go.clusters.contains(cluster)) {
                continue;
            }
            *stop = None;
            for t in &mut st.threads {
                t.status = Status::Runnable;
            }
        }
    }
}

/// What may go on in a launch of several devices once every block has stopped: every block of the launch (a `System`
/// barrier), the blocks of these devices (a `Device` barrier), and the blocks of these clusters (a `Cluster` barrier).
#[derive(Debug, Default, PartialEq, Eq)]
struct Go {
    system: bool,
    devices: Vec<u32>,
    clusters: Vec<u32>,
}

/// [`grid_release`] per device, after the `System` barrier: `None` when every block finished. A `System` barrier needs
/// every block of every device at it; the devices otherwise release independently, as the clusters of one device do,
/// and a device that finished strands none of the others (`per_device` blocks each, devices end to end).
fn release_devices(
    stops: &[Option<BlockStop>],
    clusters: &[u32],
    per_device: usize,
) -> Result<Option<Go>, KernelTrap> {
    let stop = |i: usize| stops[i].expect("every block has stopped");
    if (0..stops.len()).all(|i| stop(i) == BlockStop::Done) {
        return Ok(None);
    }
    let on_system = |i: usize| {
        matches!(
            stop(i),
            BlockStop::Wait {
                scope: Scope::System,
                ..
            }
        )
    };
    if (0..stops.len()).any(on_system) {
        return if (0..stops.len()).all(|i| stop(i) == stop(0)) {
            Ok(Some(Go {
                system: true,
                ..Go::default()
            }))
        } else {
            Err(KernelTrap::BarrierMismatch)
        };
    }
    let mut go = Go::default();
    for (d, (s, c)) in stops
        .chunks(per_device)
        .zip(clusters.chunks(per_device))
        .enumerate()
    {
        match grid_release(s, c)? {
            Released::Finished => {}
            Released::Grid => go.devices.push(d as u32),
            Released::Clusters(ids) => go.clusters.extend(ids),
        }
    }
    Ok(Some(go))
}

/// What the blocks that have all stopped may do next.
#[derive(Debug, PartialEq, Eq)]
enum Released {
    /// Every block finished.
    Finished,
    /// Every block waits at one `Device` barrier.
    Grid,
    /// These clusters (by id) have every block at one `Cluster` barrier; the others are finished.
    Clusters(Vec<u32>),
}

/// What may go on once every block has stopped (`clusters[i]` is the cluster of block `i`). A `Device` barrier needs
/// every block of the grid at it; a `Cluster` barrier needs every block of its cluster at it, and clusters are
/// independent, so a finished cluster does not strand another. A block that finished while others of its group wait, or
/// blocks of a group waiting at different barriers, can never be released.
fn grid_release(stops: &[Option<BlockStop>], clusters: &[u32]) -> Result<Released, KernelTrap> {
    let stop = |i: usize| stops[i].expect("every block has stopped");
    if (0..stops.len()).all(|i| stop(i) == BlockStop::Done) {
        return Ok(Released::Finished);
    }
    let on_device = |i: usize| {
        matches!(
            stop(i),
            BlockStop::Wait {
                scope: Scope::Device,
                ..
            }
        )
    };
    if (0..stops.len()).any(on_device) {
        // a finished block among the waiting ones differs from them, so one comparison covers both strandings
        return if (0..stops.len()).all(|i| stop(i) == stop(0)) {
            Ok(Released::Grid)
        } else {
            Err(KernelTrap::BarrierMismatch)
        };
    }
    let mut ids: Vec<u32> = clusters.to_vec();
    ids.sort_unstable();
    ids.dedup();
    let mut go = Vec::new();
    for c in ids {
        let members: Vec<usize> = (0..stops.len()).filter(|&i| clusters[i] == c).collect();
        let first = stop(members[0]);
        if members.iter().any(|&i| stop(i) != first) {
            return Err(KernelTrap::BarrierMismatch);
        }
        if first != BlockStop::Done {
            go.push(c);
        }
    }
    Ok(Released::Clusters(go))
}

/// The launch with its grid made concrete: a grid read from scalar arguments (`LaunchShape::grid_dims`, ROADMAP U22) is
/// resolved from `args` (device 0's, the same on every device), so the rest of the run sees a static grid.
fn resolve_launch(
    k: &KernelFn,
    args: &[KernelArg],
    mut launch: LaunchShape,
) -> Result<LaunchShape, KernelTrap> {
    if launch.grid_dims.is_none() {
        return Ok(launch);
    }
    let value = |local: u32| {
        let (at, field, _) = k.argument_of_local(local)?;
        int_argument(args.get(at)?, field)
    };
    let grid = launch.resolve_grid(value).ok_or(KernelTrap::ArgMismatch(
        "a grid extent needs a positive integer argument that fits a u32",
    ))?;
    launch.grid = grid;
    launch.grid_dims = None;
    Ok(launch)
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

/// The integer `arg` carries: itself, or field `field` of a struct argument (`None` for anything else).
fn int_argument(arg: &KernelArg, field: Option<usize>) -> Option<i128> {
    match (arg, field) {
        (KernelArg::Int(v), None) => Some(*v),
        (KernelArg::Struct(vals), Some(j)) => match vals.get(j) {
            Some(KernelArg::Int(v)) => Some(*v),
            _ => None,
        },
        _ => None,
    }
}

/// The element count of `a`, resolving a dynamic extent from its scalar argument.
fn alloc_len(k: &KernelFn, args: &[KernelArg], a: &AllocDecl) -> Result<usize, KernelTrap> {
    let bad = KernelTrap::ArgMismatch("a dynamic extent needs a positive integer argument");
    let n = match a.extent {
        Extent::Static(n) => i128::from(n),
        Extent::Param(local) => match k
            .argument_of_local(local)
            .and_then(|(i, f, _)| int_argument(&args[i], f))
        {
            Some(v) => v,
            None => return Err(bad),
        },
    };
    usize::try_from(n).ok().filter(|n| *n > 0).ok_or(bad)
}

/// Fresh instances of every allocation in `space` (and `None` for the others).
fn instances(k: &KernelFn, lens: &[usize], space: MemorySpace) -> Vec<Option<AllocMem>> {
    k.allocs
        .iter()
        .zip(lens)
        .map(|(a, &n)| {
            // tensor memory is one instance per block, stored beside `Shared`
            let here = a.space == space
                || (space == MemorySpace::Shared && a.space == MemorySpace::Tensor)
                || (space == MemorySpace::Private && a.space == MemorySpace::Tile);
            here.then(|| AllocMem::new(a.elem, n))
        })
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
    /// The block's index in the launch, devices end to end.
    lin: u32,
    limit: u64,
    /// What the launch tolerates (`run_launch_lenient`).
    lenient: Leniency,
    place: DevicePlace,
}

/// Where a block sits in a multi-device launch.
#[derive(Clone, Copy)]
struct DevicePlace {
    device: u32,
    /// The index of the device's first block.
    block_base: u32,
    /// The launch-wide id of the block's cluster.
    cluster: u32,
    /// The arguments each device owns.
    stride: usize,
}

/// Where a thread is.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Status {
    Runnable,
    Waiting {
        scope: Scope,
        at: usize,
    },
    /// Blocked at a named barrier until the phase it arrived in has opened.
    NamedWait {
        id: u32,
        phase: u32,
    },
    /// Blocked at a `Stmt::MbarWait` until the barrier's phase of that parity has completed.
    MbarWait {
        alloc: u32,
        index: usize,
        parity: u32,
    },
    Done,
}

/// One thread between phases.
struct ThreadRun<'a> {
    idx: [u32; 3],
    lin: u32,
    stack: Vec<Frame<'a>>,
    locals: Vec<Option<Val>>,
    private: Vec<Option<AllocMem>>,
    /// Its `CopyAsync` copies that have not landed.
    asyncq: AsyncQueue,
    steps: u64,
    started: bool,
    status: Status,
    /// The function the thread is in (the entry unless it is inside a call) and its buffer map.
    fn_idx: u32,
    bufmap: Vec<usize>,
    /// What the thread contributed to the collective it waits at, until the rendezvous completes.
    contrib: Option<Contribution>,
    /// What the thread handed over to the `Stmt::Mma` it waits at.
    mma: Option<Box<MmaContribution>>,
    /// What the thread handed over to the `Stmt::Wgmma` it waits at.
    wgmma: Option<Box<WgmmaContribution>>,
    /// The `D` fragments (allocation, values) of this thread's wgmma operations not yet committed, then the committed
    /// groups still in flight, oldest first.
    wg_open: Vec<(u32, Vec<Val>)>,
    wg_groups: std::collections::VecDeque<Vec<(u32, Vec<Val>)>>,
    /// What the thread handed over to the `Stmt::MatrixLoad` it waits at.
    matrix: Option<Box<MatrixContribution>>,
    /// What the thread handed over to the `Stmt::MatrixStore` it waits at.
    matrix_store: Option<Box<StoreContribution>>,
    /// `Shared` writes a completed matrix store left for this thread, performed when it next runs.
    pending: Vec<PendingStore>,
    /// What the thread contributed to the tuple collective it waits at.
    tuple: Option<Box<TupleContribution>>,
    /// The named barrier that opened while the thread waited at it: the acquire it owes before it runs on.
    named_acq: Option<(u32, u32)>,
}

/// One block between `Device` barriers: its threads and its memory.
struct BlockState<'a> {
    threads: Vec<ThreadRun<'a>>,
}

/// Where a block stopped: it finished, or all its threads wait at the `Device` or `Cluster` barrier `at`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum BlockStop {
    Done,
    Wait { scope: Scope, at: usize },
}

/// A block's memory at the start: its `Shared` arrays, an empty access log and no barrier released.
fn start_mem(b: &BlockRun<'_>) -> BlockMem {
    let [bx, by, bz] = b.launch.block;
    BlockMem {
        shared: instances(b.k, b.lens, MemorySpace::Shared)
            .into_iter()
            .map(|m| {
                m.map(|mut m| {
                    if let Some(seed) = b.lenient.shared_pattern {
                        for (i, byte) in m.bytes.iter_mut().enumerate() {
                            *byte = 1 + ((37 * i as u64 + u64::from(seed)) % 58) as u8;
                        }
                        m.init.fill(true);
                    } else if b.lenient.zero_shared {
                        m.init.fill(true);
                    }
                    m
                })
            })
            .collect(),
        log: Log::new(),
        block_epoch: 0,
        sub_epoch: vec![0; (bx * by * bz).div_ceil(b.sg) as usize],
        mbar: Default::default(),
        named: Default::default(),
        same_value_writes: b.lenient.same_value_writes,
        tmem: Default::default(),
    }
}

fn start_block<'a>(b: &BlockRun<'a>) -> BlockState<'a> {
    let [bx, by, bz] = b.launch.block;
    let n = bx * by * bz;
    let threads: Vec<ThreadRun<'_>> = (0..n)
        .map(|lin| ThreadRun {
            idx: [lin % bx, (lin / bx) % by, lin / (bx * by)],
            lin,
            stack: Vec::new(),
            locals: vec![None; b.k.locals.len()],
            private: instances(b.k, b.lens, MemorySpace::Private),
            asyncq: AsyncQueue::default(),
            steps: 0,
            started: false,
            status: Status::Runnable,
            fn_idx: b.module.map_or(0, |(_, e)| e),
            bufmap: Vec::new(),
            contrib: None,
            mma: None,
            matrix: None,
            matrix_store: None,
            pending: Vec::new(),
            tuple: None,
            wgmma: None,
            wg_open: Vec::new(),
            wg_groups: std::collections::VecDeque::new(),
            named_acq: None,
        })
        .collect();
    BlockState { threads }
}

/// Run a block until it finishes or all its threads wait at one `Device` barrier.
fn advance<'a>(
    b: &BlockRun<'a>,
    (st, own, peers): (&mut BlockState<'a>, &mut BlockMem, Peers<'_>),
    args: &mut [KernelArg],
    global: &mut Log,
    edges: &mut Edges,
) -> Result<BlockStop, KernelTrap> {
    loop {
        for t in st
            .threads
            .iter_mut()
            .filter(|t| t.status == Status::Runnable)
        {
            let peers = Peers {
                before: &mut *peers.before,
                after: &mut *peers.after,
            };
            run_phase(b, t, ((&mut *own, peers), global, &mut *edges), args)?;
        }
        if st.threads.iter().all(|t| t.status == Status::Done) {
            own.tmem.check_released()?;
            return Ok(BlockStop::Done);
        }
        if wake_mbar_waiters(&mut st.threads, own) | wake_named_waiters(&mut st.threads, own) {
            continue;
        }
        // a thread that stopped only to have a wgmma commit or wait applied is runnable again
        if st.threads.iter().any(|t| t.status == Status::Runnable) {
            continue;
        }
        if let Some((scope, at)) = at_grid_barrier(&st.threads) {
            return Ok(BlockStop::Wait { scope, at });
        }
        let blocked = st
            .threads
            .iter()
            .any(|t| matches!(t.status, Status::MbarWait { .. } | Status::NamedWait { .. }));
        match release(b, &mut st.threads, own) {
            Err(KernelTrap::BarrierMismatch) if blocked => {
                return Err(KernelTrap::MbarrierDeadlock);
            }
            r => r?,
        }
    }
}

/// Make runnable every thread blocked at a `Stmt::MbarWait` whose phase has completed; whether any was.
fn wake_mbar_waiters(threads: &mut [ThreadRun<'_>], mem: &BlockMem) -> bool {
    let mut woke = false;
    for t in threads {
        let Status::MbarWait {
            alloc,
            index,
            parity,
        } = t.status
        else {
            continue;
        };
        if mem
            .mbar
            .get(&(alloc, index))
            .is_some_and(|s| s.complete(parity))
        {
            t.status = Status::Runnable;
            woke = true;
        }
    }
    woke
}

/// Make runnable every thread blocked at a named barrier whose phase has opened (each owes an acquire on it); whether any was.
fn wake_named_waiters(threads: &mut [ThreadRun<'_>], mem: &BlockMem) -> bool {
    let mut woke = false;
    for t in threads {
        let Status::NamedWait { id, phase } = t.status else {
            continue;
        };
        if mem.named.get(&id).is_some_and(|s| s.opened_after(phase)) {
            t.status = Status::Runnable;
            t.named_acq = Some((id, phase));
            woke = true;
        }
    }
    woke
}

/// The `Device` or `Cluster` barrier every thread of the block waits at, if they all wait at the same one.
fn at_grid_barrier(threads: &[ThreadRun<'_>]) -> Option<(Scope, usize)> {
    let Status::Waiting { scope, at } = threads.first()?.status else {
        return None;
    };
    if !matches!(scope, Scope::Device | Scope::Cluster | Scope::System) {
        return None;
    }
    threads
        .iter()
        .all(|t| t.status == Status::Waiting { scope, at })
        .then_some((scope, at))
}

/// Run one thread until it finishes or reaches a barrier.
fn run_phase<'a>(
    b: &BlockRun<'a>,
    t: &mut ThreadRun<'a>,
    ((mem, peers), global, edges): ((&mut BlockMem, Peers<'_>), &mut Log, &mut Edges),
    args: &mut [KernelArg],
) -> Result<(), KernelTrap> {
    let ctx = ThreadCtx {
        launch: b.launch,
        block_idx: b.idx,
        thread_idx: t.idx,
        block_lin: b.lin,
        thread_lin: t.lin,
        subgroup: b.sg,
        device: b.place.device,
        cluster: b.place.cluster,
        block_base: b.place.block_base,
        stride: b.place.stride,
        private: std::mem::take(&mut t.private),
        asyncq: std::mem::take(&mut t.asyncq),
        block: mem,
        peers,
        peer: None,
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
    let pending = std::mem::take(&mut t.pending);
    if let Some((id, phase)) = t.named_acq.take() {
        m.named_acquire(id, phase);
    }
    let stop = if t.started {
        m.perform_pending(pending)
            .and_then(|()| m.resume(&mut t.stack, module))
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
        t.asyncq = c.asyncq;
    }
    t.status = status_after(stop?, t);
    if matches!(t.status, Status::Done) && !t.asyncq.is_empty() {
        return Err(KernelTrap::AsyncUnwaited);
    }
    Ok(())
}

/// Where a thread is once it stopped: done, or waiting at a barrier or a collective (holding its contribution).
fn status_after(stop: Stop, t: &mut ThreadRun<'_>) -> Status {
    match stop {
        Stop::Done => Status::Done,
        Stop::Barrier { scope, at } => Status::Waiting { scope, at },
        Stop::NamedWait { id, phase } => Status::NamedWait { id, phase },
        Stop::Collective { scope, at, c } => {
            t.contrib = Some(c);
            Status::Waiting { scope, at }
        }
        Stop::Tuple { scope, at, c } => {
            t.tuple = Some(c);
            Status::Waiting { scope, at }
        }
        Stop::Wgmma { at, c } => {
            t.wgmma = Some(c);
            Status::Waiting {
                scope: Scope::Warpgroup,
                at,
            }
        }
        Stop::WgmmaCommit => {
            let open = std::mem::take(&mut t.wg_open);
            t.wg_groups.push_back(open);
            Status::Runnable
        }
        Stop::WgmmaWait { pending } => {
            land_wgmma(t, pending);
            Status::Runnable
        }
        Stop::Mma { at, c } => {
            t.mma = Some(c);
            Status::Waiting {
                scope: Scope::Subgroup,
                at,
            }
        }
        Stop::MbarWait {
            alloc,
            index,
            parity,
        } => Status::MbarWait {
            alloc,
            index,
            parity,
        },
        Stop::MatrixStore { at, c } => {
            t.matrix_store = Some(c);
            Status::Waiting {
                scope: Scope::Subgroup,
                at,
            }
        }
        Stop::Matrix { at, c } => {
            t.matrix = Some(c);
            Status::Waiting {
                scope: Scope::Subgroup,
                at,
            }
        }
    }
}

/// A collective whose every lane has arrived: compute the results and put each in its lane's local. It exchanges
/// registers and orders no memory, so no epoch is counted.
fn deliver(
    members: &mut [ThreadRun<'_>],
    module: Option<(&Module, u32)>,
) -> Result<(), KernelTrap> {
    let lanes: Vec<Contribution> = members
        .iter()
        .map(|t| t.contrib.expect("every lane contributed"))
        .collect();
    let out = results(&lanes, &mut |f, a, b| match module {
        Some((m, _)) => super::kernel_call::eval_combine(m, f, a, b),
        None => Err(KernelTrap::Unsupported(
            "a combining collective outside a module (check_kernel refuses it)",
        )),
    })?;
    for ((t, c), v) in members.iter_mut().zip(&lanes).zip(out) {
        t.locals[c.result as usize] = Some(v);
        t.contrib = None;
    }
    Ok(())
}

/// A tuple collective whose every lane has arrived: fold the lanes' states under the combine function and put each
/// lane's state in its result locals.
fn deliver_tuple(
    members: &mut [ThreadRun<'_>],
    module: Option<(&Module, u32)>,
) -> Result<(), KernelTrap> {
    let lanes: Vec<TupleContribution> = members
        .iter_mut()
        .map(|t| *t.tuple.take().expect("every lane contributed"))
        .collect();
    let out = tuple_results(&lanes, &mut |f, a, b| match module {
        Some((m, _)) => super::kernel_call::eval_combine_tuple(m, f, a, b),
        None => Err(KernelTrap::Unsupported(
            "a combining collective outside a module (check_kernel refuses it)",
        )),
    })?;
    for ((t, c), state) in members.iter_mut().zip(&lanes).zip(out) {
        for (local, v) in c.results.iter().zip(state) {
            t.locals[*local as usize] = Some(v);
        }
    }
    Ok(())
}

/// A wgmma whose 128 threads have all arrived: compute `D` and queue each thread's fragment on it, to land at the wait
/// that covers its group. Like an MMA it orders no memory.
fn deliver_wgmma(members: &mut [ThreadRun<'_>]) -> Result<(), KernelTrap> {
    let lanes: Vec<WgmmaContribution> = members
        .iter_mut()
        .map(|t| *t.wgmma.take().expect("every thread handed over"))
        .collect();
    let out = wgmma_compute(&lanes)?;
    for ((t, c), frag) in members.iter_mut().zip(&lanes).zip(out) {
        t.wg_open.push((c.d, frag));
    }
    Ok(())
}

/// Land this thread's committed wgmma groups until at most `pending` remain in flight.
fn land_wgmma(t: &mut ThreadRun<'_>, pending: u32) {
    while t.wg_groups.len() > pending as usize {
        for (d, frag) in t.wg_groups.pop_front().expect("longer than pending") {
            if let Some(mem) = t.private[d as usize].as_mut() {
                for (i, v) in frag.into_iter().enumerate() {
                    mem.set(i, v);
                }
            }
        }
    }
}

/// An MMA whose every lane has arrived: compute `D` and store each lane's fragment in its `Private` array. Like a
/// collective it exchanges registers and orders no memory.
fn deliver_mma(members: &mut [ThreadRun<'_>]) -> Result<(), KernelTrap> {
    let lanes: Vec<MmaContribution> = members
        .iter_mut()
        .map(|t| *t.mma.take().expect("every lane handed over"))
        .collect();
    let out = compute(&lanes)?;
    for ((t, c), frag) in members.iter_mut().zip(&lanes).zip(out) {
        let mem = t.private[c.d as usize]
            .as_mut()
            .ok_or(KernelTrap::Unsupported(
                "an MMA fragment that is not Private",
            ))?;
        for (i, v) in frag.into_iter().enumerate() {
            mem.set(i, v);
        }
    }
    Ok(())
}

/// A matrix load whose every lane has arrived: distribute the tiles and store each lane's fragment in its `Private`
/// array. The rows were read (and race-checked) when the lanes arrived, so this orders no memory.
fn deliver_matrix(members: &mut [ThreadRun<'_>]) -> Result<(), KernelTrap> {
    let lanes: Vec<MatrixContribution> = members
        .iter_mut()
        .map(|t| *t.matrix.take().expect("every lane handed over"))
        .collect();
    let out = distribute(&lanes)?;
    for ((t, c), frag) in members.iter_mut().zip(&lanes).zip(out) {
        let mem = t.private[c.dst as usize]
            .as_mut()
            .ok_or(KernelTrap::Unsupported(
                "a matrix fragment that is not Private",
            ))?;
        for (i, v) in frag.into_iter().enumerate() {
            mem.set(i, v);
        }
    }
    Ok(())
}

/// A matrix store whose every lane has arrived: assemble the tiles and leave each supplying lane the 8 writes of its row,
/// which it performs (and logs) when it next runs, so a race with another lane's access is the usual `DataRace`.
fn deliver_matrix_store(members: &mut [ThreadRun<'_>]) -> Result<(), KernelTrap> {
    let lanes: Vec<StoreContribution> = members
        .iter_mut()
        .map(|t| *t.matrix_store.take().expect("every lane handed over"))
        .collect();
    let rows = assemble(&lanes)?;
    for ((t, c), row) in members.iter_mut().zip(&lanes).zip(rows) {
        let (Some((alloc, first, ty)), Some(vals)) = (c.row, row) else {
            continue;
        };
        for (j, value) in vals.into_iter().enumerate() {
            t.pending.push(PendingStore {
                alloc,
                index: ScalarExpr::ConstInt((first + j as i128) as i64, ty),
                value,
            });
        }
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
        Scope::Warpgroup => {
            let w = 4 * b.sg;
            let s = lin / w;
            (s * w)..((s + 1) * w).min(total)
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
        if members.iter().all(|t| t.wgmma.is_some()) {
            deliver_wgmma(members)?;
        } else if members.iter().all(|t| t.mma.is_some()) {
            deliver_mma(members)?;
        } else if members.iter().all(|t| t.matrix.is_some()) {
            deliver_matrix(members)?;
        } else if members.iter().all(|t| t.matrix_store.is_some()) {
            deliver_matrix_store(members)?;
        } else if members.iter().all(|t| t.tuple.is_some()) {
            deliver_tuple(members, b.module)?;
        } else if members.iter().all(|t| t.contrib.is_some()) {
            deliver(members, b.module)?;
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
            lenient: Leniency::default(),
            place: DevicePlace {
                device: 0,
                block_base: 0,
                cluster: 0,
                stride: 0,
            },
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
                asyncq: AsyncQueue::default(),
                steps: 0,
                started: true,
                status: *s,
                fn_idx: 0,
                bufmap: Vec::new(),
                contrib: None,
                mma: None,
                matrix: None,
                matrix_store: None,
                pending: Vec::new(),
                tuple: None,
                wgmma: None,
                wg_open: Vec::new(),
                wg_groups: std::collections::VecDeque::new(),
                named_acq: None,
            })
            .collect();
        let mut mem = BlockMem {
            shared: Vec::new(),
            log: Log::new(),
            block_epoch: 0,
            sub_epoch: vec![0; n.div_ceil(sg) as usize],
            mbar: Default::default(),
            named: Default::default(),
            same_value_writes: false,
            tmem: Default::default(),
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
    fn at(scope: Scope, at: usize) -> Option<BlockStop> {
        Some(BlockStop::Wait { scope, at })
    }

    fn dev(n: usize) -> Option<BlockStop> {
        at(Scope::Device, n)
    }

    fn clu(n: usize) -> Option<BlockStop> {
        at(Scope::Cluster, n)
    }

    #[test]
    fn the_grid_is_released_only_when_every_block_waits_at_one_device_barrier() {
        let done = Some(BlockStop::Done);
        let one = [0, 0, 0];
        assert_eq!(grid_release(&[done, done], &[0, 0]), Ok(Released::Finished));
        assert_eq!(
            grid_release(&[dev(3), dev(3), dev(3)], &one),
            Ok(Released::Grid)
        );
        // a finished block strands the others, wherever it sits in the order
        for stops in [[done, dev(3)], [dev(3), done]] {
            assert_eq!(
                grid_release(&stops, &[0, 0]),
                Err(KernelTrap::BarrierMismatch)
            );
        }
        // blocks waiting at different barriers can never be released
        assert_eq!(
            grid_release(&[dev(3), dev(4)], &[0, 0]),
            Err(KernelTrap::BarrierMismatch)
        );
        // a Device wait beside a Cluster wait cannot be released either
        assert_eq!(
            grid_release(&[dev(3), clu(3)], &[0, 1]),
            Err(KernelTrap::BarrierMismatch)
        );
    }

    #[test]
    fn a_cluster_is_released_on_its_own_and_a_finished_cluster_strands_nobody() {
        let done = Some(BlockStop::Done);
        // clusters 0 = blocks 0,1 and 1 = blocks 2,3
        let c = [0, 0, 1, 1];
        assert_eq!(
            grid_release(&[clu(1), clu(1), clu(2), clu(2)], &c),
            Ok(Released::Clusters(vec![0, 1]))
        );
        // cluster 1 finished: cluster 0 still goes on
        assert_eq!(
            grid_release(&[clu(1), clu(1), done, done], &c),
            Ok(Released::Clusters(vec![0]))
        );
        // a block finished inside a waiting cluster strands it, and so do different barriers
        assert_eq!(
            grid_release(&[clu(1), done, done, done], &c),
            Err(KernelTrap::BarrierMismatch)
        );
        assert_eq!(
            grid_release(&[clu(1), clu(2), done, done], &c),
            Err(KernelTrap::BarrierMismatch)
        );
    }
}
