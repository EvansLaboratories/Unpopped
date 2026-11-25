// SPDX-License-Identifier: MIT OR Apache-2.0
//! Why a kernel run stopped without finishing: the one type the evaluators, the memory rules and the call machinery all
//! report in. A leaf module, so none of them has to reach back into `kernel_eval` for it (ROADMAP U35); the path
//! `kernel_eval::KernelTrap` is kept by a re-export.

use crate::kernel::KernelError;

/// Why a run stopped without finishing.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum KernelTrap {
    /// `check_kernel` refuses the kernel; the interpreter never runs a malformed one.
    Malformed(KernelError),
    /// The arguments do not match the parameters.
    ArgMismatch(&'static str),
    /// A `For` with `step == 0`.
    ZeroStep,
    /// More than `STEP_LIMIT` (`kernel_eval`) statements.
    StepLimit,
    /// A coordinate outside its buffer's logical shape.
    OutOfBounds {
        /// The buffer parameter index.
        buffer: u32,
    },
    /// A `Place::Raw` whose byte offset is negative or not a multiple of the access type's size (ROADMAP U9c).
    MisalignedAccess {
        /// The buffer parameter index.
        buffer: u32,
        /// The byte offset the kernel computed.
        byte_offset: i128,
    },
    /// A construct this interpreter does not evaluate (named), by design not guessed at.
    Unsupported(&'static str),
    /// A `Place::Peer` names a device rank outside the launch.
    DeviceRankOutOfRange {
        /// The rank the kernel computed.
        rank: i128,
    },
    /// A `Place::ClusterPeer` names a block rank outside the cluster.
    ClusterRankOutOfRange {
        /// The rank the kernel computed.
        rank: i128,
    },
    /// An index outside a launched kernel's allocation (never wrapped, never silent).
    AllocOutOfBounds {
        /// The allocation index.
        alloc: u32,
    },
    /// A block ran past the step limit after an acquire-class atomic while later blocks had not yet run. The
    /// oracle runs blocks in order, so a wait on a later block's write can never be satisfied: this is that wait, named,
    /// not a hang (`docs/question-u4-memory-model.md`, sub-question 1; the `Device` barrier model of S6 would run it).
    CrossBlockWait,
    /// A read of a `Shared` element no thread has stored. Shared memory is not zero-filled: a
    /// zero would hide the missing-barrier bug the memory space exists to catch.
    UninitializedShared {
        /// The allocation index.
        alloc: u32,
    },
    /// A read of a `Private` element this thread has not stored.
    UninitializedPrivate {
        /// The allocation index.
        alloc: u32,
    },
    /// Threads of one barrier's scope did not all reach the same barrier: one finished while
    /// peers waited, or they waited at different barriers and none could be released.
    BarrierMismatch,
    /// Two threads touched one element with no ordering between them and at least one wrote it:
    /// a store racing a store, or a read of another thread's store.
    DataRace,
    /// A thread touched a `Shared` element that one of its own `CopyAsync` copies has not yet landed on (read it,
    /// stored it, or copied onto it again before the `AsyncWait` that covers the copy). Hardware gives the old or
    /// the new value; a reference should not.
    AsyncRace {
        /// The allocation index.
        alloc: u32,
    },
    /// A thread finished with `CopyAsync` copies it never waited for (committed or not).
    AsyncUnwaited,
    /// An async-proxy read ([`Stmt::Wgmma`](crate::kernel::Stmt::Wgmma) or `Tcgen05Mma` tiles) reached a `Shared`
    /// element whose last write was a generic-proxy store not followed by that writer's own
    /// [`Stmt::ProxyFence`](crate::kernel::Stmt::ProxyFence) (PTX ISA 9.5, 9.7.17.7.1: "An async proxy fence must be
    /// used ..."). Hardware may read the old or the new value.
    MissingProxyFence {
        /// The allocation index.
        alloc: u32,
    },
    /// An `mbarrier` used outside its contract: arriving at or waiting on one never initialised, initialised with an
    /// arrival count below 1, more arrivals than are pending, or a parity other than 0 or 1.
    MbarrierMisuse {
        /// What was wrong.
        what: &'static str,
    },
    /// Every thread that has not finished waits on an `mbarrier` phase (or at a barrier) that no running thread can
    /// complete.
    MbarrierDeadlock,
}
