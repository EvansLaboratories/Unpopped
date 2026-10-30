// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Execution scopes, ids, launch shape and memory spaces** for the kernel layer
//! (`docs/u3-row4-scopes-and-memory-spaces.md`, sections 2 and 3).
//!
//! Passive data. [`ExecId`] is the payload of the single [`crate::ir::ScalarExpr::Exec`] leaf
//! (decision D-R4-1); a new id is an additive change to this `#[non_exhaustive]` enum, not
//! another breaking `ScalarExpr` variant. [`crate::kernel::check_kernel`] enforces the rules;
//! the interpreter does not run any of it yet (PR 1 of the row lands types and checks only).

use serde::{Deserialize, Serialize};
use unpopped_vocab::ElementKind;

/// A level of the execution hierarchy. A `Barrier` synchronizes one of these.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Scope {
    /// One thread.
    Thread,
    /// The lanes of one subgroup (warp, wave, simdgroup).
    Subgroup,
    /// The threads of one block (workgroup).
    Block,
    /// The blocks of one cluster.
    Cluster,
    /// The whole launch.
    Device,
}

/// Which execution id a [`crate::ir::ScalarExpr::Exec`] leaf reads. Its type is `U32`
/// (decision D-R4-2).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ExecId {
    /// Index of the thread in its block along axis `0..=2`.
    ThreadIdx(u8),
    /// Index of the block in the grid along axis `0..=2`.
    BlockIdx(u8),
    /// Extent of the block along axis `0..=2`.
    BlockDim(u8),
    /// Extent of the grid along axis `0..=2`.
    GridDim(u8),
    /// Index of the thread in its subgroup.
    LaneId,
    /// Index of the subgroup in its block.
    SubgroupId,
    /// Lanes per subgroup.
    SubgroupSize,
}

impl ExecId {
    /// The axis this id reads, when it has one.
    #[must_use]
    pub const fn axis(self) -> Option<u8> {
        match self {
            Self::ThreadIdx(a) | Self::BlockIdx(a) | Self::BlockDim(a) | Self::GridDim(a) => {
                Some(a)
            }
            Self::LaneId | Self::SubgroupId | Self::SubgroupSize => None,
        }
    }

    /// `true` for the ids that need a subgroup size to mean anything.
    #[must_use]
    pub const fn needs_subgroup(self) -> bool {
        matches!(self, Self::LaneId | Self::SubgroupId | Self::SubgroupSize)
    }

    /// The name used in diagnostics.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::ThreadIdx(_) => "ThreadIdx",
            Self::BlockIdx(_) => "BlockIdx",
            Self::BlockDim(_) => "BlockDim",
            Self::GridDim(_) => "GridDim",
            Self::LaneId => "LaneId",
            Self::SubgroupId => "SubgroupId",
            Self::SubgroupSize => "SubgroupSize",
        }
    }
}

/// A subgroup size as a **range**, never one number: a target decides within it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct SubgroupSize {
    /// Smallest size the kernel accepts (at least 1).
    pub min: u32,
    /// Largest size the kernel accepts.
    pub max: u32,
}

impl SubgroupSize {
    /// A range `min..=max`.
    #[must_use]
    pub const fn new(min: u32, max: u32) -> Self {
        Self { min, max }
    }
}

/// The launch geometry of a kernel: a property of the kernel, not of a statement.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct LaunchShape {
    /// Threads per block, per axis.
    pub block: [u32; 3],
    /// Blocks per grid, per axis.
    pub grid: [u32; 3],
    /// `None`: the target decides. `Some`: the kernel requires a size in this range.
    pub subgroup: Option<SubgroupSize>,
}

impl LaunchShape {
    /// A launch of `grid` blocks of `block` threads, with no subgroup requirement.
    #[must_use]
    pub const fn new(block: [u32; 3], grid: [u32; 3]) -> Self {
        Self {
            block,
            grid,
            subgroup: None,
        }
    }

    /// The same launch requiring a subgroup size in `min..=max`.
    #[must_use]
    pub const fn with_subgroup(mut self, min: u32, max: u32) -> Self {
        self.subgroup = Some(SubgroupSize::new(min, max));
        self
    }
}

/// Where an allocation lives.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum MemorySpace {
    /// Device memory. Not declarable as an allocation: a global buffer is a buffer parameter.
    Global,
    /// One instance per block.
    Shared,
    /// One instance per thread.
    Private,
    /// Read-only after launch.
    Constant,
}

/// How many elements an allocation holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Extent {
    /// Fixed at compile time (non-zero).
    Static(u32),
    /// The value of scalar parameter local `.0`, fixed per launch (dynamic shared memory).
    Param(u32),
}

/// A named array in a memory space, addressed by [`crate::kernel::Place::Alloc`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct AllocDecl {
    /// Name, for diagnostics.
    pub name: String,
    /// Where it lives.
    pub space: MemorySpace,
    /// Element type.
    #[serde(with = "crate::text::ek")]
    pub elem: ElementKind,
    /// Element count.
    pub extent: Extent,
}

impl AllocDecl {
    /// An allocation of `extent` elements of `elem` in `space`.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        space: MemorySpace,
        elem: ElementKind,
        extent: Extent,
    ) -> Self {
        Self {
            name: name.into(),
            space,
            elem,
            extent,
        }
    }
}
