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
use unpopped_vocab::layout_algebra::{ComposedLayout, Layout, LayoutError, Swizzle};

/// A level of the execution hierarchy. A `Barrier` synchronizes one of these.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Scope {
    /// One thread.
    Thread,
    /// The lanes of one subgroup (warp, wave, simdgroup).
    Subgroup,
    /// The four consecutive subgroups (warps) of a warpgroup, 128 threads of a launch whose subgroup size is 32
    /// (`wgmma`; ROADMAP U7 S4). A `Barrier` of this scope waits for exactly those threads.
    Warpgroup,
    /// The threads of one block (workgroup).
    Block,
    /// The blocks of one cluster.
    Cluster,
    /// The whole launch on one device.
    Device,
    /// Every device of a multi-device launch (`LaunchShape::devices`, ROADMAP U12 slice 2).
    System,
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
    /// Subgroups in the block: the block's threads divided by the subgroup size, rounded up (WGSL `num_subgroups`, MSL
    /// `simdgroups_per_threadgroup`).
    SubgroupCount,
    /// Index of the device the thread runs on, `0..devices` (`0` in a launch of one device).
    DeviceRank,
    /// Devices of the launch (`LaunchShape::devices`; `1` for a launch of one device).
    DeviceCount,
    /// Index of the block in its cluster, `x` fastest (PTX `%cluster_ctarank`); `0` in a launch without clusters,
    /// where every block is a cluster of one.
    ClusterCtaRank,
    /// Index of the block in its cluster along axis `0..=2` (PTX `%cluster_ctaid`); `0` in a launch without clusters.
    ClusterCtaId(u8),
    /// Extent of the cluster, in blocks, along axis `0..=2` (PTX `%cluster_nctaid`); `1` in a launch without clusters.
    ClusterDim(u8),
    /// Index of the warpgroup in its block: the thread's subgroup index divided by four (a warpgroup is four consecutive
    /// subgroups, [`Scope::Warpgroup`]).
    WarpgroupId,
}

impl ExecId {
    /// The axis this id reads, when it has one.
    #[must_use]
    pub const fn axis(self) -> Option<u8> {
        match self {
            Self::ThreadIdx(a)
            | Self::BlockIdx(a)
            | Self::BlockDim(a)
            | Self::GridDim(a)
            | Self::ClusterCtaId(a)
            | Self::ClusterDim(a) => Some(a),
            Self::LaneId
            | Self::SubgroupId
            | Self::SubgroupSize
            | Self::SubgroupCount
            | Self::DeviceRank
            | Self::DeviceCount
            | Self::ClusterCtaRank
            | Self::WarpgroupId => None,
        }
    }

    /// `true` for the ids that need a subgroup size to mean anything.
    #[must_use]
    pub const fn needs_subgroup(self) -> bool {
        matches!(
            self,
            Self::LaneId
                | Self::SubgroupId
                | Self::SubgroupSize
                | Self::SubgroupCount
                | Self::WarpgroupId
        )
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
            Self::SubgroupCount => "SubgroupCount",
            Self::DeviceRank => "DeviceRank",
            Self::DeviceCount => "DeviceCount",
            Self::ClusterCtaRank => "ClusterCtaRank",
            Self::ClusterCtaId(_) => "ClusterCtaId",
            Self::ClusterDim(_) => "ClusterDim",
            Self::WarpgroupId => "WarpgroupId",
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
    /// Blocks per cluster, per axis (`None`: the launch has no clusters). The grid must be a multiple of it on every
    /// axis. A `Cluster` barrier, atomic scope or fence needs it (since 0.42.18).
    #[serde(default)]
    pub cluster: Option<[u32; 3]>,
    /// Devices the launch spans (at least 1). With more than one, every device runs the whole grid on its own copy of
    /// the buffers, `ExecId::DeviceRank` tells them apart, and a `Place::Peer` reaches another device's copy; a
    /// `System`-scope barrier, atomic or fence orders accesses across devices (ROADMAP U12 slice 2, since 0.44.39).
    #[serde(default = "one_device")]
    pub devices: u32,
    /// A grid derived from scalar parameters (ROADMAP U22). `Some`: these three extents are the grid and `grid` is **not
    /// meaningful** (it is `[1, 1, 1]`); a consumer reads the grid through [`LaunchShape::resolve_grid`] with the launch's
    /// scalar arguments. `None` (the default): `grid` is the grid.
    #[serde(default)]
    pub grid_dims: Option<[GridDim; 3]>,
}

/// One extent of a launch grid that is not a number fixed in the kernel (ROADMAP U22).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum GridDim {
    /// A fixed number of blocks (at least 1).
    Static(u32),
    /// The value of scalar parameter local `.0` (an integer parameter, at least 1 at launch).
    Param(u32),
    /// `ceil(param / by)`: the blocks that cover `param` elements `by` at a time (`by` at least 1; the value of the
    /// parameter at least 1 at launch).
    CeilDiv {
        /// The scalar parameter local holding the element count.
        param: u32,
        /// Elements per block along this axis.
        by: u32,
    },
}

impl GridDim {
    /// The scalar parameter local this extent reads, if any.
    #[must_use]
    pub const fn param(self) -> Option<u32> {
        match self {
            Self::Static(_) => None,
            Self::Param(p) | Self::CeilDiv { param: p, .. } => Some(p),
        }
    }

    /// The number of blocks, given the value of each scalar parameter local (`None` if it has none). `None` when a value is
    /// missing, is below 1, or the result does not fit a `u32`.
    #[must_use]
    pub fn resolve(self, mut value: impl FnMut(u32) -> Option<i128>) -> Option<u32> {
        let n = match self {
            Self::Static(n) => i128::from(n),
            Self::Param(p) => value(p)?,
            Self::CeilDiv { param, by } => {
                if by == 0 {
                    return None;
                }
                let v = value(param)?;
                (v + i128::from(by) - 1).div_euclid(i128::from(by)) * i128::from(v >= 1)
            }
        };
        u32::try_from(n).ok().filter(|n| *n >= 1)
    }
}

const fn one_device() -> u32 {
    1
}

impl LaunchShape {
    /// A launch of `grid` blocks of `block` threads, with no subgroup requirement.
    #[must_use]
    pub const fn new(block: [u32; 3], grid: [u32; 3]) -> Self {
        Self {
            block,
            grid,
            subgroup: None,
            cluster: None,
            devices: 1,
            grid_dims: None,
        }
    }

    /// The same launch with a grid derived from scalar parameters (ROADMAP U22). `grid` becomes `[1, 1, 1]` and is not
    /// meaningful; read the grid through [`LaunchShape::resolve_grid`].
    #[must_use]
    pub const fn with_grid_dims(mut self, dims: [GridDim; 3]) -> Self {
        self.grid = [1, 1, 1];
        self.grid_dims = Some(dims);
        self
    }

    /// The grid of this launch given the value of each scalar parameter local: `grid` itself when it is static, else each
    /// [`GridDim`] resolved. `None` when a parameter has no value, a grid extent resolves to less than 1, or overflows.
    #[must_use]
    pub fn resolve_grid(&self, mut value: impl FnMut(u32) -> Option<i128>) -> Option<[u32; 3]> {
        let Some(d) = self.grid_dims else {
            return Some(self.grid);
        };
        Some([
            d[0].resolve(&mut value)?,
            d[1].resolve(&mut value)?,
            d[2].resolve(&mut value)?,
        ])
    }

    /// The same launch with clusters of `dims` blocks (the grid must be a multiple of it on every axis).
    #[must_use]
    pub const fn with_cluster(mut self, dims: [u32; 3]) -> Self {
        self.cluster = Some(dims);
        self
    }

    /// The same launch spanning `n` devices (at least 1).
    #[must_use]
    pub const fn with_devices(mut self, n: u32) -> Self {
        self.devices = n;
        self
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
    /// Tensor memory (`tcgen05`; ROADMAP U7 S5): one instance per block, 128 lanes of 32-bit cells, declared as
    /// `128 * columns` elements and addressed `lane * columns + column`. Only the `tcgen05` statements reach it: an
    /// ordinary load, store or atomic on it is refused.
    Tensor,
    /// An opaque matrix tile (ROADMAP T2, `docs/design-opaque-tile-mma.md`): `rows * cols` elements that no ordinary access
    /// reaches (a load, a store or an atomic on it is refused); only the tile statements ([`crate::kernel::Stmt::TileLoad`],
    /// `TileMma`, `TileFill`, `TileStore`) read and write it, and the program cannot name an element. The declaration carries
    /// its shape in [`AllocDecl::tile`].
    Tile,
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
    /// How a logical index maps to a physical element (ROADMAP T1, `docs/design-swizzled-allocation.md`). `None` is the identity,
    /// today's meaning, so a document written before the field loads unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layout: Option<AllocLayout>,
    /// The alignment of the allocation's base address in bytes (a power of two). A swizzle pattern is anchored at address zero, so
    /// a swizzled allocation states an alignment that is a multiple of the swizzle's period.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub align_bytes: Option<u32>,
    /// The shape of a [`MemorySpace::Tile`] allocation (`None` on every other space; `check_kernel` requires the two to agree).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tile: Option<TileShape>,
}

/// The shape of an opaque tile: `rows` by `cols` elements of the allocation's element type (ROADMAP T2). A `TileMma` multiplies
/// an `m x k` tile by a `k x n` tile into an `m x n` tile, so the shapes are checked against each other, not against a table.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct TileShape {
    /// Rows.
    pub rows: u32,
    /// Columns.
    pub cols: u32,
}

impl TileShape {
    /// A tile of `rows` by `cols` elements.
    #[must_use]
    pub const fn new(rows: u32, cols: u32) -> Self {
        Self { rows, cols }
    }
}

/// A layout, and optionally a swizzle, on an allocation: `index -> swizzle(layout(index))`, the swizzle in **bytes** (the PTX form
/// `swz(3,4,3)` for the 128-byte mode). Written as one canonical string, `(shape:stride)` or `swz(B,M,S)((shape:stride))`.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct AllocLayout {
    /// Logical index to element offset.
    pub layout: Layout,
    /// The byte-address swizzle applied to the offset, if any.
    pub swizzle: Option<Swizzle>,
}

impl AllocLayout {
    /// A layout with no swizzle.
    #[must_use]
    pub fn new(layout: Layout) -> Self {
        Self {
            layout,
            swizzle: None,
        }
    }

    /// The same layout under the byte swizzle `swizzle`.
    #[must_use]
    pub fn with_swizzle(mut self, swizzle: Swizzle) -> Self {
        self.swizzle = Some(swizzle);
        self
    }

    /// The physical element of logical index `index` for elements of `elem_bytes` bytes: the layout, then the swizzle read over
    /// elements ([`Swizzle::in_elements`]).
    ///
    /// # Errors
    ///
    /// The [`LayoutError`] of the layout or the swizzle (an index outside the layout, or a swizzle base below the element width).
    pub fn physical(&self, index: i64, elem_bytes: u32) -> Result<i64, LayoutError> {
        let off = self.layout.call(index)?;
        match self.swizzle {
            Some(s) => s.in_elements(elem_bytes)?.apply(off),
            None => Ok(off),
        }
    }
}

impl std::fmt::Display for AllocLayout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.swizzle {
            Some(s) => write!(f, "{}", ComposedLayout::new(s, self.layout.clone())),
            None => write!(f, "{}", self.layout),
        }
    }
}

impl std::str::FromStr for AllocLayout {
    type Err = LayoutError;

    fn from_str(text: &str) -> Result<Self, LayoutError> {
        if text.starts_with("swz(") {
            let c: ComposedLayout = text.parse()?;
            Ok(Self::new(c.layout().clone()).with_swizzle(c.swizzle()))
        } else {
            Ok(Self::new(text.parse()?))
        }
    }
}

impl Serialize for AllocLayout {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.collect_str(self)
    }
}

impl<'de> Deserialize<'de> for AllocLayout {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let text = String::deserialize(d)?;
        text.parse().map_err(serde::de::Error::custom)
    }
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
            layout: None,
            align_bytes: None,
            tile: None,
        }
    }

    /// An opaque tile of `rows` by `cols` elements of `elem` ([`MemorySpace::Tile`], extent `rows * cols`).
    #[must_use]
    pub fn tile(name: impl Into<String>, elem: ElementKind, rows: u32, cols: u32) -> Self {
        let mut a = Self::new(
            name,
            MemorySpace::Tile,
            elem,
            Extent::Static(rows.saturating_mul(cols)),
        );
        a.tile = Some(TileShape::new(rows, cols));
        a
    }

    /// The same allocation with `layout` (and its swizzle).
    #[must_use]
    pub fn with_layout(mut self, layout: AllocLayout) -> Self {
        self.layout = Some(layout);
        self
    }

    /// The same allocation with its base address aligned to `bytes` (a power of two; checked by `check_kernel`).
    #[must_use]
    pub fn with_align_bytes(mut self, bytes: u32) -> Self {
        self.align_bytes = Some(bytes);
        self
    }
}
