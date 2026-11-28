// SPDX-License-Identifier: MIT OR Apache-2.0
//! **The schedule vocabulary** (board 500 Q2 A, 2026-11-26; `docs/design-tile-and-schedule-plan.md`, phase S1).
//!
//! A [`ScheduleSpec`] states *how* a kernel is to be run, never *what* it computes: the tile shape hierarchy, the stage
//! count, how the work is divided among the blocks, and the epilogue tile. It sits beside the plan-layer
//! [`crate::plan::Schedule`] (which is about the iteration of one op, is `Copy` and is `non_exhaustive`); the name differs on
//! purpose. The fields come from the union rows (CUTLASS 52, 53, 54, 41, 14; CUDA 13, 26, 78, 80; ThunderKittens 59), not from a
//! library's own enum.
//!
//! **What this module is.** Vocabulary and a checker, and nothing else: no field of [`crate::kernel::KernelFn`] carries a
//! spec yet (that is phase S2, in the one breaking release of the plan), and **the oracle ignores a schedule** because a
//! schedule changes how fast a deterministic kernel runs, never its result.
//!
//! **The rule that makes it safe.** An emitter that is handed a spec **honours** every field it sets or **declines it by
//! name**; it never silently ignores one. [`ScheduleSpec::set_fields`] lists the fields a spec sets, and
//! [`ScheduleField::name`] is the name a decline carries.

use serde::{Deserialize, Serialize};

/// Three extents of a tile: `m`, `n` and `k`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TileDims {
    /// Rows of the output tile.
    pub m: u32,
    /// Columns of the output tile.
    pub n: u32,
    /// The contracted extent.
    pub k: u32,
}

impl TileDims {
    /// A tile of `m` x `n` x `k`.
    #[must_use]
    pub const fn new(m: u32, n: u32, k: u32) -> Self {
        Self { m, n, k }
    }

    /// Every extent is nonzero.
    fn positive(self) -> bool {
        self.m > 0 && self.n > 0 && self.k > 0
    }

    /// Every extent of `self` divides the matching extent of `outer`.
    fn divides(self, outer: Self) -> bool {
        self.positive() && outer.m % self.m == 0 && outer.n % self.n == 0 && outer.k % self.k == 0
    }
}

/// The tile shape hierarchy (CUTLASS 52, CUDA 80): the block's tile, optionally the warp's share of it and the instruction's
/// share of that. Each level divides the one above it.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TileHierarchy {
    /// The tile one block (CTA) computes.
    pub cta: TileDims,
    /// The tile one warp computes, when stated.
    pub warp: Option<TileDims>,
    /// The tile one matrix instruction computes, when stated.
    pub instr: Option<TileDims>,
}

/// How many pipeline stages the kernel keeps in flight (CUTLASS 53, CUDA 78 and 26).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum StageCount {
    /// The emitter chooses (CUTLASS `StageCountAuto`).
    Auto,
    /// Exactly this many stages (at least one).
    Fixed(u32),
}

/// How the output tiles are divided among the blocks (CUDA 13, CUTLASS 54).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum WorkDivision {
    /// One tile per block, launched as a grid of them.
    DataParallel,
    /// A fixed number of resident blocks loop over the tiles (`blocks`: that number, or the emitter's choice when `None`).
    Persistent {
        /// Resident blocks, when stated (at least one).
        blocks: Option<u32>,
    },
    /// Stream-K: the contracted axis is split across blocks so every block gets an equal share of the work.
    StreamK,
    /// Split-K: the contracted axis is split into `slices` independent partial products that are combined afterwards.
    SplitK {
        /// How many slices (at least two: one slice is [`WorkDivision::DataParallel`]).
        slices: u32,
    },
}

/// The epilogue's own tiling (CUTLASS 41).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EpilogueSchedule {
    /// The epilogue tile `[m, n]`, when stated.
    pub tile: Option<[u32; 2]>,
    /// The sub-tile `[m, n]` the epilogue works through, when stated; it divides `tile`.
    pub subtile: Option<[u32; 2]>,
}

/// A schedule: how a kernel is run. See the module documentation. A stored schedule may state only the fields it sets: every
/// field not present loads as [`ScheduleSpec::default`]'s (the emitter chooses).
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(default)]
pub struct ScheduleSpec {
    /// The tile shape hierarchy, when stated.
    pub tile: Option<TileHierarchy>,
    /// The stage count.
    pub stages: StageCount,
    /// The shared memory set aside for the pipeline, in bytes, when stated (CUTLASS 53, "smem carveout").
    pub smem_carveout_bytes: Option<u32>,
    /// How the tiles are divided among the blocks.
    pub work: WorkDivision,
    /// The epilogue's tiling, when stated.
    pub epilogue: Option<EpilogueSchedule>,
    /// The cluster shape `[x, y, z]`, when stated.
    pub cluster: Option<[u32; 3]>,
}

impl Default for ScheduleSpec {
    /// The schedule that states nothing: the emitter chooses everything.
    fn default() -> Self {
        Self {
            tile: None,
            stages: StageCount::Auto,
            smem_carveout_bytes: None,
            work: WorkDivision::DataParallel,
            epilogue: None,
            cluster: None,
        }
    }
}

/// A field of a [`ScheduleSpec`]: the unit an emitter honours or declines by name.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ScheduleField {
    /// [`ScheduleSpec::tile`], the block tile.
    Tile,
    /// The warp tile inside [`ScheduleSpec::tile`].
    WarpTile,
    /// The instruction tile inside [`ScheduleSpec::tile`].
    InstrTile,
    /// [`ScheduleSpec::stages`], set to a fixed count.
    Stages,
    /// [`ScheduleSpec::smem_carveout_bytes`].
    SmemCarveout,
    /// [`ScheduleSpec::work`], set to anything but [`WorkDivision::DataParallel`].
    WorkDivision,
    /// [`ScheduleSpec::epilogue`].
    Epilogue,
    /// [`ScheduleSpec::cluster`].
    Cluster,
}

impl ScheduleField {
    /// The name a decline carries.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Tile => "tile",
            Self::WarpTile => "warp tile",
            Self::InstrTile => "instruction tile",
            Self::Stages => "stage count",
            Self::SmemCarveout => "shared-memory carveout",
            Self::WorkDivision => "work division",
            Self::Epilogue => "epilogue schedule",
            Self::Cluster => "cluster shape",
        }
    }
}

impl ScheduleSpec {
    /// The fields this spec sets (a field left at the emitter's choice is not listed), in a fixed order. An emitter that does
    /// not honour one of them declines it by [`ScheduleField::name`].
    #[must_use]
    pub fn set_fields(&self) -> Vec<ScheduleField> {
        let mut out = Vec::new();
        if let Some(t) = &self.tile {
            out.push(ScheduleField::Tile);
            if t.warp.is_some() {
                out.push(ScheduleField::WarpTile);
            }
            if t.instr.is_some() {
                out.push(ScheduleField::InstrTile);
            }
        }
        if matches!(self.stages, StageCount::Fixed(_)) {
            out.push(ScheduleField::Stages);
        }
        if self.smem_carveout_bytes.is_some() {
            out.push(ScheduleField::SmemCarveout);
        }
        if self.work != WorkDivision::DataParallel {
            out.push(ScheduleField::WorkDivision);
        }
        if self.epilogue.is_some() {
            out.push(ScheduleField::Epilogue);
        }
        if self.cluster.is_some() {
            out.push(ScheduleField::Cluster);
        }
        out
    }
}

/// Why a [`ScheduleSpec`] is not well formed. The first violation, in field order.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ScheduleError {
    /// A tile extent is zero (the field named).
    ZeroExtent(ScheduleField),
    /// The warp tile does not divide the block tile, or the instruction tile does not divide the warp tile (or, with no warp
    /// tile, the block tile).
    TileDoesNotNest(ScheduleField),
    /// [`StageCount::Fixed`] of zero.
    NoStages,
    /// A shared-memory carveout of zero bytes.
    ZeroCarveout,
    /// [`WorkDivision::Persistent`] with zero blocks.
    NoPersistentBlocks,
    /// [`WorkDivision::SplitK`] with fewer than two slices.
    SplitKNeedsTwoSlices,
    /// An epilogue tile extent is zero, or the sub-tile does not divide the tile.
    BadEpilogue,
    /// A cluster extent is zero.
    ZeroCluster,
}

/// Check a [`ScheduleSpec`]. It checks the spec's own consistency (extents are positive, the tile levels nest, counts are
/// meaningful); it does **not** say whether a kernel can run under it, which is the emitter's to honour or decline by name.
///
/// # Errors
///
/// The first [`ScheduleError`], in field order.
pub fn check_schedule(s: &ScheduleSpec) -> Result<(), ScheduleError> {
    if let Some(t) = &s.tile {
        if !t.cta.positive() {
            return Err(ScheduleError::ZeroExtent(ScheduleField::Tile));
        }
        if let Some(w) = t.warp {
            if !w.positive() {
                return Err(ScheduleError::ZeroExtent(ScheduleField::WarpTile));
            }
            if !w.divides(t.cta) {
                return Err(ScheduleError::TileDoesNotNest(ScheduleField::WarpTile));
            }
        }
        if let Some(i) = t.instr {
            if !i.positive() {
                return Err(ScheduleError::ZeroExtent(ScheduleField::InstrTile));
            }
            let outer = t.warp.unwrap_or(t.cta);
            if !i.divides(outer) {
                return Err(ScheduleError::TileDoesNotNest(ScheduleField::InstrTile));
            }
        }
    }
    if s.stages == StageCount::Fixed(0) {
        return Err(ScheduleError::NoStages);
    }
    if s.smem_carveout_bytes == Some(0) {
        return Err(ScheduleError::ZeroCarveout);
    }
    match s.work {
        WorkDivision::Persistent { blocks: Some(0) } => {
            return Err(ScheduleError::NoPersistentBlocks);
        }
        WorkDivision::SplitK { slices } if slices < 2 => {
            return Err(ScheduleError::SplitKNeedsTwoSlices);
        }
        _ => {}
    }
    if let Some(e) = &s.epilogue {
        let positive = |t: [u32; 2]| t[0] > 0 && t[1] > 0;
        if e.tile.is_some_and(|t| !positive(t)) || e.subtile.is_some_and(|t| !positive(t)) {
            return Err(ScheduleError::BadEpilogue);
        }
        if let (Some(t), Some(sub)) = (e.tile, e.subtile) {
            if t[0] % sub[0] != 0 || t[1] % sub[1] != 0 {
                return Err(ScheduleError::BadEpilogue);
            }
        }
    }
    if s.cluster.is_some_and(|c| c.contains(&0)) {
        return Err(ScheduleError::ZeroCluster);
    }
    Ok(())
}

/// Specs for an emitter's decline test: each with the result [`check_schedule`] gives. The first is a spec that sets every field
/// and passes, so a decline test can name each field of [`ScheduleSpec::set_fields`]; the rest are refused, one error each.
/// Always present (a few specs, no feature needed), unlike `unpopped::fixtures`, so an integration test can use it.
#[must_use]
pub fn schedule_fixtures() -> Vec<(&'static str, ScheduleSpec, Result<(), ScheduleError>)> {
    let full = ScheduleSpec {
        tile: Some(TileHierarchy {
            cta: TileDims::new(128, 128, 64),
            warp: Some(TileDims::new(64, 64, 64)),
            instr: Some(TileDims::new(16, 8, 16)),
        }),
        stages: StageCount::Fixed(4),
        smem_carveout_bytes: Some(98_304),
        work: WorkDivision::Persistent { blocks: Some(132) },
        epilogue: Some(EpilogueSchedule {
            tile: Some([64, 64]),
            subtile: Some([64, 16]),
        }),
        cluster: Some([2, 1, 1]),
    };
    let with = |f: fn(&mut ScheduleSpec)| {
        let mut s = full.clone();
        f(&mut s);
        s
    };
    vec![
        ("FullySet", full.clone(), Ok(())),
        ("Default", ScheduleSpec::default(), Ok(())),
        (
            "WarpTileDoesNotDivideCta",
            with(|s| {
                if let Some(t) = &mut s.tile {
                    t.warp = Some(TileDims::new(96, 64, 64));
                }
            }),
            Err(ScheduleError::TileDoesNotNest(ScheduleField::WarpTile)),
        ),
        (
            "ZeroStages",
            with(|s| s.stages = StageCount::Fixed(0)),
            Err(ScheduleError::NoStages),
        ),
        (
            "SplitKOfOne",
            with(|s| s.work = WorkDivision::SplitK { slices: 1 }),
            Err(ScheduleError::SplitKNeedsTwoSlices),
        ),
    ]
}
