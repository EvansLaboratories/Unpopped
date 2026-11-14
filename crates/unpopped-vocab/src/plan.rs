// SPDX-License-Identifier: MIT OR Apache-2.0
//! Plan-layer *descriptors* shared across kernel families: caller
//! preferences and the numerical-guarantee record every plan exposes.
//!
//! These are pure data (driver-free), so they live in the vocabulary crate.
//! The device-coupled `Workspace<'a>` handle (borrowed `DeviceSliceMut`) stays
//! in `baracuda-kernels-types::plan`.

use crate::sku::BackendKind;

/// Hints that influence kernel selection inside a plan's `select`
/// method.
///
/// The fields are intentionally generic across kernel families — each
/// op category may layer its own `*PlanPreference` wrapper on top
/// (e.g. `GroupedPlanPreference` adds grouped-specific knobs) that
/// embeds this struct.
#[derive(Copy, Clone, Debug)]
pub struct PlanPreference {
    /// Maximum workspace the caller is willing to provide. The selector
    /// only considers kernels whose workspace size for the descriptor
    /// fits in this budget. Use `usize::MAX` to disable the constraint.
    pub max_workspace_bytes: usize,
    /// Allow Hopper-specialized (`sm_90a`) kernels in selection. Has no
    /// effect when the `sm90a` feature is off in the underlying kernel
    /// crate (no such kernels exist in the build).
    pub allow_sm90a: bool,
    /// Force a particular backend at plan-selection time, bypassing the
    /// plan's built-in heuristic.
    ///
    /// `None` (the default) lets the plan's per-op-category heuristic
    /// decide. `Some(BackendKind::Cublas)` / `Some(BackendKind::Cutlass)`
    /// override the heuristic when a caller has profiling-driven
    /// information the heuristic doesn't have (or wants deterministic
    /// kernel selection for golden-output testing).
    ///
    /// Plans surface their actual choice through their `sku()` accessor —
    /// inspect `sku.backend` to see what the heuristic picked.
    ///
    /// Returns `Error::Unsupported` from `select` if the requested
    /// backend doesn't have a kernel for the requested
    /// `(layout, epilogue, element)` triple. For example, the cuBLAS
    /// backend doesn't support `EpilogueKind::BiasRelu` (cuBLAS has no
    /// fused-bias-activation GEMM); forcing it on a Bias* epilogue
    /// returns an error rather than silently falling back to CUTLASS.
    pub prefer_backend: Option<BackendKind>,
}

impl Default for PlanPreference {
    fn default() -> Self {
        Self {
            max_workspace_bytes: usize::MAX,
            allow_sm90a: true,
            prefer_backend: None,
        }
    }
}

/// Re-exported from [`crate::sku`], where [`crate::sku::KernelSku`] (its consumer) lives: moving it there removes the
/// `plan` <-> `sku` cycle (ROADMAP U29) without changing this path.
pub use crate::sku::PrecisionGuarantee;
