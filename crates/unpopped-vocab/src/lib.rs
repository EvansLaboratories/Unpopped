//! # unpopped-vocab
//!
//! The **driver-free classifier vocabulary** for kernel generation: the
//! pure-data types that describe *what a kernel operates on* and *how it is
//! keyed for dispatch*, with **no dependency on any device driver, backend, or
//! vendor crate**.
//!
//! - The [`KernelDtype`] umbrella trait + the [`Element`] / [`IntElement`] /
//!   [`FpElement`] / [`BinElement`] / [`BiasElement`] hierarchy and the dtype
//!   wrapper types ([`S8`], [`U8`], [`S4`], [`U4`], [`Bin`], [`F32Strict`],
//!   [`Fp8E4M3`], [`Fp8E5M2`]).
//! - Tag enums ([`ElementKind`], [`MathPrecision`], [`ArchSku`], [`LayoutSku`],
//!   [`EpilogueKind`], [`ActivationKind`], [`OpCategory`], [`BackendKind`], the
//!   op-family discriminants in [`ops`], …).
//! - The structure-key vocabulary ([`StructureKey`], [`OperandDesc`],
//!   [`structure_key`], the per-operand axes) — the classifier INPUT both
//!   Baracuda and Fuel key on.
//! - The dispatch-table types ([`DispatchTable`], [`DispatchEntry`],
//!   [`Implementor`], [`Provenance`], …) and the plan descriptors
//!   ([`PlanPreference`], [`PrecisionGuarantee`]).
//!
//! ## Why this crate exists
//!
//! A kernel generator, a kernel selector and a runtime all have to agree on
//! *what a kernel is keyed by* — but only the runtime needs a device. Bundling
//! the vocabulary with device views means every consumer inherits a driver FFI
//! it has no use for.
//!
//! So the vocabulary is a leaf. It owns the [`DeviceRepr`](crate::DeviceRepr)
//! memory-layout marker and needs only `half` / `float8`; it depends on no
//! backend, no driver, and no vendor crate. Device-side types — tensor views,
//! workspaces, the adapters that turn them into an [`OperandDesc`] — live in the
//! vendor crates that own the device, and those crates depend on *this* one.
//! The dependency only ever points this way.
//!
//! ## Provenance
//!
//! These types were developed in the [Baracuda] workspace as
//! `baracuda-kernel-vocab`, carved out of `baracuda-kernels-types` to shed
//! exactly the CUDA-driver pull described above, and extracted here so that
//! generators and backends from different vendors can share one vocabulary. The
//! commit history came with them — see `docs/history.md` in the repository for
//! how to read it.
//!
//! This crate seeds the reference implementation of the **classifier-vocabulary**
//! sub-standard of KISS (the Kernel Interface Standards Suite).
//!
//! [Baracuda]: https://github.com/ciresnave/baracuda
//!
//! # 1.0-freeze stability
//!
//! The op-family discriminant enums plus the category / backend tags and the
//! auxiliary index-dtype tags are `#[non_exhaustive]`; downstream `match` arms
//! need a `_ =>` catch-all. The kernel-dispatch-keying enums (`ElementKind`,
//! `BiasElementKind`, `LayoutSku`, `ArchSku`, `EpilogueKind`, `ActivationKind`)
//! are **intentionally** exhaustive — adding a variant is a deliberate
//! workspace-wide event that should surface as a build break at every match
//! site.

#![deny(missing_docs)]

pub mod device_repr;
pub mod dispatch;
pub mod element;
pub mod layout;
pub mod ops;
pub mod plan;
pub mod shape_expr;
pub mod sku;
pub mod structure_key;

pub use device_repr::DeviceRepr;
pub use dispatch::{
    CandidateResult, DispatchEntry, DispatchTable, HwStamp, Implementor, MIN_FLIP_MARGIN,
    Provenance, ReportedCandidate, merge, reported_entry, seed_winner, winner_of,
};
pub use element::{
    BiasElement, BiasElementKind, Bin, BinElement, Bool, Complex32, Complex64, Element,
    ElementKind, F32Strict, Fp8E4M3, Fp8E5M2, FpElement, IndexElement, IndexElementKind,
    IndexOutputElement, IndexOutputKind, IntElement, KernelDtype, MathPrecision, S4, S8,
    ScalarType, U4, U8,
};
pub use layout::{ActivationKind, ArchSku, EpilogueKind, LayoutSku};
pub use ops::{
    ArgReduceKind, AttentionKind, BinaryCmpKind, BinaryKind, ConvKind, CrossEntropyTargetKind,
    EmbeddingKind, FftKind, FillMode, GatedActivationKind, GgufBlockFormat, ImageKind,
    IndexingKind, LinalgKind, LossKind, LossReduction, MoeKind, NormalizationKind, PadMode,
    PoolKind, QuantizeKind, RandomKind, ReduceKind, ReduceToOp, ScanKind, SegmentKind,
    ShapeLayoutKind, SoftmaxKind, SortKind, TernaryKind, UnaryKind,
};
pub use plan::{PlanPreference, PrecisionGuarantee};
pub use shape_expr::{
    Axis, CodecDecline, DimExpr, DimValue, Extent, ShapeDecline, ShapeExpr, decode_dim,
    decode_shape, encode_dim, encode_shape, eval_dim, eval_shape,
};
pub use sku::{BackendKind, KernelSku, OpCategory};
pub use structure_key::{
    AxisMask, Contiguity, ContractionKey, DivBucket, IdxWidth, MAX_OPERANDS, MAX_RANK, MpCode,
    OperandDesc, OperandKey, QuantFacts, QuantFamily, STRUCTURE_KEY_VERSION, ScalePlacement,
    SizeClass, StructureKey, SymExtent, SymKind, VecWidth, WorkClass, dtype_token, structure_key,
    structure_key_token,
};
