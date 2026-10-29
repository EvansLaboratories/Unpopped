// SPDX-License-Identifier: MIT OR Apache-2.0
//! The **capability set** (U3 build row 2, `docs/u3-program-model-and-seam.md` sections 3 and 4.4).
//!
//! What a *device* reports or was measured to do, as keyed facts with provenance. Distinct
//! from [`crate::capability::TargetCapabilities`], which is the static per-compute-capability
//! resource table a schedule chooser reads.
//!
//! **Standalone by decision.** Seam hold: no `Backend::capabilities`, and
//! `Backend::supports_dtype` does not consult this yet. Types only.
//!
//! Three rules the types enforce:
//! - **Absent is unknown, never denied.** [`CapabilitySet::value`] of an unrecorded key is
//!   [`CapValue::Unknown`]; [`CapabilitySet::dtype_support`] of one is `NotExpressible`.
//! - **Denorm mode defaults to `Unknown`**, never `Preserve` (a compile line with no denorm
//!   flag declares nothing).
//! - **Every value carries its provenance** ([`CapSource`]); a key is recorded once, so a
//!   `Reported` value and a `Measured` one never silently shadow each other.
//!
//! Whether the `vulkan:` token set can carry a subgroup range or a measured tag is
//! UNVERIFIED (spec section 6); nothing here reads the vocabulary.

use crate::artifact::FeatureToken;
use crate::backend::DtypeSupport;
use unpopped_vocab::ElementKind;

/// Denormal handling of a device for one float type.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum DenormMode {
    /// Not stated. Never read as `Preserve`.
    #[default]
    Unknown,
    /// Denormals are kept.
    Preserve,
    /// Denormals flush to zero.
    Flush,
}

/// What a capability is about.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum CapKey {
    /// Whether a dtype is usable.
    Dtype(ElementKind),
    /// Denorm handling of a float dtype.
    Denorm(ElementKind),
    /// Subgroup size (a range: it may vary per launch).
    SubgroupSize,
    /// Largest invocation count of one workgroup.
    MaxWorkgroupInvocations,
    /// A named device feature token.
    Feature(FeatureToken),
}

/// The value of a capability.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum CapValue {
    /// A yes/no fact.
    Bool(bool),
    /// An inclusive range; `fixed` says the device cannot pick other than one end.
    Range {
        /// Smallest.
        min: u32,
        /// Largest.
        max: u32,
        /// `true` when the value is one number (`min == max`) the device pins.
        fixed: bool,
    },
    /// A denorm mode.
    Denorm(DenormMode),
    /// Not known.
    Unknown,
}

/// Where a value came from.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum CapSource {
    /// The driver or API said so.
    Reported,
    /// A probe observed it.
    Measured,
    /// A caller declared it.
    Declared,
}

/// One keyed fact.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct Capability {
    /// What it is about.
    pub key: CapKey,
    /// Its value.
    pub value: CapValue,
    /// Where the value came from.
    pub source: CapSource,
}

impl Capability {
    /// A capability.
    #[must_use]
    pub fn new(key: CapKey, value: CapValue, source: CapSource) -> Self {
        Self { key, value, source }
    }
}

/// Why [`CapabilitySet::insert`] refused.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum CapabilityError {
    /// The key is already recorded; replacing it would pick a winner between sources.
    DuplicateKey(CapKey),
    /// A `Range` with `min > max`, or `fixed` with `min != max`.
    BadRange {
        /// Smallest.
        min: u32,
        /// Largest.
        max: u32,
        /// The `fixed` flag.
        fixed: bool,
    },
}

/// The facts known about one device.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct CapabilitySet {
    /// The recorded facts, in insertion order, one per key.
    pub entries: Vec<Capability>,
}

impl CapabilitySet {
    /// An empty set: everything unknown.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a fact.
    ///
    /// # Errors
    /// [`CapabilityError::DuplicateKey`] if the key is recorded; [`CapabilityError::BadRange`]
    /// for an inconsistent range.
    pub fn insert(&mut self, c: Capability) -> Result<(), CapabilityError> {
        if let CapValue::Range { min, max, fixed } = c.value {
            if min > max || (fixed && min != max) {
                return Err(CapabilityError::BadRange { min, max, fixed });
            }
        }
        if self.get(&c.key).is_some() {
            return Err(CapabilityError::DuplicateKey(c.key));
        }
        self.entries.push(c);
        Ok(())
    }

    /// The fact for a key, if recorded.
    #[must_use]
    pub fn get(&self, key: &CapKey) -> Option<&Capability> {
        self.entries.iter().find(|c| &c.key == key)
    }

    /// The value for a key; [`CapValue::Unknown`] when unrecorded.
    #[must_use]
    pub fn value(&self, key: &CapKey) -> CapValue {
        self.get(key).map_or(CapValue::Unknown, |c| c.value.clone())
    }

    /// The denorm mode of a float dtype; [`DenormMode::Unknown`] unless a `Denorm` value is
    /// recorded for it.
    #[must_use]
    pub fn denorm(&self, dtype: ElementKind) -> DenormMode {
        match self.value(&CapKey::Denorm(dtype)) {
            CapValue::Denorm(m) => m,
            _ => DenormMode::Unknown,
        }
    }

    /// Whether a dtype is usable, as the answer `Backend::supports_dtype` gives: recorded
    /// `true` is `Supported`, recorded `false` is `Unsupported`, and unrecorded (or a value
    /// of another shape) is `NotExpressible`, because silence is not denial.
    #[must_use]
    pub fn dtype_support(&self, dtype: ElementKind) -> DtypeSupport {
        match self.value(&CapKey::Dtype(dtype)) {
            CapValue::Bool(true) => DtypeSupport::Supported,
            CapValue::Bool(false) => DtypeSupport::Unsupported,
            _ => DtypeSupport::NotExpressible,
        }
    }
}
