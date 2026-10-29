// SPDX-License-Identifier: MIT OR Apache-2.0
//! The **declared resource map** a frontend lift can be given (U3 build row 3,
//! `docs/u3-row3-resource-map-and-frontend.md`).
//!
//! Without a map, [`crate::convert`] finds a kernel's buffers by *name convention*
//! (`out`/`in{K}`, `output`/`input{K}`). That is brittle by measurement: 0 of Fuel's 147
//! Slang kernels lift on it. A [`ResourceMap`] is the caller saying what the buffers and the
//! `Params` block are, and the lift resolves names through it.
//!
//! It is the *input* half of what a [`crate::artifact::KernelManifest`] entry later reports,
//! so it reuses that module's vocabulary ([`BindingRole`], [`Packing`], [`ParamsLayout`]) and
//! is checked by the same rules. It names no language, and it is not a field of
//! [`crate::convert::Frontend`] (which stays `Copy` and `const`-constructible).

use unpopped_vocab::ElementKind;

use crate::artifact::{
    Binding, BindingKind, BindingRole, EntryManifest, ManifestError, Packing, ParamsLayout,
};

/// One buffer the caller declares.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct DeclaredBuffer {
    /// The name the kernel source uses for it.
    pub name: String,
    /// Descriptor set.
    pub set: u32,
    /// Index within the set.
    pub index: u32,
    /// Read, write or both.
    pub role: BindingRole,
    /// The element type stored.
    pub elem: ElementKind,
    /// The storage packing.
    pub packing: Packing,
}

impl DeclaredBuffer {
    /// A declared buffer with every field stated.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        set: u32,
        index: u32,
        role: BindingRole,
        elem: ElementKind,
        packing: Packing,
    ) -> Self {
        Self {
            name: name.into(),
            set,
            index,
            role,
            elem,
            packing,
        }
    }
}

/// The buffers and `Params` block of one kernel, as the caller declares them.
///
/// **Operand order is declaration order**: the `K`th [`BindingRole::Read`] buffer is operand
/// `K` (`ScalarExpr::Input(K)`), and the `K`th `Params` field is `ScalarExpr::Param(K)`. The
/// one [`BindingRole::Write`] / [`BindingRole::ReadWrite`] buffer is the output.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct ResourceMap {
    /// The buffers, in declaration order.
    pub buffers: Vec<DeclaredBuffer>,
    /// The `Params` block, if any. Its [`crate::artifact::LayoutRule`] is required: there is
    /// no default.
    pub params: Option<ParamsLayout>,
}

/// Why a [`ResourceMap`] is not usable.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ResourceMapError {
    /// A buffer has an empty name.
    EmptyName,
    /// Two buffers, or a buffer and a `Params` field, share a name.
    DuplicateName(String),
    /// The lifts read one output; the map declares this many `Write` / `ReadWrite` buffers.
    OutputCount(usize),
    /// The binding or `Params` description fails the manifest rules.
    Manifest(ManifestError),
}

impl ResourceMap {
    /// A map from its buffers and optional `Params` block.
    #[must_use]
    pub fn new(buffers: Vec<DeclaredBuffer>, params: Option<ParamsLayout>) -> Self {
        Self { buffers, params }
    }

    /// Reject a map the lifts cannot resolve names through. Reports the first defect.
    ///
    /// # Errors
    /// See [`ResourceMapError`].
    pub fn check(&self) -> Result<(), ResourceMapError> {
        let names: Vec<&str> = self
            .buffers
            .iter()
            .map(|b| b.name.as_str())
            .chain(
                self.params
                    .iter()
                    .flat_map(|p| p.fields.iter().map(|f| f.name.as_str())),
            )
            .collect();
        for (i, n) in names.iter().enumerate() {
            if n.is_empty() {
                return Err(ResourceMapError::EmptyName);
            }
            if names[..i].contains(n) {
                return Err(ResourceMapError::DuplicateName((*n).to_string()));
            }
        }
        let outputs = self
            .buffers
            .iter()
            .filter(|b| b.role != BindingRole::Read)
            .count();
        if outputs != 1 {
            return Err(ResourceMapError::OutputCount(outputs));
        }
        // The same rules a manifest entry is held to (duplicate `(set, index)`, packing,
        // `Params` alignment and overlap), not a second copy of them.
        let mut entry = EntryManifest::new("resource-map", [1, 1, 1]);
        entry.bindings = self
            .buffers
            .iter()
            .map(|b| {
                Binding::new(
                    b.set,
                    b.index,
                    BindingKind::Storage,
                    b.role,
                    b.elem,
                    b.packing,
                )
            })
            .collect();
        entry.params.clone_from(&self.params);
        crate::artifact::check_entry(&entry).map_err(ResourceMapError::Manifest)
    }

    /// The output buffer's name. Only meaningful after [`check`](Self::check).
    #[cfg(feature = "convert")]
    pub(crate) fn output_name(&self) -> Option<&str> {
        self.buffers
            .iter()
            .find(|b| b.role != BindingRole::Read)
            .map(|b| b.name.as_str())
    }

    /// Operand names, in operand order (`Read` buffers in declaration order).
    #[cfg(feature = "convert")]
    pub(crate) fn input_names(&self) -> Vec<&str> {
        self.buffers
            .iter()
            .filter(|b| b.role == BindingRole::Read)
            .map(|b| b.name.as_str())
            .collect()
    }

    /// `Params` field names, in `ScalarExpr::Param` order.
    #[cfg(feature = "convert")]
    pub(crate) fn param_names(&self) -> Vec<&str> {
        self.params
            .iter()
            .flat_map(|p| p.fields.iter().map(|f| f.name.as_str()))
            .collect()
    }
}
