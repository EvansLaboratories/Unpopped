// SPDX-License-Identifier: MIT OR Apache-2.0
//! The kernel **artifact and manifest** types (U3 build row 1,
//! `docs/u3-program-model-and-seam.md` section 4.1).
//!
//! A CUDA kernel is source text and its argument ABI is positional, so
//! [`crate::backend::GeneratedKernel`]'s `String` and nothing else was enough. A SPIR-V
//! word stream cannot ride a `String`, and a descriptor-bound kernel needs its bindings and
//! `Params` block described next to it. These types are that description.
//!
//! **Standalone by decision.** The seam hold is in force: nothing here is a field of
//! `GeneratedKernel` or an item of `Backend` / `Frontend`. Wiring waits for the hold to lift;
//! until then these are types a caller can build and [`KernelManifest::check`], and no
//! backend reads them.
//!
//! **Nothing defaults.** [`LayoutRule`] has no `Default`, and a [`KernelManifest`] that
//! does not say how its `Params` block is laid out is not a manifest (spec 4.1). The shapes of
//! [`Packing`] and [`BindingKind`] are the spec's sketch, **not a ruling**: the spec names
//! the fields and leaves their variants open.

use unpopped_vocab::ElementKind;

/// What a backend produced, as bytes a compiler or driver consumes.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum KernelArtifact {
    /// Source text (CUDA C, C, Slang, GLSL).
    Source(String),
    /// A SPIR-V module as 32-bit words.
    SpirV(Vec<u32>),
}

impl KernelArtifact {
    /// The source text, or `None` for a binary artifact. A `SpirV` artifact is never
    /// rendered as text: a consumer that wants source must be told there is none.
    #[must_use]
    pub fn as_source(&self) -> Option<&str> {
        match self {
            KernelArtifact::Source(s) => Some(s),
            _ => None,
        }
    }

    /// The SPIR-V words, or `None` for any other artifact.
    #[must_use]
    pub fn as_spirv(&self) -> Option<&[u32]> {
        match self {
            KernelArtifact::SpirV(w) => Some(w),
            _ => None,
        }
    }
}

/// How a `Params` block (or a buffer element) is laid out. **No default**: a manifest
/// that does not say is not a manifest.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum LayoutRule {
    /// GLSL `std140` (array elements pad to 16 bytes).
    Std140,
    /// GLSL `std430`.
    Std430,
    /// Scalar block layout (every member at its natural alignment).
    Scalar,
}

/// The resource a binding names.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum BindingKind {
    /// A storage buffer.
    Storage,
    /// A uniform buffer.
    Uniform,
    /// A sampled texture.
    Texture,
    /// A sampler.
    Sampler,
}

/// What the kernel does with a binding.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum BindingRole {
    /// Read only.
    Read,
    /// Write only.
    Write,
    /// Read and written.
    ReadWrite,
}

/// How elements sit in a binding's storage.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Packing {
    /// One element per addressable unit.
    Unpacked,
    /// `elems_per_word` elements share one 32-bit word (sub-byte and 16-bit storage).
    Packed {
        /// Elements per 32-bit word; at least 2.
        elems_per_word: u32,
    },
}

/// A target feature a kernel entry requires (an opaque token from the target's vocabulary,
/// carried byte-exact and never interpreted here).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct FeatureToken(pub String);

/// One binding of one entry.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct Binding {
    /// Descriptor set.
    pub set: u32,
    /// Index within the set.
    pub index: u32,
    /// The resource kind.
    pub kind: BindingKind,
    /// Read, write or both.
    pub role: BindingRole,
    /// The element type stored.
    pub elem: ElementKind,
    /// The storage packing.
    pub packing: Packing,
}

impl Binding {
    /// A binding with every field stated.
    #[must_use]
    pub fn new(
        set: u32,
        index: u32,
        kind: BindingKind,
        role: BindingRole,
        elem: ElementKind,
        packing: Packing,
    ) -> Self {
        Self {
            set,
            index,
            kind,
            role,
            elem,
            packing,
        }
    }
}

/// One member of a `Params` block.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct ParamField {
    /// Member name.
    pub name: String,
    /// Byte offset in the block.
    pub offset: u32,
    /// Scalar type of the member.
    pub ty: ElementKind,
}

impl ParamField {
    /// A param field.
    #[must_use]
    pub fn new(name: impl Into<String>, offset: u32, ty: ElementKind) -> Self {
        Self {
            name: name.into(),
            offset,
            ty,
        }
    }
}

/// A `Params` block: its layout rule and its members.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct ParamsLayout {
    /// The layout rule the offsets were computed under.
    pub rule: LayoutRule,
    /// The members.
    pub fields: Vec<ParamField>,
}

impl ParamsLayout {
    /// A params layout; the rule is required.
    #[must_use]
    pub fn new(rule: LayoutRule, fields: Vec<ParamField>) -> Self {
        Self { rule, fields }
    }
}

/// What compiled (or will compile) the artifact. Free text is carried verbatim.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct ToolchainRecord {
    /// SPIR-V version, e.g. `"1.5"`.
    pub spirv_version: String,
    /// Target environment, e.g. `"vulkan1.3"`.
    pub target_env: String,
    /// Compiler profile, if any.
    pub profile: Option<String>,
    /// Compile flags, in order. A flag that is **absent** is unknown, never "default on".
    pub flags: Vec<String>,
}

impl ToolchainRecord {
    /// A toolchain record.
    #[must_use]
    pub fn new(
        spirv_version: impl Into<String>,
        target_env: impl Into<String>,
        profile: Option<String>,
        flags: Vec<String>,
    ) -> Self {
        Self {
            spirv_version: spirv_version.into(),
            target_env: target_env.into(),
            profile,
            flags,
        }
    }
}

/// One entry point of an artifact. Entry-per-module and entry-per-file are both valid.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct EntryManifest {
    /// Entry-point name.
    pub name: String,
    /// Local workgroup size.
    pub workgroup_size: [u32; 3],
    /// The bindings.
    pub bindings: Vec<Binding>,
    /// The `Params` block, if the entry takes one.
    pub params: Option<ParamsLayout>,
    /// Features the entry requires of the device.
    pub required_features: Vec<FeatureToken>,
}

impl EntryManifest {
    /// An entry with no bindings, params or required features; fill the pub fields.
    #[must_use]
    pub fn new(name: impl Into<String>, workgroup_size: [u32; 3]) -> Self {
        Self {
            name: name.into(),
            workgroup_size,
            bindings: Vec::new(),
            params: None,
            required_features: Vec::new(),
        }
    }
}

/// The description of an artifact: its toolchain and its entries.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct KernelManifest {
    /// What built it.
    pub toolchain: ToolchainRecord,
    /// The entry points.
    pub entries: Vec<EntryManifest>,
}

impl KernelManifest {
    /// A manifest.
    #[must_use]
    pub fn new(toolchain: ToolchainRecord, entries: Vec<EntryManifest>) -> Self {
        Self { toolchain, entries }
    }

    /// Reject a manifest that cannot describe a launchable artifact. Reports the first
    /// defect, in entry order.
    ///
    /// # Errors
    /// See [`ManifestError`].
    pub fn check(&self) -> Result<(), ManifestError> {
        if self.entries.is_empty() {
            return Err(ManifestError::NoEntries);
        }
        for (i, e) in self.entries.iter().enumerate() {
            if self.entries[..i].iter().any(|p| p.name == e.name) {
                return Err(ManifestError::DuplicateEntry(e.name.clone()));
            }
            check_entry(e)?;
        }
        Ok(())
    }
}

/// Why [`KernelManifest::check`] refused.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ManifestError {
    /// No entries at all.
    NoEntries,
    /// Two entries share a name.
    DuplicateEntry(String),
    /// An entry's workgroup size has a zero dimension.
    ZeroWorkgroup(String),
    /// Two bindings of one entry share `(set, index)`.
    DuplicateBinding {
        /// The entry.
        entry: String,
        /// The set.
        set: u32,
        /// The index.
        index: u32,
    },
    /// A `Packed` binding with fewer than 2 elements per word.
    DegeneratePacking {
        /// The entry.
        entry: String,
    },
    /// A `Params` member whose type has no scalar size here (refused, not skipped).
    UnsizedParamType {
        /// The entry.
        entry: String,
        /// The member.
        field: String,
    },
    /// A `Params` member not at a multiple of its own size.
    MisalignedParam {
        /// The entry.
        entry: String,
        /// The member.
        field: String,
    },
    /// Two `Params` members overlap.
    OverlappingParams {
        /// The entry.
        entry: String,
        /// The first member.
        a: String,
        /// The second member.
        b: String,
    },
}

/// Byte size of a scalar `Params` member; `None` for a type with no plain scalar size.
fn scalar_bytes(t: ElementKind) -> Option<u32> {
    use ElementKind as K;
    match t {
        K::I8 | K::U8 | K::Bool => Some(1),
        K::F16 | K::Bf16 | K::I16 | K::U16 => Some(2),
        K::F32 | K::F32Strict | K::I32 => Some(4),
        K::F64 | K::I64 => Some(8),
        _ => None,
    }
}

pub(crate) fn check_entry(e: &EntryManifest) -> Result<(), ManifestError> {
    let name = || e.name.clone();
    if e.workgroup_size.contains(&0) {
        return Err(ManifestError::ZeroWorkgroup(name()));
    }
    for (i, b) in e.bindings.iter().enumerate() {
        if e.bindings[..i]
            .iter()
            .any(|p| (p.set, p.index) == (b.set, b.index))
        {
            return Err(ManifestError::DuplicateBinding {
                entry: name(),
                set: b.set,
                index: b.index,
            });
        }
        if matches!(b.packing, Packing::Packed { elems_per_word } if elems_per_word < 2) {
            return Err(ManifestError::DegeneratePacking { entry: name() });
        }
    }
    if let Some(p) = &e.params {
        for (i, f) in p.fields.iter().enumerate() {
            let Some(size) = scalar_bytes(f.ty) else {
                return Err(ManifestError::UnsizedParamType {
                    entry: name(),
                    field: f.name.clone(),
                });
            };
            if f.offset % size != 0 {
                return Err(ManifestError::MisalignedParam {
                    entry: name(),
                    field: f.name.clone(),
                });
            }
            for g in &p.fields[..i] {
                // `g` was sized when it was visited as `f`.
                let gs = scalar_bytes(g.ty).unwrap_or(0);
                if f.offset < g.offset + gs && g.offset < f.offset + size {
                    return Err(ManifestError::OverlappingParams {
                        entry: name(),
                        a: g.name.clone(),
                        b: f.name.clone(),
                    });
                }
            }
        }
    }
    Ok(())
}
