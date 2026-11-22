// SPDX-License-Identifier: MIT OR Apache-2.0
//! The **entry model** of a kernel language whose entry point is not "a function whose parameters are the buffers" (ROADMAP U19
//! census slice 4, `docs/options-u19-slang-entry-model.md`, option B agreed with fishstringer). A Slang compute kernel keeps its
//! resources as **file-scope globals** (`[[vk::binding(0, 0)]] StructuredBuffer<float> input;`) and takes its thread ids as
//! **semantics on the entry parameters** (`uint3 gid : SV_DispatchThreadID`), with the block shape in an attribute
//! (`[numthreads(256, 1, 1)]`). The grammar knowledge to read those lives in the language's own crate; it hands the walker this
//! model through [`Frontend::with_entry_model`](crate::Frontend::with_entry_model), and the walker stays neutral.
//!
//! The conventions (agreed with fishstringer, whose E17 slice 3 emits the same layout): the resources are the ones the entry
//! **references**, in **slot order**, which is the parameter order, with a constant buffer after the buffers; a gap, a repeat
//! or a non-zero start declines; `numthreads` wins over the context's block (the caller still supplies the grid); `elem` is the
//! source's own type text and the walker resolves it.

/// What the language reads off an entry point.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct EntryModel {
    /// The block shape the entry declares (`[numthreads(x, y, z)]`), if it does. It wins over the context's block.
    pub block: Option<[u32; 3]>,
    /// The entry parameters that are execution ids, by parameter name. Every parameter of the entry must be one.
    pub semantics: Vec<(String, Semantic)>,
    /// The file-scope resources the entry references.
    pub resources: Vec<Resource>,
}

impl EntryModel {
    /// A model from its three parts.
    #[must_use]
    pub fn new(
        block: Option<[u32; 3]>,
        semantics: Vec<(String, Semantic)>,
        resources: Vec<Resource>,
    ) -> Self {
        Self {
            block,
            semantics,
            resources,
        }
    }
}

/// The execution id an entry parameter carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Semantic {
    /// `SV_DispatchThreadID`: `blockIdx * blockDim + threadIdx` per axis (a `uint3`).
    DispatchThreadId,
    /// `SV_GroupID`: `blockIdx` (a `uint3`).
    GroupId,
    /// `SV_GroupThreadID`: `threadIdx` (a `uint3`).
    GroupThreadId,
    /// `SV_GroupIndex`: the linear thread index in the block (a `uint`).
    GroupIndex,
}

/// How a file-scope resource is declared.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ResourceKind {
    /// `StructuredBuffer<T>`: a read-only buffer.
    StructuredRO,
    /// `RWStructuredBuffer<T>`: a buffer the entry may write.
    StructuredRW,
    /// `ConstantBuffer<S>`: one struct of scalars; `elem` names the struct.
    ConstantBuffer,
}

/// One file-scope resource an entry references.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct Resource {
    /// The global's name.
    pub name: String,
    /// How it is declared.
    pub kind: ResourceKind,
    /// The element type's source text (`float`, `uint`), or the struct's name for a constant buffer.
    pub elem: String,
    /// The binding slot (`[[vk::binding(slot, 0)]]`, or declaration order where the source gives none).
    pub slot: u32,
}

impl Resource {
    /// A resource from its four parts.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        kind: ResourceKind,
        elem: impl Into<String>,
        slot: u32,
    ) -> Self {
        Self {
            name: name.into(),
            kind,
            elem: elem.into(),
            slot,
        }
    }
}

/// Why a language could not read an entry model; the walker declines with this text.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct EntryModelError {
    /// What it could not read.
    pub reason: String,
}

impl EntryModelError {
    /// An error with this reason.
    #[must_use]
    pub fn new(reason: impl Into<String>) -> Self {
        Self {
            reason: reason.into(),
        }
    }
}

/// A frontend's reader of an entry point: the CST, the source and the entry's name to its [`EntryModel`].
pub type EntryModelFn = fn(&tree_sitter::Tree, &str, &str) -> Result<EntryModel, EntryModelError>;
