// SPDX-License-Identifier: MIT OR Apache-2.0
//! What a consumer needs to **launch** a kernel a backend lowered with [`crate::Backend::lower_kernel`], as data
//! (`GeneratedKernel::interface`, PM ruling 2026-11-11 on fishstringer#17, with the field lists of Fishstringer, Vulkane and
//! Baracuda's launchers).
//!
//! Two halves. The **core-derived** half comes from the [`KernelFn`] alone, so every emitter agrees on it:
//! [`KernelInterface::of`] reads the launch geometry, the parameter list and the dynamic `Shared` allocations. The
//! **emitter-set** half is what only the emitter knows: backend-assigned binding slots, the layout of a parameter block,
//! the capabilities and compiler arguments the emitted source needs, launch bounds, the entry symbol. It is empty by
//! default and set through the `with_*` builders, so a backend that sets nothing is unchanged. Nothing here is part of
//! the kernel's revision hash: the emitter-set fields follow from the source text the emitter wrote.

use crate::capability::TargetCapabilities;
use crate::exec::{Extent, LaunchShape, MemorySpace};
use crate::kernel::{KernelFn, Param, ParamKind};
use unpopped_vocab::ElementKind;

/// A `Shared` allocation whose element count is a launch argument (`Extent::Param`: CUDA `extern __shared__`, sized by the
/// launch). The size in bytes is `size_param`'s value times the element size; a consumer that needs more than the target's
/// default dynamic limit raises it before the launch (CUDA: `cudaFuncSetAttribute(MaxDynamicSharedMemorySize)`).
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct DynamicShared {
    /// Index into [`KernelFn::allocs`].
    pub alloc: u32,
    /// The allocation's name.
    pub name: String,
    /// Its element type.
    pub elem: ElementKind,
    /// Index into [`KernelInterface::params`] of the parameter that gives the element count: a scalar, or a struct
    /// whose field `size_field` does.
    pub size_param: usize,
    /// For a struct `size_param`, the index of the integer field in its `fields`; `None` for a scalar parameter.
    pub size_field: Option<usize>,
}

/// A binding slot the emitter assigned to a parameter (a Vulkan set and binding, an HLSL register space and register).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct Binding {
    /// Index into [`KernelInterface::params`].
    pub param: usize,
    /// The descriptor set or register space.
    pub space: u32,
    /// The binding or register number.
    pub slot: u32,
}

impl Binding {
    /// Parameter `param` is bound at `slot` of `space`.
    #[must_use]
    pub const fn new(param: usize, space: u32, slot: u32) -> Self {
        Self { param, space, slot }
    }
}

/// One field of the parameter block the emitter packs the scalars, the grid dimensions and each buffer's shape, strides and
/// base offset into.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct ParamsBlockField {
    /// The field's name.
    pub name: String,
    /// Its byte offset in the block.
    pub offset: u32,
    /// Its element type.
    pub ty: ElementKind,
}

impl ParamsBlockField {
    /// A field `name` of type `ty` at byte `offset`.
    #[must_use]
    pub fn new(name: impl Into<String>, offset: u32, ty: ElementKind) -> Self {
        Self {
            name: name.into(),
            offset,
            ty,
        }
    }
}

/// The layout of the parameter block: its fields, total size and alignment, and whether it is passed by address
/// (a CUDA `__grid_constant__` parameter) rather than copied by value.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct ParamsBlock {
    /// The fields in offset order.
    pub fields: Vec<ParamsBlockField>,
    /// Total size in bytes.
    pub size: u32,
    /// Alignment in bytes.
    pub align: u32,
    /// Passed by address of the argument (`__grid_constant__`) instead of by value.
    pub by_address: bool,
}

impl ParamsBlock {
    /// An empty block of `size` bytes aligned to `align`.
    #[must_use]
    pub fn new(size: u32, align: u32) -> Self {
        Self {
            fields: Vec::new(),
            size,
            align,
            by_address: false,
        }
    }

    /// The block with `field` appended.
    #[must_use]
    pub fn with_field(mut self, field: ParamsBlockField) -> Self {
        self.fields.push(field);
        self
    }

    /// The block passed by address (`__grid_constant__`).
    #[must_use]
    pub fn passed_by_address(mut self) -> Self {
        self.by_address = true;
        self
    }

    /// The block of a [`ParamKind::Struct`] parameter, derived from its fields (ROADMAP U23): each field at the next
    /// offset aligned to its own size (the natural C layout, no packing), the block aligned to its widest field and sized
    /// to a multiple of that alignment. A `Complex64`/`Complex128` field is a pair aligned to one half. By value; chain
    /// [`ParamsBlock::passed_by_address`] for `__grid_constant__`. **No thread-set, memory-kind or copy-count annotation** (the U23 guard; test module `layout_only_guard`).
    ///
    /// # Errors
    /// [`ParamsBlockError::NotAStruct`] for a buffer or scalar parameter; [`ParamsBlockError::Unsized`] for a field type
    /// with no whole-byte width (`I4`, `U4`, `B1`) or no defined one.
    pub fn of_struct(param: &Param) -> Result<Self, ParamsBlockError> {
        let ParamKind::Struct { fields } = &param.kind else {
            return Err(ParamsBlockError::NotAStruct);
        };
        let mut block = Self::new(0, 1);
        let mut offset: u32 = 0;
        let mut align: u32 = 1;
        for f in fields {
            let (size, a) = field_size_align(f.ty).ok_or_else(|| ParamsBlockError::Unsized {
                field: f.name.clone(),
                ty: f.ty,
            })?;
            offset = offset.next_multiple_of(a);
            block
                .fields
                .push(ParamsBlockField::new(f.name.clone(), offset, f.ty));
            offset += size;
            align = align.max(a);
        }
        block.size = offset.next_multiple_of(align);
        block.align = align;
        Ok(block)
    }
}

/// Why [`ParamsBlock::of_struct`] declined.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ParamsBlockError {
    /// The parameter is not a [`ParamKind::Struct`].
    NotAStruct,
    /// A field's type has no whole-byte width to lay out.
    Unsized {
        /// The field's name.
        field: String,
        /// Its type.
        ty: ElementKind,
    },
}

impl std::fmt::Display for ParamsBlockError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotAStruct => f.write_str("parameter is not a struct"),
            Self::Unsized { field, ty } => {
                write!(
                    f,
                    "struct field `{field}` of type {ty:?} has no whole-byte layout"
                )
            }
        }
    }
}

impl std::error::Error for ParamsBlockError {}

/// `(size, alignment)` in bytes of a laid-out field, or `None` for a type with no whole-byte width.
fn field_size_align(ty: ElementKind) -> Option<(u32, u32)> {
    Some(match ty {
        ElementKind::I8
        | ElementKind::U8
        | ElementKind::Bool
        | ElementKind::Fp8E4M3FN
        | ElementKind::Fp8E5M2 => (1, 1),
        ElementKind::F16 | ElementKind::Bf16 | ElementKind::I16 | ElementKind::U16 => (2, 2),
        ElementKind::F32 | ElementKind::F32Strict | ElementKind::I32 | ElementKind::U32 => (4, 4),
        ElementKind::F64 | ElementKind::I64 | ElementKind::U64 => (8, 8),
        ElementKind::Complex64 => (8, 4),
        ElementKind::Complex128 => (16, 8),
        _ => return None,
    })
}

/// What the emitter guarantees about a buffer parameter, beyond its [`crate::kernel::BufferRole`]: the launcher asserts it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct BufferGuarantee {
    /// Index into [`KernelInterface::params`].
    pub param: usize,
    /// The byte alignment the buffer's base must have, if the emitted code relies on one.
    pub align_bytes: Option<u32>,
    /// The buffer does not alias another parameter (`__restrict__`, `noalias`).
    pub restrict: bool,
}

impl BufferGuarantee {
    /// Parameter `param` needs `align_bytes` alignment (if `Some`) and is `restrict` (if true).
    #[must_use]
    pub const fn new(param: usize, align_bytes: Option<u32>, restrict: bool) -> Self {
        Self {
            param,
            align_bytes,
            restrict,
        }
    }
}

/// Occupancy bounds the emitted kernel was compiled with (`__launch_bounds__`): they change the legal block size.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct LaunchBounds {
    /// The most threads per block a launch may use.
    pub max_threads: u32,
    /// The fewest blocks per multiprocessor the compiler must leave room for, if stated.
    pub min_blocks_per_sm: Option<u32>,
}

impl LaunchBounds {
    /// At most `max_threads` threads per block and, if `Some`, at least `min_blocks_per_sm` blocks per multiprocessor.
    #[must_use]
    pub const fn new(max_threads: u32, min_blocks_per_sm: Option<u32>) -> Self {
        Self {
            max_threads,
            min_blocks_per_sm,
        }
    }
}

/// What a specialization constant stands for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum SpecTarget {
    /// The block size on axis `.0` (0, 1 or 2): its value in [`KernelInterface::launch`] is the default.
    BlockDim(u8),
    /// The element count of the dynamic `Shared` allocation at index `.0` of [`KernelInterface::dynamic_shared`].
    DynamicShared(usize),
}

/// A value the emitter left as a specialization constant instead of baking it into the source (a SPIR-V `SpecId`, an HLSL
/// or Slang specialization parameter): the host supplies it when it builds the pipeline.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct SpecConstant {
    /// The constant's id.
    pub id: u32,
    /// What it stands for.
    pub target: SpecTarget,
}

impl SpecConstant {
    /// Specialization constant `id` stands for `target`.
    #[must_use]
    pub const fn new(id: u32, target: SpecTarget) -> Self {
        Self { id, target }
    }
}

/// The launch interface of a lowered kernel. See the module docs for the two halves.
#[derive(Clone, Debug, PartialEq, Default)]
#[non_exhaustive]
pub struct KernelInterface {
    /// The kernel's launch geometry (`None`: a kernel with no launch shape).
    pub launch: Option<LaunchShape>,
    /// The parameters in declaration order: buffers with element type, role and rank, scalars with type and local.
    pub params: Vec<Param>,
    /// The `Shared` allocations sized by a launch argument.
    pub dynamic_shared: Vec<DynamicShared>,
    /// The symbol to link or look up, when it differs from `GeneratedKernel::name` (a templated instantiation's mangled name).
    pub entry_symbol: Option<String>,
    /// Backend-assigned binding slots, by parameter.
    pub bindings: Vec<Binding>,
    /// The layout of the parameter block.
    pub params_block: Option<ParamsBlock>,
    /// What the emitted code relies on about a buffer: alignment, no aliasing.
    pub buffer_guarantees: Vec<BufferGuarantee>,
    /// The occupancy bounds the kernel was compiled with.
    pub launch_bounds: Option<LaunchBounds>,
    /// The most registers per thread the kernel was compiled with (`__maxnreg__`).
    pub max_regs: Option<u32>,
    /// The kernel must be launched cooperatively (`cudaLaunchCooperativeKernel`).
    pub cooperative: bool,
    /// Capabilities the emitted source needs (`shaderInt64`, 16-bit arithmetic, float or 64-bit atomics, a subgroup size),
    /// by name, so a consumer can decline before it builds a pipeline.
    pub requires: Vec<String>,
    /// Compiler arguments the emitted source needs (`-denorm-mode-fp32 preserve`).
    pub compiler_args: Vec<String>,
    /// The values left as specialization constants rather than baked in.
    pub spec_constants: Vec<SpecConstant>,
    /// Where the emitted kernel differs from the kernel's semantics in a way the source does not show (an async copy lowered
    /// synchronously, a barrier widened), one sentence each, so a consumer can decline or warn.
    pub fidelity_notes: Vec<String>,
}

impl KernelInterface {
    /// The core-derived half for `kernel`: its launch geometry, its parameters, and the `Shared` allocations a scalar
    /// parameter sizes. The emitter-set half is empty.
    #[must_use]
    pub fn of(kernel: &KernelFn) -> Self {
        let dynamic_shared = kernel
            .allocs
            .iter()
            .enumerate()
            .filter(|(_, a)| a.space == MemorySpace::Shared)
            .filter_map(|(i, a)| match a.extent {
                Extent::Param(local) => {
                    let (size_param, size_field, _) = kernel.argument_of_local(local)?;
                    Some(DynamicShared {
                        alloc: u32::try_from(i).unwrap_or(u32::MAX),
                        name: a.name.clone(),
                        elem: a.elem,
                        size_param,
                        size_field,
                    })
                }
                _ => None,
            })
            .collect();
        Self {
            launch: kernel.launch,
            params: kernel.params.clone(),
            dynamic_shared,
            ..Self::default()
        }
    }

    /// The interface with the symbol to link or look up.
    #[must_use]
    pub fn with_entry_symbol(mut self, symbol: impl Into<String>) -> Self {
        self.entry_symbol = Some(symbol.into());
        self
    }

    /// The interface with `binding` appended.
    #[must_use]
    pub fn with_binding(mut self, binding: Binding) -> Self {
        self.bindings.push(binding);
        self
    }

    /// The interface with the parameter block's layout.
    #[must_use]
    pub fn with_params_block(mut self, block: ParamsBlock) -> Self {
        self.params_block = Some(block);
        self
    }

    /// The interface with `guarantee` appended.
    #[must_use]
    pub fn with_buffer_guarantee(mut self, guarantee: BufferGuarantee) -> Self {
        self.buffer_guarantees.push(guarantee);
        self
    }

    /// The interface with its launch bounds.
    #[must_use]
    pub fn with_launch_bounds(mut self, bounds: LaunchBounds) -> Self {
        self.launch_bounds = Some(bounds);
        self
    }

    /// The interface with its register cap.
    #[must_use]
    pub fn with_max_regs(mut self, regs: u32) -> Self {
        self.max_regs = Some(regs);
        self
    }

    /// The interface marked as needing a cooperative launch.
    #[must_use]
    pub fn cooperative(mut self) -> Self {
        self.cooperative = true;
        self
    }

    /// The interface with the capability `name` appended to what the source needs.
    #[must_use]
    pub fn with_requires(mut self, name: impl Into<String>) -> Self {
        self.requires.push(name.into());
        self
    }

    /// The interface with the compiler argument `arg` appended.
    #[must_use]
    pub fn with_compiler_arg(mut self, arg: impl Into<String>) -> Self {
        self.compiler_args.push(arg.into());
        self
    }

    /// The interface with `constant` appended to the specialization constants.
    #[must_use]
    pub fn with_spec_constant(mut self, constant: SpecConstant) -> Self {
        self.spec_constants.push(constant);
        self
    }

    /// The interface with the fidelity note `note` appended.
    #[must_use]
    pub fn with_fidelity_note(mut self, note: impl Into<String>) -> Self {
        self.fidelity_notes.push(note.into());
        self
    }
}

/// Why a kernel's `Shared` memory could not be sized or does not fit ([`check_shared_size`]).
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum SharedSizeError {
    /// An allocation's element has no byte size here: a sub-byte dtype (packed or container width is a layout decision this
    /// check does not make) or a dtype the vocabulary does not size.
    UnsizedElement {
        /// The allocation's name.
        alloc: String,
        /// Its element type.
        elem: ElementKind,
    },
    /// `dynamic_counts` has no entry for the dynamic allocation at this index of [`KernelInterface::dynamic_shared`].
    DynamicCountMissing {
        /// The allocation's name.
        alloc: String,
    },
    /// The byte total does not fit `u64`.
    Overflow,
    /// The total is above the target's per-block limit.
    OverLimit {
        /// Static plus dynamic `Shared` bytes.
        bytes: u64,
        /// [`TargetCapabilities::max_shared_mem_per_block`].
        limit: u32,
    },
}

/// Byte size of one `Shared` element, for the dtypes whose size is the plain width (`None` otherwise, by name).
const fn shared_elem_bytes(e: ElementKind) -> Option<u64> {
    match e {
        ElementKind::I8
        | ElementKind::U8
        | ElementKind::Bool
        | ElementKind::Fp8E4M3FN
        | ElementKind::Fp8E5M2 => Some(1),
        ElementKind::F16 | ElementKind::Bf16 | ElementKind::I16 | ElementKind::U16 => Some(2),
        ElementKind::F32 | ElementKind::F32Strict | ElementKind::I32 | ElementKind::U32 => Some(4),
        ElementKind::F64 | ElementKind::I64 | ElementKind::U64 | ElementKind::Complex64 => Some(8),
        ElementKind::Complex128 => Some(16),
        _ => None,
    }
}

/// The bytes of `Shared` memory `kernel` declares: every static allocation, plus each dynamic one at the element count in
/// `dynamic_counts` (in the order of [`KernelInterface::dynamic_shared`], the launch's values for the sizing parameters).
/// No alignment padding is added: the figure is the sum of `count * element size`.
///
/// # Errors
/// [`SharedSizeError::UnsizedElement`], [`SharedSizeError::DynamicCountMissing`] or [`SharedSizeError::Overflow`].
pub fn shared_bytes(kernel: &KernelFn, dynamic_counts: &[u64]) -> Result<u64, SharedSizeError> {
    let mut total: u64 = 0;
    let mut next_dynamic = 0usize;
    for a in kernel
        .allocs
        .iter()
        .filter(|a| a.space == MemorySpace::Shared)
    {
        let width = shared_elem_bytes(a.elem).ok_or_else(|| SharedSizeError::UnsizedElement {
            alloc: a.name.clone(),
            elem: a.elem,
        })?;
        let count = match a.extent {
            Extent::Static(n) => u64::from(n),
            Extent::Param(_) => {
                let c = dynamic_counts.get(next_dynamic).copied().ok_or_else(|| {
                    SharedSizeError::DynamicCountMissing {
                        alloc: a.name.clone(),
                    }
                })?;
                next_dynamic += 1;
                c
            }
        };
        total = count
            .checked_mul(width)
            .and_then(|b| total.checked_add(b))
            .ok_or(SharedSizeError::Overflow)?;
    }
    Ok(total)
}

/// Does `kernel`'s `Shared` memory ([`shared_bytes`]) fit one block of the target? Returns the byte total.
///
/// The limit is [`TargetCapabilities::max_shared_mem_per_block`], the opt-in maximum (on CUDA it needs
/// `cudaFuncSetAttribute(MaxDynamicSharedMemorySize)`); the smaller default static limit is not modelled, and neither is
/// the per-SM figure or any reserved system shared memory. Limits are not monotonic in the architecture number
/// ([`TargetCapabilities`]), so pass the row of the actual target.
///
/// # Errors
/// Those of [`shared_bytes`], and [`SharedSizeError::OverLimit`].
pub fn check_shared_size(
    kernel: &KernelFn,
    dynamic_counts: &[u64],
    target: &TargetCapabilities,
) -> Result<u64, SharedSizeError> {
    let bytes = shared_bytes(kernel, dynamic_counts)?;
    if bytes > u64::from(target.max_shared_mem_per_block) {
        return Err(SharedSizeError::OverLimit {
            bytes,
            limit: target.max_shared_mem_per_block,
        });
    }
    Ok(bytes)
}

/// Why a kernel's launch shape does not fit a target ([`check_launch_limits`]).
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum LaunchLimitError {
    /// The kernel has no launch shape, so there is no block to size.
    NoLaunch,
    /// A block axis is zero: the block has no thread.
    EmptyBlock,
    /// The threads of a block do not fit `u64`.
    Overflow,
    /// The block has more threads than the target allows.
    OverThreads {
        /// Threads in the block (the product of the three axes).
        threads: u64,
        /// [`TargetCapabilities::max_threads_per_block`].
        limit: u32,
    },
    /// The kernel requires a subgroup size range that does not contain the target's.
    SubgroupSize {
        /// The smallest size the kernel accepts.
        min: u32,
        /// The largest size the kernel accepts.
        max: u32,
        /// [`TargetCapabilities::warp_size`].
        warp_size: u32,
    },
}

/// Does `kernel`'s launch shape fit one block of the target? Returns the threads per block.
///
/// Checks the block's total thread count against [`TargetCapabilities::max_threads_per_block`] and a required subgroup size
/// range against [`TargetCapabilities::warp_size`]. Not modelled: the per-axis block limits, the grid limits, the cluster
/// size limit, and occupancy (registers per thread and shared memory per SM); pass the row of the actual target.
///
/// # Errors
/// [`LaunchLimitError`].
pub fn check_launch_limits(
    kernel: &KernelFn,
    target: &TargetCapabilities,
) -> Result<u64, LaunchLimitError> {
    let launch = kernel.launch.as_ref().ok_or(LaunchLimitError::NoLaunch)?;
    if launch.block.contains(&0) {
        return Err(LaunchLimitError::EmptyBlock);
    }
    let threads = launch
        .block
        .iter()
        .try_fold(1u64, |acc, &d| acc.checked_mul(u64::from(d)))
        .ok_or(LaunchLimitError::Overflow)?;
    if threads > u64::from(target.max_threads_per_block) {
        return Err(LaunchLimitError::OverThreads {
            threads,
            limit: target.max_threads_per_block,
        });
    }
    if let Some(range) = launch.subgroup
        && !(range.min..=range.max).contains(&target.warp_size)
    {
        return Err(LaunchLimitError::SubgroupSize {
            min: range.min,
            max: range.max,
            warp_size: target.warp_size,
        });
    }
    Ok(threads)
}

/// **The U23 patent guard, as a compile-time check** (`docs/PATENT-SWEEP.md`, U23 section; PM 2026-11-17). `Param`,
/// `ParamKind` and `ParamsBlock` carry a layout and no storage decision: no thread-set, memory-kind or copy-count annotation, and no
/// choice of where data lives by which threads read it (US20230305845A1, pending, claims 1, 9, 15, 21, 27). Each type is
/// destructured with every field named and no `..`, and `ParamKind` is matched with no wildcard, so adding a field or a
/// variant stops this module compiling. The fix is a person reading that section first, not editing this test.
#[cfg(test)]
mod layout_only_guard {
    use super::*;
    use crate::kernel::{ParamKind, StructField};

    #[test]
    fn param_types_carry_layout_and_nothing_else() {
        let Param { name: _, kind } = Param::scalar("x", ElementKind::I32, 0);
        match kind {
            ParamKind::Buffer {
                elem: _,
                role: _,
                rank: _,
            } => {}
            ParamKind::Scalar { ty: _, local: _ } => {}
            ParamKind::Struct { fields } => {
                for f in fields {
                    let StructField {
                        name: _,
                        ty: _,
                        local: _,
                    } = f;
                }
            }
        }
        let ParamsBlock {
            fields,
            size: _,
            align: _,
            by_address: _,
        } = ParamsBlock::new(4, 4).with_field(ParamsBlockField::new("x", 0, ElementKind::I32));
        for f in fields {
            let ParamsBlockField {
                name: _,
                offset: _,
                ty: _,
            } = f;
        }
    }
}
